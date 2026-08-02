from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from models.crf_layer import LinearChainCRF

#TODO 
'''
self.attention = nn.TransformerEncoderLayer(
    d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, dropout=dropout, batch_first=True
)
'''
#emission_temperature = trial.suggest_float("emission_temperature", 0.5, 2.0)
#also tune transformer, unfreeze top 2 layers and attach lora adapter r=8
class MultiTaskNeuralCRFTagger(nn.Module):

    def __init__(
        self,
        dense_dim: int = 768,
        stylo_dim: int = 84,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        feature_input_dropout: float = 0.0,  
        rnn_type: str = "LSTM",              
        aux_boundary_weight: float = 0.4,
        boundary_pos_weight: float = 5.0,    
        emission_temp: float = 1.0,
        use_attention: bool = True,
        aux_pos_weight: Optional[float] = None, 
    ):
        super().__init__()
        self.dense_dim = dense_dim
        self.stylo_dim = stylo_dim
        self.hidden_dim = hidden_dim
        self.aux_boundary_weight = aux_boundary_weight
        self.rnn_type = rnn_type.upper()
        self.emission_temp = emission_temp

        # Handle backward compatibility if aux_pos_weight was passed instead
        pos_weight_val = boundary_pos_weight if aux_pos_weight is None else aux_pos_weight

        # 1A. Stylometric normalization safeguard
        self.stylo_layernorm = nn.LayerNorm(stylo_dim)

        # 1B. Dual-Branch Projection Network
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.stylo_projection = nn.Sequential(
            nn.Linear(stylo_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # [NEW] Gated Multimodal Unit (GMU) Fusion Gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()  # Outputs values strictly between 0.0 and 1.0
        )

        # 1C. [NEW] Feature Input Dropout
        # Applied directly to fused embeddings before entering the RNN
        self.input_dropout = nn.Dropout(p=feature_input_dropout)

        # 2. [UPDATED] Sequence Context Encoder (BiLSTM or BiGRU)
        rnn_cls = nn.GRU if self.rnn_type == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=hidden_dim,  # e.g., 192 + 64 = 256
            hidden_size=hidden_dim // 2,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        self.use_attention = use_attention
        if self.use_attention:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, 
                num_heads=4, 
                dropout=dropout, 
                batch_first=True
            )
            self.attn_layernorm = nn.LayerNorm(hidden_dim)

        # 3. Primary Emissions Head [Human=0, AI=1]
        self.emission_head = nn.Linear(hidden_dim, 2)

        # 4. Auxiliary Boundary Head
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim // 2),  # <--- Changed from * 2 to * 4
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 5. Linear-Chain CRF Layer
        self.crf = LinearChainCRF(num_tags=2)

        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32),
            reduction="none"
        )

    def forward(
        self,
        fused_features: torch.Tensor,
        mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        boundaries: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:

        # Unpack inputs dynamically
        dense_feats = fused_features[..., : self.dense_dim]
        stylo_feats = fused_features[..., self.dense_dim :]

        # Normalize stylometrics & project branches
        stylo_feats = self.stylo_layernorm(stylo_feats)

        proj_dense = self.dense_projection(dense_feats)  # Shape: (batch, seq_len, hidden_dim)
        proj_stylo = self.stylo_projection(stylo_feats)  # Shape: (batch, seq_len, hidden_dim)

        # Calculate gating weights g in [0, 1]
        gate_input = torch.cat([proj_dense, proj_stylo], dim=-1) # Shape: (batch, seq_len, hidden_dim * 2)
        gate = self.fusion_gate(gate_input)                     # Shape: (batch, seq_len, hidden_dim)

        # Dynamic Gated Fusion: g * dense + (1 - g) * stylo
        projected = gate * proj_dense + (1.0 - gate) * proj_stylo # Shape: (batch, seq_len, hidden_dim)

        # [NEW] Apply feature input dropout across fused projected sequence
        projected = self.input_dropout(projected)

        # Pack sequences before RNN (lengths moved to CPU for PyTorch RNN requirement)
        seq_lengths_cpu = torch.clamp(mask.sum(dim=1).long(), min=1).cpu()

        packed_input = nn.utils.rnn.pack_padded_sequence(
            projected, seq_lengths_cpu, batch_first=True, enforce_sorted=False
        )
        
        # [UPDATED] Pass packed sequences through selected RNN (LSTM or GRU)
        packed_output, _ = self.rnn(packed_input)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=fused_features.size(1)
        )

        # ... 2. [NEW] Optional Residual Self-Attention ...
        if self.use_attention:
            # Key, Query, Value are all rnn_out
            attn_out, _ = self.self_attn(
                query=rnn_out, key=rnn_out, value=rnn_out, key_padding_mask=~mask.bool()
            )
            # Residual Connection + LayerNorm
            rnn_out = self.attn_layernorm(rnn_out + attn_out)

        # Emissions & Auxiliary Boundary Predictions
        emissions = self.emission_head(rnn_out) / self.emission_temp

        zeros = torch.zeros_like(rnn_out[:, :1, :])
        shifted_rnn = torch.cat([zeros, rnn_out[:, :-1, :]], dim=1)

        # Compute pair interaction terms
        diff = torch.abs(rnn_out - shifted_rnn)  # Element-wise style magnitude delta
        prod = rnn_out * shifted_rnn            # Element-wise feature alignment

        # Concatenate [h_i, h_{i-1}, |h_i - h_{i-1}|, h_i * h_{i-1}] -> Shape: (batch, seq_len, hidden_dim * 4)
        boundary_inputs = torch.cat([rnn_out, shifted_rnn, diff, prod], dim=-1)
        boundary_logits = self.boundary_head(boundary_inputs).squeeze(-1)

        outputs = {"emissions": emissions, "boundary_logits": boundary_logits}

        # Loss Computation
        if labels is not None and boundaries is not None:
            # Unreduced sequence NLL
            crf_seq_nll = self.crf(emissions, labels.long(), mask, reduction="none")

            # Exact per-token CRF loss calculation
            total_active_tokens = torch.clamp(mask.float().sum(), min=1.0)
            crf_token_loss = crf_seq_nll.sum() / total_active_tokens

            # Boundary BCE loss per active token (uses self.pos_weight buffer)
            bce_raw = self.bce_loss(boundary_logits, boundaries.float())
            bce_masked = (bce_raw * mask.float()).sum() / total_active_tokens

            total_loss = crf_token_loss + self.aux_boundary_weight * bce_masked

            outputs["loss"] = total_loss
            outputs["crf_loss"] = crf_token_loss
            outputs["boundary_loss"] = bce_masked

        return outputs

    def predict(
        self, fused_features: torch.Tensor, mask: torch.Tensor
    ) -> Dict[str, Any]:
        """Runs Viterbi sequence decoding and marginal state probability inference without side-effects."""
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                outputs = self.forward(fused_features, mask)
                emissions = outputs["emissions"]

                # 1. Global Viterbi Decode Path
                viterbi_paths = self.crf.viterbi_decode(emissions, mask)

                # 2. Continuous Calibrated Marginal Probabilities P(y_i = 1 | x)
                marginals = self.crf.compute_marginal_probabilities(emissions, mask)
                ai_probabilities = marginals[:, :, 1] * mask.float()

                boundary_probs = (
                    torch.sigmoid(outputs["boundary_logits"]) * mask.float()
                )

            return {
                "viterbi_paths": viterbi_paths,
                "probabilities": ai_probabilities,
                "boundary_probabilities": boundary_probs,
            }
        finally:
            if was_training:
                self.train()