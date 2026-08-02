import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional, Tuple, List, Dict, Any


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size must be odd to preserve sequence length, got {kernel_size}"
        padding = (kernel_size - 1) * dilation // 2
        
        num_groups = 8 if channels % 8 == 0 else (4 if channels % 4 == 0 else 1)

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)  # <-- Fixed line 19 here
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        if mask is not None:
            out = out * mask.unsqueeze(1)
        out = self.dropout(self.act(self.norm1(out)))
        
        out = self.conv2(out)
        if mask is not None:
            out = out * mask.unsqueeze(1)
        out = self.norm2(out)
        
        out = self.act(out + residual)
        if mask is not None:
            out = out * mask.unsqueeze(1)
            
        return out


class BoundaryDetectorTCN(nn.Module):
    def __init__(self, input_dim: int = 31, hidden_dim: int = 64, num_layers: int = 4, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(channels=hidden_dim, kernel_size=kernel_size, dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x.transpose(1, 2)
        h = self.input_proj(h)

        for block in self.blocks:
            h = block(h, mask=mask)

        hidden_feats = h.transpose(1, 2)
        boundary_probs = self.boundary_head(hidden_feats)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).to(boundary_probs.dtype)
            boundary_probs = boundary_probs * mask_expanded
            hidden_feats = hidden_feats * mask_expanded

        return boundary_probs, hidden_feats


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    Pulls same-author segment embeddings together on the unit hypersphere
    and pushes different-author embeddings apart.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        N = embeddings.size(0)
        if N < 2:
            return torch.tensor(0.0, device=device)

        embeddings = F.normalize(embeddings, p=2, dim=-1)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        logits_mask = torch.ones((N, N), dtype=torch.bool, device=device)
        logits_mask.fill_diagonal_(False)

        labels_view = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels_view, labels_view.T) & logits_mask

        # Max-trick computed strictly over off-diagonal entries
        sim_matrix_masked = sim_matrix.masked_fill(~logits_mask, -1e9)
        logits_max, _ = torch.max(sim_matrix_masked, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)

        pos_counts = pos_mask.float().sum(dim=1)
        valid_anchors = pos_counts > 0

        if not valid_anchors.any():
            return torch.tensor(0.0, device=device)

        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1)[valid_anchors] / pos_counts[valid_anchors]
        loss = -mean_log_prob_pos.mean()
        return loss


class BoundaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)
        if preds.dim() == 2:
            preds = preds.unsqueeze(-1)

        preds = torch.clamp(preds, self.eps, 1.0 - self.eps)
        
        loss_pos = -self.alpha * ((1.0 - preds) ** self.gamma) * torch.log(preds) * targets
        loss_neg = -(1.0 - self.alpha) * (preds ** self.gamma) * torch.log(1.0 - preds) * (1.0 - targets)
        
        loss = loss_pos + loss_neg

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).to(loss.dtype) if mask.dim() == 2 else mask.to(loss.dtype)
            return (loss * mask_expanded).sum() / torch.clamp(mask_expanded.sum(), min=1.0)
        return loss.mean()


class SegmentProfiler(nn.Module):
    def __init__(
        self, 
        feature_dim: int = 31, 
        hidden_dim: int = 64, 
        projection_dim: int = 32, 
        dense_dim: int = 0,         # Set to 1024 if passing RobBERT embeddings
        macro_start_idx: int = 3
    ):
        super().__init__()
        self.macro_start_idx = macro_start_idx
        self.dense_dim = dense_dim
        
        combined_dim = hidden_dim + feature_dim + 1 + dense_dim

        self.segment_encoder = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )

    def _find_peaks(self, boundary_probs: torch.Tensor, threshold: float = 0.5, min_dist: int = 3) -> List[int]:
        """GPU-native NMS peak finding with safe 1D dimension flattening."""
        probs = boundary_probs.squeeze(-1)  # Shape [seq_len]
        seq_len = probs.size(0)
        if seq_len == 0:
            return []

        kernel_size = 2 * min_dist + 1
        probs_3d = probs.view(1, 1, seq_len)
        
        # Identify local maxima using GPU 1D max pooling
        max_pooled = F.max_pool1d(probs_3d, kernel_size=kernel_size, stride=1, padding=min_dist)
        is_peak = (probs_3d == max_pooled) & (probs_3d > threshold)
        
        # Explicit .view(-1) prevents 0D scalar tensor crash on seq_len == 1 or single peak
        peak_indices = is_peak.view(-1).nonzero(as_tuple=False)
        if peak_indices.numel() == 0:
            return []

        peak_list = peak_indices.view(-1).tolist()
        
        filtered_peaks = []
        for idx in peak_list:
            if not filtered_peaks or (idx - filtered_peaks[-1]) >= min_dist:
                filtered_peaks.append(idx)
        return filtered_peaks

    def extract_single_document_segments(
        self,
        seq_features: torch.Tensor,
        hidden_feats: torch.Tensor,
        boundary_probs: torch.Tensor,
        valid_len: int,
        threshold: float = 0.5,
        dense_embs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        device = seq_features.device

        valid_features = seq_features[:valid_len]
        valid_hidden = hidden_feats[:valid_len]
        valid_b_probs = boundary_probs[:valid_len]

        if dense_embs is not None and self.dense_dim > 0:
            if dense_embs.size(0) == 1:
                token_dense = dense_embs.repeat(valid_len, 1)
            else:
                token_dense = F.interpolate(
                    dense_embs.unsqueeze(0).transpose(1, 2),
                    size=valid_len,
                    mode="nearest"
                ).squeeze(0).transpose(0, 1)
        else:
            token_dense = None

        peaks = self._find_peaks(valid_b_probs, threshold=threshold)
        split_points = sorted(list(set([0] + peaks + [valid_len])))

        doc_macro_baseline = torch.mean(valid_features[:, self.macro_start_idx:], dim=0)

        segment_embeddings, delta_styles, segment_spans = [], [], []

        for i in range(len(split_points) - 1):
            start_idx, end_idx = split_points[i], split_points[i + 1]
            if start_idx == end_idx:
                continue

            seg_raw = valid_features[start_idx:end_idx]
            seg_hidden = valid_hidden[start_idx:end_idx]

            pooled_raw = torch.mean(seg_raw, dim=0)
            pooled_hidden = torch.mean(seg_hidden, dim=0)

            seg_macro_mean = pooled_raw[self.macro_start_idx:]
            delta_sty = torch.norm(seg_macro_mean - doc_macro_baseline, p=2, keepdim=True)

            feature_list = [pooled_hidden, pooled_raw, delta_sty]

            if token_dense is not None:
                seg_dense = token_dense[start_idx:end_idx]
                pooled_dense = torch.mean(seg_dense, dim=0)
                feature_list.append(pooled_dense)

            combined = torch.cat(feature_list, dim=-1)
            seg_emb = self.segment_encoder(combined)

            segment_embeddings.append(seg_emb)
            delta_styles.append(delta_sty)
            segment_spans.append((start_idx, end_idx))

        if not segment_embeddings:
            pooled_raw = torch.mean(valid_features, dim=0)
            pooled_hidden = torch.mean(valid_hidden, dim=0)
            delta_sty = torch.tensor([0.0], device=device)
            
            feature_list = [pooled_hidden, pooled_raw, delta_sty]
            if token_dense is not None:
                feature_list.append(torch.mean(token_dense, dim=0))

            combined = torch.cat(feature_list, dim=-1)
            segment_embeddings.append(self.segment_encoder(combined))
            delta_styles.append(delta_sty)
            segment_spans.append((0, valid_len))

        return torch.stack(segment_embeddings), torch.stack(delta_styles), segment_spans

    def forward(
        self,
        batch_seq_features: torch.Tensor,
        batch_hidden_feats: torch.Tensor,
        batch_boundary_probs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        batch_dense_embs: Optional[List[torch.Tensor]] = None
    ) -> List[Dict[str, Any]]:
        batch_size = batch_seq_features.size(0)
        batch_results = []

        for b in range(batch_size):
            valid_len = int(mask[b].sum().item()) if mask is not None else batch_seq_features.size(1)
            dense_emb = batch_dense_embs[b] if batch_dense_embs is not None else None

            seg_embs, deltas, spans = self.extract_single_document_segments(
                seq_features=batch_seq_features[b],
                hidden_feats=batch_hidden_feats[b],
                boundary_probs=batch_boundary_probs[b],
                valid_len=valid_len,
                threshold=threshold,
                dense_embs=dense_emb
            )

            batch_results.append({
                "segment_embeddings": seg_embs,
                "delta_styles": deltas,
                "spans": spans,
                "num_segments": seg_embs.size(0)
            })

        return batch_results


class GraphDiarizer(nn.Module):
    def __init__(
        self,
        in_dim: int = 32,
        hidden_dim: int = 32,
        out_dim: int = 16,
        num_classes: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        affinity_threshold: float = 0.2,
        alpha: float = 2.0
    ):
        super().__init__()
        self.in_dim = in_dim
        self.affinity_threshold = affinity_threshold
        self.alpha = alpha

        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout, add_self_loops=False, edge_dim=1)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)

        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout, add_self_loops=False, edge_dim=1)
        self.norm2 = nn.LayerNorm(out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, num_classes)
        )

    def build_dynamic_graph(self, seg_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        K = seg_embeddings.size(0)
        device = seg_embeddings.device

        if K == 1:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
            edge_attr = torch.tensor([[1.0]], dtype=torch.float, device=device)
            return edge_index, edge_attr

        diff = seg_embeddings.unsqueeze(1) - seg_embeddings.unsqueeze(0)
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)

        similarity = torch.exp(-self.alpha * dist_matrix)
        adjacency_mask = (similarity >= self.affinity_threshold) | torch.eye(K, dtype=torch.bool, device=device)
        
        edge_index = adjacency_mask.nonzero(as_tuple=False).t().contiguous()
        edge_attr = similarity[edge_index[0], edge_index[1]].unsqueeze(-1)

        return edge_index, edge_attr

    def forward_single_document(self, seg_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index, edge_attr = self.build_dynamic_graph(seg_embeddings)

        h = self.gat1(seg_embeddings, edge_index, edge_attr=edge_attr)
        h = self.norm1(h)
        h = F.gelu(h)

        h_out = self.gat2(h, edge_index, edge_attr=edge_attr)
        h_out = self.norm2(h_out)

        author_embeddings = F.normalize(h_out, p=2, dim=-1)
        logits = self.classifier(author_embeddings)

        return logits, author_embeddings

    def forward(self, batch_segment_data: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        batch_results = []

        for doc in batch_segment_data:
            seg_embs = doc["segment_embeddings"]
            logits, author_embs = self.forward_single_document(seg_embs)

            batch_results.append({
                "logits": logits,
                "author_embeddings": author_embs,
                "num_segments": seg_embs.size(0)
            })

        return batch_results


class HISDiarizer(nn.Module):
    """
    Hierarchical Information-Stylometric Diarization Network (HIS-Diarizer).
    """
    def __init__(
        self,
        feature_dim: int = 31,
        hidden_dim: int = 64,
        num_layers: int = 4,
        tcn_dropout: float = 0.1,
        segment_proj_dim: int = 32,
        dense_dim: int = 0,
        gat_hidden_dim: int = 32,
        graph_out_dim: int = 16,
        num_classes: int = 2,
        gat_heads: int = 4,
        gat_dropout: float = 0.1,
        affinity_threshold: float = 0.2,
        alpha_scale: float = 2.0
    ):
        super().__init__()
        
        self.boundary_detector = BoundaryDetectorTCN(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=tcn_dropout
        )
        
        self.profiler = SegmentProfiler(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            projection_dim=segment_proj_dim,
            dense_dim=dense_dim
        )
        
        self.graph_diarizer = GraphDiarizer(
            in_dim=segment_proj_dim,
            hidden_dim=gat_hidden_dim,
            out_dim=graph_out_dim,
            num_classes=num_classes,
            heads=gat_heads,
            dropout=gat_dropout,
            affinity_threshold=affinity_threshold,
            alpha=alpha_scale
        )

    def forward_stage1(self, seq_features: torch.Tensor, mask: Optional[torch.Tensor] = None):
        boundary_probs, hidden_feats = self.boundary_detector(seq_features, mask=mask)
        return boundary_probs, hidden_feats

    def forward_stage2(
        self,
        seq_features: torch.Tensor,
        hidden_feats: torch.Tensor,
        boundary_probs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        batch_dense_embs: Optional[List[torch.Tensor]] = None
    ):
        segment_data = self.profiler(
            batch_seq_features=seq_features,
            batch_hidden_feats=hidden_feats,
            batch_boundary_probs=boundary_probs,
            mask=mask,
            threshold=threshold,
            batch_dense_embs=batch_dense_embs
        )
        diarization_results = self.graph_diarizer(segment_data)
        return diarization_results, segment_data

    def forward(
        self, 
        seq_features: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        threshold: float = 0.5,
        batch_dense_embs: Optional[List[torch.Tensor]] = None
    ):
        boundary_probs, hidden_feats = self.forward_stage1(seq_features, mask=mask)
        diarization_results, segment_data = self.forward_stage2(
            seq_features, hidden_feats, boundary_probs, mask=mask, threshold=threshold, batch_dense_embs=batch_dense_embs
        )
        return boundary_probs, diarization_results, segment_data