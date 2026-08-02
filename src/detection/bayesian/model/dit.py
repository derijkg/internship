# models/dit_flow_model.py

import math
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelMMDChangePointLayer(nn.Module):
    """
    Differentiable Bayesian/Kernel Change-Point Detection (KCPD) Layer.
    Computes Maximum Mean Discrepancy (MMD) with an RBF kernel between 
    sliding sentence windows: W_left = [i-k, ..., i] and W_right = [i+1, ..., i+k].
    
    A high MMD score indicates an abrupt distribution shift (authorship boundary).
    """

    def __init__(self, feature_dim: int, window_size: int = 2, gamma: float = 1.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.gamma = gamma

    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Computes RBF Kernel matrix K(X, Y) = exp(-gamma * ||x - y||^2)
        X: [B, N, D], Y: [B, M, D] -> returns [B, N, M]
        """
        X_norm = (X ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
        Y_norm = (Y ** 2).sum(dim=-1, keepdim=True)  # [B, M, 1]
        
        # Distances: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
        dist_sq = X_norm + Y_norm.transpose(-1, -2) - 2 * torch.bmm(X, Y.transpose(-1, -2))
        dist_sq = torch.clamp(dist_sq, min=0.0)
        return torch.exp(-self.gamma * dist_sq)

    def forward(self, H: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        H: Hidden sequence tensor of shape [B, seq_len, D]
        mask: Boolean mask of shape [B, seq_len]
        Returns: MMD score tensor of shape [B, seq_len, 1]
        """
        B, N, D = H.shape
        mmd_scores = torch.zeros(B, N, 1, device=H.device, dtype=H.dtype)
        k = self.window_size

        for i in range(1, N):
            # Left window: max(0, i-k) to i
            left_start = max(0, i - k)
            W_left = H[:, left_start:i, :]  # [B, len_left, D]

            # Right window: i to min(N, i+k)
            right_end = min(N, i + k)
            W_right = H[:, i:right_end, :]  # [B, len_right, D]

            if W_left.shape[1] == 0 or W_right.shape[1] == 0:
                continue

            # Compute RBF Kernel matrices
            K_XX = self._rbf_kernel(W_left, W_left)     # [B, len_l, len_l]
            K_YY = self._rbf_kernel(W_right, W_right)   # [B, len_r, len_r]
            K_XY = self._rbf_kernel(W_left, W_right)    # [B, len_l, len_r]

            # MMD^2 = E[K(X,X)] + E[K(Y,Y)] - 2*E[K(X,Y)]
            mmd = K_XX.mean(dim=(1, 2)) + K_YY.mean(dim=(1, 2)) - 2 * K_XY.mean(dim=(1, 2))
            mmd_scores[:, i, 0] = torch.sqrt(torch.clamp(mmd, min=1e-8))

        if mask is not None:
            mmd_scores = mmd_scores * mask.unsqueeze(-1).float()

        return mmd_scores


class DITFlowModel(nn.Module):
    """
    DIT-Flow Architecture:
    Dynamic Information-Trajectory & Style Flow Network for Mixed-Text Attribution.
    """

    def __init__(
        self,
        input_dim: int = 862,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        window_size: int = 2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # -----------------------------------------------------------------
        # 1. Feature Projection & Dynamic Trajectory Computation
        # Feature vector per sentence is expanded to include:
        # [X_i, Velocity (v_i), Acceleration (a_i), Cosine Distance]
        # Trajectory Dimension = 3 * input_dim + 1
        # -----------------------------------------------------------------
        trajectory_dim = 3 * input_dim + 1
        self.input_projection = nn.Sequential(
            nn.Linear(trajectory_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # -----------------------------------------------------------------
        # 2. Contextual Sequence Encoder (BiLSTM)
        # -----------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # -----------------------------------------------------------------
        # 3. Bayesian Kernel Change-Point Detection Layer
        # -----------------------------------------------------------------
        self.kcpd_layer = KernelMMDChangePointLayer(
            feature_dim=hidden_dim,
            window_size=window_size
        )

        # -----------------------------------------------------------------
        # 4. Multi-Task Prediction Heads
        # -----------------------------------------------------------------
        # Head 1: Continuous AICS Predictor (AI Contribution Score ∈ [0, 1])
        self.aics_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),  # Hidden state + MMD score
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Head 2: Boundary Change-Point Predictor (Transition Probability ∈ [0, 1])
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def _compute_trajectory_derivatives(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculates sentence-to-sentence Velocity (v_i = X_i - X_{i-1}),
        Acceleration (a_i = v_i - v_{i-1}), and Cosine distance.
        X: [B, seq_len, input_dim]
        Returns: [B, seq_len, 3 * input_dim + 1]
        """
        B, N, D = X.shape

        # Velocity v_i = X_i - X_{i-1}
        X_pad_prev = F.pad(X[:, :-1, :], (0, 0, 1, 0), value=0.0)  # Shift right by 1
        velocity = X - X_pad_prev                                 # [B, N, D]

        # Acceleration a_i = v_i - v_{i-1}
        V_pad_prev = F.pad(velocity[:, :-1, :], (0, 0, 1, 0), value=0.0)
        acceleration = velocity - V_pad_prev                      # [B, N, D]

        # Cosine distance between X_i and X_{i-1}
        norm_X = F.normalize(X, p=2, dim=-1)
        norm_X_prev = F.normalize(X_pad_prev, p=2, dim=-1)
        cosine_sim = (norm_X * norm_X_prev).sum(dim=-1, keepdim=True)
        cosine_dist = 1.0 - cosine_sim                             # [B, N, 1]

        # Concatenate into full trajectory representation
        trajectory_feats = torch.cat([X, velocity, acceleration, cosine_dist], dim=-1)
        return trajectory_feats

    def forward(
        self, 
        fused_features: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        fused_features: [B, seq_len, 862]
        mask: [B, seq_len] boolean tensor (True for valid sentences, False for padded)
        """
        B, N, _ = fused_features.shape

        # 1. Compute dynamic trajectory features
        traj_feats = self._compute_trajectory_derivatives(fused_features)  # [B, N, 3*D + 1]
        proj_feats = self.input_projection(traj_feats)                      # [B, N, hidden_dim]

        # 2. Sequence Encoding via BiLSTM
        lstm_out, _ = self.lstm(proj_feats)                                 # [B, N, hidden_dim]

        # 3. Kernel Change-Point Detection (MMD Divergence)
        mmd_scores = self.kcpd_layer(lstm_out, mask=mask)                  # [B, N, 1]

        # Fuse hidden representations with MMD divergence scores
        fused_rep = torch.cat([lstm_out, mmd_scores], dim=-1)               # [B, N, hidden_dim + 1]

        # 4. Multi-Task Predictions
        aics_preds = self.aics_head(fused_rep).squeeze(-1)                  # [B, N]
        boundary_preds = self.boundary_head(fused_rep).squeeze(-1)          # [B, N]

        if mask is not None:
            aics_preds = aics_preds * mask.float()
            boundary_preds = boundary_preds * mask.float()

        return {
            "aics_preds": aics_preds,
            "boundary_preds": boundary_preds,
            "mmd_scores": mmd_scores.squeeze(-1)
        }


class DITFlowLoss(nn.Module):
    """
    Joint Multi-Task Loss for DIT-Flow:
    - MSE / Smooth L1 Loss for Continuous AICS Predictions [0.0 - 1.0]
    - BCE with Focal Loss for Sparse Boundary Transition Detection
    """

    def __init__(self, alpha_boundary: float = 1.0, focal_gamma: float = 2.0):
        super().__init__()
        self.alpha_boundary = alpha_boundary
        self.focal_gamma = focal_gamma
        self.mse_loss = nn.SmoothL1Loss(reduction="none")

    def _focal_bce_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal Binary Cross Entropy for imbalanced boundary markers."""
        eps = 1e-7
        preds = torch.clamp(preds, eps, 1.0 - eps)
        bce = - (targets * torch.log(preds) + (1.0 - targets) * torch.log(1.0 - preds))
        p_t = torch.where(targets == 1.0, preds, 1.0 - preds)
        focal_factor = (1.0 - p_t) ** self.focal_gamma
        return focal_factor * bce

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        boundaries: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        preds: Dictionary from model forward pass
        labels: Ground truth continuous AICS [B, N]
        boundaries: Ground truth binary boundaries [B, N]
        mask: Boolean mask [B, N]
        """
        aics_preds = preds["aics_preds"]
        boundary_preds = preds["boundary_preds"]

        # Masked AICS Loss
        loss_aics_element = self.mse_loss(aics_preds, labels)
        loss_aics = (loss_aics_element * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

        # Masked Boundary Focal Loss
        loss_boundary_element = self._focal_bce_loss(boundary_preds, boundaries)
        loss_boundary = (loss_boundary_element * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

        # Joint Loss
        total_loss = loss_aics + self.alpha_boundary * loss_boundary

        return total_loss, {
            "total_loss": float(total_loss.item()),
            "aics_loss": float(loss_aics.item()),
            "boundary_loss": float(loss_boundary.item())
        }