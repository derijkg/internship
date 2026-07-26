# training/calibration.py

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_curve


class NeymanPearsonCalibrator:
    """Calculates decision threshold tau* that enforces a strict upper bound on

    False Positive Rate (FPR <= target_fpr).
    """

    def __init__(self, target_fpr: float = 0.01):
        self.target_fpr = target_fpr
        self.optimal_threshold = 0.50

    def fit(
        self, y_true: np.ndarray, y_probs: np.ndarray
    ) -> float:
        """Solves for tau* on Out-Of-Fold continuous marginal probabilities."""
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)

        # Filter thresholds where FPR <= target_fpr
        valid_indices = np.where(fpr <= self.target_fpr)[0]

        if len(valid_indices) > 0:
            # Pick highest TPR (which corresponds to lowest valid threshold)
            best_idx = valid_indices[-1]
            self.optimal_threshold = float(np.clip(thresholds[best_idx], 0.0, 1.0))
        else:
            self.optimal_threshold = 0.50

        print(
            f"-> Neyman-Pearson Calibrated Threshold (FPR <= {self.target_fpr*100:.1f}%): τ* = {self.optimal_threshold:.6f}"
        )
        return self.optimal_threshold

    def predict(
        self, y_probs: np.ndarray, threshold: float = None
    ) -> np.ndarray:
        tau = threshold if threshold is not None else self.optimal_threshold
        return (y_probs >= tau).astype(int)