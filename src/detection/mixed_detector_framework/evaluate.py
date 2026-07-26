# evaluate.py

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def evaluate_mixed_authorship_performance(
    all_y_true: List[np.ndarray],
    all_y_probs: List[np.ndarray],
    all_b_true: List[np.ndarray],
    all_b_probs: List[np.ndarray],
    threshold: float = 0.50,
) -> Dict[str, float]:
    """Computes comprehensive multi-level evaluation metrics."""
    flat_y_true = np.concatenate(all_y_true)
    flat_y_probs = np.concatenate(all_y_probs)
    flat_y_pred = (flat_y_probs >= threshold).astype(int)

    flat_b_true = np.concatenate(all_b_true)
    flat_b_probs = np.concatenate(all_b_probs)
    
    best_b_thresh = 0.50
    best_b_f1 = 0.0

    for t in np.arange(0.10, 0.90, 0.05):
        b_pred_t = (flat_b_probs >= t).astype(int)
        f1_t = f1_score(flat_b_true, b_pred_t, pos_label=1, zero_division=0)
        if f1_t > best_b_f1:
            best_b_f1 = f1_t
            best_b_thresh = t

    flat_b_pred = (flat_b_probs >= best_b_thresh).astype(int)

    # 1. Sentence-Level Metrics
    sent_prec = precision_score(flat_y_true, flat_y_pred, pos_label=1, zero_division=0)
    sent_rec = recall_score(flat_y_true, flat_y_pred, pos_label=1, zero_division=0)
    sent_f1 = f1_score(flat_y_true, flat_y_pred, pos_label=1, zero_division=0)
    sent_auc = roc_auc_score(flat_y_true, flat_y_probs)

    # 2. Boundary Transition Metrics
    bound_prec = precision_score(flat_b_true, flat_b_pred, pos_label=1, zero_division=0)
    bound_rec = recall_score(flat_b_true, flat_b_pred, pos_label=1, zero_division=0)
    bound_f1 = f1_score(flat_b_true, flat_b_pred, pos_label=1, zero_division=0)

    # 3. Document-Level Span IoU & AI Ratio MAE
    ious = []
    ratio_errors = []

    for y_t, y_p_prob in zip(all_y_true, all_y_probs):
        y_p = (y_p_prob >= threshold).astype(int)

        # Span IoU
        intersection = np.sum((y_t == 1) & (y_p == 1))
        union = np.sum((y_t == 1) | (y_p == 1))
        iou = (intersection / union) if union > 0 else 1.0
        ious.append(iou)

        # AI Ratio MAE
        true_ratio = np.mean(y_t)
        pred_ratio = np.mean(y_p)
        ratio_errors.append(abs(true_ratio - pred_ratio))

    mean_iou = float(np.mean(ious))
    mean_ratio_mae = float(np.mean(ratio_errors))

    return {
        "sent_precision_ai": sent_prec,
        "sent_recall_ai": sent_rec,
        "sent_f1_ai": sent_f1,
        "sent_roc_auc": sent_auc,
        "boundary_precision": bound_prec,
        "boundary_recall": bound_rec,
        "boundary_f1": bound_f1,
        "span_iou": mean_iou,
        "ai_ratio_mae": mean_ratio_mae,
    }