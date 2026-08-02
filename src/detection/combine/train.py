import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from his import HISDiarizer, BoundaryFocalLoss, SupervisedContrastiveLoss

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = str(BASE_DIR / "processed_features")
DEFAULT_CKPT_DIR = str(BASE_DIR / "checkpoints")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# =====================================================================
# EARLY STOPPER CLASS
# =====================================================================

class EarlyStopper:
    def __init__(self, patience: int = 5, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.best_state_dict = None

    def check_and_update(self, score: float, model: nn.Module) -> bool:
        improved = (score > self.best_score) if self.mode == "max" else (score < self.best_score)
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best_weights(self, model: nn.Module):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


# =====================================================================
# DATA SPLITTING (GROUP-BASED TO PREVENT LEAKAGE - 3 FOLDS)
# =====================================================================

def split_dataset_kfold(data_dir: str, n_splits: int = 3, test_ratio: float = 0.1):
    data_path = Path(data_dir)
    print(f"\n--- Creating {n_splits}-Fold Group Cross-Validation Splits in {data_path} ---")
    
    metadata_df = pd.read_parquet(data_path / "synthetic_metadata.parquet")
    features_dict = torch.load(data_path / "precomputed_features.pt", weights_only=False)

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(metadata_df, groups=metadata_df['original_id']))

    train_val_df = metadata_df.iloc[train_val_idx].copy().reset_index(drop=True)
    test_df = metadata_df.iloc[test_idx].copy().reset_index(drop=True)

    test_df.to_parquet(data_path / "test_metadata.parquet", index=False)
    torch.save(
        {syn_id: features_dict[syn_id] for syn_id in test_df["synthetic_id"] if syn_id in features_dict},
        data_path / "test_features.pt"
    )

    gkf = GroupKFold(n_splits=n_splits)
    for fold, (t_idx, v_idx) in enumerate(gkf.split(train_val_df, groups=train_val_df['original_id'])):
        fold_train_df = train_val_df.iloc[t_idx]
        fold_val_df = train_val_df.iloc[v_idx]

        fold_train_df.to_parquet(data_path / f"fold_{fold}_train_metadata.parquet", index=False)
        fold_val_df.to_parquet(data_path / f"fold_{fold}_val_metadata.parquet", index=False)

        torch.save(
            {syn_id: features_dict[syn_id] for syn_id in fold_train_df["synthetic_id"] if syn_id in features_dict},
            data_path / f"fold_{fold}_train_features.pt"
        )
        torch.save(
            {syn_id: features_dict[syn_id] for syn_id in fold_val_df["synthetic_id"] if syn_id in features_dict},
            data_path / f"fold_{fold}_val_features.pt"
        )
        print(f" -> Fold {fold}: Train={len(fold_train_df)} samples | Val={len(fold_val_df)} samples")

    print(f"{n_splits}-Fold Group Cross-Validation splits created cleanly!\n")


# =====================================================================
# DATASET & COLLATION
# =====================================================================

class HISDiarizerDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "fold_0_train"):
        self.data_dir = Path(data_dir)
        self.split = split
        
        metadata_path = self.data_dir / f"{split}_metadata.parquet"
        features_path = self.data_dir / f"{split}_features.pt"
        
        if not metadata_path.exists() or not features_path.exists():
            metadata_path = self.data_dir / "synthetic_metadata.parquet"
            features_path = self.data_dir / "precomputed_features.pt"

        self.metadata_df = pd.read_parquet(metadata_path)
        self.features_dict = torch.load(features_path, weights_only=False)
        self.samples = list(self.metadata_df["synthetic_id"])

        self.label_map = {
            "Human": 0, 
            "qwen3.6:27b": 1, 
            "gemma4:e4b": 1, 
            "qwen3.5:4b": 1, 
            "gemma4:26b": 1
        }

    def __len__(self) -> int:
        return len(self.samples)

    def get_class_counts(self) -> torch.Tensor:

        counts = torch.zeros(2, dtype=torch.float32)
        for idx in range(len(self.samples)):
            syn_id = self.samples[idx]
            feat_data = self.features_dict[syn_id]
            seq_len = feat_data["fused_sequence"].size(0)
            meta_row = self.metadata_df.iloc[idx]
            
            _, token_author_ids = self._build_targets_and_token_labels(
                seq_len=seq_len, sentences=meta_row["sentences"], labels=meta_row["labels"]
            )
            for c in range(2):
                counts[c] += (token_author_ids == c).sum().item()
        return counts

    def _build_targets_and_token_labels(self, seq_len: int, sentences: Any, labels: Any):

        boundary_targets = torch.zeros((seq_len, 1), dtype=torch.float32)
        token_author_ids = torch.zeros(seq_len, dtype=torch.long)

        sentences_list = list(sentences) if isinstance(sentences, (list, np.ndarray, pd.Series)) else []
        labels_list = list(labels) if isinstance(labels, (list, np.ndarray, pd.Series)) else []

        if len(sentences_list) == 0 or len(labels_list) == 0:
            return boundary_targets, token_author_ids

        sent_lengths = [len(str(s)) for s in sentences_list]
        total_chars = max(sum(sent_lengths), 1)

        cum_chars = 0
        prev_end_tok = 0

        for i, (sent_len, lbl) in enumerate(zip(sent_lengths, labels_list)):
            cum_chars += sent_len
            if i == len(sentences_list) - 1:
                end_tok = seq_len
            else:
                end_tok = int(round((cum_chars / total_chars) * seq_len))
                end_tok = min(max(end_tok, prev_end_tok + 1), seq_len)

            start_tok = prev_end_tok
            author_id = self.label_map.get(str(lbl), 1)
            token_author_ids[start_tok:end_tok] = author_id

            if i < len(labels_list) - 1 and labels_list[i] != labels_list[i + 1]:
                b_idx = max(0, min(end_tok - 1, seq_len - 1))
                boundary_targets[b_idx, 0] = 1.0

            prev_end_tok = end_tok

        return boundary_targets, token_author_ids

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        syn_id = self.samples[idx]
        feat_data = self.features_dict[syn_id]
        meta_row = self.metadata_df.iloc[idx]

        fused_seq = feat_data["fused_sequence"]
        seq_len = fused_seq.size(0)

        boundary_targets, token_author_ids = self._build_targets_and_token_labels(
            seq_len=seq_len, sentences=meta_row["sentences"], labels=meta_row["labels"]
        )

        return {
            "synthetic_id": syn_id,
            "fused_sequence": fused_seq,
            "boundary_targets": boundary_targets,
            "token_author_ids": token_author_ids,
            "seq_len": seq_len
        }


def pad_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_size = len(batch)
    max_len = max(item["seq_len"] for item in batch)
    feature_dim = batch[0]["fused_sequence"].size(-1)

    padded_features = torch.zeros(batch_size, max_len, feature_dim)
    padded_targets = torch.zeros(batch_size, max_len, 1)
    padded_token_author_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len)

    for i, item in enumerate(batch):
        l = item["seq_len"]
        padded_features[i, :l] = item["fused_sequence"]
        padded_targets[i, :l] = item["boundary_targets"]
        padded_token_author_ids[i, :l] = item["token_author_ids"]
        mask[i, :l] = 1.0

    return {
        "fused_sequence": padded_features,
        "boundary_targets": padded_targets,
        "token_author_ids": padded_token_author_ids,
        "mask": mask
    }


# =====================================================================
# DYNAMIC THRESHOLD CALIBRATION & OPTIMAL MATCHING METRICS
# =====================================================================

def compute_boundary_counts_with_tolerance(
    pred_probs: torch.Tensor,
    target_boundaries: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
    tolerance: int = 3
) -> Tuple[int, int, int]:
    tp, fp, fn = 0, 0, 0
    batch_size = pred_probs.size(0)

    for b in range(batch_size):
        valid_len = int(mask[b].sum().item())
        if valid_len == 0:
            continue
        p_probs = pred_probs[b, :valid_len].squeeze(-1).detach().cpu().numpy()
        t_bounds = target_boundaries[b, :valid_len].squeeze(-1).detach().cpu().numpy()

        pred_indices = np.where(p_probs > threshold)[0]
        true_indices = np.where(t_bounds > 0.5)[0]

        n_pred = len(pred_indices)
        n_true = len(true_indices)

        if n_pred == 0 and n_true == 0:
            continue
        elif n_pred > 0 and n_true == 0:
            fp += n_pred
        elif n_pred == 0 and n_true > 0:
            fn += n_true
        else:
            # Hungarian algorithm for optimal bipartite matching
            cost_matrix = np.abs(pred_indices[:, None] - true_indices[None, :])
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_preds = set()
            matched_trues = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= tolerance:
                    tp += 1
                    matched_preds.add(r)
                    matched_trues.add(c)

            fp += (n_pred - len(matched_preds))
            fn += (n_true - len(matched_trues))

    return tp, fp, fn


def calibrate_optimal_boundary_threshold(model, val_loader, device) -> float:
    model.eval()
    candidate_thresholds = np.arange(0.20, 0.80, 0.05)
    best_tau = 0.5
    best_f1 = -1.0

    print(" -> Calibrating optimal boundary decision threshold (tau*) on validation set...")

    with torch.no_grad():
        all_pred_probs = []
        all_targets = []
        all_masks = []

        # 1. Forward pass over validation set; store batch tensors
        for batch in val_loader:
            features = batch["fused_sequence"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["boundary_targets"].to(device)

            boundary_probs, _ = model.forward_stage1(features, mask=mask)
            all_pred_probs.append(boundary_probs.cpu())
            all_targets.append(targets.cpu())
            all_masks.append(mask.cpu())

        # 2. Iterate over candidate thresholds without attempting invalid torch.cat across variable seq lengths
        for tau in candidate_thresholds:
            total_tp, total_fp, total_fn = 0, 0, 0
            
            for b_probs, b_targets, b_mask in zip(all_pred_probs, all_targets, all_masks):
                tp, fp, fn = compute_boundary_counts_with_tolerance(
                    b_probs, b_targets, b_mask, threshold=tau, tolerance=3
                )
                total_tp += tp
                total_fp += fp
                total_fn += fn

            precision = total_tp / max(total_tp + total_fp, 1e-9)
            recall = total_tp / max(total_tp + total_fn, 1e-9)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-9)

            if f1 > best_f1:
                best_f1 = f1
                best_tau = float(tau)

    print(f" -> Optimal Threshold Calibrated: tau* = {best_tau:.2f} (Val Boundary F1: {best_f1:.4f})")
    return best_tau


def compute_stage1_boundary_f1(model, val_loader, device, threshold=0.5) -> float:

    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            features, mask = batch["fused_sequence"].to(device), batch["mask"].to(device)
            target_boundaries = batch["boundary_targets"].to(device)

            boundary_probs, _ = model.forward_stage1(features, mask=mask)
            tp, fp, fn = compute_boundary_counts_with_tolerance(
                boundary_probs, target_boundaries, mask, threshold=threshold, tolerance=3
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn

    precision = total_tp / max(total_tp + total_fp, 1e-9)
    recall = total_tp / max(total_tp + total_fn, 1e-9)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-9)
    return float(f1)


def compute_validation_score(model, val_loader, device, threshold=0.5) -> Dict[str, float]:

    model.eval()
    all_preds, all_trues = [], []
    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            features, mask = batch["fused_sequence"].to(device), batch["mask"].to(device)
            target_boundaries = batch["boundary_targets"].to(device)
            token_author_ids = batch["token_author_ids"].to(device)

            boundary_probs, hidden_feats = model.forward_stage1(features, mask=mask)
            diarization_results, segment_data_batch = model.forward_stage2(
                features, hidden_feats, boundary_probs, mask=mask, threshold=threshold
            )

            tp, fp, fn = compute_boundary_counts_with_tolerance(
                boundary_probs, target_boundaries, mask, threshold=threshold, tolerance=3
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn

            for doc_idx, doc_res in enumerate(diarization_results):
                valid_len = int(mask[doc_idx].sum().item())
                if valid_len == 0:
                    continue

                true_tokens = token_author_ids[doc_idx, :valid_len].cpu().numpy()
                pred_tokens = np.zeros(valid_len, dtype=int)

                pred_logits = doc_res["logits"]
                spans = segment_data_batch[doc_idx]["spans"]

                if pred_logits.size(0) > 0 and len(spans) > 0:
                    pred_classes = torch.argmax(pred_logits, dim=-1).cpu().numpy()
                    for seg_idx, (st, end) in enumerate(spans):
                        if seg_idx < len(pred_classes):
                            st_c, end_c = min(st, valid_len), min(end, valid_len)
                            if st_c < end_c:
                                pred_tokens[st_c:end_c] = pred_classes[seg_idx]

                all_preds.extend(pred_tokens)
                all_trues.extend(true_tokens)

    precision = total_tp / max(total_tp + total_fp, 1e-9)
    recall = total_tp / max(total_tp + total_fn, 1e-9)
    mean_boundary_f1 = 2 * (precision * recall) / max(precision + recall, 1e-9)

    macro_f1 = f1_score(all_trues, all_preds, average="macro", zero_division=0) if all_trues else 0.0
    combined_score = 0.5 * mean_boundary_f1 + 0.5 * macro_f1

    return {"boundary_f1": mean_boundary_f1, "macro_f1": macro_f1, "combined_score": combined_score}


# =====================================================================
# TRAINING LOOPS (WITH SAFE BATCH-WIDE CE + SUPCON LOSS)
# =====================================================================

def train_stage1_boundary_detector(
    model, train_loader, val_loader, optimizer, criterion, device, 
    max_epochs: int = 25, patience: int = 5, verbose: bool = True
) -> float:
    print("\n==================================================")
    print(f"STARTING STAGE 1: TCN Boundary Detector (Max {max_epochs} Epochs, Patience {patience})")
    print("==================================================")

    stopper = EarlyStopper(patience=patience, mode="max")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    for epoch in range(max_epochs):
        model.boundary_detector.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Stage 1 Epoch {epoch+1}/{max_epochs}") if verbose else train_loader

        for batch in pbar:
            features, targets, mask = batch["fused_sequence"].to(device), batch["boundary_targets"].to(device), batch["mask"].to(device)
            optimizer.zero_grad()
            boundary_probs, _ = model.forward_stage1(features, mask=mask)
            
            loss = criterion(boundary_probs, targets, mask=mask)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.boundary_detector.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        val_b_f1 = compute_stage1_boundary_f1(model, val_loader, device, threshold=0.5)
        scheduler.step(val_b_f1)

        if verbose:
            print(f"  -> Epoch {epoch+1} Train Loss: {total_loss/max(len(train_loader),1):.5f} | Val Boundary F1: {val_b_f1:.4f}")

        if stopper.check_and_update(val_b_f1, model):
            if verbose:
                print(f"Early stopping triggered at Stage 1 Epoch {epoch+1}! Best Val Boundary F1: {stopper.best_score:.4f}")
            break

    stopper.restore_best_weights(model)
    calibrated_tau = calibrate_optimal_boundary_threshold(model, val_loader, device)

    for param in model.boundary_detector.parameters():
        param.requires_grad = False

    return calibrated_tau


def train_stage2_graph_diarizer(
    model, train_loader, val_loader, optimizer, criterion_ce, criterion_supcon, device, 
    max_epochs: int = 25, patience: int = 5, threshold: float = 0.5, lambda_sup: float = 0.5, verbose: bool = True
):
    print("==================================================")
    print(f"STARTING STAGE 2: Graph Diarizer with Joint CE + SupCon Loss (Threshold={threshold:.2f})")
    print("==================================================")

    model.boundary_detector.eval()
    stopper = EarlyStopper(patience=patience, mode="max")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    for epoch in range(max_epochs):
        model.profiler.train()
        model.graph_diarizer.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Stage 2 Epoch {epoch+1}/{max_epochs}") if verbose else train_loader

        for batch in pbar:
            features, mask, token_author_ids = batch["fused_sequence"].to(device), batch["mask"].to(device), batch["token_author_ids"].to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                boundary_probs, hidden_feats = model.forward_stage1(features, mask=mask)

            diarization_results, segment_data_batch = model.forward_stage2(
                features, hidden_feats, boundary_probs, mask=mask, threshold=threshold
            )

            batch_ce_loss = torch.tensor(0.0, device=device)
            valid_docs = 0
            all_batch_embs = []
            all_batch_labels = []

            for doc_idx, doc_res in enumerate(diarization_results):
                pred_logits = doc_res["logits"]
                author_embs = doc_res["author_embeddings"]
                spans = segment_data_batch[doc_idx]["spans"]

                if pred_logits.size(0) == 0 or len(spans) == 0:
                    continue

                target_segment_labels = []
                for start_tok, end_tok in spans:
                    seg_tokens = token_author_ids[doc_idx, start_tok:end_tok]
                    if seg_tokens.numel() > 0:
                        majority_label = torch.mode(seg_tokens).values.item()
                    else:
                        majority_label = 0
                    target_segment_labels.append(majority_label)

                true_labels_tensor = torch.tensor(target_segment_labels, dtype=torch.long, device=device)
                l_ce = criterion_ce(pred_logits, true_labels_tensor)

                batch_ce_loss = batch_ce_loss + l_ce
                valid_docs += 1

                all_batch_embs.append(author_embs)
                all_batch_labels.append(true_labels_tensor)

            if valid_docs > 0:
                batch_loss = batch_ce_loss / valid_docs

                # Safely aggregate SupCon across entire mini-batch of segments
                if len(all_batch_embs) > 0:
                    cat_embs = torch.cat(all_batch_embs, dim=0)
                    cat_labels = torch.cat(all_batch_labels, dim=0)
                    if cat_embs.size(0) > 1 and len(torch.unique(cat_labels)) > 1:
                        l_supcon = criterion_supcon(cat_embs, cat_labels)
                        if not torch.isnan(l_supcon):
                            batch_loss = batch_loss + (lambda_sup * l_supcon)

                batch_loss.backward()
                nn.utils.clip_grad_norm_(list(model.profiler.parameters()) + list(model.graph_diarizer.parameters()), max_norm=1.0)
                optimizer.step()
                total_loss += batch_loss.item()

        val_metrics = compute_validation_score(model, val_loader, device, threshold=threshold)
        val_score = val_metrics["combined_score"]
        scheduler.step(val_score)

        if verbose:
            print(f"  -> Epoch {epoch+1} Train Loss: {total_loss/max(len(train_loader),1):.5f} | Val Combined Score: {val_score:.4f} (Macro F1: {val_metrics['macro_f1']:.4f})")

        if stopper.check_and_update(val_score, model):
            if verbose:
                print(f"Early stopping triggered at Stage 2 Epoch {epoch+1}! Best Val Score: {stopper.best_score:.4f}")
            break

    stopper.restore_best_weights(model)
    print("Stage 2 Training Complete! Restored best Graph Diarizer weights.\n")


# =====================================================================
# HYPERPARAMETER TUNING (OPTUNA)
# =====================================================================

BEST_PARAMS_FILE = "best_hyperparameters.json"

def run_hyperparameter_tuning(
    data_dir: str, 
    ckpt_dir: Path, 
    n_trials: int = 15, 
    device: str = "cuda",
    epochs: int = 5,       # Add epochs parameter
    patience: int = 3      # Add patience parameter
    ):
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not installed. Run 'pip install optuna' to run tuning.")

    print("\n==================================================")
    print(f"STARTING HYPERPARAMETER TUNING ({n_trials} Trials on Fold 0)")
    print("==================================================")

    train_dataset = HISDiarizerDataset(data_dir=data_dir, split="fold_0_train")
    val_dataset = HISDiarizerDataset(data_dir=data_dir, split="fold_0_val")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=pad_collate_fn)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 2, 4),
            "tcn_dropout": trial.suggest_float("tcn_dropout", 0.05, 0.3),
            "segment_proj_dim": trial.suggest_categorical("segment_proj_dim", [16, 32, 64]),
            "gat_hidden_dim": trial.suggest_categorical("gat_hidden_dim", [16, 32, 64]),
            "gat_heads": trial.suggest_categorical("gat_heads", [2, 4, 8]),
            "gat_dropout": trial.suggest_float("gat_dropout", 0.05, 0.3),
            "affinity_threshold": trial.suggest_float("affinity_threshold", 0.1, 0.4),
            "alpha_scale": trial.suggest_float("alpha_scale", 1.0, 3.0),
            "focal_alpha": trial.suggest_float("focal_alpha", 0.50, 0.85),
            "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 3.0),
            "lr_stage1": trial.suggest_float("lr_stage1", 1e-4, 3e-3, log=True),
            "lr_stage2": trial.suggest_float("lr_stage2", 1e-4, 3e-3, log=True),
            "lambda_sup": trial.suggest_float("lambda_sup", 0.1, 1.0)
        }

        model = HISDiarizer(
            feature_dim=31,
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            tcn_dropout=params["tcn_dropout"],
            segment_proj_dim=params["segment_proj_dim"],
            gat_hidden_dim=params["gat_hidden_dim"],
            graph_out_dim=16,
            num_classes=2,
            gat_heads=params["gat_heads"],
            gat_dropout=params["gat_dropout"],
            affinity_threshold=params["affinity_threshold"],
            alpha_scale=params["alpha_scale"]
        ).to(device)

        focal_loss = BoundaryFocalLoss(alpha=params["focal_alpha"], gamma=params["focal_gamma"])
        ce_loss = nn.CrossEntropyLoss()
        supcon_loss = SupervisedContrastiveLoss(temperature=0.07)

        opt_s1 = torch.optim.AdamW(model.boundary_detector.parameters(), lr=params["lr_stage1"])
        calibrated_tau = train_stage1_boundary_detector(
            model, train_loader, val_loader, opt_s1, focal_loss, device, 
            max_epochs=epochs, patience=patience, verbose=False  # Use passed args
        )
        opt_s2 = torch.optim.AdamW(list(model.profiler.parameters()) + list(model.graph_diarizer.parameters()), lr=params["lr_stage2"])
        train_stage2_graph_diarizer(
            model, train_loader, val_loader, opt_s2, ce_loss, supcon_loss, device, 
            max_epochs=epochs, patience=patience, threshold=calibrated_tau, 
            lambda_sup=params["lambda_sup"], verbose=False  # Use passed args
        )

        val_metrics = compute_validation_score(model, val_loader, device, threshold=calibrated_tau)
        score = val_metrics["combined_score"]

        del model, opt_s1, opt_s2, focal_loss, ce_loss, supcon_loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"\nTuning Complete! Best Validation Score: {study.best_value:.4f}")
    best_params = study.best_params

    save_path = ckpt_dir / BEST_PARAMS_FILE
    with open(save_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Saved optimal hyperparameters to: {save_path.resolve()}\n")

    return best_params


def load_best_hyperparameters(ckpt_dir: Path) -> Dict[str, Any]:
    save_path = ckpt_dir / BEST_PARAMS_FILE
    if save_path.exists():
        with open(save_path, "r") as f:
            params = json.load(f)
        print(f"Loaded optimal hyperparameters from {save_path.resolve()}")
        return params
    return {}


# =====================================================================
# ENSEMBLE TEST EVALUATION (3 MODELS - ALIGNED PROFILER EVALUATION)
# =====================================================================

def evaluate_ensemble_on_test_set(
    models: List[HISDiarizer], 
    test_loader: DataLoader, 
    device: torch.device, 
    thresholds: List[float]
):
    for m in models:
        m.eval()

    all_pred_token_labels = []
    all_true_token_labels = []
    total_tp, total_fp, total_fn = 0, 0, 0
    avg_threshold = float(np.mean(thresholds))

    print("\n==================================================")
    print(f"EVALUATING {len(models)}-MODEL ENSEMBLE ON HELD-OUT TEST SET (Unified tau* = {avg_threshold:.2f})")
    print("==================================================")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Ensemble Testing"):
            features, mask = batch["fused_sequence"].to(device), batch["mask"].to(device)
            target_boundaries = batch["boundary_targets"].to(device)
            token_author_ids = batch["token_author_ids"].to(device)

            batch_size = features.size(0)
            ensemble_boundary_probs = torch.zeros_like(target_boundaries)
            model_hidden_feats_list = []

            # 1. Ensemble Stage 1: Average boundary probabilities
            for model in models:
                b_probs, hidden_f = model.forward_stage1(features, mask=mask)
                ensemble_boundary_probs += b_probs / len(models)
                model_hidden_feats_list.append(hidden_f)

            tp, fp, fn = compute_boundary_counts_with_tolerance(
                ensemble_boundary_probs, target_boundaries, mask, threshold=avg_threshold, tolerance=3
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn

            # 2. Ensemble Stage 2: Profile segments per model to prevent feature space mismatch
            for doc_idx in range(batch_size):
                valid_len = int(mask[doc_idx].sum().item())
                if valid_len == 0:
                    continue

                true_tokens = token_author_ids[doc_idx, :valid_len].cpu().numpy()
                pred_tokens = np.zeros(valid_len, dtype=int)

                model_logits_list = []
                spans_ref = None

                for m_idx, model in enumerate(models):
                    seg_data_list = model.profiler(
                        batch_seq_features=features[doc_idx:doc_idx+1],
                        batch_hidden_feats=model_hidden_feats_list[m_idx][doc_idx:doc_idx+1],
                        batch_boundary_probs=ensemble_boundary_probs[doc_idx:doc_idx+1],
                        mask=mask[doc_idx:doc_idx+1],
                        threshold=avg_threshold
                    )
                    spans_ref = seg_data_list[0]["spans"]
                    if len(spans_ref) > 0:
                        doc_res = model.graph_diarizer(seg_data_list)[0]
                        model_logits_list.append(doc_res["logits"])

                if len(model_logits_list) > 0 and spans_ref is not None and len(spans_ref) > 0:
                    ensemble_doc_logits = torch.stack(model_logits_list).mean(dim=0)
                    pred_classes = torch.argmax(ensemble_doc_logits, dim=-1).cpu().numpy()

                    for seg_idx, (st, end) in enumerate(spans_ref):
                        if seg_idx < len(pred_classes):
                            st_c, end_c = min(st, valid_len), min(end, valid_len)
                            if st_c < end_c:
                                pred_tokens[st_c:end_c] = pred_classes[seg_idx]

                all_pred_token_labels.extend(pred_tokens)
                all_true_token_labels.extend(true_tokens)

    precision = total_tp / max(total_tp + total_fp, 1e-9)
    recall = total_tp / max(total_tp + total_fn, 1e-9)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-9)

    print("\n--------------------------------------------------")
    print(f"ENSEMBLE STAGE 1: BOUNDARY DETECTION (Tolerance ±3 tokens)")
    print("--------------------------------------------------")
    print(f"Boundary Precision: {precision:.4f}")
    print(f"Boundary Recall:    {recall:.4f}")
    print(f"Boundary F1-Score:  {f1:.4f}")

    print("\n--------------------------------------------------")
    print("ENSEMBLE STAGE 2: TOKEN-LEVEL AUTHOR DIARIZATION PERFORMANCE")
    print("--------------------------------------------------")
    class_names = ["Human (0)", "LLM-Generated (1)"]
    print(classification_report(all_true_token_labels, all_pred_token_labels, target_names=class_names, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_true_token_labels, all_pred_token_labels))


# =====================================================================
# MAIN EXECUTION PIPELINE (3 FOLDS)
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate HIS-Diarizer 3-Fold Ensemble")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Path to processed features folder")
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR, help="Path to checkpoints output folder")
    parser.add_argument("--epochs", type=int, default=25, help="Max training epochs per stage (with Early Stopping)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--batch_size", type=int, default=8, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Default learning rate")
    parser.add_argument("--n_folds", type=int, default=3, help="Number of Group K-Fold splits (Set to 3)")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning before training")
    parser.add_argument("--n_trials", type=int, default=15, help="Number of Optuna tuning trials")
    parser.add_argument("--limit_samples", type=int, default=None, help="Limit dataset size for fast dry run")
    parser.add_argument("--force_resplit", action="store_true", help="Force re-creation of Group K-Fold splits")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.data_dir)
    output_ckpt_dir = Path(args.ckpt_dir)
    output_ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path.resolve()}")

    split_fold0_file = data_path / "fold_0_train_metadata.parquet"
    if not split_fold0_file.exists() or args.force_resplit:
        split_dataset_kfold(data_dir=args.data_dir, n_splits=args.n_folds)

    if args.tune:
        run_hyperparameter_tuning(data_dir=args.data_dir, ckpt_dir=output_ckpt_dir, n_trials=args.n_trials, device=device)

    best_hp = load_best_hyperparameters(output_ckpt_dir)

    hidden_dim = best_hp.get("hidden_dim", 64)
    num_layers = best_hp.get("num_layers", 4)
    tcn_dropout = best_hp.get("tcn_dropout", 0.1)
    segment_proj_dim = best_hp.get("segment_proj_dim", 32)
    gat_hidden_dim = best_hp.get("gat_hidden_dim", 32)
    gat_heads = best_hp.get("gat_heads", 4)
    gat_dropout = best_hp.get("gat_dropout", 0.1)
    affinity_threshold = best_hp.get("affinity_threshold", 0.2)
    alpha_scale = best_hp.get("alpha_scale", 2.0)
    focal_alpha = best_hp.get("focal_alpha", 0.75)
    focal_gamma = best_hp.get("focal_gamma", 2.0)
    lr_s1 = best_hp.get("lr_stage1", args.lr)
    lr_s2 = best_hp.get("lr_stage2", args.lr)
    lambda_sup = best_hp.get("lambda_sup", 0.5)

    trained_ensemble_models = []
    calibrated_thresholds = []

    for fold in range(args.n_folds):
        print(f"\n==================================================")
        print(f"TRAINING MODEL FOR FOLD {fold + 1} / {args.n_folds}")
        print("==================================================")

        train_dataset = HISDiarizerDataset(data_dir=args.data_dir, split=f"fold_{fold}_train")
        val_dataset = HISDiarizerDataset(data_dir=args.data_dir, split=f"fold_{fold}_val")

        if args.limit_samples is not None and args.limit_samples < len(train_dataset):
            train_dataset.samples = train_dataset.samples[:args.limit_samples]

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn)

        class_counts = train_dataset.get_class_counts()
        class_weights = 1.0 / torch.sqrt(torch.clamp(class_counts, min=1.0))
        class_weights = (class_weights / class_weights.sum()).to(device)

        fold_model = HISDiarizer(
            feature_dim=31,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            tcn_dropout=tcn_dropout,
            segment_proj_dim=segment_proj_dim,
            gat_hidden_dim=gat_hidden_dim,
            graph_out_dim=16,
            num_classes=2,
            gat_heads=gat_heads,
            gat_dropout=gat_dropout,
            affinity_threshold=affinity_threshold,
            alpha_scale=alpha_scale
        ).to(device)

        focal_loss = BoundaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        supcon_loss = SupervisedContrastiveLoss(temperature=0.07)

        opt_stage1 = torch.optim.AdamW(fold_model.boundary_detector.parameters(), lr=lr_s1, weight_decay=1e-4)
        fold_tau = train_stage1_boundary_detector(
            fold_model, train_loader, val_loader, opt_stage1, focal_loss, device, 
            max_epochs=args.epochs, patience=args.patience
        )
        calibrated_thresholds.append(fold_tau)

        stage2_params = list(fold_model.profiler.parameters()) + list(fold_model.graph_diarizer.parameters())
        opt_stage2 = torch.optim.AdamW(stage2_params, lr=lr_s2, weight_decay=1e-4)
        train_stage2_graph_diarizer(
            fold_model, train_loader, val_loader, opt_stage2, ce_loss, supcon_loss, device, 
            max_epochs=args.epochs, patience=args.patience, threshold=fold_tau, lambda_sup=lambda_sup
        )

        torch.save(fold_model.state_dict(), output_ckpt_dir / f"his_diarizer_fold_{fold}.pt")
        trained_ensemble_models.append(fold_model)

    torch.save(trained_ensemble_models[0].state_dict(), output_ckpt_dir / "his_diarizer_full_model.pt")

    test_dataset = HISDiarizerDataset(data_dir=args.data_dir, split="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn)

    evaluate_ensemble_on_test_set(
        models=trained_ensemble_models,
        test_loader=test_loader,
        device=device,
        thresholds=calibrated_thresholds
    )


if __name__ == "__main__":
    main()