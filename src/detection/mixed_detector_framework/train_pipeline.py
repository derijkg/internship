# train_pipeline.py

#TODO limit cpu usage like train_optuna

import os
import copy
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List

from config import parse_args_into_config, MasterConfig
from data.synthetic_generator import MixedAuthorshipDataEngine
from data.dataset import MixedAuthorshipDataset, pad_collate_fn
from features.stylometrics import StylometricFeatureEngine
from features.dense_encoder import DenseTransformerEncoder
from models.neural_crf_tagger import MultiTaskNeuralCRFTagger
from training.calibration import NeymanPearsonCalibrator
from evaluate import evaluate_mixed_authorship_performance


class ScaledFoldDataset(Dataset):
    """Wraps MixedAuthorshipDataset for a specific fold, supplying feature
    matrices that have been standardized using a fold-specific StandardScaler.
    """

    def __init__(
        self,
        base_dataset: MixedAuthorshipDataset,
        scaled_features_list: List[np.ndarray],
        indices: np.ndarray,
    ):
        self.base_dataset = base_dataset
        self.scaled_features_list = scaled_features_list
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self.indices[idx]
        doc = self.base_dataset.docs[real_idx]
        fused_features = self.scaled_features_list[real_idx]
        feature_len = len(fused_features)

        return {
            "doc_id": doc.doc_id,
            "parent_doc_id": doc.parent_doc_id,
            "scenario": doc.scenario,
            "fused_features": torch.tensor(
                fused_features, dtype=torch.float32
            ),
            # Derived from fused_features row count to guarantee 100% tensor alignment
            "labels": torch.tensor(doc.labels[:feature_len], dtype=torch.long),
            "boundaries": torch.tensor(doc.boundaries[:feature_len], dtype=torch.long),
            "seq_len": feature_len,
        }


def run_training_pipeline(cfg: MasterConfig):
    print("=========================================================")
    print("      STAGE 1: DATA ENGINE & SYNTHETIC GENERATION        ")
    print("=========================================================")

    # OPTUNA PARAMETER LOADING
    if os.path.exists("best_params.json"):
        print("\n[INFO] Found 'best_params.json'. Applying Optuna tuned hyperparameters!")
        with open("best_params.json", "r") as f:
            best_params = json.load(f)
        
        # Override config attributes with Optuna params
        cfg.training.learning_rate = best_params.get("learning_rate", cfg.training.learning_rate)
        cfg.training.weight_decay = best_params.get("weight_decay", cfg.training.weight_decay)
        cfg.model.aux_boundary_weight = best_params.get("aux_boundary_weight", cfg.model.aux_boundary_weight)
        cfg.model.dropout = best_params.get("dropout", cfg.model.dropout)
        cfg.model.hidden_dim = best_params.get("hidden_dim", cfg.model.hidden_dim)
        cfg.model.num_lstm_layers = best_params.get("num_lstm_layers", cfg.model.num_lstm_layers)
        
        print(f"  -> Learning Rate:        {cfg.training.learning_rate}")
        print(f"  -> Weight Decay:          {cfg.training.weight_decay}")
        print(f"  -> Aux Boundary Weight:  {cfg.model.aux_boundary_weight}")
        print(f"  -> Dropout:              {cfg.model.dropout}")
        print(f"  -> Hidden Dim:           {cfg.model.hidden_dim}")
        print(f"  -> Num LSTM Layers:      {cfg.model.num_lstm_layers}\n")
    else:
        print("\n[INFO] No 'best_params.json' found. Using default config parameters.")

    print(f"Loading raw dataset from: {cfg.data.data_path}")

    if cfg.data.data_path.endswith('.csv'):
        raw_df = pd.read_csv(cfg.data.data_path)
    else:
        raw_df = pd.read_parquet(cfg.data.data_path)

    # Subsample raw records if requested
    if cfg.data.sample_size and len(raw_df) > cfg.data.sample_size:
        print(f"Subsampling raw dataset down to {cfg.data.sample_size} records...")
        raw_df = raw_df.sample(n=cfg.data.sample_size, random_state=cfg.data.random_seed).reset_index(drop=True)

    engine = MixedAuthorshipDataEngine(
        random_state=cfg.data.random_seed,
        min_sentences=cfg.data.min_sentences
    )
    synthetic_docs = engine.process_dataframe(raw_df)

    print("\n=========================================================")
    print("      STAGE 2: PRECOMPUTING & CACHING FUSED FEATURES     ")
    print("=========================================================")
    style_engine = StylometricFeatureEngine(
        include_w3=cfg.model.include_w3,
        include_w5=cfg.model.include_w5
    )
    dense_encoder = DenseTransformerEncoder(
        model_name=cfg.model.transformer_name,
        device=cfg.model.device
    )

    dataset = MixedAuthorshipDataset(
        synthetic_docs=synthetic_docs,
        style_engine=style_engine,
        dense_encoder=dense_encoder,
        cache_dir=cfg.data.cache_dir,
        cache_file=cfg.data.cache_file
    )

    print("\n=========================================================")
    print("      STAGE 3: STRATIFIED GROUP K-FOLD TRAINING         ")
    print("=========================================================")
    # Derived from dataset.docs to guarantee 100% size alignment
    parent_ids = np.array([doc.parent_doc_id for doc in dataset.docs])
    scenarios = np.array([doc.scenario for doc in dataset.docs])

    # Safeguard Group K-Fold splitting
    scenario_counts = pd.Series(scenarios).value_counts()
    if (scenario_counts < cfg.training.n_splits).any():
        print("Notice: Using standard GroupKFold grouped by parent_doc_id.")
        kfold_splitter = GroupKFold(n_splits=cfg.training.n_splits)
        split_generator = kfold_splitter.split(dataset, groups=parent_ids)
    else:
        kfold_splitter = StratifiedGroupKFold(n_splits=cfg.training.n_splits)
        split_generator = kfold_splitter.split(dataset, scenarios, groups=parent_ids)

    oof_y_true = []
    oof_y_probs = []
    oof_b_true = []
    oof_b_probs = []

    device = torch.device(cfg.model.device if hasattr(cfg.model, 'device') else 'cpu')
    print(f"Target Execution Device: {device}")

    for fold, (train_idx, val_idx) in enumerate(split_generator):
        print(f"\n--- Training Fold {fold + 1}/{cfg.training.n_splits} ({len(train_idx)} Train Docs, {len(val_idx)} Val Docs) ---")

        # =========================================================
        # FOLD-SAFE SCALING: STYLOMETRICS ONLY (COLUMNS 768..851)
        # =========================================================
        train_features = [dataset.features_list[i] for i in train_idx]
        flat_train = np.vstack(train_features)
        train_stylo_cols = flat_train[:, 768:]

        style_scaler = StandardScaler()
        style_scaler.fit(train_stylo_cols)

        scaled_features_list = []
        for doc_matrix in dataset.features_list:
            dense_part = doc_matrix[:, :768]
            stylo_part = doc_matrix[:, 768:]

            scaled_stylo = style_scaler.transform(stylo_part).astype(np.float32)
            scaled_doc = np.hstack([dense_part, scaled_stylo]).astype(np.float32)
            scaled_features_list.append(scaled_doc)

        train_sub = ScaledFoldDataset(dataset, scaled_features_list, train_idx)
        val_sub = ScaledFoldDataset(dataset, scaled_features_list, val_idx)

        train_loader = DataLoader(
            train_sub, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=pad_collate_fn
        )
        val_loader = DataLoader(
            val_sub, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=pad_collate_fn
        )

        # =========================================================
        # DUAL-BRANCH NEURAL CRF MODEL INITIALIZATION
        # =========================================================
        model = MultiTaskNeuralCRFTagger(
            dense_dim=768,
            stylo_dim=84,
            hidden_dim=cfg.model.hidden_dim,
            num_lstm_layers=cfg.model.num_lstm_layers,
            dropout=cfg.model.dropout,
            aux_boundary_weight=cfg.model.aux_boundary_weight
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
        )

        # Early Stopping Variables
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        best_model_weights = None

        for epoch in range(cfg.training.epochs):
            # -----------------------------------------------------
            # 1. TRAINING PHASE
            # -----------------------------------------------------
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                fused_features = batch["fused_features"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["labels"].to(device)
                boundaries = batch["boundaries"].to(device)

                optimizer.zero_grad()
                out = model(
                    fused_features,
                    mask,
                    labels=labels,
                    boundaries=boundaries
                )
                loss = out["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # -----------------------------------------------------
            # 2. VALIDATION PHASE (Computes Val Loss per Epoch)
            # -----------------------------------------------------
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    fused_features = batch["fused_features"].to(device)
                    mask = batch["mask"].to(device)
                    labels = batch["labels"].to(device)
                    boundaries = batch["boundaries"].to(device)

                    out = model(
                        fused_features,
                        mask,
                        labels=labels,
                        boundaries=boundaries
                    )
                    val_loss += out["loss"].item()

            avg_val_loss = val_loss / len(val_loader)

            print(f"    Epoch {epoch + 1:02d}/{cfg.training.epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # -----------------------------------------------------
            # 3. EARLY STOPPING & BEST MODEL CHECKPOINTING
            # -----------------------------------------------------
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    --> Early stopping triggered at epoch {epoch + 1}! (Best Val Loss: {best_val_loss:.4f})")
                    break

        # ---------------------------------------------------------
        # 4. RESTORE BEST WEIGHTS (Runs AFTER training completes)
        # ---------------------------------------------------------
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)

        # ---------------------------------------------------------
        # 5. OUT-OF-FOLD EVALUATION PREDICTIONS
        # ---------------------------------------------------------
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                fused_features = batch["fused_features"].to(device)
                mask = batch["mask"].to(device)

                preds = model.predict(fused_features, mask)
                mask_np = mask.cpu().numpy()

                probs_np = preds["probabilities"].cpu().numpy()
                b_probs_np = preds["boundary_probabilities"].cpu().numpy()
                y_true_np = batch["labels"].numpy()
                b_true_np = batch["boundaries"].numpy()

                for b_i in range(len(batch["doc_ids"])):
                    seq_len = mask_np[b_i].sum()
                    oof_y_true.append(y_true_np[b_i, :seq_len])
                    oof_y_probs.append(probs_np[b_i, :seq_len])
                    oof_b_true.append(b_true_np[b_i, :seq_len])
                    oof_b_probs.append(b_probs_np[b_i, :seq_len])

    print("\n=========================================================")
    print("      STAGE 4: NEYMAN-PEARSON CALIBRATION & EVALUATION    ")
    print("=========================================================")
    flat_oof_y = np.concatenate(oof_y_true)
    flat_oof_p = np.concatenate(oof_y_probs)

    calibrator = NeymanPearsonCalibrator(target_fpr=cfg.training.target_fpr)
    optimal_tau = calibrator.fit(flat_oof_y, flat_oof_p)

    metrics = evaluate_mixed_authorship_performance(
        oof_y_true, oof_y_probs, oof_b_true, oof_b_probs, threshold=optimal_tau
    )

    print("\n=========================================================")
    print("            OUT-OF-FOLD PERFORMANCE METRICS              ")
    print("=========================================================")
    print(f"Calibrated Threshold (τ*): {optimal_tau:.6f} (Target FPR <= {cfg.training.target_fpr*100:.1f}%)")
    print(f"Sentence AI Precision:     {metrics['sent_precision_ai']:.4f}")
    print(f"Sentence AI Recall:        {metrics['sent_recall_ai']:.4f}")
    print(f"Sentence AI F1 Score:      {metrics['sent_f1_ai']:.4f}")
    print(f"Sentence ROC-AUC:          {metrics['sent_roc_auc']:.4f}")
    print(f"Boundary Precision:        {metrics['boundary_precision']:.4f}")
    print(f"Boundary Recall:           {metrics['boundary_recall']:.4f}")
    print(f"Boundary F1 Score:         {metrics['boundary_f1']:.4f}")
    print(f"Segment Span IoU:          {metrics['span_iou']:.4f}")
    print(f"AI Ratio MAE:              {metrics['ai_ratio_mae']:.4f}")


if __name__ == "__main__":
    config = parse_args_into_config()
    run_training_pipeline(config)