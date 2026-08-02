# tune_optuna.py

# =========================================================================
# 0. STRICT CPU THREAD LIMITS (MUST BE SET BEFORE IMPORTING TORCH/NUMPY)
# =========================================================================
import os

# Limit OpenMP, MKL, OpenBLAS, and NumExpr C++ background thread pools
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# Disable Hugging Face Rust tokenizer background thread pools (prevents fork deadlocks)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import optuna

# Set PyTorch internal intra-op and inter-op CPU threads
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

from config import parse_args_into_config
from data.synthetic_generator import MixedAuthorshipDataEngine
from data.dataset import MixedAuthorshipDataset, pad_collate_fn
from features.stylometrics import StylometricFeatureEngine
from features.dense_encoder import DenseTransformerEncoder
from models.neural_crf_tagger import MultiTaskNeuralCRFTagger
from train_pipeline import ScaledFoldDataset 


def objective(trial: optuna.Trial, cfg, dataset) -> float:
    device = torch.device(cfg.model.device)

    # =========================================================================
    # 1. OPTUNA HYPERPARAMETER SUGGESTIONS
    # =========================================================================
    lr = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    aux_boundary_weight = trial.suggest_float("aux_boundary_weight", 0.1, 2.0)
    dropout = trial.suggest_float("dropout", 0.2, 0.65)
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768])
    num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 2)

    boundary_pos_weight = trial.suggest_float("boundary_pos_weight", 1.0, 5.0, step=0.5)
    rnn_type = trial.suggest_categorical("rnn_type", ["LSTM", "GRU"])
    scheduler_type = trial.suggest_categorical("scheduler_type", ["cosine", "reduce_on_plateau", "none"])
    feature_input_dropout = trial.suggest_float("feature_input_dropout", 0.0, 0.3)
    crf_lr_mult = trial.suggest_float("crf_lr_mult", 2.0, 10.0, step=1.0)
    emission_temp = trial.suggest_float('emission_temp', 0.5, 1.2, step=0.1)
    use_attention = trial.suggest_categorical('use_attention', [True, False])


    print(f"\n  --> [Trial {trial.number}] Params: {trial.params}", flush=True)
    # =========================================================================
    # 2. FAST SINGLE FOLD SPLIT & OPTIMIZED DATALOADERS
    # =========================================================================
    parent_ids = np.array([doc.parent_doc_id for doc in dataset.docs])
    scenarios = np.array([doc.scenario for doc in dataset.docs])

    kfold_splitter = StratifiedGroupKFold(n_splits=cfg.training.n_splits)
    train_idx, val_idx = next(kfold_splitter.split(dataset, scenarios, groups=parent_ids))

    # Scale Stylometrics
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
        scaled_features_list.append(np.hstack([dense_part, scaled_stylo]).astype(np.float32))

    train_sub = ScaledFoldDataset(dataset, scaled_features_list, train_idx)
    val_sub = ScaledFoldDataset(dataset, scaled_features_list, val_idx)

    # pin_memory=True speeds up CPU-to-GPU data transfers
    train_loader = DataLoader(
        train_sub, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        collate_fn=pad_collate_fn,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_sub, 
        batch_size=cfg.training.batch_size, 
        shuffle=False, 
        collate_fn=pad_collate_fn,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # =========================================================================
    # 3. BUILD MODEL & OPTIMIZER
    # =========================================================================
    model = MultiTaskNeuralCRFTagger(
        dense_dim=768,
        stylo_dim=84,
        hidden_dim=hidden_dim,
        num_lstm_layers=num_lstm_layers,
        dropout=dropout,
        feature_input_dropout=feature_input_dropout, 
        rnn_type=rnn_type,                          
        aux_boundary_weight=aux_boundary_weight,
        boundary_pos_weight=boundary_pos_weight,   
        emission_temp=emission_temp,
        use_attention=use_attention
    ).to(device)

    crf_params = [p for n, p in model.named_parameters() if "crf" in n]
    other_params = [p for n, p in model.named_parameters() if "crf" not in n]
    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
        {"params": crf_params, "lr": lr * crf_lr_mult, "weight_decay": 0.0}
    ])

    scheduler = None
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.optuna.search_epochs)
    elif scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_val_auc = 0.0
    max_search_epochs = cfg.optuna.search_epochs
    patience = 3
    patience_counter = 0

    # =========================================================================
    # 4. TRAIN AND EVALUATE
    # =========================================================================
    for epoch in range(max_search_epochs):
        # --- 4a. TRAINING PHASE ---
        model.train()
        for batch in train_loader:
            fused_features = batch["fused_features"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            boundaries = batch["boundaries"].to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model(fused_features, mask, labels=labels, boundaries=boundaries)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # --- 4b. VALIDATION PHASE ---
        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                fused_features = batch["fused_features"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)

                preds = model.predict(fused_features, mask)

                mask_np = mask.cpu().numpy()
                probs_np = preds["probabilities"].cpu().numpy()
                labels_np = batch["labels"].numpy()

                for b_i in range(len(batch["doc_ids"])):
                    seq_len = mask_np[b_i].sum()
                    val_probs.extend(probs_np[b_i, :seq_len])
                    val_targets.extend(labels_np[b_i, :seq_len])

        val_auc = roc_auc_score(val_targets, val_probs)

        # [LIVE PROGRESS MONITORING]
        print(f"  --> [Trial {trial.number}] Epoch {epoch+1}/{max_search_epochs} | Val AUC: {val_auc:.4f}", flush=True)

        if scheduler is not None:
            if scheduler_type == "cosine":
                scheduler.step()
            elif scheduler_type == "reduce_on_plateau":
                scheduler.step(val_auc)

        # PRUNING CHECK
        trial.report(val_auc, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # EARLY STOPPING CHECK
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    --> Early stopping trial at epoch {epoch + 1} (Best AUC: {best_val_auc:.4f})", flush=True)
                break

    return best_val_auc


def run_tuning():
    cfg = parse_args_into_config()

    print("=========================================================")
    print("      OPTUNA HYPERPARAMETER TUNING SEARCH                ")
    print("=========================================================")
    print(f"Loading data from: {cfg.data.data_path}")

    raw_df = pd.read_parquet(cfg.data.data_path) if cfg.data.data_path.endswith('.parquet') else pd.read_csv(cfg.data.data_path)
    
    if cfg.data.sample_size and len(raw_df) > cfg.data.sample_size:
        raw_df = raw_df.sample(n=cfg.data.sample_size, random_state=cfg.data.random_seed).reset_index(drop=True)

    engine = MixedAuthorshipDataEngine(random_state=cfg.data.random_seed, min_sentences=cfg.data.min_sentences)
    synthetic_docs = engine.process_dataframe(raw_df)

    style_engine = StylometricFeatureEngine(include_w3=cfg.model.include_w3, include_w5=cfg.model.include_w5)
    dense_encoder = DenseTransformerEncoder(model_name=cfg.model.transformer_name, device=cfg.model.device)

    dataset = MixedAuthorshipDataset(
        synthetic_docs=synthetic_docs,
        style_engine=style_engine,
        dense_encoder=dense_encoder,
        cache_dir=cfg.data.cache_dir,
        cache_file=cfg.data.cache_file
    )

    db_file = f"sqlite:///{cfg.optuna.study_name}.db"
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(
        study_name=cfg.optuna.study_name,
        direction="maximize",
        pruner=pruner,
        storage=db_file,
        load_if_exists=True
    )

    # Seed database if it's new
    if os.path.exists("best_params.json") and len(study.trials) == 0:
        print("-> Pre-seeding new Optuna database with your previous best parameters from 'best_params.json'...")
        with open("best_params.json", "r") as f:
            best_params = json.load(f)
        study.enqueue_trial(best_params)

    print(f"\nConnected to database: '{cfg.optuna.study_name}.db'")
    print(f"Existing trials in study: {len(study.trials)}")
    print(f"Running {cfg.optuna.n_trials} additional trials...")

    print(f"\nStarting {cfg.optuna.n_trials} tuning trials...")
    study.optimize(lambda trial: objective(trial, cfg, dataset), n_trials=cfg.optuna.n_trials)

    print("\n=========================================================")
    print("           OPTIMIZATION COMPLETE                        ")
    print("=========================================================")
    print(f"Best Validation ROC-AUC: {study.best_value:.4f}")
    print("Best Parameters Found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    output_path = "best_params.json"
    with open(output_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\nSaved best hyperparameters to '{output_path}'")


if __name__ == "__main__":
    run_tuning()