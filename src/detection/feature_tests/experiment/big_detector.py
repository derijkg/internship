#!/usr/bin/env python3
"""
Modular XGBoost Hyperparameter Tuning & Evaluation Pipeline
Features:
  - Exclusively uses aggregated features CSV files (Trajectory token files are ignored)
  - Separate model training per LLM generator via `--llm_model` / `--model` flag
  - Strict 80/20 Grouped Holdout Split (prevents data leakage across documents)
  - Optuna Tuning strictly inside 80% Dev Set (10-Fold Stratified Group CV)
  - Unseen 20% Holdout Test Set evaluation & threshold calibration
  - Robust SHAP compatibility wrapper
  - Dynamic CPU thread capping via threadpoolctl
  - Subfolder-based artifact & SQLite Optuna DB management per model & dataset
  - Automated column auditing to prevent target (is_llm) and metadata leakage
"""

import os
import sys
import glob
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from threadpoolctl import threadpool_limits

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
import optuna
import shap

# Suppress Optuna & SHAP verbosity warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")


# =====================================================================
# 0. DYNAMIC CPU THREAD MANAGEMENT (Leaves 5 CPUs free)
# =====================================================================
try:
    TOTAL_CPUS = len(os.sched_getaffinity(0))
except AttributeError:
    TOTAL_CPUS = os.cpu_count() or 4

TARGET_CPUS = max(1, TOTAL_CPUS - 7)
threadpool_limits(limits=TARGET_CPUS)


# =====================================================================
# 1. COLUMN AUDITOR & SUBFOLDER HELPERS
# =====================================================================
def extract_feature_columns(df):
    """
    Audits all columns in the DataFrame and returns ONLY valid numeric feature columns.
    Guarantees that target labels (is_llm), text, identifiers, and metadata do not leak into X.
    """
    ignore_cols = {
        # Targets & Labels
        "is_llm", "label", "target", "y",
        # Metadata & Identifiers
        "sentence_id", "_id", "doc_id", "id", "page_link", "synthetic_id",
        # Text & Context
        "text", "abstract", "abstract_sentence", "context", "full_context",
        # Model & Source Information
        "generator_model", "model", "model_name", "source", "dataset", "llm_col", "eval_unit",
        # Pandas / CSV Index Artifacts
        "Unnamed: 0", "Unnamed: 0.1", "index"
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in ignore_cols]

    assert "is_llm" not in feature_cols, "CRITICAL LEAKAGE: 'is_llm' target column found in feature_cols!"
    assert "label" not in feature_cols, "CRITICAL LEAKAGE: 'label' column found in feature_cols!"
    assert "generator_model" not in feature_cols, "CRITICAL LEAKAGE: 'generator_model' column found in feature_cols!"

    return sorted(feature_cols)


def get_input_subfolder_name(file_paths):
    """
    Derives an output subfolder name based on the input file or parent folder name.
    """
    if not file_paths:
        return "default_run"

    generic_names = {"aggregated_features", "data", "features", "sampled"}

    if len(file_paths) == 1:
        path = file_paths[0]
        parent_dir = os.path.basename(os.path.dirname(path))
        filename = os.path.splitext(os.path.basename(path))[0]

        if filename in generic_names and parent_dir and parent_dir not in generic_names:
            return parent_dir
        return filename
    else:
        stems = []
        for p in file_paths:
            p_parent = os.path.basename(os.path.dirname(p))
            p_file = os.path.splitext(os.path.basename(p))[0]
            name = p_parent if p_file in generic_names and p_parent not in generic_names else p_file
            stems.append(name)
        
        unique_stems = sorted(list(set(stems)))
        if len(unique_stems) <= 3:
            return "_".join(unique_stems)
        else:
            return f"merged_{len(file_paths)}_inputs"


# =====================================================================
# 2. DATA RESOLUTION, LOADING & HOLDOUT SPLIT
# =====================================================================
def create_holdout_split(sent_df, test_size=0.20):
    """
    Splits dataset into 80% Dev Set and 20% Holdout Test Set grouped by document ID (_id).
    Guarantees no sentence from the same document appears in both Dev and Test sets.
    """
    doc_col = "_id" if "_id" in sent_df.columns else "sentence_id"
    groups = sent_df[doc_col].values
    y = sent_df["is_llm"].values
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    dev_idx, test_idx = next(sgkf.split(sent_df, y, groups=groups))
    
    dev_df = sent_df.iloc[dev_idx].copy().reset_index(drop=True)
    test_df = sent_df.iloc[test_idx].copy().reset_index(drop=True)
    
    print(f"[HOLDOUT SPLIT] Dev Set (80%): {len(dev_df)} rows | Holdout Test Set (20%): {len(test_df)} rows")
    return dev_df, test_df


def resolve_input_files(inputs):
    """
    Resolves input file paths/directories strictly for aggregated_features.csv.
    Automatically EXCLUDES full_abstract and trajectory_tokens files.
    """
    csv_files = []
    for item in inputs:
        if os.path.isdir(item):
            found_feat = glob.glob(os.path.join(item, "**", "aggregated_features.csv"), recursive=True)
            csv_files.extend(found_feat)
        elif "*" in item:
            csv_files.extend(glob.glob(item, recursive=True))
        elif os.path.isfile(item):
            csv_files.append(item)

    # Strictly include only aggregated feature files and ignore full_abstract / trajectory_tokens
    valid_files = [
        f for f in set(csv_files) 
        if "full_abstract" not in f and "trajectory_tokens" not in f and "trajectory" not in f
    ]
    return sorted(valid_files)


def load_and_merge_feature_csvs(file_paths, target_model=None):
    """
    Loads and merges sentence-level aggregated feature CSV files.
    Optionally filters LLM data to a specific model target (e.g. GPT4_LLM).
    """
    human_df = None
    llm_dfs = []
    all_found_llm_labels = set()

    print(f"\n[DATA LOADER] Processing {len(file_paths)} aggregated feature file(s):")
    for path in file_paths:
        print(f"  -> {path}")

    for path in file_paths:
        df = pd.read_csv(path)
        
        # Explicit check to reject trajectory token files if passed directly
        if "token_pos" in df.columns:
            print(f"  [WARNING] Skipping '{path}': Token trajectory files are not supported. Please pass aggregated_features.csv instead.")
            continue

        if "is_llm" not in df.columns and "label" in df.columns:
            df["is_llm"] = (df["label"] != "Human").astype(int)

        if human_df is None:
            human_df = df[df["is_llm"] == 0].copy()

        curr_llm = df[df["is_llm"] == 1].copy()

        if "label" in curr_llm.columns:
            all_found_llm_labels.update(curr_llm["label"].dropna().unique().tolist())

        # Filter to single model if requested
        if target_model:
            target_clean = target_model.replace("_LLM", "").strip().lower()
            # Flexible match: works with 'GPT4', 'GPT4_LLM', or case-insensitive matching
            mask = (
                curr_llm["label"].astype(str).str.strip().str.lower() == target_model.strip().lower()
            ) | (
                curr_llm["label"].astype(str).str.replace("_LLM", "", regex=False).str.strip().str.lower() == target_clean
            )
            curr_llm = curr_llm[mask].copy()

        if not curr_llm.empty:
            llm_dfs.append(curr_llm)

    if human_df is None:
        raise ValueError("Failed to load valid Human feature data!")

    if not llm_dfs:
        err_msg = f"No LLM samples found matching target model '{target_model}'!" if target_model else "Failed to load valid LLM feature data!"
        if all_found_llm_labels:
            err_msg += f"\nAvailable LLM labels in input dataset(s): {sorted(list(all_found_llm_labels))}"
        raise ValueError(err_msg)

    combined_llm_df = pd.concat(llm_dfs, ignore_index=True)
    merged_df = pd.concat([human_df, combined_llm_df], ignore_index=True)

    print("\n" + "=" * 80)
    mode_str = f"SINGLE MODEL TARGET: '{target_model}'" if target_model else "ALL LLMs COMBINED"
    print(f"  DATASET SUMMARY [{mode_str}]")
    print("=" * 80)
    print(f"Total Human Samples: {len(human_df)}")
    print(f"Total LLM Samples:   {len(combined_llm_df)}")
    if target_model and "label" in combined_llm_df.columns:
        print(f"Matched LLM Label(s):{sorted(combined_llm_df['label'].unique().tolist())}")
    elif "label" in combined_llm_df.columns:
        print(f"Included Generator Models: {sorted(combined_llm_df['label'].unique().tolist())}")
    print(f"Total Dataset Size:  {len(merged_df)} sentences")
    print("=" * 80 + "\n")

    return merged_df


# =====================================================================
# 3. OPTUNA OBJECTIVE & TUNING MANAGER
# =====================================================================
def xgboost_objective(trial, X, y, groups, base_scale_pos_weight, n_splits=10):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 600, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 2.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", 
            max(0.05, base_scale_pos_weight * 0.5), 
            min(1.0, base_scale_pos_weight * 2.5),
            log=True
        ),
        "random_state": 42,
        "eval_metric": "logloss",
        "n_jobs": TARGET_CPUS
    }

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**params, early_stopping_rounds=20)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds_prob = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds_prob))

    return float(np.mean(aucs))


def print_trial_progress(study, trial):
    current_auc = trial.value
    best_auc = study.best_value
    best_num = study.best_trial.number

    print(
        f"  [Optuna Trial #{trial.number:03d}] "
        f"10-Fold Dev ROC-AUC: {current_auc:.4f} | "
        f"Best Dev ROC-AUC: {best_auc:.4f} (Trial #{best_num:03d})"
    )


def run_or_resume_optuna(X_dev, y_dev, groups_dev, scale_pos_weight, target_out_dir, n_trials, reset_study, study_name="xgboost_aggregated_study"):
    os.makedirs(target_out_dir, exist_ok=True)
    db_path = os.path.abspath(os.path.join(target_out_dir, "xgboost_study.db"))
    json_path = os.path.join(target_out_dir, "best_params.json")
    storage_url = f"sqlite:///{db_path}"

    if reset_study:
        print(f"\n[RESET STUDY] Deleting existing Optuna database in '{target_out_dir}'...")
        if os.path.exists(db_path):
            os.remove(db_path)
        if os.path.exists(json_path):
            os.remove(json_path)

    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=42)
    pruner = optuna.pruners.NopPruner()

    sanitized_study_name = "".join([c if c.isalnum() or c in "_-" else "_" for c in study_name])

    study = optuna.create_study(
        study_name=sanitized_study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

    existing_trials = len(study.trials)
    print(f"[OPTUNA MANAGER] SQLite DB Path: '{db_path}'")
    print(f"[OPTUNA MANAGER] Existing completed trials in DB: {existing_trials}")

    if n_trials > 0:
        print(f"\n[OPTUNA TUNER] Running {n_trials} trial(s) strictly on 80% Dev Set (10-Fold CV)...")
        print("-" * 85)
        study.optimize(
            lambda trial: xgboost_objective(trial, X_dev, y_dev, groups_dev, scale_pos_weight, n_splits=10),
            n_trials=n_trials,
            callbacks=[print_trial_progress]
        )
        print("-" * 85)

    if len(study.trials) == 0 or study.best_trial is None:
        best_params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight
        }
    else:
        best_params = study.best_params
        print(f"\n[BEST OPTUNA TRIAL] Trial #{study.best_trial.number} | Best Dev 10-Fold ROC-AUC: {study.best_value:.4f}")

    best_params["random_state"] = 42
    best_params["eval_metric"] = "logloss"
    best_params["n_jobs"] = TARGET_CPUS

    with open(json_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"[SAVED BEST PARAMS]: {json_path}")

    return best_params


# =====================================================================
# 4. EVALUATION ON DEV CV & UNTOUCHED 20% HOLDOUT TEST SET
# =====================================================================
def find_optimal_threshold(y_true, y_probs):
    thresholds = np.linspace(0.1, 0.9, 81)
    f1s = [f1_score(y_true, (y_probs >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx]), float(f1s[best_idx])


def evaluate_final_xgboost(dev_df, test_df, feature_cols, best_params, target_out_dir, n_splits=10):
    doc_col = "_id" if "_id" in dev_df.columns else "sentence_id"

    X_dev = dev_df[feature_cols].values
    y_dev = dev_df["is_llm"].values
    groups_dev = dev_df[doc_col].values

    X_test = test_df[feature_cols].values
    y_test = test_df["is_llm"].values

    print("\n" + "=" * 85)
    print("  1. DEV SET 10-FOLD CROSS-VALIDATION PERFORMANCE (80% Data)")
    print("=" * 85)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, accs, f1s, precs, recs = [], [], [], [], []
    all_dev_probs, all_dev_y = [], []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dev, y_dev, groups=groups_dev)):
        X_tr, X_va = X_dev[train_idx], X_dev[val_idx]
        y_tr, y_va = y_dev[train_idx], y_dev[val_idx]

        model = xgb.XGBClassifier(**best_params)
        model.fit(X_tr, y_tr)

        preds_prob = model.predict_proba(X_va)[:, 1]
        preds_class = model.predict(X_va)

        all_dev_probs.extend(preds_prob)
        all_dev_y.extend(y_va)

        aucs.append(roc_auc_score(y_va, preds_prob))
        accs.append(accuracy_score(y_va, preds_class))
        f1s.append(f1_score(y_va, preds_class))
        precs.append(precision_score(y_va, preds_class))
        recs.append(recall_score(y_va, preds_class))

    opt_thresh, opt_f1 = find_optimal_threshold(np.array(all_dev_y), np.array(all_dev_probs))

    dev_metrics_df = pd.DataFrame([{
        "Dataset": "Dev 10-Fold CV (Thresh=0.50)",
        "ROC-AUC": f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}",
        "Accuracy": f"{np.mean(accs):.3f} ± {np.std(accs):.3f}",
        "F1-Score": f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}",
        "Precision": f"{np.mean(precs):.3f} ± {np.std(precs):.3f}",
        "Recall": f"{np.mean(recs):.3f} ± {np.std(recs):.3f}"
    }])
    print(dev_metrics_df.to_string(index=False))
    print(f"\n[DEV THRESHOLD CALIBRATION] Optimal Cutoff = {opt_thresh:.2f} (Dev CV Max F1 = {opt_f1:.3f})")

    print("\n" + "=" * 85)
    print("  2. UNTOUCHED HOLDOUT TEST SET PERFORMANCE (20% Data - Zero Leakage)")
    print("=" * 85)

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_dev, y_dev)

    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_preds_default = (test_probs >= 0.50).astype(int)
    test_preds_optimal = (test_probs >= opt_thresh).astype(int)

    test_metrics = [
        {
            "Dataset": "Holdout Test (Thresh=0.50)",
            "ROC-AUC": f"{roc_auc_score(y_test, test_probs):.3f}",
            "Accuracy": f"{accuracy_score(y_test, test_preds_default):.3f}",
            "F1-Score": f"{f1_score(y_test, test_preds_default):.3f}",
            "Precision": f"{precision_score(y_test, test_preds_default):.3f}",
            "Recall": f"{recall_score(y_test, test_preds_default):.3f}"
        },
        {
            "Dataset": f"Holdout Test (Thresh={opt_thresh:.2f})",
            "ROC-AUC": f"{roc_auc_score(y_test, test_probs):.3f}",
            "Accuracy": f"{accuracy_score(y_test, test_preds_optimal):.3f}",
            "F1-Score": f"{f1_score(y_test, test_preds_optimal):.3f}",
            "Precision": f"{precision_score(y_test, test_preds_optimal):.3f}",
            "Recall": f"{recall_score(y_test, test_preds_optimal):.3f}"
        }
    ]

    test_metrics_df = pd.DataFrame(test_metrics)
    print(test_metrics_df.to_string(index=False))
    print("=" * 85 + "\n")

    print("[SHAP ANALYSIS] Computing SHAP values on Holdout Test Set...")
    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_test)
    except Exception:
        print("[SHAP FALLBACK] Using compatibility explainer mode...")
        explainer = shap.Explainer(final_model.predict, X_test[:100])
        shap_values = explainer(X_test).values

    if isinstance(shap_values, list):
        shap_vals_to_plot = shap_values[1]
    else:
        shap_vals_to_plot = shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_to_plot, pd.DataFrame(X_test, columns=feature_cols), show=False)
    plt.title("XGBoost SHAP Feature Importance (Holdout Test Set)", fontsize=13, fontweight='bold')
    plt.tight_layout()

    shap_output_png = os.path.join(target_out_dir, "shap_feature_importance.png")
    plt.savefig(shap_output_png, dpi=300)
    print(f"[SAVED SHAP PLOT]: {shap_output_png}")
    plt.close('all')


# =====================================================================
# 5. MAIN CLI PARSER & EXECUTION
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost Tuning & Evaluation Pipeline")

    parser.add_argument(
        "inputs", 
        nargs="*", 
        default=["experiments_output/sampled"],
        help="Aggregated feature CSV files, directories, or glob patterns (default: experiments_output/sampled)"
    )
    parser.add_argument(
        "--llm_model", "--model",
        type=str,
        default=None,
        help="Target LLM model to train on (e.g. 'GPT4', 'Claude-3', or 'GPT4_LLM'). If omitted or 'all', trains on all LLMs combined."
    )
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials to run (set 0 to skip tuning)")
    parser.add_argument("--reset_study", action="store_true", help="Reset/delete existing Optuna database and study history")
    parser.add_argument("--out_dir", type=str, default="optuna_xgboost", help="Base output directory (subfolders created per input file & model)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Normalize model filter target
    target_llm = None if (not args.llm_model or args.llm_model.lower() == "all") else args.llm_model

    # 1. Resolve input files
    file_paths = resolve_input_files(args.inputs)
    if not file_paths:
        print("[ERROR] No valid sentence-level aggregated_features.csv files found!")
        sys.exit(1)

    subfolder_name = get_input_subfolder_name(file_paths)
    if target_llm:
        clean_model_folder = target_llm.replace("_LLM", "").strip()
        subfolder_name = os.path.join(subfolder_name, clean_model_folder)

    target_out_dir = os.path.join(args.out_dir, subfolder_name)
    os.makedirs(target_out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"  PIPELINE OUTPUT SUBFOLDER: {target_out_dir}")
    print("=" * 80)

    # 2. Load and filter dataset for target model
    sent_df = load_and_merge_feature_csvs(file_paths, target_model=target_llm)

    dev_df, test_df = create_holdout_split(sent_df, test_size=0.20)

    # 3. Extract Feature Matrices
    feature_cols = extract_feature_columns(dev_df)

    print(f"[FEATURE AUDIT] Identified {len(feature_cols)} numeric feature columns for XGBoost.")
    print(f"[FEATURE AUDIT] Verified 'is_llm' target column is EXCLUDED from feature set: True")

    doc_col = "_id" if "_id" in dev_df.columns else "sentence_id"
    X_dev = dev_df[feature_cols].values
    y_dev = dev_df["is_llm"].values
    groups_dev = dev_df[doc_col].values

    n_human_dev = int(np.sum(y_dev == 0))
    n_llm_dev = int(np.sum(y_dev == 1))
    scale_pos_weight_dev = float(n_human_dev / n_llm_dev) if n_llm_dev > 0 else 1.0

    print(f"[THREAD MANAGER] Total CPUs: {TOTAL_CPUS} | Capped to: {TARGET_CPUS} threads")

    # 4. Optuna Hyperparameter Search strictly inside target_out_dir
    best_params = run_or_resume_optuna(
        X_dev=X_dev, 
        y_dev=y_dev, 
        groups_dev=groups_dev, 
        scale_pos_weight=scale_pos_weight_dev, 
        target_out_dir=target_out_dir,
        n_trials=args.n_trials, 
        reset_study=args.reset_study,
        study_name=f"xgboost_{subfolder_name}"
    )

    # 5. Full Final Evaluation on Dev CV AND 20% Untouched Holdout Test Set
    evaluate_final_xgboost(
        dev_df=dev_df, 
        test_df=test_df, 
        feature_cols=feature_cols, 
        best_params=best_params, 
        target_out_dir=target_out_dir
    )


if __name__ == "__main__":
    main()