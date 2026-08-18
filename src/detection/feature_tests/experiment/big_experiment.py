#!/usr/bin/env python3
"""
Experiment Scheduler for LLM Sentence & Abstract Trajectory Analysis
Features: Two-Tier Data Cache Architecture, VRAM model grouping optimization, Selective Reset, Master Comparison Report.
Includes document _id tracking for leakage-free GroupKFold train/test splits.
"""

import argparse
import gc
import json
import os
import re
import sys
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import roc_curve, auc, roc_auc_score

import nltk
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =====================================================================
# 1. DEFINE YOUR EXPERIMENT CONFIGURATIONS MATRIX
# =====================================================================

STUDY_CONFIGS = [
    # --- Dutch Abstracts: Sentence-Level (ALL DATA: n_samples=None) ---
    {
        "study_name": "exp01_dutch_qwen27b_sent",
        "model_name": "Qwen/Qwen2.5-7B",
        "language": "dutch",
        "dataset": "abstracts",
        "llm_col": "qwen3.6:27b_single",
        "parquet_path": "/home/gderijck/internship/data/gold/llm_added.parquet",
        "eval_unit": "sentence",  # 'sentence' or 'abstract'
        "n_samples": None,        # None or -1 runs on ALL DATA
        "min_words": 8,
    },
    {
        "study_name": "exp02_dutch_gemma4_e4b_sent",
        "model_name": "Qwen/Qwen2.5-7B",
        "language": "dutch",
        "dataset": "abstracts",
        "llm_col": "gemma4:e4b_single",
        "parquet_path": "/home/gderijck/internship/data/gold/llm_added.parquet",
        "eval_unit": "sentence",
        "n_samples": None,
        "min_words": 8,
    },
    {
        "study_name": "exp03_dutch_qwen4b_sent",
        "model_name": "Qwen/Qwen2.5-7B",
        "language": "dutch",
        "dataset": "abstracts",
        "llm_col": "qwen3.5:4b_single",
        "parquet_path": "/home/gderijck/internship/data/gold/llm_added.parquet",
        "eval_unit": "sentence",
        "n_samples": None,
        "min_words": 8,
    },
    {
        "study_name": "exp04_dutch_gemma4_26b_sent",
        "model_name": "Qwen/Qwen2.5-7B",
        "language": "dutch",
        "dataset": "abstracts",
        "llm_col": "gemma4:26b_single",
        "parquet_path": "/home/gderijck/internship/data/gold/llm_added.parquet",
        "eval_unit": "sentence",
        "n_samples": None,
        "min_words": 8,
    },
    
    # --- Dutch Abstracts: Full Abstract-Level (Higher Context Depth) ---
    {
        "study_name": "exp05_dutch_qwen27b_full_abstract",
        "model_name": "Qwen/Qwen2.5-7B",
        "language": "dutch",
        "dataset": "abstracts",
        "llm_col": "qwen3.6:27b_single",
        "parquet_path": "/home/gderijck/internship/data/gold/llm_added.parquet",
        "eval_unit": "abstract",  # Full Abstract!
        "n_samples": None,
        "min_words": 8,
    },
]


STUDY_CONFIGS_DEBUG = [
    {
        "study_name": "test_run_dutch_qwen05b",
        "model_name": "Qwen/Qwen2.5-0.5B",  # Uses your cached 0.5B model!
        "language": "dutch",
        "dataset": "abstracts",
        "llm_col": "qwen3.6:27b_single",
        "parquet_path": "/home/gderijck/internship/data/gold/llm_added.parquet",
        "eval_unit": "sentence",
        "n_samples": 5,                     # Small sample size for instant verification
        "min_words": 8,
    }
]

# =====================================================================
# 2. GLOBAL DATA CACHE MANAGER (TIER 1: STUDY-AGNOSTIC)
# =====================================================================
def get_global_cache_paths(config, cache_root="data_cache"):
    """
    Constructs study-agnostic file paths for global feature/token caching based purely on:
    (evaluator_model, dataset, llm_col, eval_unit, n_samples)
    """
    model_clean = config["model_name"].replace("/", "_").replace(":", "_")
    llm_col_clean = str(config.get("llm_col", "default")).replace(":", "_").replace("/", "_")
    ds_name = config.get("dataset", "dataset")
    eval_unit = config.get("eval_unit", "sentence")
    
    n_samples_cfg = config.get("n_samples")
    is_full = (n_samples_cfg is None) or (n_samples_cfg <= 0)
    sample_tag = "FULL" if is_full else f"SAMPLE_{n_samples_cfg}"
    
    model_dir = os.path.join(cache_root, model_clean)
    os.makedirs(model_dir, exist_ok=True)
    
    prefix = f"{ds_name}_{llm_col_clean}_{eval_unit}_{sample_tag}"
    feat_path = os.path.join(model_dir, f"{prefix}_features.csv")
    token_path = os.path.join(model_dir, f"{prefix}_tokens.csv")
    
    return feat_path, token_path

def append_chunk_to_csv(records, csv_path):
    """Appends a batch of records to CSV incrementally to save RAM."""
    if not records:
        return
    df_chunk = pd.DataFrame(records)
    file_exists = os.path.exists(csv_path)
    df_chunk.to_csv(csv_path, mode='a', index=False, header=not file_exists)

# =====================================================================
# 3. UTILITY & VALIDATION FUNCTIONS
# =====================================================================
def setup_nltk():
    """Ensure NLTK sentence tokenizers are downloaded."""
    for resource in ['punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)


def is_valid_sentence(s, min_words):
    if s is None or pd.isna(s):
        return False
    s_str = str(s).strip()
    invalid_flags = {'FAILED_GENERATION', 'FAILED_VALIDATION', 'NONE', 'NAN', 'NA', 'NULL', '<NA>', ''}
    if s_str.upper() in invalid_flags:
        return False
    if len(s_str.split()) < min_words:
        return False
    return True


def load_data(language, dataset_type, n_samples, min_words, parquet_path=None, llm_col=None, eval_unit="sentence"):
    setup_nltk()
    from nltk.tokenize import sent_tokenize

    human_texts = []  # List of tuples: (doc_id, text)
    llm_texts = []    # List of tuples: (doc_id, text)

    if dataset_type == "abstracts":
        print(f"Loading Custom Parquet Dataset: {parquet_path}")
        if not llm_col:
            raise ValueError("Must specify llm_col when dataset='abstracts'.")

        df_parquet = pd.read_parquet(parquet_path)

        for idx, row in df_parquet.iterrows():
            doc_id = row['_id'] if '_id' in row else f'doc_{idx}'
            h_val = row["abstract_sentence"]
            h_sents = h_val if isinstance(h_val, (list, np.ndarray)) else [h_val]
            valid_h = [str(s).strip() for s in h_sents if is_valid_sentence(s, min_words)]

            l_val = row[llm_col]
            l_sents = l_val if isinstance(l_val, (list, np.ndarray)) else [l_val]
            valid_l = [str(s).strip() for s in l_sents if is_valid_sentence(s, min_words)]

            if eval_unit == "abstract":
                if len(valid_h) > 0:
                    human_texts.append((doc_id, " ".join(valid_h)))
                if len(valid_l) > 0:
                    llm_texts.append((doc_id, " ".join(valid_l)))
            else:
                for s in valid_h:
                    human_texts.append((doc_id, s))
                for s in valid_l:
                    llm_texts.append((doc_id, s))

    else:
        print(f"Loading Standard Dataset for Language: [{language.upper()}]")
        if language == "english":
            try:
                ds = load_dataset("Hello-SimpleAI/HC3", name="all", split="train")
            except Exception:
                ds = load_dataset("Hello-SimpleAI/HC3", revision="refs/convert/parquet", split="train")

            for entry_idx, entry in enumerate(ds):
                doc_id = entry.get("_id", entry.get("id", f"hc3_{entry_idx}"))
                for ans in entry["human_answers"]:
                    for s in sent_tokenize(ans):
                        if is_valid_sentence(s, min_words):
                            human_texts.append((doc_id, s.strip()))
                for ans in entry["chatgpt_answers"]:
                    for s in sent_tokenize(ans):
                        if is_valid_sentence(s, min_words):
                            llm_texts.append((doc_id, s.strip()))

    if len(human_texts) == 0 or len(llm_texts) == 0:
        raise ValueError(f"Extracted 0 valid texts! (Human: {len(human_texts)}, LLM: {len(llm_texts)})")

    # Sample logic preserving (_id, text) tuple pairs safely
    np.random.seed(42)
    if n_samples is not None and n_samples > 0 and n_samples < len(human_texts):
        idx_h = np.random.choice(len(human_texts), n_samples, replace=False)
        idx_l = np.random.choice(len(llm_texts), n_samples, replace=False)
        human_sample = [human_texts[i] for i in idx_h]
        llm_sample = [llm_texts[i] for i in idx_l]
        print(f"Sampled {len(human_sample)} Human and {len(llm_sample)} LLM [{eval_unit.upper()}] samples.")
    else:
        print(f"Using ALL available data ({len(human_texts)} Human, {len(llm_texts)} LLM [{eval_unit.upper()}] samples).")
        human_sample = human_texts
        llm_sample = llm_texts

    return human_sample, llm_sample


@torch.no_grad()
def extract_token_trajectory(text, model, tokenizer, max_length=2048):
    device = next(model.parameters()).device
    
    inputs = tokenizer(
        text.strip(), 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length
    ).to(device)
    
    input_ids = inputs["input_ids"][0]

    if len(input_ids) < 3:
        return None

    outputs = model(inputs["input_ids"], use_cache=False)
    logits = outputs.logits[0]

    shift_logits = logits[:-1, :]
    shift_labels = input_ids[1:]

    special_ids = set(tokenizer.all_special_ids)
    
    valid_positions = [
        pos for pos, token_id in enumerate(shift_labels)
        if token_id.item() not in special_ids
    ]
    
    total_valid_tokens = len(valid_positions)
    if total_valid_tokens < 2:
        return None

    trajectory = []

    for norm_idx, pos in enumerate(valid_positions):
        token_id = shift_labels[pos].item()
        token_logits = shift_logits[pos]

        probs = F.softmax(token_logits, dim=-1)
        log_probs = F.log_softmax(token_logits, dim=-1)

        raw_log_prob = log_probs[token_id].item()
        entropy = -torch.sum(probs * log_probs).item()
        norm_score = raw_log_prob + entropy
        rank = (token_logits > token_logits[token_id]).sum().item() + 1
        log_rank = float(np.log(rank))

        trajectory.append({
            "token_pos": norm_idx + 1,
            "norm_pos": (norm_idx + 1) / total_valid_tokens,
            "raw_log_prob": raw_log_prob,
            "entropy": entropy,
            "entropy_norm_score": norm_score,
            "rank": rank,
            "log_rank": log_rank
        })

    return trajectory

@torch.no_grad()
def process_batch(batch_items, label_prefix, start_sentence_id, model, tokenizer, max_length=2048):
    """
    Memory-optimized GPU trajectory extraction for large vocabulary models (e.g., Qwen2.5 with V=152k).
    """
    if not batch_items:
        return [], start_sentence_id

    doc_ids, raw_texts = zip(*batch_items)
    texts = [t.strip() for t in raw_texts]
    
    device = next(model.parameters()).device
    special_ids = set(tokenizer.all_special_ids)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    inputs = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    input_ids = inputs["input_ids"]          # [B, L]
    attention_mask = inputs["attention_mask"] # [B, L]

    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits                  # [B, L, V]

    shift_logits = logits[:, :-1, :]         # [B, L-1, V]
    shift_labels = input_ids[:, 1:]          # [B, L-1]
    shift_mask = attention_mask[:, 1:]        # [B, L-1]

    # 1. Rank calculation (performed directly on logits)
    target_logits = torch.gather(shift_logits, dim=-1, index=shift_labels.unsqueeze(-1)) # [B, L-1, 1]
    rank = (shift_logits > target_logits).sum(dim=-1) + 1 # [B, L-1]
    log_rank = torch.log(rank.float())                    # [B, L-1]
    del target_logits

    # 2. Log-softmax & raw_log_prob
    log_probs = F.log_softmax(shift_logits, dim=-1) # [B, L-1, V]
    del shift_logits                                # Free logits immediately

    raw_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1) # [B, L-1]

    # 3. Entropy calculation (using exp in-place to avoid duplicate probs tensor)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1) # [B, L-1]
    del probs, log_probs                            # Free large VRAM tensors immediately
    
    entropy_norm_score = raw_log_probs + entropy          # [B, L-1]

    # 4. Single transfer to CPU NumPy
    raw_log_probs_np = raw_log_probs.cpu().numpy()
    entropy_np = entropy.cpu().numpy()
    norm_score_np = entropy_norm_score.cpu().numpy()
    rank_np = rank.cpu().numpy()
    log_rank_np = log_rank.cpu().numpy()
    mask_np = shift_mask.cpu().numpy()
    labels_np = shift_labels.cpu().numpy()

    # Free remaining GPU tensors
    del raw_log_probs, entropy, entropy_norm_score, rank, log_rank, shift_mask, shift_labels, outputs, logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    records = []
    current_sentence_id = start_sentence_id

    # Fast CPU extraction
    for b_idx in range(len(batch_items)):
        doc_id = doc_ids[b_idx]
        text = texts[b_idx]

        seq_mask = mask_np[b_idx]
        seq_labels = labels_np[b_idx]

        valid_positions = [
            pos for pos in range(len(seq_labels))
            if seq_mask[pos] == 1 and seq_labels[pos] not in special_ids
        ]

        total_valid_tokens = len(valid_positions)
        if total_valid_tokens < 2:
            continue

        sid = f"{label_prefix}_{current_sentence_id}"
        label_name = "Human" if label_prefix == "H" else "LLM"

        for norm_idx, pos in enumerate(valid_positions):
            records.append({
                "token_pos": norm_idx + 1,
                "norm_pos": (norm_idx + 1) / total_valid_tokens,
                "raw_log_prob": float(raw_log_probs_np[b_idx, pos]),
                "entropy": float(entropy_np[b_idx, pos]),
                "entropy_norm_score": float(norm_score_np[b_idx, pos]),
                "rank": int(rank_np[b_idx, pos]),
                "log_rank": float(log_rank_np[b_idx, pos]),
                "sentence_id": sid,
                "_id": doc_id,
                "label": label_name,
                "text": text
            })

        current_sentence_id += 1

    return records, current_sentence_id

# =====================================================================
# 4. FEATURE AGGREGATION, STATS & VISUALIZATION
# =====================================================================
def aggregate_sentence_features(token_df):
    records = []
    def calc_slope(x, y):
        if len(x) >= 2:
            return stats.linregress(x, y).slope
        return 0.0

    for sid, group in token_df.groupby("sentence_id"):
        label = group["label"].iloc[0]
        doc_id = group['_id'].iloc[0]
        is_llm = 1 if label == 'LLM' else 0
        group = group.sort_values("token_pos")
        
        norm_pos = group["norm_pos"].values
        log_rank = group["log_rank"].values
        raw_log_prob = group["raw_log_prob"].values
        entropy = group["entropy"].values
        entropy_norm_score = group["entropy_norm_score"].values
        length = len(group)
        
        diff_log_rank = np.diff(log_rank) if length > 1 else np.array([0.0])
        diff_log_prob = np.diff(raw_log_prob) if length > 1 else np.array([0.0])

        record = {
            "sentence_id": sid,
            '_id': doc_id,  # Retained for GroupKFold train/test splits
            "label": label,
            "is_llm": is_llm,
            "token_length": length,
            "mean_log_rank": np.mean(log_rank),
            "std_log_rank": np.std(log_rank),
            "max_log_rank": np.max(log_rank),
            "p75_log_rank": np.percentile(log_rank, 75),
            "slope_log_rank": calc_slope(norm_pos, log_rank),
            "volatility_log_rank": np.var(diff_log_rank),
            "mean_entropy_norm": np.mean(entropy_norm_score),
            "std_entropy_norm": np.std(entropy_norm_score),
            "slope_entropy_norm": calc_slope(norm_pos, entropy_norm_score),
            "volatility_entropy_norm": np.var(np.diff(entropy_norm_score)) if length > 1 else 0.0,
            "mean_log_prob": np.mean(raw_log_prob),
            "std_log_prob": np.std(raw_log_prob),
            "slope_log_prob": calc_slope(norm_pos, raw_log_prob),
            "volatility_log_prob": np.var(diff_log_prob),
            "mean_entropy": np.mean(entropy),
            "std_entropy": np.std(entropy),
            "slope_entropy": calc_slope(norm_pos, entropy),
        }
        records.append(record)
        
    return pd.DataFrame(records)


def compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (np.mean(group2) - np.mean(group1)) / s_pooled if s_pooled != 0 else 0.0


def format_p_value(p):
    if p < 1e-4:
        return f"{p:.2e} ***"
    elif p < 0.001:
        return f"{p:.2e} **"
    elif p < 0.05:
        return f"{p:.4f} *"
    else:
        return f"{p:.4f} (ns)"


def calculate_significance(sent_df):
    human_df = sent_df[sent_df["is_llm"] == 0]
    llm_df = sent_df[sent_df["is_llm"] == 1]
    
    ignore_cols = ["sentence_id", "_id", "doc_id", "label", "is_llm", "text"]
    feature_cols = [col for col in sent_df.columns if col not in ignore_cols]
    results = []
    
    for feat in feature_cols:
        h_vals = human_df[feat].values
        l_vals = llm_df[feat].values
        
        u_stat, p_mw = stats.mannwhitneyu(h_vals, l_vals, alternative='two-sided')
        lev_stat, p_lev = stats.levene(h_vals, l_vals)
        d_val = compute_cohens_d(h_vals, l_vals)
        
        if len(np.unique(sent_df["is_llm"])) > 1:
            auc_val = roc_auc_score(sent_df["is_llm"], sent_df[feat])
            directional_auc = max(auc_val, 1.0 - auc_val)
        else:
            directional_auc = 0.5
        
        results.append({
            "feature": feat,
            "human_mean_std": f"{np.mean(h_vals):.3f} ± {np.std(h_vals):.3f}",
            "llm_mean_std": f"{np.mean(l_vals):.3f} ± {np.std(l_vals):.3f}",
            "p_location (MW-U)": format_p_value(p_mw),
            "p_variance (Levene)": format_p_value(p_lev),
            "cohens_d": round(d_val, 3),
            "roc_auc": round(directional_auc, 3),
            "_raw_auc": directional_auc,
            "_raw_p_mw": p_mw,
            "_raw_p_lev": p_lev
        })
        
    res_df = pd.DataFrame(results)
    return res_df.sort_values(by="_raw_auc", ascending=False).reset_index(drop=True)


def generate_visualizations(token_df, sent_df, sig_df, output_png_path, exp_title):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    palette = {"Human": "#2b5c8f", "LLM": "#d95f02"}

    # 1. Token Trajectory Across Depth
    token_df["pos_bin"] = pd.cut(
        token_df["norm_pos"], 
        bins=np.linspace(0, 1, 11), 
        labels=np.linspace(0.1, 1.0, 10)
    ).astype(float)

    sns.lineplot(data=token_df, x="pos_bin", y="log_rank", hue="label", palette=palette, ax=axes[0, 0], marker="o", err_style="band")
    axes[0, 0].set_title("1. Token Log-Rank Trajectory Across Depth", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Normalized Depth (0.0 = Start, 1.0 = End)")
    axes[0, 0].set_ylabel("Log Rank")
    axes[0, 0].legend(title="Source")  # Renamed from 'Model' to 'Source'

    # 2. Slope KDE Distribution (Fixes the empty legend box!)
    human_slopes = sent_df[sent_df["label"] == "Human"]["slope_log_rank"]
    llm_slopes = sent_df[sent_df["label"] == "LLM"]["slope_log_rank"]
    _, p_val = stats.mannwhitneyu(human_slopes, llm_slopes, alternative='two-sided')

    sns.kdeplot(data=sent_df, x="slope_log_rank", hue="label", fill=True, common_norm=False, palette=palette, ax=axes[0, 1], alpha=0.4)
    axes[0, 1].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].set_title(f"2. Trajectory Slope Distribution (p={p_val:.2e})", fontsize=11, fontweight='bold')
    # Cleanly move Seaborn's KDE legend instead of overwriting it
    if axes[0, 1].get_legend() is not None:
        sns.move_legend(axes[0, 1], "upper right", title="Source")

    # 3. Dynamic ROC Curve using the Best Feature from Significance Testing
    top_feat = sig_df.iloc[0]["feature"] if (sig_df is not None and not sig_df.empty) else "mean_log_rank"
    y_true = (sent_df["label"] == "LLM").astype(int)
    y_scores = sent_df[top_feat]
    
    # Orient scores so higher score corresponds to LLM
    if roc_auc_score(y_true, y_scores) < 0.5:
        y_scores = -y_scores
        
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc_val = auc(fpr, tpr)
    else:
        fpr, tpr, roc_auc_val = [0, 1], [0, 1], 0.5

    axes[1, 0].plot(fpr, tpr, color='#d95f02', lw=2.5, label=f'AUC ({top_feat}) = {roc_auc_val:.3f}')
    axes[1, 0].plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    axes[1, 0].set_title(f'3. ROC Curve: Top Feature ({top_feat})', fontsize=11, fontweight='bold')
    axes[1, 0].legend(loc="lower right")

    # 4. Mean Log-Rank vs Token Length
    sns.scatterplot(data=sent_df, x="token_length", y="mean_log_rank", hue="label", palette=palette, alpha=0.5, ax=axes[1, 1], s=30)
    sns.regplot(data=sent_df[sent_df['label'] == 'Human'], x='token_length', y='mean_log_rank', ax=axes[1, 1], scatter=False, color='#2b5c8f')
    sns.regplot(data=sent_df[sent_df['label'] == 'LLM'], x='token_length', y='mean_log_rank', ax=axes[1, 1], scatter=False, color='#d95f02')
    axes[1, 1].set_title("4. Mean Log-Rank vs. Token Length", fontsize=11, fontweight='bold')
    axes[1, 1].legend(title="Source")

    plt.suptitle(exp_title, fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.close('all')


# =====================================================================
# 5. EXPERIMENT SCHEDULER CLASS (TIER 2: STANDARDIZED EXPERIMENTS)
# =====================================================================
class ExperimentScheduler:
    def __init__(self, configs, root_dir="experiments_output", cache_dir="data_cache", reset_studies=None, reset_all=False):
        self.configs = configs
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.reset_studies = set(reset_studies) if reset_studies else set()
        self.reset_all = reset_all
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def is_study_completed(self, config):
        study_name = config["study_name"]
        if self.reset_all or study_name in self.reset_studies:
            return False
        
        exp_dir = os.path.join(self.root_dir, study_name)
        required_files = [
            "trajectory_tokens.csv", 
            "aggregated_features.csv", 
            "feature_significance.csv", 
            "visualization_dashboard.png",
            "study_config.json"
        ]
        for f in required_files:
            if not os.path.exists(os.path.join(exp_dir, f)):
                return False
        return True

    def run_all(self):
        pending_studies = []
        for config in self.configs:
            if self.is_study_completed(config):
                print(f"[SKIP] Study '{config['study_name']}' is already completed.")
            else:
                pending_studies.append(config)

        if not pending_studies:
            print("\nAll scheduled studies are already completed! Generating Master Summary Report...")
            self.generate_master_summary()
            return

        print(f"\n==================================================================")
        print(f"  SCHEDULING {len(pending_studies)} STUDIES FOR EXECUTION")
        print(f"==================================================================")

        grouped_studies = defaultdict(list)
        for config in pending_studies:
            grouped_studies[config["model_name"]].append(config)

        for model_name, study_group in grouped_studies.items():
            print(f"\n" + "="*80)
            print(f"EVALUATOR MODEL GROUP: [{model_name}] ({len(study_group)} STUDIES)")
            print("="*80)

            model, tokenizer = None, None

            try:
                # Lazy loading: load GPU model ONLY if at least one study requires extraction
                for config in study_group:
                    global_feat_path, global_tok_path = get_global_cache_paths(config, self.cache_dir)
                    needs_gpu = not (os.path.exists(global_feat_path) and os.path.exists(global_tok_path))
                    
                    if needs_gpu and model is None:
                        print(f"Loading Evaluator Model onto GPU: [{model_name}]...")
                        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto",
                            trust_remote_code=True
                        )
                        model.eval()

                    self._execute_single_study(config, model, tokenizer)

            except Exception as e:
                print(f"[CRITICAL ERROR] Execution failed for model '{model_name}': {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

            finally:
                if model is not None:
                    print(f"\n[Memory Manager] Unloading [{model_name}] from GPU...")
                    del model
                    del tokenizer
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    plt.close('all')

        self.generate_master_summary()

    def _execute_single_study(self, config, model, tokenizer):
        study_name = config["study_name"]
        exp_dir = os.path.join(self.root_dir, study_name)
        os.makedirs(exp_dir, exist_ok=True)

        print(f"\n>>> EXECUTING STUDY: [{study_name}] <<<")
        print(f"Dataset: {config['dataset']} | LLM Col: {config.get('llm_col')} | Unit: {config.get('eval_unit', 'sentence')}")

        global_feat_path, global_tok_path = get_global_cache_paths(config, self.cache_dir)
        cache_hit = (
            os.path.exists(global_feat_path) and 
            os.path.exists(global_tok_path) and 
            not self.reset_all and 
            study_name not in self.reset_studies
        )

        if cache_hit:
            print(f"[GLOBAL CACHE HIT] Reusing pre-computed trajectories from:\n  -> {global_feat_path}")
            sent_df = pd.read_csv(global_feat_path)
            token_df = pd.read_csv(global_tok_path)
        else:
            print(f"[CACHE MISS] Extracting trajectories on GPU...")
            human_sample, llm_sample = load_data(
                language=config.get("language", "english"),
                dataset_type=config.get("dataset", "default"),
                n_samples=config.get("n_samples"),
                min_words=config.get("min_words", 8),
                parquet_path=config.get("parquet_path"),
                llm_col=config.get("llm_col"),
                eval_unit=config.get("eval_unit", "sentence")
            )

            if os.path.exists(global_tok_path):
                os.remove(global_tok_path)

            eval_unit = config.get("eval_unit", "sentence")
            BATCH_SIZE = 32 if eval_unit == "sentence" else 8
            FLUSH_THRESHOLD = 100_000  # Flush every ~100k token records (~3,000 sentences)
            
            chunk_records = []
            sentence_id = 0

            human_sample = sorted(human_sample, key=lambda x: len(x[1]))
            llm_sample = sorted(llm_sample, key=lambda x: len(x[1]))

            # 1. Process Human Texts in GPU Batches
            pbar_h = tqdm(total=len(human_sample), desc="Evaluating Human Texts (Batched)")
            for i in range(0, len(human_sample), BATCH_SIZE):
                batch_items = human_sample[i : i + BATCH_SIZE]
                records, sentence_id = process_batch(batch_items, "H", sentence_id, model, tokenizer)
                chunk_records.extend(records)
                pbar_h.update(len(batch_items))

                # Flush to disk and free memory every 100k records
                if len(chunk_records) >= FLUSH_THRESHOLD:
                    append_chunk_to_csv(chunk_records, global_tok_path)
                    chunk_records.clear()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            pbar_h.close()

            # 2. Process LLM Texts in GPU Batches
            sentence_id_llm = 0
            pbar_l = tqdm(total=len(llm_sample), desc="Evaluating LLM Texts (Batched)")
            for i in range(0, len(llm_sample), BATCH_SIZE):
                batch_items = llm_sample[i : i + BATCH_SIZE]
                records, sentence_id_llm = process_batch(batch_items, "L", sentence_id_llm, model, tokenizer)
                chunk_records.extend(records)
                pbar_l.update(len(batch_items))

                if len(chunk_records) >= FLUSH_THRESHOLD:
                    append_chunk_to_csv(chunk_records, global_tok_path)
                    chunk_records.clear()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            pbar_l.close()

            # Final flush for leftover records
            if chunk_records:
                append_chunk_to_csv(chunk_records, global_tok_path)
                chunk_records.clear()

            token_df = pd.read_csv(global_tok_path) if os.path.exists(global_tok_path) else pd.DataFrame()
            sent_df = aggregate_sentence_features(token_df)
            sent_df.to_csv(global_feat_path, index=False)
            print(f"[SAVED TO GLOBAL CACHE]:\n  -> {global_feat_path}")

        # Save Standardized Artifacts
        std_feat_path = os.path.join(exp_dir, "aggregated_features.csv")
        std_tok_path = os.path.join(exp_dir, "trajectory_tokens.csv")
        sent_df.to_csv(std_feat_path, index=False)
        token_df.to_csv(std_tok_path, index=False)

        # Reports & Plots
        sig_df = calculate_significance(sent_df)
        report_cols = ["feature", "human_mean_std", "llm_mean_std", "p_location (MW-U)", "p_variance (Levene)", "cohens_d", "roc_auc"]
        export_df = sig_df[report_cols]

        print("\n" + "="*80)
        print(f"  STATISTICAL SIGNIFICANCE REPORT: [{study_name.upper()}]")
        print("="*80)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(export_df.to_string(index=False))
        print("="*80 + "\n")

        sig_csv_path = os.path.join(exp_dir, "feature_significance.csv")
        export_df.to_csv(sig_csv_path, index=False)

        if not token_df.empty:
            exp_title = f"Study: {study_name}\nModel: {config['model_name']} | Unit: {config.get('eval_unit', 'sentence')}"
            generate_visualizations(token_df, sent_df, sig_df, os.path.join(exp_dir, "visualization_dashboard.png"), exp_title)

        with open(os.path.join(exp_dir, "study_config.json"), "w") as f:
            json.dump(config, f, indent=4)

        top_feature = sig_df.iloc[0]
        print(f"[FINISHED] [{study_name}] Top Feature: '{top_feature['feature']}' (AUC = {top_feature['roc_auc']:.3f})")



    def generate_master_summary(self):
        """Scans all completed studies and outputs a master comparison table."""
        summary_rows = []
        for config in self.configs:
            study_name = config["study_name"]
            sig_csv = os.path.join(self.root_dir, study_name, "feature_significance.csv")
            
            if os.path.exists(sig_csv):
                sig_df = pd.read_csv(sig_csv)
                top_row = sig_df.iloc[0]
                summary_rows.append({
                    "study_name": study_name,
                    "evaluator_model": config["model_name"],
                    "dataset": config["dataset"],
                    "llm_col": config.get("llm_col", "default"),
                    "eval_unit": config.get("eval_unit", "sentence"),
                    "top_feature": top_row["feature"],
                    "top_roc_auc": top_row["roc_auc"],
                    "cohens_d": top_row["cohens_d"],
                    "p_location (MW-U)": top_row["p_location (MW-U)"],
                    "p_variance (Levene)": top_row["p_variance (Levene)"]
                })

        if summary_rows:
            master_df = pd.DataFrame(summary_rows).sort_values(by="top_roc_auc", ascending=False)
            master_path = os.path.join(self.root_dir, "MASTER_SUMMARY_REPORT.csv")
            master_df.to_csv(master_path, index=False)
            print("\n" + "="*85)
            print("  MASTER EXPERIMENTS COMPARISON REPORT")
            print("="*85)
            print(master_df.to_string(index=False))
            print("="*85)
            print(f"Master summary report saved to: {master_path}\n")


# =====================================================================
# 6. CLI INTERFACE
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Automated Experiment Scheduler.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode using STUDY_CONFIGS_DEBUG (small test sample)."
    )
    parser.add_argument(
        "--reset_studies",
        type=str,
        nargs="+",
        default=None,
        help="List of study_names to force re-running (e.g. --reset_studies exp01_dutch_qwen27b_sent)."
    )
    parser.add_argument(
        "--reset_all",
        action="store_true",
        help="Force re-run ALL scheduled studies, overwriting previous results."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments_output",
        help="Root output directory for all experiment folders."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data_cache",
        help="Root cache directory for global dataset trajectory extractions."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    #DEBUG
    active_configs = STUDY_CONFIGS_DEBUG if args.debug else STUDY_CONFIGS
    if args.debug:
        print("\n" + "=" * 80)
        print("  [DEBUG MODE ACTIVATED] Running with STUDY_CONFIGS_DEBUG matrix")
        print("=" * 80 + "\n")


    if args.reset_all or args.reset_studies:
        if args.reset_all:
            target_names = [cfg["study_name"] for cfg in STUDY_CONFIGS]
            reset_msg = "ALL STUDIES IN CONFIG:\n  - " + "\n  - ".join(target_names)
        else:
            reset_msg = "SPECIFIED STUDIES:\n  - " + "\n  - ".join(args.reset_studies)

        print("\n" + "!" * 80)
        print("  WARNING: SAFETY RESET CONFIRMATION REQUIRED")
        print("!" * 80)
        print(f"You are about to RESET / OVERWRITE the following study results:\n")
        print(reset_msg)
        print("\n" + "-" * 80)
        
        confirm = input("Are you sure you want to proceed and delete/overwrite existing data? [y/N]: ")
        if confirm.strip().lower() not in ['y', 'yes']:
            print("\n[ABORTED] Reset cancelled by user. No existing files were modified. Exiting safely.\n")
            sys.exit(0)
        else:
            print("\n[CONFIRMED] Safety check passed. Proceeding with study reset...\n")

    scheduler = ExperimentScheduler(
        configs=active_configs,
        root_dir=args.output_dir,
        cache_dir=args.cache_dir,
        reset_studies=args.reset_studies,
        reset_all=args.reset_all
    )
    scheduler.run_all()