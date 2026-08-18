import os
import gc
import argparse
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.special
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import nltk
import glob
import ast
from scipy.optimize import curve_fit
from tqdm import tqdm

from huggingface_hub import hf_hub_download
from datasets import load_dataset
from llama_cpp import Llama

# Download NLTK tokenizer data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Set random seed
np.random.seed(420)

# ==========================================
# 0. CONFIGURATION & PATHS
# ==========================================
CLIN33_CSV_PATH = r"E:\code\dta\internship\src\detection\feature_tests\data\clin33_shared_task_generated_dutch.csv"
PARQUET_HUMAN_PATH = r"E:\code\dta\internship\src\detection\feature_tests\data\llm_added.parquet"

MODELS_CONFIG = {
    "qwen2.5-0.5b-base": {
        "repo_id": "QuantFactory/Qwen2.5-0.5B-GGUF",
        "filename": "Qwen2.5-0.5B.Q8_0.gguf",
        "description": "Qwen 2.5 0.5B Base (Q8_0)",
        "llama_kwargs": {
            "n_gpu_layers": -1,
            "tensor_split": None,
            "n_ctx": 2048,
            "n_batch": 512,
            "logits_all": True,
            "flash_attn": False,
            "verbose": False
        }
    },
    "qwen2.5-3b-base": {
        "repo_id": "QuantFactory/Qwen2.5-3B-GGUF",
        "filename": "Qwen2.5-3B.Q8_0.gguf",
        "description": "Qwen 2.5 3B Base (Q8_0)",
        "llama_kwargs": {
            "n_gpu_layers": -1,
            "tensor_split": None,
            "n_ctx": 2048,
            "n_batch": 512,
            "logits_all": True,
            "flash_attn": False,
            "verbose": False
        }
    },
    "qwen2.5-7b-base": {
        "repo_id": "QuantFactory/Qwen2.5-7B-GGUF",
        "filename": "Qwen2.5-7B.Q4_K_M.gguf",
        "description": "Qwen 2.5 7B Base (Q4_K_M)",
        "llama_kwargs": {
            "n_gpu_layers": -1,
            "tensor_split": None,
            "n_ctx": 2048,
            "n_batch": 512,
            "logits_all": True,
            "flash_attn": False,
            "verbose": False
        }
    },
    "eurollm-1.7b-base": {
        "repo_id": "QuantFactory/EuroLLM-1.7B-GGUF",
        "filename": "EuroLLM-1.7B.Q8_0.gguf",
        "description": "EuroLLM 1.7B Base (Q8_0)",
        "llama_kwargs": {
            "n_gpu_layers": -1,
            "tensor_split": None,
            "n_ctx": 2048,
            "n_batch": 512,
            "logits_all": True,
            "flash_attn": False,
            "verbose": False
        }
    },
    "eurollm-9b-base": {
        "repo_id": "QuantFactory/EuroLLM-9B-GGUF",
        "filename": "EuroLLM-9B.Q4_K_M.gguf",
        "description": "EuroLLM 9B Base (Q4_K_M)",
        "llama_kwargs": {
            "n_gpu_layers": -1,
            "tensor_split": None,
            "n_ctx": 2048,
            "n_batch": 512,
            "logits_all": True,
            "flash_attn": False,
            "verbose": False
        }
    }
}

# Define target LLM columns to analyze from the abstracts parquet file



if os.path.exists(PARQUET_HUMAN_PATH):
    _df_meta = pd.read_parquet(PARQUET_HUMAN_PATH)
    TARGET_LLM_COLS = [c for c in _df_meta.columns if c.endswith("_full")]
else:
    TARGET_LLM_COLS = [
        "qwen3.5:4b_full",
        "qwen3.6:27b_full",
        'gemma4:e4b_full',
        'gemma4:26b_full'
    ]

# 2. Define ONE SINGLE RUN configuration containing all _full columns
RUN_CONFIGS = {
    "abstracts_combined_all_models": {
        "model": "qwen",
        "data": "abstracts",
        "llm_columns": TARGET_LLM_COLS,  # All models combined in 1 run!
    }
}



# ==========================================
# 1. HELPER & STATISTICAL FUNCTIONS
# ==========================================
def compute_cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    if dof <= 0:
        return 0.0
    var_x = np.var(x, ddof=1) if nx > 1 else 0.0
    var_y = np.var(y, ddof=1) if ny > 1 else 0.0
    pooled_std = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / dof)
    if pooled_std < 1e-8:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


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
    if sent_df is None or sent_df.empty:
        return pd.DataFrame()

    human_df = sent_df[sent_df["is_llm"] == 0]
    llm_df = sent_df[sent_df["is_llm"] == 1]
    
    ignore_cols = ["sentence_id", "_id", "doc_id", "label", "generator_model", "is_llm", "text", "genre", "token_length"]
    feature_cols = [col for col in sent_df.columns if col not in ignore_cols]
    results = []
    
    for feat in feature_cols:
        h_raw = pd.to_numeric(human_df[feat], errors='coerce').values
        l_raw = pd.to_numeric(llm_df[feat], errors='coerce').values

        h_vals = h_raw[np.isfinite(h_raw)]
        l_vals = l_raw[np.isfinite(l_raw)]

        if len(h_vals) < 2 or len(l_vals) < 2:
            continue
        
        try:
            u_stat, p_mw = stats.mannwhitneyu(h_vals, l_vals, alternative='two-sided')
        except Exception:
            p_mw = 1.0

        try:
            lev_stat, p_lev = stats.levene(h_vals, l_vals)
        except Exception:
            p_lev = 1.0

        d_val = compute_cohens_d(h_vals, l_vals)
        
        try:
            feat_series = pd.to_numeric(sent_df[feat], errors='coerce').values
            valid_mask = np.isfinite(feat_series) & np.isfinite(sent_df["is_llm"].values)
            
            if len(np.unique(sent_df.loc[valid_mask, "is_llm"])) > 1:
                clean_y_true = sent_df.loc[valid_mask, "is_llm"].values
                clean_y_scores = feat_series[valid_mask]
                auc_val = float(roc_auc_score(clean_y_true, clean_y_scores))
            else:
                auc_val = 0.5
        except Exception:
            auc_val = 0.5
        
        results.append({
            "feature": feat,
            "human_mean_std": f"{np.mean(h_vals):.3f} ± {np.std(h_vals):.3f}",
            "llm_mean_std": f"{np.mean(l_vals):.3f} ± {np.std(l_vals):.3f}",
            "cohens_d": round(d_val, 3),
            "roc_auc": round(auc_val, 3),
            "_raw_auc": auc_val,
            "_raw_p_mw": p_mw,
            "_raw_p_lev": p_lev
        })
        
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df["p_mw_fdr"] = stats.false_discovery_control(res_df["_raw_p_mw"].values)
        res_df["p_location (MW-U FDR)"] = res_df["p_mw_fdr"].apply(format_p_value)
        res_df["p_variance (Levene)"] = res_df["_raw_p_lev"].apply(format_p_value)
        
        res_df["_auc_dist"] = np.abs(res_df["_raw_auc"] - 0.5)
        res_df = res_df.sort_values(by="_auc_dist", ascending=False).drop(columns=["_auc_dist"]).reset_index(drop=True)
        res_df = res_df[res_df["p_mw_fdr"] < 0.05].reset_index(drop=True)
        
    return res_df


def is_config_completed(run_dir_name, config):
    output_dir = run_dir_name
    if not os.path.exists(output_dir):
        return False

    data_setting = config.get("data")
    model_setting = config.get("model")

    matching_models = [
        m_name for m_name in MODELS_CONFIG.keys()
        if model_setting in m_name
    ]

    if not matching_models:
        return False

    for model_name in matching_models:
        sig_file = os.path.join(output_dir, f"{model_name}_{data_setting}_significance_results.csv")
        feat_file = os.path.join(output_dir, f"{model_name}_{data_setting}_sentence_features.csv")
        if not (os.path.exists(sig_file) and os.path.exists(feat_file)):
            return False

    return True


# ==========================================
# 2. FEATURE EXTRACTION PIPELINE
# ==========================================
def compute_vectorized_gini(probs, top_k=500):
    probs_arr = np.asarray(probs, dtype=np.float64)
    is_1d = probs_arr.ndim == 1
    probs_2d = np.atleast_2d(probs_arr)
    M, V = probs_2d.shape
    actual_k = min(top_k, V)

    if actual_k < 2:
        gini = np.zeros(M, dtype=np.float64)
        return gini[0] if is_1d else gini

    # 1. Extract top-k probabilities
    topk_probs = np.partition(probs_2d, -actual_k, axis=-1)[:, -actual_k:]
    sorted_topk = np.sort(topk_probs, axis=-1)  # Ascending order: (M, actual_k)

    # 2. Correct Gini weights over local top-k subset
    k_idx = np.arange(1, actual_k + 1, dtype=np.float64)  # 1-based local rank
    weights = (actual_k - k_idx + 0.5) / actual_k         # Shape: (actual_k,)

    # 3. Normalize by total probability mass of the top-k subset
    total_mass = np.sum(sorted_topk, axis=-1, keepdims=True)  # Shape: (M, 1)
    lorenz_area = np.sum(sorted_topk * weights, axis=-1, keepdims=True) / (total_mass + 1e-12)
    
    gini = (1.0 - 2.0 * lorenz_area).squeeze(-1)

    return gini[0] if is_1d else gini


def compute_zipf_exponent(v_logits, top_k=20):
    v_logits_arr = np.asarray(v_logits, dtype=np.float64)
    is_1d = v_logits_arr.ndim == 1
    v_logits_2d = np.atleast_2d(v_logits_arr)
    M, V = v_logits_2d.shape
    actual_k = min(top_k, V)

    if actual_k < 2:
        alphas = np.zeros(M, dtype=np.float64)
        return alphas[0] if is_1d else alphas

    topk_logits = np.partition(v_logits_2d, -actual_k, axis=-1)[:, -actual_k:]
    sorted_topk = np.sort(topk_logits, axis=-1)[:, ::-1]  # Descending order

    log_ranks = np.log(np.arange(1, actual_k + 1, dtype=np.float64))

    mean_x = np.mean(log_ranks)
    var_x = np.var(log_ranks)
    mean_y = np.mean(sorted_topk, axis=-1, keepdims=True)

    cov_xy = np.mean((log_ranks - mean_x) * (sorted_topk - mean_y), axis=-1)
    zipf_alpha = -cov_xy / (var_x + 1e-12)

    return zipf_alpha[0] if is_1d else zipf_alpha

def compute_zipf_mandelbrot_params(
    v_logits, 
    top_k=20, 
    beta_min=0.0, 
    beta_max=10.0, 
    beta_steps=101
):
    v_logits_arr = np.asarray(v_logits, dtype=np.float64)
    is_1d = (v_logits_arr.ndim == 1)
    v_logits_2d = np.atleast_2d(v_logits_arr)
    M, V = v_logits_2d.shape
    actual_k = min(top_k, V)

    if actual_k < 2:
        alphas = np.zeros(M, dtype=np.float64)
        betas = np.zeros(M, dtype=np.float64)
        return (alphas[0], betas[0]) if is_1d else (alphas, betas)

    topk_logits = np.partition(v_logits_2d, -actual_k, axis=-1)[:, -actual_k:]
    sorted_topk = np.sort(topk_logits, axis=-1)[:, ::-1]

    ranks = np.arange(1, actual_k + 1, dtype=np.float64)
    beta_grid = np.linspace(beta_min, beta_max, beta_steps, dtype=np.float64)
    
    log_r_beta = np.log(ranks[None, :] + beta_grid[:, None])
    mean_x = np.mean(log_r_beta, axis=-1, keepdims=True)
    x_cent = log_r_beta - mean_x
    var_x = np.mean(x_cent**2, axis=-1)

    mean_y = np.mean(sorted_topk, axis=-1, keepdims=True)
    y_cent = sorted_topk - mean_y
    var_y = np.mean(y_cent**2, axis=-1, keepdims=True)

    cov_xy = (y_cent @ x_cent.T) / actual_k

    alphas_grid = -cov_xy / (var_x[None, :] + 1e-12)
    alphas_clipped = np.clip(alphas_grid, 1e-4, 20.0)

    mse_grid = var_y + 2.0 * alphas_clipped * cov_xy + (alphas_clipped**2) * var_x[None, :]
    best_beta_idx = np.argmin(mse_grid, axis=-1)

    h = beta_grid[1] - beta_grid[0]
    B = len(beta_grid)
    idx_mid = best_beta_idx
    idx_left = np.clip(idx_mid - 1, 0, B - 1)
    idx_right = np.clip(idx_mid + 1, 0, B - 1)

    row_idx = np.arange(M)
    y1 = mse_grid[row_idx, idx_left]
    y2 = mse_grid[row_idx, idx_mid]
    y3 = mse_grid[row_idx, idx_right]

    denom = y1 - 2.0 * y2 + y3
    is_interior = (idx_mid > 0) & (idx_mid < B - 1)
    valid_parabola = is_interior & (denom > 1e-12)

    delta = np.zeros(M, dtype=np.float64)
    delta[valid_parabola] = -0.5 * h * (y3[valid_parabola] - y1[valid_parabola]) / denom[valid_parabola]
    delta = np.clip(delta, -h, h)

    best_betas = np.clip(beta_grid[idx_mid] + delta, beta_grid[0], beta_grid[-1])

    log_r_ref = np.log(ranks[None, :] + best_betas[:, None])
    x_ref_cent = log_r_ref - np.mean(log_r_ref, axis=-1, keepdims=True)
    var_x_ref = np.mean(x_ref_cent**2, axis=-1)
    cov_xy_ref = np.mean(y_cent * x_ref_cent, axis=-1)

    best_alphas = -cov_xy_ref / (var_x_ref + 1e-12)
    best_alphas = np.clip(best_alphas, 1e-4, 20.0)

    if is_1d:
        return best_alphas[0], best_betas[0]
    return best_alphas, best_betas

def extract_logit_trajectory(
    text,
    doc_id,
    label_prefix,
    sentence_id,
    llm,
    model_source="LLM",
    max_tokens=2048,
    unigram_log_probs=None,
):
    text_clean = text.strip()
    tokens = llm.tokenize(text_clean.encode("utf-8"))

    bos_id = llm.token_bos()
    eos_id = llm.token_eos()
    start_id = bos_id if (bos_id is not None and bos_id != -1) else eos_id

    if start_id is not None and start_id != -1:
        if len(tokens) == 0 or tokens[0] != start_id:
            tokens = [start_id] + tokens

    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    if len(tokens) < 3:
        return [], sentence_id + 1

    llm.reset()
    llm.eval(tokens)

    logits = np.array(llm.eval_logits, dtype=np.float32)
    shift_logits = logits[:-1, :]
    shift_labels = np.array(tokens[1:], dtype=np.int64)

    special_ids = {
        tid for tid in (bos_id, eos_id) if tid is not None and tid != -1
    }
    n_vocab = llm.n_vocab()

    valid_mask = np.array(
        [(tok not in special_ids) and (tok < n_vocab) for tok in shift_labels],
        dtype=bool,
    )

    valid_positions = np.where(valid_mask)[0]
    total_valid_tokens = len(valid_positions)

    if total_valid_tokens < 2:
        return [], sentence_id + 1

    v_logits = shift_logits[valid_positions]
    v_labels = shift_labels[valid_positions]

    lse = scipy.special.logsumexp(v_logits, axis=-1, keepdims=True)
    log_probs = v_logits - lse
    probs = np.exp(log_probs)

    raw_log_probs = log_probs[np.arange(total_valid_tokens), v_labels]
    surprisal = -raw_log_probs
    entropies = -np.sum(probs * log_probs, axis=-1)

    entropy_norm_score = entropies - surprisal
    surprisal_entropy_gap = surprisal - entropies

    top_k_partition = min(2, n_vocab)
    top2_logits = np.partition(v_logits, -top_k_partition, axis=-1)[
        :, -top_k_partition:
    ]
    sorted_top2_logits = np.sort(top2_logits, axis=-1)

    p_top1 = np.exp(sorted_top2_logits[:, -1] - lse.squeeze(-1))
    p_top2 = (
        np.exp(sorted_top2_logits[:, -2] - lse.squeeze(-1))
        if top_k_partition >= 2
        else np.zeros_like(p_top1)
    )

    min_entropies = -np.log(p_top1 + 1e-12)
    renyi2_entropies = -np.log(np.sum(probs**2, axis=-1) + 1e-12)

    target_logits = v_logits[np.arange(total_valid_tokens), v_labels, None]
    ranks = np.sum(v_logits > target_logits, axis=-1) + 1
    cdf_mass = np.sum((v_logits>=target_logits)*probs, axis=-1)

    eff_vocab = np.exp(entropies)
    rank_eff_ratio = ranks / (eff_vocab + 1e-8)
    logit_std = np.std(v_logits, axis=-1)
    top1_top2_margins = p_top1 - p_top2

    gini_coefs = compute_vectorized_gini(probs)
    zipf_alphas = compute_zipf_exponent(v_logits, top_k=20)
    mb_alphas, mb_betas = compute_zipf_mandelbrot_params(v_logits, top_k=20)

    k50 = min(50, n_vocab)
    top50_p = np.partition(probs, -k50, axis=-1)[:, -k50:]
    sorted_top50_p = np.sort(top50_p, axis=-1)

    k5 = min(5, n_vocab)
    k10 = min(10, n_vocab)
    top5_mass = np.sum(sorted_top50_p[:, -k5:], axis=-1)
    top10_mass = np.sum(sorted_top50_p[:, -k10:], axis=-1)
    top50_mass = np.sum(sorted_top50_p, axis=-1)
    concentration_gradient = top5_mass / (top50_mass + 1e-8)

    mean_logit = np.mean(v_logits, axis=-1, keepdims=True)
    std_logit = np.std(v_logits, axis=-1, keepdims=True) + 1e-8
    norm_diff = (v_logits - mean_logit) / std_logit

    logit_skewness = np.mean(norm_diff ** 3, axis=-1)
    logit_kurtosis = np.mean(norm_diff ** 4, axis=-1) - 3.0  # Fisher Kurtosis

    if unigram_log_probs is not None:
        unigram_log_probs_arr = np.asarray(unigram_log_probs)
        unigram_prior_surprisal = -unigram_log_probs_arr[v_labels]
    else:
        unigram_prior_surprisal = np.full_like(surprisal, np.log(n_vocab))

    unigram_igr = (unigram_prior_surprisal - surprisal) / (
        unigram_prior_surprisal + 1e-8
    )
    bci = gini_coefs * (1.0 - top1_top2_margins)

    k_ranks = np.arange(1, n_vocab + 1, dtype=np.float64)
    log_k = np.log(k_ranks)
    log_z = scipy.special.logsumexp(
        -zipf_alphas[:, None] * log_k[None, :], axis=-1
    )
    predicted_zipf_surprisal = zipf_alphas * np.log(ranks) + log_z
    zipf_anomaly = np.abs(surprisal - predicted_zipf_surprisal)

    max_entropy = np.log(n_vocab)
    norm_entropy = entropies / max_entropy
    gini_entropy_gap = gini_coefs - norm_entropy

    if total_valid_tokens >= 3:
        acc_vals = np.diff(surprisal, n=2)
        surprisal_acc = np.pad(acc_vals, (2, 0), mode="edge")
    else:
        surprisal_acc = np.zeros(total_valid_tokens, dtype=np.float32)

    sid = f"{label_prefix}_{sentence_id}"
    label_name = (
        "Human"
        if str(model_source).upper() == "HUMAN"
        else (
            f"{model_source}_LLM"
            if not str(model_source).endswith("_LLM")
            else str(model_source)
        )
    )

    records = [
        {
            "token_pos": idx + 1,
            "norm_pos": (idx + 1) / total_valid_tokens,
            "raw_log_prob": float(raw_log_probs[idx]),
            "surprisal": float(surprisal[idx]),
            "entropy": float(entropies[idx]),
            "entropy_norm_score": float(entropy_norm_score[idx]),
            "surprisal_entropy_gap": float(surprisal_entropy_gap[idx]),
            "min_entropy": float(min_entropies[idx]),
            "renyi2_entropy": float(renyi2_entropies[idx]),
            "cdf_mass": float(cdf_mass[idx]),
            "rank_eff_ratio": float(rank_eff_ratio[idx]),
            "logit_std": float(logit_std[idx]),
            "gini_coef": float(gini_coefs[idx]),
            "zipf_alpha": float(zipf_alphas[idx]),
            "mandelbrot_alpha": float(mb_alphas[idx]),
            "mandelbrot_beta": float(mb_betas[idx]),
            "top5_mass": float(top5_mass[idx]),
            "top10_mass": float(top10_mass[idx]),
            "top50_mass": float(top50_mass[idx]),
            "logit_skewness": float(logit_skewness[idx]),
            "logit_kurtosis": float(logit_kurtosis[idx]),
            "unigram_igr": float(unigram_igr[idx]),
            "rank": int(ranks[idx]),
            "log_rank": float(np.log(ranks[idx])),
            "top1_top2_margin": float(top1_top2_margins[idx]),
            "surprisal_acc": float(surprisal_acc[idx]),
            "bci": float(bci[idx]),
            "concentration_gradient": float(concentration_gradient[idx]),
            "zipf_anomaly": float(zipf_anomaly[idx]),
            "gini_entropy_gap": float(gini_entropy_gap[idx]),
            "sentence_id": sid,
            "_id": doc_id,
            "generator_model": model_source,
            "label": label_name,
        }
        for idx in range(total_valid_tokens)
    ]

    return records, sentence_id + 1


def calc_slope(x, y):
    if len(x) >= 2 and np.std(x) > 1e-8:
        return float(scipy.stats.linregress(x, y).slope)
    return 0.0




def extract_array_trajectory_features(norm_pos, array_vals, feature_prefix, num_bins=10):
    features = {}
    if len(array_vals) == 0:
        return features

    # 1. Positional Binned Interpolation (Steps 1 to 10)
    target_bins = np.linspace(0.1, 1.0, num_bins)
    interpolated = np.interp(target_bins, norm_pos, array_vals)
    
    for i in range(num_bins):
        features[f"{feature_prefix}_step_{i+1:02d}"] = float(interpolated[i])

    # 2. Local Volatility (1st-Order Adjacent Differences)
    adj_diffs = np.abs(np.diff(interpolated))
    
    for i in range(len(adj_diffs)):
        features[f"{feature_prefix}_diff_step_{i+1:02d}_{i+2:02d}"] = float(adj_diffs[i])
        
    features[f"{feature_prefix}_total_variation"] = float(np.sum(adj_diffs))
    features[f"{feature_prefix}_max_local_jump"] = float(np.max(adj_diffs)) if len(adj_diffs) > 0 else 0.0
    features[f"{feature_prefix}_mean_local_jump"] = float(np.mean(adj_diffs)) if len(adj_diffs) > 0 else 0.0
    features[f"{feature_prefix}_std_local_jump"] = float(np.std(adj_diffs)) if len(adj_diffs) > 0 else 0.0

    # 3. Macro Structural Spans
    start_val = interpolated[0]
    mid_val = interpolated[num_bins // 2]
    end_val = interpolated[-1]

    features[f"{feature_prefix}_span_start_to_end"] = float(end_val - start_val)
    features[f"{feature_prefix}_abs_span_start_to_end"] = float(abs(end_val - start_val))
    features[f"{feature_prefix}_span_start_to_mid"] = float(mid_val - start_val)
    features[f"{feature_prefix}_abs_span_start_to_mid"] = float(abs(mid_val - start_val))
    features[f"{feature_prefix}_span_mid_to_end"] = float(end_val - mid_val)
    features[f"{feature_prefix}_abs_span_mid_to_end"] = float(abs(end_val - mid_val))

    # 4. Pairwise Distance Matrix Aggregations
    pairwise_dist_matrix = np.abs(interpolated[:, None] - interpolated[None, :])
    tril_indices = np.tril_indices(num_bins, k=-1)
    pairwise_diffs = pairwise_dist_matrix[tril_indices]

    if len(pairwise_diffs) > 0:
        features[f"{feature_prefix}_pairwise_diff_mean"] = float(np.mean(pairwise_diffs))
        features[f"{feature_prefix}_pairwise_diff_max"] = float(np.max(pairwise_diffs))
        features[f"{feature_prefix}_pairwise_diff_std"] = float(np.std(pairwise_diffs))
    else:
        features[f"{feature_prefix}_pairwise_diff_mean"] = 0.0
        features[f"{feature_prefix}_pairwise_diff_max"] = 0.0
        features[f"{feature_prefix}_pairwise_diff_std"] = 0.0

    # 5. Spectral (FFT) Features on Uniform Interpolated Grid
    centered_interp = interpolated - np.mean(interpolated)
    fft_raw = np.fft.rfft(centered_interp)[1:]  # Drop DC offset component
    power_spectrum = np.abs(fft_raw) ** 2
    
    if len(power_spectrum) > 0:
        mid = max(1, len(power_spectrum) // 2)
        low_energy = float(np.sum(power_spectrum[:mid]))
        high_energy = float(np.sum(power_spectrum[mid:]))
        
        features[f"{feature_prefix}_fft_low_energy"] = low_energy
        features[f"{feature_prefix}_fft_high_energy"] = high_energy
        features[f"{feature_prefix}_fft_spectral_ratio"] = float(high_energy / (low_energy + 1e-8))
        
        total_power = float(np.sum(power_spectrum))
        if total_power > 1e-12:
            power_norm = power_spectrum / total_power
            nonzero_p = power_norm[power_norm > 0]
            features[f"{feature_prefix}_fft_spectral_entropy"] = float(-np.sum(nonzero_p * np.log(nonzero_p)))
        else:
            features[f"{feature_prefix}_fft_spectral_entropy"] = 0.0
    else:
        features[f"{feature_prefix}_fft_low_energy"] = 0.0
        features[f"{feature_prefix}_fft_high_energy"] = 0.0
        features[f"{feature_prefix}_fft_spectral_ratio"] = 0.0
        features[f"{feature_prefix}_fft_spectral_entropy"] = 0.0
        
    return features


def _process_single_sentence_group(sid, group, text_map=None, log_base=np.e):
    group = group.sort_values("token_pos")
    length = len(group)

    label = group["label"].iloc[0]
    generator_model = group["generator_model"].iloc[0] if "generator_model" in group.columns else ("Human" if str(label).upper() == "HUMAN" else "LLM")
    is_llm = 0 if (str(label).upper() == "HUMAN" or str(generator_model).upper() == "HUMAN") else 1
    doc_id = group['_id'].iloc[0]

    # Core Arrays
    norm_pos = np.ascontiguousarray(group["norm_pos"].values.astype(np.float64))
    log_rank = np.ascontiguousarray(group["log_rank"].values.astype(np.float64))
    ranks = group["rank"].values.astype(np.float64) if "rank" in group.columns else np.exp(np.clip(log_rank, -100, 100))
    raw_log_prob = np.ascontiguousarray(group["raw_log_prob"].values.astype(np.float64))
    surprisal = np.ascontiguousarray(group["surprisal"].values.astype(np.float64))
    entropy = np.ascontiguousarray(group["entropy"].values.astype(np.float64))

    # Structural Arrays (Fix for np.zeroes typo)
    zipf_alpha = np.ascontiguousarray(group["zipf_alpha"].values.astype(np.float64)) if "zipf_alpha" in group.columns else np.zeros(length, dtype=np.float64)
    mb_beta = np.ascontiguousarray(group["mandelbrot_beta"].values.astype(np.float64)) if "mandelbrot_beta" in group.columns else np.zeros(length, dtype=np.float64)
    gini_coef = np.ascontiguousarray(group["gini_coef"].values.astype(np.float64)) if "gini_coef" in group.columns else np.zeros(length, dtype=np.float64)
    bci = np.ascontiguousarray(group["bci"].values.astype(np.float64)) if "bci" in group.columns else np.zeros(length, dtype=np.float64)
    conc_grad = np.ascontiguousarray(group["concentration_gradient"].values.astype(np.float64)) if "concentration_gradient" in group.columns else np.zeros(length, dtype=np.float64)
    zipf_anomaly = np.ascontiguousarray(group["zipf_anomaly"].values.astype(np.float64)) if "zipf_anomaly" in group.columns else np.zeros(length, dtype=np.float64)
    top5_mass = np.ascontiguousarray(group["top5_mass"].values.astype(np.float64)) if "top5_mass" in group.columns else np.zeros(length, dtype=np.float64)
    top10_mass = np.ascontiguousarray(group["top10_mass"].values.astype(np.float64)) if "top10_mass" in group.columns else np.zeros(length, dtype=np.float64)
    top50_mass = np.ascontiguousarray(group["top50_mass"].values.astype(np.float64)) if "top50_mass" in group.columns else np.zeros(length, dtype=np.float64)
    logit_skew = np.ascontiguousarray(group["logit_skewness"].values.astype(np.float64)) if "logit_skewness" in group.columns else np.zeros(length, dtype=np.float64)
    logit_kurt = np.ascontiguousarray(group["logit_kurtosis"].values.astype(np.float64)) if "logit_kurtosis" in group.columns else np.zeros(length, dtype=np.float64)
    unigram_igr = np.ascontiguousarray(group["unigram_igr"].values.astype(np.float64)) if "unigram_igr" in group.columns else np.zeros(length, dtype=np.float64)
    margins = np.ascontiguousarray(group["top1_top2_margin"].values.astype(np.float64)) if "top1_top2_margin" in group.columns else np.zeros(length, dtype=np.float64)
    surp_acc = np.ascontiguousarray(group["surprisal_acc"].values.astype(np.float64)) if "surprisal_acc" in group.columns else np.zeros(length, dtype=np.float64)

    # 1. ENTROPY & SURPRISAL STATISTICAL MOMENTS
    mean_e = float(np.mean(entropy))
    std_e = float(np.std(entropy, ddof=1)) if length > 1 else 0.0

    entropy_surprisal_diff = np.ascontiguousarray(group["entropy_norm_score"].values.astype(np.float64)) if "entropy_norm_score" in group.columns else (entropy - surprisal)
    mean_entropy_surprisal_diff = float(np.mean(entropy_surprisal_diff))
    std_entropy_surprisal_diff = float(np.std(entropy_surprisal_diff, ddof=1)) if length > 1 else 0.0
    p25_diff = float(np.percentile(entropy_surprisal_diff, 25))
    p75_diff = float(np.percentile(entropy_surprisal_diff, 75))
    iqr_entropy_surprisal_diff = float(p75_diff - p25_diff)

    diff_entropy = np.diff(entropy) if length > 1 else np.array([0.0])
    mean_abs_diff_entropy = float(np.mean(np.abs(diff_entropy))) if length > 1 else 0.0

    if length >= 3:
        std_diff_entropy = float(np.std(diff_entropy, ddof=1))
        volatility_log_rank = float(np.var(np.diff(log_rank), ddof=1))
        volatility_log_prob = float(np.var(np.diff(raw_log_prob), ddof=1))
    else:
        std_diff_entropy, volatility_log_rank, volatility_log_prob = 0.0, 0.0, 0.0

    # Mean Crossing Rate
    centered_entropy = entropy - mean_e
    zero_crossings = np.where(np.diff(centered_entropy >= 0))[0] if length > 1 else np.array([])
    entropy_mean_crossing_rate = float(len(zero_crossings) / (length - 1)) if length > 1 else 0.0

    # 2. LOCAL SURPRISAL SHOCKS (LOO 5-Window, contiguous array safe)
    if length >= 5:
        shape = (length - 5 + 1, 5)
        strides = (surprisal.strides[0], surprisal.strides[0])
        windows = np.lib.stride_tricks.as_strided(surprisal, shape=shape, strides=strides)
        center_tokens = windows[:, 2]
        
        loo_sum = np.sum(windows, axis=1) - center_tokens
        loo_mean = loo_sum / 4.0
        loo_sq_sum = np.sum(windows**2, axis=1) - (center_tokens**2)
        loo_var = np.maximum(0.0, (loo_sq_sum - 4.0 * (loo_mean**2)) / 3.0)
        loo_std = np.sqrt(loo_var)
        
        local_shocks = np.abs(center_tokens - loo_mean) / (loo_std + 1e-8)
        max_local_surprisal_shock = float(np.max(local_shocks))
        mean_local_surprisal_shock = float(np.mean(local_shocks))
    else:
        max_local_surprisal_shock, mean_local_surprisal_shock = 0.0, 0.0

    # 3. MARKOV REGIME ENTROPY & TIME-SERIES DYNAMICS
    if length >= 4:
        states = np.digitize(entropy, bins=[1.0, 3.0])
        trans_matrix = np.zeros((3, 3), dtype=np.float64)
        
        for i_st, j_st in zip(states[:-1], states[1:]):
            trans_matrix[i_st, j_st] += 1.0
            
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        total_transitions = trans_matrix.sum()
        
        if total_transitions > 0:
            pi = (row_sums / total_transitions).squeeze(-1)
            trans_probs = np.zeros_like(trans_matrix)
            np.divide(trans_matrix, row_sums, out=trans_probs, where=row_sums > 0)
            
            row_entropies = np.zeros(3, dtype=np.float64)
            for i in range(3):
                p_row = trans_probs[i]
                nonzero_p = p_row[p_row > 0]
                if len(nonzero_p) > 0:
                    row_entropies[i] = -np.sum(nonzero_p * np.log2(nonzero_p))
            
            markov_regime_entropy = float(np.sum(pi * row_entropies))
        else:
            markov_regime_entropy = 0.0
    else:
        markov_regime_entropy = 0.0

    if length >= 4 and std_e > 1e-8:
        local_stds = [np.std(entropy[i:i+3], ddof=1) for i in range(length - 2)]
        mean_local_entropy_std = float(np.mean(local_stds))
        local_global_std_ratio = float(mean_local_entropy_std / std_e)
    else:
        local_global_std_ratio = 1.0

    if length >= 3 and std_e > 1e-8 and np.std(surprisal, ddof=1) > 1e-8:
        cov_matrix = np.cov(surprisal, entropy, ddof=1)
        surprisal_entropy_cov = float(cov_matrix[0, 1])
        surprisal_entropy_corr = float(np.nan_to_num(np.corrcoef(surprisal, entropy)[0, 1], nan=0.0))
    else:
        surprisal_entropy_cov, surprisal_entropy_corr = 0.0, 0.0

    if length >= 3 and std_e > 1e-8:
        h_centered = entropy - mean_e
        denom = np.sum(h_centered[:-1] ** 2)
        if denom > 1e-8:
            phi1 = np.sum(h_centered[1:] * h_centered[:-1]) / denom
            ar1_residuals = h_centered[1:] - phi1 * h_centered[:-1]
            ar1_residual_var = float(np.var(ar1_residuals, ddof=1)) if len(ar1_residuals) > 1 else 0.0
        else:
            ar1_residual_var = 0.0
    else:
        ar1_residual_var = 0.0

    if std_e > 1e-8 and length >= 4:
        ac = float(np.corrcoef(entropy[:-1], entropy[1:])[0, 1])
        entropy_autocorr = 0.0 if np.isnan(ac) else ac
    else:
        entropy_autocorr = 0.0

    surprisal_var = float(np.var(surprisal, ddof=1)) if length > 1 else 0.0
    surprisal_mean = float(np.mean(surprisal))
    fano_factor = float(np.nan_to_num(surprisal_var / (surprisal_mean + 1e-8), nan=0.0))

    surprisal_skew = float(np.nan_to_num(scipy.stats.skew(surprisal), nan=0.0)) if length >= 3 else 0.0
    surprisal_kurtosis = float(np.nan_to_num(scipy.stats.kurtosis(surprisal), nan=0.0)) if length >= 4 else 0.0
    entropy_skew = float(np.nan_to_num(scipy.stats.skew(entropy), nan=0.0)) if length >= 3 else 0.0

    p25_e = float(np.percentile(entropy, 25))
    p75_e = float(np.percentile(entropy, 75))
    iqr_entropy = p75_e - p25_e
    median_e = float(np.median(entropy))
    iqr_entropy_ratio = float(iqr_entropy / (median_e + 1e-8))

    p25_lp = float(np.percentile(raw_log_prob, 25))
    p75_lp = float(np.percentile(raw_log_prob, 75))

    cdf_vals = np.ascontiguousarray(group["cdf_mass"].values.astype(np.float64)) if "cdf_mass" in group.columns else np.zeros(length)
    mean_cdf = float(np.mean(cdf_vals))
    tail_breach_90 = float(np.mean(cdf_vals > 0.90))
    tail_breach_95 = float(np.mean(cdf_vals > 0.95))

    min_e = np.ascontiguousarray(group["min_entropy"].values.astype(np.float64)) if "min_entropy" in group.columns else np.zeros(length)
    renyi2_e = np.ascontiguousarray(group["renyi2_entropy"].values.astype(np.float64)) if "renyi2_entropy" in group.columns else np.zeros(length)
    min_shannon_ratio = float(np.nan_to_num(np.mean(min_e / (entropy + 1e-8)), nan=0.0))
    renyi2_shannon_ratio = float(np.nan_to_num(np.mean(renyi2_e / (entropy + 1e-8)), nan=0.0))

    entropy_spike_ratio = float(np.mean(entropy > (mean_e + 1.5 * std_e))) if std_e > 1e-8 else 0.0

    terminal_mask = norm_pos >= 0.70
    terminal_entropy_slope = calc_slope(norm_pos[terminal_mask], entropy[terminal_mask]) if np.sum(terminal_mask) >= 2 else 0.0

    p90_margin = float(np.percentile(margins, 90)) if length > 0 else 0.0
    max_rank = float(np.max(ranks)) if length > 0 else 1.0
    bimodal_extreme_index = float(np.nan_to_num((max_rank * p90_margin) / (mean_e + 1e-8), nan=0.0, posinf=0.0))

    entropy_texture_index = float((np.abs(surprisal_kurtosis) * entropy_autocorr) / (std_e + 1e-8))
    surprisal_jitter_index = float(np.mean(np.abs(np.diff(surprisal)))) if length > 1 else 0.0

    head_mask, tail_mask = norm_pos <= 0.25, norm_pos > 0.75
    head_lp = float(np.mean(raw_log_prob[head_mask])) if np.any(head_mask) else float(np.mean(raw_log_prob))
    tail_lp = float(np.mean(raw_log_prob[tail_mask])) if np.any(tail_mask) else float(np.mean(raw_log_prob))

    # 4. EXTRACT TRAJECTORY GRIDS
    zipf_traj_features = extract_array_trajectory_features(norm_pos, zipf_alpha, "zipf")
    gini_traj_features = extract_array_trajectory_features(norm_pos, gini_coef, "gini")
    ent_traj_features = extract_array_trajectory_features(norm_pos, entropy, "ent")
    lp_traj_features = extract_array_trajectory_features(norm_pos, raw_log_prob, "lp")
    mb_beta_traj_features = extract_array_trajectory_features(norm_pos, mb_beta, 'mb_beta')

    # 5. ASSEMBLE FINAL DICTIONARY
    return {
        "sentence_id": sid,
        '_id': doc_id,
        "label": label,                      
        "generator_model": generator_model,  
        "is_llm": is_llm,                    
        "token_length": length,
        
        "mean_log_rank": float(np.mean(log_rank)),
        "std_log_rank": float(np.std(log_rank, ddof=1)) if length > 1 else 0.0,
        "slope_log_rank": calc_slope(norm_pos, log_rank),
        "volatility_log_rank": volatility_log_rank,

        "mean_log_prob": float(np.mean(raw_log_prob)),
        "std_log_prob": float(np.std(raw_log_prob, ddof=1)) if length > 1 else 0.0,
        "slope_log_prob": calc_slope(norm_pos, raw_log_prob),
        "volatility_log_prob": volatility_log_prob,
        "p25_log_prob": p25_lp,
        "p75_log_prob": p75_lp,
        "iqr_log_prob": p75_lp - p25_lp,
        "surprisal_skew": surprisal_skew,
        "surprisal_kurtosis": surprisal_kurtosis,
        "fano_factor_burstiness": fano_factor,

        "mean_entropy": mean_e,
        "std_entropy": std_e,
        "slope_entropy": calc_slope(norm_pos, entropy),
        "entropy_skew": entropy_skew,
        "entropy_spike_ratio": entropy_spike_ratio,
        "entropy_autocorr": entropy_autocorr,
        "iqr_entropy_ratio": iqr_entropy_ratio,

        "mean_gini_coef": float(np.mean(gini_coef)),
        "std_gini_coef": float(np.std(gini_coef, ddof=1)) if length > 1 else 0.0,
        "mean_zipf_alpha": float(np.mean(zipf_alpha)),
        "std_zipf_alpha": float(np.std(zipf_alpha, ddof=1)) if length > 1 else 0.0,
        "slope_zipf_alpha": calc_slope(norm_pos, zipf_alpha),
        "iqr_zipf_alpha": float(np.percentile(zipf_alpha, 75) - np.percentile(zipf_alpha, 25)) if length > 1 else 0.0,

        "mean_mandelbrot_beta": float(np.mean(mb_beta)),
        "std_mandelbrot_beta": float(np.std(mb_beta, ddof=1)) if length > 1 else 0.0,
        "slope_mandelbrot_beta": calc_slope(norm_pos, mb_beta),
        "iqr_mandelbrot_beta": float(np.percentile(mb_beta, 75) - np.percentile(mb_beta, 25)) if length > 1 else 0.0,

        "mean_top5_mass": float(np.mean(top5_mass)),
        "mean_top10_mass": float(np.mean(top10_mass)),
        "mean_top50_mass": float(np.mean(top50_mass)),
        "mean_logit_skewness": float(np.mean(logit_skew)),
        "mean_logit_kurtosis": float(np.mean(logit_kurt)),
        "mean_unigram_igr": float(np.mean(unigram_igr)),

        "ar1_residual_var": ar1_residual_var,
        "terminal_entropy_slope": terminal_entropy_slope,
        "bimodal_extreme_index": bimodal_extreme_index,
        "entropy_texture_index": entropy_texture_index,
        "surprisal_jitter_index": surprisal_jitter_index,

        "mean_entropy_surprisal_diff": mean_entropy_surprisal_diff,
        "std_entropy_surprisal_diff": std_entropy_surprisal_diff,
        "iqr_entropy_surprisal_diff": iqr_entropy_surprisal_diff,

        "std_diff_entropy": std_diff_entropy,
        "mean_abs_diff_entropy": mean_abs_diff_entropy,
        "entropy_mean_crossing_rate": entropy_mean_crossing_rate,
        "local_global_std_ratio": local_global_std_ratio,
        "surprisal_entropy_cov": surprisal_entropy_cov,
        "surprisal_entropy_corr": surprisal_entropy_corr,
        "markov_regime_entropy": markov_regime_entropy,

        "min_shannon_ratio": min_shannon_ratio,
        "renyi2_shannon_ratio": renyi2_shannon_ratio,
        "mean_cdf_mass": mean_cdf,
        "tail_breach_ratio_90": tail_breach_90,
        "tail_breach_ratio_95": tail_breach_95,
        "rank_eff_ratio": float(np.mean(group["rank_eff_ratio"].values.astype(np.float64))) if "rank_eff_ratio" in group.columns else 0.0,
        "mean_logit_std": float(np.mean(group["logit_std"].values.astype(np.float64))) if "logit_std" in group.columns else 0.0,

        "diff_head_tail_log_prob": head_lp - tail_lp,
        
        "mean_top1_top2_margin": float(np.mean(margins)) if length > 0 else 0.0,
        "std_top1_top2_margin": float(np.std(margins, ddof=1)) if length > 1 else 0.0,
        "mean_surprisal_acc": float(np.mean(surp_acc)) if length > 0 else 0.0,
        "std_surprisal_acc": float(np.std(surp_acc, ddof=1)) if length > 1 else 0.0,
        "mean_bci": float(np.mean(bci)),
        "mean_concentration_gradient": float(np.mean(conc_grad)),
        "mean_zipf_anomaly": float(np.mean(zipf_anomaly)),
        "max_local_surprisal_shock": max_local_surprisal_shock,
        "mean_local_surprisal_shock": mean_local_surprisal_shock,

        **zipf_traj_features,
        **gini_traj_features,
        **ent_traj_features,
        **lp_traj_features,
        **mb_beta_traj_features
    }


def aggregate_sentence_features(token_df, text_map=None, n_jobs=-1):
    if token_df is None or token_df.empty or "sentence_id" not in token_df.columns:
        print("[WARNING] token_df is empty or missing 'sentence_id'. Returning empty DataFrame.")
        return pd.DataFrame()

    groups = [group for _, group in token_df.groupby("sentence_id")]
    
    # Wrap groups generator with tqdm
    records = Parallel(n_jobs=n_jobs, batch_size=100, max_nbytes=None)(
        delayed(_process_single_sentence_group)(
            group["sentence_id"].iloc[0], group, text_map
        ) 
        for group in tqdm(groups, desc="Aggregating sentence features")
    )
    return pd.DataFrame(records)


# ==========================================
# 3. LOAD DATASETS
# ==========================================
def load_clin33_dutch_llm_sentences(csv_path, max_samples=100, seed=42):
    print(f"Loading new Dutch LLM dataset from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find new LLM CSV file at: {csv_path}")
        
    df_raw = pd.read_csv(csv_path)
    llm_candidates = []
    
    for idx, row in df_raw.iterrows():
        text = str(row["generated_text"])
        genre = row.get("genre", "Unknown")
        doc_id = f"clin33_{idx}"
        
        sents = nltk.sent_tokenize(text, language='dutch')
        for s in sents:
            if len(s.split()) >= 10:
                llm_candidates.append({
                    "text": s,
                    "doc_id": doc_id,
                    "is_llm": 1,
                    "generator_model": "CLIN33_Dutch_ChatGPT",
                    "genre": genre
                })

    rng = np.random.default_rng(seed)
    rng.shuffle(llm_candidates)
    selected_llm = llm_candidates[:max_samples]

    res_df = pd.DataFrame(selected_llm)
    print(f"Extracted {len(res_df)} random Dutch LLM sentences (Min length: 10 words)")
    return res_df


def load_hc3_sentences(max_samples_per_class=100, seed=42):
    print("Loading Hello-SimpleAI/HC3 dataset...")
    hc3 = load_dataset(
        "json", 
        data_files="hf://datasets/Hello-SimpleAI/HC3/all.jsonl", 
        split="train"
    ).shuffle(seed=seed)
    
    human_candidates = []
    llm_candidates = []
    
    for idx, item in enumerate(hc3):
        doc_id = item.get("id", item.get("idx", f"hc3_{idx}"))
        
        for ans in item.get("human_answers", []):
            sents = nltk.sent_tokenize(ans)
            for s in sents:
                if len(s.split()) >= 10:
                    human_candidates.append({
                        "text": s,
                        "doc_id": doc_id,
                        "is_llm": 0,
                        "generator_model": "Human"
                    })

        for ans in item.get("chatgpt_answers", []):
            sents = nltk.sent_tokenize(ans)
            for s in sents:
                if len(s.split()) >= 10:
                    llm_candidates.append({
                        "text": s,
                        "doc_id": doc_id,
                        "is_llm": 1,
                        "generator_model": "ChatGPT"
                    })

    rng = np.random.default_rng(seed)
    rng.shuffle(human_candidates)
    rng.shuffle(llm_candidates)

    selected_human = human_candidates[:max_samples_per_class]
    selected_llm = llm_candidates[:max_samples_per_class]

    final_records = selected_human + selected_llm
    rng.shuffle(final_records)

    df = pd.DataFrame(final_records)
    print(f"Extracted {len(df)} random sentences ({sum(df['is_llm']==0)} Human, {sum(df['is_llm']==1)} ChatGPT) [Min length: 10 words]")
    return df


def is_valid_sentence(s, min_words=10):
    if s is None or pd.isna(s):
        return False
    s_str = str(s).strip()
    invalid_flags = {'FAILED_GENERATION', 'FAILED_VALIDATION', 'NONE', 'NAN', 'NA', 'NULL', '<NA>', ''}
    if s_str.upper() in invalid_flags:
        return False
    if len(s_str.split()) < min_words:
        return False
    return True


def parse_sentence_list(val):
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    if isinstance(val, str):
        val_str = val.strip()
        if val_str.startswith('[') and val_str.endswith(']'):
            try:
                parsed = ast.literal_eval(val_str)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return [val_str]
    return [val] if val is not None else []


def load_parquet_human_sentences(parquet_path=PARQUET_HUMAN_PATH, max_samples=100, min_words=10, seed=42):
    print(f"Loading Human sentences from Parquet: {parquet_path}")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Could not find Parquet file at: {parquet_path}")

    df_parquet = pd.read_parquet(parquet_path)
    human_candidates = []

    for idx, row in df_parquet.iterrows():
        doc_id = row['_id'] if '_id' in row else (row['id'] if 'id' in row else f'doc_{idx}')
        
        h_sents = parse_sentence_list(row.get("abstract_sentence"))
        valid_h = [str(s).strip() for s in h_sents if is_valid_sentence(s, min_words=min_words)]

        for s in valid_h:
            human_candidates.append({
                "text": s,
                "doc_id": doc_id,
                "is_llm": 0,
                "generator_model": "Human"
            })

    rng = np.random.default_rng(seed)
    rng.shuffle(human_candidates)
    selected_human = human_candidates[:max_samples]

    print(f"Extracted {len(selected_human)} random Human sentences from Parquet (Min length: {min_words} words)")
    return selected_human


def get_clean_sentence_list(val):
    """Parses text/lists cleanly without breaking 1:1 sentence list alignment."""
    parsed = parse_sentence_list(val)
    if len(parsed) > 1:
        # Already a pre-split list of sentences (1:1 aligned across models)
        return [str(x).strip() for x in parsed]
    elif len(parsed) == 1:
        # Single string block; tokenize into sentences
        s_text = str(parsed[0]).strip()
        if not s_text or s_text.upper() in {'FAILED_GENERATION', 'FAILED_VALIDATION', 'NONE', 'NAN', 'NULL'}:
            return []
        return [str(s).strip() for s in nltk.sent_tokenize(s_text)]
    return []


def load_abstracts_dataset(
    parquet_path=PARQUET_HUMAN_PATH, 
    llm_columns=None, 
    max_samples_per_class=200, 
    min_words=8,  # Slightly relaxed from 10 to 8 to increase valid matches
    seed=42
):
    print(f"Loading Human and LLM sentences from Parquet: {parquet_path}")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Could not find Parquet file at: {parquet_path}")

    df_parquet = pd.read_parquet(parquet_path)

    # All target _full columns to enforce simultaneous presence
    all_full_cols = [c for c in df_parquet.columns if c.endswith("_full")]

    if llm_columns is None:
        llm_columns = all_full_cols
    elif isinstance(llm_columns, str):
        llm_columns = [llm_columns]

    doc_parsed = []
    valid_pairs = []  # Stores (row_idx, doc_id, sentence_idx)

    for idx, row in tqdm(df_parquet.iterrows(), total=len(df_parquet), desc="Searching aligned (doc_id, sentence_idx) pairs"):
        doc_id = row['_id'] if '_id' in row else (row['id'] if 'id' in row else f'doc_{idx}')
        
        # Parse Human sentences
        h_sents = get_clean_sentence_list(row.get("abstract_sentence"))
        
        # Parse all _full LLM columns for this document
        col_sents_map = {}
        for col in all_full_cols:
            col_sents_map[col] = get_clean_sentence_list(row.get(col))

        doc_parsed.append({
            "doc_id": doc_id,
            "h_sents": h_sents,
            "col_sents": col_sents_map
        })

        # Find sentence indices valid in Human AND ALL _full columns at the exact same position
        for s_idx in range(len(h_sents)):
            if not is_valid_sentence(h_sents[s_idx], min_words=min_words):
                continue

            all_valid = True
            for col in all_full_cols:
                col_list = col_sents_map[col]
                if s_idx >= len(col_list) or not is_valid_sentence(col_list[s_idx], min_words=min_words):
                    all_valid = False
                    break

            if all_valid:
                valid_pairs.append((idx, doc_id, s_idx))

    # Step 2: Randomly sample max_samples_per_class pairs
    rng = np.random.default_rng(seed)
    if len(valid_pairs) > max_samples_per_class:
        chosen_indices = rng.choice(len(valid_pairs), size=max_samples_per_class, replace=False)
        selected_pairs = [valid_pairs[i] for i in chosen_indices]
    else:
        selected_pairs = valid_pairs

    print(f"Selected {len(selected_pairs)} common (doc_id, sentence_idx) pairs across Human and ALL {len(all_full_cols)} '_full' columns.")

    # Step 3: Extract Human and specified LLM sentences for chosen pairs
    human_candidates = []
    llm_candidates = []

    for row_idx, doc_id, s_idx in selected_pairs:
        doc_info = doc_parsed[row_idx]

        human_candidates.append({
            "text": str(doc_info["h_sents"][s_idx]).strip(),
            "doc_id": doc_id,
            "is_llm": 0,
            "generator_model": "Human"
        })

        for col in llm_columns:
            llm_candidates.append({
                "text": str(doc_info["col_sents"][col][s_idx]).strip(),
                "doc_id": doc_id,
                "is_llm": 1,
                "generator_model": col
            })

    final_records = human_candidates + llm_candidates
    rng.shuffle(final_records)

    df = pd.DataFrame(final_records)
    print(f"Extracted {len(df)} total sentences ({len(human_candidates)} Human, {len(llm_candidates)} LLM) [Min length: {min_words} words]")
    return df

# ==========================================
# 4. VISUALIZATION & REPORT PIPELINE
# ==========================================
def generate_significance_report(model_results_dict, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "significance_report.md")
    
    lines = ["# Feature Statistical Significance Report\n\n"]
    sig_sets = {}
    
    for model_name, sig_df in model_results_dict.items():
        lines.append(f"## Model: {model_name}\n")
        if sig_df.empty:
            lines.append("No results available.\n\n")
            continue
            
        desired_cols = ["feature", "human_mean_std", "llm_mean_std", "p_location (MW-U FDR)", "cohens_d", "roc_auc"]
        available_cols = [c for c in desired_cols if c in sig_df.columns]
        
        if not available_cols:
            available_cols = [c for c in sig_df.columns if not c.startswith("_")][:6]
            
        lines.append(f"*(Total significant features: {len(sig_df)})*\n\n")
        lines.append(sig_df[available_cols].to_markdown(index=False))
        lines.append("\n\n")
        
        if "_raw_p_mw" in sig_df.columns:
            sig_feats = sig_df[sig_df["_raw_p_mw"] < 0.05]["feature"].tolist()
            sig_sets[model_name] = set(sig_feats)

    if sig_sets:
        common = set.intersection(*sig_sets.values())
        lines.append(f"## Shared Significant Features Across All Models (p < 0.05)\n")
        lines.append(f"Total Count: **{len(common)}**\n\n")
        for feat in sorted(common):
            lines.append(f"- `{feat}`")
        lines.append("\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Saved significance report to: '{report_path}'")


def plot_all_models_heatmap(model_results_dict, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "all_models_feature_heatmap.png")

    combined_auc = {}
    for model_name, res_df in model_results_dict.items():
        if res_df.empty:
            continue
        auc_series = res_df.set_index("feature")["_raw_auc"]
        combined_auc[model_name] = auc_series

    if not combined_auc:
        return

    auc_matrix = pd.DataFrame(combined_auc)
    auc_matrix["_predictive_dist"] = (auc_matrix - 0.5).abs().mean(axis=1)
    auc_matrix = auc_matrix.sort_values(by="_predictive_dist", ascending=False).drop(columns=["_predictive_dist"])

    plt.figure(figsize=(10, max(12, len(auc_matrix) * 0.25)))
    sns.heatmap(
        auc_matrix, 
        annot=True, 
        fmt=".3f", 
        cmap="vlag", 
        vmin=0.0, 
        vmax=1.0, 
        center=0.5, 
        cbar_kws={'label': 'ROC-AUC (0.5 = Chance, 0.0 / 1.0 = Strong Signal)'}
    )
    plt.title("Feature Predictive Power (ROC-AUC) Across Models\n(Sorted by Distance from Chance |AUC - 0.5|)")
    plt.xlabel("Model Scale")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved all models heatmap to: '{save_path}'")


def plot_best_metric_per_model(model_results_dict, sentence_dfs, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    valid_models = [m for m in model_results_dict.keys() if m in sentence_dfs and not model_results_dict[m].empty]
    if not valid_models:
        return

    fig, axes = plt.subplots(1, len(valid_models), figsize=(6 * len(valid_models), 5), sharey=False)
    
    for idx, model_name in enumerate(valid_models):
        ax = axes[idx] if len(valid_models) > 1 else axes
        sig_df = model_results_dict[model_name]
        sent_df = sentence_dfs[model_name].copy()
        
        top_feat = sig_df.iloc[0]["feature"]
        top_auc = sig_df.iloc[0]["_raw_auc"]
        
        sent_df["is_llm"] = pd.to_numeric(sent_df["is_llm"], errors='coerce').fillna(0).astype(int)
        
        sns.violinplot(
            data=sent_df,
            x="is_llm",
            y=top_feat,
            hue="is_llm",
            palette={0: "#2b5c8f", 1: "#d95f02"},
            legend=False,
            ax=ax,
            inner="quartile"
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Human", "LLM"])
        ax.set_title(f"{model_name}\nBest: '{top_feat}' (AUC: {top_auc:.3f})")
        ax.set_xlabel("Source Text")
        ax.set_ylabel(top_feat)

    plt.suptitle("Best Performing Metric Per Model", fontsize=14, y=1.03)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "best_metrics_per_model.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved best metric per model plot to: '{save_path}'")


def visualize_shared_feature_separation(selected_feature, sentence_dfs, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    valid_models = list(sentence_dfs.keys())
    if not valid_models:
        return

    fig, axes = plt.subplots(1, len(valid_models), figsize=(6 * len(valid_models), 5), sharey=True)
    
    for idx, model_name in enumerate(valid_models):
        ax = axes[idx] if len(valid_models) > 1 else axes
        sent_df = sentence_dfs[model_name].copy()
        sent_df["is_llm"] = pd.to_numeric(sent_df["is_llm"], errors='coerce').fillna(0).astype(int)
        
        sns.violinplot(
            data=sent_df,
            x="is_llm",
            y=selected_feature,
            hue="is_llm",
            palette={0: "#2b5c8f", 1: "#d95f02"},
            legend=False,
            ax=ax,
            inner="quartile"
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Human", "LLM"])
        ax.set_title(f"Model: {model_name}")
        ax.set_xlabel("Source Text")
        if idx == 0:
            ax.set_ylabel(f"Feature Value ({selected_feature})")
        else:
            ax.set_ylabel("")

    plt.suptitle(f"Shared Significant Feature: '{selected_feature}' Across Models", fontsize=14, y=1.03)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"shared_feature_separation_{selected_feature}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved shared feature plot to: '{save_path}'")


def plot_qwen_model_scale_comparison(model_results_dict, output_dir="."):
    """
    Plots predictive power progression across Qwen model sizes (0.5B -> 3B -> 7B) 
    for a specific dataset/run configuration folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "qwen_model_scale_performance.png")

    qwen_models = [m for m in model_results_dict.keys() if "qwen" in m.lower()]
    if not qwen_models:
        return

    scale_metrics = []
    for model_name in sorted(qwen_models):
        res_df = model_results_dict[model_name]
        if res_df.empty or "_raw_auc" not in res_df.columns:
            continue
            
        # Calculate maximum predictive AUC (distance from random chance 0.5)
        predictive_aucs = 0.5 + np.abs(res_df["_raw_auc"].values - 0.5)
        top1_auc = np.max(predictive_aucs)
        top5_mean_auc = np.mean(np.sort(predictive_aucs)[::-1][:5])

        scale_metrics.append({
            "model": model_name,
            "Top-1 Feature AUC": top1_auc,
            "Top-5 Mean AUC": top5_mean_auc
        })

    if not scale_metrics:
        return

    scale_df = pd.DataFrame(scale_metrics)

    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = np.arange(len(scale_df))
    width = 0.35

    rects1 = ax.bar(x_positions - width/2, scale_df["Top-1 Feature AUC"], width, label='Top-1 Max Feature AUC', color='#2b5c8f')
    rects2 = ax.bar(x_positions + width/2, scale_df["Top-5 Mean AUC"], width, label='Top-5 Mean Feature AUC', color='#4daf4a')

    ax.set_ylabel('Predictive Power (ROC-AUC)')
    ax.set_title('Feature Predictive Power vs Qwen Evaluator Model Size')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scale_df["model"], rotation=15)
    ax.set_ylim(0.45, 1.0)
    ax.axhline(0.5, color='gray', linestyle='--', label='Random Chance (0.5)')
    ax.legend(loc='lower right')

    # Value Labels on bars
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Qwen model scale comparison plot to: '{save_path}'")


def run_visualization_and_report_pipeline(model_significance_results, sentence_dfs, output_dir="."):
    """Master trigger that saves all plots and reports inside output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Automatic Significance Report
    generate_significance_report(model_significance_results, output_dir)
    
    # 2. All Models Heatmap Plot
    plot_all_models_heatmap(model_significance_results, output_dir)
    
    # 3. Best Performing Metric Per Model Plot
    plot_best_metric_per_model(model_significance_results, sentence_dfs, output_dir)
    
    # 4. Qwen Scale Comparison Plot for this run
    plot_qwen_model_scale_comparison(model_significance_results, output_dir)
    
    # 5. Shared Significant Feature Plot
    significant_features_by_model = {}
    for model_name, sig_df in model_significance_results.items():
        if "_raw_p_mw" in sig_df.columns:
            sig_feats = sig_df[sig_df["_raw_p_mw"] < 0.05]["feature"].tolist()
        else:
            sig_feats = []
        significant_features_by_model[model_name] = set(sig_feats)

    common_sig_features = set.intersection(*significant_features_by_model.values()) if significant_features_by_model else set()
    
    if common_sig_features and sentence_dfs:
        auc_dists = {}
        for feat in common_sig_features:
            aucs = [
                model_significance_results[m].set_index("feature").loc[feat, "_raw_auc"]
                for m in model_significance_results.keys()
                if feat in model_significance_results[m].set_index("feature").index
            ]
            if aucs:
                auc_dists[feat] = np.mean([abs(a - 0.5) for a in aucs])

        if auc_dists:
            best_common_feat = max(auc_dists, key=auc_dists.get)
            print(f"Selected shared significant feature: '{best_common_feat}' (Mean |AUC - 0.5| = {auc_dists[best_common_feat]:.3f})")
            visualize_shared_feature_separation(best_common_feat, sentence_dfs, output_dir)


# ==========================================
# 5. AGGREGATE CROSS-COLUMN COMPARISON PLOT
# ==========================================
def plot_cross_llm_col_qwen_performance(base_dir="."):
    """
    Scans all completed 'abstracts_qwen_*' run folders and plots a comprehensive summary graph
    comparing the predictive performance of different Qwen model sizes across all target llm_cols.
    """
    print("\n==================================================")
    print(" GENERATING CROSS-COLUMN SUMMARY PLOT FOR QWEN")
    print("==================================================")
    
    # Find all abstracts_qwen directory folders
    search_pattern = os.path.join(base_dir, "abstracts_qwen_*")
    matched_dirs = [d for d in glob.glob(search_pattern) if os.path.isdir(d)]
    
    if not matched_dirs:
        print("[INFO] No 'abstracts_qwen_*' directories found for summary plot.")
        return

    records = []
    
    for run_dir in sorted(matched_dirs):
        dir_name = os.path.basename(run_dir)
        # Parse column name from folder name format: abstracts_qwen_{clean_llm_col}_{num_samples}
        llm_col_identifier = dir_name.replace("abstracts_qwen_", "")
        
        # Remove trailing sample count if present (e.g., '_100')
        if "_" in llm_col_identifier and llm_col_identifier.split("_")[-1].isdigit():
            llm_col_identifier = "_".join(llm_col_identifier.split("_")[:-1])

        # Scan significance results for Qwen models
        sig_files = glob.glob(os.path.join(run_dir, "*qwen*significance_results.csv"))
        
        for sig_file in sig_files:
            file_name = os.path.basename(sig_file)
            
            # Extract Qwen model scale identifier
            qwen_scale = "Unknown"
            if "0.5b" in file_name.lower():
                qwen_scale = "Qwen 0.5B"
            elif "3b" in file_name.lower():
                qwen_scale = "Qwen 3B"
            elif "7b" in file_name.lower():
                qwen_scale = "Qwen 7B"

            df = pd.read_csv(sig_file)
            if df.empty or "_raw_auc" not in df.columns:
                continue
                
            predictive_aucs = 0.5 + np.abs(df["_raw_auc"].values - 0.5)
            top1_auc = np.max(predictive_aucs)
            top5_mean_auc = np.mean(np.sort(predictive_aucs)[::-1][:5])

            records.append({
                "Target LLM Column": llm_col_identifier,
                "Evaluator Model": qwen_scale,
                "Top-1 Feature AUC": top1_auc,
                "Top-5 Mean AUC": top5_mean_auc
            })

    if not records:
        print("[WARNING] No statistical significance results found inside abstracts_qwen folders.")
        return

    summary_df = pd.DataFrame(records)
    
    # Plotting Grouped Bar Chart
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=summary_df,
        x="Target LLM Column",
        y="Top-1 Feature AUC",
        hue="Evaluator Model",
        palette="Blues_d",
        edgecolor="black"
    )

    plt.title("Feature Predictive Power (Top-1 ROC-AUC) Across Target LLM Columns & Qwen Model Sizes", fontsize=13)
    plt.xlabel("Target LLM Column", fontsize=11)
    plt.ylabel("Top-1 Feature ROC-AUC", fontsize=11)
    plt.ylim(0.45, 1.0)
    plt.axhline(0.5, color='red', linestyle='--', label='Random Chance (0.5)')
    plt.xticks(rotation=20, ha='right')
    plt.legend(title="Qwen Model Scale", loc="lower right")

    # Annotate bars with AUC values
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, color='black',
                        xytext=(0, 2), textcoords='offset points')

    plt.tight_layout()
    summary_plot_path = os.path.join(base_dir, "qwen_model_sizes_vs_llm_cols_performance.png")
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()
    
    # Save CSV Summary
    summary_csv_path = os.path.join(base_dir, "qwen_model_sizes_vs_llm_cols_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    
    print(f"Saved cross-experiment comparison plot to: '{summary_plot_path}'")
    print(f"Saved cross-experiment CSV summary to: '{summary_csv_path}'\n")


def run_visualization_only(target_dir):
    target_dir = os.path.abspath(target_dir)
    print(f"\n[VISUALIZATION MODE] Scanning CSVs in: '{target_dir}'")
    
    model_significance_results = {}
    sentence_level_dfs = {}

    for model_name in MODELS_CONFIG.keys():
        sig_files = glob.glob(os.path.join(target_dir, f"*{model_name}*significance_results*.csv"))
        feat_files = glob.glob(os.path.join(target_dir, f"*{model_name}*sentence_features*.csv"))

        if sig_files:
            sig_file = sig_files[0]
            print(f"  Loaded significance results for {model_name}: {os.path.basename(sig_file)}")
            model_significance_results[model_name] = pd.read_csv(sig_file)

        if feat_files:
            feat_file = feat_files[0]
            print(f"  Loaded sentence features for {model_name}: {os.path.basename(feat_file)}")
            sentence_level_dfs[model_name] = pd.read_csv(feat_file)

    if model_significance_results:
        run_visualization_and_report_pipeline(model_significance_results, sentence_level_dfs, target_dir)
        
    # Generate overall summary plot across all abstracts_qwen_* folders in root
    plot_cross_llm_col_qwen_performance(".")


# ==========================================
# 6. MAIN EXECUTION ROUTINE
# ==========================================
def run_pipeline_for_config(run_dir_name, config, num_samples=100):
    print(f"\n==================================================")
    print(f" STARTING RUN SETUP: '{run_dir_name}'")
    print(f" Config: {config} | Samples per class: {num_samples}")
    print(f"==================================================")

    data_setting = config.get("data")
    model_setting = config.get("model")
    llm_columns = config.get("llm_columns", [])

    output_dir = run_dir_name
    os.makedirs(output_dir, exist_ok=True)

    if data_setting == "HC3":
        print("\n[DATASET] Loading HC3 Dataset (Human vs. ChatGPT)...")
        data_df = load_hc3_sentences(max_samples_per_class=num_samples, seed=42)
    elif data_setting == "clin33":
        print("\n[DATASET] Loading Parquet Human text + CLIN33 Dutch LLM Dataset...")
        human_samples = load_parquet_human_sentences(
            PARQUET_HUMAN_PATH, max_samples=num_samples, min_words=10, seed=42
        )
        llm_samples_df = load_clin33_dutch_llm_sentences(
            CLIN33_CSV_PATH, max_samples=num_samples, seed=42
        )

        combined_samples = human_samples + llm_samples_df.to_dict("records")
        rng = np.random.default_rng(42)
        rng.shuffle(combined_samples)
        data_df = pd.DataFrame(combined_samples)
    elif data_setting == "abstracts":
        print(f"\n[DATASET] Loading Parquet Human and LLM text from abstracts (LLM Columns: {llm_columns})...")
        data_df = load_abstracts_dataset(
            parquet_path=PARQUET_HUMAN_PATH,
            llm_columns=llm_columns,
            max_samples_per_class=num_samples,
            min_words=10,
            seed=42
        )
    else:
        raise ValueError(f"Unknown data setting: '{data_setting}' in configuration '{run_dir_name}'")

    model_significance_results = {}
    sentence_level_dfs = {}

    for model_name, model_kwargs in MODELS_CONFIG.items():
        if model_setting not in model_name:
            continue

        print(f"\n------------------------------------------")
        print(f"Processing Model: {model_kwargs['description']} [Dataset: {data_setting}]")
        print(f"------------------------------------------")

        model_path = hf_hub_download(
            repo_id=model_kwargs["repo_id"], filename=model_kwargs["filename"]
        )
        llm = Llama(model_path=model_path, **model_kwargs["llama_kwargs"])

        token_records = []
        sentence_id_counter = 0

        for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
            text = row["text"]
            doc_id = row["doc_id"]
            model_src = row["generator_model"]
            lbl_prefix = "H" if row["is_llm"] == 0 else "L"

            records, sentence_id_counter = extract_logit_trajectory(
                text=text,
                doc_id=doc_id,
                label_prefix=lbl_prefix,
                sentence_id=sentence_id_counter,
                llm=llm,
                model_source=model_src,
                max_tokens=2048,
            )
            token_records.extend(records)

        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        token_df = pd.DataFrame(token_records)
        token_csv_path = os.path.join(
            output_dir, f"{model_name}_{data_setting}_token_trajectories.csv"
        )
        token_df.to_csv(token_csv_path, index=False)
        print(f"Saved token trajectories to: '{token_csv_path}'")

        sent_df = aggregate_sentence_features(token_df, n_jobs=-1)
        sentence_level_dfs[model_name] = sent_df

        sent_csv_path = os.path.join(
            output_dir, f"{model_name}_{data_setting}_sentence_features.csv"
        )
        sent_df.to_csv(sent_csv_path, index=False)
        print(f"Saved sentence features to: '{sent_csv_path}'")

        sig_df = calculate_significance(sent_df)
        model_significance_results[model_name] = sig_df

        sig_csv_path = os.path.join(
            output_dir, f"{model_name}_{data_setting}_significance_results.csv"
        )
        sig_df.to_csv(sig_csv_path, index=False)
        print(f"Saved significance summary to: '{sig_csv_path}'")

        run_visualization_and_report_pipeline(
            model_significance_results, sentence_level_dfs, output_dir
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate & Visualize LLM vs Human Sentence Features"
    )
    parser.add_argument(
        "--visualize",
        type=str,
        default=None,
        metavar="DIR_PATH",
        help="Path to directory containing pre-calculated CSV files. If provided, skips GPU inference and generates plots directly.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        choices=list(RUN_CONFIGS.keys()),
        help="Optional: Run only a single specific configuration from . If omitted, runs ALL sequentially.",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Force rerun of configurations even if they have already been completed.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        metavar="N",
        help="Number of samples per class to extract (default: 100).",
    )

    args = parser.parse_args()

    if args.visualize is not None:
        target_dir = args.visualize if args.visualize != "" else "."
        run_visualization_only(target_dir)
        return

    if args.run:
        configs_to_run = {args.run: RUN_CONFIGS[args.run]}
    else:
        configs_to_run = RUN_CONFIGS

    for run_name, config in configs_to_run.items():
        if args.samples is not None:
            num_samples = args.samples
        else:
            num_samples = config.get("samples", 100)

        run_dir_name = f"{run_name}_{num_samples}"

        if not args.redo and is_config_completed(run_dir_name, config):
            print(f"\n[SKIP] Configuration '{run_dir_name}' is already completed. Pass --redo to re-run.")
            continue

        run_pipeline_for_config(run_dir_name, config, num_samples=num_samples)

    # After running configs, generate overall cross-column comparative plot
    plot_cross_llm_col_qwen_performance(".")


if __name__ == "__main__":
    main()