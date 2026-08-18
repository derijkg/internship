import os
import gc
import argparse
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.special
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import nltk
import glob
import ast

from huggingface_hub import hf_hub_download
from datasets import load_dataset
from llama_cpp import Llama

# Download NLTK tokenizer data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Set random seed
np.random.seed(42)

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
    
    ignore_cols = ["sentence_id", "_id", "doc_id", "label", "generator_model", "is_llm", "text", "genre","token_length"]
    feature_cols = [col for col in sent_df.columns if col not in ignore_cols]
    results = []
    
    for feat in feature_cols:
        h_raw = pd.to_numeric(human_df[feat], errors='coerce').values
        l_raw = pd.to_numeric(llm_df[feat], errors='coerce').values

        # Complete Case Analysis: Filter out NaNs and Inf values properly
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
                
                # True signed ROC-AUC (0.0 to 1.0)
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
        # Benjamini-Hochberg False Discovery Rate (FDR) adjustment
        res_df["p_mw_fdr"] = stats.false_discovery_control(res_df["_raw_p_mw"].values)
        res_df["p_location (MW-U FDR)"] = res_df["p_mw_fdr"].apply(format_p_value)
        res_df["p_variance (Levene)"] = res_df["_raw_p_lev"].apply(format_p_value)
        
        # Sort by predictive distance from chance (|AUC - 0.5|)
        res_df["_auc_dist"] = np.abs(res_df["_raw_auc"] - 0.5)
        res_df = res_df.sort_values(by="_auc_dist", ascending=False).drop(columns=["_auc_dist"]).reset_index(drop=True)

        # Filter to keep ONLY statistically significant features (FDR-adjusted p < 0.05)
        # (Note: Use '_raw_p_mw' instead of 'p_mw_fdr' if you prefer raw p-values)
        res_df = res_df[res_df["p_mw_fdr"] < 0.05].reset_index(drop=True)
        
    return res_df



# ==========================================
# 2. FEATURE EXTRACTION PIPELINE
# ==========================================
def compute_vectorized_gini(probs):
    """
    Computes standard Gini coefficient for probability distributions.
    Supports both 1D and 2D arrays.
    """
    probs = np.atleast_2d(probs)
    M, V = probs.shape
    sorted_probs = np.sort(probs, axis=-1)
    index = np.arange(1, V + 1, dtype=np.float32)
    weights = (V - index + 0.5) / V
    gini = 1.0 - 2.0 * np.sum(sorted_probs * weights, axis=-1)
    return gini.squeeze() if M == 1 else gini


def compute_zipf_exponent(v_logits, top_k=20):
    """
    Computes Zipf's exponent alpha by regressing top-k logits (log-probabilities
    up to an additive constant) against log-ranks.
    """
    v_logits = np.atleast_2d(v_logits)
    M, V = v_logits.shape
    actual_k = min(top_k, V)
    
    # Select top-k logits per token
    topk_logits = np.partition(v_logits, -actual_k, axis=-1)[:, -actual_k:]
    sorted_topk = np.sort(topk_logits, axis=-1)[:, ::-1]  # z_1 >= z_2 >= ... >= z_k
    
    log_ranks = np.log(np.arange(1, actual_k + 1, dtype=np.float32))
    
    mean_x = np.mean(log_ranks)
    var_x = np.var(log_ranks)
    mean_y = np.mean(sorted_topk, axis=-1, keepdims=True)
    
    # Linear regression slope: z_r = C - alpha * ln(r)
    cov_xy = np.mean((log_ranks - mean_x) * (sorted_topk - mean_y), axis=-1)
    zipf_alpha = -cov_xy / (var_x + 1e-8)
    return zipf_alpha


def extract_trajectory_llama_cpp(
    text, 
    doc_id, 
    label_prefix, 
    sentence_id, 
    llm, 
    model_source="LLM", 
    max_tokens=2048,
    unigram_log_probs=None  # Added missing parameter to avoid NameError
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

    special_ids = {tid for tid in (bos_id, eos_id) if tid is not None and tid != -1}
    n_vocab = llm.n_vocab()

    valid_mask = np.array([
        (tok not in special_ids) and (tok < n_vocab)
        for tok in shift_labels
    ], dtype=bool)

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

    # Top-1 and Top-2 probabilities with safe partition
    top_k_partition = min(2, n_vocab)
    top2_logits = np.partition(v_logits, -top_k_partition, axis=-1)[:, -top_k_partition:]
    p_top1 = np.exp(top2_logits[:, -1] - lse.squeeze(-1))
    p_top2 = np.exp(top2_logits[:, -2] - lse.squeeze(-1)) if top_k_partition >= 2 else np.zeros_like(p_top1)
    
    min_entropies = -np.log(p_top1 + 1e-12)
    renyi2_entropies = -np.log(np.sum(probs ** 2, axis=-1) + 1e-12)

    target_logits = v_logits[np.arange(total_valid_tokens), v_labels, None]
    ranks = np.sum(v_logits > target_logits, axis=-1) + 1
    cdf_mass = np.sum(np.where(v_logits >= target_logits, probs, 0.0), axis=-1)

    eff_vocab = np.exp(entropies)
    rank_eff_ratio = ranks / (eff_vocab + 1e-8)
    logit_std = np.std(v_logits, axis=-1)
    top1_top2_margins = p_top1 - p_top2

    gini_coefs = compute_vectorized_gini(probs)
    zipf_alphas = compute_zipf_exponent(v_logits, top_k=20)
    
    # Safe top-50 partitioning for variable vocabulary size
    k50 = min(50, n_vocab)
    top50_p = np.partition(probs, -k50, axis=-1)[:, -k50:]
    sorted_top50_p = np.sort(top50_p, axis=-1)
    
    k5 = min(5, n_vocab)
    k10 = min(10, n_vocab)
    top5_mass = np.sum(sorted_top50_p[:, -k5:], axis=-1)
    top10_mass = np.sum(sorted_top50_p[:, -k10:], axis=-1)
    top50_mass = np.sum(sorted_top50_p, axis=-1)
    concentration_gradient = top5_mass / (top50_mass + 1e-8)

    logit_skewness = scipy.stats.skew(v_logits, axis=-1)
    logit_kurtosis = scipy.stats.kurtosis(v_logits, axis=-1)

    # Unigram Prior Surprisal & IGR
    if unigram_log_probs is not None:
        unigram_prior_surprisal = -unigram_log_probs[v_labels]
    else:
        # Fallback approximation: Uniform maximum entropy prior
        unigram_prior_surprisal = np.full_like(surprisal, np.log(n_vocab))
    
    unigram_igr = (unigram_prior_surprisal - surprisal) / (unigram_prior_surprisal + 1e-8)
    bci = gini_coefs * (1.0 - top1_top2_margins)

    # Zipf Anomaly including LogSumExp normalization constant Z(alpha)
    k_ranks = np.arange(1, n_vocab + 1, dtype=np.float64)
    log_k = np.log(k_ranks)
    log_z = scipy.special.logsumexp(-zipf_alphas[:, None] * log_k[None, :], axis=-1)
    predicted_zipf_surprisal = zipf_alphas * np.log(ranks) + log_z
    zipf_anomaly = np.abs(surprisal - predicted_zipf_surprisal)

    # Concentration-Diversity Gap
    max_entropy = np.log(n_vocab)
    norm_entropy = entropies / max_entropy
    gini_entropy_gap = gini_coefs - norm_entropy

    if total_valid_tokens >= 3:
        acc_vals = np.diff(surprisal, n=2)
        surprisal_acc = np.pad(acc_vals, (2, 0), mode='edge')
    else:
        surprisal_acc = np.zeros(total_valid_tokens, dtype=np.float32)

    sid = f"{label_prefix}_{sentence_id}"
    label_name = "Human" if str(model_source).upper() == "HUMAN" else (f"{model_source}_LLM" if not str(model_source).endswith("_LLM") else str(model_source))

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
            "label": label_name
        }
        for idx in range(total_valid_tokens)
    ]

    return records, sentence_id + 1




def calc_slope(x, y):
    if len(x) >= 2 and np.std(x) > 1e-8:
        return float(scipy.stats.linregress(x, y).slope)
    return 0.0


def extract_positional_and_spectral_features(norm_pos, raw_log_prob, entropy, num_bins=10):
    features = {}
    target_bins = np.linspace(0.1, 1.0, num_bins)
    
    lp_interpolated = np.interp(target_bins, norm_pos, raw_log_prob)
    ent_interpolated = np.interp(target_bins, norm_pos, entropy)
    
    for i in range(num_bins):
        features[f"lp_step_{i+1:02d}"] = float(lp_interpolated[i])
        features[f"ent_step_{i+1:02d}"] = float(ent_interpolated[i])
        
    if len(raw_log_prob) >= 4:
        fft_raw = np.fft.rfft(raw_log_prob - np.mean(raw_log_prob))[1:] 
        fft_vals = np.abs(fft_raw)
        
        # Compute Energy using Power Spectrum (|X[k]|^2)
        power_spectrum = (fft_vals ** 2)
        
        # FIX: Dynamically partition frequency spectrum into low and high bands
        mid = max(1, len(power_spectrum) // 2)
        
        features["fft_low_freq_energy"] = float(np.sum(power_spectrum[:mid]))
        features["fft_high_freq_energy"] = float(np.sum(power_spectrum[mid:]))
        features["fft_spectral_ratio"] = float(features["fft_high_freq_energy"] / (features["fft_low_freq_energy"] + 1e-8))
        
        power_norm = power_spectrum / (np.sum(power_spectrum) + 1e-12)
        nonzero_p = power_norm[power_norm > 0]
        features["fft_spectral_entropy"] = float(-np.sum(nonzero_p * np.log(nonzero_p)))
    else:
        features["fft_low_freq_energy"] = 0.0
        features["fft_high_freq_energy"] = 0.0
        features["fft_spectral_ratio"] = 0.0
        features["fft_spectral_entropy"] = 0.0
        
    return features


def _process_single_sentence_group(sid, group, text_map=None, log_base=np.e):
    label = group["label"].iloc[0]
    generator_model = group["generator_model"].iloc[0] if "generator_model" in group.columns else ("Human" if label == "Human" else "LLM")
    is_llm = 0 if (label == "Human" or str(generator_model).upper() == "HUMAN") else 1

    doc_id = group['_id'].iloc[0]
    group = group.sort_values("token_pos")
    
    norm_pos = group["norm_pos"].values
    log_rank = group["log_rank"].values
    
    if "rank" in group.columns:
        ranks = group["rank"].values
    else:
        ranks = log_base ** log_rank if log_base != np.e else np.exp(log_rank)

    raw_log_prob = group["raw_log_prob"].values
    surprisal = group["surprisal"].values
    entropy = group["entropy"].values
    length = len(group)

    # Standardized sample standard deviation (ddof=1)
    mean_e = float(np.mean(entropy))
    std_e = float(np.std(entropy, ddof=1)) if length > 1 else 0.0

    entropy_surprisal_diff = group["entropy_norm_score"].values if "entropy_norm_score" in group.columns else (entropy - surprisal)
    mean_entropy_surprisal_diff = float(np.mean(entropy_surprisal_diff))
    std_entropy_surprisal_diff = float(np.std(entropy_surprisal_diff, ddof=1)) if length > 1 else 0.0
    p25_entropy_surprisal_diff = float(np.percentile(entropy_surprisal_diff, 25))
    p75_entropy_surprisal_diff = float(np.percentile(entropy_surprisal_diff, 75))
    iqr_entropy_surprisal_diff = float(p75_entropy_surprisal_diff - p25_entropy_surprisal_diff)

    diff_entropy = np.diff(entropy) if length > 1 else np.array([0.0])
    mean_abs_diff_entropy = float(np.mean(np.abs(diff_entropy))) if length > 1 else 0.0

    if length >= 3:
        std_diff_entropy = float(np.std(diff_entropy, ddof=1))
        volatility_log_rank = float(np.var(np.diff(log_rank), ddof=1))
        volatility_log_prob = float(np.var(np.diff(raw_log_prob), ddof=1))
    else:
        std_diff_entropy = 0.0
        volatility_log_rank = 0.0
        volatility_log_prob = 0.0

    centered_entropy = entropy - mean_e
    zero_crossings = np.where(np.diff(centered_entropy >= 0))[0] if length > 1 else np.array([])
    entropy_mean_crossing_rate = float(len(zero_crossings) / (length - 1)) if length > 1 else 0.0

    # Leave-One-Out (LOO) rolling shocks & absolute magnitude max
    if length >= 5:
        surp_series = pd.Series(surprisal)
        w = 5
        roll_sum = surp_series.rolling(window=w, min_periods=w, center=True).sum()
        roll_sq_sum = (surp_series**2).rolling(window=w, min_periods=w, center=True).sum()
        
        # Calculate leave-one-out context mean and std (excluding token i)
        loo_mean = (roll_sum - surp_series) / (w - 1)
        loo_var = (roll_sq_sum - surp_series**2 - (w - 1) * (loo_mean**2)) / (w - 2)
        loo_std = np.sqrt(np.maximum(loo_var, 0.0))
        
        local_shocks = ((surp_series - loo_mean) / (loo_std + 1e-8)).dropna().values
        
        if len(local_shocks) > 0:
            max_local_surprisal_shock = float(np.max(np.abs(local_shocks)))
            mean_local_surprisal_shock = float(np.mean(np.abs(local_shocks)))
        else:
            max_local_surprisal_shock, mean_local_surprisal_shock = 0.0, 0.0
    else:
        max_local_surprisal_shock, mean_local_surprisal_shock = 0.0, 0.0

    mean_bci = float(np.mean(group["bci"].values)) if "bci" in group.columns else 0.0
    mean_concentration_gradient = float(np.mean(group["concentration_gradient"].values)) if "concentration_gradient" in group.columns else 0.0
    mean_zipf_anomaly = float(np.mean(group["zipf_anomaly"].values)) if "zipf_anomaly" in group.columns else 0.0
    mean_gini_entropy_div = float(np.mean(group["gini_entropy_div"].values)) if "gini_entropy_div" in group.columns else 0.0

    if length >= 4 and std_e > 1e-8:
        local_stds = [np.std(entropy[i:i+3], ddof=1) for i in range(length - 2)]
        mean_local_entropy_std = float(np.mean(local_stds))
        local_global_std_ratio = float(mean_local_entropy_std / (std_e + 1e-8))
    else:
        mean_local_entropy_std = 0.0
        local_global_std_ratio = 1.0 if length < 4 else 0.0

    # Sample Covariance and Correlation (ddof=1)
    if length >= 3 and std_e > 1e-8 and np.std(surprisal, ddof=1) > 1e-8:
        cov_matrix = np.cov(surprisal, entropy, ddof=1)
        surprisal_entropy_cov = float(cov_matrix[0, 1])
        surprisal_entropy_corr = float(np.nan_to_num(np.corrcoef(surprisal, entropy)[0, 1], nan=0.0))
    else:
        surprisal_entropy_cov = 0.0
        surprisal_entropy_corr = 0.0

    # Markov Regime Entropy Rate
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

    p25_entropy = float(np.percentile(entropy, 25))
    p75_entropy = float(np.percentile(entropy, 75))
    iqr_entropy = p75_entropy - p25_entropy
    median_e = float(np.median(entropy))
    iqr_entropy_ratio = float(iqr_entropy / (median_e + 1e-8))

    p25_log_prob = float(np.percentile(raw_log_prob, 25))
    p75_log_prob = float(np.percentile(raw_log_prob, 75))
    
    surprisal_skew = float(np.nan_to_num(scipy.stats.skew(surprisal), nan=0.0)) if length >= 3 else 0.0
    surprisal_kurtosis = float(np.nan_to_num(scipy.stats.kurtosis(surprisal), nan=0.0)) if length >= 4 else 0.0
    entropy_skew = float(np.nan_to_num(scipy.stats.skew(entropy), nan=0.0)) if length >= 3 else 0.0

    surprisal_var = float(np.var(surprisal, ddof=1)) if length > 1 else 0.0
    surprisal_mean = float(np.mean(surprisal))
    fano_factor = float(np.nan_to_num(surprisal_var / (surprisal_mean + 1e-8), nan=0.0))

    cdf_vals = group["cdf_mass"].values if "cdf_mass" in group.columns else np.zeros(length)
    mean_cdf = float(np.mean(cdf_vals))
    tail_breach_90 = float(np.mean(cdf_vals > 0.90))
    tail_breach_95 = float(np.mean(cdf_vals > 0.95))

    min_e = group["min_entropy"].values if "min_entropy" in group.columns else np.zeros(length)
    renyi2_e = group["renyi2_entropy"].values if "renyi2_entropy" in group.columns else np.zeros(length)
    min_shannon_ratio = float(np.nan_to_num(np.mean(min_e / (entropy + 1e-8)), nan=0.0)) if "min_entropy" in group.columns else 0.0
    renyi2_shannon_ratio = float(np.nan_to_num(np.mean(renyi2_e / (entropy + 1e-8)), nan=0.0)) if "renyi2_entropy" in group.columns else 0.0

    entropy_spike_ratio = float(np.mean(entropy > (mean_e + 1.5 * std_e))) if std_e > 1e-8 else 0.0
    
    if std_e > 1e-8 and length >= 4:
        ac = float(np.corrcoef(entropy[:-1], entropy[1:])[0, 1])
        entropy_autocorr = 0.0 if np.isnan(ac) else ac
    else:
        entropy_autocorr = 0.0

    if length >= 3 and std_e > 1e-8:
        h_centered = entropy - mean_e
        phi1 = np.sum(h_centered[1:] * h_centered[:-1]) / (np.sum(h_centered[:-1] ** 2) + 1e-8)
        ar1_residuals = h_centered[1:] - phi1 * h_centered[:-1]
        ar1_residual_var = float(np.var(ar1_residuals, ddof=1)) if len(ar1_residuals) > 1 else 0.0
    else:
        ar1_residual_var = 0.0

    terminal_mask = norm_pos >= 0.70
    terminal_entropy_slope = calc_slope(norm_pos[terminal_mask], entropy[terminal_mask]) if np.sum(terminal_mask) >= 2 else 0.0

    margins = group["top1_top2_margin"].values if "top1_top2_margin" in group.columns else np.zeros(length)
    p90_margin = float(np.percentile(margins, 90)) if length > 0 else 0.0
    max_rank = float(np.max(ranks)) if length > 0 else 1.0
    bimodal_extreme_index = float((max_rank * p90_margin) / (mean_e + 1e-8))

    entropy_texture_index = float((np.abs(surprisal_kurtosis) * entropy_autocorr) / (std_e + 1e-8))
    surprisal_jitter_index = float(np.mean(np.abs(np.diff(surprisal)))) if length > 1 else 0.0

    mean_gini_coef = float(np.mean(group["gini_coef"].values)) if "gini_coef" in group.columns else 0.0
    mean_zipf_alpha = float(np.mean(group["zipf_alpha"].values)) if "zipf_alpha" in group.columns else 0.0
    mean_top5_mass = float(np.mean(group["top5_mass"].values)) if "top5_mass" in group.columns else 0.0
    mean_top10_mass = float(np.mean(group["top10_mass"].values)) if "top10_mass" in group.columns else 0.0
    mean_top50_mass = float(np.mean(group["top50_mass"].values)) if "top50_mass" in group.columns else 0.0
    mean_logit_skewness = float(np.mean(group["logit_skewness"].values)) if "logit_skewness" in group.columns else 0.0
    mean_logit_kurtosis = float(np.mean(group["logit_kurtosis"].values)) if "logit_kurtosis" in group.columns else 0.0
    mean_unigram_igr = float(np.mean(group["unigram_igr"].values)) if "unigram_igr" in group.columns else 0.0

    head_mask, tail_mask = norm_pos <= 0.25, norm_pos > 0.75
    head_lp = float(np.mean(raw_log_prob[head_mask])) if np.any(head_mask) else float(np.mean(raw_log_prob))
    tail_lp = float(np.mean(raw_log_prob[tail_mask])) if np.any(tail_mask) else float(np.mean(raw_log_prob))

    traj_features = extract_positional_and_spectral_features(norm_pos, raw_log_prob, entropy)

    # FIX: Safely compute optional column metrics to prevent KeyError crashes
    rank_eff_ratio = float(np.mean(group["rank_eff_ratio"].values)) if "rank_eff_ratio" in group.columns else 0.0
    mean_logit_std = float(np.mean(group["logit_std"].values)) if "logit_std" in group.columns else 0.0
    
    mean_margin = float(np.mean(margins)) if length > 0 else 0.0
    std_margin = float(np.std(margins, ddof=1)) if length > 1 else 0.0
    
    surp_acc_vals = group["surprisal_acc"].values if "surprisal_acc" in group.columns else np.zeros(length)
    mean_surp_acc = float(np.mean(surp_acc_vals)) if length > 0 else 0.0
    std_surp_acc = float(np.std(surp_acc_vals, ddof=1)) if length > 1 else 0.0

    return {
        "sentence_id": sid,
        '_id': doc_id,
        "label": label,                      
        "generator_model": generator_model,  
        "is_llm": is_llm,                    
        "token_length": length,
        
        "mean_log_rank": np.mean(log_rank),
        "std_log_rank": np.std(log_rank, ddof=1) if length > 1 else 0.0,
        "slope_log_rank": calc_slope(norm_pos, log_rank),
        "volatility_log_rank": volatility_log_rank,

        "mean_log_prob": np.mean(raw_log_prob),
        "std_log_prob": np.std(raw_log_prob, ddof=1) if length > 1 else 0.0,
        "slope_log_prob": calc_slope(norm_pos, raw_log_prob),
        "volatility_log_prob": volatility_log_prob,
        "p25_log_prob": p25_log_prob,
        "p75_log_prob": p75_log_prob,
        "iqr_log_prob": p75_log_prob - p25_log_prob,
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

        "mean_gini_coef": mean_gini_coef,
        "mean_zipf_alpha": mean_zipf_alpha,
        "mean_top5_mass": mean_top5_mass,
        "mean_top10_mass": mean_top10_mass,
        "mean_top50_mass": mean_top50_mass,
        "mean_logit_skewness": mean_logit_skewness,
        "mean_logit_kurtosis": mean_logit_kurtosis,
        "mean_unigram_igr": mean_unigram_igr,

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
        "rank_eff_ratio": rank_eff_ratio,
        "mean_logit_std": mean_logit_std,

        "diff_head_tail_log_prob": head_lp - tail_lp,
        
        "mean_top1_top2_margin": mean_margin,
        "std_top1_top2_margin": std_margin,
        "mean_surprisal_acc": mean_surp_acc,
        "std_surprisal_acc": std_surp_acc,
        "mean_bci": mean_bci,
        "mean_concentration_gradient": mean_concentration_gradient,
        "mean_zipf_anomaly": mean_zipf_anomaly,
        "mean_gini_entropy_div": mean_gini_entropy_div,
        "max_local_surprisal_shock": max_local_surprisal_shock,
        "mean_local_surprisal_shock": mean_local_surprisal_shock,
        **traj_features
    }


def aggregate_sentence_features(token_df, text_map=None, n_jobs=-1):
    if token_df is None or token_df.empty or "sentence_id" not in token_df.columns:
        print("[WARNING] token_df is empty or missing 'sentence_id'. Returning empty DataFrame.")
        return pd.DataFrame()

    groups = [group for _, group in token_df.groupby("sentence_id")]
    
    records = Parallel(n_jobs=n_jobs, batch_size=100)(
        delayed(_process_single_sentence_group)(
            group["sentence_id"].iloc[0], group, text_map
        ) 
        for group in groups
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
            if len(s.split()) >= 10:  # Enforce minimum 10 words
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
    # Load raw jsonl file directly using the built-in HF 'json' loader
    hc3 = load_dataset(
        "json", 
        data_files="hf://datasets/Hello-SimpleAI/HC3/all.jsonl", 
        split="train"
    ).shuffle(seed=seed)
    
    human_candidates = []
    llm_candidates = []
    
    for idx, item in enumerate(hc3):
        doc_id = item.get("id", item.get("idx", f"hc3_{idx}"))
        
        # Human answers
        for ans in item.get("human_answers", []):
            sents = nltk.sent_tokenize(ans)
            for s in sents:
                if len(s.split()) >= 10:  # Minimum 10 words requirement
                    human_candidates.append({
                        "text": s,
                        "doc_id": doc_id,
                        "is_llm": 0,
                        "generator_model": "Human"
                    })

        # ChatGPT answers
        for ans in item.get("chatgpt_answers", []):
            sents = nltk.sent_tokenize(ans)
            for s in sents:
                if len(s.split()) >= 10:  # Minimum 10 words requirement
                    llm_candidates.append({
                        "text": s,
                        "doc_id": doc_id,
                        "is_llm": 1,
                        "generator_model": "ChatGPT"
                    })

    # Randomly shuffle and select candidates
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
        
        # Parse Human text from 'abstract_sentence'
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

# ==========================================
# VISUALIZATION & REPORT PIPELINE (SAVED TO DIR)
# ==========================================

def generate_significance_report(model_results_dict, output_dir="."):
    """Generates an automatic Markdown significance report summarizing top features."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "significance_report.md")
    
    lines = ["# Feature Statistical Significance Report\n\n"]
    sig_sets = {}
    
    for model_name, sig_df in model_results_dict.items():
        lines.append(f"## Model: {model_name}\n")
        if sig_df.empty:
            lines.append("No results available.\n\n")
            continue
            
        # Desired columns to display in the markdown table
        desired_cols = ["feature", "human_mean_std", "llm_mean_std", "p_location (MW-U)", "cohens_d", "roc_auc"]
        
        # Dynamically keep columns that actually exist in the CSV
        available_cols = [c for c in desired_cols if c in sig_df.columns]
        
        # Fallback to first 6 non-private columns if none of the desired columns are found
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
    """Plots and saves a comparative ROC-AUC heatmap for all models."""
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
    auc_matrix["mean_auc"] = auc_matrix.mean(axis=1)
    auc_matrix = auc_matrix.sort_values(by="mean_auc", ascending=False).drop(columns=["mean_auc"])

    plt.figure(figsize=(10, max(12, len(auc_matrix) * 0.25)))
    sns.heatmap(auc_matrix, annot=True, fmt=".3f", cmap="Blues", cbar_kws={'label': 'ROC-AUC'})
    plt.title("Feature Predictive Power (ROC-AUC) Across Models")
    plt.xlabel("Model Scale")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved all models heatmap to: '{save_path}'")


def plot_best_metric_per_model(model_results_dict, sentence_dfs, output_dir="."):
    """Plots Seaborn violin plots for the best-performing metric of EACH model."""
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
    """Plots separation of a shared significant feature across all models."""
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


def run_visualization_and_report_pipeline(model_significance_results, sentence_dfs, output_dir="."):
    """Master trigger that saves all plots and reports inside output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Automatic Significance Report
    generate_significance_report(model_significance_results, output_dir)
    
    # 2. All Models Heatmap Plot
    plot_all_models_heatmap(model_significance_results, output_dir)
    
    # 3. Best Performing Metric Per Model Plot
    plot_best_metric_per_model(model_significance_results, sentence_dfs, output_dir)
    
    # 4. Shared Significant Feature Plot
    significant_features_by_model = {}
    for model_name, sig_df in model_significance_results.items():
        if "_raw_p_mw" in sig_df.columns:
            sig_feats = sig_df[sig_df["_raw_p_mw"] < 0.05]["feature"].tolist()
        else:
            sig_feats = []
        significant_features_by_model[model_name] = set(sig_feats)

    common_sig_features = set.intersection(*significant_features_by_model.values()) if significant_features_by_model else set()
    
    if common_sig_features and sentence_dfs:
        auc_means = {}
        for feat in common_sig_features:
            aucs = [
                model_significance_results[m].set_index("feature").loc[feat, "_raw_auc"]
                for m in model_significance_results.keys()
                if feat in model_significance_results[m].set_index("feature").index
            ]
            if aucs:
                auc_means[feat] = np.mean(aucs)

        if auc_means:
            best_common_feat = max(auc_means, key=auc_means.get)
            print(f"Selected shared significant feature: '{best_common_feat}' (Mean AUC = {auc_means[best_common_feat]:.3f})")
            visualize_shared_feature_separation(best_common_feat, sentence_dfs, output_dir)

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
        else:
            print(f"  [WARNING] Could not find significance CSV for '{model_name}' in '{target_dir}'")

        if feat_files:
            feat_file = feat_files[0]
            print(f"  Loaded sentence features for {model_name}: {os.path.basename(feat_file)}")
            sentence_level_dfs[model_name] = pd.read_csv(feat_file)
        else:
            print(f"  [WARNING] Could not find sentence features CSV for '{model_name}' in '{target_dir}'")

    if not model_significance_results:
        print("[ERROR] No valid significance CSV files found in directory. Exiting.")
        return

    # Trigger full visualization and report pipeline saving into target_dir
    run_visualization_and_report_pipeline(model_significance_results, sentence_level_dfs, target_dir)

# ==========================================
# 6. MAIN EXECUTION ROUTINE
# ==========================================
def run_pipeline_for_config(run_name, config):
    print(f"\n==================================================")
    print(f" STARTING RUN SETUP: '{run_name}'")
    print(f" Config: {config}")
    print(f"==================================================")

    # Extract settings from the inner dict
    data_setting = config.get("data")
    model_setting = config.get("model")

    # Create directory for saving results
    output_dir = run_name
    os.makedirs(output_dir, exist_ok=True)

    # --- SELECT DATASET BASED ON CONFIG ---
    if data_setting == "HC3":
        print("\n[DATASET] Loading HC3 Dataset (Human vs. ChatGPT)...")
        data_df = load_hc3_sentences(max_samples_per_class=100, seed=42)
        reuse_human = False
    elif data_setting == "clin33":
        print("\n[DATASET] Loading Parquet Human text + CLIN33 Dutch LLM Dataset...")
        human_samples = load_parquet_human_sentences(
            PARQUET_HUMAN_PATH, max_samples=100, min_words=10, seed=42
        )
        llm_samples_df = load_clin33_dutch_llm_sentences(
            CLIN33_CSV_PATH, max_samples=100, seed=42
        )

        # Combine Parquet Human text and CLIN33 LLM text into a single DataFrame
        combined_samples = human_samples + llm_samples_df.to_dict("records")
        rng = np.random.default_rng(42)
        rng.shuffle(combined_samples)
        data_df = pd.DataFrame(combined_samples)
    else:
        raise ValueError(f"Unknown data setting: '{data_setting}' in configuration '{run_name}'")

    model_significance_results = {}
    sentence_level_dfs = {}

    for model_name, model_kwargs in MODELS_CONFIG.items():
        if model_setting not in model_name:
            continue

        print(f"\n------------------------------------------")
        print(f"Processing Model: {model_kwargs['description']} [Dataset: {data_setting}]")
        print(f"------------------------------------------")

        # Download GGUF & Initialize Llama
        model_path = hf_hub_download(
            repo_id=model_kwargs["repo_id"], filename=model_kwargs["filename"]
        )
        llm = Llama(model_path=model_path, **model_kwargs["llama_kwargs"])

        token_records = []
        sentence_id_counter = 0

        # Extract token trajectories for ALL sentences (Human + LLM)
        for idx, row in data_df.iterrows():
            text = row["text"]
            doc_id = row["doc_id"]
            model_src = row["generator_model"]
            lbl_prefix = "H" if row["is_llm"] == 0 else "L"

            records, sentence_id_counter = extract_trajectory_llama_cpp(
                text=text,
                doc_id=doc_id,
                label_prefix=lbl_prefix,
                sentence_id=sentence_id_counter,
                llm=llm,
                model_source=model_src,
                max_tokens=2048,
            )
            token_records.extend(records)

        # Free GPU memory
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save Token Trajectories inside output folder
        token_df = pd.DataFrame(token_records)
        token_csv_path = os.path.join(
            output_dir, f"{model_name}_{data_setting}_token_trajectories.csv"
        )
        token_df.to_csv(token_csv_path, index=False)
        print(f"Saved token trajectories to: '{token_csv_path}'")

        # Aggregate Sentence Features
        sent_df = aggregate_sentence_features(token_df, n_jobs=-1)
        sentence_level_dfs[model_name] = sent_df

        # Save Sentence Features CSV inside output folder
        sent_csv_path = os.path.join(
            output_dir, f"{model_name}_{data_setting}_sentence_features.csv"
        )
        sent_df.to_csv(sent_csv_path, index=False)
        print(f"Saved sentence features to: '{sent_csv_path}'")

        # Calculate & Save Statistical Significance inside output folder
        sig_df = calculate_significance(sent_df)
        model_significance_results[model_name] = sig_df

        sig_csv_path = os.path.join(
            output_dir, f"{model_name}_{data_setting}_significance_results.csv"
        )
        sig_df.to_csv(sig_csv_path, index=False)
        print(f"Saved significance summary to: '{sig_csv_path}'")

        # Run visualizations saving directly into output_dir
        run_visualization_and_report_pipeline(
            model_significance_results, sentence_level_dfs, output_dir
        )


RUN_CONFIGS = {
    "clin_euro": {
        "model": "euro",
        "data": "clin33",
    },
    "clin_qwen": {
        "model": "qwen",
        "data": "clin33",
    },
    "Hc3_qwen": {
        "model": "qwen",
        "data": "HC3",
    },
    # Add more configurations here as needed...
}

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
        help="Optional: Run only a single specific configuration from RUN_CONFIGS (e.g., 'clin_euro'). If omitted, runs ALL sequentially.",
    )

    args = parser.parse_args()

    # --- IF --visualize IS PASSED, RUN VISUALIZATION ONLY ---
    if args.visualize is not None:
        target_dir = args.visualize if args.visualize != "" else "."
        run_visualization_only(target_dir)
        return

    # --- SELECT CONFIGURATIONS TO RUN ---
    if args.run:
        # Run only the specified configuration
        configs_to_run = {args.run: RUN_CONFIGS[args.run]}
    else:
        # Run all configurations sequentially
        configs_to_run = RUN_CONFIGS

    # --- EXECUTE RUNS ---
    for run_name, config in configs_to_run.items():
        run_pipeline_for_config(run_name, config)


if __name__ == "__main__":
    main()