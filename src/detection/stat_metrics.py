import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
from joblib import Parallel, delayed
from llama_cpp import Llama

from math_utils import (
    compute_vectorized_gini,
    compute_zipf_exponent,
    compute_zipf_mandelbrot_params,
    calc_slope,
    extract_array_trajectory_features
)


class LogitTrajectoryExtractor:
    """Evaluates text through llama-cpp and extracts token-level logit dynamics."""

    def __init__(self, llm: Llama):
        self.llm = llm

    def extract(
        self,
        text: str,
        doc_id: str,
        label_prefix: str,
        sentence_id: int,
        model_source: str = "LLM",
        max_tokens: int = 2048,
        unigram_log_probs: np.ndarray = None
    ) -> tuple[list, int]:
        text_clean = text.strip()
        tokens = self.llm.tokenize(text_clean.encode("utf-8"))

        bos_id = self.llm.token_bos()
        eos_id = self.llm.token_eos()
        start_id = bos_id if (bos_id is not None and bos_id != -1) else eos_id

        if start_id is not None and start_id != -1:
            if len(tokens) == 0 or tokens[0] != start_id:
                tokens = [start_id] + tokens

        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]

        if len(tokens) < 3:
            return [], sentence_id + 1

        self.llm.reset()
        self.llm.eval(tokens)

        logits = np.array(self.llm.eval_logits, dtype=np.float32)
        shift_logits = logits[:-1, :]
        shift_labels = np.array(tokens[1:], dtype=np.int64)

        special_ids = {tid for tid in (bos_id, eos_id) if tid is not None and tid != -1}
        n_vocab = self.llm.n_vocab()

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
        top2_logits = np.partition(v_logits, -top_k_partition, axis=-1)[:, -top_k_partition:]
        sorted_top2_logits = np.sort(top2_logits, axis=-1)

        p_top1 = np.exp(sorted_top2_logits[:, -1] - lse.squeeze(-1))
        p_top2 = np.exp(sorted_top2_logits[:, -2] - lse.squeeze(-1)) if top_k_partition >= 2 else np.zeros_like(p_top1)

        min_entropies = -np.log(p_top1 + 1e-12)
        renyi2_entropies = -np.log(np.sum(probs**2, axis=-1) + 1e-12)

        target_logits = v_logits[np.arange(total_valid_tokens), v_labels, None]
        ranks = np.sum(v_logits > target_logits, axis=-1) + 1
        cdf_mass = np.sum((v_logits >= target_logits) * probs, axis=-1)

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

        k5, k10 = min(5, n_vocab), min(10, n_vocab)
        top5_mass = np.sum(sorted_top50_p[:, -k5:], axis=-1)
        top10_mass = np.sum(sorted_top50_p[:, -k10:], axis=-1)
        top50_mass = np.sum(sorted_top50_p, axis=-1)
        concentration_gradient = top5_mass / (top50_mass + 1e-8)

        mean_logit = np.mean(v_logits, axis=-1, keepdims=True)
        std_logit = np.std(v_logits, axis=-1, keepdims=True) + 1e-8
        norm_diff = (v_logits - mean_logit) / std_logit

        logit_skewness = np.mean(norm_diff ** 3, axis=-1)
        logit_kurtosis = np.mean(norm_diff ** 4, axis=-1) - 3.0

        if unigram_log_probs is not None:
            unigram_log_probs_arr = np.asarray(unigram_log_probs)
            unigram_prior_surprisal = -unigram_log_probs_arr[v_labels]
        else:
            unigram_prior_surprisal = np.full_like(surprisal, np.log(n_vocab))

        unigram_igr = (unigram_prior_surprisal - surprisal) / (unigram_prior_surprisal + 1e-8)
        bci = gini_coefs * (1.0 - top1_top2_margins)

        k_ranks = np.arange(1, n_vocab + 1, dtype=np.float64)
        log_k = np.log(k_ranks)
        log_z = scipy.special.logsumexp(-zipf_alphas[:, None] * log_k[None, :], axis=-1)
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
        label_name = "Human" if str(model_source).upper() == "HUMAN" else (
            f"{model_source}_LLM" if not str(model_source).endswith("_LLM") else str(model_source)
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


def _process_single_sentence_group(sid: str, group: pd.DataFrame) -> dict:
    """Worker function to aggregate sentence-level dynamics from token sequences."""
    group = group.sort_values("token_pos")
    length = len(group)

    label = group["label"].iloc[0]
    generator_model = group["generator_model"].iloc[0] if "generator_model" in group.columns else ("Human" if str(label).upper() == "HUMAN" else "LLM")
    is_llm = 0 if (str(label).upper() == "HUMAN" or str(generator_model).upper() == "HUMAN") else 1
    doc_id = group['_id'].iloc[0]

    norm_pos = np.ascontiguousarray(group["norm_pos"].values.astype(np.float64))
    log_rank = np.ascontiguousarray(group["log_rank"].values.astype(np.float64))
    ranks = group["rank"].values.astype(np.float64) if "rank" in group.columns else np.exp(np.clip(log_rank, -100, 100))
    raw_log_prob = np.ascontiguousarray(group["raw_log_prob"].values.astype(np.float64))
    surprisal = np.ascontiguousarray(group["surprisal"].values.astype(np.float64))
    entropy = np.ascontiguousarray(group["entropy"].values.astype(np.float64))

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

    mean_e = float(np.mean(entropy))
    std_e = float(np.std(entropy, ddof=1)) if length > 1 else 0.0

    entropy_surprisal_diff = np.ascontiguousarray(group["entropy_norm_score"].values.astype(np.float64)) if "entropy_norm_score" in group.columns else (entropy - surprisal)
    mean_entropy_surprisal_diff = float(np.mean(entropy_surprisal_diff))
    std_entropy_surprisal_diff = float(np.std(entropy_surprisal_diff, ddof=1)) if length > 1 else 0.0
    p25_diff, p75_diff = float(np.percentile(entropy_surprisal_diff, 25)), float(np.percentile(entropy_surprisal_diff, 75))
    iqr_entropy_surprisal_diff = float(p75_diff - p25_diff)

    diff_entropy = np.diff(entropy) if length > 1 else np.array([0.0])
    mean_abs_diff_entropy = float(np.mean(np.abs(diff_entropy))) if length > 1 else 0.0

    if length >= 3:
        std_diff_entropy = float(np.std(diff_entropy, ddof=1))
        volatility_log_rank = float(np.var(np.diff(log_rank), ddof=1))
        volatility_log_prob = float(np.var(np.diff(raw_log_prob), ddof=1))
    else:
        std_diff_entropy, volatility_log_rank, volatility_log_prob = 0.0, 0.0, 0.0

    centered_entropy = entropy - mean_e
    zero_crossings = np.where(np.diff(centered_entropy >= 0))[0] if length > 1 else np.array([])
    entropy_mean_crossing_rate = float(len(zero_crossings) / (length - 1)) if length > 1 else 0.0

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
        max_local_surprisal_shock, mean_local_surprisal_shock = float(np.max(local_shocks)), float(np.mean(local_shocks))
    else:
        max_local_surprisal_shock, mean_local_surprisal_shock = 0.0, 0.0

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
        local_global_std_ratio = float(np.mean(local_stds) / std_e)
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

    entropy_autocorr = float(np.corrcoef(entropy[:-1], entropy[1:])[0, 1]) if (std_e > 1e-8 and length >= 4) else 0.0
    if np.isnan(entropy_autocorr):
        entropy_autocorr = 0.0

    surprisal_var = float(np.var(surprisal, ddof=1)) if length > 1 else 0.0
    surprisal_mean = float(np.mean(surprisal))
    fano_factor = float(np.nan_to_num(surprisal_var / (surprisal_mean + 1e-8), nan=0.0))

    surprisal_skew = float(np.nan_to_num(scipy.stats.skew(surprisal), nan=0.0)) if length >= 3 else 0.0
    surprisal_kurtosis = float(np.nan_to_num(scipy.stats.kurtosis(surprisal), nan=0.0)) if length >= 4 else 0.0
    entropy_skew = float(np.nan_to_num(scipy.stats.skew(entropy), nan=0.0)) if length >= 3 else 0.0

    p25_e, p75_e = float(np.percentile(entropy, 25)), float(np.percentile(entropy, 75))
    iqr_entropy_ratio = float((p75_e - p25_e) / (np.median(entropy) + 1e-8))

    p25_lp, p75_lp = float(np.percentile(raw_log_prob, 25)), float(np.percentile(raw_log_prob, 75))

    cdf_vals = np.ascontiguousarray(group["cdf_mass"].values.astype(np.float64)) if "cdf_mass" in group.columns else np.zeros(length)
    mean_cdf, tail_breach_90, tail_breach_95 = float(np.mean(cdf_vals)), float(np.mean(cdf_vals > 0.90)), float(np.mean(cdf_vals > 0.95))

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

    zipf_traj_features = extract_array_trajectory_features(norm_pos, zipf_alpha, "zipf")
    gini_traj_features = extract_array_trajectory_features(norm_pos, gini_coef, "gini")
    ent_traj_features = extract_array_trajectory_features(norm_pos, entropy, "ent")
    lp_traj_features = extract_array_trajectory_features(norm_pos, raw_log_prob, "lp")
    mb_beta_traj_features = extract_array_trajectory_features(norm_pos, mb_beta, 'mb_beta')

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


class SentenceFeatureAggregator:
    """Aggregates token dataframe trajectories into sentence-level metrics in parallel."""

    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs

    def aggregate(self, token_df: pd.DataFrame) -> pd.DataFrame:
        if token_df is None or token_df.empty or "sentence_id" not in token_df.columns:
            print("[WARNING] token_df is empty or missing 'sentence_id'.")
            return pd.DataFrame()

        groups = [group for _, group in token_df.groupby("sentence_id")]
        records = Parallel(n_jobs=self.n_jobs, batch_size=100)(
            delayed(_process_single_sentence_group)(group["sentence_id"].iloc[0], group)
            for group in groups
        )
        return pd.DataFrame(records)