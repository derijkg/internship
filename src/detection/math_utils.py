import numpy as np
import scipy.stats as stats


def compute_vectorized_gini(probs: np.ndarray, top_k: int = 500) -> float | np.ndarray:
    """Computes Gini inequality coefficient across top-k vocabulary probabilities."""
    probs_arr = np.asarray(probs, dtype=np.float64)
    is_1d = probs_arr.ndim == 1
    probs_2d = np.atleast_2d(probs_arr)
    M, V = probs_2d.shape
    actual_k = min(top_k, V)

    if actual_k < 2:
        gini = np.zeros(M, dtype=np.float64)
        return gini[0] if is_1d else gini

    topk_probs = np.partition(probs_2d, -actual_k, axis=-1)[:, -actual_k:]
    sorted_topk = np.sort(topk_probs, axis=-1)

    k_idx = np.arange(1, actual_k + 1, dtype=np.float64)
    weights = (actual_k - k_idx + 0.5) / actual_k

    total_mass = np.sum(sorted_topk, axis=-1, keepdims=True)
    lorenz_area = np.sum(sorted_topk * weights, axis=-1, keepdims=True) / (total_mass + 1e-12)

    gini = (1.0 - 2.0 * lorenz_area).squeeze(-1)
    return gini[0] if is_1d else gini


def compute_zipf_exponent(v_logits: np.ndarray, top_k: int = 20) -> float | np.ndarray:
    """Estimates Zipfian power-law exponent over top-k logit rankings."""
    v_logits_arr = np.asarray(v_logits, dtype=np.float64)
    is_1d = v_logits_arr.ndim == 1
    v_logits_2d = np.atleast_2d(v_logits_arr)
    M, V = v_logits_2d.shape
    actual_k = min(top_k, V)

    if actual_k < 2:
        alphas = np.zeros(M, dtype=np.float64)
        return alphas[0] if is_1d else alphas

    topk_logits = np.partition(v_logits_2d, -actual_k, axis=-1)[:, -actual_k:]
    sorted_topk = np.sort(topk_logits, axis=-1)[:, ::-1]

    log_ranks = np.log(np.arange(1, actual_k + 1, dtype=np.float64))
    mean_x = np.mean(log_ranks)
    var_x = np.var(log_ranks)
    mean_y = np.mean(sorted_topk, axis=-1, keepdims=True)

    cov_xy = np.mean((log_ranks - mean_x) * (sorted_topk - mean_y), axis=-1)
    zipf_alpha = -cov_xy / (var_x + 1e-12)

    return zipf_alpha[0] if is_1d else zipf_alpha


def compute_zipf_mandelbrot_params(
    v_logits: np.ndarray,
    top_k: int = 20,
    beta_min: float = 0.0,
    beta_max: float = 10.0,
    beta_steps: int = 101
) -> tuple:
    """Estimates Zipf-Mandelbrot rank-frequency alpha and beta parameters."""
    v_logits_arr = np.asarray(v_logits, dtype=np.float64)
    is_1d = (v_logits_arr.ndim == 1)
    v_logits_2d = np.atleast_2d(v_logits_arr)
    M, V = v_logits_2d.shape
    actual_k = min(top_k, V)

    if actual_k < 2:
        alphas, betas = np.zeros(M, dtype=np.float64), np.zeros(M, dtype=np.float64)
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
    y1, y2, y3 = mse_grid[row_idx, idx_left], mse_grid[row_idx, idx_mid], mse_grid[row_idx, idx_right]

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


def calc_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates linear regression slope between x and y."""
    if len(x) >= 2 and np.std(x) > 1e-8:
        return float(stats.linregress(x, y).slope)
    return 0.0


def extract_array_trajectory_features(
    norm_pos: np.ndarray,
    array_vals: np.ndarray,
    feature_prefix: str,
    num_bins: int = 10
) -> dict:
    """Extracts binned, volatility, macro span, pairwise, and FFT features from array trajectory."""
    features = {}
    if len(array_vals) == 0:
        return features

    target_bins = np.linspace(0.1, 1.0, num_bins)
    interpolated = np.interp(target_bins, norm_pos, array_vals)

    for i in range(num_bins):
        features[f"{feature_prefix}_step_{i+1:02d}"] = float(interpolated[i])

    adj_diffs = np.abs(np.diff(interpolated))
    for i in range(len(adj_diffs)):
        features[f"{feature_prefix}_diff_step_{i+1:02d}_{i+2:02d}"] = float(adj_diffs[i])

    features[f"{feature_prefix}_total_variation"] = float(np.sum(adj_diffs))
    features[f"{feature_prefix}_max_local_jump"] = float(np.max(adj_diffs)) if len(adj_diffs) > 0 else 0.0
    features[f"{feature_prefix}_mean_local_jump"] = float(np.mean(adj_diffs)) if len(adj_diffs) > 0 else 0.0
    features[f"{feature_prefix}_std_local_jump"] = float(np.std(adj_diffs)) if len(adj_diffs) > 0 else 0.0

    start_val, mid_val, end_val = interpolated[0], interpolated[num_bins // 2], interpolated[-1]
    features[f"{feature_prefix}_span_start_to_end"] = float(end_val - start_val)
    features[f"{feature_prefix}_abs_span_start_to_end"] = float(abs(end_val - start_val))
    features[f"{feature_prefix}_span_start_to_mid"] = float(mid_val - start_val)
    features[f"{feature_prefix}_abs_span_start_to_mid"] = float(abs(mid_val - start_val))
    features[f"{feature_prefix}_span_mid_to_end"] = float(end_val - mid_val)
    features[f"{feature_prefix}_abs_span_mid_to_end"] = float(abs(end_val - mid_val))

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

    centered_interp = interpolated - np.mean(interpolated)
    fft_raw = np.fft.rfft(centered_interp)[1:]
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