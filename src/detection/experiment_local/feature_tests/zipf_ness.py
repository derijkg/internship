import os
import ast
import re
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import curve_fit
from collections import Counter
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Normalized Zipf & Mandelbrot Fitting
# ---------------------------------------------------------
def zipf_mandelbrot_prob_func(r, alpha, beta, log_C):
    """Log formulation using probabilities: log(p) = log(C) - alpha * log(r + beta)"""
    return log_C - alpha * np.log(r + beta)

def fit_zipf_mandelbrot_normalized(tokens):
    """
    Fits Zipf and Mandelbrot laws on word PROBABILITIES p(r) = f(r) / N.
    """
    total_tokens = len(tokens)
    counts = Counter(tokens)
    freq_sorted = np.array(sorted(counts.values(), reverse=True))
    probs_sorted = freq_sorted / total_tokens  # Convert counts to probabilities p(r)

    ranks = np.arange(1, len(probs_sorted) + 1)
    log_ranks = np.log(ranks)
    log_probs = np.log(probs_sorted)

    # 1. Standard Zipf's Law on Log-Probabilities
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_probs)
    zipf_alpha = -slope
    r_squared = r_value ** 2

    # 2. Zipf-Mandelbrot Law on Log-Probabilities
    try:
        p0 = [1.0, 1.0, log_probs[0]]
        bounds = ([0.0, 0.0, -np.inf], [10.0, 500.0, np.inf])

        popt, _ = curve_fit(
            zipf_mandelbrot_prob_func,
            ranks,
            log_probs,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        mb_alpha, mb_beta, mb_log_C = popt
    except Exception as e:
        print(f"Warning: Mandelbrot fit failed ({e}). Setting to NaN.")
        mb_alpha, mb_beta = np.nan, np.nan

    return {
        "total_tokens": total_tokens,
        "vocab_size": len(counts),
        "zipf_alpha": zipf_alpha,
        "mandelbrot_alpha": mb_alpha,
        "mandelbrot_beta": mb_beta,
        "r_squared": r_squared,
        "ranks": ranks,
        "probs": probs_sorted
    }


# ---------------------------------------------------------
# 2. Token Extractor
# ---------------------------------------------------------
def extract_tokens_from_column(df, col_name):
    invalid_flags = {'FAILED_GENERATION', 'FAILED_VALIDATION', 'NONE', 'NAN', 'NA', 'NULL', '<NA>', ''}
    all_text = []

    for val in df[col_name].dropna():
        if isinstance(val, (list, np.ndarray)):
            items = list(val)
        elif isinstance(val, str):
            val_str = val.strip()
            if val_str.startswith('[') and val_str.endswith(']'):
                try:
                    parsed = ast.literal_eval(val_str)
                    items = parsed if isinstance(parsed, list) else [val_str]
                except Exception:
                    items = [val_str]
            else:
                items = [val_str]
        else:
            items = [str(val)]

        for item in items:
            s_str = str(item).strip()
            if s_str.upper() not in invalid_flags:
                all_text.append(s_str)

    combined_text = " ".join(all_text)
    tokens = re.findall(r'\b\w+\b', combined_text.lower())
    return tokens


# ---------------------------------------------------------
# 3. Pipeline with Equalized Token Sampling
# ---------------------------------------------------------
def analyze_normalized_abstracts(
    parquet_path=r"E:\code\dta\internship\data\gold\llm_added.parquet",
    human_col="abstract",
    seed=42
):
    df = pd.read_parquet(parquet_path)
    llm_full_columns = sorted([col for col in df.columns if col.endswith('_full')])
    target_columns = [human_col] + llm_full_columns

    raw_tokens = {}
    min_token_count = float('inf')

    # Step A: Extract raw tokens and find minimum corpus size
    for col in target_columns:
        tokens = extract_tokens_from_column(df, col)
        raw_tokens[col] = tokens
        min_token_count = min(min_token_count, len(tokens))

    print(f"Equalizing corpus length to target_tokens = {min_token_count:,} words per model...\n")

    # Step B: Downsample all corpora to exactly min_token_count
    rng = np.random.default_rng(seed)
    results = []
    curves_data = {}

    for col in target_columns:
        tokens = raw_tokens[col]
        # Randomly sample min_token_count tokens to make comparison 100% fair
        equalized_tokens = rng.choice(tokens, size=min_token_count, replace=False)

        fit_data = fit_zipf_mandelbrot_normalized(equalized_tokens)

        results.append({
            "generator_model": "Human (abstract)" if col == human_col else col,
            "column_name": col,
            "equalized_tokens": fit_data["total_tokens"],
            "vocab_size": fit_data["vocab_size"],
            "zipf_alpha": fit_data["zipf_alpha"],
            "mandelbrot_alpha": fit_data["mandelbrot_alpha"],
            "mandelbrot_beta": fit_data["mandelbrot_beta"],
            "r_squared": fit_data["r_squared"]
        })

        curves_data[col] = {
            "ranks": fit_data["ranks"],
            "probs": fit_data["probs"]
        }

    return pd.DataFrame(results), curves_data


# ---------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------
def plot_normalized_metrics(results_df):
    metrics = ["zipf_alpha", "mandelbrot_alpha", "mandelbrot_beta"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        colors = ['#2b5c8f' if 'Human' in m else '#2a9d8f' for m in results_df['generator_model']]

        bars = ax.bar(
            results_df['generator_model'], 
            results_df[metric], 
            color=colors, 
            edgecolor='black', 
            alpha=0.85
        )

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, 
                height + (height * 0.01 if height > 0 else 0.02), 
                f'{height:.3f}', 
                ha='center', 
                va='bottom', 
                fontsize=9, 
                fontweight='bold'
            )

        title_map = {
            "zipf_alpha": "Normalized Zipf Alpha (α)\n[Equal Corpus Size]",
            "mandelbrot_alpha": "Normalized Mandelbrot Alpha (α)\n[Equal Corpus Size]",
            "mandelbrot_beta": "Normalized Mandelbrot Beta (β)\n[Rank Offset / Head Flattening]"
        }

        ax.set_title(title_map.get(metric, metric), fontsize=12, pad=12)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_xticks(range(len(results_df)))
        ax.set_xticklabels(results_df['generator_model'], rotation=45, ha='right', fontsize=9.5)
        ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_overlay_log_prob_curves(curves_data):
    plt.figure(figsize=(10, 6))

    for col, data in curves_data.items():
        ranks = data["ranks"]
        probs = data["probs"]
        
        is_human = (col == "abstract")
        label = f"Human ({col})" if is_human else col
        color = '#2b5c8f' if is_human else None
        linewidth = 2.5 if is_human else 1.5
        alpha = 1.0 if is_human else 0.7

        plt.plot(np.log(ranks), np.log(probs), label=label, linewidth=linewidth, alpha=alpha, color=color)

    plt.xlabel("Log(Rank)", fontsize=11)
    plt.ylabel("Log(Probability)", fontsize=11)
    plt.title("Normalized Log-Log Word Rank-Probability Distribution\n(Equalized Token Count Across All Models)", fontsize=13)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 5. Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    parquet_path = r"E:\code\dta\internship\data\gold\llm_added.parquet"
    
    results_df, curves_data = analyze_normalized_abstracts(
        parquet_path=parquet_path,
        human_col="abstract"
    )

    print("="*75)
    print("      EQUALIZED & NORMALIZED ZIPF-MANDELBROT METRICS")
    print("="*75)
    print(results_df[["generator_model", "equalized_tokens", "vocab_size", "zipf_alpha", "mandelbrot_alpha", "mandelbrot_beta", "r_squared"]].to_string(index=False))

    plot_normalized_metrics(results_df)
    plot_overlay_log_prob_curves(curves_data)