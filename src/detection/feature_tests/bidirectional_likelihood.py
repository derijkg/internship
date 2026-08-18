import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

random.seed(42)


def compute_causal_likelihoods(
    text, causal_model, causal_tokenizer, device="cpu"
):
    """
    Computes Forward Log-Likelihood, Backward Log-Likelihood, and Asymmetry Score
    using a causal decoder model.
    """
    causal_model.eval()

    # --- 1. Forward Log-Likelihood ---
    enc = causal_tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]

    if input_ids.shape[1] < 3:
        return {"fwd_ll": np.nan, "bwd_ll": np.nan, "asymmetry": np.nan}

    with torch.no_grad():
        outputs = causal_model(input_ids)
        logits = outputs.logits[:, :-1, :]  # Shape: (1, seq_len - 1, vocab)
        labels = input_ids[:, 1:]  # Shape: (1, seq_len - 1)

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, 2, labels.unsqueeze(-1)
        ).squeeze(-1)
        fwd_mean_ll = token_log_probs.mean().item()

    # --- 2. Backward Log-Likelihood ---
    rev_input_ids = torch.flip(input_ids, dims=[1]).to(device)

    with torch.no_grad():
        rev_outputs = causal_model(rev_input_ids)
        rev_logits = rev_outputs.logits[:, :-1, :]
        rev_labels = rev_input_ids[:, 1:]

        rev_log_probs = F.log_softmax(rev_logits, dim=-1)
        rev_token_log_probs = torch.gather(
            rev_log_probs, 2, rev_labels.unsqueeze(-1)
        ).squeeze(-1)
        bwd_mean_ll = rev_token_log_probs.mean().item()

    asymmetry_score = fwd_mean_ll - bwd_mean_ll

    return {
        "fwd_ll": fwd_mean_ll,
        "bwd_ll": bwd_mean_ll,
        "asymmetry": asymmetry_score,
    }


def compute_bidirectional_pll(
    text, bidir_model, bidir_tokenizer, device="cpu", batch_size=32
):
    """
    Computes Masked Pseudo-Log-Likelihood (PLL) using a Masked LM with
    memory-efficient 2D log_softmax and mini-batching.
    """
    bidir_model.eval()
    tokens = bidir_tokenizer(
        text, return_tensors="pt", add_special_tokens=True
    ).to(device)
    input_ids = tokens["input_ids"][0]

    mask_id = bidir_tokenizer.mask_token_id
    special_ids = set(bidir_tokenizer.all_special_ids)
    target_indices = [
        i for i, t in enumerate(input_ids.tolist()) if t not in special_ids
    ]

    if not target_indices:
        return np.nan

    batch_input_ids = input_ids.repeat(len(target_indices), 1)
    for row_idx, token_idx in enumerate(target_indices):
        batch_input_ids[row_idx, token_idx] = mask_id

    pll_sum = 0.0

    with torch.no_grad():
        for i in range(0, len(target_indices), batch_size):
            chunk_inputs = batch_input_ids[i : i + batch_size]
            chunk_target_indices = target_indices[i : i + batch_size]
            chunk_orig_tokens = input_ids[chunk_target_indices]

            outputs = bidir_model(chunk_inputs)

            # 3D -> 2D reduction before log_softmax
            batch_idx = torch.arange(len(chunk_target_indices), device=device)
            masked_logits = outputs.logits[batch_idx, chunk_target_indices, :]

            log_probs = F.log_softmax(masked_logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs, 1, chunk_orig_tokens.unsqueeze(-1)
            ).squeeze(-1)
            pll_sum += token_log_probs.sum().item()

    return pll_sum / len(target_indices)


def process_experiment_dataset(
    data,
    causal_model,
    causal_tokenizer,
    bidir_model,
    bidir_tokenizer,
    device="cpu",
):
    """
    Processes a dataset dictionary/dataframe.
    """
    results = []

    for entry in data:
        text = entry["sentence"]
        label = entry["label"]

        causal_res = compute_causal_likelihoods(
            text, causal_model, causal_tokenizer, device=device
        )
        bidir_pll = compute_bidirectional_pll(
            text, bidir_model, bidir_tokenizer, device=device
        )

        causal_bidir_divergence = (
            causal_res["fwd_ll"] - bidir_pll
            if (not np.isnan(causal_res["fwd_ll"]) and not np.isnan(bidir_pll))
            else np.nan
        )

        results.append({
            "sentence": text,
            "label": "AI" if label == 1 else "Human",
            "fwd_ll": causal_res["fwd_ll"],
            "bwd_ll": causal_res["bwd_ll"],
            "asymmetry": causal_res["asymmetry"],
            "bidir_pll": bidir_pll,
            "causal_bidir_divergence": causal_bidir_divergence,
        })

    return pd.DataFrame(results)


def visualize_feature2_results(df, save_plot=False):
    """
    Plots score distributions, forward vs backward scatter, and reports AUROC.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Asymmetry Score Distribution
    sns.kdeplot(
        data=df,
        x="asymmetry",
        hue="label",
        fill=True,
        common_norm=False,
        ax=axes[0],
        palette={"Human": "blue", "AI": "red"},
    )
    axes[0].set_title(
        "Forward vs Backward Likelihood Asymmetry\n(fwd_ll - bwd_ll)"
    )
    axes[0].set_xlabel("Asymmetry Score")

    # 2. Causal vs Bidirectional Divergence Distribution
    sns.kdeplot(
        data=df,
        x="causal_bidir_divergence",
        hue="label",
        fill=True,
        common_norm=False,
        ax=axes[1],
        palette={"Human": "blue", "AI": "red"},
    )
    axes[1].set_title(
        "Causal vs Bidirectional Divergence\n(Causal Fwd LL - Bidir PLL)"
    )
    axes[1].set_xlabel("Divergence Score")

    # 3. Scatter Plot
    sns.scatterplot(
        data=df,
        x="fwd_ll",
        y="bwd_ll",
        hue="label",
        alpha=0.7,
        ax=axes[2],
        palette={"Human": "blue", "AI": "red"},
    )
    axes[2].set_title("Forward Log-Likelihood vs. Backward Log-Likelihood")
    axes[2].set_xlabel("Forward Mean Log-Likelihood")
    axes[2].set_ylabel("Backward Mean Log-Likelihood")

    plt.tight_layout()
    if save_plot:
        plt.savefig("feature2_experiment_results.png", dpi=300)
    plt.show()

    # --- Compute AUROC Scores ---
    print("\n" + "=" * 45)
    print("      FEATURE 2 SEPARABILITY PERFORMANCE     ")
    print("=" * 45)

    binary_labels = (df["label"] == "AI").astype(int)

    for metric in ["asymmetry", "causal_bidir_divergence", "fwd_ll"]:
        valid_mask = ~df[metric].isna()
        if valid_mask.sum() > 0:
            auc = roc_auc_score(
                binary_labels[valid_mask], df.loc[valid_mask, metric]
            )
            auc_final = max(auc, 1 - auc)
            print(f"AUROC [{metric}]: {auc_final:.4f}")
    print("=" * 45)



#right side only
def compute_right_context_gain(text, model, tokenizer, device="cpu"):
    """
    Measures how much predictive probability increases when RIGHT context is revealed
    compared to LEFT context alone.
    
    Uses RobBERT (Dutch MLM) to evaluate both contexts on the exact same tokenizer.
    """
    model.eval()
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
    input_ids = tokens["input_ids"][0]
    seq_len = len(input_ids)
    
    mask_id = tokenizer.mask_token_id
    special_ids = set(tokenizer.all_special_ids)
    
    # Target non-special tokens (skip <s> and </s>)
    target_indices = [i for i, t in enumerate(input_ids.tolist()) if t not in special_ids]
    
    if len(target_indices) < 3:
        return np.nan

    gains = []

    # Iterate over target tokens in the sentence
    for idx in target_indices:
        orig_token_id = input_ids[idx].item()
        
        # --- 1. Left-Only Context (Mask target token AND all future tokens) ---
        left_only_ids = input_ids.clone()
        left_only_ids[idx:] = mask_id  # Mask target and everything to the right
        
        # --- 2. Full Context (Mask ONLY target token) ---
        full_context_ids = input_ids.clone()
        full_context_ids[idx] = mask_id
        
        # Batch pass for speed
        batch_inputs = torch.stack([left_only_ids, full_context_ids]).to(device)
        
        with torch.no_grad():
            logits = model(batch_inputs).logits  # Shape: (2, seq_len, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Extract log probs for the original token at position `idx`
            left_only_log_prob = log_probs[0, idx, orig_token_id].item()
            full_context_log_prob = log_probs[1, idx, orig_token_id].item()
            
            # Information Gain from revealing right context
            gain = full_context_log_prob - left_only_log_prob
            gains.append(gain)
            
    # Return the mean information gain across the sentence
    return np.mean(gains)


def test_strategy_gain(data, model, tokenizer, device="cpu"):
    results = []
    print("Processing dataset with Right-Context Information Gain...")
    
    for entry in data:
        text = entry["sentence"]
        label = "AI" if entry["label"] == 1 else "Human"
        
        gain_score = compute_right_context_gain(text, model, tokenizer, device=device)
        
        results.append({
            "sentence": text,
            "label": label,
            "info_gain": gain_score
        })
        
    return pd.DataFrame(results)

def visualize_strategy_gain_results(df):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    
    # KDE Plot of Information Gain
    sns.kdeplot(data=df, x="info_gain", hue="label", fill=True, common_norm=False, palette={"Human": "blue", "AI": "red"})
    plt.title("Information Gain from Right Context\n(Full Context LogProb - Left-Only LogProb)")
    plt.xlabel("Right Context Information Gain (Higher = More Human-like)")
    plt.show()

    # Calculate AUROC
    binary_labels = (df["label"] == "AI").astype(int)
    valid_mask = ~df["info_gain"].isna()
    
    auc = roc_auc_score(binary_labels[valid_mask], df.loc[valid_mask, "info_gain"])
    auc_final = max(auc, 1 - auc)
    
    print("\n" + "="*45)
    print("   RIGHT-CONTEXT INFORMATION GAIN PERFORMANCE   ")
    print("="*45)
    print(f"AUROC [info_gain]: {auc_final:.4f}")
    print("="*45)



if __name__ == "__main__":
    # 2. Data Loading & Verification
    df = pd.read_parquet(
        Path("/home/gderijck/internship/data/gold/llm_added.parquet")
    )
    dataset = []
    targets = [("abstract_sentence", 0, 2000)] + [
        (c, 1, 500) for c in df.columns if c.endswith("_single")
    ]

    print("\n" + "=" * 55)
    print("        DATASET LOADING & CLASS VERIFICATION       ")
    print("=" * 55)

    for col, label, k in targets:
        if col in df:
            sents = [
                s.strip()
                for item in df[col].dropna()
                for s in (item if isinstance(item, (list, np.ndarray)) else [item])
                if isinstance(s, str)
                and len(s.split()) >= 12
                and "FAILED_" not in s
                and s.strip()
            ]
            if sents:
                sampled = random.sample(sents, min(k, len(sents)))
                dataset.extend(
                    {"sentence": s, "label": label}
                    for s in sampled
                )
                class_type = "Human" if label == 0 else "AI"
                print(
                    f"Column '{col}' [{class_type}, label={label}]: "
                    f"Available = {len(sents):<6} | Sampled = {len(sampled)}"
                )

    if not dataset:
        raise ValueError("Dataset is empty! Check filtering criteria.")

    # --- Class Summary Breakdown ---
    dataset_df = pd.DataFrame(dataset)
    label_counts = dataset_df["label"].value_counts().to_dict()
    human_count = label_counts.get(0, 0)
    ai_count = label_counts.get(1, 0)

    print("-" * 55)
    print(f"Total Human Sentences (Label 0): {human_count}")
    print(f"Total AI Sentences    (Label 1): {ai_count}")
    print(f"Total Dataset Size             : {len(dataset)}")
    print("=" * 55 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running experiment on device: {device}")

    strat = 'right'

    if strat=='bi':
        # 1. Load Lightweight Models
        causal_model_name = "yhavinga/gpt2-medium-dutch"
        bidir_model_name = "pdelobelle/robbert-v2-dutch-base"

        print("Loading models...")
        causal_tokenizer = AutoTokenizer.from_pretrained(causal_model_name)
        causal_model = AutoModelForCausalLM.from_pretrained(causal_model_name).to(
            device
        )

        bidir_tokenizer = AutoTokenizer.from_pretrained(bidir_model_name)
        bidir_model = AutoModelForMaskedLM.from_pretrained(bidir_model_name).to(
            device
        )

        # 3. Execute Experiment & Visualize
        results_df = process_experiment_dataset(
            data=dataset,
            causal_model=causal_model,
            causal_tokenizer=causal_tokenizer,
            bidir_model=bidir_model,
            bidir_tokenizer=bidir_tokenizer,
            device=device,
        )

        print("\nSample Results Table:")
        print(
            results_df[
                ["sentence", "label", "asymmetry", "causal_bidir_divergence"]
            ].head()
        )

        # 4. Generate Visualizations & Metrics
        visualize_feature2_results(results_df)




    if strat == 'right':
        # Load RobBERT (Dutch MLM)
        model_name = "pdelobelle/robbert-v2-dutch-base"
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        
        df_gain = test_strategy_gain(dataset, model, tokenizer, device=device)
        visualize_strategy_gain_results(df_gain)