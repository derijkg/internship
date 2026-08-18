import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# =========================================================
# 1. DATASET LOADING (YOUR PROVIDED CODE)
# =========================================================
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


# =========================================================
# 2. MODEL LOADING (Qwen 2.5)
# =========================================================
# Options: "Qwen/Qwen2.5-1.5B" (Recommended) or "Qwen/Qwen2.5-0.5B" (Ultra-fast)
model_name = "Qwen/Qwen2.5-1.5B" 

print(f"\nLoading model '{model_name}' into memory...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)
if device == "cpu":
    model.to("cpu")
model.eval()


# =========================================================
# 3. MATHEMATICAL EFFECTIVE RANK (ERank) FUNCTION
# =========================================================
def compute_normalized_erank(H_l):
    """
    Computes Normalized Effective Rank for a hidden activation matrix H_l of shape (N, d).
    
    ERank = exp(- sum(p_i * ln(p_i))) where p_i = sigma_i / sum(sigma)
    Normalized ERank = ERank / min(N, d) -> Ranges from (0, 1]
    """
    N, d = H_l.shape
    if N < 3:
        return np.nan
    
    # 1. Singular Value Decomposition of the sentence matrix
    # S has length min(N, d). Since N <= 40 and d >= 1536, len(S) = N.
    S = torch.linalg.svdvals(H_l.float()) 
    
    # 2. Normalize singular values to form a probability distribution
    S_sum = S.sum()
    if S_sum == 0:
        return np.nan
    p = S / S_sum
    
    # 3. Compute Shannon Entropy (in nats) over the singular value spectrum
    p_nz = p[p > 0]
    entropy = -torch.sum(p_nz * torch.log(p_nz)).item()
    
    # 4. Effective Rank
    erank = np.exp(entropy)
    
    # 5. Normalize by maximum theoretical rank min(N, d) = N
    max_rank = min(N, d)
    norm_erank = erank / max_rank
    
    return norm_erank


# =========================================================
# 4. PROCESS DATASET & EXTRACT LAYER TRAJECTORIES
# =========================================================
print("\nExtracting Effective Matrix Rank trajectories across layers...")

results = []

for entry in tqdm(dataset):
    text = entry["sentence"]
    label_num = entry["label"]
    label_str = "AI" if label_num == 1 else "Human"
    
    # Tokenize input sentence
    enc = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    seq_len = input_ids.shape[1]
    
    if seq_len < 4:
        continue
        
    with torch.no_grad():
        # Request all intermediate layer hidden states
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of (L + 1) layers
        
        num_layers = len(hidden_states) - 1  # Exclude raw embedding layer
        
        layer_eranks = []
        for l in range(1, num_layers + 1):
            H_l = hidden_states[l][0]  # Shape: (seq_len, hidden_dim)
            norm_erank = compute_normalized_erank(H_l)
            layer_eranks.append(norm_erank)
            
        layer_eranks = np.array(layer_eranks)
        
        # Summary Features for Scalar AUROC Testing:
        # Mid-stage layers (30% to 70% depth)
        mid_start, mid_end = int(num_layers * 0.3), int(num_layers * 0.7)
        mid_stage_erank = layer_eranks[mid_start:mid_end].mean()
        
        # Late-stage layers (70% to 95% depth)
        late_start, late_end = int(num_layers * 0.7), int(num_layers * 0.95)
        late_stage_erank = layer_eranks[late_start:late_end].mean()
        
        # Minimum ERank across all layers
        min_erank = layer_eranks.min()

        results.append({
            "sentence": text,
            "label": label_str,
            "seq_len": seq_len,
            "mid_stage_erank": mid_stage_erank,
            "late_stage_erank": late_stage_erank,
            "min_erank": min_erank,
            "full_layer_eranks": layer_eranks
        })

results_df = pd.DataFrame(results)


# =========================================================
# 5. VISUALIZATION & AUROC EVALUATION
# =========================================================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# --- Plot 1: Layer-wise ERank Trajectory Curve ---
human_eranks = np.stack(results_df[results_df["label"] == "Human"]["full_layer_eranks"].values)
ai_eranks = np.stack(results_df[results_df["label"] == "AI"]["full_layer_eranks"].values)

num_layers = human_eranks.shape[1]
x_layers = np.arange(1, num_layers + 1)

# Mean and standard error
axes[0].plot(x_layers, human_eranks.mean(axis=0), label="Human", color="blue", linewidth=2.5)
axes[0].fill_between(x_layers, 
                     human_eranks.mean(axis=0) - human_eranks.std(axis=0)*0.2,
                     human_eranks.mean(axis=0) + human_eranks.std(axis=0)*0.2,
                     color="blue", alpha=0.15)

axes[0].plot(x_layers, ai_eranks.mean(axis=0), label="AI", color="red", linewidth=2.5)
axes[0].fill_between(x_layers, 
                     ai_eranks.mean(axis=0) - ai_eranks.std(axis=0)*0.2,
                     ai_eranks.mean(axis=0) + ai_eranks.std(axis=0)*0.2,
                     color="red", alpha=0.15)

axes[0].set_title(f"Method 3: Layer-wise Normalized ERank Trajectory\n({model_name})", fontsize=12)
axes[0].set_xlabel("Transformer Layer Depth (l)", fontsize=11)
axes[0].set_ylabel("Normalized Effective Rank (Lower = Subspace Collapse)", fontsize=11)
axes[0].legend(fontsize=11)

# --- Plot 2: Late-Stage ERank Score Distribution ---
sns.kdeplot(data=results_df, x="late_stage_erank", hue="label", fill=True, common_norm=False, ax=axes[1], palette={"Human": "blue", "AI": "red"})
axes[1].set_title("Late-Stage Layer ERank Distribution", fontsize=12)
axes[1].set_xlabel("Late-Stage ERank Score", fontsize=11)

plt.tight_layout()
plt.savefig("method3_erank_dutch_results.png", dpi=300)
plt.show()

# --- Compute AUROC Scores ---
print("\n" + "="*50)
print("      METHOD 3 (ERank) SEPARABILITY PERFORMANCE     ")
print("="*50)

binary_labels = (results_df["label"] == "AI").astype(int)

for metric in ["mid_stage_erank", "late_stage_erank", "min_erank"]:
    auc = roc_auc_score(binary_labels, results_df[metric])
    # Direction: Lower ERank indicates AI (Representation Collapse)
    auc_final = max(auc, 1 - auc) 
    print(f"AUROC [{metric:<18}]: {auc_final:.4f}")

print("="*50 + "\n")