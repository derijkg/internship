import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

random.seed(42)

# =========================================================
# 1. CONTEXTUAL CAUSAL LIKELIHOOD (Forward LL given Left Context)
# =========================================================
def compute_contextual_causal_ll(left_context, target_sentence, causal_model, causal_tokenizer, device="cpu"):
    """
    Computes P(target_sentence | left_context) using a Causal LM.
    Only token log-probs corresponding to target_sentence are included in the average.
    """
    causal_model.eval()
    
    # Tokenize left_context and target_sentence cleanly using token IDs
    left_ids = causal_tokenizer.encode(left_context, add_special_tokens=True)
    target_ids = causal_tokenizer.encode(" " + target_sentence.strip(), add_special_tokens=False)
    
    if len(target_ids) < 2:
        return np.nan

    full_ids = torch.tensor([left_ids + target_ids], device=device)
    
    # Identify target token index span in the combined sequence
    target_start_idx = len(left_ids)
    target_end_idx = len(left_ids) + len(target_ids)

    with torch.no_grad():
        outputs = causal_model(full_ids)
        logits = outputs.logits[:, :-1, :]  # Predicts next token
        labels = full_ids[:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)[0]
        
        # Slice ONLY the target sentence positions (adjusting for offset)
        # Shifted range: left_ids length minus 1 up to end index minus 1
        target_log_probs = token_log_probs[target_start_idx - 1 : target_end_idx - 1]
        
        cond_fwd_ll = target_log_probs.mean().item()

    return cond_fwd_ll


# =========================================================
# 2. CONTEXTUAL RIGHT-CONTEXT INFORMATION GAIN
# =========================================================
def compute_contextual_right_gain(left_context, target_sentence, right_context, bidir_model, bidir_tokenizer, device="cpu"):
    """
    Computes Information Gain for target_sentence tokens conditioned on full abstract context vs left-only context.
    """
    bidir_model.eval()
    
    # Encode components as token ID spans
    left_ids = bidir_tokenizer.encode(left_context, add_special_tokens=False)
    target_ids = bidir_tokenizer.encode(" " + target_sentence.strip(), add_special_tokens=False)
    right_ids = bidir_tokenizer.encode(" " + right_context.strip(), add_special_tokens=False)
    
    bos = [bidir_tokenizer.bos_token_id] if bidir_tokenizer.bos_token_id is not None else []
    eos = [bidir_tokenizer.eos_token_id] if bidir_tokenizer.eos_token_id is not None else []
    
    full_token_ids = torch.tensor(bos + left_ids + target_ids + right_ids + eos, device=device)
    mask_id = bidir_tokenizer.mask_token_id
    
    target_start = len(bos) + len(left_ids)
    target_end = target_start + len(target_ids)
    
    if len(target_ids) < 2:
        return np.nan

    gains = []
    
    for idx in range(target_start, target_end):
        orig_token_id = full_token_ids[idx].item()
        
        # --- Left-Only Context: Mask target AND everything to the right ---
        left_only_ids = full_token_ids.clone()
        left_only_ids[idx:] = mask_id
        
        # --- Full Abstract Context: Mask ONLY target token ---
        full_context_ids = full_token_ids.clone()
        full_context_ids[idx] = mask_id
        
        batch_inputs = torch.stack([left_only_ids, full_context_ids]).to(device)
        
        with torch.no_grad():
            logits = bidir_model(batch_inputs).logits
            log_probs = F.log_softmax(logits, dim=-1)
            
            left_log_prob = log_probs[0, idx, orig_token_id].item()
            full_log_prob = log_probs[1, idx, orig_token_id].item()
            
            gains.append(full_log_prob - left_log_prob)
            
    return np.mean(gains)


# =========================================================
# 3. DATA PROCESSING PIPELINE WITH ABSTRACT CONTEXT
# =========================================================
def process_contextual_experiment(data, model, tokenizer, mode="right_gain", device="cpu"):
    """
    Processes dataset containing left_context, target_sentence, and right_context.
    """
    results = []
    print(f"Processing in-context dataset using mode: {mode}...")
    
    for entry in tqdm(data):
        left_ctx = entry.get("left_context", "")
        target_s = entry["target_sentence"]
        right_ctx = entry.get("right_context", "")
        label_str = "AI" if entry["label"] == 1 else "Human"
        
        if mode == "right_gain":
            score = compute_contextual_right_gain(left_ctx, target_s, right_ctx, model, tokenizer, device=device)
        elif mode == "causal_ll":
            score = compute_contextual_causal_ll(left_ctx, target_s, model, tokenizer, device=device)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        results.append({
            "target_sentence": target_s,
            "label": label_str,
            "score": score
        })
        
    return pd.DataFrame(results)


# =========================================================
# 4. EXECUTION
# =========================================================
if __name__ == "__main__":
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Running experiment on device: {device}")
    
    # Load RobBERT (Dutch MLM)
    model_name = "pdelobelle/robbert-v2-dutch-base"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

    # Helper snippet to format records with context
    dataset_with_context = []
    df = pd.read_parquet(Path('/home/gderijck/internship/data/gold/llm_added.parquet'))
    for idx, row in df.iterrows():
        # If full abstract is available as a list of sentences [S_0, S_1, ..., S_n]
        abstract_sentences = row["abstract_sentence"] # list of sentences
        
        # Target sentence index i
        target_idx = random.randint(0, len(abstract_sentences) - 1)
        
        left_ctx = " ".join(abstract_sentences[:target_idx])
        right_ctx = " ".join(abstract_sentences[target_idx + 1:])
        
        # Human Target (Label 0)
        dataset_with_context.append({
            "left_context": left_ctx,
            "target_sentence": abstract_sentences[target_idx],
            "right_context": right_ctx,
            "label": 0
        })
        
        # AI Substituted Target (Label 1 from one of the _single columns)
        ai_sentence = row["gpt4_single"]
        dataset_with_context.append({
            "left_context": left_ctx,
            "target_sentence": ai_sentence,
            "right_context": right_ctx,
            "label": 1
        })

    results_df = process_contextual_experiment(
        data=dataset_with_context, 
        model=model, 
        tokenizer=tokenizer, 
        mode="right_gain", 
        device=device
    )

    # Evaluate AUROC
    binary_labels = (results_df["label"] == "AI").astype(int)
    valid_mask = ~results_df["score"].isna()
    
    auc = roc_auc_score(binary_labels[valid_mask], results_df.loc[valid_mask, "score"])
    auc_final = max(auc, 1 - auc)
    
    print("\n" + "="*50)
    print("   IN-CONTEXT SENTENCE EVALUATION PERFORMANCE   ")
    print("="*50)
    print(f"AUROC [In-Context Score]: {auc_final:.4f}")
    print("="*50)