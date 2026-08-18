import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns


def compute_sentence_context_score(left_context, target_sentence, right_context, model, tokenizer, metric_type="right_gain", device="cpu"):
    """
    Calculates the in-context score for a single target sentence embedded inside left and right context.
    """
    model.eval()
    
    if metric_type == "right_gain":
        # Using RobBERT (Masked LM) Right-Context Gain
        left_ids = tokenizer.encode(left_context, add_special_tokens=False)
        target_ids = tokenizer.encode(" " + target_sentence.strip(), add_special_tokens=False)
        right_ids = tokenizer.encode(" " + right_context.strip(), add_special_tokens=False)
        
        bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        
        if len(target_ids) < 2:
            return 0.0

        full_token_ids = torch.tensor(bos + left_ids + target_ids + right_ids + eos, device=device)
        mask_id = tokenizer.mask_token_id
        
        target_start = len(bos) + len(left_ids)
        target_end = target_start + len(target_ids)
        
        gains = []
        for idx in range(target_start, target_end):
            orig_token_id = full_token_ids[idx].item()
            
            # Left-Only Context
            left_only_ids = full_token_ids.clone()
            left_only_ids[idx:] = mask_id
            
            # Full Abstract Context
            full_context_ids = full_token_ids.clone()
            full_context_ids[idx] = mask_id
            
            batch_inputs = torch.stack([left_only_ids, full_context_ids]).to(device)
            
            with torch.no_grad():
                logits = model(batch_inputs).logits
                log_probs = F.log_softmax(logits, dim=-1)
                
                left_log_prob = log_probs[0, idx, orig_token_id].item()
                full_log_prob = log_probs[1, idx, orig_token_id].item()
                gains.append(full_log_prob - left_log_prob)
                
        return np.mean(gains)

    elif metric_type == "causal_ll":
        # Using Causal LM Log-Likelihood P(target_sentence | left_context)
        left_ids = tokenizer.encode(left_context, add_special_tokens=True)
        target_ids = tokenizer.encode(" " + target_sentence.strip(), add_special_tokens=False)
        
        if len(target_ids) < 2:
            return 0.0

        full_ids = torch.tensor([left_ids + target_ids], device=device)
        target_start_idx = len(left_ids)
        target_end_idx = len(left_ids) + len(target_ids)

        with torch.no_grad():
            outputs = model(full_ids)
            logits = outputs.logits[:, :-1, :]
            labels = full_ids[:, 1:]
            
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)[0]
            target_log_probs = token_log_probs[target_start_idx - 1 : target_end_idx - 1]
            
            return target_log_probs.mean().item()


def scan_abstract_for_substitutions(abstract_sentences, model, tokenizer, metric_type="right_gain", device="cpu"):
    """
    Scans an entire abstract sentence-by-sentence, computes in-context scores,
    and calculates relative Z-score anomaly profiles to detect substituted sentences.
    """
    n_sents = len(abstract_sentences)
    if n_sents < 3:
        return None  # Need at least 3 sentences for robust Z-score profiling

    raw_scores = []

    # 1. Compute in-context score for every sentence position
    for i in range(n_sents):
        target_sentence = abstract_sentences[i]
        left_context = " ".join(abstract_sentences[:i])
        right_context = " ".join(abstract_sentences[i+1:])
        
        score = compute_sentence_context_score(
            left_context=left_context,
            target_sentence=target_sentence,
            right_context=right_context,
            model=model,
            tokenizer=tokenizer,
            metric_type=metric_type,
            device=device
        )
        raw_scores.append(score)

    raw_scores = np.array(raw_scores)

    # 2. Self-Calibration: Compute within-abstract Z-Scores
    mean_score = np.mean(raw_scores)
    std_score = np.std(raw_scores) + 1e-8  # Avoid division by zero
    z_scores = (raw_scores - mean_score) / std_score

    # 3. Identify Top Outlier Sentence
    # For Right-Gain / Perplexity, AI sentences manifest as anomaly spikes or drops
    abs_z_scores = np.abs(z_scores)
    predicted_sub_idx = int(np.argmax(abs_z_scores))

    return {
        "raw_scores": raw_scores,
        "z_scores": z_scores,
        "predicted_idx": predicted_sub_idx,
        "max_z_score": abs_z_scores[predicted_sub_idx]
    }

def evaluate_substitution_localization(abstract_records, model, tokenizer, metric_type="right_gain", device="cpu"):
    """
    Evaluates localization accuracy across a dataset of abstracts.
    
    `abstract_records` format:
    [
    {
        "abstract_sentences": ["S0...", "S1_AI...", "S2...", "S3..."],
        "true_substituted_idx": 1
    }, ...
    ]
    """
    correct_localizations = 0
    total_valid_abstracts = 0
    all_sentence_z_scores = []
    all_sentence_labels = []

    for record in abstract_records:
        sentences = record["abstract_sentences"]
        true_idx = record["true_substituted_idx"]
        
        scan_res = scan_abstract_for_substitutions(sentences, model, tokenizer, metric_type=metric_type, device=device)
        if scan_res is None:
            continue
            
        total_valid_abstracts += 1
        pred_idx = scan_res["predicted_idx"]
        
        # Check if the model correctly identified the exact substituted sentence
        if pred_idx == true_idx:
            correct_localizations += 1
            
        # Collect token/sentence level labels for overall AUROC
        for i, z in enumerate(scan_res["z_scores"]):
            all_sentence_z_scores.append(abs(z))
            all_sentence_labels.append(1 if i == true_idx else 0)

    top1_accuracy = correct_localizations / total_valid_abstracts
    sentence_auroc = roc_auc_score(all_sentence_labels, all_sentence_z_scores)
    
    print("\n" + "="*50)
    print("   UNKNOWN SENTENCE LOCALIZATION PERFORMANCE   ")
    print("="*50)
    print(f"Top-1 Sentence Localization Accuracy : {top1_accuracy * 100:.2f}%")
    print(f"Overall Sentence-Level AUROC (Z-Score): {sentence_auroc:.4f}")
    print("="*50)