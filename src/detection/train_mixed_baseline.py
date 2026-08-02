# train_baseline.py
import argparse
import ast
import json
import os
import random
import re
import string
import unicodedata
from collections import Counter
from datetime import datetime

import joblib
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

dutch_stopwords = stopwords.words('dutch')
DEFAULT_MODELS = ['qwen3.5:4b', 'qwen3.6:27b', 'gemma4:e4b', 'gemma4:26b']


# ==========================================
# 1. Utilities & Normalization
# ==========================================
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    try:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
    except Exception:
        pass
    text = unicodedata.normalize('NFKC', text)
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    return " ".join(text.split())


def safe_parse_list(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, tuple, np.ndarray, set)):
        return list(val)
    if isinstance(val, str):
        val_str = val.strip()
        if val_str.startswith('[') and val_str.endswith(']'):
            try:
                parsed = ast.literal_eval(val_str)
                return list(parsed) if isinstance(parsed, (list, tuple, np.ndarray, set)) else [str(parsed)]
            except (ValueError, SyntaxError):
                pass
        elif val_str:
            return [val_str]
    return []


# ==========================================
# 2. Synthetic Dataset Generator
# ==========================================
def generate_synthetic_dataset(raw_df, selected_models, llm_ratio=4, random_state=42):
    local_rng = random.Random(random_state)
    records = []

    print("Generating Synthetic Mixed-Authorship Sentences...")

    for doc_idx, (_, row) in enumerate(raw_df.iterrows()):
        source = row.get('source', 'unknown')
        parent_id = str(row['_id']) if '_id' in row and pd.notna(row['_id']) else f"parent_{doc_idx}"

        raw_human = safe_parse_list(row.get('abstract_sentence', []))
        human_sents = [normalize_text(s) for s in raw_human if normalize_text(s)]
        if len(human_sents) < 3:
            continue

        # Single Sentence Injections & Blocks
        for model in selected_models:
            col_var = f"{model}_single"
            if col_var in row:
                raw_ai = safe_parse_list(row[col_var])
                ai_sents = [normalize_text(s) for s in raw_ai if normalize_text(s)]
                if not ai_sents:
                    continue

                # Scenario: Injected Document
                mixed_sents = list(human_sents)
                labels = [0] * len(human_sents)
                
                inject_pos = local_rng.randint(1, len(human_sents) - 1)
                ai_snippet = ai_sents[:min(2, len(ai_sents))]
                
                mixed_sents[inject_pos:inject_pos] = ai_snippet
                for k in range(len(ai_snippet)):
                    labels.insert(inject_pos + k, 1)

                # Build Sliding Windows
                n = len(mixed_sents)
                doc_text = " ".join(mixed_sents)
                for i in range(n):
                    w1 = mixed_sents[i]
                    w3 = " ".join(mixed_sents[max(0, i - 1):min(n, i + 2)])
                    w5 = " ".join(mixed_sents[max(0, i - 2):min(n, i + 3)])

                    records.append({
                        'doc_id': f"{parent_id}_{model}_inj",
                        'parent_doc_id': parent_id,
                        'source': source,
                        'sentence_idx': i,
                        'label': labels[i],
                        'text_w1': w1,
                        'text_w3': w3,
                        'text_w5': w5,
                        'doc_text': doc_text
                    })

    df = pd.DataFrame(records)
    print(f"-> Generated {len(df)} total sentence records across synthetic documents.")
    return df


# ==========================================
# 3. Simple Sequence Smoothing
# ==========================================
def smooth_probabilities(df_eval, raw_probs, alpha=0.75):
    df = df_eval.copy().reset_index(drop=True)
    df['prob'] = raw_probs
    smoothed = np.zeros(len(df))

    for doc_id, group in df.groupby('doc_id'):
        group_sorted = group.sort_values('sentence_idx')
        idxs = group_sorted.index.values
        probs = group_sorted['prob'].values
        n = len(probs)

        if n <= 2 or alpha >= 1.0:
            smoothed[idxs] = probs
            continue

        fwd, bwd = np.zeros(n), np.zeros(n)
        fwd[0], bwd[-1] = probs[0], probs[-1]

        for i in range(1, n):
            fwd[i] = alpha * probs[i] + (1 - alpha) * fwd[i - 1]
            bwd[n - 1 - i] = alpha * probs[n - 1 - i] + (1 - alpha) * bwd[n - i]

        ema = 0.5 * (fwd + bwd)
        smoothed[idxs] = np.maximum(probs * 0.85, ema)

    return smoothed


# ==========================================
# 4. Main Clean Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Baseline Mixed-Authorship Detector")
    parser.add_argument('--data_path', type=str, required=True, help="Path to parquet dataset")
    parser.add_argument('--features', type=str, choices=['word_w1', 'word_w3', 'word_char_w3'], default='word_char_w3')
    parser.add_argument('--C', type=float, default=1.0, help="Logistic Regression C penalty")
    parser.add_argument('--alpha', type=float, default=0.75, help="Sequence smoothing alpha")
    args = parser.parse_args()

    print(f"Loading raw data: {args.data_path}")
    raw_df = pd.read_parquet(args.data_path)

    # 80/20 Document-Level Split
    train_raw, test_raw = train_test_split(raw_df, test_size=0.20, random_state=42)

    # Generate Synthetic Data
    train_df = generate_synthetic_dataset(train_raw, DEFAULT_MODELS)
    test_df = generate_synthetic_dataset(test_raw, DEFAULT_MODELS)

    # Feature Extraction
    print(f"\nExtracting features using mode: '{args.features}'...")
    text_col = 'text_w3' if 'w3' in args.features else 'text_w1'

    matrices_tr, matrices_te = [], []

    # 1. Word TF-IDF
    vec_word = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True, stop_words=dutch_stopwords)
    matrices_tr.append(vec_word.fit_transform(train_df[text_col]))
    matrices_te.append(vec_word.transform(test_df[text_col]))

    # 2. Char TF-IDF (Optional)
    if 'char' in args.features:
        vec_char = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=10000, sublinear_tf=True)
        matrices_tr.append(vec_char.fit_transform(train_df[text_col]))
        matrices_te.append(vec_char.transform(test_df[text_col]))

    X_train = hstack(matrices_tr).tocsr()
    X_test = hstack(matrices_te).tocsr()
    y_train = train_df['label'].values
    y_test = test_df['label'].values

    # Train Logistic Regression Model
    print(f"\nTraining Logistic Regression (C={args.C})...")
    clf = LogisticRegression(C=args.C, class_weight='balanced', max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Predict Probabilities
    raw_test_probs = clf.predict_proba(X_test)[:, 1]
    smoothed_test_probs = smooth_probabilities(test_df, raw_test_probs, alpha=args.alpha)

    # Calibrate 1% FPR Threshold
    raw_tr_probs = clf.predict_proba(X_train)[:, 1]
    smoothed_tr_probs = smooth_probabilities(train_df, raw_tr_probs, alpha=args.alpha)
    fpr, tpr, thresholds = roc_curve(y_train, smoothed_tr_probs)
    valid = np.where(fpr <= 0.01)[0]
    opt_threshold = float(np.clip(thresholds[valid[-1]], 0.0, 1.0)) if len(valid) > 0 else 0.5

    # Evaluate Metrics
    preds = (smoothed_test_probs >= opt_threshold).astype(int)
    roc_auc = roc_auc_score(y_test, smoothed_test_probs)
    rec_at_1fp = recall_score(y_test, preds, pos_label=1)

    print("\n" + "=" * 50)
    print("           BASELINE EVALUATION RESULTS           ")
    print("=" * 50)
    print(f"Calibrated 1% FPR Threshold: {opt_threshold:.4f}")
    print(f"Recall @ 1% FPR (TPR):       {rec_at_1fp:.4f}")
    print(f"Sentence-Level ROC-AUC:      {roc_auc:.4f}\n")
    print(classification_report(y_test, preds, digits=4))


if __name__ == "__main__":
    main()