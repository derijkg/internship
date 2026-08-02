# train_mixed_authorship.py
import argparse
import ast
import hashlib
import json
import os
import random
import re
import shutil
import string
import unicodedata
from collections import Counter
from datetime import datetime

import joblib
import nltk
import numpy as np
import optuna
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

# Quietly ensure NLTK Dutch Stopwords are available
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

dutch_stopwords = stopwords.words('dutch')

# Lazy-loaded NLP and Stopwords objects
_nlp = None
_dutch_stopwords_lemmatized = None

DEFAULT_MODELS = ['qwen3.5:4b', 'qwen3.6:27b', 'gemma4:e4b', 'gemma4:26b']
DUTCH_TRANSITIONS = {
    "echter", "bovendien", "daarnaast", "desalniettemin", "kortom",
    "tevens", "daardoor", "derhalve", "bijgevolg", "namelijk", "hoewel"
}


# ==========================================
# 1. NLP & Normalization Utilities
# ==========================================
def get_nlp():
    """Lazy loads spaCy model only when needed to save memory."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("nl_core_news_sm", disable=["parser", "ner"])
        except Exception:
            import spacy.cli
            spacy.cli.download('nl_core_news_sm')
            _nlp = spacy.load("nl_core_news_sm", disable=["parser", "ner"])
    return _nlp


def get_dutch_stopwords_lemmatized():
    """Lazy loads and computes lemmatized Dutch stopwords."""
    global _dutch_stopwords_lemmatized
    if _dutch_stopwords_lemmatized is None:
        nlp_model = get_nlp()
        _dutch_stopwords_lemmatized = list(set([
            token.lemma_.lower() for doc in nlp_model.pipe(dutch_stopwords) for token in doc
        ]))
    return _dutch_stopwords_lemmatized


def strip_markdown(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(?<!\w)(\*|_)(.*?)\1(?!\w)', r'\2', text)
    text = re.sub(r'(~~)(.*?)\1', r'\2', text)
    text = re.sub(r'(`)(.*?)\1', r'\2', text)
    text = re.sub(r'^\s*[#>]+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    return text


def clean_html_markdown(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
    except Exception:
        pass
    return strip_markdown(text)


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = clean_html_markdown(text)
    text = unicodedata.normalize('NFKC', text)
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    text = text.replace('—', '-').replace('–', '-')
    return " ".join(text.split())


def safe_parse_list(val):
    """Safely parses list-like structures (lists, numpy arrays, JSON/AST stringified lists)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    # Convert NumPy arrays, tuples, and sets to standard Python lists
    if isinstance(val, (list, tuple, np.ndarray, set)):
        return list(val)
    if isinstance(val, str):
        val_str = val.strip()
        if val_str.startswith('[') and val_str.endswith(']'):
            try:
                parsed = ast.literal_eval(val_str)
                return list(parsed) if isinstance(parsed, (list, tuple, np.ndarray, set)) else [str(parsed)]
            except (ValueError, SyntaxError):
                try:
                    parsed = json.loads(val_str)
                    return list(parsed) if isinstance(parsed, (list, tuple, np.ndarray, set)) else [str(parsed)]
                except Exception:
                    pass
        elif val_str:
            return [val_str]
    return []


def pre_lemmatize_dataset(df, text_column='text', target_column=None, n_process=1, cache_dir="./.dataset_cache"):
    """Lemmatizes dataset sequentially or in parallel, with automatic hash-based disk caching."""
    df = df.copy()
    if target_column is None:
        target_column = f"{text_column}_lemmatized" if text_column != 'text' else 'text_lemmatized'

    os.makedirs(cache_dir, exist_ok=True)

    # Generate a unique MD5 hash based on dataset length, column name, and first 100 record IDs
    if 'doc_id' in df.columns:
        sample_ids = df['doc_id'].astype(str).iloc[:100].tolist()
    else:
        sample_ids = [str(i) for i in df.index[:100]]

    hash_input = f"{len(df)}_{text_column}_" + "_".join(sample_ids)
    dataset_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
    cache_path = os.path.join(cache_dir, f"lem_{text_column}_{dataset_hash}.parquet")

    # 1. Check if cache exists
    if os.path.exists(cache_path):
        print(f"-> Found cached lemmatized column for '{text_column}'. Loading from '{cache_path}'...")
        try:
            cached_series = pd.read_parquet(cache_path)[target_column]
            if len(cached_series) == len(df):
                df[target_column] = cached_series.values
                return df
        except Exception as e:
            print(f"Warning: Cache load failed ({e}). Re-computing lemmatization...")

    # 2. Compute lemmatization if cache is missing
    print(f"Pre-lemmatizing {len(df)} records from '{text_column}' into '{target_column}'...")
    normalized_texts = [normalize_text(str(t)) for t in df[text_column]]
    nlp_model = get_nlp()
    docs = nlp_model.pipe(normalized_texts, batch_size=256, n_process=n_process)
    lemmatized_texts = [" ".join([token.lemma_ for token in doc if not token.is_punct]) for doc in docs]

    df[target_column] = lemmatized_texts

    # 3. Save to disk cache for future runs
    try:
        pd.DataFrame({target_column: lemmatized_texts}).to_parquet(cache_path)
        print(f"-> Successfully saved lemmatized column to cache: '{cache_path}'")
    except Exception as e:
        print(f"Warning: Could not write cache file ({e})")

    return df

# ==========================================
# 2. Stylometric & Delta Feature Extraction
# ==========================================
def calculate_ttr(words):
    return len(set(words)) / len(words) if words else 0.0


def calculate_hapax_ratio(words):
    if not words:
        return 0.0
    counts = Counter(words)
    return sum(1 for w, c in counts.items() if c == 1) / len(words)


def extract_raw_stylometrics(text, sentences):
    """Extracts a 12-dimensional vector of linguistic metrics."""
    words = re.findall(r'\w+', text.lower())
    total_chars = len(text)

    if not words or not sentences:
        return np.zeros(12)

    if len(sentences) <= 1:
        mean_sent_len = float(len(words))
        var_sent_len = 0.0
        burstiness = 0.0
    else:
        sent_lengths = [len(re.findall(r'\w+', s)) for s in sentences if len(re.findall(r'\w+', s)) > 0]
        mean_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
        var_sent_len = float(np.var(sent_lengths)) if sent_lengths else 0.0
        burstiness = (float(np.std(sent_lengths)) / mean_sent_len) if mean_sent_len > 0 else 0.0

    word_lengths = [len(w) for w in words]
    mean_word_len = float(np.mean(word_lengths))
    var_word_len = float(np.var(word_lengths))

    ttr = calculate_ttr(words)
    hapax_ratio = calculate_hapax_ratio(words)

    transition_count = sum(1 for w in words if w in DUTCH_TRANSITIONS)
    transition_ratio = transition_count / len(words)

    spaces_count = text.count(' ')
    double_spaces = text.count('  ')
    space_ratio = spaces_count / total_chars if total_chars > 0 else 0.0
    double_space_ratio = double_spaces / total_chars if total_chars > 0 else 0.0

    punc_count = sum(1 for c in text if c in string.punctuation)
    punc_ratio = punc_count / total_chars if total_chars > 0 else 0.0

    return np.array([
        mean_sent_len, var_sent_len, burstiness,
        mean_word_len, var_word_len,
        ttr, hapax_ratio,
        transition_ratio, space_ratio, double_space_ratio, punc_ratio,
        float(total_chars)
    ], dtype=np.float64)


def parse_sents_field(val):
    parsed = safe_parse_list(val)
    if isinstance(parsed, list):
        return parsed
    if isinstance(val, str):
        return [val]
    return []


def extract_sentence_stylometrics_with_delta(record):
    """Extracts local multi-scale stylometrics plus Relative Delta Stylometrics."""
    doc_text = str(record.get('doc_text', ''))
    doc_sentences = parse_sents_field(record.get('doc_sentences', []))
    doc_style = extract_raw_stylometrics(doc_text, doc_sentences)

    sents_w1 = parse_sents_field(record.get('sents_w1', []))
    sents_w3 = parse_sents_field(record.get('sents_w3', []))
    sents_w5 = parse_sents_field(record.get('sents_w5', []))

    w1_style = extract_raw_stylometrics(str(record.get('text_w1', '')), sents_w1)
    w3_style = extract_raw_stylometrics(str(record.get('text_w3', '')), sents_w3)
    w5_style = extract_raw_stylometrics(str(record.get('text_w5', '')), sents_w5)

    # Relative Delta Features (Window Style minus Document Mean Style)
    w1_delta = w1_style - doc_style
    w3_delta = w3_style - doc_style
    w5_delta = w5_style - doc_style

    return np.hstack([w1_style, w3_style, w5_style, w1_delta, w3_delta, w5_delta])


# ==========================================
# 3. Synthetic Mixed-Authorship Dataset Generator
# ==========================================
def generate_synthetic_mixed_dataset(raw_df, selected_models, llm_ratio=4, random_state=42):
    local_rng = random.Random(random_state)
    synthetic_docs = []

    print("Generating Synthetic Mixed-Authorship Dataset...")

    for doc_idx, (_, row) in enumerate(raw_df.iterrows()):
        source = row.get('source', 'unknown')
        
        # Use existing _id if present, otherwise fall back to doc_idx
        if '_id' in row and pd.notna(row['_id']):
            parent_doc_id = str(row['_id'])
        else:
            parent_doc_id = f"parent_{doc_idx}"

        raw_human_sents = []
        for col in ['abstract_sentence', 'abstract_sentences']:
            if col in row:
                parsed = safe_parse_list(row.get(col, []))
                if isinstance(parsed, list) and len(parsed) > 0:
                    raw_human_sents = parsed
                    break

        human_sents = [normalize_text(s) for s in raw_human_sents if normalize_text(s)]
        if len(human_sents) < 3:
            continue

        valid_models = []
        for model in selected_models:
            for col_var in [f"{model}_single", f"{model}_sentence", f"{model}_sentences"]:
                if col_var in row:
                    raw_ai = safe_parse_list(row[col_var])
                    if isinstance(raw_ai, list) and len(raw_ai) > 0:
                        valid_models.append((model, col_var))
                        break

        if not valid_models:
            continue

        sampled_models = local_rng.sample(valid_models, k=min(llm_ratio, len(valid_models)))

        # Scenario 1: Pure Human Document
        synthetic_docs.append({
            'doc_id': f"{parent_doc_id}_pure_human",
            'parent_doc_id': parent_doc_id,
            'source': source,
            'sentences': human_sents,
            'labels': [0] * len(human_sents),
            'scenario': 'pure_human'
        })

        for model_name, col_var in sampled_models:
            parsed_ai = safe_parse_list(row[col_var])
            ai_sents = [normalize_text(s) for s in parsed_ai if normalize_text(s)]

            if not ai_sents:
                continue

            # Scenario 2: Pure LLM Document
            synthetic_docs.append({
                'doc_id': f"{parent_doc_id}_{model_name}_pure_ai",
                'parent_doc_id': parent_doc_id,
                'source': source,
                'sentences': ai_sents,
                'labels': [1] * len(ai_sents),
                'scenario': 'pure_ai'
            })

            # Scenario 3: Single/Multi Sentence Injection
            mixed_sents_inj = list(human_sents)
            mixed_labels_inj = [0] * len(human_sents)

            inject_pos = local_rng.randint(1, len(human_sents) - 1)
            if len(ai_sents) > 2:
                start_idx = local_rng.randint(0, len(ai_sents) - 2)
                ai_snippet = ai_sents[start_idx : start_idx + local_rng.randint(1, 2)]
            else:
                ai_snippet = ai_sents

            mixed_sents_inj[inject_pos:inject_pos] = ai_snippet
            for k in range(len(ai_snippet)):
                mixed_labels_inj.insert(inject_pos + k, 1)

            synthetic_docs.append({
                'doc_id': f"{parent_doc_id}_{model_name}_injection",
                'parent_doc_id': parent_doc_id,
                'source': source,
                'sentences': mixed_sents_inj,
                'labels': mixed_labels_inj,
                'scenario': 'sentence_injection'
            })

            # Scenario 4: Paragraph / Block Substitution
            if len(human_sents) >= 4 and len(ai_sents) >= 2:
                mixed_sents_sub = list(human_sents)
                mixed_labels_sub = [0] * len(human_sents)

                sub_start = local_rng.randint(1, len(human_sents) - 2)
                sub_len = min(2, len(human_sents) - sub_start)

                for k in range(sub_len):
                    mixed_sents_sub[sub_start + k] = ai_sents[k % len(ai_sents)]
                    mixed_labels_sub[sub_start + k] = 1

                synthetic_docs.append({
                    'doc_id': f"{parent_doc_id}_{model_name}_substitution",
                    'parent_doc_id': parent_doc_id,
                    'source': source,
                    'sentences': mixed_sents_sub,
                    'labels': mixed_labels_sub,
                    'scenario': 'block_substitution'
                })

    print(f"-> Generated {len(synthetic_docs)} synthetic documents across 4 mixed-authorship scenarios.")
    return synthetic_docs


def build_multiscale_sentence_dataframe(synthetic_docs):
    records = []

    for doc in synthetic_docs:
        doc_id = doc['doc_id']
        parent_doc_id = doc['parent_doc_id']
        source = doc['source']
        scenario = doc['scenario']
        sents = doc['sentences']
        labels = doc['labels']
        n_sents = len(sents)
        doc_text = " ".join(sents)

        for i in range(n_sents):
            sents_w1 = [sents[i]]
            text_w1 = sents[i]

            start_3 = max(0, i - 1)
            end_3 = min(n_sents, i + 2)
            sents_w3 = sents[start_3:end_3]
            text_w3 = " ".join(sents_w3)

            start_5 = max(0, i - 2)
            end_5 = min(n_sents, i + 3)
            sents_w5 = sents[start_5:end_5]
            text_w5 = " ".join(sents_w5)

            records.append({
                'doc_id': doc_id,
                'parent_doc_id': parent_doc_id,
                'source': source,
                'scenario': scenario,
                'sentence_idx': i,
                'label': labels[i],
                'text': sents[i],
                'text_w1': text_w1,
                'text_w3': text_w3,
                'text_w5': text_w5,
                'sents_w1': sents_w1,
                'sents_w3': sents_w3,
                'sents_w5': sents_w5,
                'doc_text': doc_text,
                'doc_sentences': sents
            })

    return pd.DataFrame(records)


# ==========================================
# 4. Custom Feature Pipeline Builders
# ==========================================
class MultiScaleStylometricExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            records = X.to_dict('records')
        else:
            records = X
        features = [extract_sentence_stylometrics_with_delta(item) for item in records]
        return np.array(features)


class TextExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, key='text_w3'):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            if self.key in X.columns:
                return X[self.key].fillna('').astype(str).tolist()
            fallback_key = self.key.replace('_lemmatized', '')
            return X[fallback_key].fillna('').astype(str).tolist() if fallback_key in X.columns else [''] * len(X)
        elif isinstance(X, list):
            return [str(item.get(self.key, item.get(self.key.replace('_lemmatized', ''), ''))) for item in X]
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")


def get_dynamic_feature_pipeline(params, use_pre_lemmatized=True):
    features = []

    # Common Parameters for Word TF-IDF
    word_params = {
        'ngram_range': (params.get('word_min_ngram', 1), params.get('word_max_ngram', 2)),
        'max_features': params.get('word_max_features', 5000),
        'min_df': params.get('word_min_df', 2),
        'max_df': 0.95,
        'sublinear_tf': True,
        'stop_words': get_dutch_stopwords_lemmatized() if use_pre_lemmatized else None
    }

    # Common Parameters for Char TF-IDF
    char_params = {
        'analyzer': 'char',
        'ngram_range': (params.get('char_min_ngram', 2), params.get('char_max_ngram', 4)),
        'max_features': params.get('char_max_features', 5000),
        'min_df': params.get('char_min_df', 2),
        'max_df': 0.95,
        'sublinear_tf': True
    }

    # Dynamic Word TF-IDF Branches
    for win in ['w1', 'w3', 'w5']:
        if params.get(f'use_{win}_word', (win == 'w3')):
            key = f'text_{win}_lemmatized' if use_pre_lemmatized else f'text_{win}'
            features.append((f'word_{win}', Pipeline([
                ('extract', TextExtractor(key=key)),
                ('tfidf', TfidfVectorizer(**word_params))
            ])))

    # Dynamic Char TF-IDF Branches
    for win in ['w1', 'w3', 'w5']:
        if params.get(f'use_{win}_char', (win == 'w3')):
            features.append((f'char_{win}', Pipeline([
                ('extract', TextExtractor(key=f'text_{win}')),
                ('tfidf', TfidfVectorizer(**char_params))
            ])))

    # Safety Fallback: Default to W3 Word if Optuna disables all TF-IDF branches
    if not features:
        key = 'text_w3_lemmatized' if use_pre_lemmatized else 'text_w3'
        features.append(('word_w3', Pipeline([
            ('extract', TextExtractor(key=key)),
            ('tfidf', TfidfVectorizer(**word_params))
        ])))

    # Always Include Stylometrics + Relative Deltas
    features.append(('stylometrics', Pipeline([
        ('extractor', MultiScaleStylometricExtractor()),
        ('scaler', StandardScaler())
    ])))

    return FeatureUnion(features)


def get_cached_split_features(X_train_raw, X_val_raw, params, cache_dir="./.optuna_mixed_cache"):
    os.makedirs(cache_dir, exist_ok=True)

    def get_data_hash(X_raw):
        if isinstance(X_raw, pd.DataFrame):
            sample_ids = X_raw['doc_id'].astype(str).iloc[:100].tolist()
            length = len(X_raw)
        else:
            sample_ids = [str(x.get('doc_id', '')) for x in X_raw[:100]]
            length = len(X_raw)
        serialized = f"{length}_" + "_".join(sample_ids)
        return hashlib.md5(serialized.encode('utf-8')).hexdigest()

    train_hash = get_data_hash(X_train_raw)
    val_hash = get_data_hash(X_val_raw)

    train_sty_path = os.path.join(cache_dir, f"tr_sty_{train_hash}.joblib")
    val_sty_path = os.path.join(cache_dir, f"val_sty_{train_hash}_{val_hash}.joblib")

    # Word & Char TF-IDF Param Dictionaries
    word_params = {
        'ngram_range': (params.get('word_min_ngram', 1), params.get('word_max_ngram', 2)),
        'max_features': params.get('word_max_features', 5000),
        'min_df': params.get('word_min_df', 2),
        'max_df': 0.95,
        'sublinear_tf': True,
        'stop_words': get_dutch_stopwords_lemmatized()
    }

    char_params = {
        'analyzer': 'char',
        'ngram_range': (params.get('char_min_ngram', 2), params.get('char_max_ngram', 4)),
        'max_features': params.get('char_max_features', 5000),
        'min_df': params.get('char_min_df', 2),
        'max_df': 0.95,
        'sublinear_tf': True
    }

    matrices_tr = []
    matrices_va = []

    # 1. Dynamic Word TF-IDF Branches
    for win in ['w1', 'w3', 'w5']:
        if params.get(f'use_{win}_word', (win == 'w3')):
            vec_word = TfidfVectorizer(**word_params)
            ext_word = TextExtractor(key=f'text_{win}_lemmatized')
            matrices_tr.append(vec_word.fit_transform(ext_word.transform(X_train_raw)))
            matrices_va.append(vec_word.transform(ext_word.transform(X_val_raw)))

    # 2. Dynamic Char TF-IDF Branches
    for win in ['w1', 'w3', 'w5']:
        if params.get(f'use_{win}_char', (win == 'w3')):
            vec_char = TfidfVectorizer(**char_params)
            ext_char = TextExtractor(key=f'text_{win}')
            matrices_tr.append(vec_char.fit_transform(ext_char.transform(X_train_raw)))
            matrices_va.append(vec_char.transform(ext_char.transform(X_val_raw)))

    # Fallback if no TF-IDF window was selected
    if not matrices_tr:
        vec_word = TfidfVectorizer(**word_params)
        ext_word = TextExtractor(key='text_w3_lemmatized')
        matrices_tr.append(vec_word.fit_transform(ext_word.transform(X_train_raw)))
        matrices_va.append(vec_word.transform(ext_word.transform(X_val_raw)))

    # 3. Cached Stylometrics + Deltas
    if os.path.exists(train_sty_path) and os.path.exists(val_sty_path):
        X_tr_sty = joblib.load(train_sty_path)
        X_va_sty = joblib.load(val_sty_path)
    else:
        ext_sty = MultiScaleStylometricExtractor()
        scaler = StandardScaler()
        raw_tr = ext_sty.transform(X_train_raw)
        raw_va = ext_sty.transform(X_val_raw)

        X_tr_sty = csr_matrix(scaler.fit_transform(raw_tr))
        X_va_sty = csr_matrix(scaler.transform(raw_va))

        joblib.dump(X_tr_sty, train_sty_path)
        joblib.dump(X_va_sty, val_sty_path)

    matrices_tr.append(X_tr_sty)
    matrices_va.append(X_va_sty)

    X_tr_combined = hstack(matrices_tr).tocsr()
    X_va_combined = hstack(matrices_va).tocsr()

    return X_tr_combined, X_va_combined


# ==========================================
# 5. Classifier & Post-Processing Engine
# ==========================================
def get_classifier(kernel, c_val, gamma='scale', calibrate=False, groups_cv=None):
    if kernel == 'linear':
        base_clf = LinearSVC(C=c_val, random_state=42, class_weight='balanced', dual='auto', max_iter=5000)
    else:
        base_clf = SVC(C=c_val, kernel=kernel, gamma=gamma, random_state=42, class_weight='balanced', cache_size=2000)

    if calibrate:
        cv_splitter = StratifiedGroupKFold(n_splits=3) if groups_cv is not None else 3
        return CalibratedClassifierCV(estimator=base_clf, cv=cv_splitter, method='sigmoid')
    return base_clf


def apply_adaptive_sequence_smoothing(df_eval, raw_probs, alpha=0.6):
    df = df_eval.copy().reset_index(drop=True)
    df["raw_prob"] = raw_probs
    smoothed_probs = np.zeros(len(df))

    for doc_id, group in df.groupby("doc_id"):
        group_sorted = group.sort_values("sentence_idx")
        idxs = group_sorted.index.values
        probs = group_sorted["raw_prob"].values
        n = len(probs)

        if n <= 2:
            smoothed_probs[idxs] = probs
            continue

        fwd = np.zeros(n)
        bwd = np.zeros(n)
        fwd[0] = probs[0]
        bwd[-1] = probs[-1]

        for i in range(1, n):
            fwd[i] = alpha * probs[i] + (1 - alpha) * fwd[i - 1]
            bwd[n - 1 - i] = alpha * probs[n - 1 - i] + (1 - alpha) * bwd[n - i]

        ema_smooth = 0.5 * (fwd + bwd)

        # Allow raw high-confidence predictions to bypass smoothing suppression
        # Detections with confidence > 0.85 retain their original confidence
        smoothed_probs[idxs] = np.where(probs > 0.85, probs, ema_smooth)

    return smoothed_probs

# ==========================================
# 6. Optuna Hyperparameter Optimization
# ==========================================
def print_best_trial_callback(study, trial):
    if trial.state == optuna.trial.TrialState.COMPLETE:
        best = study.best_trial
        print(f"-> [Optuna Progress] Current Best: Trial {best.number} | Value ({study.direction.name}): {best.value:.4f}")


def optimize_mixed_detector_with_optuna(
        train_df, kernel_choice='linear', trials=30, score_metric='f1',
        tuning_sample_size=9000, study_name=None, n_jobs_optuna=1, reset_study=False,
        args=None
):
    print("\n--- Optuna Mixed-Authorship Tuning Initialized (3-Fold Group CV) ---")

    db_path = "sqlite:///optuna_mixed.db?timeout=60"
    if study_name is None:
        study_name = f"mixed_detector_{kernel_choice}_{score_metric}"

    if len(train_df) > tuning_sample_size:
        print(f"Subsampling document clusters down to ~{tuning_sample_size} sentences for tuning...")
        unique_parents = train_df['parent_doc_id'].unique()
        rng = np.random.RandomState(42)
        shuffled_parents = rng.permutation(unique_parents)

        selected_parents = []
        sentence_count = 0
        for parent_id in shuffled_parents:
            p_len = (train_df['parent_doc_id'] == parent_id).sum()
            selected_parents.append(parent_id)
            sentence_count += p_len
            if sentence_count >= tuning_sample_size:
                break

        train_sub = train_df[train_df['parent_doc_id'].isin(selected_parents)].copy().reset_index(drop=True)
    else:
        train_sub = train_df.copy().reset_index(drop=True)

    X_train_raw_all = train_sub.to_dict(orient='records')
    y_train_all = train_sub['label'].values
    groups_all = train_sub['parent_doc_id'].values

    sgkf = StratifiedGroupKFold(n_splits=3)

    # Allowed window sets passed from CLI
    word_win_allowed = args.word_windows if args and hasattr(args, 'word_windows') else ['w1', 'w3']
    char_win_allowed = args.char_windows if args and hasattr(args, 'char_windows') else ['w3']
    tune_windows = args.tune_windows if args and hasattr(args, 'tune_windows') else False

    def objective(trial):
        # 1. Alpha Parameter (Fixed or Tuned)
        if args and args.fixed_alpha is not None:
            alpha = args.fixed_alpha
        else:
            alpha = round(trial.suggest_float('alpha', 0.60, 0.90, step=0.05), 2)

        # 2. SVM Penalty C Parameter (Fixed or Tuned)
        if args and args.fixed_c is not None:
            c_val = args.fixed_c
        else:
            c_val = trial.suggest_float('C', 1e-2, 1e2, log=True)

        # 3. Dynamic or Fixed Window Selections
        trial_params = {
            'alpha': alpha,
            'C': c_val,
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']) if kernel_choice == 'all' else kernel_choice,
            'word_min_ngram': 1,
            'word_max_ngram': trial.suggest_int('word_max_ngram', 1, 3),
            'word_max_features': trial.suggest_int('word_max_features', 3000, 10000, step=1000),
            'word_min_df': trial.suggest_int('word_min_df', 1, 5),
            'char_min_ngram': trial.suggest_int('char_min_ngram', 2, 3),
            'char_max_ngram': trial.suggest_int('char_max_ngram', 3, 5),
            'char_max_features': trial.suggest_int('char_max_features', 3000, 10000, step=1000),
            'char_min_df': trial.suggest_int('char_min_df', 1, 5),
        }

        # Assign window flags based on CLI choices and tune_windows flag
        for win in ['w1', 'w3', 'w5']:
            if win in word_win_allowed:
                trial_params[f'use_{win}_word'] = trial.suggest_categorical(f'use_{win}_word', [True, False]) if tune_windows else True
            else:
                trial_params[f'use_{win}_word'] = False

            if win in char_win_allowed:
                trial_params[f'use_{win}_char'] = trial.suggest_categorical(f'use_{win}_char', [True, False]) if tune_windows else True
            else:
                trial_params[f'use_{win}_char'] = False

        # Safety Fallback: Ensure at least one word window is active
        if not any(trial_params.get(f'use_{w}_word') for w in ['w1', 'w3', 'w5']):
            trial_params['use_w3_word'] = True

        # Evaluate Trial Across 3 Group Folds
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train_raw_all, y_train_all, groups=groups_all)):
            X_tr_raw = [X_train_raw_all[i] for i in train_idx]
            X_va_raw = [X_train_raw_all[i] for i in val_idx]
            y_tr_fold = y_train_all[train_idx]
            y_va_fold = y_train_all[val_idx]

            X_tr, X_va = get_cached_split_features(X_tr_raw, X_va_raw, trial_params)

            clf = get_classifier(kernel=trial_params['kernel'], c_val=trial_params['C'], calibrate=False)
            clf.fit(X_tr, y_tr_fold)

            decision_scores = clf.decision_function(X_va)
            val_probs = 1.0 / (1.0 + np.exp(-decision_scores))

            val_df_fold = pd.DataFrame(X_va_raw)
            smoothed_val_probs = apply_adaptive_sequence_smoothing(val_df_fold, val_probs, alpha=alpha)

            if score_metric == 'set_fp':
                fpr, tpr, _ = roc_curve(y_va_fold, smoothed_val_probs)
                valid = np.where(fpr <= 0.01)[0]
                fold_score = tpr[valid[-1]] if len(valid) > 0 else 0.0
            else:
                preds = (smoothed_val_probs >= 0.50).astype(int)
                fold_score = f1_score(y_va_fold, preds, pos_label=1, zero_division=0)

            fold_scores.append(fold_score)

            intermediate_mean = float(np.mean(fold_scores))
            trial.report(intermediate_mean, step=fold)
            if trial.should_prune():
                print(f"-> [Trial {trial.number}] PRUNED at Fold {fold + 1}")
                raise optuna.TrialPruned()

        mean_score = float(np.mean(fold_scores))
        print(f"-> [Trial {trial.number}] Mean 3-Fold Score: {mean_score:.4f}")
        return mean_score

    if reset_study:
        try:
            optuna.delete_study(study_name=study_name, storage=db_path)
        except Exception:
            pass

    study = optuna.create_study(
        study_name=study_name, 
        storage=db_path, 
        direction='maximize', 
        load_if_exists=True, 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
    )

    # 4. Enqueue Strong Baseline Trial (Warm-Start Optuna)
    baseline_trial = {
        'alpha': 0.75,
        'C': 1.0,
        'word_max_ngram': 2,
        'word_max_features': 6000,
        'word_min_df': 2,
        'char_min_ngram': 2,
        'char_max_ngram': 4,
        'char_max_features': 6000,
        'char_min_df': 2,
    }
    if tune_windows:
        for w in ['w1', 'w3', 'w5']:
            if w in word_win_allowed: baseline_trial[f'use_{w}_word'] = True
            if w in char_win_allowed: baseline_trial[f'use_{w}_char'] = True

    try:
        study.enqueue_trial(baseline_trial)
        print("-> Enqueued strong baseline trial (Alpha=0.75, C=1.0, W1+W3 Word) for Trial 0 warm-start.")
    except Exception:
        pass

    study.optimize(objective, n_trials=trials, n_jobs=n_jobs_optuna, callbacks=[print_best_trial_callback])

    return study.best_params

# ==========================================
# 7. Experiment Registry & Evaluation
# ==========================================
def evaluate_mixed_authorship_metrics(test_df, y_test, smoothed_probs, threshold):
    preds = (smoothed_probs >= threshold).astype(int)

    sent_precision = precision_score(y_test, preds, pos_label=1, zero_division=0)
    sent_recall = recall_score(y_test, preds, pos_label=1, zero_division=0)
    sent_f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test, smoothed_probs)

    df_eval = test_df.copy().reset_index(drop=True)
    df_eval['pred'] = preds

    ious = []
    ratio_errors = []

    for doc_id, group in df_eval.groupby('doc_id'):
        y_true_doc = group['label'].values
        y_pred_doc = group['pred'].values

        intersection = np.sum((y_true_doc == 1) & (y_pred_doc == 1))
        union = np.sum((y_true_doc == 1) | (y_pred_doc == 1))
        iou = (intersection / union) if union > 0 else 1.0
        ious.append(iou)

        true_ratio = np.mean(y_true_doc)
        pred_ratio = np.mean(y_pred_doc)
        ratio_errors.append(abs(true_ratio - pred_ratio))

    mean_iou = float(np.mean(ious))
    mean_ratio_mae = float(np.mean(ratio_errors))

    return {
        'sent_precision_ai': sent_precision,
        'sent_recall_ai': sent_recall,
        'sent_f1_ai': sent_f1,
        'sentence_roc_auc': roc_auc,
        'span_iou': mean_iou,
        'ai_ratio_mae': mean_ratio_mae,
        'preds': preds
    }


def log_experiment_to_registry(record_dict, registry_path="mixed_authorship_experiments.csv"):
    df_new = pd.DataFrame([record_dict])
    if os.path.exists(registry_path):
        try:
            df_existing = pd.read_csv(registry_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined.to_csv(registry_path, index=False)
    print(f"-> Experiment record appended to '{registry_path}'")


# ==========================================
# 8. Main Orchestrator
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Multi-Scale Mixed-Authorship LLM Segment Detector")

    parser.add_argument('--data_path', type=str, required=True, help="Path to raw parquet/csv dataset")
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS, help="LLM models to include")
    parser.add_argument('--source', nargs='+', default=['UG', 'SB', 'HBO'], help="Sources to isolate")
    parser.add_argument('--llm_ratio', type=int, default=4, help="Number of LLM models to sample per document")
    parser.add_argument('--study_name', type=str, default="mixed_detector_exp1", help="Name for study and saved .pkl model")

    parser.add_argument('--kernel', type=str, choices=['linear', 'rbf', 'all'], default='linear', help="SVM kernel choice")
    parser.add_argument('--trials', type=int, default=200, help="Optuna trials")
    parser.add_argument('--tuning_sample_size', type=int, default=9000, help="Subsample size for Optuna tuning")
    parser.add_argument('--score', type=str, choices=['f1', 'set_fp'], default='set_fp', help="Optimization metric")
    parser.add_argument('--reset_study', action='store_true', help="Reset Optuna database entry")

    parser.add_argument('--word_windows', nargs='+', default=['w1', 'w3'], choices=['w1', 'w3', 'w5'], help="Word TF-IDF windows to use")
    parser.add_argument('--char_windows', nargs='+', default=['w3'], choices=['w1', 'w3', 'w5'], help="Char TF-IDF windows to use")
    parser.add_argument('--tune_windows', action='store_true', help="Let Optuna search over window combinations")
    parser.add_argument('--fixed_alpha', type=float, default=None, help="Fix alpha smoothing factor (e.g. 0.75)")
    parser.add_argument('--fixed_c', type=float, default=None, help="Fix SVM C parameter (e.g. 1.0)")

    args = parser.parse_args()

    print(f"Loading dataset from: {args.data_path}")
    raw_df = pd.read_csv(args.data_path) if args.data_path.endswith('.csv') else pd.read_parquet(args.data_path)

    if args.source and 'source' in raw_df.columns:
        raw_df = raw_df[raw_df['source'].isin(args.source)].copy()

    # Split Raw Documents 80% Train, 20% Test
    train_raw, test_raw = train_test_split(raw_df, test_size=0.20, random_state=42)

    # Synthetic Dataset Generation
    train_docs = generate_synthetic_mixed_dataset(train_raw, selected_models=args.models, llm_ratio=args.llm_ratio)
    test_docs = generate_synthetic_mixed_dataset(test_raw, selected_models=args.models, llm_ratio=args.llm_ratio)

    train_df = build_multiscale_sentence_dataframe(train_docs)
    test_df = build_multiscale_sentence_dataframe(test_docs)

    # Pre-lemmatize all multi-scale text windows
    for col in ['text_w1', 'text_w3', 'text_w5']:
        train_df = pre_lemmatize_dataset(train_df, text_column=col, target_column=f"{col}_lemmatized")
        test_df = pre_lemmatize_dataset(test_df, text_column=col, target_column=f"{col}_lemmatized")

    # Run Optuna Hyperparameter Tuning
    best_params = optimize_mixed_detector_with_optuna(
        train_df, kernel_choice=args.kernel, trials=args.trials, score_metric=args.score,
        tuning_sample_size=args.tuning_sample_size, study_name=args.study_name, 
        reset_study=args.reset_study, args=args
    )

    c_val = best_params.get('C', 1.0)
    kernel = best_params.get('kernel', args.kernel)
    best_alpha = best_params.get('alpha', 0.60)  # Correctly extract alpha with default
    print(f"-> Selected Optimal Sequence Smoothing Factor (Alpha): {best_alpha:.2f}")

    groups = train_df['parent_doc_id'].values

    # Build Deployable Pipeline
    feature_pipeline = get_dynamic_feature_pipeline(best_params, use_pre_lemmatized=True)
    calibrated_clf = get_classifier(kernel=kernel, c_val=c_val, calibrate=True, groups_cv=groups)

    full_pipeline = Pipeline([
        ('features', feature_pipeline),
        ('classifier', calibrated_clf)
    ])

    X_train_raw = train_df.to_dict(orient='records')
    X_test_raw = test_df.to_dict(orient='records')
    y_train = train_df['label'].values
    y_test = test_df['label'].values

    print("\nTraining final probability-calibrated multi-scale pipeline on 100% training sentences...")
    full_pipeline.fit(X_train_raw, y_train, classifier__groups=groups)

    optimal_threshold = 0.5
    if args.score == 'set_fp':
        print("Calibrating sentence decision threshold for 1% max FPR constraint...")
        sgkf = StratifiedGroupKFold(n_splits=3)
        oof_probs = np.zeros(len(y_train))

        for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_train_raw, y_train, groups=groups)):
            tr_sub = [X_train_raw[i] for i in tr_idx]
            va_sub = [X_train_raw[i] for i in va_idx]
            y_tr_f = y_train[tr_idx]

            X_tr_f, X_va_f = get_cached_split_features(tr_sub, va_sub, best_params)
            clf_f = get_classifier(kernel=kernel, c_val=c_val, calibrate=True, groups_cv=[groups[i] for i in tr_idx])
            clf_f.fit(X_tr_f, y_tr_f)
            oof_probs[va_idx] = clf_f.predict_proba(X_va_f)[:, 1]

        smoothed_oof = apply_adaptive_sequence_smoothing(train_df, oof_probs, alpha=best_alpha)
        fpr, tpr, thresholds = roc_curve(y_train, smoothed_oof)

        thresholds = np.clip(thresholds, 0.0, 1.0)
        valid = np.where(fpr <= 0.01)[0]
        optimal_threshold = float(thresholds[valid[-1]]) if len(valid) > 0 else 0.5
        print(f"-> Calibrated Sentence Threshold (1% Max FPR Probability): {optimal_threshold:.6f}")

    # Attach deployment attributes to the pipeline object so they persist in the saved .pkl
    full_pipeline.optimal_threshold = optimal_threshold
    full_pipeline.alpha = best_alpha

    # Evaluate on Unseen Test Documents
    raw_test_probs = full_pipeline.predict_proba(X_test_raw)[:, 1]
    smoothed_test_probs = apply_adaptive_sequence_smoothing(test_df, raw_test_probs, alpha=best_alpha)

    metrics = evaluate_mixed_authorship_metrics(test_df, y_test, smoothed_test_probs, optimal_threshold)

    print("\n" + "=" * 55)
    print("      MIXED-AUTHORSHIP SENTENCE PERFORMANCE      ")
    print("=" * 55)
    print(classification_report(y_test, metrics['preds'], digits=4))
    print(f"Sentence-Level ROC-AUC: {metrics['sentence_roc_auc']:.4f}")
    print(f"Segment Span IoU:       {metrics['span_iou']:.4f}")
    print(f"AI Ratio MAE:           {metrics['ai_ratio_mae']:.4f}\n")

    # Save Pipeline & Log Registry
    study_clean = args.study_name[:-4] if args.study_name.endswith('.pkl') else args.study_name
    save_path = f"{study_clean}.pkl"
    joblib.dump(full_pipeline, save_path)
    print(f"Deployable pipeline saved successfully to '{save_path}'")

    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'study_name': study_clean,
        'save_path': save_path,
        'kernel': kernel,
        'score_metric': args.score,
        'calibrated_threshold': optimal_threshold,
        'alpha': best_alpha,
        'C': c_val,
        'sent_precision_ai': metrics['sent_precision_ai'],
        'sent_recall_ai': metrics['sent_recall_ai'],
        'sent_f1_ai': metrics['sent_f1_ai'],
        'sentence_roc_auc': metrics['sentence_roc_auc'],
        'span_iou': metrics['span_iou'],
        'ai_ratio_mae': metrics['ai_ratio_mae']
    }
    log_experiment_to_registry(record)


if __name__ == "__main__":
    main()