# features.py
import ast
import hashlib
import json
import os
import re
import string
import random
from collections import Counter

import joblib
import nltk
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from nltk.stem.snowball import SnowballStemmer

# Ensure NLTK Dutch Stopwords are available quietly
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
dutch_stopwords = stopwords.words('dutch')

dutch_stemmer = SnowballStemmer("dutch")

def stemming_tokenizer(text):
    """Tokenizes raw text and stems the tokens to match Dutch morphological forms."""
    # Matches alphanumeric words of at least 2 characters (mimics sklearn's default token_pattern)
    tokens = re.findall(r'\b\w\w+\b', text.lower())
    return [dutch_stemmer.stem(token) for token in tokens]
dutch_stopwords_stemmed = list(set([dutch_stemmer.stem(word) for word in dutch_stopwords]))

# ALTERNATIVE SUGGESTION (Commented out): For high-accuracy lemmatization, spaCy's Dutch model is a good option:
# import spacy
# nlp = spacy.load("nl_core_news_sm", disable=["parser", "ner"])
# def lemmatizing_tokenizer(text):
#     return [token.lemma_ for token in nlp(text.lower()) if not token.is_punct]

# Dutch transition words used for stylometric analysis
DUTCH_TRANSITIONS = {
    "echter", "bovendien", "daarnaast", "desalniettemin", "kortom", 
    "tevens", "daardoor", "derhalve", "bijgevolg", "namelijk"
}


# ==========================================
# 1. Stylometric Helper Functions
# ==========================================

def calculate_ttr(words):
    """Calculates Type-Token Ratio (lexical diversity measure)."""
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def calculate_hapax_ratio(words):
    """Calculates the ratio of words that appear exactly once."""
    if not words:
        return 0.0
    counts = Counter(words)
    hapaxes = sum(1 for w, c in counts.items() if c == 1)
    return hapaxes / len(words)


def extract_stylometric_features(text, sentences):
    """Extracts a 12-dimensional array of statistical and syntactic metrics from raw text."""
    words = re.findall(r'\w+', text.lower())
    total_chars = len(text)
    
    if not words or not sentences:
        return np.zeros(12)
    
    # #ADDED: Optimize calculations when we are dealing with a single sentence (sentence granularity)
    if len(sentences) <= 1:
        # Avoid running a redundant regex pass on the sentence
        mean_sent_len = float(len(words))
        var_sent_len = 0.0
        burstiness = 0.0
    else:
        # Multi-sentence branch: run the sentence tokenization and statistical operations
        sent_lengths = [len(re.findall(r'\w+', s)) for s in sentences if len(re.findall(r'\w+', s)) > 0]
        mean_sent_len = np.mean(sent_lengths) if sent_lengths else 0.0
        var_sent_len = np.var(sent_lengths) if sent_lengths else 0.0
        burstiness = (np.std(sent_lengths) / mean_sent_len) if mean_sent_len > 0 else 0.0
    
    word_lengths = [len(w) for w in words]
    mean_word_len = np.mean(word_lengths)
    var_word_len = np.var(word_lengths)
    
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
    ])

# ==========================================
# 2. Custom Scikit-Learn Transformers
# ==========================================

class TextExtractor(BaseEstimator, TransformerMixin):
    """Extracts raw text strings from lists of dictionary records."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [item['text'] for item in X]


class StylometricExtractor(BaseEstimator, TransformerMixin):
    """Computes multidimensional linguistic and structural metrics in parallel."""
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Prevent parallelization overhead on small validation folds
        if len(X) < 100 or self.n_jobs == 1:
            features = [extract_stylometric_features(item['text'], item['sentences']) for item in X]
        else:
            features = Parallel(n_jobs=self.n_jobs)(
                delayed(extract_stylometric_features)(item['text'], item['sentences']) 
                for item in X
            )
        return np.array(features)


# ==========================================
# 3. Core Preprocessing & Caching Utilities
# ==========================================

def generate_df_hash(df):
    """Generates a reproducible MD5 hash of the DataFrame's size and head content."""
    serialized = f"{len(df)}_" + "".join(df['text'].head(50).astype(str))
    return hashlib.md5(serialized.encode('utf-8')).hexdigest()


def safe_parse_list(val):
    """Safely parses string-serialized lists back to Python lists."""
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('[') and val.endswith(']'):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                try:
                    return json.loads(val)
                except Exception:
                    pass
    return val


def prepare_classification_dataset(df, selected_models, granularity='full', source_filter=None, llm_ratio=4, random_state=42):
    import random
    local_rng = random.Random(random_state) if random_state is not None else random

    if source_filter:
        df = df[df['source'].isin(source_filter)].copy()
        
    records = []
    
    active_granularities = ['full', 'sentence'] if granularity == 'both' else [granularity]
    
    for _, row in df.iterrows():
        source = row.get('source', 'unknown')
        
        row_human_records = []
        row_llm_records = []
        
        for gran in active_granularities:
            raw_human_sents = safe_parse_list(row.get('abstract_sentence', []))
            human_sents = raw_human_sents if isinstance(raw_human_sents, list) else []
            
            # --- Human Generation ---
            if gran == 'full':
                human_text = row.get('abstract', '')
                if pd.notna(human_text) and human_text != "":
                    row_human_records.append({'text': human_text, 'sentences': human_sents, 'label': 0, 'source': source})
            else:
                for sent in human_sents:
                    if sent is not None and sent != "":
                        row_human_records.append({'text': sent, 'sentences': [sent], 'label': 0, 'source': source})
            
            # --- LLM Parsing ---
            valid_models = []
            for model in selected_models:
                if gran == 'full':
                    col_name = f"{model}_full"
                    if col_name in row and pd.notna(row[col_name]) and row[col_name] != "":
                        valid_models.append(model)
                else:
                    sent_col = f"{model}_single" if f"{model}_single" in row else f"{model}_sentence"
                    if sent_col in row:
                        raw_sent_list = safe_parse_list(row[sent_col])
                        if isinstance(raw_sent_list, list) and any(s is not None and s != "" for s in raw_sent_list):
                            valid_models.append(model)
            
            if not valid_models:
                continue
                
            # --- Ratio Sampling Logic ---
            if llm_ratio > 0 and len(valid_models) > llm_ratio:
                models_to_process = local_rng.sample(valid_models, k=llm_ratio)
            else:
                models_to_process = valid_models
                
            # --- LLM Generation ---
            for model in models_to_process:
                if gran == 'full':
                    col_name = f"{model}_full"
                    ai_text = row[col_name]
                    sent_col = f"{model}_single" if f"{model}_single" in row else f"{model}_sentence"
                    raw_ai_sents = safe_parse_list(row.get(sent_col, []))
                    ai_sents = raw_ai_sents if isinstance(raw_ai_sents, list) else []
                    
                    row_llm_records.append({'text': ai_text, 'sentences': ai_sents, 'label': 1, 'source': source})
                else:
                    sent_col = f"{model}_single" if f"{model}_single" in row else f"{model}_sentence"
                    raw_sent_list = safe_parse_list(row.get(sent_col, []))
                    if isinstance(raw_sent_list, list):
                        for sent in raw_sent_list:
                            if sent is not None and sent != "":
                                row_llm_records.append({'text': sent, 'sentences': [sent], 'label': 1, 'source': source})
                                
        if row_human_records and row_llm_records:
            records.extend(row_human_records)
            records.extend(row_llm_records)
            
    # #ADDED: Build DataFrame, calculate the empirical distribution ratio, and print details
    final_df = pd.DataFrame(records)
    
    if not final_df.empty and 'label' in final_df.columns:
        num_human = (final_df['label'] == 0).sum()
        num_llm = (final_df['label'] == 1).sum()
        
        if num_human > 0:
            actual_ratio = num_llm / num_human
            print(f"[Dataset Stats] Granularity: {granularity} | "
                  f"Total Entries: {len(final_df)} | "
                  f"Empirical Ratio -> Human: 1 to LLM: {actual_ratio:.2f} "
                  f"(Counts: {num_human} Human, {num_llm} LLM)")
        else:
            print(f"[Dataset Stats] Granularity: {granularity} | "
                  f"Total Entries: {len(final_df)} | "
                  f"No Human entries found (LLM Count: {num_llm})")
                  
    return final_df

def get_feature_extraction_pipeline():
    """Defines and returns the core feature extractor blueprint for Sklearn pipelines."""
    return FeatureUnion([
        ('word_ngrams', Pipeline([
            ('extract', TextExtractor()),
            # #ADDED: Configured custom Dutch stemming tokenizer and stemmed stopwords.
            # #ADDED: Set token_pattern=None to silence scikit-learn warnings when passing a tokenizer.
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3), 
                max_features=5000, 
                tokenizer=stemming_tokenizer, 
                token_pattern=None,
                stop_words=dutch_stopwords_stemmed
            ))
        ])),
        ('char_ngrams', Pipeline([
            ('extract', TextExtractor()),
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=5000))
        ])),
        ('stylometrics', Pipeline([
            ('extractor', StylometricExtractor(n_jobs=-1)),
            ('scaler', StandardScaler())  
        ]))
    ])

#TODO CHECK CACHING OPERATION AND DATA USAGE
#TODO check test df but good or just set default 
def get_or_create_cached_features(train_df, val_df, test_df=None, cache_dir="./cache", granularity="full", clear_cache=False):
    """Processes and caches features cleanly for distinct datasets, tracking training state dependencies."""
    os.makedirs(cache_dir, exist_ok=True)
    
    train_hash = generate_df_hash(train_df)
    val_hash = generate_df_hash(val_df)
    
    train_cache_path = os.path.join(cache_dir, f"X_train_{granularity}_{train_hash}.joblib")
    val_cache_path = os.path.join(cache_dir, f"X_val_{granularity}_train_{train_hash}_val_{val_hash}.joblib")
    
    if test_df is not None:
        test_hash = generate_df_hash(test_df)
        test_cache_path = os.path.join(cache_dir, f"X_test_{granularity}_train_{train_hash}_test_{test_hash}.joblib")
    else:
        test_cache_path = None
    
    if clear_cache:
        paths = [train_cache_path, val_cache_path]
        if test_cache_path:
            paths.append(test_cache_path)
        for p in paths:
            if os.path.exists(p): 
                os.remove(p)

    cache_exists = os.path.exists(train_cache_path) and os.path.exists(val_cache_path)
    if test_cache_path:
        cache_exists = cache_exists and os.path.exists(test_cache_path)

    if cache_exists:
        print("Loading pre-calculated features from cache...")
        X_train_scaled = joblib.load(train_cache_path)
        X_val_scaled = joblib.load(val_cache_path)
        X_test_scaled = joblib.load(test_cache_path) if test_cache_path else None
    else:
        print("No matched cache found. Extracting features...")
        X_train_raw = train_df[['text', 'sentences']].to_dict(orient='records')
        X_val_raw = val_df[['text', 'sentences']].to_dict(orient='records')
        
        feature_union = get_feature_extraction_pipeline()
        
        X_train_scaled = feature_union.fit_transform(X_train_raw)
        X_val_scaled = feature_union.transform(X_val_raw)
        
        # Write training and validation cache
        joblib.dump(X_train_scaled, train_cache_path)
        joblib.dump(X_val_scaled, val_cache_path)
        
        if test_df is not None:
            X_test_raw = test_df[['text', 'sentences']].to_dict(orient='records')
            X_test_scaled = feature_union.transform(X_test_raw)
            joblib.dump(X_test_scaled, test_cache_path)
        else:
            X_test_scaled = None
            
        print(f"Features calculated and cached in '{cache_dir}'")
        
    return X_train_scaled, X_val_scaled, X_test_scaled