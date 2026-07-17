# utils.py
import os
import ast
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from features import StylometricExtractor, TextExtractor
import hashlib

def generate_df_hash(df):
    serialized = f"{len(df)}_" + "".join(df['text'].head(50).astype(str))
    return hashlib.md5(serialized.encode('utf-8')).hexdigest()

def safe_parse_list(val):
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

def prepare_classification_dataset(df, selected_models, granularity='full', source_filter=None, sample_one_llm=False, random_state=42):
    import random
    if random_state is not None:
        random.seed(random_state)

    if source_filter:
        df = df[df['source'].isin(source_filter)].copy()
        
    records = []
    
    for _, row in df.iterrows():
        source = row.get('source', 'unknown')
        raw_human_sents = safe_parse_list(row.get('abstract_sentence', []))
        human_sents = raw_human_sents if isinstance(raw_human_sents, list) else [] #TODO why empty list of not a list???
        
        # 1. Add Human Data (Label 0)
        if granularity == 'full':
            human_text = row.get('abstract', '')
            if pd.isna(human_text) or human_text == "":
                continue
            records.append({'text': human_text, 'sentences': human_sents, 'label': 0, 'source': source})
        else:
            for sent in human_sents:
                if sent is not None and sent != "":
                    records.append({'text': sent, 'sentences': [sent], 'label': 0, 'source': source})
                    
        # 2. Add LLM Data (Label 1)
        valid_models = []
        for model in selected_models:
            if granularity == 'full':
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
            
        # Select models to process (Either exactly 1 representative rewrite per row, or all of them)
        if sample_one_llm:
            models_to_process = [random.choice(valid_models)]
        else:
            models_to_process = valid_models
            
        for model in models_to_process:
            if granularity == 'full':
                col_name = f"{model}_full"
                ai_text = row[col_name]
                
                sent_col = f"{model}_single" if f"{model}_single" in row else f"{model}_sentence"
                raw_ai_sents = safe_parse_list(row.get(sent_col, []))
                ai_sents = raw_ai_sents if isinstance(raw_ai_sents, list) else []
                
                records.append({'text': ai_text, 'sentences': ai_sents, 'label': 1, 'source': source})
            else:
                sent_col = f"{model}_single" if f"{model}_single" in row else f"{model}_sentence"
                raw_sent_list = safe_parse_list(row.get(sent_col, []))
                if isinstance(raw_sent_list, list):
                    for sent in raw_sent_list:
                        if sent is not None and sent != "":
                            records.append({'text': sent, 'sentences': [sent], 'label': 1, 'source': source})
                            
    return pd.DataFrame(records)

def get_feature_extraction_pipeline():
    """Defines and returns the core feature extractor blueprint for Sklearn pipelines."""
    return FeatureUnion([
        ('word_ngrams', Pipeline([
            ('extract', TextExtractor()),
            ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=5000))
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

def get_or_create_cached_features(train_df, val_df, test_df, cache_dir="./cache", granularity="full", clear_cache=False):
    """Processes and caches features cleanly for three distinct datasets."""
    train_hash = generate_df_hash(train_df)
    val_hash = generate_df_hash(val_df)
    test_hash = generate_df_hash(test_df)
    
    train_cache_path = os.path.join(cache_dir, f"X_train_{granularity}_{train_hash}.joblib")
    val_cache_path = os.path.join(cache_dir, f"X_val_{granularity}_{val_hash}.joblib")
    test_cache_path = os.path.join(cache_dir, f"X_test_{granularity}_{test_hash}.joblib")
    
    if clear_cache:
        for p in [train_cache_path, val_cache_path, test_cache_path]:
            if os.path.exists(p): os.remove(p)

    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path) and os.path.exists(test_cache_path):
        print("Loading pre-calculated features from cache...")
        X_train_scaled = joblib.load(train_cache_path)
        X_val_scaled = joblib.load(val_cache_path)
        X_test_scaled = joblib.load(test_cache_path)
    else:
        print("No cache found. Extracting features for all splits...")
        X_train_raw = train_df[['text', 'sentences']].to_dict(orient='records')
        X_val_raw = val_df[['text', 'sentences']].to_dict(orient='records')
        X_test_raw = test_df[['text', 'sentences']].to_dict(orient='records')
        
        feature_union = get_feature_extraction_pipeline()
        
        X_train_scaled = feature_union.fit_transform(X_train_raw)
        X_val_scaled = feature_union.transform(X_val_raw)
        X_test_scaled = feature_union.transform(X_test_raw)
        
        joblib.dump(X_train_scaled, train_cache_path)
        joblib.dump(X_val_scaled, val_cache_path)
        joblib.dump(X_test_scaled, test_cache_path)
        print(f"Features calculated and cached in '{cache_dir}'")
        
    return X_train_scaled, X_val_scaled, X_test_scaled