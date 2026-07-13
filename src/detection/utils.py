# utils.py
import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from models.svm_model import StylometricExtractor


def prepare_classification_dataset(df, selected_models, granularity='full', source_filter=None):
    """
    Reshapes the wide dataframe into a binary classification format:
    text | sentences | label (0 for human, 1 for AI) | source
    """
    if source_filter:
        # Filter source (e.g. ['UG', 'SB', 'HBO'])
        df = df[df['source'].isin(source_filter)].copy()
        
    records = []
    
    for _, row in df.iterrows():
        source = row.get('source', 'unknown')
        
        # Human Baseline
        if granularity == 'full':
            human_text = row['abstract']
            human_sents = row['abstract_sentence'] # Assumed list of strings
            records.append({'text': human_text, 'sentences': human_sents, 'label': 0, 'source': source})
        else:
            for sent in row['abstract_sentence']:
                records.append({'text': sent, 'sentences': [sent], 'label': 0, 'source': source})
                
        # LLM Outputs
        for model in selected_models:
            if granularity == 'full':
                col_name = f"{model}_full"
                # Fallback to reconstructing from sentences if full col is absent
                if col_name in row and pd.notna(row[col_name]):
                    ai_text = row[col_name]
                    ai_sents = row.get(f"{model}_sentence", [])
                    records.append({'text': ai_text, 'sentences': ai_sents, 'label': 1, 'source': source})
            else:
                sent_col = f"{model}_sentence"
                if sent_col in row and isinstance(row[sent_col], list):
                    for sent in row[sent_col]:
                        records.append({'text': sent, 'sentences': [sent], 'label': 1, 'source': source})
                        
    return pd.DataFrame(records)


def get_or_create_cached_features(train_df, test_df, cache_dir="./cache", granularity="full"):
    """
    Extracts and caches the entire scaled feature matrix (TF-IDF + Stylometrics) to disk.
    If the files already exist, it instantly loads them.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    train_cache_path = os.path.join(cache_dir, f"X_train_{granularity}.joblib")
    test_cache_path = os.path.join(cache_dir, f"X_test_{granularity}.joblib")
    
    # Format inputs as dictionary lists for the custom pipeline transformers
    X_train_raw = train_df[['text', 'sentences']].to_dict(orient='records')
    X_test_raw = test_df[['text', 'sentences']].to_dict(orient='records')
    
    if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
        print("Loading pre-calculated features from cache...")
        X_train_scaled = joblib.load(train_cache_path)
        X_test_scaled = joblib.load(test_cache_path)
    else:
        print("No cache found. Extracting features (this might take a minute)...")
        
        # Build the feature union
        feature_union = FeatureUnion([
            ('word_ngrams', TfidfVectorizer(preprocessor=lambda x: x['text'], ngram_range=(1, 3), max_features=5000)),
            ('char_ngrams', TfidfVectorizer(preprocessor=lambda x: x['text'], analyzer='char', ngram_range=(2, 5), max_features=5000)),
            ('stylometrics', StylometricExtractor())
        ])
        
        # Scale the features
        scaler = StandardScaler(with_mean=False)
        
        # Fit and transform training features
        X_train_transformed = feature_union.fit_transform(X_train_raw)
        X_train_scaled = scaler.fit_transform(X_train_transformed)
        
        # Transform test features
        X_test_transformed = feature_union.transform(X_test_raw)
        X_test_scaled = scaler.transform(X_test_transformed)
        
        # Save to disk
        joblib.dump(X_train_scaled, train_cache_path)
        joblib.dump(X_test_scaled, test_cache_path)
        print(f"Features calculated and cached in '{cache_dir}'")
        
    return X_train_scaled, X_test_scaled