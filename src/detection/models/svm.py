# optuna_svm.py
import optuna
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
from features import extract_stylometric_features
from sklearn.base import BaseEstimator, TransformerMixin
from utils import get_or_create_cached_features

class StylometricExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # No initialization arguments needed for this transformer
        pass

    def fit(self, X, y=None):
        # Rule-based stylometrics do not "learn" anything from the data,
        # so fit simply returns self without doing anything.
        return self
    
    def transform(self, X):
        """
        X is expected to be a list of dicts: 
        [{'text': "...", 'sentences': ["...", "..."]}, ...]
        """
        # Parallelize the extraction loop across all available CPU cores
        features = Parallel(n_jobs=-1)(
            delayed(extract_stylometric_features)(item['text'], item['sentences']) 
            for item in X
        )
        return np.array(features)





# Assuming `train_df` is prepared and split
def optimize_svm_with_optuna(train_df):
    # Form input dictionaries
    X_train = train_df[['text', 'sentences']].to_dict(orient='records')
    y_train = train_df['label'].values
    
    # Pre-extracting features avoids doing this step repeatedly inside the loop
    # Let's build a static feature pipeline to transform the raw text into a matrix once
    print("Pre-building static TF-IDF and stylometric feature matrix...")
    
    static_feature_pipeline = FeatureUnion([
        ('word_ngrams', TfidfVectorizer(preprocessor=lambda x: x['text'], ngram_range=(1, 3), max_features=5000)),
        ('char_ngrams', TfidfVectorizer(preprocessor=lambda x: x['text'], analyzer='char', ngram_range=(2, 5), max_features=5000)),
        ('stylometrics', StylometricExtractor())
    ])
    
    # Transform data once
    X_train_scaled, X_test_scaled = get_or_create_cached_features(train_df, test_df, granularity=args.granularity)
    y_train = train_df['label'].values
    y_test = test_df['label'].values

    
    def objective(trial):
        # Hyperparameters to search
        c_val = trial.suggest_float('C', 1e-3, 1e2, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid'])
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto']) if kernel != 'linear' else 'scale'
        
        # Initialize SVM with trial parameters
        clf = SVC(C=c_val, kernel=kernel, gamma=gamma, random_state=42, class_weight='balanced')
        
        # Use cross-validation (e.g., 3-fold to save time) to score the trial
        scores = cross_val_score(clf, X_train_scaled, y_train, cv=3, scoring='f1_macro', n_jobs=-1)
        
        return np.mean(scores)

    # Run the study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    print("\n--- Best SVM Hyperparameters ---")
    print(study.best_params)
    print(f"Best cross-validated F1-score: {study.best_value:.4f}")
    
    return study.best_params