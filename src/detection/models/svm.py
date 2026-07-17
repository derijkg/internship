# models/svm.py
import optuna
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from utils import get_or_create_cached_features, get_feature_extraction_pipeline

def optimize_svm_with_optuna(train_df, val_df, test_df, granularity):
    print("Pre-building static TF-IDF and stylometric feature matrix...") #TODO why again here, just pass from train.py
    X_train_scaled, X_val_scaled, _ = get_or_create_cached_features(
        train_df, val_df, test_df, granularity=granularity
    )
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    
    def objective(trial):
        c_val = trial.suggest_float('C', 1e-3, 1e2, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid'])
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto']) if kernel != 'linear' else 'scale'
        
        clf = SVC(C=c_val, kernel=kernel, gamma=gamma, random_state=42, class_weight='balanced')
        clf.fit(X_train_scaled, y_train)
        
        # Evaluate on the validation split
        preds = clf.predict(X_val_scaled)
        return f1_score(y_val, preds, average='macro')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    print("\n--- Best SVM Hyperparameters ---")
    print(study.best_params)
    print(f"Best validation F1-score: {study.best_value:.4f}")
    
    return study.best_params

def train_svm(train_df, val_df, test_df, c_val, kernel, save_path, granularity, run_optuna=False):
    if run_optuna:
        print("Running Hyperparameter Optimization via Optuna...")
        best_params = optimize_svm_with_optuna(train_df, val_df, test_df, granularity)
        c_val = best_params['C']
        kernel = best_params['kernel']
        gamma = best_params.get('gamma', 'scale')
    else:
        gamma = 'scale'
        
    # Prepare raw pipeline records
    X_train_raw = train_df[['text', 'sentences']].to_dict(orient='records')
    X_test_raw = test_df[['text', 'sentences']].to_dict(orient='records')
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    feature_pipeline = get_feature_extraction_pipeline()
    clf = SVC(C=c_val, kernel=kernel, gamma=gamma, random_state=42, class_weight='balanced')
    
    # Bundle feature pipeline and classifier into a single deployable artifact
    full_pipeline = Pipeline([
        ('features', feature_pipeline),
        ('classifier', clf)
    ])
    
    print(f"Training final unified SVM pipeline...")
    full_pipeline.fit(X_train_raw, y_train)
    
    # Evaluate on the held-out test split
    preds = full_pipeline.predict(X_test_raw)
    print("\nSVM Test Performance Evaluation:")
    print(classification_report(y_test, preds))
    
    joblib.dump(full_pipeline, save_path)
    print(f"Deployable pipeline saved successfully to {save_path}")