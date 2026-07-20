# models/svm.py
import sys
import optuna
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from features import get_or_create_cached_features, get_feature_extraction_pipeline, generate_df_hash
from sklearn.model_selection import train_test_split

def optimize_svm_with_optuna(train_df, val_df, test_df, granularity, sample_size=3000, reset_study=False):
    """
    Finds the best SVM parameters using a stratified random subset 
    of the training/validation data to speed up execution.
    """
    print(f"\n--- Optuna Subsampling Enabled ---")
    
    # 1. Stratified subsample of training data
    # #ADDED: Robust stratification fallback check if subset selection fails due to limited classes
    stratify_train = train_df['label'] if ('label' in train_df.columns and train_df['label'].value_counts().min() > 1) else None
    if len(train_df) > sample_size:
        print(f"Subsampling training set from {len(train_df)} down to {sample_size} for tuning...")
        train_sub, _ = train_test_split(
            train_df,
            train_size=sample_size,
            random_state=42,
            stratify=stratify_train
        )
    else:
        train_sub = train_df

    # 2. Stratified subsample of validation data (tuning evaluate step should also be fast)
    val_sample_size = min(len(val_df), 1000)
    stratify_val = val_df['label'] if ('label' in val_df.columns and val_df['label'].value_counts().min() > 1) else None
    if len(val_df) > val_sample_size:
        print(f"Subsampling validation set from {len(val_df)} down to {val_sample_size}...")
        val_sub, _ = train_test_split(
            val_df,
            train_size=val_sample_size,
            random_state=42,
            stratify=stratify_val
        )
    else:
        val_sub = val_df

    # 3. Extract features ONLY for the subsets (creates unique, independent cache files)
    X_train_scaled, X_val_scaled, _ = get_or_create_cached_features(
        train_sub, val_sub, test_df=None, granularity=granularity
    )
    
    y_train = train_sub['label'].values
    y_val = val_sub['label'].values

    # 4. Define the objective function inside using the subset variables
    def objective(trial):
        c_val = trial.suggest_float('C', 1e-3, 1e2, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid'])
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto']) if kernel != 'linear' else 'scale'
        
        print(f"\n>>> [Trial {trial.number}] Training on {X_train_scaled.shape[0]} rows: C={c_val:.4f}, gamma={gamma}, kernel={kernel}...")

        # cache_size=2000 is safe since the subset matrix is even smaller
        clf = SVC(C=c_val, kernel=kernel, gamma=gamma, random_state=42, class_weight='balanced', cache_size=2000)
        clf.fit(X_train_scaled, y_train)
        
        preds = clf.predict(X_val_scaled)
        score = f1_score(y_val, preds, average='macro')
        print(f"<<< [Trial {trial.number}] Score: {score:.4f}")
        return score

    # 5. Run the study with saving best and optional reset with --reset_study
    db_path = "sqlite:///optuna_svm.db"
    study_name = f"svm_optimization_{granularity}"

    # Calculate the hash of the current full training dataset
    current_hash = generate_df_hash(train_df)

    # Temporarily create/load study to check for changes
    study = optuna.create_study(
        study_name=study_name,
        storage=db_path,
        direction='maximize',
        load_if_exists=True
    )

    # Check if dataset has changed
    stored_hash = study.user_attrs.get("dataset_hash")
    if stored_hash is not None and stored_hash != current_hash:
        print("\n⚠️  WARNING: The training dataset has changed since the last Optuna run!")
        
        if not reset_study:
            if sys.stdin.isatty():
                try:
                    response = input("Would you like to reset the Optuna database to start fresh? [y/N]: ").strip().lower()
                    if response in ['y', 'yes']:
                        reset_study = True
                except Exception:
                    print("Interactive prompt failed. Proceeding with existing study.")
            else:
                print("Non-interactive environment detected. Proceeding with existing study.")

    # Handle the reset flag (requested via prompt or CLI argument)
    if reset_study:
        print(f"\n[Resetting Study] Deleting existing study '{study_name}' from {db_path}...")
        try:
            optuna.delete_study(study_name=study_name, storage=db_path)
        except (KeyError, Exception):
            print("No existing study found to delete. Starting fresh.")
        
        # Recreate fresh study
        study = optuna.create_study(
            study_name=study_name,
            storage=db_path,
            direction='maximize'
        )

    # Update or save the current hash in study attributes
    study.set_user_attr("dataset_hash", current_hash)

    # Check if we already have trials in this database
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 0 and not reset_study:
        print(f"\n---> Resuming existing study '{study_name}' with {len(completed_trials)} completed trials found in database.")
    
    # Run the optimization loop (if resume, it will run *additional* trials up to 15)
    study.optimize(objective, n_trials=15)
    
    # --- Custom Tie-Breaker (Occam's Razor) ---
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best_value = study.best_value
    
    # Filter trials that are within a tiny floating-point tolerance of the best score
    tolerance = 1e-6
    top_trials = [t for t in completed_trials if abs(t.value - best_value) < tolerance]
    
    # From those top trials, select the one with the lowest C value to maximize regularization
    best_trial = min(top_trials, key=lambda t: t.params.get('C', float('inf')))
    best_params = best_trial.params

    print("\nBest trial (with custom tie-breaker for lowest C):")
    print(f"  Value (F1): {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")
        
    return best_params

def train_svm(train_df, val_df, test_df, c_val, kernel, save_path, granularity, run_optuna=False, reset_study=False):
    if run_optuna:
        print("Running Hyperparameter Optimization via Optuna...")
        best_params = optimize_svm_with_optuna(train_df, val_df, test_df, granularity, reset_study=reset_study)
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