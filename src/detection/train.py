# train.py
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from features import prepare_classification_dataset, pre_lemmatize_dataset
from models.svm import train_svm
from models.robbert import train_transformer

DEFAULT_MODELS = ['qwen3.5:4b', 'qwen3.6:27b', 'gemma4:e4b', 'gemma4:26b']


def parse_sample_size(value):
    """Parses training sample size as an absolute integer or float percentage."""
    try:
        if '.' in value:
            val = float(value)
            if not (0.0 < val <= 1.0):
                raise argparse.ArgumentTypeError(
                    f"Float sample size must be between 0.0 and 1.0, got {val}"
                )
            return val
        val = int(value)
        if val <= 0:
            raise argparse.ArgumentTypeError(
                f"Integer sample size must be positive, got {val}"
            )
        return val
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Sample size must be an int or float, but got '{value}'"
        )


def main():
    parser = argparse.ArgumentParser(description="LLM Detection Training Orchestrator (Auto-Tuned)")

    # Core Controls
    parser.add_argument('--data_path', type=str, required=True, help="Path to raw parquet or csv file")
    parser.add_argument('--classifier', type=str, choices=['svm', 'bert', 'both'], default='both', help="Model type to train")
    parser.add_argument('--granularity', type=str, choices=['full', 'sentence', 'both'], default='full', help="Train on full abstracts, split sentences, or both")

    # Dataset Filtering Parameters
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS, help="LLM models to include in classification task")
    parser.add_argument('--source', nargs='+', default=['UG', 'SB', 'HBO'], help="Sources to isolate (default all: UG, SB, HBO)")
    parser.add_argument('--llm_ratio', type=int, default=-1, help="Number of LLM rewrites to sample per human record (-1 for all)")
    parser.add_argument('--reset_study', action='store_true', help="Clear and restart the Optuna hyperparameter study database")

    # Hyperparameter Tuning Controls
    parser.add_argument('--trials', type=int, default=50, help="Number of trials to execute during tuning (if single-stage)")
    parser.add_argument(
        '--tuning',
        type=str,
        choices=['model', '2stage', 'merged'],
        default='2stage',
        help="Tuning strategy: 'model', '2stage', or 'merged'."
    )
    parser.add_argument(
        '--tuning_sample_size',
        type=parse_sample_size,
        default=3000,
        help="Subsample size for tuning (absolute int or float percentage)."
    )
    parser.add_argument('--trials_stage1', type=int, default=30, help="Stage 1 trials (TF-IDF)")
    parser.add_argument('--trials_stage2', type=int, default=30, help="Stage 2 trials (SVM)")

    # #explained Collapsed --studyname and --svmname into a single consolidated --study_name argument.
    parser.add_argument('--study_name', type=str, default=None, help="Name for the Optuna study database entry and saved .pkl model file.")

    parser.add_argument('--kernel', type=str, choices=['linear', 'rbf', 'sigmoid', 'poly', 'all'], default='all', help="SVM kernel choice")
    parser.add_argument('--transformer_name', type=str, default='pdelobelle/robbert-2023-dutch-base', help="HuggingFace model string")

    parser.add_argument('--score', type=str, choices=['f1', 'precision', 'f0.5', 'roc_auc', 'set_fp'], default='f0.5', help="Metric to optimize")
    parser.add_argument('--n_jobs_optuna', type=int, default=1, help="Number of parallel jobs for Optuna tuning (default 1)")

    args = parser.parse_args()

    print(f"Loading data from: {args.data_path}")
    if args.data_path.endswith('.csv'):
        raw_df = pd.read_csv(args.data_path)
    elif args.data_path.endswith('.parquet') or args.data_path.endswith('.pq'):
        raw_df = pd.read_parquet(args.data_path)
    else:
        raise ValueError("Unsupported data format (Must be CSV or Parquet)")

    # Early Filtering
    if args.source:
        if 'source' in raw_df.columns:
            raw_df = raw_df[raw_df['source'].isin(args.source)].copy()
            print(f"Early filtered dataset to sources: {args.source} ({len(raw_df)} rows remaining)")
        else:
            print("Warning: 'source' column not found in dataset. Skipping early filtering.")

    stratify_col = raw_df['source'] if 'source' in raw_df.columns else None

    train_raw_df, test_raw_df = train_test_split(
        raw_df,
        test_size=0.20,
        random_state=42,
        stratify=stratify_col
    )

    # Dataset Shaping
    train_df = prepare_classification_dataset(
        train_raw_df, selected_models=args.models, granularity=args.granularity, source_filter=None, llm_ratio=args.llm_ratio
    )
    test_df = prepare_classification_dataset(
        test_raw_df, selected_models=args.models, granularity=args.granularity, source_filter=None, llm_ratio=args.llm_ratio
    )

    # Sequential Pre-Lemmatization
    train_df = pre_lemmatize_dataset(train_df, text_column='text')
    test_df = pre_lemmatize_dataset(test_df, text_column='text')

    print(f"\nDataset compiled successfully:")
    print(f"-> Combined Train Split: {len(train_df)} rows")
    print(f"-> Test Split:           {len(test_df)} rows\n")

    if args.classifier in ['svm', 'both']:
        # #explained Automatically derives save_path (.pkl) and study_name from the consolidated --study_name CLI argument.
        if args.study_name:
            study_name_clean = args.study_name[:-4] if args.study_name.endswith('.pkl') else args.study_name
            save_path = f"{study_name_clean}.pkl"
        else:
            study_name_clean = None
            save_path = f"svm_{args.granularity}.pkl"

        train_svm(
            train_df=train_df,
            test_df=test_df,
            c_val=1.0,
            kernel=args.kernel,
            save_path=save_path,
            granularity=args.granularity,
            run_optuna=True,
            reset_study=args.reset_study,
            trials=args.trials,
            trials_stage1=args.trials_stage1,
            trials_stage2=args.trials_stage2,
            tuning_strategy=args.tuning,
            tuning_sample_size=args.tuning_sample_size,
            score_metric=args.score,
            study_name=study_name_clean,
            n_jobs_optuna=args.n_jobs_optuna
        )

    if args.classifier in ['bert', 'both']:
        train_transformer(
            train_df=train_df,
            val_df=None,
            test_df=test_df,
            model_name=args.transformer_name,
            epochs=3,
            batch_size=8,
            lr=2e-5,
            save_path=f"./transformer_{args.granularity}_model",
            run_optuna=True
        )


if __name__ == "__main__":
    main()