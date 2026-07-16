# train.py
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import prepare_classification_dataset
from models.svm import train_svm
from models.robbert import train_transformer

DEFAULT_MODELS = ['qwen3.5:4b', 'qwen3.6:27b', 'gemma4:e4b', 'gemma4:26b']

def main():
    parser = argparse.ArgumentParser(description="LLM Detection Training Orchestrator (Auto-Tuned)")
    
    # Core Controls
    parser.add_argument('--data_path', type=str, required=True, help="Path to raw parquet or csv file")
    parser.add_argument('--classifier', type=str, choices=['svm', 'bert', 'both'], default='both', help="Model type to train")
    parser.add_argument('--granularity', type=str, choices=['full', 'sentence'], default='full', help="Train on full abstracts or split sentences")
    
    # Dataset Filtering Parameters
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS, help="LLM models to include in classification task")
    parser.add_argument('--source', nargs='+', default=['UG', 'SB', 'HBO'], help="Sources to isolate (default all: UG, SB, HBO)")
    
    # Model Architecture Selection
    parser.add_argument('--transformer_name', type=str, default='pdelobelle/robbert-2023-dutch-base', help="HuggingFace model string")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading data from: {args.data_path}")
    if args.data_path.endswith('.csv'):
        raw_df = pd.read_csv(args.data_path)
    elif args.data_path.endswith('.parquet') or args.data_path.endswith('.pq'):
        raw_df = pd.read_parquet(args.data_path)
    else:
        raise ValueError("Unsupported data format (Must be CSV or Parquet)")
        
    # --- 1. Clean 3-Way Split on Raw Data ---
    stratify_col = raw_df['source'] if 'source' in raw_df.columns else None
    
    # Split out the 20% test set
    train_val_raw_df, test_raw_df = train_test_split(
        raw_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=stratify_col
    )
    
    # Split remaining 80% into Train (70% total) and Val (10% total)
    train_val_stratify = train_val_raw_df['source'] if 'source' in train_val_raw_df.columns else None
    train_raw_df, val_raw_df = train_test_split(
        train_val_raw_df,
        test_size=0.125, # 0.125 * 0.8 = 0.10 validation portion
        random_state=42,
        stratify=train_val_stratify
    )
    
    # --- 2. Shape splits independently ---
    train_df = prepare_classification_dataset(
        train_raw_df, selected_models=args.models, granularity=args.granularity, source_filter=args.source
    )
    val_df = prepare_classification_dataset(
        val_raw_df, selected_models=args.models, granularity=args.granularity, source_filter=args.source
    )
    test_df = prepare_classification_dataset(
        test_raw_df, selected_models=args.models, granularity=args.granularity, source_filter=args.source
    )
    
    print(f"\nDataset compiled successfully:")
    print(f"-> Train Split: {len(train_df)} rows")
    print(f"-> Val Split:   {len(val_df)} rows")
    print(f"-> Test Split:  {len(test_df)} rows\n")
    
    # Train Models depending on user choice (passing all 3 splits cleanly)
    if args.classifier in ['svm', 'both']:
        train_svm(
            train_df=train_df, 
            val_df=val_df,
            test_df=test_df, 
            c_val=1.0,               
            kernel='rbf',            
            save_path=f"svm_{args.granularity}.pkl",
            granularity=args.granularity,
            run_optuna=True          
        )
        
    if args.classifier in ['bert', 'both']:
        train_transformer(
            train_df=train_df, 
            val_df=val_df,
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