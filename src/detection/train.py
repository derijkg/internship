# train.py
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import prepare_classification_dataset
from models.svm_model import train_svm
from models.transformer_model import train_transformer

DEFAULT_MODELS = ['qwen3.5:4b', 'qwen3.6:27b', 'gemma4:e4b', 'gemma4:26b']

def main():
    parser = argparse.ArgumentParser(description="LLM Detection Training Orchestrator")
    
    # Core Controls
    parser.add_argument('--data_path', type=str, required=True, help="Path to raw parquet or csv file")
    parser.add_argument('--classifier', type=str, choices=['svm', 'bert', 'both'], default='both', help="Model type to train")
    parser.add_argument('--granularity', type=str, choices=['full', 'sentence'], default='full', help="Train on full abstracts or split sentences")
    
    # Dataset Filtering Parameters
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS, help="LLM models to include in classification task")
    parser.add_argument('--source', nargs='+', default=['UG', 'SB', 'HBO'], help="Sources to isolate (default all: UG, SB, HBO)")
    
    # SVM Parameters
    parser.add_argument('--svm_c', type=float, default=1.0, help="C value penalty for SVM")
    parser.add_argument('--svm_kernel', type=str, default='rbf', help="Kernel type for SVM")
    
    # Transformer Parameters
    parser.add_argument('--transformer_name', type=str, default='pdelobelle/robbert-2023-dutch-base', help="HuggingFace model string")
    parser.add_argument('--epochs', type=int, default=3, help="Transformer training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Transformer batch size")
    parser.add_argument('--lr', type=float, default=2e-5, help="Transformer learning rate")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading data from: {args.data_path}")
    if args.data_path.endswith('.csv'):
        raw_df = pd.read_csv(args.data_path)
    elif args.data_path.endswith('.parquet') or args.data_path.endswith('.pq'):
        raw_df = pd.read_parquet(args.data_path)
    else:
        raise ValueError("Unsupported data format (Must be CSV or Parquet)")
        
    # Re-shape into balanced binary dataset
    processed_df = prepare_classification_dataset(
        raw_df, 
        selected_models=args.models, 
        granularity=args.granularity, 
        source_filter=args.source
    )
    
    print(f"Dataset compiled. Size: {len(processed_df)} rows. Target class balance:")
    print(processed_df['label'].value_counts())
    
    # Standard train/test split
    train_df, test_df = train_test_split(processed_df, test_size=0.2, random_state=42, stratify=processed_df['label'])
    
    # Train Models depending on user choice
    if args.classifier in ['svm', 'both']:
        train_svm(
            train_df=train_df, 
            test_df=test_df, 
            c_val=args.svm_c, 
            kernel=args.svm_kernel,
            save_path=f"svm_{args.granularity}.pkl"
        )
        
    if args.classifier in ['bert', 'both']:
        train_transformer(
            train_df=train_df, 
            test_df=test_df, 
            model_name=args.transformer_name, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            lr=args.lr,
            save_path=f"./transformer_{args.granularity}_model"
        )

if __name__ == "__main__":
    main()