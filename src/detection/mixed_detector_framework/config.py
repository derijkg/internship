# config.py
import torch
import argparse
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    data_path: str = "/home/gderijck/internship/data/gold/llm_added.parquet"
    sample_size: Optional[int] = None  # Set default to None so full dataset is processed unless specified
    min_sentences: int = 3
    cache_dir: str = "./.feature_cache"
    cache_file: Optional[str] = None  # Explicit cache filename/path to force load
    random_seed: int = 42


@dataclass
class ModelConfig:
    transformer_name: str = "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"
    device: str = "cuda" if torch.cuda.is_available() else 'cpu'
    fused_in_dim: int = 852  # 768 dense + 84 stylometrics
    hidden_dim: int = 256
    num_lstm_layers: int = 2
    dropout: float = 0.3
    include_w3: bool = True
    include_w5: bool = True
    aux_boundary_weight: float = 0.5


@dataclass
class TrainingConfig:
    n_splits: int = 3
    epochs: int = 15
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    target_fpr: float = 0.01  # 1% Max FPR target for Neyman-Pearson calibration


@dataclass
class OptunaConfig:
    n_trials: int = 15          # Number of hyperparameter trials
    search_epochs: int = 6       # Max epochs per trial during hyperparameter search
    study_name: str = "neural_crf_optuna"
    storage: Optional[str] = None


@dataclass
class MasterConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)


def parse_args_into_config() -> MasterConfig:
    """Parses command-line arguments and merges them into a structured
    MasterConfig object.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Scale Neural CRF Mixed-Authorship Segment Detector"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else 'cpu',
        help="execution device (cuda cpu or guillotine)"
    )

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to raw dataset (.parquet or .csv)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Subsample raw docs (None for full dataset)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./.feature_cache",
        help="Directory for caching fused feature matrices",
    )
    parser.add_argument(
        "--cache_file",
        type=str,
        default=None,
        help="Explicit filename or path of cached features to force load (e.g. fused_features_xyz.pt)",
    )

    # Model arguments
    parser.add_argument(
        "--transformer_name",
        type=str,
        default="NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for projection and BiLSTM",
    )
    parser.add_argument(
        "--aux_boundary_weight",
        type=float,
        default=0.5,
        help="Weight for auxiliary boundary loss",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Training epochs per fold",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (number of documents per batch)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=3,
        help="Number of Stratified Group K-Fold splits",
    )
    parser.add_argument(
        "--target_fpr",
        type=float,
        default=0.01,
        help="Neyman-Pearson target FPR constraint (default: 0.01 = 1%)",
    )

    # Optuna Arguments
    parser.add_argument(
        "--optuna_trials",
        type=int,
        default=15,
        help="Number of trials for Optuna hyperparameter optimization",
    )
    parser.add_argument(
        "--optuna_epochs",
        type=int,
        default=6,
        help="Maximum epochs per Optuna search trial",
    )
    parser.add_argument(
        "--optuna_study_name",
        type=str,
        default="neural_crf_optuna",
        help="Name of the Optuna study",
    )

    args = parser.parse_args()

    default_data_cfg = DataConfig()
    data_cfg = DataConfig(
        data_path=args.data_path if args.data_path else default_data_cfg.data_path,
        sample_size=args.sample_size,  # Kept as None if omitted
        cache_dir=args.cache_dir,
        cache_file=args.cache_file,
    )

    model_cfg = ModelConfig(
        transformer_name=args.transformer_name,
        hidden_dim=args.hidden_dim,
        aux_boundary_weight=args.aux_boundary_weight,
        device=args.device
    )

    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_splits=args.n_splits,
        target_fpr=args.target_fpr,
    )

    optuna_cfg = OptunaConfig(
        n_trials=args.optuna_trials,
        search_epochs=args.optuna_epochs,
        study_name=args.optuna_study_name,
    )

    return MasterConfig(
        data=data_cfg,
        model=model_cfg,
        training=train_cfg,
        optuna=optuna_cfg,
    )