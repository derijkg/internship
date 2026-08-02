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
    feature_input_dropout: float = 0.0     # [NEW] Regularization before RNN
    rnn_type: str = "LSTM"                 # [NEW] "LSTM" or "GRU"
    include_w3: bool = True
    include_w5: bool = True
    aux_boundary_weight: float = 0.5
    boundary_pos_weight: float = 5.0       # [NEW] Class imbalance weight for boundary loss
    emission_temp: float = 1.0             # [NEW] Temperature scaling for emissions
    use_attention: bool = True             # [NEW] Residual Multi-Head Self-Attention


@dataclass
class TrainingConfig:
    n_splits: int = 3
    epochs: int = 15
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    crf_lr_mult: float = 5.0               # [NEW] Higher LR multiplier for CRF layer
    scheduler_type: str = "cosine"         # [NEW] "cosine", "reduce_on_plateau", or "none"
    target_fpr: float = 0.01               # 1% Max FPR target for Neyman-Pearson calibration


@dataclass
class OptunaConfig:
    n_trials: int = 15                     # Number of hyperparameter trials
    search_epochs: int = 6                 # Max epochs per trial during hyperparameter search
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
        help="execution device (cuda or cpu)"
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
        help="Hidden dimension for projection and BiRNN",
    )
    parser.add_argument(
        "--rnn_type",
        type=str,
        default="LSTM",
        choices=["LSTM", "GRU"],
        help="Recurrent cell type (LSTM or GRU)",
    )
    parser.add_argument(
        "--feature_input_dropout",
        type=float,
        default=0.0,
        help="Dropout applied to fused features before RNN",
    )
    parser.add_argument(
        "--aux_boundary_weight",
        type=float,
        default=0.5,
        help="Weight for auxiliary boundary loss",
    )
    parser.add_argument(
        "--boundary_pos_weight",
        type=float,
        default=5.0,
        help="Positive class weight for boundary BCE loss",
    )
    parser.add_argument(
        "--emission_temp",
        type=float,
        default=1.0,
        help="Temperature scaling factor for emissions before CRF",
    )
    parser.add_argument(
        "--use_attention",
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="Enable residual multi-head self-attention layer (True/False)",
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
        "--crf_lr_mult",
        type=float,
        default=5.0,
        help="Learning rate multiplier for CRF layer parameters",
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["cosine", "reduce_on_plateau", "none"],
        help="Learning rate decay scheduler strategy",
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
        sample_size=args.sample_size,
        cache_dir=args.cache_dir,
        cache_file=args.cache_file,
    )

    model_cfg = ModelConfig(
        transformer_name=args.transformer_name,
        hidden_dim=args.hidden_dim,
        rnn_type=args.rnn_type,
        feature_input_dropout=args.feature_input_dropout,
        aux_boundary_weight=args.aux_boundary_weight,
        boundary_pos_weight=args.boundary_pos_weight,
        emission_temp=args.emission_temp,
        use_attention=args.use_attention,
        device=args.device,
    )

    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        crf_lr_mult=args.crf_lr_mult,
        scheduler_type=args.scheduler_type,
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