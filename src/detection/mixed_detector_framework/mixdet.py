#!/usr/bin/env python3
"""
===============================================================================
MULTI-SCALE NEURAL-CRF MIXED-AUTHORSHIP SEGMENT DETECTOR (CUDA-OPTIMIZED)
===============================================================================
Consolidated Framework containing:
  - Section 0: CPU Thread Limits & Environment Setup
  - Section 1: Configuration Dataclasses & Argument Parser
  - Section 2: Data Abstractions & Collator
  - Section 3: Feature Extraction Engines (Dense Transformer & Stylometrics)
  - Section 4: Neural CRF Model Architecture
  - Section 5: Neyman-Pearson Calibration & Multi-Level Evaluation
  - Section 6: Training Pipeline & Optuna Hyperparameter Tuning
  - Section 7: Main Entry Point
===============================================================================
"""

# =============================================================================
# SECTION 0: CPU THREAD LIMITS & ENVIRONMENT SETUP
# =============================================================================
import os
import gc
import json
import re
import string
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

# Restrict background thread pools to prevent CPU contention/deadlocks
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from data.dataset import build_dataset


# Set PyTorch CPU thread counts
torch.set_num_threads(4)
torch.set_num_interop_threads(2)


# =============================================================================
# SECTION 1: CONFIGURATION DATACLASSES
# =============================================================================
@dataclass
class DataConfig:
    """Dataset and caching configurations."""
    data_path: str = "/home/gderijck/internship/data/gold/llm_added.parquet"
    sample_size: Optional[int] = None
    min_sentences: int = 3
    cache_dir: str = "./.feature_cache"
    cache_file: Optional[str] = None
    random_seed: int = 42

@dataclass
class ModelConfig:
    """Neural Network & CRF Layer Hyperparameters."""
    transformer_name: str = "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fused_in_dim: int = 852  # 768 dense + 84 stylometrics
    hidden_dim: int = 256
    num_lstm_layers: int = 2
    dropout: float = 0.3
    feature_input_dropout: float = 0.0
    rnn_type: str = "LSTM"  # "LSTM" or "GRU"
    include_w3: bool = True
    include_w5: bool = True
    aux_boundary_weight: float = 0.5
    boundary_pos_weight: float = 5.0
    emission_temp: float = 1.0
    use_attention: bool = True

@dataclass
class TrainingConfig:
    """Cross-validation and optimization settings."""
    n_splits: int = 3
    epochs: int = 15
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    crf_lr_mult: float = 5.0
    scheduler_type: str = "cosine"  # "cosine", "reduce_on_plateau", or "none"
    target_fpr: float = 0.01  # 1% Max FPR target for Neyman-Pearson calibration

@dataclass
class OptunaConfig:
    """Optuna hyperparameter tuning settings."""
    n_trials: int = 50
    search_epochs: int = 8
    study_name: str = "neural_crf_optuna"
    storage_dir: str = "/home/gderijck/internship/optuna_studies"

@dataclass
class MasterConfig:
    """Master configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)

def parse_args_into_config() -> MasterConfig:
    """Parses command-line arguments and merges them into MasterConfig."""
    parser = argparse.ArgumentParser(description="Multi-Scale Neural CRF Segment Detector")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="/home/gderijck/internship/data/gold/llm_added.parquet")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default="./.feature_cache")
    parser.add_argument("--cache_file", type=str, default=None)

    # Model arguments
    parser.add_argument("--transformer_name", type=str, default="NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--rnn_type", type=str, default="LSTM", choices=["LSTM", "GRU"])
    parser.add_argument("--feature_input_dropout", type=float, default=0.0)
    parser.add_argument("--aux_boundary_weight", type=float, default=0.5)
    parser.add_argument("--boundary_pos_weight", type=float, default=5.0)
    parser.add_argument("--emission_temp", type=float, default=1.0)
    parser.add_argument("--use_attention", type=lambda x: (str(x).lower() == 'true'), default=True)

    # Training arguments
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--crf_lr_mult", type=float, default=5.0)
    parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["cosine", "reduce_on_plateau", "none"])
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--target_fpr", type=float, default=0.01)

    # Optuna Arguments
    parser.add_argument("--optuna_trials", type=int, default=50)
    parser.add_argument("--optuna_epochs", type=int, default=8)
    parser.add_argument("--optuna_study_name", type=str, default=None)
    parser.add_argument("--optuna_storage_dir", type=str, default="/home/gderijck/internship/optuna_studies")

    args = parser.parse_args()

    # Construct dynamic Optuna study name based on non-tuned launch parameters if not explicitly provided
    if not args.optuna_study_name or args.optuna_study_name == "neural_crf_optuna":
        clean_model_tag = args.transformer_name.split("/")[-1].replace("-sentence-transformers", "").replace("-", "_")
        args.optuna_study_name = f"crf_{clean_model_tag}_k{args.n_splits}_b{args.batch_size}_fpr{int(args.target_fpr * 1000)}"

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
        storage_dir=args.optuna_storage_dir,
    )

    return MasterConfig(data=data_cfg, model=model_cfg, training=train_cfg, optuna=optuna_cfg)


# =============================================================================
# SECTION 2: DATA STRUCTURES, DATASET, & COLLATOR
# =============================================================================
class ScaledFoldDataset(Dataset):
    """
    Wraps document datasets for a specific fold, applying standardized 
    fold-safe stylometric feature matrices.
    """
    def __init__(self, base_dataset: Any, scaled_features_list: List[np.ndarray], indices: np.ndarray):
        self.base_dataset = base_dataset
        self.scaled_features_list = scaled_features_list
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self.indices[idx]
        doc = self.base_dataset.docs[real_idx]
        fused_features = self.scaled_features_list[real_idx]
        feature_len = len(fused_features)

        return {
            "doc_id": doc.doc_id,
            "parent_doc_id": doc.parent_doc_id,
            "scenario": doc.scenario,
            "fused_features": torch.tensor(fused_features, dtype=torch.float32),
            "labels": torch.tensor(doc.labels[:feature_len], dtype=torch.long),
            "boundaries": torch.tensor(doc.boundaries[:feature_len], dtype=torch.long),
            "seq_len": feature_len,
        }

def pad_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Dynamic padding collator for batching variable-length document sequences.
    Directly packages CPU sequence lengths to avoid GPU synchronization stalls.
    """
    batch_size = len(batch)
    max_len = max(item["seq_len"] for item in batch)
    feat_dim = batch[0]["fused_features"].shape[1]

    padded_features = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
    padded_labels = torch.zeros((batch_size, max_len), dtype=torch.long)
    padded_boundaries = torch.zeros((batch_size, max_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)

    doc_ids, parent_doc_ids, scenarios = [], [], []

    for i, item in enumerate(batch):
        seq_len = item["seq_len"]
        padded_features[i, :seq_len] = item["fused_features"]
        padded_labels[i, :seq_len] = item["labels"]
        padded_boundaries[i, :seq_len] = item["boundaries"]
        mask[i, :seq_len] = True
        seq_lengths[i] = seq_len

        doc_ids.append(item["doc_id"])
        parent_doc_ids.append(item["parent_doc_id"])
        scenarios.append(item["scenario"])

    return {
        "doc_ids": doc_ids,
        "parent_doc_ids": parent_doc_ids,
        "scenarios": scenarios,
        "fused_features": padded_features,
        "labels": padded_labels,
        "boundaries": padded_boundaries,
        "mask": mask,
        "seq_lengths": seq_lengths,
    }


# =============================================================================
# SECTION 3: FEATURE EXTRACTION ENGINES
# =============================================================================
def get_optimal_device(requested_device: str = "cpu") -> torch.device:
    """Safely determines execution device."""
    if "cuda" in requested_device and torch.cuda.is_available():
        return torch.device(requested_device)
    return torch.device("cpu")

class DenseTransformerEncoder(nn.Module):
    """
    Extracts dense sentence embeddings using a Dutch Transformer 
    (e.g., NFI RobBERT Sentence Transformer).
    """
    def __init__(
        self,
        model_name: str = "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers",
        device: str = "cpu",
        max_length: int = 128
    ):
        super().__init__()
        self.device = get_optimal_device(device)
        self.model_name = model_name
        self.max_length = max_length
        self.hidden_dim = 768

        print(f"Loading Pretrained Dutch Transformer '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.encoder.eval()

        for param in self.encoder.parameters():
            param.requires_grad = False

    def extract_sentence_embeddings(self, sents: List[str], batch_size: int = 64) -> torch.Tensor:
        """Extracts mean-pooled sentence embeddings in mini-batches."""
        if not sents:
            return torch.zeros((0, self.hidden_dim))

        all_embeddings = []
        for i in range(0, len(sents), batch_size):
            batch_sents = sents[i : i + batch_size]
            encoded = self.tokenizer(
                batch_sents,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.encoder(**encoded)
                token_embeddings = outputs.last_hidden_state  # [Batch, SeqLen, 768]
                
                mask = encoded['attention_mask'].unsqueeze(-1).float()
                sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                
                batch_embeddings = (sum_embeddings / sum_mask).cpu()
                all_embeddings.append(batch_embeddings)

        return torch.cat(all_embeddings, dim=0)

DUTCH_TRANSITIONS = {
    "echter", "bovendien", "daarnaast", "desalniettemin", "kortom",
    "tevens", "daardoor", "derhalve", "bijgevolg", "namelijk",
    "hoewel", "aldus", "immers", "enerzijds", "anderzijds",
}

class StylometricFeatureEngine:
    """
    Computes 12-dim stylometrics across multi-scale windows (W1, W3, W5),
    global relative deltas (Δ_global), and local gradients (∇_local).
    """
    def __init__(self, include_w3: bool = True, include_w5: bool = True):
        self.include_w3 = include_w3
        self.include_w5 = include_w5

    @staticmethod
    def extract_raw_vector(text: str, sentences: List[str]) -> np.ndarray:
        """Extracts a 12-dimensional statistical surface feature vector."""
        words = re.findall(r"\w+", text.lower())
        total_chars = max(1, len(text))
        num_words = max(1, len(words))

        if not words or not sentences:
            return np.zeros(12, dtype=np.float32)

        # 1-3. Sentence length metrics
        sent_lengths = [
            len(re.findall(r"\w+", s))
            for s in sentences if len(re.findall(r"\w+", s)) > 0
        ]
        mean_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
        var_sent_len = float(np.var(sent_lengths)) if sent_lengths else 0.0
        burstiness = (float(np.std(sent_lengths)) / mean_sent_len) if mean_sent_len > 0 else 0.0

        # 4-5. Word length metrics
        word_lengths = [len(w) for w in words]
        mean_word_len = float(np.mean(word_lengths))
        var_word_len = float(np.var(word_lengths))

        # 6-7. Vocabulary richness
        unique_words = set(words)
        ttr = len(unique_words) / num_words
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        hapax_ratio = sum(1 for w, c in word_counts.items() if c == 1) / num_words

        # 8. Discourse transition ratio
        transition_count = sum(1 for w in words if w in DUTCH_TRANSITIONS)
        transition_ratio = transition_count / num_words

        # 9-11. Whitespace and punctuation ratios
        spaces_count = text.count(" ")
        double_spaces = text.count("  ")
        punc_count = sum(1 for c in text if c in string.punctuation)

        space_ratio = spaces_count / total_chars
        double_space_ratio = double_spaces / total_chars
        punc_ratio = punc_count / total_chars

        # 12. Log character length
        log_char_len = float(np.log1p(total_chars))

        return np.array([
            mean_sent_len, var_sent_len, burstiness, mean_word_len,
            var_word_len, ttr, hapax_ratio, transition_ratio,
            space_ratio, double_space_ratio, punc_ratio, log_char_len
        ], dtype=np.float32)

    def compute_document_features(self, sents: List[str]) -> Tuple[np.ndarray, int]:
        """Calculates fused multi-scale features for a sequence of sentences."""
        N = len(sents)
        if N == 0:
            return np.zeros((0, 12), dtype=np.float32), 12

        doc_text = " ".join(sents)
        doc_style = self.extract_raw_vector(doc_text, sents)

        w1_styles = np.zeros((N, 12), dtype=np.float32)
        w3_styles = np.zeros((N, 12), dtype=np.float32)
        w5_styles = np.zeros((N, 12), dtype=np.float32)

        for i in range(N):
            # W1: Sentence i
            w1_styles[i] = self.extract_raw_vector(sents[i], [sents[i]])

            # W3: Window [i-1, i, i+1]
            if self.include_w3:
                w3_sents = sents[max(0, i - 1) : min(N, i + 2)]
                w3_styles[i] = self.extract_raw_vector(" ".join(w3_sents), w3_sents)

            # W5: Window [i-2 ... i+2]
            if self.include_w5:
                w5_sents = sents[max(0, i - 2) : min(N, i + 3)]
                w5_styles[i] = self.extract_raw_vector(" ".join(w5_sents), w5_sents)

        feature_blocks = [w1_styles]

        if self.include_w3:
            w3_delta_global = w3_styles - doc_style
            w3_grad_local = np.zeros_like(w3_styles)
            w3_grad_local[1:] = w3_styles[1:] - w3_styles[:-1]
            feature_blocks.extend([w3_styles, w3_delta_global, w3_grad_local])

        if self.include_w5:
            w5_delta_global = w5_styles - doc_style
            w5_grad_local = np.zeros_like(w5_styles)
            w5_grad_local[1:] = w5_styles[1:] - w5_styles[:-1]
            feature_blocks.extend([w5_styles, w5_delta_global, w5_grad_local])

        fused_matrix = np.hstack(feature_blocks).astype(np.float32)
        return fused_matrix, fused_matrix.shape[1]


# =============================================================================
# SECTION 4: NEURAL CRF MODEL ARCHITECTURE
# =============================================================================
class LinearChainCRF(nn.Module):
    """
    Pure PyTorch Linear-Chain Conditional Random Field (CRF).
    Supports Log-partition Z(x), Viterbi decoding, and Forward-Backward marginals.
    """
    def __init__(self, num_tags: int = 2):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def _compute_score(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        if seq_len == 0:
            return torch.zeros(batch_size, device=emissions.device)

        mask_float = mask.float()
        score = self.start_transitions[tags[:, 0]] * mask_float[:, 0]
        emit_scores = emissions.gather(2, tags.unsqueeze(-1)).squeeze(-1)
        score = score + (emit_scores * mask_float).sum(dim=1)

        if seq_len > 1:
            trans_scores = self.transitions[tags[:, :-1], tags[:, 1:]]
            trans_mask = mask_float[:, 1:] * mask_float[:, :-1]
            score = score + (trans_scores * trans_mask).sum(dim=1)

        seq_lengths = mask.sum(dim=1).long()
        last_valid_indices = torch.clamp(seq_lengths - 1, min=0)
        last_tags = tags.gather(1, last_valid_indices.unsqueeze(1)).squeeze(1)

        has_valid = (seq_lengths > 0).float()
        score = score + self.end_transitions[last_tags] * has_valid
        return score

    def _compute_log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_tags = emissions.shape
        if seq_len == 0:
            return torch.zeros(batch_size, device=emissions.device)

        mask_bool = mask.bool()
        forward_var = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        init_mask = mask_bool[:, 0].unsqueeze(1)
        forward_var = torch.where(init_mask, forward_var, torch.zeros_like(forward_var))

        for i in range(1, seq_len):
            mask_i = mask_bool[:, i].unsqueeze(1)
            emit_score = emissions[:, i].unsqueeze(1)
            trans_score = self.transitions.unsqueeze(0)
            next_tag_var = forward_var.unsqueeze(2) + trans_score + emit_score
            forward_var_next = torch.logsumexp(next_tag_var, dim=1)
            forward_var = torch.where(mask_i, forward_var_next, forward_var)

        forward_var = forward_var + self.end_transitions.unsqueeze(0)
        has_valid = mask.sum(dim=1) > 0
        log_z = torch.logsumexp(forward_var, dim=1)
        return torch.where(has_valid, log_z, torch.zeros_like(log_z))

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: torch.Tensor,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> torch.Tensor:
        """Computes Negative Log-Likelihood loss: Loss = Z(x) - S(x, y)."""
        log_partition = self._compute_log_partition(emissions, mask)
        path_score = self._compute_score(emissions, tags, mask)
        nll = log_partition - path_score

        if reduction == "none":
            return nll
        elif reduction == "sum":
            return torch.sum(nll)
        return torch.mean(nll)

    def viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """Finds globally optimal sequence path y* using Viterbi decoding."""
        batch_size, seq_len, _ = emissions.shape
        if seq_len == 0:
            return [[] for _ in range(batch_size)]

        mask_bool = mask.bool()
        viterbi_vars = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        history = []

        for i in range(1, seq_len):
            mask_i = mask_bool[:, i].unsqueeze(1)
            broadcast_viterbi = viterbi_vars.unsqueeze(2) + self.transitions.unsqueeze(0)
            max_vars, bptrs = torch.max(broadcast_viterbi, dim=1)

            viterbi_vars_next = max_vars + emissions[:, i]
            viterbi_vars = torch.where(mask_i, viterbi_vars_next, viterbi_vars)
            history.append(bptrs)

        viterbi_vars = viterbi_vars + self.end_transitions.unsqueeze(0)
        best_last_tags = torch.argmax(viterbi_vars, dim=1).cpu().tolist()
        seq_lengths = mask.sum(dim=1).long().cpu().tolist()

        history_numpy = torch.stack(history, dim=0).cpu().numpy() if history else None

        best_paths = []
        for b in range(batch_size):
            seq_len_b = seq_lengths[b]
            if seq_len_b == 0:
                best_paths.append([])
                continue

            best_tag = best_last_tags[b]
            best_path = [best_tag]

            if history_numpy is not None:
                for step_idx in range(seq_len_b - 2, -1, -1):
                    best_tag = int(history_numpy[step_idx, b, best_tag])
                    best_path.append(best_tag)

            best_path.reverse()
            best_paths.append(best_path)

        return best_paths

    def compute_marginal_probabilities(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Computes marginal posterior probabilities P(y_i = k | x) via Forward-Backward (Numerically Safe)."""
        batch_size, seq_len, num_tags = emissions.shape
        if seq_len == 0:
            return torch.zeros((batch_size, 0, num_tags), device=emissions.device)

        mask_bool = mask.bool()

        # 1. Forward Pass
        forward_vars = torch.zeros_like(emissions)
        forward_var = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        init_mask = mask_bool[:, 0].unsqueeze(1)
        forward_var = torch.where(init_mask, forward_var, torch.zeros_like(forward_var))
        forward_vars[:, 0] = forward_var

        for i in range(1, seq_len):
            mask_i = mask_bool[:, i].unsqueeze(1)
            emit_score = emissions[:, i].unsqueeze(1)
            trans_score = self.transitions.unsqueeze(0)
            next_tag_var = forward_var.unsqueeze(2) + trans_score + emit_score
            forward_var_next = torch.logsumexp(next_tag_var, dim=1)
            forward_var = torch.where(mask_i, forward_var_next, forward_var)
            forward_vars[:, i] = forward_var

        # 2. Backward Pass
        backward_vars = torch.zeros_like(emissions)
        backward_var = self.end_transitions.unsqueeze(0)

        for i in range(seq_len - 1, -1, -1):
            if i < seq_len - 1:
                mask_i = mask_bool[:, i + 1].unsqueeze(1)
                emit_score = emissions[:, i + 1].unsqueeze(1)
                trans_score = self.transitions.unsqueeze(0)
                next_tag_var = backward_var.unsqueeze(1) + trans_score + emit_score
                backward_var_next = torch.logsumexp(next_tag_var, dim=2)
                backward_var = torch.where(mask_i, backward_var_next, backward_var)
            backward_vars[:, i] = backward_var

        # 3. Combine Marginals safely: P(y_i | x) = exp(fwd + bwd - log_Z)
        log_marginals = forward_vars + backward_vars
        log_z = torch.logsumexp(forward_vars[:, -1] + self.end_transitions.unsqueeze(0), dim=-1, keepdim=True)
        
        # Clamp log difference to prevent float overflow/underflow and nan multiplication
        log_diff = torch.clamp(log_marginals - log_z.unsqueeze(1), min=-80.0, max=0.0)
        marginals = torch.exp(log_diff)
        marginals = torch.nan_to_num(marginals, nan=0.0, posinf=0.0, neginf=0.0)

        return marginals * mask.float().unsqueeze(-1)


class MultiTaskNeuralCRFTagger(nn.Module):
    """
    Multimodal Neural CRF Tagger with Gated Multimodal Unit (GMU) dynamic fusion,
    BiRNN (LSTM/GRU), Residual Self-Attention, and auxiliary boundary loss.
    Fully optimized for CUDA execution and safe attention masking.
    """
    def __init__(
        self,
        dense_dim: int = 768,
        stylo_dim: int = 84,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        feature_input_dropout: float = 0.0,
        rnn_type: str = "LSTM",
        aux_boundary_weight: float = 0.4,
        boundary_pos_weight: float = 5.0,
        emission_temp: float = 1.0,
        use_attention: bool = True,
        aux_pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.dense_dim = dense_dim
        self.stylo_dim = stylo_dim
        self.hidden_dim = hidden_dim
        self.aux_boundary_weight = aux_boundary_weight
        self.rnn_type = rnn_type.upper()
        self.emission_temp = emission_temp

        pos_weight_val = boundary_pos_weight if aux_pos_weight is None else aux_pos_weight

        # Register pos_weight as a persistent module buffer to prevent device mismatches on CUDA
        self.register_buffer("boundary_pos_weight", torch.tensor([pos_weight_val], dtype=torch.float32))

        # Stylometric Normalization & Dual-Branch Projections
        self.stylo_layernorm = nn.LayerNorm(stylo_dim)

        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.stylo_projection = nn.Sequential(
            nn.Linear(stylo_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Gated Multimodal Unit (GMU) Fusion Gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.input_dropout = nn.Dropout(p=feature_input_dropout)

        # Sequence Context Encoder (BiLSTM or BiGRU)
        rnn_cls = nn.GRU if self.rnn_type == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        self.use_attention = use_attention
        if self.use_attention:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.attn_layernorm = nn.LayerNorm(hidden_dim)

        # Emission & Boundary Heads
        self.emission_head = nn.Linear(hidden_dim, 2)
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.crf = LinearChainCRF(num_tags=2)

    def forward(
        self,
        fused_features: torch.Tensor,
        mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        boundaries: Optional[torch.Tensor] = None,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        dense_feats = fused_features[..., : self.dense_dim]
        stylo_feats = fused_features[..., self.dense_dim :]

        stylo_feats = self.stylo_layernorm(stylo_feats)

        proj_dense = self.dense_projection(dense_feats)
        proj_stylo = self.stylo_projection(stylo_feats)

        gate_input = torch.cat([proj_dense, proj_stylo], dim=-1)
        gate = self.fusion_gate(gate_input)

        projected = gate * proj_dense + (1.0 - gate) * proj_stylo
        projected = self.input_dropout(projected)

        # Use CPU lengths tensor directly if provided to prevent CPU-GPU sync stalls
        if seq_lengths is None:
            seq_lengths_cpu = torch.clamp(mask.sum(dim=1).long(), min=1).cpu()
        else:
            seq_lengths_cpu = torch.clamp(seq_lengths, min=1).cpu()

        packed_input = nn.utils.rnn.pack_padded_sequence(
            projected, seq_lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.rnn(packed_input)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=fused_features.size(1)
        )

        if self.use_attention:
            padding_mask = ~mask.bool()
            # Prevent all-True rows in key_padding_mask to avoid NaNs in PyTorch SDPA CUDA kernels
            all_masked_rows = padding_mask.all(dim=-1, keepdim=True)
            safe_padding_mask = padding_mask & ~all_masked_rows

            attn_out, _ = self.self_attn(
                query=rnn_out, key=rnn_out, value=rnn_out, key_padding_mask=safe_padding_mask
            )
            # Mask attn_out for padded tokens before residual addition
            attn_out = attn_out * mask.unsqueeze(-1).float()
            rnn_out = self.attn_layernorm(rnn_out + attn_out)

        emissions = self.emission_head(rnn_out) / self.emission_temp

        zeros = torch.zeros_like(rnn_out[:, :1, :])
        shifted_rnn = torch.cat([zeros, rnn_out[:, :-1, :]], dim=1)

        diff = torch.abs(rnn_out - shifted_rnn)
        prod = rnn_out * shifted_rnn

        boundary_inputs = torch.cat([rnn_out, shifted_rnn, diff, prod], dim=-1)
        boundary_logits = self.boundary_head(boundary_inputs).squeeze(-1)

        outputs = {"emissions": emissions, "boundary_logits": boundary_logits}

        if labels is not None and boundaries is not None:
            crf_seq_nll = self.crf(emissions, labels.long(), mask, reduction="none")
            total_active_tokens = torch.clamp(mask.float().sum(), min=1.0)
            crf_token_loss = crf_seq_nll.sum() / total_active_tokens

            bce_raw = F.binary_cross_entropy_with_logits(
                boundary_logits, boundaries.float(), pos_weight=self.boundary_pos_weight, reduction="none"
            )
            bce_masked = (bce_raw * mask.float()).sum() / total_active_tokens

            total_loss = crf_token_loss + self.aux_boundary_weight * bce_masked

            outputs["loss"] = total_loss
            outputs["crf_loss"] = crf_token_loss
            outputs["boundary_loss"] = bce_masked

        return outputs

    def predict(
        self,
        fused_features: torch.Tensor,
        mask: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Runs sequence decoding and marginal state inference."""
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                outputs = self.forward(fused_features, mask, seq_lengths=seq_lengths)
                emissions = outputs["emissions"]

                marginals = self.crf.compute_marginal_probabilities(emissions, mask)
                ai_probabilities = marginals[:, :, 1] * mask.float()
                boundary_probs = torch.sigmoid(outputs["boundary_logits"]) * mask.float()

                return {
                    "probabilities": ai_probabilities,
                    "boundary_probabilities": boundary_probs,
                }
        finally:
            if was_training:
                self.train()


# =============================================================================
# SECTION 5: CALIBRATION & EVALUATION METRICS
# =============================================================================
class NeymanPearsonCalibrator:
    """
    Calculates optimal decision threshold tau* that enforces a strict upper bound
    on False Positive Rate (FPR <= target_fpr).
    """
    def __init__(self, target_fpr: float = 0.01):
        self.target_fpr = target_fpr
        self.optimal_threshold = 0.50

    def fit(self, y_true: np.ndarray, y_probs: np.ndarray) -> float:
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        valid_indices = np.where(fpr <= self.target_fpr)[0]

        if len(valid_indices) > 0:
            best_idx = valid_indices[-1]
            self.optimal_threshold = float(np.clip(thresholds[best_idx], 0.0, 1.0))
        else:
            self.optimal_threshold = 0.50

        print(f"-> Neyman-Pearson Calibrated Threshold (FPR <= {self.target_fpr*100:.1f}%): τ* = {self.optimal_threshold:.6f}")
        return self.optimal_threshold

    def predict(self, y_probs: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        tau = threshold if threshold is not None else self.optimal_threshold
        return (y_probs >= tau).astype(int)

def calibrate_boundary_threshold(
    b_true_list: List[np.ndarray],
    b_probs_list: List[np.ndarray],
    search_step: float = 0.02,
) -> float:
    """
    Finds the optimal boundary decision threshold by maximizing F1 score 
    on validation/calibration data.
    """
    flat_b_true = np.concatenate(b_true_list)
    flat_b_probs = np.concatenate(b_probs_list)

    best_thresh = 0.50
    best_f1 = -1.0

    for t in np.arange(0.10, 0.90, search_step):
        b_pred_t = (flat_b_probs >= t).astype(int)
        f1_t = f1_score(flat_b_true, b_pred_t, pos_label=1, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = float(t)

    print(f"-> Calibrated Boundary Threshold (Max F1): τ_boundary = {best_thresh:.4f}")
    return best_thresh

def evaluate_mixed_authorship_performance(
    all_y_true: List[np.ndarray],
    all_y_probs: List[np.ndarray],
    all_b_true: List[np.ndarray],
    all_b_probs: List[np.ndarray],
    threshold: float = 0.50,
    boundary_threshold: float = 0.50,
) -> Dict[str, float]:
    """Computes comprehensive multi-level evaluation metrics without threshold leakage."""
    flat_y_true = np.concatenate(all_y_true)
    flat_y_probs = np.concatenate(all_y_probs)
    flat_y_pred = (flat_y_probs >= threshold).astype(int)

    flat_b_true = np.concatenate(all_b_true)
    flat_b_probs = np.concatenate(all_b_probs)
    flat_b_pred = (flat_b_probs >= boundary_threshold).astype(int)

    # 1. Sentence-Level Metrics
    sent_prec = precision_score(flat_y_true, flat_y_pred, pos_label=1, zero_division=0)
    sent_rec = recall_score(flat_y_true, flat_y_pred, pos_label=1, zero_division=0)
    sent_f1 = f1_score(flat_y_true, flat_y_pred, pos_label=1, zero_division=0)

    # Safe ROC-AUC calculation to prevent exception on single class splits
    if len(np.unique(flat_y_true)) > 1:
        sent_auc = roc_auc_score(flat_y_true, flat_y_probs)
    else:
        sent_auc = 0.50

    # 2. Boundary Transition Metrics
    bound_prec = precision_score(flat_b_true, flat_b_pred, pos_label=1, zero_division=0)
    bound_rec = recall_score(flat_b_true, flat_b_pred, pos_label=1, zero_division=0)
    bound_f1 = f1_score(flat_b_true, flat_b_pred, pos_label=1, zero_division=0)

    # 3. Document-Level Span IoU & AI Ratio MAE
    ious = []
    ratio_errors = []

    for y_t, y_p_prob in zip(all_y_true, all_y_probs):
        y_p = (y_p_prob >= threshold).astype(int)
        intersection = np.sum((y_t == 1) & (y_p == 1))
        union = np.sum((y_t == 1) | (y_p == 1))
        iou = (intersection / union) if union > 0 else 1.0
        ious.append(iou)

        true_ratio = np.mean(y_t)
        pred_ratio = np.mean(y_p)
        ratio_errors.append(abs(true_ratio - pred_ratio))

    mean_iou = float(np.mean(ious))
    mean_ratio_mae = float(np.mean(ratio_errors))

    return {
        "sent_precision_ai": sent_prec,
        "sent_recall_ai": sent_rec,
        "sent_f1_ai": sent_f1,
        "sent_roc_auc": sent_auc,
        "boundary_precision": bound_prec,
        "boundary_recall": bound_rec,
        "boundary_f1": bound_f1,
        "span_iou": mean_iou,
        "ai_ratio_mae": mean_ratio_mae,
    }


# =============================================================================
# SECTION 6: TRAINING PIPELINE & OPTUNA HYPERPARAMETER TUNING
# =============================================================================
def run_training_pipeline(cfg: MasterConfig, dataset: Any):
    """Executes full Stratified Group K-Fold Cross-Validation training & evaluation."""
    print("=========================================================")
    print(" STAGE 1: TRAINING PIPELINE SETUP ")
    print("=========================================================")

    # Look for best_params.json in current directory or optuna_studies directory
    optuna_best_params_file = os.path.join(cfg.optuna.storage_dir, f"best_params_{cfg.optuna.study_name}.json")
    param_file = None
    if os.path.exists("best_params.json"):
        param_file = "best_params.json"
    elif os.path.exists(optuna_best_params_file):
        param_file = optuna_best_params_file

    if param_file:
        print(f"\n[INFO] Found '{param_file}'. Applying all Optuna tuned hyperparameters!")
        with open(param_file, "r") as f:
            best_params = json.load(f)

        # Optimization & Training params
        cfg.training.learning_rate = best_params.get("learning_rate", cfg.training.learning_rate)
        cfg.training.weight_decay = best_params.get("weight_decay", cfg.training.weight_decay)
        cfg.training.crf_lr_mult = best_params.get("crf_lr_mult", cfg.training.crf_lr_mult)
        cfg.training.scheduler_type = best_params.get("scheduler_type", cfg.training.scheduler_type)

        # Model architecture params
        cfg.model.aux_boundary_weight = best_params.get("aux_boundary_weight", cfg.model.aux_boundary_weight)
        cfg.model.boundary_pos_weight = best_params.get("boundary_pos_weight", cfg.model.boundary_pos_weight)
        cfg.model.dropout = best_params.get("dropout", cfg.model.dropout)
        cfg.model.feature_input_dropout = best_params.get("feature_input_dropout", cfg.model.feature_input_dropout)
        cfg.model.hidden_dim = best_params.get("hidden_dim", cfg.model.hidden_dim)
        cfg.model.num_lstm_layers = best_params.get("num_lstm_layers", cfg.model.num_lstm_layers)
        cfg.model.rnn_type = best_params.get("rnn_type", cfg.model.rnn_type)
        cfg.model.emission_temp = best_params.get("emission_temp", cfg.model.emission_temp)
        cfg.model.use_attention = best_params.get("use_attention", cfg.model.use_attention)

    parent_ids = np.array([doc.parent_doc_id for doc in dataset.docs])
    scenarios = np.array([doc.scenario for doc in dataset.docs])

    scenario_counts = pd.Series(scenarios).value_counts()
    if (scenario_counts < cfg.training.n_splits).any():
        print("Notice: Using standard GroupKFold grouped by parent_doc_id.")
        kfold_splitter = GroupKFold(n_splits=cfg.training.n_splits)
        split_generator = kfold_splitter.split(dataset, groups=parent_ids)
    else:
        kfold_splitter = StratifiedGroupKFold(n_splits=cfg.training.n_splits)
        split_generator = kfold_splitter.split(dataset, scenarios, groups=parent_ids)

    oof_y_true, oof_y_probs = [], []
    oof_b_true, oof_b_probs = [], []

    device = torch.device(cfg.model.device)
    print(f"Target Execution Device: {device}")

    # Dynamically compute stylometric dimension
    stylo_dim = dataset.features_list[0].shape[1] - 768

    for fold, (train_idx, val_idx) in enumerate(split_generator):
        print(f"\n--- Training Fold {fold + 1}/{cfg.training.n_splits} ({len(train_idx)} Train, {len(val_idx)} Val) ---")

        train_features = [dataset.features_list[i] for i in train_idx]
        flat_train = np.vstack(train_features)
        train_stylo_cols = flat_train[:, 768:]

        style_scaler = StandardScaler()
        style_scaler.fit(train_stylo_cols)

        scaled_features_list = []
        for doc_matrix in dataset.features_list:
            dense_part = doc_matrix[:, :768]
            stylo_part = doc_matrix[:, 768:]
            scaled_stylo = style_scaler.transform(stylo_part).astype(np.float32)
            scaled_doc = np.hstack([dense_part, scaled_stylo]).astype(np.float32)
            scaled_features_list.append(scaled_doc)

        train_sub = ScaledFoldDataset(dataset, scaled_features_list, train_idx)
        val_sub = ScaledFoldDataset(dataset, scaled_features_list, val_idx)

        train_loader = DataLoader(train_sub, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=pad_collate_fn)
        val_loader = DataLoader(val_sub, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=pad_collate_fn)

        model = MultiTaskNeuralCRFTagger(
            dense_dim=768,
            stylo_dim=stylo_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_lstm_layers=cfg.model.num_lstm_layers,
            dropout=cfg.model.dropout,
            feature_input_dropout=cfg.model.feature_input_dropout,
            rnn_type=cfg.model.rnn_type,
            aux_boundary_weight=cfg.model.aux_boundary_weight,
            boundary_pos_weight=cfg.model.boundary_pos_weight,
            emission_temp=cfg.model.emission_temp,
            use_attention=cfg.model.use_attention,
        ).to(device)

        crf_params = [p for n, p in model.named_parameters() if "crf" in n]
        other_params = [p for n, p in model.named_parameters() if "crf" not in n]
        optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": cfg.training.learning_rate, "weight_decay": cfg.training.weight_decay},
            {"params": crf_params, "lr": cfg.training.learning_rate * cfg.training.crf_lr_mult, "weight_decay": 0.0}
        ])

        scheduler = None
        if cfg.training.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
        elif cfg.training.scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

        best_val_loss = float('inf')
        patience, patience_counter = 3, 0
        best_model_weights = None

        try:
            for epoch in range(cfg.training.epochs):
                model.train()
                train_loss = 0.0
                for batch in train_loader:
                    fused_features = batch["fused_features"].to(device)
                    mask = batch["mask"].to(device)
                    labels = batch["labels"].to(device)
                    boundaries = batch["boundaries"].to(device)
                    seq_lengths = batch["seq_lengths"]

                    optimizer.zero_grad()
                    out = model(fused_features, mask, labels=labels, boundaries=boundaries, seq_lengths=seq_lengths)
                    loss = out["loss"]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        fused_features = batch["fused_features"].to(device)
                        mask = batch["mask"].to(device)
                        labels = batch["labels"].to(device)
                        boundaries = batch["boundaries"].to(device)
                        seq_lengths = batch["seq_lengths"]

                        out = model(fused_features, mask, labels=labels, boundaries=boundaries, seq_lengths=seq_lengths)
                        val_loss += out["loss"].item()

                avg_val_loss = val_loss / len(val_loader)

                if scheduler is not None:
                    if cfg.training.scheduler_type == "cosine":
                        scheduler.step()
                    elif cfg.training.scheduler_type == "reduce_on_plateau":
                        scheduler.step(avg_val_loss)

                print(f"  Epoch {epoch + 1:02d}/{cfg.training.epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    # Store best model state_dict cloned to CPU to prevent VRAM memory accumulation
                    best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  --> Early stopping triggered at epoch {epoch + 1}!")
                        break

            if best_model_weights is not None:
                model.load_state_dict(best_model_weights)

            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    fused_features = batch["fused_features"].to(device)
                    mask = batch["mask"].to(device)
                    seq_lengths = batch["seq_lengths"]

                    preds = model.predict(fused_features, mask, seq_lengths=seq_lengths)
                    mask_np = mask.cpu().numpy()

                    probs_np = preds["probabilities"].cpu().numpy()
                    b_probs_np = preds["boundary_probabilities"].cpu().numpy()
                    y_true_np = batch["labels"].numpy()
                    b_true_np = batch["boundaries"].numpy()

                    for b_i in range(len(batch["doc_ids"])):
                        seq_len = mask_np[b_i].sum()
                        oof_y_true.append(y_true_np[b_i, :seq_len])
                        oof_y_probs.append(probs_np[b_i, :seq_len])
                        oof_b_true.append(b_true_np[b_i, :seq_len])
                        oof_b_probs.append(b_probs_np[b_i, :seq_len])
        finally:
            # Memory cleanup after each fold execution
            del model, optimizer, scheduler, train_loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print("\n=========================================================")
    print(" STAGE 2: NEYMAN-PEARSON CALIBRATION & EVALUATION ")
    print("=========================================================")
    flat_oof_y = np.concatenate(oof_y_true)
    flat_oof_p = np.concatenate(oof_y_probs)

    calibrator = NeymanPearsonCalibrator(target_fpr=cfg.training.target_fpr)
    optimal_tau = calibrator.fit(flat_oof_y, flat_oof_p)

    optimal_b_tau = calibrate_boundary_threshold(oof_b_true, oof_b_probs)

    metrics = evaluate_mixed_authorship_performance(
        oof_y_true,
        oof_y_probs,
        oof_b_true,
        oof_b_probs,
        threshold=optimal_tau,
        boundary_threshold=optimal_b_tau,
    )

    print("\n=========================================================")
    print(" OUT-OF-FOLD PERFORMANCE METRICS ")
    print("=========================================================")
    print(f"Calibrated Threshold (τ*): {optimal_tau:.6f} (Target FPR <= {cfg.training.target_fpr*100:.1f}%)")
    print(f"Sentence AI Precision: {metrics['sent_precision_ai']:.4f}")
    print(f"Sentence AI Recall:    {metrics['sent_recall_ai']:.4f}")
    print(f"Sentence AI F1 Score:  {metrics['sent_f1_ai']:.4f}")
    print(f"Sentence ROC-AUC:      {metrics['sent_roc_auc']:.4f}")
    print(f"Boundary Precision:    {metrics['boundary_precision']:.4f}")
    print(f"Boundary Recall:       {metrics['boundary_recall']:.4f}")
    print(f"Boundary F1 Score:     {metrics['boundary_f1']:.4f}")
    print(f"Segment Span IoU:      {metrics['span_iou']:.4f}")
    print(f"AI Ratio MAE:          {metrics['ai_ratio_mae']:.4f}")

def run_tuning(cfg: MasterConfig, dataset: Any):
    """Runs Optuna hyperparameter optimization with memory safety."""
    import optuna

    stylo_dim = dataset.features_list[0].shape[1] - 768

    def objective(trial: optuna.Trial) -> float:
        device = torch.device(cfg.model.device)

        # Properly scaled hyperparameter spaces
        lr = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        aux_boundary_weight = trial.suggest_float("aux_boundary_weight", 0.05, 2.0, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768])
        num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 2)
        boundary_pos_weight = trial.suggest_float("boundary_pos_weight", 1.0, 10.0, log=True)
        rnn_type = trial.suggest_categorical("rnn_type", ["LSTM", "GRU"])
        scheduler_type = trial.suggest_categorical("scheduler_type", ["cosine", "reduce_on_plateau", "none"])
        feature_input_dropout = trial.suggest_float("feature_input_dropout", 0.0, 0.3)
        crf_lr_mult = trial.suggest_float("crf_lr_mult", 1.0, 10.0, log=True)
        emission_temp = trial.suggest_float("emission_temp", 0.5, 1.5)
        use_attention = trial.suggest_categorical("use_attention", [True, False])

        parent_ids = np.array([doc.parent_doc_id for doc in dataset.docs])
        scenarios = np.array([doc.scenario for doc in dataset.docs])

        kfold_splitter = StratifiedGroupKFold(n_splits=cfg.training.n_splits)
        train_idx, val_idx = next(kfold_splitter.split(dataset, scenarios, groups=parent_ids))

        train_features = [dataset.features_list[i] for i in train_idx]
        flat_train = np.vstack(train_features)
        train_stylo_cols = flat_train[:, 768:]

        style_scaler = StandardScaler()
        style_scaler.fit(train_stylo_cols)

        scaled_features_list = []
        for doc_matrix in dataset.features_list:
            dense_part = doc_matrix[:, :768]
            stylo_part = doc_matrix[:, 768:]
            scaled_stylo = style_scaler.transform(stylo_part).astype(np.float32)
            scaled_features_list.append(np.hstack([dense_part, scaled_stylo]).astype(np.float32))

        train_sub = ScaledFoldDataset(dataset, scaled_features_list, train_idx)
        val_sub = ScaledFoldDataset(dataset, scaled_features_list, val_idx)

        train_loader = DataLoader(train_sub, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=pad_collate_fn)
        val_loader = DataLoader(val_sub, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=pad_collate_fn)

        model = MultiTaskNeuralCRFTagger(
            dense_dim=768,
            stylo_dim=stylo_dim,
            hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            feature_input_dropout=feature_input_dropout,
            rnn_type=rnn_type,
            aux_boundary_weight=aux_boundary_weight,
            boundary_pos_weight=boundary_pos_weight,
            emission_temp=emission_temp,
            use_attention=use_attention
        ).to(device)

        crf_params = [p for n, p in model.named_parameters() if "crf" in n]
        other_params = [p for n, p in model.named_parameters() if "crf" not in n]
        optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": lr, "weight_decay": weight_decay},
            {"params": crf_params, "lr": lr * crf_lr_mult, "weight_decay": 0.0}
        ])

        scheduler = None
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.optuna.search_epochs)
        elif scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

        best_val_auc = 0.0
        patience, patience_counter = 3, 0

        try:
            for epoch in range(cfg.optuna.search_epochs):
                model.train()
                for batch in train_loader:
                    fused_features = batch["fused_features"].to(device)
                    mask = batch["mask"].to(device)
                    labels = batch["labels"].to(device)
                    boundaries = batch["boundaries"].to(device)
                    seq_lengths = batch["seq_lengths"]

                    optimizer.zero_grad()
                    out = model(fused_features, mask, labels=labels, boundaries=boundaries, seq_lengths=seq_lengths)
                    loss = out["loss"]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                model.eval()
                val_probs, val_targets = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        fused_features = batch["fused_features"].to(device)
                        mask = batch["mask"].to(device)
                        seq_lengths = batch["seq_lengths"]

                        preds = model.predict(fused_features, mask, seq_lengths=seq_lengths)

                        mask_np = mask.cpu().numpy()
                        probs_np = preds["probabilities"].cpu().numpy()
                        labels_np = batch["labels"].numpy()

                        for b_i in range(len(batch["doc_ids"])):
                            seq_len = mask_np[b_i].sum()
                            val_probs.extend(probs_np[b_i, :seq_len])
                            val_targets.extend(labels_np[b_i, :seq_len])

                # Safe ROC-AUC calculation to handle single-class target arrays
                if len(np.unique(val_targets)) > 1:
                    val_auc = roc_auc_score(val_targets, val_probs)
                else:
                    val_auc = 0.50

                if scheduler is not None:
                    if scheduler_type == "cosine":
                        scheduler.step()
                    elif scheduler_type == "reduce_on_plateau":
                        scheduler.step(val_auc)

                trial.report(val_auc, step=epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            return best_val_auc
        finally:
            # Memory safety cleanup after every trial
            del model, optimizer, scheduler, train_loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()

    # Optuna database storage location setup
    os.makedirs(cfg.optuna.storage_dir, exist_ok=True)
    db_file_path = os.path.join(cfg.optuna.storage_dir, f"{cfg.optuna.study_name}.db")
    db_uri = f"sqlite:////{db_file_path}"

    print(f"\n[INFO] Saving Optuna study to: {db_file_path}")
    print(f"[INFO] Optuna Study Name: {cfg.optuna.study_name}")

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(
        study_name=cfg.optuna.study_name,
        direction="maximize",
        pruner=pruner,
        storage=db_uri,
        load_if_exists=True
    )

    if os.path.exists("best_params.json") and len(study.trials) == 0:
        with open("best_params.json", "r") as f:
            best_params = json.load(f)
        study.enqueue_trial(best_params)

    study.optimize(objective, n_trials=cfg.optuna.n_trials)

    print("\n=========================================================")
    print(" OPTIMIZATION COMPLETE ")
    print("=========================================================")
    print(f"Best Validation ROC-AUC: {study.best_value:.4f}")

    # Save best parameters to both current working dir and optuna_studies dir
    best_params_local = "best_params.json"
    best_params_optuna_dir = os.path.join(cfg.optuna.storage_dir, f"best_params_{cfg.optuna.study_name}.json")

    with open(best_params_local, "w") as f:
        json.dump(study.best_params, f, indent=4)
    with open(best_params_optuna_dir, "w") as f:
        json.dump(study.best_params, f, indent=4)

    print(f"Saved best hyperparameters to '{best_params_local}' and '{best_params_optuna_dir}'")


# =============================================================================
# SECTION 7: MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    config = parse_args_into_config()
    print(f"Initialized configuration for device: {config.model.device}")

    # 1. Load parquet data, synthesize documents, and compute/load cached features
    dataset = build_dataset(config)

    # 2. Execute Optuna hyperparameter tuning:
    run_tuning(config, dataset)

    # OR execute final model training & cross-validation evaluation:
    # run_training_pipeline(config, dataset)


    #CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python mixdet.py