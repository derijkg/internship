# data/dataset.py

import os
import hashlib
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from data.synthetic_generator import SyntheticDocument
from features.stylometrics import StylometricFeatureEngine
from features.dense_encoder import DenseTransformerEncoder
from features.lpt_extractor import LocalProbabilityTopologyEngine


class MixedAuthorshipDataset(Dataset):
    """
    PyTorch Dataset that extracts and disk-caches fused features
    (Dense Embeddings [768] + Stylometrics [84] + LPT Features [10] = 862 dims).
    """
    def __init__(
        self,
        synthetic_docs: List[SyntheticDocument],
        style_engine: StylometricFeatureEngine,
        dense_encoder: DenseTransformerEncoder,
        lpt_engine: LocalProbabilityTopologyEngine,
        cache_dir: str = "./.feature_cache",
        cache_file: Optional[str] = None
    ):
        self.docs = synthetic_docs
        self.style_engine = style_engine
        self.dense_encoder = dense_encoder
        self.lpt_engine = lpt_engine
        self.cache_dir = cache_dir
        self.cache_file = cache_file
        os.makedirs(cache_dir, exist_ok=True)

        # Precompute or load cached feature matrices as torch Tensors
        self.features_list: List[torch.Tensor] = self._precompute_or_load_cache()

    def _get_cache_hash(self) -> str:
        """Generates MD5 hash for cache validation including model specs."""
        hasher = hashlib.md5()
        hasher.update(str(len(self.docs)).encode('utf-8'))
        
        doc_signature = "|".join(f"{d.parent_doc_id}:{d.num_sentences}" for d in self.docs)
        hasher.update(doc_signature.encode('utf-8'))
        
        encoder_name = getattr(self.dense_encoder, "model_name", "dense_encoder")
        lpt_name = getattr(self.lpt_engine, "model_name", "lpt_engine")
        hasher.update(f"{encoder_name}_{lpt_name}".encode('utf-8'))
        
        return hasher.hexdigest()

    def _precompute_or_load_cache(self) -> List[torch.Tensor]:
        """Precomputes fused features or loads specified/hashed cache from disk."""
        
        # 1. FORCED CACHE LOAD
        if self.cache_file:
            cache_filename = self.cache_file if self.cache_file.endswith(".pt") else f"{self.cache_file}.pt"
            target_path = cache_filename if (os.path.isabs(cache_filename) or os.path.exists(cache_filename)) else os.path.join(self.cache_dir, cache_filename)

            if not os.path.exists(target_path):
                raise FileNotFoundError(f"[ERROR] Specified cache file '{target_path}' does not exist!")

            print(f"-> FORCED CACHE LOAD: Loading feature matrices directly from '{target_path}'...")
            loaded_features = torch.load(target_path, weights_only=False)

            if len(loaded_features) != len(self.docs):
                raise ValueError(f"Cache file contains {len(loaded_features)} docs, but current run generated {len(self.docs)} docs!")

            return loaded_features

        # 2. AUTOMATIC MD5 HASH LOAD / COMPUTATION
        cache_hash = self._get_cache_hash()
        cache_path = os.path.join(self.cache_dir, f"fused_features_{cache_hash}.pt")

        if os.path.exists(cache_path):
            print(f"-> Loading cached fused feature matrices from '{cache_path}'...")
            return torch.load(cache_path, weights_only=False)

        print(f"-> Extracting & caching features for {len(self.docs)} documents...")
        cached_features: List[torch.Tensor] = []

        for idx, doc in enumerate(self.docs):
            if (idx + 1) % 100 == 0 or idx == len(self.docs) - 1:
                print(f"   Processing doc {idx + 1}/{len(self.docs)}...", end="\r")

            if len(doc.sentences) == 0:
                # Fallback zero array if document is empty
                total_dim = getattr(self.dense_encoder, "embedding_dim", 768) + 84 + self.lpt_engine.feature_dim
                fused_mat = np.zeros((0, total_dim), dtype=np.float32)
            else:
                # A. Stylometrics [N, 84]
                style_mat, _ = self.style_engine.compute_document_features(doc.sentences)
                style_mat = np.atleast_2d(style_mat)

                # B. Dense Sentence Embeddings [N, 768]
                dense_tensor = self.dense_encoder.extract_sentence_embeddings(doc.sentences)
                dense_mat = np.atleast_2d(dense_tensor.cpu().numpy())

                # C. Local Probability Topology (LPT) [N, 10]
                lpt_mat = self.lpt_engine.extract_document_lpt(doc.sentences)
                lpt_mat = np.atleast_2d(lpt_mat)

                # Validate row alignment across feature extractors
                if not (dense_mat.shape[0] == style_mat.shape[0] == lpt_mat.shape[0]):
                    raise ValueError(
                        f"Mismatch sentence count in doc {doc.doc_id}: "
                        f"dense {dense_mat.shape}, style {style_mat.shape}, lpt {lpt_mat.shape}"
                    )

                # Fuse all vectors along feature dimension [N, 862]
                fused_mat = np.hstack([dense_mat, style_mat, lpt_mat]).astype(np.float32)

            cached_features.append(torch.from_numpy(fused_mat))

        print(f"\n-> Successfully cached all document features to '{cache_path}'")
        torch.save(cached_features, cache_path)
        return cached_features

    def __len__(self) -> int:
        return len(self.docs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        doc = self.docs[idx]
        
        return {
            "doc_id": doc.doc_id,
            "parent_doc_id": doc.parent_doc_id,
            "scenario": doc.scenario,
            "fused_features": self.features_list[idx],
            # Continuous Float32 tensors for AICS labels [0.0, 0.5, 1.0] and boundaries
            "labels": torch.tensor(doc.labels, dtype=torch.float32),
            "boundaries": torch.tensor(doc.boundaries, dtype=torch.float32),
            "seq_len": len(doc.sentences)
        }


def pad_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pads variable length sentence sequences in a batch to uniform max length.
    """
    batch_size = len(batch)
    max_len = max(item["seq_len"] for item in batch)
    
    if max_len == 0:
        max_len = 1
        
    fused_dim = batch[0]["fused_features"].shape[-1]

    padded_features = torch.zeros(batch_size, max_len, fused_dim, dtype=torch.float32)
    padded_labels = torch.zeros(batch_size, max_len, dtype=torch.float32)
    padded_boundaries = torch.zeros(batch_size, max_len, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    doc_ids = []
    parent_doc_ids = []
    scenarios = []

    for i, item in enumerate(batch):
        seq_len = item["seq_len"]
        if seq_len > 0:
            padded_features[i, :seq_len] = item["fused_features"]
            padded_labels[i, :seq_len] = item["labels"]
            padded_boundaries[i, :seq_len] = item["boundaries"]
            mask[i, :seq_len] = True

        doc_ids.append(item["doc_id"])
        parent_doc_ids.append(item["parent_doc_id"])
        scenarios.append(item["scenario"])

    return {
        "fused_features": padded_features,
        "labels": padded_labels,
        "boundaries": padded_boundaries,
        "mask": mask,
        "doc_ids": doc_ids,
        "parent_doc_ids": parent_doc_ids,
        "scenarios": scenarios
    }