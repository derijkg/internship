#mixed author
def generate_synthetic_mixed_dataset(raw_df, selected_models, llm_ratio=4, random_state=42):
    local_rng = random.Random(random_state)
    synthetic_docs = []

    print("Generating Synthetic Mixed-Authorship Dataset...")

    for doc_idx, (_, row) in enumerate(raw_df.iterrows()):
        source = row.get('source', 'unknown')
        
        # Use existing _id if present, otherwise fall back to doc_idx
        if '_id' in row and pd.notna(row['_id']):
            parent_doc_id = str(row['_id'])
        else:
            parent_doc_id = f"parent_{doc_idx}"

        raw_human_sents = []
        for col in ['abstract_sentence', 'abstract_sentences']:
            if col in row:
                parsed = safe_parse_list(row.get(col, []))
                if isinstance(parsed, list) and len(parsed) > 0:
                    raw_human_sents = parsed
                    break

        human_sents = [normalize_text(s) for s in raw_human_sents if normalize_text(s)]
        if len(human_sents) < 3:
            continue

        valid_models = []
        for model in selected_models:
            for col_var in [f"{model}_single", f"{model}_sentence", f"{model}_sentences"]:
                if col_var in row:
                    raw_ai = safe_parse_list(row[col_var])
                    if isinstance(raw_ai, list) and len(raw_ai) > 0:
                        valid_models.append((model, col_var))
                        break

        if not valid_models:
            continue

        sampled_models = local_rng.sample(valid_models, k=min(llm_ratio, len(valid_models)))

        # Scenario 1: Pure Human Document
        synthetic_docs.append({
            'doc_id': f"{parent_doc_id}_pure_human",
            'parent_doc_id': parent_doc_id,
            'source': source,
            'sentences': human_sents,
            'labels': [0] * len(human_sents),
            'scenario': 'pure_human'
        })

        for model_name, col_var in sampled_models:
            parsed_ai = safe_parse_list(row[col_var])
            ai_sents = [normalize_text(s) for s in parsed_ai if normalize_text(s)]

            if not ai_sents:
                continue

            # Scenario 2: Pure LLM Document
            synthetic_docs.append({
                'doc_id': f"{parent_doc_id}_{model_name}_pure_ai",
                'parent_doc_id': parent_doc_id,
                'source': source,
                'sentences': ai_sents,
                'labels': [1] * len(ai_sents),
                'scenario': 'pure_ai'
            })

            # Scenario 3: Single/Multi Sentence Injection
            mixed_sents_inj = list(human_sents)
            mixed_labels_inj = [0] * len(human_sents)

            inject_pos = local_rng.randint(1, len(human_sents) - 1)
            if len(ai_sents) > 2:
                start_idx = local_rng.randint(0, len(ai_sents) - 2)
                ai_snippet = ai_sents[start_idx : start_idx + local_rng.randint(1, 2)]
            else:
                ai_snippet = ai_sents

            mixed_sents_inj[inject_pos:inject_pos] = ai_snippet
            for k in range(len(ai_snippet)):
                mixed_labels_inj.insert(inject_pos + k, 1)

            synthetic_docs.append({
                'doc_id': f"{parent_doc_id}_{model_name}_injection",
                'parent_doc_id': parent_doc_id,
                'source': source,
                'sentences': mixed_sents_inj,
                'labels': mixed_labels_inj,
                'scenario': 'sentence_injection'
            })

            # Scenario 4: Paragraph / Block Substitution
            if len(human_sents) >= 4 and len(ai_sents) >= 2:
                mixed_sents_sub = list(human_sents)
                mixed_labels_sub = [0] * len(human_sents)

                sub_start = local_rng.randint(1, len(human_sents) - 2)
                sub_len = min(2, len(human_sents) - sub_start)

                for k in range(sub_len):
                    mixed_sents_sub[sub_start + k] = ai_sents[k % len(ai_sents)]
                    mixed_labels_sub[sub_start + k] = 1

                synthetic_docs.append({
                    'doc_id': f"{parent_doc_id}_{model_name}_substitution",
                    'parent_doc_id': parent_doc_id,
                    'source': source,
                    'sentences': mixed_sents_sub,
                    'labels': mixed_labels_sub,
                    'scenario': 'block_substitution'
                })

    print(f"-> Generated {len(synthetic_docs)} synthetic documents across 4 mixed-authorship scenarios.")
    return synthetic_docs


def build_multiscale_sentence_dataframe(synthetic_docs):
    records = []

    for doc in synthetic_docs:
        doc_id = doc['doc_id']
        parent_doc_id = doc['parent_doc_id']
        source = doc['source']
        scenario = doc['scenario']
        sents = doc['sentences']
        labels = doc['labels']
        n_sents = len(sents)
        doc_text = " ".join(sents)

        for i in range(n_sents):
            sents_w1 = [sents[i]]
            text_w1 = sents[i]

            start_3 = max(0, i - 1)
            end_3 = min(n_sents, i + 2)
            sents_w3 = sents[start_3:end_3]
            text_w3 = " ".join(sents_w3)

            start_5 = max(0, i - 2)
            end_5 = min(n_sents, i + 3)
            sents_w5 = sents[start_5:end_5]
            text_w5 = " ".join(sents_w5)

            records.append({
                'doc_id': doc_id,
                'parent_doc_id': parent_doc_id,
                'source': source,
                'scenario': scenario,
                'sentence_idx': i,
                'label': labels[i],
                'text': sents[i],
                'text_w1': text_w1,
                'text_w3': text_w3,
                'text_w5': text_w5,
                'sents_w1': sents_w1,
                'sents_w3': sents_w3,
                'sents_w5': sents_w5,
                'doc_text': doc_text,
                'doc_sentences': sents
            })

    return pd.DataFrame(records)





#SVM
import random
import zlib
import numpy as np
import pandas as pd


def mix_abstract(human_sentences, llm_sentences, target_ratio, random_state=42):
  """Substitutes target_ratio % of human sentences with corresponding LLM sentences.

  Assumes 1-to-1 parallel sentence lists.
  """
  n_sentences = len(human_sentences)
  if n_sentences != len(llm_sentences):
    raise ValueError("Human and LLM sentence lists must be equal in length.")

  # Determine exact number of sentences to replace
  k = int(round(target_ratio * n_sentences))

  # Clamp k between 0 and n_sentences
  k = max(0, min(n_sentences, k))

  # Set seed for reproducibility
  rng = random.Random(random_state)

  # Randomly select indices for LLM replacement
  llm_indices = set(rng.sample(range(n_sentences), k))

  # Construct the mixed list of sentences
  mixed_sentences = [
      llm_sentences[i] if i in llm_indices else human_sentences[i]
      for i in range(n_sentences)
  ]

  actual_ratio = k / n_sentences if n_sentences > 0 else 0.0
  mixed_text = " ".join(mixed_sentences)

  return mixed_text, actual_ratio, sorted(list(llm_indices))


def generate_mixed_test_dataset(
    test_df, target_ratios=[0.25, 0.50, 0.75], seed=42
):
  """test_df expected columns:

  - 'doc_id': Unique abstract identifier
  - 'human_sentences': List[str]
  - 'llm_sentences': List[str]
  """
  mixed_records = []

  for idx, row in test_df.iterrows():
    doc_id = row['doc_id']
    h_sents = row['human_sentences']
    l_sents = row['llm_sentences']

    # Skip very short abstracts where percentages cannot be meaningfully split
    if len(h_sents) < 4 or len(h_sents) != len(l_sents):
      continue

    for ratio in target_ratios:
      # Generate a unique, deterministic seed per document and ratio using CRC32
      seed_str = f"{seed}_{doc_id}_{ratio}"
      pair_seed = zlib.crc32(seed_str.encode("utf-8"))

      mixed_text, actual_ratio, llm_indices = mix_abstract(
          h_sents, l_sents, target_ratio=ratio, random_state=pair_seed
      )

      mixed_records.append({
          "doc_id": doc_id,
          "target_ratio": ratio,
          "actual_ratio": actual_ratio,
          "llm_sentence_indices": llm_indices,
          "mixed_text": mixed_text,
          "num_sentences": len(h_sents),
      })

  return pd.DataFrame(mixed_records)




#BAYESIAN
# data/synthetic_generator.py

import ast
import json
import random
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SyntheticDocument:
    """Dataclass representing a single generated synthetic document."""

    doc_id: str
    parent_doc_id: str
    source: str
    scenario: str
    llm_models_used: List[str]
    sentences: List[str]
    labels: List[float]  # AICS: y_i ∈ [0.0, 0.5, 1.0]
    boundaries: List[int]  # b_i ∈ {0, 1}, transition indicators
    num_sentences: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MixedAuthorshipDataEngine:
    """Production-grade synthetic dataset generator for mixed human-LLM authorship."""

    AVAILABLE_MODELS = [
        "qwen3.6:27b",
        "gemma4:e4b",
        "qwen3.5:4b",
        "gemma4:26b",
    ]

    def __init__(
        self,
        random_state: int = 42,
        min_sentences: int = 3,
        scenario_weights: Optional[Dict[str, float]] = None,
    ):
        self.rng = random.Random(random_state)
        self.min_sentences = min_sentences

    @staticmethod
    def normalize_text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        text = text.replace("—", "-").replace("–", "-")
        return " ".join(text.split())

    @classmethod
    def split_into_sentences(cls, text: str) -> List[str]:
        """Splits full string ({model}_full) into clean sentences via regex."""
        if not text or not isinstance(text, str):
            return []
        # Regex split on sentence-ending punctuation followed by space/newline
        raw_sents = re.split(r'(?<=[.!?])\s+', text)
        cleaned = [cls.normalize_text(s) for s in raw_sents if len(s.strip()) > 3]
        return cleaned

    @classmethod
    def parse_sentence_array(cls, val: Any) -> List[str]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return []
        parsed_list = []
        if isinstance(val, (list, tuple, np.ndarray)):
            parsed_list = list(val)
        elif isinstance(val, str):
            val_str = val.strip()
            if val_str.startswith("[") and val_str.endswith("]"):
                try:
                    parsed = ast.literal_eval(val_str)
                    parsed_list = list(parsed) if isinstance(parsed, (list, tuple, np.ndarray)) else [str(parsed)]
                except (ValueError, SyntaxError):
                    try:
                        parsed = json.loads(val_str)
                        parsed_list = list(parsed) if isinstance(parsed, (list, tuple, np.ndarray)) else [str(parsed)]
                    except Exception:
                        parsed_list = [val_str]
            elif val_str:
                parsed_list = [val_str]

        cleaned = [cls.normalize_text(str(s)) for s in parsed_list if str(s).strip()]
        return [s for s in cleaned if len(s) > 3]

    @staticmethod
    def compute_boundaries(labels: List[float]) -> List[int]:
        if not labels:
            return []
        boundaries = [0] * len(labels)
        for i in range(1, len(labels)):
            # Boundary exists if authorship score shifts significantly
            if abs(labels[i] - labels[i - 1]) > 0.1:
                boundaries[i] = 1
        return boundaries

    # ==========================================
    # Scenario Generators with Continuous AICS
    # ==========================================
    def _create_pure_human(self, parent_id: str, source: str, human_sents: List[str]) -> SyntheticDocument:
        labels = [0.0] * len(human_sents)
        return SyntheticDocument(
            doc_id=f"{parent_id}_pure_human",
            parent_doc_id=parent_id,
            source=source,
            scenario="pure_human",
            llm_models_used=[],
            sentences=human_sents,
            labels=labels,
            boundaries=self.compute_boundaries(labels),
            num_sentences=len(human_sents),
        )

    def _create_pure_ai(self, parent_id: str, source: str, model_name: str, ai_full_sents: List[str]) -> SyntheticDocument:
        labels = [1.0] * len(ai_full_sents)
        return SyntheticDocument(
            doc_id=f"{parent_id}_{model_name}_pure_ai",
            parent_doc_id=parent_id,
            source=source,
            scenario="pure_ai",
            llm_models_used=[model_name],
            sentences=ai_full_sents,
            labels=labels,
            boundaries=self.compute_boundaries(labels),
            num_sentences=len(ai_full_sents),
        )

    def _create_sentence_substitution(
        self, parent_id: str, source: str, model_name: str, human_sents: List[str], ai_single_sents: List[str]
    ) -> SyntheticDocument:
        """Replaces specific sentences with single-sentence LLM rewrites (AICS = 0.5 for polish)."""
        sents = list(human_sents)
        labels = [0.0] * len(human_sents)

        sub_idx = self.rng.randint(0, len(human_sents) - 1)
        if sub_idx < len(ai_single_sents):
            sents[sub_idx] = ai_single_sents[sub_idx]
            labels[sub_idx] = 0.5  # Polished sentence level rewrite

        return SyntheticDocument(
            doc_id=f"{parent_id}_{model_name}_substitution",
            parent_doc_id=parent_id,
            source=source,
            scenario="substitution",
            llm_models_used=[model_name],
            sentences=sents,
            labels=labels,
            boundaries=self.compute_boundaries(labels),
            num_sentences=len(sents),
        )

    def process_dataframe(self, df: pd.DataFrame) -> List[SyntheticDocument]:
        synthetic_docs: List[SyntheticDocument] = []

        for idx, row in df.iterrows():
            parent_id = str(row["_id"]) if "_id" in row and pd.notna(row["_id"]) else f"doc_{idx}"
            source = str(row.get("source", "unknown"))

            human_sents = self.parse_sentence_array(row.get("abstract_sentence", []))
            if len(human_sents) < self.min_sentences:
                continue

            # Pure Human
            synthetic_docs.append(self._create_pure_human(parent_id, source, human_sents))

            for model in self.AVAILABLE_MODELS:
                col_single = f"{model}_single"
                col_full = f"{model}_full"

                # Parse full string rewrite
                if col_full in row and isinstance(row[col_full], str):
                    ai_full_sents = self.split_into_sentences(row[col_full])
                    if len(ai_full_sents) >= self.min_sentences:
                        synthetic_docs.append(self._create_pure_ai(parent_id, source, model, ai_full_sents))

                # Parse sentence-level rewrites
                if col_single in row:
                    ai_single_sents = self.parse_sentence_array(row[col_single])
                    if len(ai_single_sents) >= len(human_sents):
                        synthetic_docs.append(
                            self._create_sentence_substitution(parent_id, source, model, human_sents, ai_single_sents)
                        )

        return synthetic_docs


## data/dataset.py

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




#combine
class DutchSyntheticMixer:
    """Generates synthetic mixed-author Dutch documents from Parquet row data."""
    def __init__(self, seed: int = 42):
        random.seed(seed)

    def mix_sample(self, human_sents: List[str], single_rewrites: Dict[str, List[str]]) -> Dict[str, Any]:
        available_models = list(single_rewrites.keys())
        
        # Fallback to pure human if no valid LLM rewrites exist for this row
        if not available_models or len(human_sents) == 0:
            return {
                "text": " ".join(human_sents),
                "sentences": human_sents,
                "labels": ["Human"] * len(human_sents),
                "scenario": "pure_human"
            }

        scenario = random.choice(["needle", "block_swap", "interleaved"])
        mixed_sents, labels = [], []

        if scenario == "needle":
            mixed_sents, labels = list(human_sents), ["Human"] * len(human_sents)
            idx = random.randint(0, len(human_sents) - 1)
            model = random.choice(available_models)
            mixed_sents[idx] = single_rewrites[model][idx]
            labels[idx] = model

        elif scenario == "block_swap":
            model = random.choice(available_models)
            n_sents = len(human_sents)
            split = random.randint(1, max(1, n_sents - 1))
            m_sents = single_rewrites[model]
            
            if random.random() > 0.5:
                mixed_sents = human_sents[:split] + m_sents[split:]
                labels = ["Human"] * split + [model] * (n_sents - split)
            else:
                mixed_sents = m_sents[:split] + human_sents[split:]
                labels = [model] * split + ["Human"] * (n_sents - split)

        else:  # interleaved
            curr_author = "Human"
            authors = ["Human"] + available_models
            for i, h_sent in enumerate(human_sents):
                if random.random() < 0.4:
                    curr_author = random.choice([a for a in authors if a != curr_author])
                
                if curr_author == "Human":
                    mixed_sents.append(h_sent)
                else:
                    mixed_sents.append(single_rewrites[curr_author][i])
                labels.append(curr_author)

        return {
            "text": " ".join(mixed_sents),
            "sentences": mixed_sents,
            "labels": labels,
            "scenario": scenario
        }

    

#mixed_author
@dataclass
class SyntheticDocument:
    """Dataclass representing a single generated synthetic document."""

    doc_id: str
    parent_doc_id: str
    source: str
    scenario: str
    llm_models_used: List[str]
    sentences: List[str]
    labels: List[int]  # y_i ∈ {0, 1}
    boundaries: List[int]  # b_i ∈ {0, 1}, where b_i = I(y_i != y_{i-1})
    num_sentences: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MixedAuthorshipDataEngine:
    """Production-grade synthetic dataset generator for mixed human-LLM authorship detection."""

    AVAILABLE_MODELS = [
        "qwen3.6:27b",
        "gemma4:e4b",
        "qwen3.5:4b",
        "gemma4:26b",
    ]

    def __init__(
        self,
        random_state: int = 42,
        min_sentences: int = 3,
        scenario_weights: Optional[Dict[str, float]] = None,
    ):
        self.rng = random.Random(random_state)
        self.min_sentences = min_sentences
        self.scenario_weights = scenario_weights or {
            "pure_human": 0.15,
            "pure_ai": 0.15,
            "continuation": 0.20,
            "infilling": 0.20,
            "substitution": 0.20,
            "multi_model_hybrid": 0.10,
        }

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalizes unicode characters and strips unwanted whitespace."""
        if not isinstance(text, str) or not text.strip():
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = (
            text.replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("‘", "'")
        )
        text = text.replace("—", "-").replace("–", "-")
        return " ".join(text.split())

    @classmethod
    def parse_sentence_array(cls, val: Any) -> List[str]:
        """Safely parses stringified arrays, lists, or numpy arrays into clean str lists."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return []
        parsed_list = []
        if isinstance(val, (list, tuple, np.ndarray)):
            parsed_list = list(val)
        elif isinstance(val, str):
            val_str = val.strip()
            if val_str.startswith("[") and val_str.endswith("]"):
                try:
                    parsed = ast.literal_eval(val_str)
                    parsed_list = (
                        list(parsed)
                        if isinstance(parsed, (list, tuple, np.ndarray))
                        else [str(parsed)]
                    )
                except (ValueError, SyntaxError):
                    try:
                        parsed = json.loads(val_str)
                        parsed_list = (
                            list(parsed)
                            if isinstance(parsed, (list, tuple, np.ndarray))
                            else [str(parsed)]
                        )
                    except Exception:
                        parsed_list = [val_str]
            elif val_str:
                parsed_list = [val_str]

        cleaned = [
            cls.normalize_text(str(s)) for s in parsed_list if str(s).strip()
        ]
        return [s for s in cleaned if len(s) > 3]

    @staticmethod
    def compute_boundaries(labels: List[int]) -> List[int]:
        """Computes explicit boundary transition indicators: b_i = I(y_i != y_{i-1})."""
        if not labels:
            return []
        boundaries = [0] * len(labels)
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                boundaries[i] = 1
        return boundaries

    def _create_pure_human(
        self, parent_id: str, source: str, human_sents: List[str]
    ) -> SyntheticDocument:
        labels = [0] * len(human_sents)
        return SyntheticDocument(
            doc_id=f"{parent_id}_pure_human",
            parent_doc_id=parent_id,
            source=source,
            scenario="pure_human",
            llm_models_used=[],
            sentences=human_sents,
            labels=labels,
            boundaries=self.compute_boundaries(labels),
            num_sentences=len(human_sents),
        )

    def _create_pure_ai(
        self,
        parent_id: str,
        source: str,
        model_name: str,
        ai_sents: List[str],
    ) -> SyntheticDocument:
        labels = [1] * len(ai_sents)
        return SyntheticDocument(
            doc_id=f"{parent_id}_{model_name}_pure_ai",
            parent_doc_id=parent_id,
            source=source,
            scenario="pure_ai",
            llm_models_used=[model_name],
            sentences=ai_sents,
            labels=labels,
            boundaries=self.compute_boundaries(labels),
            num_sentences=len(ai_sents),
        )

    def _create_continuation(
        self,
        parent_id: str,
        source: str,
        model_name: str,
        human_sents: List[str],
        ai_sents: List[str],
    ) -> SyntheticDocument:
        split_point = self.rng.randint(1, len(human_sents) - 1)
        h_part = human_sents[:split_point]

        ai_part = (
            ai_sents[split_point:]
            if len(ai_sents) > split_point
            else ai_sents[: max(1, len(human_sents) - split_point)]
        )

        sents = h_part + ai_part
        labels = [0] * len(h_part) + [1] * len(ai_part)

        return SyntheticDocument(
            doc_id=f"{parent_id}_{model_name}_continuation",
            parent_doc_id=parent_id,
            source=source,
            scenario="continuation",
            llm_models_used=[model_name],
            sentences=sents,
            labels=labels,
            boundaries=self.compute_boundaries(labels),
            num_sentences=len(sents),
        )

    def _create_infilling(
        self,
        parent_id: str,
        source: str,
        model_name: str,
        human_sents: List[str],
        ai_sents: List[str],
    ) -> SyntheticDocument:
        sents = list(human_sents)
        labels = [0] * len(human_sents)

        insert_pos = self.rng.randint(1, len(human_sents) - 1)

        snippet_len = min(self.rng.randint(1, 2), len(ai_sents))
        start_idx = (
            self.rng.randint(0, len(ai_sents) - snippet_len)
            if len(ai_sents) > snippet_len
            else 0
        )
        ai_snippet = ai_sents[start_idx : start_idx + snippet_len]

        sents[insert_pos:insert_pos] = ai_snippet
        for k in range(len(ai_snippet)):
            labels.insert(insert_pos + k, 1)

        return SyntheticDocument(
            doc_id=f"{parent_id}_{model_name}_infilling",
            parent_doc_id=parent_id,
            source=source,
            scenario="infilling",
            llm_models_used=[model_name],
            sentences=sents,
            labels=labels,
            boundaries=self.compute_boundaries(labels),
            num_sentences=len(sents),
        )

    def _create_substitution(
        self,
        parent_id: str,
        source: str,
        model_name: str,
        human_sents: List[str],
        ai_sents: List[str],
    ) -> SyntheticDocument:
        sents = list(human_sents)
        labels = [0] * len(human_sents)

        sub_start = self.rng.randint(1, len(human_sents) - 2)
        sub_len = min(self.rng.randint(1, 2), len(human_sents) - sub_start)

        for k in range(sub_len):
            ai_idx = (sub_start + k) % len(ai_sents)
            sents[sub_start + k] = ai_sents[ai_idx]
            labels[sub_start + k] = 1

        return SyntheticDocument(
            doc_id=f"{parent_id}_{model_name}_substitution",
            parent_doc_id=parent_id,
            source=source,
            scenario="substitution",
            llm_models_used=[model_name],
            sentences=sents,
            labels=labels,
            boundaries=self.compute_boundaries(labels),
            num_sentences=len(sents),
        )

    def _create_multi_model_hybrid(
        self,
        parent_id: str,
        source: str,
        available_ai_dict: Dict[str, List[str]],
        human_sents: List[str],
    ) -> Optional[SyntheticDocument]:
        if len(available_ai_dict) < 2 or len(human_sents) < 5:
            return None

        selected_models = self.rng.sample(list(available_ai_dict.keys()), k=2)
        m1, m2 = selected_models[0], selected_models[1]

        sents = list(human_sents)
        labels = [0] * len(human_sents)

        pos1 = self.rng.randint(1, max(1, (len(human_sents) // 2) - 1))
        ai1_sent = self.rng.choice(available_ai_dict[m1])
        sents[pos1] = ai1_sent
        labels[pos1] = 1

        pos2_start = max(pos1 + 1, len(human_sents) // 2)
        pos2_end = len(human_sents) - 1

        # Prevent empty range ValueError in self.rng.randint
        if pos2_start >= pos2_end:
            return None

        pos2 = self.rng.randint(pos2_start, pos2_end)
        
        ai2_sent = self.rng.choice(available_ai_dict[m2])
        sents[pos2] = ai2_sent
        labels[pos2] = 1

        return SyntheticDocument(
            doc_id=f"{parent_id}_multimodel_hybrid",
            parent_doc_id=parent_id,
            source=source,
            scenario="multi_model_hybrid",
            llm_models_used=[m1, m2],
            sentences=sents,
            labels=labels,
            boundaries=self.compute_boundaries(labels),
            num_sentences=len(sents),
        )

    def process_dataframe(self, df: pd.DataFrame) -> List[SyntheticDocument]:
        synthetic_docs: List[SyntheticDocument] = []
        print(f"--- Data Engine: Processing {len(df)} Raw Documents into Synthetic Mixed-Authorship ---")

        for idx, row in df.iterrows():
            parent_id = str(row["_id"]) if "_id" in row and pd.notna(row["_id"]) else f"doc_{idx}"
            source = str(row.get("source", "unknown"))

            human_sents = self.parse_sentence_array(row.get("abstract_sentence", []))
            if len(human_sents) < self.min_sentences:
                continue

            available_ai: Dict[str, List[str]] = {}
            for model in self.AVAILABLE_MODELS:
                col_single = f"{model}_single"
                if col_single in row:
                    ai_parsed = self.parse_sentence_array(row[col_single])
                    if len(ai_parsed) >= 2:
                        available_ai[model] = ai_parsed

            if not available_ai:
                continue

            # 1. Pure Human
            synthetic_docs.append(self._create_pure_human(parent_id, source, human_sents))

            # 2. Model-specific mixed scenarios
            for model_name, ai_sents in available_ai.items():
                synthetic_docs.append(self._create_pure_ai(parent_id, source, model_name, ai_sents))

                if len(human_sents) >= 4:
                    synthetic_docs.append(self._create_continuation(parent_id, source, model_name, human_sents, ai_sents))

                if len(human_sents) >= 3:
                    synthetic_docs.append(self._create_infilling(parent_id, source, model_name, human_sents, ai_sents))

                if len(human_sents) >= 4:
                    synthetic_docs.append(self._create_substitution(parent_id, source, model_name, human_sents, ai_sents))

            # 3. Multi-Model Hybrid
            if len(available_ai) >= 2 and len(human_sents) >= 5:
                hybrid_doc = self._create_multi_model_hybrid(parent_id, source, available_ai, human_sents)
                if hybrid_doc:
                    synthetic_docs.append(hybrid_doc)

        print(f"-> Successfully generated {len(synthetic_docs)} synthetic documents.")
        return synthetic_docs


# =============================================================================
# 2. FEATURE CACHED DATASET & COLLATOR
# =============================================================================
class MixedAuthorshipDataset(Dataset):
    """
    PyTorch Dataset that extracts and disk-caches multi-scale fused feature matrices
    (Dense Transformer Embeddings + Stylometric Deltas + Boundary Gradients).
    """
    def __init__(
        self,
        synthetic_docs: List[SyntheticDocument],
        style_engine: Any,
        dense_encoder: Any,
        cache_dir: str = "./.feature_cache",
        cache_file: Optional[str] = None
    ):
        self.docs = synthetic_docs
        self.style_engine = style_engine
        self.dense_encoder = dense_encoder
        self.cache_dir = cache_dir
        self.cache_file = cache_file
        os.makedirs(cache_dir, exist_ok=True)

        # Features stored as List[np.ndarray] for fold-safe standardization
        self.features_list: List[np.ndarray] = self._precompute_or_load_cache()

    def _get_cache_hash(self) -> str:
        """Generates a robust MD5 hash for cache validation."""
        hasher = hashlib.md5()
        hasher.update(str(len(self.docs)).encode('utf-8'))
        
        doc_signature = "|".join(f"{d.doc_id}:{d.num_sentences}" for d in self.docs)
        hasher.update(doc_signature.encode('utf-8'))
        
        encoder_name = getattr(self.dense_encoder, "model_name", "dense_encoder")
        hasher.update(encoder_name.encode('utf-8'))

        w_flags = f"w3:{getattr(self.style_engine, 'include_w3', True)}_w5:{getattr(self.style_engine, 'include_w5', True)}"
        hasher.update(w_flags.encode('utf-8'))
        
        return hasher.hexdigest()

    def _precompute_or_load_cache(self) -> List[np.ndarray]:
        """Precomputes fused features or loads specified/hashed cache safely on CPU."""
        if self.cache_file:
            cache_filename = self.cache_file if self.cache_file.endswith(".npy") or self.cache_file.endswith(".pt") else f"{self.cache_file}.pt"
            target_path = cache_filename if os.path.isabs(cache_filename) or os.path.exists(cache_filename) else os.path.join(self.cache_dir, cache_filename)

            if not os.path.exists(target_path):
                raise FileNotFoundError(f"[ERROR] Specified cache file '{target_path}' does not exist!")

            print(f"-> FORCED CACHE LOAD: Loading feature matrices directly from '{target_path}'...")
            loaded_features = torch.load(target_path, map_location="cpu", weights_only=False)

            if len(loaded_features) != len(self.docs):
                raise ValueError(f"Forced cache contains {len(loaded_features)} docs, but run generated {len(self.docs)} docs!")

            return [f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in loaded_features]

        cache_hash = self._get_cache_hash()
        cache_path = os.path.join(self.cache_dir, f"fused_features_{cache_hash}.pt")

        if os.path.exists(cache_path):
            print(f"-> Loading cached fused feature matrices from '{cache_path}'...")
            loaded_features = torch.load(cache_path, map_location="cpu", weights_only=False)
            return [f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in loaded_features]

        print(f"-> Extracting & caching features for {len(self.docs)} documents...")
        cached_features: List[np.ndarray] = []
        dense_dim = getattr(self.dense_encoder, "hidden_dim", 768)

        for idx, doc in enumerate(self.docs):
            if (idx + 1) % 500 == 0 or idx == len(self.docs) - 1:
                print(f"   Processing doc {idx + 1}/{len(self.docs)}...", end="\r")

            if len(doc.sentences) == 0:
                style_mat, style_dim = self.style_engine.compute_document_features([])
                fused_mat = np.zeros((0, dense_dim + style_dim), dtype=np.float32)
            else:
                style_mat, _ = self.style_engine.compute_document_features(doc.sentences)
                style_mat = np.atleast_2d(style_mat)

                dense_tensor = self.dense_encoder.extract_sentence_embeddings(doc.sentences)
                dense_mat = np.atleast_2d(dense_tensor.cpu().numpy())

                if dense_mat.shape[0] != style_mat.shape[0]:
                    raise ValueError(f"Sentence count mismatch in doc {doc.doc_id}: dense {dense_mat.shape} vs style {style_mat.shape}")

                fused_mat = np.hstack([dense_mat, style_mat]).astype(np.float32)

            cached_features.append(fused_mat)

        print(f"\n-> Successfully cached all document features to '{cache_path}'")
        torch.save(cached_features, cache_path)
        return cached_features

    def __len__(self) -> int:
        return len(self.docs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        doc = self.docs[idx]
        fused = self.features_list[idx]
        
        return {
            "doc_id": doc.doc_id,
            "parent_doc_id": doc.parent_doc_id,
            "scenario": doc.scenario,
            "fused_features": torch.tensor(fused, dtype=torch.float32),
            "labels": torch.tensor(doc.labels[:len(fused)], dtype=torch.long),
            "boundaries": torch.tensor(doc.boundaries[:len(fused)], dtype=torch.long),
            "seq_len": len(fused)
        }