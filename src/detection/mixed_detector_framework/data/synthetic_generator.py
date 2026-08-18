# data/synthetic_generator.py

import ast
import hashlib
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

    # ==========================================
    # 1. Cleaning & Parsing Utilities
    # ==========================================
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

    # ==========================================
    # 2. Scenario Generators
    # ==========================================
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
        """Human writes first k sentences, LLM writes remaining sentences."""
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
        """Inserts an AI snippet (1-2 sentences) inside a human document."""
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
        """Replaces a mid-document span of human sentences with AI rewrites."""
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
        """Involves 2 different LLM models modifying different parts of the document."""
        if len(available_ai_dict) < 2 or len(human_sents) < 5:
            return None

        selected_models = self.rng.sample(list(available_ai_dict.keys()), k=2)
        m1, m2 = selected_models[0], selected_models[1]

        sents = list(human_sents)
        labels = [0] * len(human_sents)

        # Restrain pos1 to allow room for pos2 in second half of sequence
        pos1 = self.rng.randint(1, max(1, (len(human_sents) // 2) - 1))
        ai1_sent = self.rng.choice(available_ai_dict[m1])
        sents[pos1] = ai1_sent
        labels[pos1] = 1

        pos2_start = max(pos1 + 1, len(human_sents) // 2)
        pos2_end = len(human_sents) - 1
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

    # ==========================================
    # 3. Main Processing Engine
    # ==========================================
    def process_dataframe(
        self, df: pd.DataFrame
    ) -> List[SyntheticDocument]:
        """Generates a list of SyntheticDocument objects from raw pandas DataFrame."""
        synthetic_docs: List[SyntheticDocument] = []

        print(
            f"--- Data Engine: Processing {len(df)} Raw Documents into Synthetic Mixed-Authorship ---"
        )

        for idx, row in df.iterrows():
            parent_id = str(row["_id"]) if "_id" in row and pd.notna(row["_id"]) else f"doc_{idx}"
            source = str(row.get("source", "unknown"))

            # Extract human sentences
            human_sents = self.parse_sentence_array(
                row.get("abstract_sentence", [])
            )
            if len(human_sents) < self.min_sentences:
                continue

            # Extract available AI sentence arrays per model
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
            synthetic_docs.append(
                self._create_pure_human(parent_id, source, human_sents)
            )

            # 2. Model-specific mixed scenarios
            for model_name, ai_sents in available_ai.items():
                synthetic_docs.append(
                    self._create_pure_ai(
                        parent_id, source, model_name, ai_sents
                    )
                )

                if len(human_sents) >= 4:
                    synthetic_docs.append(
                        self._create_continuation(
                            parent_id, source, model_name, human_sents, ai_sents
                        )
                    )

                if len(human_sents) >= 3:
                    synthetic_docs.append(
                        self._create_infilling(
                            parent_id, source, model_name, human_sents, ai_sents
                        )
                    )

                if len(human_sents) >= 4:
                    synthetic_docs.append(
                        self._create_substitution(
                            parent_id, source, model_name, human_sents, ai_sents
                        )
                    )

            # 3. Multi-Model Hybrid
            if len(available_ai) >= 2 and len(human_sents) >= 5:
                hybrid_doc = self._create_multi_model_hybrid(
                    parent_id, source, available_ai, human_sents
                )
                if hybrid_doc:
                    synthetic_docs.append(hybrid_doc)

        print(
            f"-> Successfully generated {len(synthetic_docs)} synthetic documents."
        )
        return synthetic_docs