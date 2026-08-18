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