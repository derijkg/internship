'''
TODO
add train, dev, test split already stratified
add stratifiedkfoldcrossval


'''

import ast
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

try:
    import nltk
except ImportError:
    nltk = None

# Compiled regex for fast non-printable / zero-width character removal
NON_PRINTABLE_RE = re.compile(r"[\u200b\x00-\x1f\x7f-\x9f]")


def normalize_text(text: Any) -> str:
    """Fast text normalization using UNICODE NFKC and regex filtering."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text_str = str(text)
    text_str = unicodedata.normalize("NFKC", text_str)
    # Fast regex replacement instead of pure Python character loop
    text_str = NON_PRINTABLE_RE.sub("", text_str)
    return text_str.strip()


@dataclass
class TextRecord:
    """Dataclass representing a single extracted text record."""

    text: str
    doc_id: str
    is_llm: bool
    generator_model: str
    source_col: str
    level: str
    sentence_idx: Optional[int] = None
    source: Optional[str] = None
    year: Optional[int] = None
    keywords: Optional[str] = None

    @property
    def key(self) -> tuple:
        return (
            self.doc_id,
            self.generator_model,
            self.source_col,
            self.level,
            self.sentence_idx,
            self.text,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "is_llm": int(self.is_llm),
            "generator_model": self.generator_model,
            "source_col": self.source_col,
            "level": self.level,
            "sentence_idx": self.sentence_idx,
            "source": self.source,
            "year": self.year,
            "keywords": self.keywords,
        }


@dataclass
class AbstractData:
    records: List[TextRecord] = field(default_factory=list)

    def to_pandas(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.records])

    def shuffle(self, seed: Optional[int] = 42) -> "AbstractData":
        rng = np.random.default_rng(seed)
        records_copy = list(self.records)
        rng.shuffle(records_copy)
        return AbstractData(records_copy)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> TextRecord:
        return self.records[idx]

    def __iter__(self):
        return iter(self.records)

    def __add__(self, other: "AbstractData") -> "AbstractData":
        return AbstractData(self.records + other.records)


class AbstractDataloader:

    INVALID_FLAGS: Set[str] = {
        "FAILED_GENERATION",
        "FAILED_VALIDATION",
        "NONE",
        "NAN",
        "NA",
        "NULL",
        "<NA>",
        "",
    }

    def __init__(
        self,
        source: Union[str, Path, pd.DataFrame],
        id_col: str = "_id",
        human_abstract_col: str = "abstract",
        human_sentence_col: str = "abstract_sentence",
    ):
        if isinstance(source, (str, Path)):
            path_str = str(source)
            if path_str.endswith(".parquet"):
                self.df = pd.read_parquet(path_str)
            elif path_str.endswith(".csv"):
                self.df = pd.read_csv(path_str)
            else:
                raise ValueError(
                    f"Unsupported format: {path_str}. Expected parquet or csv."
                )
        elif isinstance(source, pd.DataFrame):
            self.df = source.copy()
        else:
            raise TypeError("Source must be a file path or pd.DataFrame.")

        if id_col in self.df.columns:
            self.id_col = id_col
        elif "id" in self.df.columns:
            self.id_col = "id"
        else:
            self.id_col = "_id"
            self.df["_id"] = [f"doc_{idx}" for idx in range(len(self.df))]

        self.human_abstract_col = human_abstract_col
        self.human_sentence_col = human_sentence_col
        self._discover_columns()

    def _discover_columns(self):
        known_meta = {
            self.id_col,
            "source",
            "keywords",
            "year",
            self.human_abstract_col,
            self.human_sentence_col,
        }
        self.llm_columns = [c for c in self.df.columns if c not in known_meta]
        self.llm_col_map = {}
        for col in self.llm_columns:
            if "_" in col:
                model, suffix = col.rsplit("_", 1)
                self.llm_col_map[col] = {"model": model, "suffix": suffix}
            else:
                self.llm_col_map[col] = {"model": col, "suffix": "unknown"}

    @classmethod
    def _validate_and_normalize(
        cls, text: Any, min_words: int = 0
    ) -> Optional[str]:
        """Validates and normalizes text in a single pass."""
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return None
        norm = normalize_text(text)
        if not norm or norm.upper() in cls.INVALID_FLAGS:
            return None
        if min_words > 0 and len(norm.split()) < min_words:
            return None
        return norm

    @classmethod
    def _parse_list_fast(cls, val: Any) -> List[Any]:
        """Fast list parsing using json.loads with fallback to ast.literal_eval."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return []
        if isinstance(val, (list, np.ndarray)):
            return list(val)
        if isinstance(val, str):
            val_str = val.strip()
            if val_str.startswith("[") and val_str.endswith("]"):
                try:
                    return json.loads(val_str)
                except Exception:
                    try:
                        return ast.literal_eval(val_str)
                    except Exception:
                        pass
            return [val_str]
        return [val]

    def _tokenize_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        if nltk is not None:
            try:
                return nltk.sent_tokenize(text)
            except Exception:
                pass
        return [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()
        ]

    def load_human(
        self,
        level: str = "sentence",
        min_words: int = 10,
        offset: int = 0,
        max_samples: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> AbstractData:
        records = []
        # Fast dictionary conversion instead of iterrows()
        rows = self.df.to_dict("records")
        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(rows)

        target_count = (
            (offset + max_samples) if max_samples is not None else float("inf")
        )

        for row in rows:
            if len(records) >= target_count:
                break

            doc_id = str(row[self.id_col])
            meta = {
                "source": row.get("source"),
                "year": row.get("year"),
                "keywords": row.get("keywords"),
            }

            if level == "abstract":
                norm_text = self._validate_and_normalize(
                    row.get(self.human_abstract_col), min_words=min_words
                )
                if norm_text:
                    records.append(
                        TextRecord(
                            text=norm_text,
                            doc_id=doc_id,
                            is_llm=False,
                            generator_model="Human",
                            source_col=self.human_abstract_col,
                            level="abstract",
                            **meta,
                        )
                    )

            elif level == "sentence":
                raw_sents = self._parse_list_fast(
                    row.get(self.human_sentence_col)
                )
                for s_idx, s in enumerate(raw_sents):
                    norm_sent = self._validate_and_normalize(
                        s, min_words=min_words
                    )
                    if norm_sent:
                        records.append(
                            TextRecord(
                                text=norm_sent,
                                doc_id=doc_id,
                                is_llm=False,
                                generator_model="Human",
                                source_col=self.human_sentence_col,
                                level="sentence",
                                sentence_idx=s_idx,
                                **meta,
                            )
                        )
            else:
                raise ValueError("level must be 'abstract' or 'sentence'.")

        data = AbstractData(records)
        if offset > 0 or max_samples is not None:
            end = (
                (offset + max_samples)
                if max_samples is not None
                else len(data)
            )
            data = AbstractData(data.records[offset:end])

        return data

    def load_llm(
        self,
        columns: Optional[Union[str, List[str]]] = None,
        models: Optional[Union[str, List[str]]] = None,
        suffixes: Optional[Union[str, List[str]]] = None,
        level: str = "sentence",
        min_words: int = 10,
        offset: int = 0,
        total_max_samples: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> AbstractData:
        target_cols = (
            [columns]
            if isinstance(columns, str)
            else (
                columns
                if columns is not None
                else self.get_llm_columns(models=models, suffixes=suffixes)
            )
        )

        all_records = []
        rows = self.df.to_dict("records")
        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(rows)

        target_count = (
            (offset + total_max_samples)
            if total_max_samples is not None
            else float("inf")
        )

        for col in target_cols:
            if col not in self.df.columns or len(all_records) >= target_count:
                continue

            suffix = self.llm_col_map.get(col, {}).get("suffix", "")

            for row in rows:
                if len(all_records) >= target_count:
                    break

                doc_id = str(row[self.id_col])
                val = row.get(col)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue

                meta = {
                    "source": row.get("source"),
                    "year": row.get("year"),
                    "keywords": row.get("keywords"),
                }
                parsed_items = self._parse_list_fast(val)

                if level == "abstract":
                    if suffix == "single":
                        valid_items = [
                            self._validate_and_normalize(i)
                            for i in parsed_items
                        ]
                        full_text = " ".join([i for i in valid_items if i])
                    else:
                        full_text = (
                            self._validate_and_normalize(parsed_items[0])
                            if parsed_items
                            else ""
                        )

                    if self._validate_and_normalize(
                        full_text, min_words=min_words
                    ):
                        all_records.append(
                            TextRecord(
                                text=full_text,
                                doc_id=doc_id,
                                is_llm=True,
                                generator_model=col,
                                source_col=col,
                                level="abstract",
                                **meta,
                            )
                        )

                elif level == "sentence":
                    for s_idx, item in enumerate(parsed_items):
                        norm_item = self._validate_and_normalize(item)
                        if not norm_item:
                            continue

                        if suffix == "single":
                            if self._validate_and_normalize(
                                norm_item, min_words=min_words
                            ):
                                all_records.append(
                                    TextRecord(
                                        text=norm_item,
                                        doc_id=doc_id,
                                        is_llm=True,
                                        generator_model=col,
                                        source_col=col,
                                        level="sentence",
                                        sentence_idx=s_idx,
                                        **meta,
                                    )
                                )
                        else:
                            sents = self._tokenize_sentences(norm_item)
                            for sub_idx, s in enumerate(sents):
                                norm_s = self._validate_and_normalize(
                                    s, min_words=min_words
                                )
                                if norm_s:
                                    all_records.append(
                                        TextRecord(
                                            text=norm_s,
                                            doc_id=doc_id,
                                            is_llm=True,
                                            generator_model=col,
                                            source_col=col,
                                            level="sentence",
                                            sentence_idx=sub_idx,
                                            **meta,
                                        )
                                    )

        data = AbstractData(all_records)
        if offset > 0 or total_max_samples is not None:
            end = (
                (offset + total_max_samples)
                if total_max_samples is not None
                else len(data)
            )
            data = AbstractData(data.records[offset:end])

        return data

    def load_dataset(
        self,
        level: str = "sentence",
        llm_columns: Optional[Union[str, List[str]]] = None,
        models: Optional[Union[str, List[str]]] = None,
        suffixes: Optional[Union[str, List[str]]] = None,
        offset_human: int = 0,
        offset_llm: int = 0,
        max_samples_human: Optional[int] = None,
        max_samples_llm: Optional[int] = None,
        samples_per_class: Optional[int] = None,
        min_words: int = 10,
        seed: Optional[int] = 42,
    ) -> AbstractData:
        h_limit = (
            max_samples_human
            if max_samples_human is not None
            else samples_per_class
        )
        l_limit = (
            max_samples_llm
            if max_samples_llm is not None
            else samples_per_class
        )

        human_data = self.load_human(
            level=level,
            min_words=min_words,
            offset=offset_human,
            max_samples=h_limit,
            seed=seed,
        )

        llm_data = self.load_llm(
            columns=llm_columns,
            models=models,
            suffixes=suffixes,
            level=level,
            min_words=min_words,
            offset=offset_llm,
            total_max_samples=l_limit,
            seed=seed,
        )

        return (human_data + llm_data).shuffle(seed=seed)