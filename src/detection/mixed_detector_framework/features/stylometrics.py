# features/stylometrics.py

import re
import string
from typing import Dict, List, Tuple

import numpy as np

# Core Dutch transition markers for stylometric profiling
DUTCH_TRANSITIONS = {
    "echter",
    "bovendien",
    "daarnaast",
    "desalniettemin",
    "kortom",
    "tevens",
    "daardoor",
    "derhalve",
    "bijgevolg",
    "namelijk",
    "hoewel",
    "aldus",
    "immers",
    "enerzijds",
    "anderzijds",
}


class StylometricFeatureEngine:
    """Calculates multi-scale stylometrics, global relative deltas (Δ_global),

    and local boundary gradients (∇_local) across sentence sequences.
    """

    def __init__(self, include_w3: bool = True, include_w5: bool = True):
        self.include_w3 = include_w3
        self.include_w5 = include_w5

    @staticmethod
    def extract_raw_vector(text: str, sentences: List[str]) -> np.ndarray:
        """Extracts a 12-dimensional statistical surface vector from a text

        snippet.
        """
        words = re.findall(r"\w+", text.lower())
        total_chars = max(1, len(text))
        num_words = max(1, len(words))

        if not words or not sentences:
            return np.zeros(12, dtype=np.float32)

        # 1-3. Sentence length metrics
        sent_lengths = [
            len(re.findall(r"\w+", s))
            for s in sentences
            if len(re.findall(r"\w+", s)) > 0
        ]
        mean_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
        var_sent_len = float(np.var(sent_lengths)) if sent_lengths else 0.0
        burstiness = (
            (float(np.std(sent_lengths)) / mean_sent_len)
            if mean_sent_len > 0
            else 0.0
        )

        # 4-5. Word length metrics
        word_lengths = [len(w) for w in words]
        mean_word_len = float(np.mean(word_lengths))
        var_word_len = float(np.var(word_lengths))

        # 6-7. Vocabulary richness (TTR & Hapax Legomena)
        unique_words = set(words)
        ttr = len(unique_words) / num_words
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        hapax_ratio = (
            sum(1 for w, c in word_counts.items() if c == 1) / num_words
        )

        # 8. Discourse transition ratio
        transition_count = sum(1 for w in words if w in DUTCH_TRANSITIONS)
        transition_ratio = transition_count / num_words

        # 9-11. Punctuation and whitespace formatting features
        spaces_count = text.count(" ")
        double_spaces = text.count("  ")
        punc_count = sum(1 for c in text if c in string.punctuation)

        space_ratio = spaces_count / total_chars
        double_space_ratio = double_spaces / total_chars
        punc_ratio = punc_count / total_chars

        # 12. Log length scaling
        log_char_len = float(np.log1p(total_chars))

        return np.array(
            [
                mean_sent_len,
                var_sent_len,
                burstiness,
                mean_word_len,
                var_word_len,
                ttr,
                hapax_ratio,
                transition_ratio,
                space_ratio,
                double_space_ratio,
                punc_ratio,
                log_char_len,
            ],
            dtype=np.float32,
        )

    def compute_document_features(
        self, sents: List[str]
    ) -> Tuple[np.ndarray, int]:
        """Calculates fused multi-scale features for an entire sequence of

        sentences.

        Returns:
            features: Matrix of shape [N_sentences, Feature_Dim]
            feature_dim: Integer total feature dimension
        """
        N = len(sents)
        if N == 0:
            return np.zeros((0, 12), dtype=np.float32), 12

        doc_text = " ".join(sents)
        doc_style = self.extract_raw_vector(doc_text, sents)

        # Precompute window texts & raw vectors for all sentences
        w1_styles = np.zeros((N, 12), dtype=np.float32)
        w3_styles = np.zeros((N, 12), dtype=np.float32)
        w5_styles = np.zeros((N, 12), dtype=np.float32)

        for i in range(N):
            # W1: Sentence i
            w1_sents = [sents[i]]
            w1_styles[i] = self.extract_raw_vector(sents[i], w1_sents)

            # W3: Window [i-1, i, i+1]
            if self.include_w3:
                w3_sents = sents[max(0, i - 1) : min(N, i + 2)]
                w3_styles[i] = self.extract_raw_vector(
                    " ".join(w3_sents), w3_sents
                )

            # W5: Window [i-2 ... i+2]
            if self.include_w5:
                w5_sents = sents[max(0, i - 2) : min(N, i + 3)]
                w5_styles[i] = self.extract_raw_vector(
                    " ".join(w5_sents), w5_sents
                )

        feature_blocks = [w1_styles]

        # Process W3 features & gradients
        if self.include_w3:
            w3_delta_global = w3_styles - doc_style  # Δ_global
            w3_grad_local = np.zeros_like(w3_styles)  # ∇_local = W3_i - W3_{i-1}
            w3_grad_local[1:] = w3_styles[1:] - w3_styles[:-1]

            feature_blocks.extend(
                [w3_styles, w3_delta_global, w3_grad_local]
            )

        # Process W5 features & gradients
        if self.include_w5:
            w5_delta_global = w5_styles - doc_style  # Δ_global
            w5_grad_local = np.zeros_like(w5_styles)  # ∇_local = W5_i - W5_{i-1}
            w5_grad_local[1:] = w5_styles[1:] - w5_styles[:-1]

            feature_blocks.extend(
                [w5_styles, w5_delta_global, w5_grad_local]
            )

        fused_matrix = np.hstack(feature_blocks).astype(np.float32)
        return fused_matrix, fused_matrix.shape[1]