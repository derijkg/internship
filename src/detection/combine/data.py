import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

# =====================================================================
# FEATURE EXTRACTORS
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = str(BASE_DIR / "processed_features")

class FeatureExtractorEngine:
    """
    Extracts Micro-Stats (Causal LLM split across GPUs), 
    Macro-Stylometrics (SpaCy), and Dense Embeddings (RobBERT 2023).
    """
    # 1. Define POS_TAGS right at the class level
    POS_TAGS = [
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
        "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "VERB"
    ]
    DUTCH_SIGNAL_WORDS: Set[str] = {
        "bovendien", "daarnaast", "tevens", "hierbij", "daarentegen", 
        "desalniettemin", "weliswaar", "echter", "enerzijds", "anderzijds",
        "folgelijk", "concluderend", "derhalve", "immers", "zodoende", 
        "kortom", "alsnog", "hoewel", "doordat", "nadat", "tot slot"
    }

    def __init__(
        self,
        causal_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        dense_model_name: str = "DTAI-KULeuven/robbert-2023-dutch-large",
        spacy_model: str = "nl_core_news_lg",
    ):
        print(f"Initializing Feature Extractors across available GPUs...")

        # 1. Causal LLM - Automatically split across both GPUs with memory limits
        max_memory = {0: "10GiB", 1: "10GiB"}
        print(f"Loading Causal LLM ({causal_model_name}) split across GPUs...")
        self.causal_tok = AutoTokenizer.from_pretrained(causal_model_name, trust_remote_code=True)
        self.causal_llm = AutoModelForCausalLM.from_pretrained(
            causal_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
        )
        self.causal_llm.eval()

        # 2. Dense Model (RobBERT 2023) - Placed on GPU 1
        self.dense_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        print(f"Loading Dense Encoder ({dense_model_name}) on {self.dense_device}...")
        self.dense_tok = AutoTokenizer.from_pretrained(dense_model_name)
        self.dense_model = AutoModel.from_pretrained(dense_model_name).to(self.dense_device)
        self.dense_model.eval()

        # 3. SpaCy for Macro-Stylometrics
        print(f"Loading SpaCy ({spacy_model})...")
        self.nlp = spacy.load(spacy_model, disable=["ner"])
        self.function_words = self.nlp.Defaults.stop_words

        # 4. Initialize pos_map AFTER POS_TAGS is accessible
        self.pos_map = {tag: i for i, tag in enumerate(self.POS_TAGS)}

    @torch.no_grad()
    def extract_micro_features(self, text: str) -> torch.Tensor:
        """Surprisal, Log-Rank, Token Entropy."""
        # For multi-GPU models, feed inputs to model.device (cuda:0)
        inputs = self.causal_tok(text, return_tensors="pt", truncation=True, max_length=1024).to(self.causal_llm.device)
        input_ids = inputs["input_ids"]

        if input_ids.shape[1] < 2:
            return torch.zeros((1, 3), device="cpu")

        outputs = self.causal_llm(input_ids)
        logits = outputs.logits[0, :-1, :].float()
        targets = input_ids[0, 1:]

        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)

        surprisal = -log_probs[torch.arange(len(targets)), targets]

        target_logits = logits[torch.arange(len(targets)), targets].unsqueeze(-1)
        ranks = (logits > target_logits).sum(dim=-1).float()
        log_rank = torch.log(ranks + 1.0)

        entropy = -torch.sum(probs * log_probs, dim=-1)

        micro_feats = torch.stack([surprisal, log_rank, entropy], dim=-1)
        return micro_feats.cpu()

    def extract_macro_features(self, text: str, target_len: int) -> torch.Tensor:
        """
        Returns a [target_len, 28] macro-stylometric feature matrix covering:
        - Structural (3): Tree Depth, Function Word Flag, Punctuation Flag
        - POS One-Hot (15): Standard SpaCy Universal POS tags
        - Discourse Signals (2): Signal Word Flag, Dutch Subordinate Conjunction Flag
        - Voice & Syntax (2): Passive Auxiliary Flag ('worden'/'zijn'), Word Length
        - Sentence Length Pacing & Burstiness (6): Relative sentence length, Z-score, TTR, etc.
        """
        doc = self.nlp(text)
        words = [t.text.lower() for t in doc if not t.is_punct]
        num_words = max(len(words), 1)

        # Document-wide sentence length statistics for pacing & burstiness
        sentences = list(doc.sents)
        sent_lengths = [len(s) for s in sentences] if sentences else [1]
        mean_sent_len = float(np.mean(sent_lengths))
        std_sent_len = float(np.std(sent_lengths)) + 1e-6

        # Vocabulary richness (Type-Token Ratio - TTR)
        ttr = len(set(words)) / float(num_words)

        token_features = []

        for token in doc:
            # 1. Base Structural Features (3)
            depth = float(len(list(token.ancestors)))
            is_func = 1.0 if token.text.lower() in self.function_words else 0.0
            is_punct = 1.0 if token.is_punct else 0.0

            # 2. POS One-Hot Vector (15)
            pos_vec = [0.0] * len(self.POS_TAGS)
            if token.pos_ in self.pos_map:
                pos_vec[self.pos_map[token.pos_]] = 1.0

            # 3. Dutch Discourse & Signal Words (2)
            is_signal = 1.0 if token.text.lower() in self.DUTCH_SIGNAL_WORDS else 0.0
            is_sconj = 1.0 if token.pos_ == "SCONJ" else 0.0

            # 4. Voice & Morphology (2)
            is_passive_aux = 1.0 if (token.lemma_ in ["worden", "zijn"] and token.pos_ == "AUX") else 0.0
            word_len = float(len(token.text)) / 10.0  # Normalized word length

            # 5. Sentence Pacing & Burstiness Metrics (6)
            sent_len = float(len(token.sent))
            rel_sent_len = sent_len / max(mean_sent_len, 1.0)
            sent_len_zscore = (sent_len - mean_sent_len) / std_sent_len

            pacing_vec = [
                sent_len / 50.0,       # Absolute sentence length (scaled)
                rel_sent_len,          # Relative sentence length to mean
                sent_len_zscore,       # Sentence length Z-score (burstiness)
                ttr,                   # Type-token ratio (lexical diversity)
                std_sent_len / 20.0,   # Document sentence length variance
                float(len(sentences))  # Sentence count
            ]

            # Combine all features for this token (Total: 3 + 15 + 2 + 2 + 6 = 28 features)
            tok_vector = [depth, is_func, is_punct] + pos_vec + [is_signal, is_sconj, is_passive_aux, word_len] + pacing_vec
            token_features.append(tok_vector)

        if not token_features:
            return torch.zeros((target_len, 28), device="cpu")

        macro_tensor = torch.tensor(token_features, dtype=torch.float32)

        # Resample/align SpaCy token count to target subword token length
        if macro_tensor.shape[0] != target_len:
            if macro_tensor.shape[0] == 1:
                macro_tensor = macro_tensor.repeat(target_len, 1)
            else:
                macro_tensor = F.interpolate(
                    macro_tensor.unsqueeze(0).transpose(1, 2),
                    size=target_len,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).transpose(0, 1)

        return macro_tensor.cpu()

    @torch.no_grad()
    def extract_dense_sentence_embeddings(self, sentences: List[str]) -> torch.Tensor:
        """Extracts dense embeddings using RobBERT on dense_device."""
        if not sentences:
            return torch.zeros((1, 1024), device="cpu")

        inputs = self.dense_tok(
            sentences, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(self.dense_device)

        outputs = self.dense_model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return embeddings.cpu()


# =====================================================================
# SYNTHETIC MIXER
# =====================================================================

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


# =====================================================================
# MAIN DATASET GENERATOR & PRE-COMPUTER
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset and pre-compute all features.")
    parser.add_argument("-d", "--data_path", type=str, default="/home/gderijck/internship/data/gold/llm_added.parquet")
    parser.add_argument("-o", "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-s", "--samples_per_row", type=int, default=2)
    parser.add_argument("--causal_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dense_model", type=str, default="DTAI-KULeuven/robbert-2023-dutch-large")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data_path)
    print(f"Loaded parquet file with {len(df)} rows.")

    mixer = DutchSyntheticMixer()
    engine = FeatureExtractorEngine(
        causal_model_name=args.causal_model,
        dense_model_name=args.dense_model,
    )

    metadata_records = []
    features_dict = {}

    models_cols = ["qwen3.6:27b", "gemma4:e4b", "qwen3.5:4b", "gemma4:26b"]

    print("Generating mixed dataset and computing features...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        raw_sents = row.get("abstract_sentence", [])
        
        # 1. Safely parse and sanitize human sentences
        if raw_sents is None or (isinstance(raw_sents, float) and np.isnan(raw_sents)):
            continue
        
        raw_human_list = list(raw_sents) if isinstance(raw_sents, (list, np.ndarray)) else []
        
        # Keep only valid non-empty human string sentences
        human_sents = [
            str(s).strip() for s in raw_human_list 
            if s is not None and isinstance(s, str) and len(str(s).strip()) > 0
        ]
        
        if len(human_sents) == 0:
            continue

        # 2. Parse and strictly validate single-sentence rewrites per model
        single_rewrites = {}
        for m in models_cols:
            col = f"{m}_single"
            col_val = row.get(col)
            
            if col_val is not None and isinstance(col_val, (list, np.ndarray)) and len(col_val) > 0:
                candidate_sents = list(col_val)
                
                # REJECTION RULE: Reject model if sentence count mismatch or if ANY item is None/non-string
                if len(candidate_sents) != len(human_sents):
                    continue
                
                is_valid = True
                cleaned_cand_sents = []
                for item in candidate_sents:
                    if item is None or (isinstance(item, float) and np.isnan(item)) or not isinstance(item, str) or len(str(item).strip()) == 0:
                        is_valid = False
                        break
                    cleaned_cand_sents.append(str(item).strip())

                if is_valid:
                    single_rewrites[m] = cleaned_cand_sents

        row_id = str(row["_id"])

        for sample_i in range(args.samples_per_row):
            # Mix sample (will fall back to pure_human if single_rewrites is empty)
            sample = mixer.mix_sample(human_sents, single_rewrites)
            syn_id = f"{row_id}_syn_{sample_i}"

            # 1. Extract Micro Features
            micro_feats = engine.extract_micro_features(sample["text"])

            # 2. Extract Macro Features
            macro_feats = engine.extract_macro_features(sample["text"], target_len=micro_feats.shape[0])

            # 3. Extract Dense Sentence Embeddings
            dense_sents = engine.extract_dense_sentence_embeddings(sample["sentences"])

            # Save PyTorch Tensors into feature dictionary
            features_dict[syn_id] = {
                "fused_sequence": torch.cat([micro_feats, macro_feats], dim=-1),  # [seq_len, 31]
                "dense_sentence_embs": dense_sents,                                # [num_sents, 1024]
                "labels": sample["labels"],
                "scenario": sample["scenario"]
            }

            metadata_records.append({
                "synthetic_id": syn_id,
                "original_id": row_id,
                "text": sample["text"],
                "sentences": sample["sentences"],
                "labels": sample["labels"],
                "scenario": sample["scenario"]
            })

    # Save Parquet Metadata
    parquet_path = out_dir / "synthetic_metadata.parquet"
    pd.DataFrame(metadata_records).to_parquet(parquet_path, index=False)
    print(f"Saved metadata Parquet to {parquet_path}")

    # Save Sidecar PyTorch Tensors File
    tensors_path = out_dir / "precomputed_features.pt"
    torch.save(features_dict, tensors_path)
    print(f"Saved pre-computed PyTorch tensors to {tensors_path}")



'''
#surprisal of three models
# Assuming s1, s2, s3 are aligned surprisal tensors of shape [seq_len, 1]
surprisals = torch.cat([s1, s2, s3], dim=-1)  # [seq_len, 3]

# 1. Consensus metrics
mean_surprisal = torch.mean(surprisals, dim=-1, keepdim=True)  # [seq_len, 1]
std_surprisal = torch.std(surprisals, dim=-1, keepdim=True)    # [seq_len, 1]

# 2. Pairwise differences
delta_1_2 = s1 - s2
delta_1_3 = s1 - s3
delta_2_3 = s2 - s3

# Combined 8-dim Surprisal Profile
surprisal_block = torch.cat([
    surprisals,      # [S1, S2, S3]  (Raw values)
    mean_surprisal,  # Mean          (Consensus difficulty)
    std_surprisal,   # Std           (Model mismatch level)
    delta_1_2,       # M1 vs M2      (Directional bias)
    delta_1_3,       # M1 vs M3      (Directional bias)
    delta_2_3        # M2 vs M3      (Directional bias)
], dim=-1)
'''

if __name__ == "__main__":
    main()
