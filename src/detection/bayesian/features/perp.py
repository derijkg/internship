# features/lpt_extractor.py

from typing import List, Union, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class MultiModelLPTEngine:
    """
    Extracts 10-dimensional Local Probability Topology (LPT) statistical features
    per sentence using a reference open-source Causal SLM/LLM.
    
    Extracted Features per Sentence:
    --------------------------------
    1. mean_log_prob       : Mean log-likelihood log P(t_i | t_<i)
    2. perplexity          : exp(-mean_log_prob)
    3. std_log_prob        : Token log-prob volatility across sentence
    4. min_log_prob        : Minimum log-prob (captures unexpected human words)
    5. mean_entropy        : Average next-token distribution entropy H(P_i)
    6. mean_log_rank       : Mean log(1 + rank(t_i)) under reference model
    7. rank_top1_ratio     : Fraction of tokens where reference model predicted Top-1
    8. rank_top5_ratio     : Fraction of tokens in Top-5 predictions
    9. delta_log_prob_std  : Volatility of token-to-token probability transitions
    10. log_prob_skewness  : Asymmetry of log-probability distribution
    """


    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        device: Optional[str] = None
    ):
        if model_names is None:
            model_names = ["Qwen/Qwen2.5-0.5B-Instruct", "gpt2"]

        self.model_names = model_names
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizers = []
        self.models = []

        print(f"-> Initializing Multi-Model LPT Engine with {len(model_names)} models on {self.device}...")
        for name in model_names:
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            
            mod = AutoModelForCausalLM.from_pretrained(
                name, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
            ).to(self.device)
            mod.eval()

            self.tokenizers.append(tok)
            self.models.append(mod)

        # 10 features per model * num_models
        self.feature_dim = 10 * len(self.model_names)

    @torch.no_grad()
    def compute_sentence_lpt(self, sentence: str) -> np.ndarray:
        if not sentence or not sentence.strip():
            return np.zeros(self.feature_dim, dtype=np.float32)

        ensemble_features = []

        for tok, mod in zip(self.tokenizers, self.models):
            inputs = tok(sentence, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]

            if seq_len <= 1:
                ensemble_features.append(np.zeros(10, dtype=np.float32))
                continue

            outputs = mod(**inputs)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            log_probs_all = F.log_softmax(shift_logits, dim=-1)
            target_log_probs = torch.gather(log_probs_all, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1).squeeze(0)
            target_log_probs_np = target_log_probs.cpu().numpy().astype(np.float32)

            # Token Ranks
            sorted_indices = torch.argsort(shift_logits, dim=-1, descending=True)
            matches = (sorted_indices == shift_labels.unsqueeze(-1))
            target_ranks = torch.where(matches)[-1].squeeze(0).cpu().numpy().astype(np.float32) + 1

            # Next-token entropy
            probs_all = torch.exp(log_probs_all)
            entropy_per_token = -torch.sum(probs_all * log_probs_all, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)

            mean_log_prob = float(np.mean(target_log_probs_np))
            perplexity = float(np.exp(-mean_log_prob))
            std_log_prob = float(np.std(target_log_probs_np)) if len(target_log_probs_np) > 1 else 0.0
            min_log_prob = float(np.min(target_log_probs_np))
            mean_entropy = float(np.mean(entropy_per_token))

            log_ranks = np.log(target_ranks)
            mean_log_rank = float(np.mean(log_ranks))
            rank_top1_ratio = float(np.mean(target_ranks == 1))
            rank_top5_ratio = float(np.mean(target_ranks <= 5))
            delta_log_prob_std = float(np.std(np.diff(target_log_probs_np))) if len(target_log_probs_np) > 1 else 0.0
            skewness = float(np.mean(((target_log_probs_np - mean_log_prob) / (std_log_prob + 1e-6)) ** 3)) if len(target_log_probs_np) > 2 else 0.0

            ensemble_features.append(np.array([
                mean_log_prob, perplexity, std_log_prob, min_log_prob, mean_entropy,
                mean_log_rank, rank_top1_ratio, rank_top5_ratio, delta_log_prob_std, skewness
            ], dtype=np.float32))

        return np.concatenate(ensemble_features, axis=-1).astype(np.float32)

    def extract_document_lpt(self, sentences: List[str]) -> np.ndarray:
        if not sentences:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        return np.vstack([self.compute_sentence_lpt(s) for s in sentences]).astype(np.float32)