import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class DCSRConfig:
    model_name: str = "gpt2"               # Try "Qwen/Qwen2.5-1.5B" for even higher accuracy
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    top_n_logits: int = 50
    temperature: float = 0.7
    head_tokens: int = 5                  # Number of prefix tokens to evaluate for scaffold friction
    
    scaffolds: List[str] = field(default_factory=lambda: [
        "In this peer-reviewed academic monograph, we rigorously establish that ",
        "```python\n# Utility script to process unstructured text records\n# ",
        "Hark! Upon the foggy, tempestuous moors of 19th-century Yorkshire, ",
        "yo bro honestly IMO if you ask me about this, ",
        "CLINICAL CASE REPORT: The patient presented with acute onset of "
    ])


@dataclass
class DCSRResult:
    text: str
    num_tokens: int
    head_cfi: float                    # Contextual Friction on first H tokens
    angular_trajectory_variance: float  # Var(theta) across sentence
    head_context_gain: float            # Mean Delta Gain on first H tokens
    dcsr_score: float                   # Higher = More likely AI


class DCSRDetector:
    def __init__(self, config: Optional[DCSRConfig] = None):
        self.config = config or DCSRConfig()
        print(f"[DCSR] Initializing model '{self.config.model_name}' on {self.config.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            device_map="auto" if self.config.device == "cuda" else None
        )
        if self.config.device == "cpu":
            self.model.to("cpu")
            
        self.model.eval()

        self._scaffold_data = {}
        self._precompute_scaffolds()

    @torch.no_grad()
    def _precompute_scaffolds(self):
        """Pre-computes KV-caches AND the tail logit for predicting t_0."""
        print("[DCSR] Pre-computing static scaffold caches...")
        for i, scaffold in enumerate(self.config.scaffolds):
            p_ids = self.tokenizer.encode(scaffold, return_tensors="pt").to(self.config.device)
            outputs = self.model(p_ids, use_cache=True)
            
            # The last logit of the scaffold predicts the FIRST token of the candidate sentence
            tail_logit = outputs.logits[0, -1, :] 
            
            self._scaffold_data[f"scaffold_{i}"] = {
                "past_key_values": outputs.past_key_values,
                "tail_logit": tail_logit
            }

    @torch.no_grad()
    def _get_exact_token_log_probs(self, sentence: str, scaffold_key: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes mathematically exact per-token log-probs for ALL tokens t_0...t_{M-1}.
        Returns (per_token_log_probs, logits_matrix).
        """
        s_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
        s_tensor = torch.tensor([s_ids], device=self.config.device)
        M = len(s_ids)

        if scaffold_key is None:
            # Standalone Baseline Pass
            outputs = self.model(s_tensor)
            logits = outputs.logits[0] # (M, vocab_size)
            
            # Unconditioned t_0 log_prob is approximated via uniform/unigram or shift
            # For baseline: logits[:-1] predicts s_ids[1:]
            shift_logits = logits[:-1, :]
            shift_labels = s_tensor[0, 1:]
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_lps = log_probs[torch.arange(M - 1), shift_labels]
            return token_lps, logits

        else:
            # Conditioned Pass with Exact Alignment
            scaff = self._scaffold_data[scaffold_key]
            past_kv = scaff["past_key_values"]
            tail_logit = scaff["tail_logit"] # Logit predicting t_0
            
            # Predict t_1...t_{M-1} given past_kv + s_tensor
            outputs = self.model(s_tensor, past_key_values=past_kv, use_cache=False)
            logits = outputs.logits[0] # (M, vocab_size)
            
            # Combine tail_logit (for t_0) with logits[:-1] (for t_1...t_{M-1})
            combined_logits = torch.cat([tail_logit.unsqueeze(0), logits[:-1, :]], dim=0) # (M, vocab_size)
            
            log_probs = F.log_softmax(combined_logits, dim=-1)
            token_lps = log_probs[torch.arange(M), s_tensor[0]]
            return token_lps, combined_logits

    def _compute_trajectory_variance(self, logits: torch.Tensor) -> float:
        scaled_logits = logits / self.config.temperature
        top_logits, _ = torch.topk(scaled_logits, k=self.config.top_n_logits, dim=-1)
        probs = F.softmax(top_logits, dim=-1)
        
        if probs.shape[0] < 2:
            return 0.0
            
        norm_probs = F.normalize(probs, p=2, dim=-1)
        v1, v2 = norm_probs[:-1, :], norm_probs[1:, :]
        
        cos_sim = torch.clamp(torch.sum(v1 * v2, dim=-1), -1.0 + 1e-7, 1.0 - 1e-7)
        angles = torch.acos(cos_sim)
        
        ang_var = torch.var(angles).item()
        return ang_var if not math.isnan(ang_var) else 0.0

    def analyze_sentence(self, sentence: str) -> DCSRResult:
        sentence = sentence.strip()
        
        # 1. Standalone baseline
        base_lps, base_logits = self._get_exact_token_log_probs(sentence, scaffold_key=None)
        
        # Slice Head Tokens (first H tokens, offset by 1 for baseline comparison)
        H = min(self.config.head_tokens, base_lps.shape[0])
        base_head_lps = base_lps[:H]
        
        # 2. Conditioned Head Deltas
        scaffold_head_gains = []
        
        for i in range(len(self.config.scaffolds)):
            key = f"scaffold_{i}"
            cond_lps, _ = self._get_exact_token_log_probs(sentence, scaffold_key=key)
            
            # Compare head tokens (cond_lps has exact t_0...t_{H-1})
            cond_head_lps = cond_lps[:H]
            
            # Delta gain on sentence head
            head_gain = (cond_head_lps - base_head_lps).mean().item()
            scaffold_head_gains.append(head_gain)

        # 3. Compute Head Metrics
        head_cfi = float(np.var(scaffold_head_gains))
        head_mean_gain = float(np.mean(scaffold_head_gains))
        ang_var = self._compute_trajectory_variance(base_logits)

        # Final Formula:
        # AI text = High Head Mean Gain, Low Head CFI, Low AngVar
        # Human text = Low Head Mean Gain, High Head CFI, High AngVar
        alpha, beta, gamma = 1.0, 5.0, 1.0
        dcsr_score = (gamma * head_mean_gain) - (alpha * head_cfi) - (beta * ang_var)

        return DCSRResult(
            text=sentence,
            num_tokens=len(self.tokenizer.encode(sentence, add_special_tokens=False)),
            head_cfi=head_cfi,
            angular_trajectory_variance=ang_var,
            head_context_gain=head_mean_gain,
            dcsr_score=dcsr_score
        )


if __name__ == "__main__":
    ai_sentence = "Quantum computing leverages superposition and entanglement to perform complex computations exponentially faster than classical systems."
    human_sentence = "Honestly though, quantum bits are weird enough to make any sane programmer reconsider their entire career choice."

    config = DCSRConfig(model_name="gpt2", head_tokens=5, temperature=0.7)
    detector = DCSRDetector(config)

    print("\n================ RUNNING EXACT DCSR ENGINE ================\n")
    
    res_ai = detector.analyze_sentence(ai_sentence)
    print(f"[AI Sample]\n  Score: {res_ai.dcsr_score:.4f} | HeadGain: {res_ai.head_context_gain:.4f} | HeadCFI: {res_ai.head_cfi:.4f} | AngVar: {res_ai.angular_trajectory_variance:.4f}\n")
    
    res_human = detector.analyze_sentence(human_sentence)
    print(f"[Human Sample]\n  Score: {res_human.dcsr_score:.4f} | HeadGain: {res_human.head_context_gain:.4f} | HeadCFI: {res_human.head_cfi:.4f} | AngVar: {res_human.angular_trajectory_variance:.4f}\n")