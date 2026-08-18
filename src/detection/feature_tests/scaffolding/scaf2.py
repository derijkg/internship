import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class DCSRConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B"  # Try Qwen2.5-1.5B or gpt2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    top_n_logits: int = 50
    temperature: float = 0.7              # Temperature scaling for logit trajectory
    
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
    contextual_friction_index: float  # Var(Delta L_k)
    angular_trajectory_variance: float # Var(theta) with temperature scaling
    mean_context_gain: float           # Mean(Delta L_k)
    dcsr_score: float                  # Combined Zero-Shot Metric
    deltas: Dict[str, float]


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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Pre-compute and store KV Caches for all scaffold prompts for high speed
        self._scaffold_caches = {}
        self._precompute_scaffold_caches()

    @torch.no_grad()
    def _precompute_scaffold_caches(self):
        """Pre-computes Key-Value caches for all static scaffold prompts."""
        print("[DCSR] Pre-computing static KV-caches for scaffolds...")
        for i, scaffold in enumerate(self.config.scaffolds):
            p_ids = self.tokenizer.encode(scaffold, return_tensors="pt").to(self.config.device)
            outputs = self.model(p_ids, use_cache=True)
            # Store past_key_values and length
            self._scaffold_caches[f"scaffold_{i}"] = {
                "past_key_values": outputs.past_key_values,
                "length": p_ids.shape[1]
            }

    @torch.no_grad()
    def _get_sentence_log_probs(self, sentence: str, scaffold_key: Optional[str] = None) -> Tuple[float, torch.Tensor]:
        """
        Evaluates sentence token log-probs.
        If `scaffold_key` is provided, utilizes pre-computed KV cache.
        """
        sentence_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
        s_tensor = torch.tensor([sentence_ids], device=self.config.device)
        sentence_len = len(sentence_ids)

        if scaffold_key is None:
            # Unconditioned baseline pass (Empty Prefix)
            outputs = self.model(s_tensor)
            logits = outputs.logits[0] # (seq_len, vocab_size)
            # Position i predicts sentence_ids[i] (shift by 1)
            shift_logits = logits[:-1, :]
            shift_labels = s_tensor[0, 1:]
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs[torch.arange(len(shift_labels)), shift_labels]
            return token_log_probs.mean().item(), logits
        else:
            # Fast Conditioned Pass using KV-Cache
            cache_info = self._scaffold_caches[scaffold_key]
            past_kv = cache_info["past_key_values"]
            
            outputs = self.model(s_tensor, past_key_values=past_kv, use_cache=False)
            logits = outputs.logits[0] # (sentence_len, vocab_size)
            
            # Here, the first logit predicts sentence_ids[0] using the scaffold's past KV state!
            # We construct a full logit sequence where scaffold tail predicts s_ids[0], logits[0] predicts s_ids[1], etc.
            # To be exact: forward on s_tensor with past_kv outputs logits for each token in s_tensor.
            # The prompt's last KV state + s_tensor[:-1] predicts s_tensor
            
            # Simple loss mask over sentence tokens:
            log_probs = F.log_softmax(logits[:-1, :], dim=-1)
            token_log_probs = log_probs[torch.arange(sentence_len - 1), s_tensor[0, 1:]]
            
            return token_log_probs.mean().item(), logits

    def _compute_temperature_trajectory_curvature(self, logits: torch.Tensor) -> float:
        """
        Computes angular variance of Top-N probability vectors scaled by Temperature T.
        """
        scaled_logits = logits / self.config.temperature
        top_logits, _ = torch.topk(scaled_logits, k=self.config.top_n_logits, dim=-1)
        probs = F.softmax(top_logits, dim=-1)
        
        if probs.shape[0] < 2:
            return 0.0
            
        norm_probs = F.normalize(probs, p=2, dim=-1)
        v1 = norm_probs[:-1, :]
        v2 = norm_probs[1:, :]
        
        cos_sim = torch.clamp(torch.sum(v1 * v2, dim=-1), -1.0 + 1e-7, 1.0 - 1e-7)
        angles = torch.acos(cos_sim)
        
        ang_var = torch.var(angles).item()
        return ang_var if not math.isnan(ang_var) else 0.0

    def analyze_sentence(self, sentence: str) -> DCSRResult:
        sentence = sentence.strip()
        
        # 1. Standalone Baseline Log-Likelihood (L_0)
        baseline_mean_ll, baseline_logits = self._get_sentence_log_probs(sentence, scaffold_key=None)
        
        # 2. Context-Conditioned Log-Likelihoods & Delta Gains
        deltas = {}
        delta_list = []
        
        for i in range(len(self.config.scaffolds)):
            key = f"scaffold_{i}"
            cond_mean_ll, _ = self._get_sentence_log_probs(sentence, scaffold_key=key)
            
            # Delta = Scaffold LL - Baseline LL
            delta = cond_mean_ll - baseline_mean_ll
            deltas[f"Scaffold_{i+1}"] = delta
            delta_list.append(delta)

        # 3. Compute Refined Metrics
        cfi = float(np.var(delta_list))              # Contextual Friction Index
        mean_gain = float(np.mean(delta_list))      # Mean Delta Context Gain
        ang_var = self._compute_temperature_trajectory_curvature(baseline_logits)

        # Refined Formula:
        # AI text benefits smoothly from scaffolds (High mean context gain, Low CFI friction, Low AngVar)
        # Human text responds erratically to scaffolds (Low/Negative mean gain, High CFI friction, High AngVar)
        dcsr_score = mean_gain - (1.5 * cfi) - (2.0 * ang_var)

        return DCSRResult(
            text=sentence,
            num_tokens=len(self.tokenizer.encode(sentence, add_special_tokens=False)),
            contextual_friction_index=cfi,
            angular_trajectory_variance=ang_var,
            mean_context_gain=mean_gain,
            dcsr_score=dcsr_score,
            deltas=deltas
        )


if __name__ == "__main__":
    ai_sentence = "Quantum computing leverages superposition and entanglement to perform complex computations exponentially faster than classical systems."
    human_sentence = "Honestly though, quantum bits are weird enough to make any sane programmer reconsider their entire career choice."

    # Using GPT-2 for local dry run. For real benchmarks, use "Qwen/Qwen2.5-1.5B"
    config = DCSRConfig(model_name="gpt2", top_n_logits=50, temperature=0.7)
    detector = DCSRDetector(config)

    print("\n================ RUNNING REFINED DCSR ENGINE ================\n")
    
    res_ai = detector.analyze_sentence(ai_sentence)
    print(f"[AI Sample]\n  Score: {res_ai.dcsr_score:.4f} | MeanGain: {res_ai.mean_context_gain:.4f} | CFI: {res_ai.contextual_friction_index:.4f} | AngVar: {res_ai.angular_trajectory_variance:.4f}\n")
    
    res_human = detector.analyze_sentence(human_sentence)
    print(f"[Human Sample]\n  Score: {res_human.dcsr_score:.4f} | MeanGain: {res_human.mean_context_gain:.4f} | CFI: {res_human.contextual_friction_index:.4f} | AngVar: {res_human.angular_trajectory_variance:.4f}\n")