import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class DCSRConfig:
    """Configuration for DCSR Zero-Shot Detection."""
    model_name: str = "Qwen/Qwen2.5-1.5B"  # Fast, highly capable observer model
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    top_n_logits: int = 50                 # Dimensionality for vocabulary trajectory geometry
    
    # Orthogonal Scaffold Prompts spanning wildly different domains/styles
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
    contextual_friction_index: float  # CFI: Variance across scaffold responses
    angular_trajectory_variance: float # Var(theta): Micro-logit directional turbulence
    dcsr_score: float                  # Combined detection metric (Higher = likely AI)
    scaffold_log_likelihoods: Dict[str, float]


class DCSRDetector:
    def __init__(self, config: Optional[DCSRConfig] = None):
        self.config = config or DCSRConfig()
        print(f"[DCSR] Loading observer model '{self.config.model_name}' onto {self.config.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            device_map="auto" if self.config.device == "cuda" else None
        )
        if self.config.device == "cpu":
            self.model.to("cpu")
            
        self.model.eval()
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def _get_sentence_log_prob_under_prefix(self, prefix: str, sentence: str) -> Tuple[float, torch.Tensor]:
        """
        Calculates log-likelihood of target `sentence` conditioned on `prefix`,
        and returns the logit dynamics corresponding to the sentence tokens.
        """
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        sentence_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
        
        if len(sentence_ids) < 2:
            raise ValueError("Sentence must contain at least 2 tokens for trajectory analysis.")
            
        full_input_ids = torch.tensor([prefix_ids + sentence_ids], device=self.config.device)
        
        outputs = self.model(full_input_ids)
        logits = outputs.logits[0]  # Shape: (seq_len, vocab_size)
        
        # We care about predicting sentence tokens.
        # Position i in logits predicts token at full_input_ids[i+1].
        prefix_len = len(prefix_ids)
        sentence_len = len(sentence_ids)
        
        target_logits_start = prefix_len - 1
        target_logits_end = prefix_len + sentence_len - 1
        
        # Extract target logits predicting sentence tokens
        sentence_logits = logits[target_logits_start:target_logits_end, :] # (sentence_len, vocab_size)
        target_token_ids = torch.tensor(sentence_ids, device=self.config.device)
        
        # Calculate Log Probabilities
        log_probs = F.log_softmax(sentence_logits, dim=-1)
        token_log_probs = log_probs[torch.arange(sentence_len), target_token_ids]
        
        total_log_likelihood = token_log_probs.sum().item()
        
        return total_log_likelihood, sentence_logits

    def _compute_trajectory_curvature(self, logits: torch.Tensor) -> float:
        """
        Computes the step-wise angular variance (persistent topology curvature) 
        of the top-N probability distribution vector across sentence tokens.
        """
        # logits shape: (sentence_len, vocab_size)
        top_logits, _ = torch.topk(logits, k=self.config.top_n_logits, dim=-1)
        probs = F.softmax(top_logits, dim=-1) # (sentence_len, top_n_logits)
        
        if probs.shape[0] < 2:
            return 0.0
            
        # Normalize top-N probability vectors
        norm_probs = F.normalize(probs, p=2, dim=-1)
        
        # Cosine similarity between consecutive token distribution vectors
        v1 = norm_probs[:-1, :] # (M-1, top_n)
        v2 = norm_probs[1:, :]  # (M-1, top_n)
        
        cos_sim = torch.sum(v1 * v2, dim=-1)
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Compute angles in radians
        angles = torch.acos(cos_sim)
        
        # Variance of directional transitions
        angular_variance = torch.var(angles).item()
        return angular_variance if not math.isnan(angular_variance) else 0.0

    def analyze_sentence(self, sentence: str) -> DCSRResult:
        """
        Executes the full DCSR pipeline on a single sentence.
        """
        sentence = sentence.strip()
        scaffold_lls = {}
        
        raw_ll_list = []
        sentence_logits_baseline = None
        
        for i, scaffold in enumerate(self.config.scaffolds):
            ll, sentence_logits = self.get_scaffold_ll(scaffold, sentence)
            scaffold_lls[f"Scaffold_{i+1}"] = ll
            raw_ll_list.append(ll)
            
            if i == 0:
                sentence_logits_baseline = sentence_logits

        # 1. Compute Contextual Friction Index (CFI)
        # Standardize log-likelihoods by token length to get per-token average cross-scaffold
        sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
        num_tokens = len(sentence_tokens)
        
        normalized_lls = [ll / num_tokens for ll in raw_ll_list]
        
        # Lower CFI = Low response friction across contexts = More likely AI
        # Higher CFI = Erratic friction / anisotropic clashes = More likely Human
        cfi = float(np.var(normalized_lls))
        
        # 2. Compute Top-N Angular Trajectory Variance
        angular_variance = self._compute_trajectory_curvature(sentence_logits_baseline)
        
        # 3. Compute Composite Zero-Shot Score
        # AI text typically exhibits: Low CFI + Low Angular Variance + High Mean Likelihood
        mean_ll = float(np.mean(normalized_lls))
        
        # We construct dcsr_score such that HIGHER value = MORE LIKELY AI
        # Note: Weights can be calibrated empirically.
        alpha, beta, gamma = 1.0, 2.0, 0.5
        dcsr_score = (gamma * mean_ll) - (alpha * cfi) - (beta * angular_variance)
        
        return DCSRResult(
            text=sentence,
            num_tokens=num_tokens,
            contextual_friction_index=cfi,
            angular_trajectory_variance=angular_variance,
            dcsr_score=dcsr_score,
            scaffold_log_likelihoods=scaffold_lls
        )

    def get_scaffold_ll(self, scaffold: str, sentence: str) -> Tuple[float, torch.Tensor]:
        return self._get_sentence_log_prob_under_prefix(scaffold, sentence)


# --- Quick Test Execution ---
if __name__ == "__main__":
    # Test sentences (Short sentence comparison)
    ai_sentence = "Quantum computing leverages superposition and entanglement to perform complex computations exponentially faster than classical systems."
    human_sentence = "Honestly though, quantum bits are weird enough to make any sane programmer reconsider their entire career choice."

    # Initialize Detector (Uses GPT-2 as a small local baseline if Qwen isn't downloaded yet)
    # Switch to "Qwen/Qwen2.5-1.5B" or "meta-llama/Meta-Llama-3-8B" for real benchmarks!
    config = DCSRConfig(model_name="gpt2", top_n_logits=50)
    detector = DCSRDetector(config)

    print("\n--- ANALYZING SENTENCES ---\n")
    
    res_ai = detector.analyze_sentence(ai_sentence)
    print(f"[AI Sample] Score: {res_ai.dcsr_score:.4f} | CFI: {res_ai.contextual_friction_index:.4f} | AngVar: {res_ai.angular_trajectory_variance:.4f}")
    
    res_human = detector.analyze_sentence(human_sentence)
    print(f"[Human Sample] Score: {res_human.dcsr_score:.4f} | CFI: {res_human.contextual_friction_index:.4f} | AngVar: {res_human.angular_trajectory_variance:.4f}")