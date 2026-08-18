import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


@dataclass
class DetectionResult:
    """Dataclass holding complete zero-shot detection metrics for a sentence."""
    sentence: str
    num_tokens: int
    resonance_vector: np.ndarray        # R(S) = [l_1, l_2, ..., l_K]
    mean_likelihood: float             # Mean(R(S))
    cfi: float                          # Contextual Friction Index
    angles: np.ndarray                 # Token-to-token transition angles (theta)
    theta_var: float                   # Var(theta) - Logit Trajectory Curvature
    theta_mean: float                  # Mean(theta)
    decision_score: float              # Combined zero-shot decision score D(S)
    is_human_predicted: bool           # True if high turbulence/friction (Human)


class ContextualFrictionDetector:
    """
    Sentence-Level Zero-Shot LLM Detector using Contextual Scaffolding,
    Contextual Friction Index (CFI), and Top-K Vocabulary Trajectory Geometry.
    """

    DEFAULT_SCAFFOLDS = [
        # P1: 18th-century philosophical prose
        "In considering the fundamental nature of human understanding and moral sentiment, one must observe that ",
        # P2: Modern Python docstring / code context
        "def process_pipeline_data(input_stream: Dict[str, Any], validate_schema: bool = True) -> Dict:\n    \"\"\"Execute telemetry aggregation ",
        # P3: Clinical medical report
        "PATIENT ADMISSION SUMMARY: The 58-year-old subject presented to emergency care exhibiting acute onset of ",
        # P4: Casual conversational transcript
        "hey so I was talking to Sarah yesterday and she was like total non-believer about the whole situation but ",
        # P5: Formal legal contract / statute
        "IN WITNESS WHEREOF, the parties hereto have executed this Agreement as of the effective date. Pursuant to Section 4.2, ",
        # P6: Sci-Fi / Worldbuilding narrative
        "The sub-quantum drift engines hummed softly in the vacuum of sector 7, casting a faint azure radiation across "
    ]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        device: Optional[str] = None,
        top_n: int = 50,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.1,
        scaffolds: Optional[List[str]] = None,
        scaffold_std_baseline: Optional[Union[float, List[float]]] = None,
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the Observer Model and Tokenizer.

        Args:
            model_name: HuggingFace model path/identifier.
            device: 'cuda', 'cpu', or 'mps'.
            top_n: Size N for Top-N vocabulary trajectory extraction.
            alpha: Calibration weight for CFI.
            beta: Calibration weight for Logit Trajectory Curvature Var(theta).
            gamma: Calibration weight for Likelihood Baseline Mean(R(S)).
            scaffolds: Custom list of K diverse prompt prefixes.
            scaffold_std_baseline: Baseline standard deviation sigma_P across prompts.
            torch_dtype: Model torch precision (e.g., float16, bfloat16, float32).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_n = top_n
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Prompts (Scaffolds)
        self.scaffolds = scaffolds if scaffolds is not None else self.DEFAULT_SCAFFOLDS
        self.num_scaffolds = len(self.scaffolds)

        # Prompt standard deviation scale sigma_P
        if scaffold_std_baseline is None:
            self.sigma_p = torch.ones(self.num_scaffolds, device=self.device)
        elif isinstance(scaffold_std_baseline, (float, int)):
            self.sigma_p = torch.full((self.num_scaffolds,), float(scaffold_std_baseline), device=self.device)
        else:
            self.sigma_p = torch.tensor(scaffold_std_baseline, dtype=torch.float32, device=self.device)

        # Load Observer Model and Tokenizer
        print(f"Loading observer model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=self.device
        )
        self.model.eval()

        # Storage for pre-computed KV-caches (Optimization 1)
        self._precomputed_prompt_caches = None

    # =========================================================================
    # Step 1 & Step 2: Contextual Scaffolding & Contextual Friction Matrix
    # =========================================================================
    @torch.inference_mode()
    def compute_contextual_resonance(self, sentence: str) -> Tuple[np.ndarray, float, float]:
        """
        Evaluates sentence S across K diverse scaffold prompts in parallel.

        Returns:
            resonance_vector: R(S) array of log-likelihoods ell_k(S) [K].
            mean_likelihood: Mean(R(S)).
            cfi: Contextual Friction Index CFI(S).
        """
        # Tokenize sentence independently to maintain exact token alignment
        sentence_ids = self.tokenizer.encode(sentence, add_special_tokens=False, return_tensors="pt")[0]
        M = len(sentence_ids)
        if M == 0:
            raise ValueError("Candidate sentence produced 0 tokens.")

        # Construct padded batched inputs [P_k + S] for parallel forward pass
        batch_input_ids = []
        prompt_lengths = []

        for p_str in self.scaffolds: 
            p_ids = self.tokenizer.encode(p_str, add_special_tokens=False)
            prompt_lengths.append(len(p_ids))
            combined_ids = p_ids + sentence_ids.tolist()
            batch_input_ids.append(torch.tensor(combined_ids, dtype=torch.long))

        # Max length in batch
        max_len = max(len(ids) for ids in batch_input_ids)
        pad_id = self.tokenizer.pad_token_id

        # Tensor allocations with left-padding #TODO CHECK PADDING & ATTENTION MASK
        B = self.num_scaffolds
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=self.device)

        for k, ids in enumerate(batch_input_ids):
            seq_len = len(ids)
            offset = max_len - seq_len
            input_ids[k, offset:] = ids.to(self.device)
            attention_mask[k, offset:] = 1

        # Single Batched Forward Pass: O(M) sequence compute cost
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Shape: (K, max_len, Vocab_Size)

        # Log-softmax over vocabulary
        log_probs = F.log_softmax(logits, dim=-1)

        # Compute conditional log-likelihood ell_k(S) for each scaffold k #TODO??
        resonance_vector = torch.zeros(B, device=self.device, dtype=torch.float32)

        for k in range(B):
            L_k = prompt_lengths[k]
            Z_k = max_len - (L_k + M)  # Padding offset for scaffold k

            # Candidate sentence target tokens: indices [Z_k + L_k : Z_k + L_k + M]
            target_ids = input_ids[k, Z_k + L_k : Z_k + L_k + M]  # Shape: (M,)

            # Predicting logits: indices [Z_k + L_k - 1 : Z_k + L_k + M - 1]
            pred_log_probs = log_probs[k, Z_k + L_k - 1 : Z_k + L_k + M - 1, :]  # Shape: (M, Vocab)

            # Gather log probs at target token positions
            token_log_probs = pred_log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            
            # ell_k(S) = sum_{i=1}^M log P_M(t_i | P_k, t_1...t_{i-1})
            resonance_vector[k] = token_log_probs.sum()

        # Calculations
        R_mean = resonance_vector.mean()
        
        # Normalized Residuals: (R(S) - Mean(R(S))) / sigma_P
        normalized_residuals = (resonance_vector - R_mean) / self.sigma_p
        
        # Contextual Friction Index: Var( normalized_residuals )
        cfi_val = torch.var(normalized_residuals, unbiased=False).item()

        return resonance_vector.cpu().numpy(), R_mean.item(), cfi_val

    # =========================================================================
    # Step 3: Top-K Vocabulary Trajectory Geometry (Micro-Topology)
    # =========================================================================
    @torch.inference_mode()
    def compute_trajectory_geometry(self, sentence: str) -> Tuple[np.ndarray, float, float, int]:
        """
        Extracts Top-N logit vectors v_i across sequence positions and calculates
        the differential angle trajectory curvature Var(theta).

        Returns:
            angles: Differential angular changes theta_i [M-2].
            theta_var: Var(theta) - Angular Path Curvature.
            theta_mean: Mean(theta).
            num_tokens: Number of candidate sentence tokens M.
        """
        sentence_ids = self.tokenizer.encode(sentence, add_special_tokens=False, return_tensors="pt").to(self.device)
        M = sentence_ids.shape[1]

        if M < 2:
            # Cannot compute angular curvature for sequences shorter than 2 tokens
            return np.array([]), 0.0, 0.0, M

        # Forward pass on candidate sentence alone
        outputs = self.model(sentence_ids)
        logits = outputs.logits[0]  # Shape: (M, Vocab_Size)

        # Retrieve Top-N logit vector v_i in R^N at each step
        # top_logits shape: (M, Top_N)
        top_logits, _ = torch.topk(logits, k=self.top_n, dim=-1)
        top_logits = top_logits.to(torch.float32)

        # Compute pairwise consecutive angular transitions theta_i
        # v_i and v_{i+1} for i = 0 ... M-2
        v_i = top_logits[:-1]      # Shape: (M-1, Top_N)
        v_next = top_logits[1:]    # Shape: (M-1, Top_N)

        # Cosine Similarity along top logit vector dimensions
        dot_product = (v_i * v_next).sum(dim=-1)
        norm_v_i = torch.norm(v_i, p=2, dim=-1)
        norm_v_next = torch.norm(v_next, p=2, dim=-1)

        cos_sim = dot_product / (norm_v_i * norm_v_next + 1e-8)
        
        # Numerical clamping for stable arccos
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
        angles = torch.arccos(cos_sim)  # Shape: (M-1,)

        if len(angles) > 1:
            theta_var = torch.var(angles, unbiased=False).item()
            theta_mean = torch.mean(angles).item()
        else:
            theta_var = 0.0
            theta_mean = angles[0].item() if len(angles) == 1 else 0.0

        return angles.cpu().numpy(), theta_var, theta_mean, M

    # =========================================================================
    # Combined Zero-Shot Detection Pipeline
    # =========================================================================
    def detect(self, sentence: str, threshold: float = 0.0) -> DetectionResult:
        """
        Executes full detection pipeline and computes combined decision metric D(S):
            D(S) = alpha * CFI(S) + beta * Var(theta) - gamma * Mean(R(S))
        """
        # Step 1 & 2: Contextual Scaffolding & Friction Matrix
        resonance_vec, r_mean, cfi = self.compute_contextual_resonance(sentence)

        # Step 3: Top-K Vocabulary Trajectory Geometry
        angles, theta_var, theta_mean, M = self.compute_trajectory_geometry(sentence)

        # Combined Decision Metric D(S)
        decision_score = (
            self.alpha * cfi +
            self.beta * theta_var -
            self.gamma * r_mean
        )

        is_human = decision_score > threshold

        return DetectionResult(
            sentence=sentence,
            num_tokens=M,
            resonance_vector=resonance_vec,
            mean_likelihood=r_mean,
            cfi=cfi,
            angles=angles,
            theta_var=theta_var,
            theta_mean=theta_mean,
            decision_score=decision_score,
            is_human_predicted=is_human
        )

    # =========================================================================
    # Optimization 1: KV-Cache Pre-computation Helper (Optional Execution Path)
    # =========================================================================
    @torch.inference_mode()
    def precompute_scaffold_kv_caches(self):
        """
        Pre-computes and stores KV-caches for static prompt scaffolds.
        Allows sentence evaluation by re-using KV cache across evaluations.
        """
        print("Pre-computing Key-Value (KV) caches for static scaffolds...")
        self._precomputed_prompt_caches = []

        for prompt in self.scaffolds:
            p_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            outputs = self.model(p_ids, use_cache=True)
            self._precomputed_prompt_caches.append({
                "past_key_values": outputs.past_key_values,
                "prompt_length": p_ids.shape[1],
                "last_logit": outputs.logits[:, -1, :]
            })
        print(f"Successfully cached {len(self.scaffolds)} scaffold KV states.")


# =============================================================================
# Practical Execution Demonstration
# =============================================================================
if __name__ == "__main__":
    # Initialize Detector with an open-weights observer model
    # (Use lightweight Qwen2.5-1.5B or GPT2 for fast CPU/GPU execution)
    detector = ContextualFrictionDetector(
        model_name="Qwen/Qwen2.5-1.5B",
        top_n=50,
        alpha=1.0,
        beta=1.0,
        gamma=0.05
    )

    # Test Candidate Sentences
    ai_sentence = (
        "Furthermore, it is crucial to recognize that artificial intelligence "
        "plays an increasingly vital role in optimizing modern organizational workflows."
    )

    human_sentence = (
        "Honestly, I stumbled across that weird obscure bug again at 3 AM "
        "and almost threw my laptop right out the bedroom window."
    )

    print("\n" + "=" * 70)
    print("RUNNING SENTENCE-LEVEL LLM DETECTION EXPERIMENT")
    print("=" * 70)

    for label, text in [("AI-Generated Candidate", ai_sentence), ("Human Candidate", human_sentence)]:
        result = detector.detect(text)

        print(f"\n--- Candidate [{label}] ---")
        print(f"Sentence: \"{result.sentence}\"")
        print(f"Token Count (M)            : {result.num_tokens}")
        print(f"Mean Likelihood Mean(R)    : {result.mean_likelihood:.4f}")
        print(f"Contextual Friction (CFI)  : {result.cfi:.6f}")
        print(f"Trajectory Curvature Var(θ): {result.theta_var:.6f}")
        print(f"----------------------------------------")
        print(f"Combined Decision Score D(S): {result.decision_score:.6f}")
        print(f"Predicted Class             : {'HUMAN TEXT' if result.is_human_predicted else 'LLM GENERATED'}")
        print("Resonance Profile R(S)      :", np.round(result.resonance_vector, 2))