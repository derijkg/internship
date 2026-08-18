import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union, Any, Literal
import torch
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    PreTrainedModel, 
    PreTrainedTokenizerBase
)


DEFAULT_SCAFFOLDS = [
    # P1: 18th-century philosophical prose
    "In considering the fundamental nature of human understanding and moral sentiment, one must observe that ",
    # P2: Modern Python docstring / code context
    'def process_pipeline_data(input_stream: Dict[str, Any], validate_schema: bool = True) -> Dict:\n    """Execute telemetry aggregation ',
    # P3: Clinical medical report
    "PATIENT ADMISSION SUMMARY: The 58-year-old subject presented to emergency care exhibiting acute onset of ",
    # P4: Casual conversational transcript
    "hey so I was talking to Sarah yesterday and she was like total non-believer about the whole situation but ",
    # P5: Formal legal contract / statute
    "IN WITNESS WHEREOF, the parties hereto have executed this Agreement as of the effective date. Pursuant to Section 4.2, ",
    # P6: Sci-Fi / Worldbuilding narrative
    "The sub-quantum drift engines hummed softly in the vacuum of sector 7, casting a faint azure radiation across "
]


@dataclass
class Step2FrictionResult:
    """Dataclass holding Step 2 Differential Resonance & Contextual Friction metrics."""
    sentence: str
    tokens: List[str]
    num_tokens: int
    resonance_vector: np.ndarray             # R(S): Shape (K,)
    differential_resonance: np.ndarray       # Delta R(S) = R(S) - R_uncond(S): Shape (K,)
    unconditioned_ll: float                  # Unconditioned log-likelihood (length-normalized)
    mean_likelihood: float                   # Mean(R(S)) across scaffolds
    cfi: float                               # Global Contextual Friction Index
    per_token_cfi: np.ndarray                # Per-token friction profile: Shape (M,)
    token_log_probs: np.ndarray              # Scaffold token log-probs matrix: Shape (K, M)


@dataclass
class Step3TrajectoryResult:
    """Dataclass holding Step 3 Vocabulary Trajectory Geometry metrics."""
    sentence: str
    num_tokens: int
    angles: np.ndarray                       # Transition angles theta_1..theta_{M-1} (radians)
    theta_var: float                         # Var(theta) - Logit Path Curvature Variance
    theta_mean: float                        # Mean(theta) - Average Angular Deviation
    trajectory_matrix: np.ndarray            # Trajectory matrix V of shape (M, N or V)


@dataclass
class DetectionResult:
    """Dataclass holding complete zero-shot detection metrics and decision score D(S)."""
    sentence: str
    num_tokens: int
    resonance_vector: np.ndarray             # R(S)
    differential_resonance: np.ndarray       # Delta R(S)
    mean_likelihood: float                   # Mean(R(S))
    cfi: float                               # Contextual Friction Index
    per_token_cfi: np.ndarray                # Per-token friction array
    angles: np.ndarray                       # Transition angles theta
    theta_var: float                         # Var(theta)
    theta_mean: float                        # Mean(theta)
    decision_score: float                    # Combined zero-shot decision metric D(S)


# ==========================================
# STEP 1: Contextual Scaffolding Engine
# ==========================================

@torch.inference_mode()
def compute_scaffold_log_likelihoods(
    sentence: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    scaffolds: List[str] = DEFAULT_SCAFFOLDS,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Step 1: Contextual Scaffolding (Virtual Context Expansion)

    Computes the conditional log-likelihood profile ell_k(S) of a candidate 
    sentence S across K orthogonally diverse synthetic prefix prompts (scaffolds).
    """
    if device is None:
        device = next(model.parameters()).device

    sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
    M = len(sentence_ids)
    if M == 0:
        raise ValueError("Candidate sentence produced 0 tokens.")

    scaffold_ids_list = [
        tokenizer.encode(p_str, add_special_tokens=True) 
        for p_str in scaffolds
    ]
    K = len(scaffold_ids_list)

    batch_input_ids = [ids + sentence_ids for ids in scaffold_ids_list]
    max_len = max(len(ids) for ids in batch_input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    input_ids = torch.full((K, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((K, max_len), dtype=torch.long, device=device)

    for k, ids in enumerate(batch_input_ids):
        seq_len = len(ids)
        offset = max_len - seq_len
        input_ids[k, offset:] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[k, offset:] = 1

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids
    )

    target_ids = input_ids[:, max_len - M :]  # Shape: (K, M)
    pred_logits = outputs.logits[:, max_len - M - 1 : max_len - 1, :].float()  # Shape: (K, M, Vocab)
    pred_log_probs = F.log_softmax(pred_logits, dim=-1)

    token_log_probs = pred_log_probs.gather(
        dim=-1, index=target_ids.unsqueeze(-1)
    ).squeeze(-1)  # Shape: (K, M)

    scaffold_log_likelihoods = token_log_probs.sum(dim=-1)  # Shape: (K,)

    return scaffold_log_likelihoods, token_log_probs


@torch.inference_mode()
def compute_unconditioned_log_likelihood(
    sentence: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes unconditioned token log-probabilities log P_M(t_i | t_1..t_{i-1})
    for candidate sentence S without prefix scaffolding.
    """
    if device is None:
        device = next(model.parameters()).device

    sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
    M = len(sentence_ids)
    if M == 0:
        raise ValueError("Candidate sentence produced 0 tokens.")

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.tensor([[bos_id] + sentence_ids], dtype=torch.long, device=device)

    outputs = model(input_ids=input_ids)

    pred_logits = outputs.logits[:, :-1, :].float()  # Shape: (1, M, Vocab)
    pred_log_probs = F.log_softmax(pred_logits, dim=-1)

    target_ids = torch.tensor([sentence_ids], dtype=torch.long, device=device)  # Shape: (1, M)
    uncond_token_log_probs = pred_log_probs.gather(
        dim=-1, index=target_ids.unsqueeze(-1)
    ).squeeze(-1)  # Shape: (1, M)

    uncond_log_likelihood = uncond_token_log_probs.sum()

    return uncond_log_likelihood, uncond_token_log_probs


# ==========================================
# STEP 3: Vocabulary Trajectory Geometry
# ==========================================

@torch.inference_mode()
def compute_vocabulary_trajectory_geometry(
    sentence: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    top_n: int = 100,
    use_softmax: bool = True,
    alignment: Literal["vocab_space", "confidence_spectrum"] = "vocab_space",
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """
    Step 3: Vocabulary Trajectory Geometry (Micro-Topology)

    Analyzes path curvature (differential angles theta_i) of prediction distributions 
    across vocabulary space or confidence spectrum space.
    """
    if device is None:
        device = next(model.parameters()).device

    sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
    M = len(sentence_ids)
    if M == 0:
        raise ValueError("Candidate sentence produced 0 tokens.")

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.tensor([[bos_id] + sentence_ids], dtype=torch.long, device=device)

    outputs = model(input_ids=input_ids)
    
    pred_logits = outputs.logits[:, :-1, :].squeeze(0).float()  # Shape: (M, Vocab)

    if use_softmax:
        probs_or_logits = F.softmax(pred_logits, dim=-1)  # Shape: (M, Vocab)
    else:
        # Subtract mean across vocabulary dimension to make logit representations
        # shift-invariant before L2 normalization.
        probs_or_logits = pred_logits - pred_logits.mean(dim=-1, keepdim=True)

    if alignment == "vocab_space":
        # Full V-dimensional probability/logit vectors aligned by token index
        v_matrix = probs_or_logits
        # Create an inspection matrix for display (top-N probabilities)
        probs = F.softmax(pred_logits, dim=-1)
        inspection_matrix, _ = torch.topk(probs, k=min(top_n, probs.shape[-1]), dim=-1)
    else:
        # Top-N probability profile sorted by rank (Confidence Spectrum)
        N = min(top_n, probs_or_logits.shape[-1])
        v_matrix, _ = torch.topk(probs_or_logits, k=N, dim=-1)  # Shape: (M, N)
        inspection_matrix = v_matrix

    if M <= 1:
        angles = np.array([], dtype=np.float32)
        return angles, 0.0, 0.0, inspection_matrix.cpu().numpy()

    # L2-normalize prediction vectors
    v_norm = F.normalize(v_matrix, p=2, dim=-1)

    # Differential cosine similarity between successive steps v_i and v_{i+1}
    v_i = v_norm[:-1, :]     # Shape: (M-1, dim)
    v_next = v_norm[1:, :]   # Shape: (M-1, dim)

    cos_sim = (v_i * v_next).sum(dim=-1)  # Shape: (M-1,)
    
    # Clamp cosine similarity to avoid acos numerical NaN errors
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)

    # Transition angles in radians: theta_i = arccos(cos_sim_i)
    angles_tensor = torch.acos(cos_sim)  # Shape: (M-1,)

    angles = angles_tensor.cpu().numpy()
    theta_mean = float(np.mean(angles))
    theta_var = float(np.var(angles, ddof=0))

    return angles, theta_var, theta_mean, inspection_matrix.cpu().numpy()


# ==========================================
# Main Detector Pipeline Class
# ==========================================

class ContextualFrictionDetector:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        device: Optional[str] = None,
        scaffolds: Optional[List[str]] = None,
        scaffold_std_baseline: Optional[Union[float, List[float]]] = None,
        torch_dtype: torch.dtype = torch.float16
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaffolds = scaffolds if scaffolds is not None else DEFAULT_SCAFFOLDS
        self.num_scaffolds = len(self.scaffolds)

        if scaffold_std_baseline is None:
            self.sigma_p = torch.ones(self.num_scaffolds, device=self.device)
        elif isinstance(scaffold_std_baseline, (float, int)):
            self.sigma_p = torch.full((self.num_scaffolds,), float(scaffold_std_baseline), device=self.device)
        else:
            self.sigma_p = torch.tensor(scaffold_std_baseline, dtype=torch.float32, device=self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        ).to(self.device)
        self.model.eval()

    def get_scaffold_profile(self, sentence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Step 1: Get raw scaffold conditional log-likelihoods and token log probabilities."""
        return compute_scaffold_log_likelihoods(
            sentence=sentence,
            model=self.model,
            tokenizer=self.tokenizer,
            scaffolds=self.scaffolds,
            device=self.device
        )

    def get_unconditioned_baseline(self, sentence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get unconditioned baseline log-likelihood and token log probabilities."""
        return compute_unconditioned_log_likelihood(
            sentence=sentence,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

    def compute_contextual_friction(
        self, 
        sentence: str,
        length_normalized: bool = True
    ) -> Step2FrictionResult:
        """
        Step 2: Differential Resonance & Contextual Friction Matrix (CFI)
        """
        scaffold_ll, token_log_probs = self.get_scaffold_profile(sentence)
        uncond_ll, _ = self.get_unconditioned_baseline(sentence)

        sentence_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
        M = len(sentence_ids)
        tokens = self.tokenizer.convert_ids_to_tokens(sentence_ids)

        scale_factor = (1.0 / M) if length_normalized else 1.0
        
        resonance_vector = scaffold_ll * scale_factor  # Shape: (K,)
        uncond_ll_norm = (uncond_ll * scale_factor).item()

        differential_resonance = resonance_vector - uncond_ll_norm  # Shape: (K,)

        # Global Contextual Friction Index (CFI)
        # Standardized residuals: z_k = (R_k - R_mean) / sigma_k
        R_mean = resonance_vector.mean()
        residuals = (resonance_vector - R_mean) / self.sigma_p
        cfi_global = torch.mean(residuals ** 2).item()

        # Per-Token Contextual Friction Profile
        T_matrix = token_log_probs  # Shape: (K, M)
        T_mean = T_matrix.mean(dim=0, keepdim=True)  # Shape: (1, M)
        
        token_residuals = (T_matrix - T_mean) / self.sigma_p.unsqueeze(1)
        per_token_cfi = torch.mean(token_residuals ** 2, dim=0).cpu().numpy()  # Shape: (M,)

        return Step2FrictionResult(
            sentence=sentence,
            tokens=tokens,
            num_tokens=M,
            resonance_vector=resonance_vector.cpu().numpy(),
            differential_resonance=differential_resonance.cpu().numpy(),
            unconditioned_ll=uncond_ll_norm,
            mean_likelihood=R_mean.item(),
            cfi=cfi_global,
            per_token_cfi=per_token_cfi,
            token_log_probs=token_log_probs.cpu().numpy()
        )

    def compute_trajectory_geometry(
        self,
        sentence: str,
        top_n: int = 100,
        use_softmax: bool = True,
        alignment: Literal["vocab_space", "confidence_spectrum"] = "vocab_space"
    ) -> Step3TrajectoryResult:
        """
        Step 3: Vocabulary Trajectory Geometry (Micro-Topology)
        """
        angles, theta_var, theta_mean, traj_matrix = compute_vocabulary_trajectory_geometry(
            sentence=sentence,
            model=self.model,
            tokenizer=self.tokenizer,
            top_n=top_n,
            use_softmax=use_softmax,
            alignment=alignment,
            device=self.device
        )

        sentence_ids = self.tokenizer.encode(sentence, add_special_tokens=False)

        return Step3TrajectoryResult(
            sentence=sentence,
            num_tokens=len(sentence_ids),
            angles=angles,
            theta_var=theta_var,
            theta_mean=theta_mean,
            trajectory_matrix=traj_matrix
        )

    def analyze_sentence(
        self,
        sentence: str,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        top_n: int = 100,
        alignment: Literal["vocab_space", "confidence_spectrum"] = "vocab_space"
    ) -> DetectionResult:
        """
        Combines Steps 1, 2, and 3 to yield the zero-shot decision metric D(S):
            D(S) = alpha * CFI(S) + beta * Var(theta) - gamma * Mean(R(S))
        """
        step2_res = self.compute_contextual_friction(sentence)
        step3_res = self.compute_trajectory_geometry(sentence, top_n=top_n, alignment=alignment)

        decision_score = (
            alpha * step2_res.cfi 
            + beta * step3_res.theta_var 
            - gamma * step2_res.mean_likelihood
        )

        return DetectionResult(
            sentence=sentence,
            num_tokens=step2_res.num_tokens,
            resonance_vector=step2_res.resonance_vector,
            differential_resonance=step2_res.differential_resonance,
            mean_likelihood=step2_res.mean_likelihood,
            cfi=step2_res.cfi,
            per_token_cfi=step2_res.per_token_cfi,
            angles=step3_res.angles,
            theta_var=step3_res.theta_var,
            theta_mean=step3_res.theta_mean,
            decision_score=decision_score
        )