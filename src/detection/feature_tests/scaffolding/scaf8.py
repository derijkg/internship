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
import unicodedata
import os
import json



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
# STEP 3 Helper Routine
# ==========================================

def _compute_trajectory_geometry_from_logits(
    pred_logits: torch.Tensor,
    top_n: int = 100,
    use_softmax: bool = True,
    alignment: Literal["vocab_space", "confidence_spectrum"] = "vocab_space"
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Computes trajectory geometry angles and statistics directly from pre-computed logits (M, V)."""
    M, V = pred_logits.shape

    if use_softmax:
        probs_or_logits = F.softmax(pred_logits, dim=-1)
    else:
        probs_or_logits = pred_logits - pred_logits.mean(dim=-1, keepdim=True)

    if alignment == "vocab_space":
        v_matrix = probs_or_logits
        probs = probs_or_logits if use_softmax else F.softmax(pred_logits, dim=-1)
        inspection_matrix, _ = torch.topk(probs, k=min(top_n, V), dim=-1)
    else:
        N = min(top_n, V)
        v_matrix, _ = torch.topk(probs_or_logits, k=N, dim=-1)
        inspection_matrix = v_matrix

    if M <= 1:
        angles = np.array([], dtype=np.float32)
        return angles, 0.0, 0.0, inspection_matrix.cpu().numpy()

    v_norm = F.normalize(v_matrix, p=2, dim=-1)
    cos_sim = (v_norm[:-1, :] * v_norm[1:, :]).sum(dim=-1)
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)

    angles_tensor = torch.acos(cos_sim)
    angles = angles_tensor.cpu().numpy()
    
    theta_mean = float(angles_tensor.mean().item())
    theta_var = float(angles_tensor.var(unbiased=False).item())

    return angles, theta_var, theta_mean, inspection_matrix.cpu().numpy()


# ==========================================
# Standalone Functional API
# ==========================================

@torch.inference_mode()
def compute_scaffold_log_likelihoods(
    sentence: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    scaffolds: List[str] = DEFAULT_SCAFFOLDS,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    max_len = max(len(ids) for ids in scaffold_ids_list) + M

    input_ids = torch.full((K, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((K, max_len), dtype=torch.long, device=device)
    sentence_tensor = torch.tensor(sentence_ids, dtype=torch.long, device=device)

    for k, ids in enumerate(scaffold_ids_list):
        p_len = len(ids)
        start_pos = max_len - p_len - M
        input_ids[k, start_pos : start_pos + p_len] = torch.tensor(ids, dtype=torch.long, device=device)
        input_ids[k, max_len - M :] = sentence_tensor
        attention_mask[k, start_pos :] = 1

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids
    )

    pred_logits = outputs.logits[:, max_len - M - 1 : max_len - 1, :].float()
    target_ids = input_ids[:, max_len - M :]

    token_log_probs = -F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.size(-1)),
        target_ids.reshape(-1),
        reduction="none"
    ).reshape(K, M)

    scaffold_log_likelihoods = token_log_probs.sum(dim=-1)
    return scaffold_log_likelihoods, token_log_probs


@torch.inference_mode()
def compute_unconditioned_log_likelihood(
    sentence: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = next(model.parameters()).device

    sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
    M = len(sentence_ids)
    if M == 0:
        raise ValueError("Candidate sentence produced 0 tokens.")

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.tensor([[bos_id] + sentence_ids], dtype=torch.long, device=device)

    outputs = model(input_ids=input_ids)

    pred_logits = outputs.logits[:, :-1, :].float()
    target_ids = input_ids[:, 1:]

    uncond_token_log_probs = -F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.size(-1)),
        target_ids.reshape(-1),
        reduction="none"
    ).unsqueeze(0)

    uncond_log_likelihood = uncond_token_log_probs.sum()
    return uncond_log_likelihood, uncond_token_log_probs


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
    if device is None:
        device = next(model.parameters()).device

    sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
    M = len(sentence_ids)
    if M == 0:
        raise ValueError("Candidate sentence produced 0 tokens.")

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.tensor([[bos_id] + sentence_ids], dtype=torch.long, device=device)

    outputs = model(input_ids=input_ids)
    pred_logits = outputs.logits[0, :-1, :].float()

    return _compute_trajectory_geometry_from_logits(
        pred_logits=pred_logits,
        top_n=top_n,
        use_softmax=use_softmax,
        alignment=alignment
    )


# ==========================================
# Main Detector Class
# ==========================================

class ContextualFrictionDetector:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        device: Optional[str] = None,
        scaffolds: Optional[List[str]] = None,
        scaffold_std_baseline: Optional[Union[float, List[float], str]] = None,
        reference_corpus: Optional[List[str]] = None,
        torch_dtype: torch.dtype = torch.float16
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaffolds = scaffolds if scaffolds is not None else DEFAULT_SCAFFOLDS
        self.num_scaffolds = len(self.scaffolds)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        ).to(self.device)
        self.model.eval()

        # Token & Scaffold Setup
        self.bos_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        self.scaffold_ids_list = [
            self.tokenizer.encode(p_str, add_special_tokens=True) 
            for p_str in self.scaffolds
        ]

        self.all_prompt_ids = [[self.bos_id]] + self.scaffold_ids_list
        self.all_prompt_tensors = [
            torch.tensor(ids, dtype=torch.long, device=self.device)
            for ids in self.all_prompt_ids
        ]
        self.all_prompt_lens = [len(ids) for ids in self.all_prompt_ids]
        self.max_prompt_len = max(self.all_prompt_lens)

        # Initialize or Calibrate sigma_p
        self._initialize_sigma_p(scaffold_std_baseline, reference_corpus)

    def _initialize_sigma_p(
        self, 
        baseline_input: Optional[Union[float, List[float], str]], 
        reference_corpus: Optional[List[str]]
    ):
        """Helper to resolve sigma_p from file, reference corpus, list, or fallback default."""
        if reference_corpus is not None:
            print(f"Calibrating scaffold standard deviations on {len(reference_corpus)} reference texts...")
            self.calibrate_scaffold_baselines(reference_corpus)
            return

        if isinstance(baseline_input, str) and os.path.isfile(baseline_input):
            print(f"Loading calibrated sigma_p from {baseline_input}...")
            with open(baseline_input, "r") as f:
                data = json.load(f)
                self.sigma_p = torch.tensor(data["sigma_p"], dtype=torch.float32, device=self.device)
            return

        if baseline_input is None:
            self.sigma_p = torch.ones(self.num_scaffolds, device=self.device)
        elif isinstance(baseline_input, (float, int)):
            self.sigma_p = torch.full((self.num_scaffolds,), float(baseline_input), device=self.device)
        else:
            self.sigma_p = torch.tensor(baseline_input, dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def calibrate_scaffold_baselines(
        self, 
        reference_texts: List[str], 
        save_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Calibrates sigma_p across a reference dataset of human-written sentences.
        
        Args:
            reference_texts: List of 100-500 natural human sentences in the target language.
            save_path: Optional JSON path to store calibrated sigma_p values.
        """
        all_residuals = []

        for text in reference_texts:
            try:
                # Get scaffold log-likelihoods for this reference sentence
                scaffold_ll, _, _, _, _, sentence_ids, _ = self._forward_all(text)
                M = len(sentence_ids)
                
                # Length-normalized resonance vector R(S): Shape (K,)
                resonance_vector = scaffold_ll / M
                
                # Mean resonance across scaffolds
                R_mean = resonance_vector.mean()
                
                # Residual delta_k = R_k - R_mean: Shape (K,)
                residuals = resonance_vector - R_mean
                all_residuals.append(residuals.unsqueeze(0))
            except ValueError:
                continue  # Skip empty or un-tokenizable texts

        if not all_residuals:
            raise ValueError("Reference dataset produced no valid tokenized sequences.")

        # Stack residuals matrix: Shape (N_samples, K)
        residual_matrix = torch.cat(all_residuals, dim=0)

        # Standard deviation along sample dimension (dim=0)
        sigma_p_calibrated = torch.std(residual_matrix, dim=0, unbiased=True) + 1e-6
        self.sigma_p = sigma_p_calibrated

        print(f"Calibration Complete. Computed sigma_p values: {self.sigma_p.cpu().tolist()}")

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump({
                    "num_scaffolds": self.num_scaffolds,
                    "sigma_p": self.sigma_p.cpu().tolist(),
                    "scaffolds": self.scaffolds
                }, f, indent=2)
            print(f"Saved calibrated sigma_p to {save_path}")

        return self.sigma_p

    @torch.inference_mode()
    def _forward_all(
        self, 
        sentence: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[str]]:
        """
        Executes a SINGLE unified forward pass for both unconditioned (Row 0)
        and all K scaffold prompts (Rows 1..K).
        """
        sentence_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
        M = len(sentence_ids)
        if M == 0:
            raise ValueError("Candidate sentence produced 0 tokens.")

        tokens = self.tokenizer.convert_ids_to_tokens(sentence_ids)
        num_prompts = self.num_scaffolds + 1
        max_len = self.max_prompt_len + M

        input_ids = torch.full((num_prompts, max_len), self.pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((num_prompts, max_len), dtype=torch.long, device=self.device)
        sentence_tensor = torch.tensor(sentence_ids, dtype=torch.long, device=self.device)

        for k in range(num_prompts):
            p_len = self.all_prompt_lens[k]
            start_pos = max_len - p_len - M
            input_ids[k, start_pos : start_pos + p_len] = self.all_prompt_tensors[k]
            input_ids[k, max_len - M :] = sentence_tensor
            attention_mask[k, start_pos :] = 1

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        pred_logits = outputs.logits[:, max_len - M - 1 : max_len - 1, :].float()
        target_ids = input_ids[:, max_len - M :]

        token_log_probs = -F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            target_ids.reshape(-1),
            reduction="none"
        ).reshape(num_prompts, M)

        uncond_token_log_probs = token_log_probs[0:1]         # Shape: (1, M)
        uncond_ll = uncond_token_log_probs.sum()             # Scalar
        scaffold_token_log_probs = token_log_probs[1:]       # Shape: (K, M)
        scaffold_ll = scaffold_token_log_probs.sum(dim=-1)   # Shape: (K,)
        uncond_logits = pred_logits[0]                       # Shape: (M, Vocab)

        return (
            scaffold_ll,
            scaffold_token_log_probs,
            uncond_ll,
            uncond_token_log_probs,
            uncond_logits,
            sentence_ids,
            tokens
        )

    def get_scaffold_profile(self, sentence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        scaffold_ll, scaffold_token_log_probs, _, _, _, _, _ = self._forward_all(sentence)
        return scaffold_ll, scaffold_token_log_probs

    def get_unconditioned_baseline(self, sentence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, uncond_ll, uncond_token_log_probs, _, _, _ = self._forward_all(sentence)
        return uncond_ll, uncond_token_log_probs

    def compute_contextual_friction(
        self, 
        sentence: str,
        length_normalized: bool = True
    ) -> Step2FrictionResult:
        (
            scaffold_ll,
            token_log_probs,
            uncond_ll,
            _,
            _,
            sentence_ids,
            tokens
        ) = self._forward_all(sentence)

        M = len(sentence_ids)
        scale_factor = (1.0 / M) if length_normalized else 1.0
        
        resonance_vector = scaffold_ll * scale_factor
        uncond_ll_norm = (uncond_ll * scale_factor).item()

        differential_resonance = resonance_vector - uncond_ll_norm

        R_mean = resonance_vector.mean()
        residuals = (resonance_vector - R_mean) / self.sigma_p
        cfi_global = torch.mean(residuals ** 2).item()

        T_matrix = token_log_probs
        T_mean = T_matrix.mean(dim=0, keepdim=True)
        
        token_residuals = (T_matrix - T_mean) / self.sigma_p.unsqueeze(1)
        per_token_cfi = torch.mean(token_residuals ** 2, dim=0).cpu().numpy()

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
        (
            _,
            _,
            _,
            _,
            uncond_logits,
            sentence_ids,
            _
        ) = self._forward_all(sentence)

        angles, theta_var, theta_mean, traj_matrix = _compute_trajectory_geometry_from_logits(
            pred_logits=uncond_logits,
            top_n=top_n,
            use_softmax=use_softmax,
            alignment=alignment
        )

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
        (
            scaffold_ll,
            token_log_probs,
            uncond_ll,
            _,
            uncond_logits,
            sentence_ids,
            tokens
        ) = self._forward_all(sentence)

        M = len(sentence_ids)
        scale_factor = 1.0 / M
        
        resonance_vector = scaffold_ll * scale_factor
        uncond_ll_norm = (uncond_ll * scale_factor).item()

        differential_resonance = resonance_vector - uncond_ll_norm

        R_mean = resonance_vector.mean()
        residuals = (resonance_vector - R_mean) / self.sigma_p
        cfi_global = torch.mean(residuals ** 2).item()

        T_matrix = token_log_probs
        T_mean = T_matrix.mean(dim=0, keepdim=True)
        token_residuals = (T_matrix - T_mean) / self.sigma_p.unsqueeze(1)
        per_token_cfi = torch.mean(token_residuals ** 2, dim=0).cpu().numpy()

        angles, theta_var, theta_mean, _ = _compute_trajectory_geometry_from_logits(
            pred_logits=uncond_logits,
            top_n=top_n,
            use_softmax=True,
            alignment=alignment
        )

        decision_score = (
            alpha * cfi_global 
            + beta * theta_var 
            - gamma * R_mean.item()
        )

        return DetectionResult(
            sentence=sentence,
            num_tokens=M,
            resonance_vector=resonance_vector.cpu().numpy(),
            differential_resonance=differential_resonance.cpu().numpy(),
            mean_likelihood=R_mean.item(),
            cfi=cfi_global,
            per_token_cfi=per_token_cfi,
            angles=angles,
            theta_var=theta_var,
            theta_mean=theta_mean,
            decision_score=decision_score
        )



if __name__ == '__main__':
    #run test
    from ...data_loader import AbstractDataloader
    loader = AbstractDataloader('/home/gderijck/internship/data/gold/llm_added.parquet')
    data_config = loader.load_dataset(
        level='abstract',
        suffixes=['full'],
        max_samples_human=2000,
        max_samples_llm=2000,
        seed=42
    )

