import math
from dataclasses import dataclass, field
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

try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


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
    """Dataclass holding complete zero-shot detection metrics and calibrated decision score D(S)."""
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
    decision_score: float                    # Combined calibrated logit score
    ai_probability: Optional[float] = None   # Calibrated probability P(AI | S) in [0, 1]
    subscore_weights: Optional[Dict[str, float]] = None # Applied sub-score weights


# ==========================================
# Fitting Helper Routine (PyTorch Fallback)
# ==========================================

def _fit_logistic_regression_pytorch(
    X: torch.Tensor, 
    y: torch.Tensor, 
    l2_reg: float = 1e-4
) -> Tuple[np.ndarray, float]:
    """Fits binary logistic regression using PyTorch L-BFGS optimizer when scikit-learn is absent."""
    N, D = X.shape
    w = torch.zeros(D, requires_grad=True, device=X.device)
    b = torch.zeros(1, requires_grad=True, device=X.device)
    optimizer = torch.optim.LBFGS([w, b], lr=1.0, max_iter=300)

    def closure():
        optimizer.zero_grad()
        logits = X @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, y.float()) + 0.5 * l2_reg * torch.sum(w ** 2)
        loss.backward()
        return loss

    optimizer.step(closure)
    return w.detach().cpu().numpy(), float(b.detach().cpu().numpy())


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
    theta_var = float(angles_tensor.var(unbiased=False).item()) if M > 2 else 0.0

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

        # Calibration State Attributes
        self.scaler_mean: Optional[torch.Tensor] = None   # Shape (3,)
        self.scaler_std: Optional[torch.Tensor] = None    # Shape (3,)
        self.weights: Optional[torch.Tensor] = None       # Shape (3,)
        self.intercept: float = 0.0
        self.is_calibrated: bool = False

        # Initialize or Calibrate sigma_p
        self._initialize_sigma_p(scaffold_std_baseline, reference_corpus)

    def _initialize_sigma_p(
        self, 
        baseline_input: Optional[Union[float, List[float], str]], 
        reference_corpus: Optional[List[str]]
    ):
        """Helper to resolve sigma_p from file, reference corpus, list, or fallback default."""
        if reference_corpus is not None:
            print(f"Calibrating scaffold standard deviations on {len(reference_corpus)} reference human texts...")
            self.calibrate_scaffold_baselines(reference_corpus)
            return

        if isinstance(baseline_input, str) and os.path.isfile(baseline_input):
            self.load_calibration(baseline_input)
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
        Calibrates sigma_p across a reference dataset of natural human-written sentences.
        """
        all_residuals = []

        for text in reference_texts:
            try:
                scaffold_ll, _, _, _, _, sentence_ids, _ = self._forward_all(text)
                M = len(sentence_ids)
                
                resonance_vector = scaffold_ll / M
                R_mean = resonance_vector.mean()
                
                residuals = resonance_vector - R_mean
                all_residuals.append(residuals.unsqueeze(0))
            except ValueError:
                continue

        if len(all_residuals) < 2:
            raise ValueError("Scaffold baseline calibration requires at least 2 valid tokenized human texts.")

        residual_matrix = torch.cat(all_residuals, dim=0)

        # Standard deviation along sample dimension (dim=0) with zero-variance clamping
        sigma_p_calibrated = torch.std(residual_matrix, dim=0, unbiased=True)
        sigma_p_calibrated = torch.clamp(sigma_p_calibrated, min=1e-5)
        self.sigma_p = sigma_p_calibrated

        print(f"Scaffold Baseline Calibration Complete. Computed sigma_p: {self.sigma_p.cpu().tolist()}")

        if save_path:
            self.save_calibration(save_path)

        return self.sigma_p

    def calibrate_detector(
        self,
        reference_texts: List[str],
        labels: List[int],
        save_path: Optional[str] = None,
        top_n: int = 100,
        alignment: Literal["vocab_space", "confidence_spectrum"] = "vocab_space"
    ) -> Dict[str, Any]:
        """
        Calibrates both the scaffold baselines (sigma_p) and sub-score fusion weights 
        by fitting a Logistic Regression model on normalized sub-scores.

        Args:
            reference_texts: List of text samples.
            labels: List of binary class labels (0 = Human, 1 = LLM/AI).
            save_path: Optional path to save full JSON calibration state.
            top_n: Top-N vocabulary inspection count for trajectory geometry.
            alignment: Trajectory vector alignment space.

        Returns:
            Dict containing calibration parameters, effective unnormalized weights, and statistics.
        """
        if len(reference_texts) != len(labels):
            raise ValueError(f"Mismatch: len(reference_texts)={len(reference_texts)} vs len(labels)={len(labels)}")

        unique_labels = set(labels)
        if not unique_labels.issubset({0, 1}) or len(unique_labels) < 2:
            raise ValueError("Labels must contain binary values (both 0 = Human and 1 = LLM/AI).")

        # Step 1: Calibrate Scaffold Baselines (sigma_p) exclusively on Human texts (label == 0)
        human_texts = [text for text, label in zip(reference_texts, labels) if label == 0]
        if len(human_texts) < 2:
            raise ValueError("Calibration requires at least 2 Human-written samples (label=0) for sigma_p.")

        print(f"Step 1/2: Calibrating scaffold standard deviations (sigma_p) on {len(human_texts)} human texts...")
        self.calibrate_scaffold_baselines(human_texts)

        # Step 2: Extract sub-scores for ALL reference texts
        print(f"Step 2/2: Extracting sub-scores across {len(reference_texts)} reference samples...")
        features = []
        valid_labels = []

        for text, label in zip(reference_texts, labels):
            try:
                # Run evaluation pass
                res = self.analyze_sentence(text, top_n=top_n, alignment=alignment)
                if res.num_tokens < 3:
                    continue  # Skip ultra-short sequences to prevent trajectory variance noise #TODO parallellize and early filter
                features.append([res.cfi, res.theta_var, res.mean_likelihood])
                valid_labels.append(label)
            except ValueError:
                continue

        if len(features) < 10:
            raise ValueError("Too few valid samples produced token sequences for calibration.")

        X_raw = np.array(features, dtype=np.float32)
        y = np.array(valid_labels, dtype=np.int32)

        # Step 3: Compute Standardization Parameters (Mean & Standard Deviation)
        scaler_mean = np.mean(X_raw, axis=0)
        scaler_std = np.std(X_raw, axis=0) + 1e-8  # Epsilon to avoid division by zero

        # Standardize features (Z-score normalization)
        X_norm = (X_raw - scaler_mean) / scaler_std

        # Step 4: Fit Binary Logistic Regression
        if SKLEARN_AVAILABLE:
            clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
            clf.fit(X_norm, y)
            weights = clf.coef_[0]  # Shape (3,)
            intercept = float(clf.intercept_[0])
        else:
            X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=self.device)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
            weights, intercept = _fit_logistic_regression_pytorch(X_tensor, y_tensor)

        # Step 5: Store parameters in Detector state
        self.scaler_mean = torch.tensor(scaler_mean, dtype=torch.float32, device=self.device)
        self.scaler_std = torch.tensor(scaler_std, dtype=torch.float32, device=self.device)
        self.weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.intercept = intercept
        self.is_calibrated = True

        # Unstandardized effective weights for raw features:
        # Logit = (w1/std1)*cfi + (w2/std2)*theta_var + (w3/std3)*R_mean + (intercept - sum(w_i*mu_i/std_i))
        effective_alpha = float(weights[0] / scaler_std[0])
        effective_beta = float(weights[1] / scaler_std[1])
        effective_gamma = float(-weights[2] / scaler_std[2])  # Negated to match paper formula direction

        stats = {
            "num_samples": len(valid_labels),
            "num_human": int(np.sum(y == 0)),
            "num_ai": int(np.sum(y == 1)),
            "scaffold_sigma_p": self.sigma_p.cpu().tolist(),
            "scaler_mean": scaler_mean.tolist(),
            "scaler_std": scaler_std.tolist(),
            "normalized_weights": {
                "w_cfi": float(weights[0]),
                "w_theta_var": float(weights[1]),
                "w_R_mean": float(weights[2]),
                "intercept": intercept
            },
            "effective_unnormalized_weights": {
                "alpha": effective_alpha,
                "beta": effective_beta,
                "gamma": effective_gamma
            }
        }

        print("\n=== Calibration Complete ===")
        print(f"Normalized Weights (z-scores): CFI={weights[0]:.4f}, Theta_Var={weights[1]:.4f}, R_Mean={weights[2]:.4f}, Intercept={intercept:.4f}")
        print(f"Effective Raw Weights: Alpha={effective_alpha:.4f}, Beta={effective_beta:.4f}, Gamma={effective_gamma:.4f}\n")

        if save_path:
            self.save_calibration(save_path, stats)

        return stats

    def save_calibration(self, save_path: str, stats_dict: Optional[Dict[str, Any]] = None):
        """Saves current calibration attributes and baseline sigma_p to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        data = {
            "num_scaffolds": self.num_scaffolds,
            "sigma_p": self.sigma_p.cpu().tolist(),
            "scaffolds": self.scaffolds,
            "is_calibrated": self.is_calibrated,
            "scaler_mean": self.scaler_mean.cpu().tolist() if self.scaler_mean is not None else None,
            "scaler_std": self.scaler_std.cpu().tolist() if self.scaler_std is not None else None,
            "weights": self.weights.cpu().tolist() if self.weights is not None else None,
            "intercept": float(self.intercept),
            "stats": stats_dict
        }
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved calibration parameters to {save_path}")

    def load_calibration(self, load_path: str):
        """Loads calibration state and baseline sigma_p from a JSON file."""
        with open(load_path, "r") as f:
            data = json.load(f)

        self.sigma_p = torch.tensor(data["sigma_p"], dtype=torch.float32, device=self.device)
        if data.get("is_calibrated", False):
            self.scaler_mean = torch.tensor(data["scaler_mean"], dtype=torch.float32, device=self.device)
            self.scaler_std = torch.tensor(data["scaler_std"], dtype=torch.float32, device=self.device)
            self.weights = torch.tensor(data["weights"], dtype=torch.float32, device=self.device)
            self.intercept = float(data["intercept"])
            self.is_calibrated = True
        print(f"Loaded calibration parameters from {load_path}")

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
        Computes zero-shot metrics and calibrated detection score D(S).
        If calibrated via `calibrate_detector()`, normalizes features and returns logit score & AI probability.
        """
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

        raw_subscores = np.array([cfi_global, theta_var, R_mean.item()], dtype=np.float32)

        if self.is_calibrated and self.weights is not None:
            # Standardize sub-scores using calibrated mean and std
            scaler_mean = self.scaler_mean.cpu().numpy()
            scaler_std = self.scaler_std.cpu().numpy()
            norm_subscores = (raw_subscores - scaler_mean) / scaler_std

            w = self.weights.cpu().numpy()
            logit = float(np.dot(w, norm_subscores) + self.intercept)
            ai_probability = 1.0 / (1.0 + math.exp(-logit))
            decision_score = logit
            subscore_weights = {
                "w_cfi": float(w[0]),
                "w_theta_var": float(w[1]),
                "w_R_mean": float(w[2]),
                "intercept": self.intercept
            }
        else:
            # Fallback uncalibrated decision score
            decision_score = (
                alpha * cfi_global 
                + beta * theta_var 
                - gamma * R_mean.item()
            )
            ai_probability = 1.0 / (1.0 + math.exp(-decision_score))
            subscore_weights = {"alpha": alpha, "beta": beta, "gamma": gamma}

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
            decision_score=decision_score,
            ai_probability=ai_probability,
            subscore_weights=subscore_weights
        )


if __name__ == '__main__':
    # Usage Example with Calibration
    detector = ContextualFrictionDetector(model_name="Qwen/Qwen2.5-1.5B")


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
    reference_corpus = []
    reference_labels = []
    reference_id_idx = () #TODO EXCLUDE REFERENCE CORPUS FROM TEST

    for tr in data_config.records:
        reference_corpus.append(tr.text)
        reference_labels.append(int(tr.is_llm))



    # Calibrate both sigma_p and subscore weights
    stats = detector.calibrate_detector(
        reference_texts=reference_corpus,
        labels=reference_labels,
        save_path="./calibration_params.json"
    )

    # Analyze new sentence with calibrated probability output
    res = detector.analyze_sentence("Artificial intelligence is transforming the technological landscape rapidly.")
    print(f"Decision Score (Logit): {res.decision_score:.4f}")
    print(f"Calibrated P(AI): {res.ai_probability:.4f}")



    #TODO 

    #maybe
    #weigth calibration exclude samples used for scaf std calibration

    #LARGER PROJECT
    #find good scaffolds
    #multiple scaffold sets / multilpe languages
    #diff models



    #QUICK ADD
    #add loading calibration from file
    # add tqdm to functions
    #check bos prepend baseline non scaffold sent
    #check scaffold kv cacheprecompute


    #CALC CHANGE
    #parallellize calibration sigma scaf
    #parallellize weight cal 
    #bayesian length smoothing or dynamic prior weight based on token count (for scaf std or weights?)
    #add robust microtopological features (avg token rank/log, pred. entropy curve)

