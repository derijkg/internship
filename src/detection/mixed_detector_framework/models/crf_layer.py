from typing import List, Literal, Tuple
import torch
import torch.nn as nn

#TODO lr crf

class LinearChainCRF(nn.Module):
    """Pure PyTorch Linear-Chain Conditional Random Field (CRF) for sequence tagging.

    Implements Forward-Backward log-partition Z(x), Viterbi path decoding, and
    marginal posterior state probability calculation.
    """

    def __init__(self, num_tags: int = 2):
        super().__init__()
        self.num_tags = num_tags

        # Transition matrix: transitions[i, j] is score of transitioning FROM tag i TO tag j
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes transition parameters uniformly."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates exact path score S(x, y) for target tag sequence y vectorially."""
        batch_size, seq_len, _ = emissions.shape
        if seq_len == 0:
            return torch.zeros(batch_size, device=emissions.device)

        mask_float = mask.float()

        # 1. Start transition score (masked at token 0)
        score = self.start_transitions[tags[:, 0]] * mask_float[:, 0]

        # 2. Emission scores across sequence
        emit_scores = emissions.gather(2, tags.unsqueeze(-1)).squeeze(-1)
        score = score + (emit_scores * mask_float).sum(dim=1)

        # 3. Transition scores between adjacent valid tokens
        if seq_len > 1:
            trans_scores = self.transitions[tags[:, :-1], tags[:, 1:]]
            trans_mask = mask_float[:, 1:] * mask_float[:, :-1]
            score = score + (trans_scores * trans_mask).sum(dim=1)

        # 4. End transition score based on last valid tag index
        seq_lengths = mask.sum(dim=1).long()
        last_valid_indices = torch.clamp(seq_lengths - 1, min=0)
        last_tags = tags.gather(1, last_valid_indices.unsqueeze(1)).squeeze(1)

        has_valid = (seq_lengths > 0).float()
        score = score + self.end_transitions[last_tags] * has_valid

        return score

    def _compute_log_partition(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Computes log-partition function Z(x) via Forward algorithm in log-space."""
        batch_size, seq_len, num_tags = emissions.shape
        if seq_len == 0:
            return torch.zeros(batch_size, device=emissions.device)

        mask_bool = mask.bool()

        # Log-space forward variables: [batch_size, num_tags]
        forward_var = self.start_transitions.unsqueeze(0) + emissions[:, 0]

        init_mask = mask_bool[:, 0].unsqueeze(1)
        forward_var = torch.where(init_mask, forward_var, torch.zeros_like(forward_var))

        for i in range(1, seq_len):
            mask_i = mask_bool[:, i].unsqueeze(1)
            emit_score = emissions[:, i].unsqueeze(1)
            trans_score = self.transitions.unsqueeze(0)

            next_tag_var = forward_var.unsqueeze(2) + trans_score + emit_score
            forward_var_next = torch.logsumexp(next_tag_var, dim=1)

            forward_var = torch.where(mask_i, forward_var_next, forward_var)

        forward_var = forward_var + self.end_transitions.unsqueeze(0)

        # Zero out log_partition for empty sequences
        has_valid = mask.sum(dim=1) > 0
        log_z = torch.logsumexp(forward_var, dim=1)
        return torch.where(has_valid, log_z, torch.zeros_like(log_z))

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: torch.Tensor,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> torch.Tensor:
        """Computes Negative Log-Likelihood sequence loss: Loss = Z(x) - S(x, y)."""
        log_partition = self._compute_log_partition(emissions, mask)
        path_score = self._compute_score(emissions, tags, mask)
        nll = log_partition - path_score

        if reduction == "none":
            return nll
        elif reduction == "sum":
            return torch.sum(nll)
        return torch.mean(nll)

    def viterbi_decode(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> List[List[int]]:
        """Finds globally optimal sequence path y* using Viterbi algorithm."""
        batch_size, seq_len, num_tags = emissions.shape
        if seq_len == 0:
            return [[] for _ in range(batch_size)]

        mask_bool = mask.bool()
        viterbi_vars = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        history = []

        for i in range(1, seq_len):
            mask_i = mask_bool[:, i].unsqueeze(1)
            broadcast_viterbi = viterbi_vars.unsqueeze(2) + self.transitions.unsqueeze(0)
            max_vars, bptrs = torch.max(broadcast_viterbi, dim=1)

            viterbi_vars_next = max_vars + emissions[:, i]
            viterbi_vars = torch.where(mask_i, viterbi_vars_next, viterbi_vars)
            history.append(bptrs)

        viterbi_vars = viterbi_vars + self.end_transitions.unsqueeze(0)
        best_last_tags = torch.argmax(viterbi_vars, dim=1).cpu().tolist()

        seq_lengths = mask.sum(dim=1).long().cpu().tolist()

        # Optimized CPU transfer: stack history on GPU into 1 transfer
        if history:
            history_numpy = torch.stack(history, dim=0).cpu().numpy()
        else:
            history_numpy = None

        best_paths = []
        for b in range(batch_size):
            seq_len_b = seq_lengths[b]
            if seq_len_b == 0:
                best_paths.append([])
                continue

            best_tag = best_last_tags[b]
            best_path = [best_tag]

            if history_numpy is not None:
                for step_idx in range(seq_len_b - 2, -1, -1):
                    best_tag = int(history_numpy[step_idx, b, best_tag])
                    best_path.append(best_tag)

            best_path.reverse()
            best_paths.append(best_path)

        return best_paths

    def compute_marginal_probabilities(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Computes exact marginal posterior probabilities P(y_i = k | x) via Forward-Backward."""
        batch_size, seq_len, num_tags = emissions.shape
        if seq_len == 0:
            return torch.zeros((batch_size, 0, num_tags), device=emissions.device)

        mask_bool = mask.bool()

        # 1. Forward Pass
        forward_vars = torch.zeros_like(emissions)
        forward_var = self.start_transitions.unsqueeze(0) + emissions[:, 0]

        init_mask = mask_bool[:, 0].unsqueeze(1)
        forward_var = torch.where(init_mask, forward_var, torch.zeros_like(forward_var))
        forward_vars[:, 0] = forward_var

        for i in range(1, seq_len):
            mask_i = mask_bool[:, i].unsqueeze(1)
            emit_score = emissions[:, i].unsqueeze(1)
            trans_score = self.transitions.unsqueeze(0)
            next_tag_var = forward_var.unsqueeze(2) + trans_score + emit_score
            forward_var_next = torch.logsumexp(next_tag_var, dim=1)
            forward_var = torch.where(mask_i, forward_var_next, forward_var)
            forward_vars[:, i] = forward_var

        # 2. Backward Pass
        backward_vars = torch.zeros_like(emissions)
        backward_var = self.end_transitions.unsqueeze(0)

        for i in range(seq_len - 1, -1, -1):
            if i < seq_len - 1:
                mask_i = mask_bool[:, i + 1].unsqueeze(1)
                emit_score = emissions[:, i + 1].unsqueeze(1)
                trans_score = self.transitions.unsqueeze(0)
                next_tag_var = backward_var.unsqueeze(1) + trans_score + emit_score
                backward_var_next = torch.logsumexp(next_tag_var, dim=2)
                backward_var = torch.where(mask_i, backward_var_next, backward_var)
            backward_vars[:, i] = backward_var

        # 3. Combine Marginals: P(y_i | x) = exp(fwd + bwd - log_Z)
        log_marginals = forward_vars + backward_vars
        log_z = torch.logsumexp(
            forward_vars[:, -1] + self.end_transitions.unsqueeze(0),
            dim=-1,
            keepdim=True,
        )
        marginals = torch.exp(log_marginals - log_z.unsqueeze(1))

        return marginals * mask.float().unsqueeze(-1)