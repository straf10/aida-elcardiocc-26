from __future__ import annotations

from typing import List

import torch
from torch import nn


class PartialCRF(nn.Module):
    """Linear-chain CRF with support for partial token labels via allow-mask."""

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags = int(num_tags)
        self.start_transitions = nn.Parameter(torch.empty(self.num_tags))
        self.end_transitions = nn.Parameter(torch.empty(self.num_tags))
        self.transitions = nn.Parameter(torch.empty(self.num_tags, self.num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        allow_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        neg_inf = torch.finfo(emissions.dtype).min
        alpha = self.start_transitions.unsqueeze(0).expand(batch_size, -1)
        has_started = torch.zeros(batch_size, dtype=torch.bool, device=emissions.device)

        for t in range(seq_len):
            mask_t = mask[:, t]
            if not torch.any(mask_t):
                continue
            emit_t = emissions[:, t, :]
            trans_next = torch.logsumexp(
                alpha.unsqueeze(2) + self.transitions.unsqueeze(0),
                dim=1,
            ) + emit_t
            start_next = alpha + emit_t
            next_alpha = torch.where(
                has_started.unsqueeze(1),
                trans_next,
                start_next,
            )
            if allow_mask is not None:
                allow_t = allow_mask[:, t, :]
                next_alpha = next_alpha.masked_fill(~allow_t, neg_inf)
            alpha = torch.where(mask_t.unsqueeze(1), next_alpha, alpha)
            has_started = has_started | mask_t

        end_scores = torch.logsumexp(alpha + self.end_transitions.unsqueeze(0), dim=1)
        return torch.where(has_started, end_scores, torch.zeros_like(end_scores))

    def forward(
        self,
        emissions: torch.Tensor,
        allow_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.bool()
        numerator = self._log_partition(emissions, mask=mask, allow_mask=allow_mask.bool())
        denominator = self._log_partition(emissions, mask=mask, allow_mask=None)
        return -(numerator - denominator).mean()

    def decode(
        self,
        emissions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[List[int]]:
        batch_size, seq_len, num_tags = emissions.shape
        mask = attention_mask.bool()
        history = torch.zeros(
            seq_len,
            batch_size,
            num_tags,
            dtype=torch.long,
            device=emissions.device,
        )

        score = self.start_transitions.unsqueeze(0).expand(batch_size, -1)
        has_started = torch.zeros(batch_size, dtype=torch.bool, device=emissions.device)

        for t in range(seq_len):
            mask_t = mask[:, t]
            if not torch.any(mask_t):
                continue
            emit_t = emissions[:, t, :]
            transition_scores = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_prev_scores, best_prev_tags = transition_scores.max(dim=1)
            history[t] = best_prev_tags

            start_scores = score + emit_t
            continue_scores = best_prev_scores + emit_t
            next_scores = torch.where(
                has_started.unsqueeze(1),
                continue_scores,
                start_scores,
            )
            score = torch.where(mask_t.unsqueeze(1), next_scores, score)
            has_started = has_started | mask_t

        score = score + self.end_transitions.unsqueeze(0)
        best_last_tags = torch.argmax(score, dim=1)

        paths: List[List[int]] = []
        for b in range(batch_size):
            valid_positions = torch.nonzero(mask[b], as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                paths.append([0] * seq_len)
                continue

            current_tag = int(best_last_tags[b].item())
            best_path = [current_tag]
            for pos in reversed(valid_positions[1:].tolist()):
                current_tag = int(history[int(pos), b, current_tag].item())
                best_path.append(current_tag)
            best_path.reverse()

            padded = [0] * seq_len
            for idx, pos in enumerate(valid_positions.tolist()):
                padded[int(pos)] = int(best_path[idx])
            paths.append(padded)
        return paths
