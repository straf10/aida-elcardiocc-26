"""Committee score grid → multi-label logits (not stacking).

Each document is one example with tensor shape ``(n_models, n_labels)`` built only from
base-model scores for **that** document.  No patient-id embedding and no lookup in a
training patient set—new blind patients use the same forward pass.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


def resolve_torch_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


class CommitteeFlattenMLP(nn.Module):
    """Flatten ``(B, n_models, n_labels)`` → MLP → ``(B, n_labels)`` logits."""

    def __init__(
        self,
        n_models: int,
        n_labels: int,
        hidden_dims: Tuple[int, ...],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        flat = int(n_models * n_labels)
        layers: List[nn.Module] = []
        prev = flat
        for h in hidden_dims:
            layers += [nn.Linear(prev, int(h)), nn.ReLU(), nn.Dropout(float(dropout))]
            prev = int(h)
        layers.append(nn.Linear(prev, int(n_labels)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return self.net(x.reshape(b, -1))


class CommitteeConvGridMLP(nn.Module):
    """1D conv over the **label** axis mixes codes; input channels = base models."""

    def __init__(
        self,
        n_models: int,
        n_labels: int,
        conv_channels: int = 64,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        c = int(conv_channels)
        self.body = nn.Sequential(
            nn.Conv1d(int(n_models), c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(c, c, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * int(n_labels), int(mlp_hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mlp_hidden), int(n_labels)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def stack_model_grid(matrices: List[np.ndarray]) -> np.ndarray:
    """``List[(n_docs, n_labels)]`` → ``(n_docs, n_models, n_labels)`` float32."""
    g = np.stack([m.astype(np.float32, copy=False) for m in matrices], axis=0)
    return np.transpose(g, (1, 0, 2))
