"""Shim: use ``ensemble_metaheuristic.clustering.text`` (kept for ``python -m`` / old imports)."""
from __future__ import annotations

from ..clustering.text import run_text_embedding_cluster_sweep_train_routing

__all__ = ["run_text_embedding_cluster_sweep_train_routing"]
