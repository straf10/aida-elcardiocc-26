"""Shim: use ``ensemble_metaheuristic.clustering.score_matrix`` (kept for ``python -m`` / old imports)."""
from __future__ import annotations

from ..clustering.score_matrix import (
    clustering_features_from_matrices,
    run_cluster_sweep_from_features,
    run_score_matrix_cluster_sweep,
    run_score_matrix_cluster_sweep_train_routing,
    run_train_routing_champion_sweep_from_features,
)

__all__ = [
    "clustering_features_from_matrices",
    "run_cluster_sweep_from_features",
    "run_score_matrix_cluster_sweep",
    "run_score_matrix_cluster_sweep_train_routing",
    "run_train_routing_champion_sweep_from_features",
]


if __name__ == "__main__":
    from ..clustering.score_matrix import _run_standalone_cli

    _run_standalone_cli()
