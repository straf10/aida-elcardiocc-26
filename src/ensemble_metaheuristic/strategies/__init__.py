"""
Ensemble strategies (one concern per submodule).

``from src.ensemble_metaheuristic.strategies import ...`` re-exports the public API.

Each strategy submodule (except ``weighted_search`` and ``combine``, which are re-export shims)
can be executed directly, e.g. ``python -m src.ensemble_metaheuristic.strategies.correction --help``,
``...per_patient_score_routing``, or ``...per_patient_knn_train_routing``.
Label-set fusion primitives live in ``pred_merge_union``, ``pred_merge_intersection``, ``pred_merge_k_of_n``
(re-exported from ``combine``). Clustering lives under ``ensemble_metaheuristic.clustering``;
``embedding_cluster_champion`` / ``text_cluster_champion`` are thin shims for ``python -m`` compatibility.
"""
from __future__ import annotations

from typing import Any

from .per_label_champion_plus_vote import per_label_champion_plus_other_vote_predict
from .pred_merge_intersection import merge_preds_intersection
from .pred_merge_k_of_n import merge_preds_k_of_n
from .pred_merge_union import merge_preds_union
from .correction import correction_predict, search_correction_params
from .per_cluster import build_cluster_champion_routing, per_cluster_champion_predict
from .per_label_routing import (
    build_label_routing_table,
    per_label_f1,
    per_label_routed_predict,
)
from .per_patient_knn_train_routing import build_patient_routing_knn_train
from .per_patient_score_routing import per_patient_champion_from_scores, per_patient_routed_predict
from .weighted_strategy import (
    run_search,
    score_ensemble,
    weighted_ensemble_combined_matrix,
    weighted_ensemble_predict,
    weighted_ensemble_predict_frequency_buckets,
    weighted_ensemble_predict_gated_secondary,
    weighted_ensemble_predict_top_k,
    weighted_ensemble_predict_two_threshold,
)
from .weighted_vns_strategy import run_vns_search

_CLUSTERING_FROM_SCORE_MATRIX = frozenset(
    {
        "clustering_features_from_matrices",
        "run_cluster_sweep_from_features",
        "run_score_matrix_cluster_sweep",
        "run_score_matrix_cluster_sweep_train_routing",
        "run_train_routing_champion_sweep_from_features",
    },
)


def __getattr__(name: str) -> Any:
    """Lazy-load clustering helpers so ``python -m src.ensemble_metaheuristic.clustering.score_matrix`` avoids a runpy warning."""
    if name in _CLUSTERING_FROM_SCORE_MATRIX:
        from ..clustering import score_matrix as _csm

        return getattr(_csm, name)
    if name == "run_text_embedding_cluster_sweep_train_routing":
        from ..clustering import text as _ctxt

        return _ctxt.run_text_embedding_cluster_sweep_train_routing
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_cluster_champion_routing",
    "build_label_routing_table",
    "build_patient_routing_knn_train",
    "correction_predict",
    "clustering_features_from_matrices",
    "merge_preds_intersection",
    "merge_preds_k_of_n",
    "merge_preds_union",
    "per_cluster_champion_predict",
    "per_label_champion_plus_other_vote_predict",
    "per_label_f1",
    "per_label_routed_predict",
    "per_patient_champion_from_scores",
    "per_patient_routed_predict",
    "run_cluster_sweep_from_features",
    "run_score_matrix_cluster_sweep",
    "run_score_matrix_cluster_sweep_train_routing",
    "run_text_embedding_cluster_sweep_train_routing",
    "run_train_routing_champion_sweep_from_features",
    "run_search",
    "run_vns_search",
    "score_ensemble",
    "search_correction_params",
    "weighted_ensemble_combined_matrix",
    "weighted_ensemble_predict",
    "weighted_ensemble_predict_frequency_buckets",
    "weighted_ensemble_predict_gated_secondary",
    "weighted_ensemble_predict_top_k",
    "weighted_ensemble_predict_two_threshold",
]
