"""
Ensemble strategies (one concern per submodule).

``from src.ensemble_metaheuristic.strategies import ...`` re-exports the public API.
"""
from __future__ import annotations

from .combine import merge_preds_intersection, merge_preds_k_of_n, merge_preds_union
from .correction import correction_predict, search_correction_params
from .embedding_cluster_champion import (
    default_embeddings_path,
    run_embedding_cluster_sweep,
    run_embedding_kmeans_per_cluster_champion,
)
from .per_cluster import build_cluster_champion_routing, per_cluster_champion_predict
from .per_label_routing import (
    build_label_routing_table,
    per_label_f1,
    per_label_routed_predict,
)
from .weighted_search import (
    run_search,
    score_ensemble,
    weighted_ensemble_combined_matrix,
    weighted_ensemble_predict,
    weighted_ensemble_predict_frequency_buckets,
    weighted_ensemble_predict_gated_secondary,
    weighted_ensemble_predict_top_k,
    weighted_ensemble_predict_two_threshold,
)

__all__ = [
    "build_cluster_champion_routing",
    "build_label_routing_table",
    "correction_predict",
    "default_embeddings_path",
    "merge_preds_intersection",
    "merge_preds_k_of_n",
    "merge_preds_union",
    "per_cluster_champion_predict",
    "per_label_f1",
    "per_label_routed_predict",
    "run_embedding_cluster_sweep",
    "run_embedding_kmeans_per_cluster_champion",
    "run_search",
    "score_ensemble",
    "search_correction_params",
    "weighted_ensemble_combined_matrix",
    "weighted_ensemble_predict",
    "weighted_ensemble_predict_frequency_buckets",
    "weighted_ensemble_predict_gated_secondary",
    "weighted_ensemble_predict_top_k",
    "weighted_ensemble_predict_two_threshold",
]
