"""
Ensemble strategies (one concern per submodule).

``from src.ensemble_metaheuristic.strategies import ...`` re-exports the public API.

Each strategy submodule (except ``weighted_search``, which is a re-export shim) can be executed
directly, e.g. ``python -m src.ensemble_metaheuristic.strategies.correction --help``,
``...per_patient_score_routing``, or ``...per_patient_knn_train_routing``.
"""
from __future__ import annotations

from .combine import merge_preds_intersection, merge_preds_k_of_n, merge_preds_union
from .correction import correction_predict, search_correction_params
from .embedding_cluster_champion import (
    clustering_features_from_matrices,
    default_embeddings_path,
    run_cluster_sweep_from_features,
    run_embedding_cluster_sweep,
    run_embedding_kmeans_per_cluster_champion,
    run_score_matrix_cluster_sweep,
)
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

__all__ = [
    "build_cluster_champion_routing",
    "build_label_routing_table",
    "build_patient_routing_knn_train",
    "correction_predict",
    "clustering_features_from_matrices",
    "default_embeddings_path",
    "merge_preds_intersection",
    "merge_preds_k_of_n",
    "merge_preds_union",
    "per_cluster_champion_predict",
    "per_label_f1",
    "per_label_routed_predict",
    "per_patient_champion_from_scores",
    "per_patient_routed_predict",
    "run_cluster_sweep_from_features",
    "run_embedding_cluster_sweep",
    "run_embedding_kmeans_per_cluster_champion",
    "run_score_matrix_cluster_sweep",
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
