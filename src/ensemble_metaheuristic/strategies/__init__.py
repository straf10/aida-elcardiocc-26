"""
Ensemble strategies (one concern per submodule).

``from src.ensemble_metaheuristic.strategies import ...`` re-exports the public API.
"""
from __future__ import annotations

from .combine import merge_preds_intersection, merge_preds_k_of_n, merge_preds_union
from .correction import correction_predict, search_correction_params
from .per_cluster import build_cluster_champion_routing, per_cluster_champion_predict
from .per_label_routing import (
    build_label_routing_table,
    per_label_f1,
    per_label_routed_predict,
)

__all__ = [
    "build_cluster_champion_routing",
    "build_label_routing_table",
    "correction_predict",
    "merge_preds_intersection",
    "merge_preds_k_of_n",
    "merge_preds_union",
    "per_cluster_champion_predict",
    "per_label_f1",
    "per_label_routed_predict",
    "search_correction_params",
]
