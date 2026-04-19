"""Backward compatibility: re-exports classic ``weighted_strategy`` and ``weighted_vns_strategy``."""
from __future__ import annotations

from .weighted_strategy import (
    hill_climb,
    random_search,
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
    "hill_climb",
    "random_search",
    "run_search",
    "run_vns_search",
    "score_ensemble",
    "weighted_ensemble_combined_matrix",
    "weighted_ensemble_predict",
    "weighted_ensemble_predict_frequency_buckets",
    "weighted_ensemble_predict_gated_secondary",
    "weighted_ensemble_predict_top_k",
    "weighted_ensemble_predict_two_threshold",
]


if __name__ == "__main__":
    raise SystemExit(
        "weighted_search.py only re-exports APIs. Run a specific module instead:\n"
        "  python -m src.ensemble_metaheuristic.strategies.weighted_strategy\n"
        "  python -m src.ensemble_metaheuristic.strategies.weighted_vns_strategy",
    )
