"""Base ensemble predictors (one concern per slug) for val replay, export, and composition."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

import numpy as np

from ensemble_metaheuristic.strategies import (
    merge_preds_k_of_n,
    weighted_ensemble_combined_matrix,
    weighted_ensemble_predict,
    weighted_ensemble_predict_frequency_buckets,
    weighted_ensemble_predict_top_k,
)

if TYPE_CHECKING:
    from ensemble_metaheuristic.export_strategies import StrategyExportContext

PatientPreds = Dict[int, List[str]]
PredictFn = Callable[[List[np.ndarray], List[int]], PatientPreds]

# Stable table / export order (only keys that exist in the registry are used).
BASE_STRATEGY_ORDER: Tuple[str, ...] = (
    "weighted",
    "weighted_majority_restarts",
    "weighted_top_k",
    "weighted_freq_buckets",
    "weighted_global_loose",
    "weighted_global_tight",
    "weighted_top_k_loose",
    "weighted_freq_loose",
)


def build_base_strategy_functions(ctx: "StrategyExportContext") -> Dict[str, PredictFn]:
    """Return slug -> (matrices, pids) -> preds for test/blind replay and val metrics."""
    from ensemble_metaheuristic.export_strategies import StrategyExportContext as _Ctx

    if not isinstance(ctx, _Ctx):
        raise TypeError("ctx must be a StrategyExportContext")

    ism = list(ctx.is_score_model)
    labels = list(ctx.all_labels)

    def _w_pred(mats: List[np.ndarray], pids: List[int], w: np.ndarray, mt: np.ndarray, gt: float) -> PatientPreds:
        return weighted_ensemble_predict(mats, ism, w, mt, float(gt), pids, labels)

    out: Dict[str, PredictFn] = {}

    out["weighted"] = lambda mats, pids: _w_pred(mats, pids, ctx.best_w, ctx.best_mt, float(ctx.best_gt))

    rt = ctx.restart_triples
    if rt is not None and len(rt) >= 2:

        def _maj(mats: List[np.ndarray], pids: List[int]) -> PatientPreds:
            preds_list = [_w_pred(mats, pids, w, mt, gt) for w, mt, gt in rt]
            k = len(preds_list) // 2 + 1
            return merge_preds_k_of_n(preds_list, pids, k)

        out["weighted_majority_restarts"] = _maj

    def _topk(mats: List[np.ndarray], pids: List[int], gt: float) -> PatientPreds:
        comb = weighted_ensemble_combined_matrix(mats, ism, ctx.best_w, ctx.best_mt)
        return weighted_ensemble_predict_top_k(comb, float(gt), pids, labels, int(ctx.best_k))

    out["weighted_top_k"] = lambda mats, pids: _topk(mats, pids, float(ctx.best_gt))

    out["weighted_freq_buckets"] = lambda mats, pids: weighted_ensemble_predict_frequency_buckets(
        mats,
        ism,
        ctx.best_w,
        ctx.best_mt,
        float(ctx.best_gt),
        pids,
        labels,
        ctx.label_support,
        support_cutoff=25,
        rare_factor=1.08,
        freq_factor=0.97,
    )

    out["weighted_global_loose"] = lambda mats, pids: _w_pred(
        mats, pids, ctx.best_w, ctx.best_mt, float(ctx.weighted_aux_gt_loose),
    )
    out["weighted_global_tight"] = lambda mats, pids: _w_pred(
        mats, pids, ctx.best_w, ctx.best_mt, float(ctx.weighted_aux_gt_tight),
    )

    out["weighted_top_k_loose"] = lambda mats, pids: _topk(mats, pids, float(ctx.weighted_aux_gt_loose))

    out["weighted_freq_loose"] = lambda mats, pids: weighted_ensemble_predict_frequency_buckets(
        mats,
        ism,
        ctx.best_w,
        ctx.best_mt,
        float(ctx.weighted_aux_gt_loose),
        pids,
        labels,
        ctx.label_support,
        support_cutoff=25,
        rare_factor=1.08,
        freq_factor=0.97,
    )

    return out
