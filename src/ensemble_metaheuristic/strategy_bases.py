"""Base ensemble predictors (one concern per slug) for val replay, export, and composition."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np

from ensemble_metaheuristic.strategies import (
    correction_predict,
    merge_preds_k_of_n,
    per_label_champion_plus_other_vote_predict,
    per_label_routed_predict,
    per_patient_champion_from_scores,
    per_patient_routed_predict,
    weighted_ensemble_combined_matrix,
    weighted_ensemble_predict,
    weighted_ensemble_predict_frequency_buckets,
    weighted_ensemble_predict_gated_secondary,
    weighted_ensemble_predict_top_k,
    weighted_ensemble_predict_two_threshold,
)

if TYPE_CHECKING:
    from ensemble_metaheuristic.export_strategies import StrategyExportContext

PatientPreds = Dict[int, List[str]]
PredictFn = Callable[[List[np.ndarray], List[int]], PatientPreds]

# Match ``__main__.py`` per-patient sweep default.
PATIENT_SCORE_ROUTING_POLICY = "mean"

# Stable table / export order (only keys that exist in the registry are used).
BASE_STRATEGY_ORDER: Tuple[str, ...] = (
    "weighted",
    "weighted_majority_restarts",
    "weighted_gated_ir_ner",
    "weighted_top_k",
    "weighted_freq_buckets",
    "weighted_two_threshold",
    "per_label_routing",
    "per_label_champion_plus_vote",
    "per_patient_score_routing",
    "correction",
)


def build_base_strategy_functions(ctx: "StrategyExportContext") -> Dict[str, PredictFn]:
    """Return slug -> (matrices, pids) -> preds for test/blind replay and val metrics."""
    from ensemble_metaheuristic.export_strategies import StrategyExportContext as _Ctx

    if not isinstance(ctx, _Ctx):
        raise TypeError("ctx must be a StrategyExportContext")

    ism = list(ctx.is_score_model)
    names = list(ctx.names)
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

    out["weighted_gated_ir_ner"] = lambda mats, pids: weighted_ensemble_predict_gated_secondary(
        mats,
        ism,
        names,
        ctx.best_w,
        ctx.best_mt,
        float(ctx.best_gt),
        pids,
        labels,
        gate_max_base=float(ctx.best_g_gate),
    )

    def _topk(mats: List[np.ndarray], pids: List[int]) -> PatientPreds:
        comb = weighted_ensemble_combined_matrix(mats, ism, ctx.best_w, ctx.best_mt)
        return weighted_ensemble_predict_top_k(comb, float(ctx.best_gt), pids, labels, int(ctx.best_k))

    out["weighted_top_k"] = _topk

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

    out["weighted_two_threshold"] = lambda mats, pids: weighted_ensemble_predict_two_threshold(
        mats,
        ism,
        ctx.best_w,
        ctx.best_mt,
        pids,
        labels,
        t_high=float(ctx.best_gt),
        t_low=float(ctx.best_gt) * float(ctx.two_threshold_t_low_factor),
        min_votes=int(ctx.two_threshold_min_votes),
    )

    out["per_label_routing"] = lambda mats, pids: per_label_routed_predict(
        mats,
        ism,
        names,
        pids,
        labels,
        ctx.label_routing,
        score_cutoff=float(ctx.best_r_cut),
    )

    out["per_label_champion_plus_vote"] = lambda mats, pids: per_label_champion_plus_other_vote_predict(
        mats,
        ism,
        names,
        pids,
        labels,
        ctx.label_routing,
        score_cutoff=float(ctx.best_pv_cut),
        min_other_votes=int(ctx.best_pv_min_o),
    )

    _corr_kw = ("add_min_votes", "add_min_score_factor", "remove_if_zero_votes")
    cfg_corr = {k: ctx.best_cfg[k] for k in _corr_kw if k in ctx.best_cfg}

    out["correction"] = lambda mats, pids: correction_predict(
        mats,
        ism,
        names,
        pids,
        labels,
        base_model=str(ctx.best_single_name),
        **cfg_corr,
    )

    def _pp(mats: List[np.ndarray], pids: List[int]) -> PatientPreds:
        pr = per_patient_champion_from_scores(mats, names, pids, policy=PATIENT_SCORE_ROUTING_POLICY)
        return per_patient_routed_predict(
            mats,
            ism,
            names,
            pids,
            labels,
            pr,
            score_cutoff=float(ctx.best_pp_s_cut),
        )

    out["per_patient_score_routing"] = _pp

    return out
