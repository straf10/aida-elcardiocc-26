"""Base ensemble predictors (one concern per slug) for val replay, export, and composition."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

import numpy as np

from ensemble_metaheuristic.strategies import (
    correction_predict,
    merge_preds_k_of_n,
    merge_preds_union,
    per_label_routed_predict,
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
# ``best_single_model`` / ``committee_*`` are not re-weightings of the same score sum — they give
# the auto combination grid diverse inputs (single-model champion, raw OR / majority over models).
BASE_STRATEGY_ORDER: Tuple[str, ...] = (
    "best_single_model",
    "committee_or",
    "committee_majority",
    "per_label_routing",
    "correction",
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
    names = list(ctx.names)
    champ = str(ctx.best_single_name).strip()
    if champ not in names:
        raise ValueError(f"best_single_name {champ!r} not in committee names {names}")

    def _native_member_preds(mats: List[np.ndarray], pids: List[int]) -> List[PatientPreds]:
        out_list: List[PatientPreds] = []
        for mat, is_score in zip(mats, ism):
            thr = 1.0 if is_score else 0.5
            out_list.append(
                {pid: [labels[j] for j in np.where(mat[i] >= thr)[0]] for i, pid in enumerate(pids)},
            )
        return out_list

    def _w_pred(mats: List[np.ndarray], pids: List[int], w: np.ndarray, mt: np.ndarray, gt: float) -> PatientPreds:
        return weighted_ensemble_predict(mats, ism, w, mt, float(gt), pids, labels)

    out: Dict[str, PredictFn] = {}

    def _best_single(mats: List[np.ndarray], pids: List[int]) -> PatientPreds:
        idx = names.index(champ)
        mat = mats[idx]
        thr = 1.0 if ism[idx] else 0.5
        return {pid: [labels[j] for j in np.where(mat[i] >= thr)[0]] for i, pid in enumerate(pids)}

    out["best_single_model"] = _best_single

    def _committee_or(mats: List[np.ndarray], pids: List[int]) -> PatientPreds:
        parts = _native_member_preds(mats, pids)
        acc = parts[0]
        for d in parts[1:]:
            acc = merge_preds_union(acc, d, pids)
        return acc

    out["committee_or"] = _committee_or

    def _committee_majority(mats: List[np.ndarray], pids: List[int]) -> PatientPreds:
        parts = _native_member_preds(mats, pids)
        k = max(1, len(parts) // 2 + 1)
        return merge_preds_k_of_n(parts, pids, k)

    out["committee_majority"] = _committee_majority

    out["per_label_routing"] = lambda mats, pids: per_label_routed_predict(
        mats,
        ism,
        names,
        pids,
        labels,
        ctx.label_routing,
        score_cutoff=float(ctx.best_r_cut),
    )

    _cew = dict(ctx.correction_export_kw)

    out["correction"] = lambda mats, pids: correction_predict(
        mats,
        ism,
        names,
        pids,
        labels,
        base_model=str(ctx.best_single_name),
        **_cew,
    )

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
