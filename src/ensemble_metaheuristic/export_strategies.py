"""Export test + blind JSONL per ensemble strategy under ``<export_root>/<slug>/``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluation.config_utils import get_cfg, load_config
from evaluation.io_utils import load_ground_truth, save_predictions_jsonl
from evaluation.model_artifacts import load_model_artifacts
from ensemble_metaheuristic.matrices import build_score_matrix, load_thresholds_for_model
from ensemble_metaheuristic.strategies import (
    build_patient_routing_knn_train,
    correction_predict,
    merge_preds_intersection,
    merge_preds_k_of_n,
    merge_preds_union,
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

PATIENT_SCORE_ROUTING_POLICY = "mean"
PATIENT_KNN_K = 11


def matrices_for_split(
    model_cfgs: Dict[str, Any],
    model_names: Sequence[str],
    all_pids: List[int],
    all_labels: List[str],
    predictions_split: str,
    *,
    load_scores: bool,
) -> List[np.ndarray]:
    matrices: List[np.ndarray] = []
    for name in model_names:
        arts = load_model_artifacts(
            model_cfgs[name],
            all_pids,
            predictions_split=predictions_split,
            load_scores=load_scores,
        )
        thr = load_thresholds_for_model(model_cfgs[name], all_labels) if arts.scores is not None else None
        matrices.append(build_score_matrix(arts, all_pids, all_labels, thr))
    return matrices


@dataclass
class StrategyExportContext:
    """Val-tuned parameters + model list; used to replay strategies on test/blind."""

    config_path: str
    model_cfgs: Dict[str, Any]
    names: List[str]
    is_score_model: List[bool]
    all_labels: List[str]
    export_root: Path
    best_w: np.ndarray
    best_mt: np.ndarray
    best_gt: float
    fusion_label: str
    restart_triples: Optional[List[Tuple[np.ndarray, np.ndarray, float]]]
    label_routing: Dict[str, str]
    best_r_cut: float
    best_pv_cut: float
    best_pv_min_o: int
    best_cfg: Dict[str, Any]
    best_single_name: str
    best_pp_s_cut: float
    best_pp_k_cut: Optional[float]
    train_bundle: Optional[Tuple[Any, ...]]
    best_g_gate: float
    best_k: int
    label_support: Dict[str, int]


def _split_pids_and_matrices(
    ctx: StrategyExportContext,
    predictions_split: str,
) -> Tuple[Optional[List[int]], Optional[List[np.ndarray]]]:
    cfg = load_config(ctx.config_path)
    path_key = {"compare": "data.test_path", "blind": "data.blind_path"}[predictions_split]
    jsonl = str(get_cfg(cfg, path_key, ""))
    if not jsonl or not Path(jsonl).is_file():
        return None, None
    pids = list(load_ground_truth(jsonl).keys())
    if not pids:
        return None, None
    try:
        mats = matrices_for_split(
            ctx.model_cfgs, ctx.names, pids, ctx.all_labels, predictions_split, load_scores=False,
        )
    except FileNotFoundError:
        return None, None
    return pids, mats


def _write_slug(
    root: Path,
    slug: str,
    test_pred: Dict[int, List[str]],
    blind_pred: Optional[Dict[int, List[str]]],
    blind_pids: Optional[List[int]],
) -> None:
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    save_predictions_jsonl(test_pred, d / "test_predictions.jsonl")
    if blind_pids:
        bp = blind_pred if blind_pred is not None else {int(pid): [] for pid in blind_pids}
        save_predictions_jsonl(bp, d / "blind_predictions.jsonl")


def export_all_strategy_subfolders(ctx: StrategyExportContext) -> Dict[str, Any]:
    """
    For each strategy, write ``<export_root>/<slug>/test_predictions.jsonl`` and
    ``blind_predictions.jsonl`` when blind patients exist.

    Returns a small manifest dict (also written to ``export_root/manifest.json``).
    """
    root = Path(ctx.export_root)
    root.mkdir(parents=True, exist_ok=True)
    ism = list(ctx.is_score_model)
    written_slugs: List[str] = []
    errors: List[str] = []

    test_pids, test_mats = _split_pids_and_matrices(ctx, "compare")
    blind_pids, blind_mats = _split_pids_and_matrices(ctx, "blind")

    def _both(slug: str, fn: Callable[[List[np.ndarray], List[int]], Dict[int, List[str]]]) -> Tuple[
        Optional[Dict[int, List[str]]],
        Optional[Dict[int, List[str]]],
    ]:
        if test_mats is None or test_pids is None:
            return None, None
        try:
            t = fn(test_mats, test_pids)
        except Exception as exc:
            errors.append(f"{slug} test: {exc!r}")
            t = None
        b = None
        if blind_mats is not None and blind_pids is not None:
            try:
                b = fn(blind_mats, blind_pids)
            except Exception as exc:
                errors.append(f"{slug} blind: {exc!r}")
        return t, b

    def _w_pred(mats: List[np.ndarray], pids: List[int], w: np.ndarray, mt: np.ndarray, gt: float) -> Dict[int, List[str]]:
        return weighted_ensemble_predict(mats, ism, w, mt, float(gt), pids, ctx.all_labels)

    # --- weighted (chosen classic/vns) ---
    def _weighted(mats: List[np.ndarray], pids: List[int]) -> Dict[int, List[str]]:
        return _w_pred(mats, pids, ctx.best_w, ctx.best_mt, float(ctx.best_gt))

    tw, bw = _both("weighted", _weighted)
    if tw is not None:
        _write_slug(root, "weighted", tw, bw, blind_pids)
        written_slugs.append("weighted")

    # --- weighted gated (IR/NER secondary) ---
    def _gated(mats: List[np.ndarray], pids: List[int]) -> Dict[int, List[str]]:
        return weighted_ensemble_predict_gated_secondary(
            mats,
            ism,
            ctx.names,
            ctx.best_w,
            ctx.best_mt,
            float(ctx.best_gt),
            pids,
            ctx.all_labels,
            gate_max_base=float(ctx.best_g_gate),
        )

    tg, bg = _both("weighted_gated_ir_ner", _gated)
    if tg is not None:
        _write_slug(root, "weighted_gated_ir_ner", tg, bg, blind_pids)
        written_slugs.append("weighted_gated_ir_ner")

    # --- majority over weighted-search restarts ---
    if ctx.restart_triples and len(ctx.restart_triples) >= 2:

        def _maj(mats: List[np.ndarray], pids: List[int]) -> Dict[int, List[str]]:
            preds_list = [_w_pred(mats, pids, w, mt, gt) for w, mt, gt in ctx.restart_triples]
            k = len(preds_list) // 2 + 1
            return merge_preds_k_of_n(preds_list, pids, k)

        tm, bm = _both("weighted_majority_restarts", _maj)
        if tm is not None:
            _write_slug(root, "weighted_majority_restarts", tm, bm, blind_pids)
            written_slugs.append("weighted_majority_restarts")

    # --- top-k / freq / two-threshold (need combined matrix per split) ---
    def _topk(mats: List[np.ndarray], pids: List[int]) -> Dict[int, List[str]]:
        comb = weighted_ensemble_combined_matrix(mats, ism, ctx.best_w, ctx.best_mt)
        return weighted_ensemble_predict_top_k(comb, float(ctx.best_gt), pids, ctx.all_labels, int(ctx.best_k))

    tk, bk = _both("weighted_top_k", _topk)
    if tk is not None:
        _write_slug(root, "weighted_top_k", tk, bk, blind_pids)
        written_slugs.append("weighted_top_k")

    def _freq(mats: List[np.ndarray], pids: List[int]) -> Dict[int, List[str]]:
        return weighted_ensemble_predict_frequency_buckets(
            mats,
            ism,
            ctx.best_w,
            ctx.best_mt,
            float(ctx.best_gt),
            pids,
            ctx.all_labels,
            ctx.label_support,
            support_cutoff=25,
            rare_factor=1.08,
            freq_factor=0.97,
        )

    tf, bf = _both("weighted_freq_buckets", _freq)
    if tf is not None:
        _write_slug(root, "weighted_freq_buckets", tf, bf, blind_pids)
        written_slugs.append("weighted_freq_buckets")

    def _two(mats: List[np.ndarray], pids: List[int]) -> Dict[int, List[str]]:
        return weighted_ensemble_predict_two_threshold(
            mats,
            ism,
            ctx.best_w,
            ctx.best_mt,
            pids,
            ctx.all_labels,
            t_high=float(ctx.best_gt),
            t_low=float(ctx.best_gt) * 0.72,
            min_votes=3,
        )

    t2, b2 = _both("weighted_two_threshold", _two)
    if t2 is not None:
        _write_slug(root, "weighted_two_threshold", t2, b2, blind_pids)
        written_slugs.append("weighted_two_threshold")

    # --- per-label routing ---
    def _pl(mats: List[np.ndarray], pids: List[int]) -> Dict[int, List[str]]:
        return per_label_routed_predict(
            mats, ism, ctx.names, pids, ctx.all_labels, ctx.label_routing, score_cutoff=float(ctx.best_r_cut),
        )

    tpl, bpl = _both("per_label_routing", _pl)
    if tpl is not None:
        _write_slug(root, "per_label_routing", tpl, bpl, blind_pids)
        written_slugs.append("per_label_routing")

    def _plpv(mats: List[np.ndarray], pids: List[int]) -> Dict[int, List[str]]:
        return per_label_champion_plus_other_vote_predict(
            mats,
            ism,
            ctx.names,
            pids,
            ctx.all_labels,
            ctx.label_routing,
            score_cutoff=float(ctx.best_pv_cut),
            min_other_votes=int(ctx.best_pv_min_o),
        )

    tpv, bpv = _both("per_label_plus_vote", _plpv)
    if tpv is not None:
        _write_slug(root, "per_label_plus_vote", tpv, bpv, blind_pids)
        written_slugs.append("per_label_plus_vote")

    # --- correction ---
    _corr_kw = ("add_min_votes", "add_min_score_factor", "remove_if_zero_votes")
    cfg_corr = {k: ctx.best_cfg[k] for k in _corr_kw if k in ctx.best_cfg}

    def _corr(mats: List[np.ndarray], pids: List[int]) -> Dict[int, List[str]]:
        return correction_predict(
            mats,
            ism,
            ctx.names,
            pids,
            ctx.all_labels,
            base_model=ctx.best_single_name,
            **cfg_corr,
        )

    tc, bc = _both("correction", _corr)
    if tc is not None:
        _write_slug(root, "correction", tc, bc, blind_pids)
        written_slugs.append("correction")

    # --- per-patient score ---
    def _pps(mats: List[np.ndarray], pids: List[int]) -> Dict[int, List[str]]:
        pr = per_patient_champion_from_scores(mats, ctx.names, pids, policy=PATIENT_SCORE_ROUTING_POLICY)
        return per_patient_routed_predict(
            mats, ism, ctx.names, pids, ctx.all_labels, pr, score_cutoff=float(ctx.best_pp_s_cut),
        )

    tps, bps = _both("per_patient_score", _pps)
    if tps is not None:
        _write_slug(root, "per_patient_score", tps, bps, blind_pids)
        written_slugs.append("per_patient_score")

    # --- per-patient kNN train ---
    if ctx.train_bundle is not None and ctx.best_pp_k_cut is not None:
        tr_gt, tr_pids, tr_mats, _tr_path, per_train_preds = ctx.train_bundle

        def _ppk(mats: List[np.ndarray], pids: List[int]) -> Dict[int, List[str]]:
            pr = build_patient_routing_knn_train(
                tr_mats,
                mats,
                tr_gt,
                tr_pids,
                pids,
                ctx.names,
                ctx.all_labels,
                per_train_preds,
                k=PATIENT_KNN_K,
            )
            if not pr:
                return {pid: [] for pid in pids}
            return per_patient_routed_predict(
                mats, ism, ctx.names, pids, ctx.all_labels, pr, score_cutoff=float(ctx.best_pp_k_cut),
            )

        tpk, bpk = _both("per_patient_knn_train", _ppk)
        if tpk is not None:
            _write_slug(root, "per_patient_knn_train", tpk, bpk, blind_pids)
            written_slugs.append("per_patient_knn_train")

    # --- label-set merges (need component preds per split) ---
    def _merge_exports() -> None:
        if test_mats is None or test_pids is None:
            return

        def _comp(mats: List[np.ndarray], pids: List[int]) -> Tuple[
            Dict[int, List[str]],
            Dict[int, List[str]],
            Dict[int, List[str]],
            Dict[int, List[str]],
        ]:
            w = _weighted(mats, pids)
            pl = _pl(mats, pids)
            pv = _plpv(mats, pids)
            co = _corr(mats, pids)
            return w, pl, pv, co

        try:
            tw, tr, tpv, tco = _comp(test_mats, test_pids)
        except Exception as exc:
            errors.append(f"merge components test: {exc!r}")
            return

        blind_ok = blind_mats is not None and blind_pids is not None
        bw = br = bpv = bco = None
        if blind_ok:
            try:
                bw, br, bpv, bco = _comp(blind_mats, blind_pids)
            except Exception as exc:
                errors.append(f"merge components blind: {exc!r}")

        def _mb(
            fn: Callable[[Dict[int, List[str]], Dict[int, List[str]], List[int]], Dict[int, List[str]]],
            x: Dict[int, List[str]],
            y: Dict[int, List[str]],
        ) -> Optional[Dict[int, List[str]]]:
            if not blind_ok or blind_pids is None:
                return None
            return fn(x, y, blind_pids)

        combos: List[Tuple[str, Dict[int, List[str]], Optional[Dict[int, List[str]]]]] = [
            ("merge_or_per_label_weighted", merge_preds_union(tr, tw, test_pids), _mb(merge_preds_union, br, bw)),
            ("merge_and_per_label_weighted", merge_preds_intersection(tr, tw, test_pids), _mb(merge_preds_intersection, br, bw)),
            ("merge_or_per_label_vote_weighted", merge_preds_union(tpv, tw, test_pids), _mb(merge_preds_union, bpv, bw)),
            ("merge_and_per_label_vote_weighted", merge_preds_intersection(tpv, tw, test_pids), _mb(merge_preds_intersection, bpv, bw)),
            ("merge_or_per_label_correction", merge_preds_union(tr, tco, test_pids), _mb(merge_preds_union, br, bco)),
            ("merge_and_per_label_correction", merge_preds_intersection(tr, tco, test_pids), _mb(merge_preds_intersection, br, bco)),
            ("merge_or_weighted_correction", merge_preds_union(tw, tco, test_pids), _mb(merge_preds_union, bw, bco)),
            ("merge_and_weighted_correction", merge_preds_intersection(tw, tco, test_pids), _mb(merge_preds_intersection, bw, bco)),
            (
                "merge_k2_weighted_per_label_correction",
                merge_preds_k_of_n([tw, tr, tco], test_pids, 2),
                merge_preds_k_of_n([bw, br, bco], blind_pids, 2)
                if blind_ok and bw is not None and br is not None and bco is not None
                else None,
            ),
            (
                "merge_k3_weighted_per_label_correction",
                merge_preds_k_of_n([tw, tr, tco], test_pids, 3),
                merge_preds_k_of_n([bw, br, bco], blind_pids, 3)
                if blind_ok and bw is not None and br is not None and bco is not None
                else None,
            ),
        ]
        for slug, t_out, b_out in combos:
            if t_out is not None:
                _write_slug(root, slug, t_out, b_out, blind_pids)
                written_slugs.append(slug)

    _merge_exports()

    manifest = {
        "export_root": str(root.resolve()),
        "fusion": ctx.fusion_label,
        "models": list(ctx.names),
        "strategies": written_slugs,
        "errors": errors,
    }
    man_path = root / "manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if errors:
        print("[ensemble export] Some strategies skipped:", flush=True)
        for e in errors:
            print(f"  {e}", flush=True)

    return manifest
