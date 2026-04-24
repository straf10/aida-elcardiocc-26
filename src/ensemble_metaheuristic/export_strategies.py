"""Export test + blind JSONL per ensemble strategy under ``<export_root>/<slug>/``.

**Base** strategies (one mechanism each) live under their slug; **composed** strategies are defined
in ``strategy_compositions.yaml`` (OR / AND / k-of-n over base slugs) and written without new Python code.
"""

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
from ensemble_metaheuristic.strategy_bases import BASE_STRATEGY_ORDER, build_base_strategy_functions
from ensemble_metaheuristic.strategy_compositions import load_composition_specs, try_apply_composition

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
    best_g_gate: float
    best_k: int
    label_support: Dict[str, int]
    two_threshold_t_low_factor: float = 0.72
    two_threshold_min_votes: int = 3


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

    Base slugs come from ``strategy_bases.BASE_STRATEGY_ORDER``; composed slugs from
    ``strategy_compositions.yaml``. ``manifest["strategies"]`` lists every written folder (bases + compositions).

    Returns a small manifest dict (also written to ``export_root/manifest.json``).
    """
    root = Path(ctx.export_root)
    root.mkdir(parents=True, exist_ok=True)
    written_slugs: List[str] = []
    base_slugs: List[str] = []
    composed_slugs: List[str] = []
    errors: List[str] = []

    test_pids, test_mats = _split_pids_and_matrices(ctx, "compare")
    blind_pids, blind_mats = _split_pids_and_matrices(ctx, "blind")

    registry = build_base_strategy_functions(ctx)

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

    test_base: Dict[str, Dict[int, List[str]]] = {}
    blind_base: Dict[str, Dict[int, List[str]]] = {}

    for slug in BASE_STRATEGY_ORDER:
        if slug not in registry:
            continue
        fn = registry[slug]
        tw, bw = _both(slug, fn)
        if tw is not None:
            _write_slug(root, slug, tw, bw, blind_pids)
            written_slugs.append(slug)
            base_slugs.append(slug)
            test_base[slug] = tw
            if bw is not None:
                blind_base[slug] = bw

    if test_pids is not None:
        for spec in load_composition_specs():
            t_out = try_apply_composition(spec, test_base, test_pids)
            if t_out is None:
                miss = [s for s in spec.inputs if s not in test_base]
                if miss:
                    errors.append(f"{spec.slug}: skip composition (missing test base preds: {miss})")
                continue
            b_out = None
            if blind_pids and blind_mats is not None:
                if all(s in blind_base for s in spec.inputs):
                    b_out = try_apply_composition(spec, blind_base, blind_pids)
                else:
                    errors.append(f"{spec.slug}: blind composition skipped (incomplete base blind preds)")
            _write_slug(root, spec.slug, t_out, b_out, blind_pids)
            written_slugs.append(spec.slug)
            composed_slugs.append(spec.slug)

    manifest = {
        "export_root": str(root.resolve()),
        "fusion": ctx.fusion_label,
        "models": list(ctx.names),
        "strategies": written_slugs,
        "base_strategies": base_slugs,
        "composed_strategies": composed_slugs,
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
