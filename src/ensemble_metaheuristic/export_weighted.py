"""Write weighted-ensemble JSONL predictions for val / test / blind (same weights as ``__main__``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from evaluation.config_utils import get_cfg, load_config
from evaluation.io_utils import load_ground_truth, save_predictions_jsonl
from evaluation.model_artifacts import load_model_artifacts
from ensemble_metaheuristic.matrices import build_score_matrix, load_thresholds_for_model
from ensemble_metaheuristic.strategies.weighted_strategy import weighted_ensemble_predict


def _matrices_for_split(
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


def export_weighted_ensemble_jsonls(
    *,
    config_path: str,
    model_cfgs: Dict[str, Any],
    model_names: Sequence[str],
    is_score_model: Sequence[bool],
    best_w: np.ndarray,
    best_mt: np.ndarray,
    best_gt: float,
    all_labels: List[str],
    out_dir: str | Path,
    fusion_label: str,
) -> Dict[str, str | None]:
    """
    Apply the same weighted fusion as validation tuning to val / test / blind sidecars.

    Test and blind matrices use JSONL predictions only (``load_scores=False``) because val score
    bundles are not aligned to those patient_ids.

    Returns paths written (or None if skipped).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "ensemble_weighted_meta.json"
    cfg = load_config(config_path)
    written: Dict[str, str | None] = {}

    def _write(split: str, pids: List[int], matrices: List[np.ndarray]) -> str | None:
        preds = weighted_ensemble_predict(
            matrices, list(is_score_model), best_w, best_mt, float(best_gt), pids, all_labels,
        )
        dest = out / f"{split}_predictions.jsonl"
        save_predictions_jsonl(preds, dest)
        return str(dest.resolve())

    # Val — match in-memory ensemble (scores allowed)
    val_path = str(get_cfg(cfg, "data.val_path"))
    val_gt = load_ground_truth(val_path)
    val_pids = list(val_gt.keys())
    m_val = _matrices_for_split(model_cfgs, model_names, val_pids, all_labels, "val", load_scores=True)
    written["val"] = _write("val", val_pids, m_val)

    # Test
    test_path = str(get_cfg(cfg, "data.test_path", ""))
    if test_path and Path(test_path).is_file():
        test_gt = load_ground_truth(test_path)
        test_pids = list(test_gt.keys())
        try:
            m_test = _matrices_for_split(
                model_cfgs, model_names, test_pids, all_labels, "compare", load_scores=False,
            )
            written["test"] = _write("test", test_pids, m_test)
        except FileNotFoundError as exc:
            print(f"[ensemble export] WARNING: skip test — {exc}", flush=True)
            written["test"] = None
    else:
        written["test"] = None

    # Blind (no gold required)
    blind_path = str(get_cfg(cfg, "data.blind_path", ""))
    if blind_path and Path(blind_path).is_file():
        blind_gt = load_ground_truth(blind_path)
        blind_pids = list(blind_gt.keys())
        if not blind_pids:
            print("[ensemble export] WARNING: skip blind — no patient_id rows in blind JSONL.", flush=True)
            written["blind"] = None
        else:
            try:
                m_blind = _matrices_for_split(
                    model_cfgs, model_names, blind_pids, all_labels, "blind", load_scores=False,
                )
                written["blind"] = _write("blind", blind_pids, m_blind)
            except FileNotFoundError as exc:
                print(f"[ensemble export] WARNING: skip blind — {exc}", flush=True)
                written["blind"] = None
    else:
        written["blind"] = None

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fusion": fusion_label,
                "models": list(model_names),
                "weights": [float(x) for x in best_w],
                "model_thresholds": [float(x) for x in best_mt],
                "global_threshold": float(best_gt),
                "outputs": written,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    written["meta"] = str(meta_path.resolve())
    return written
