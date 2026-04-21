"""Load per-model **predictions JSONL**; optionally companion score tensors for ensemble / Recall@K."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from preprocessing.io_utils import load_labelset

from .io_utils import load_predictions


def _load_optional_scores_bundle(
    model_cfg: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[str]]]:
    """If ``scores_path`` (+ companions) exist, load them; else (None, None, None)."""
    scores_path = model_cfg.get("scores_path")
    pids_path = model_cfg.get("pids_path")
    label_path = model_cfg.get("label_names_path")
    if not scores_path or not Path(scores_path).exists():
        return None, None, None
    if not pids_path or not label_path or not Path(pids_path).exists() or not Path(label_path).exists():
        return None, None, None

    scores = np.load(scores_path)
    with open(pids_path, "r", encoding="utf-8") as handle:
        patient_ids = [int(x) for x in json.load(handle)]
    with open(label_path, "r", encoding="utf-8") as handle:
        label_names = [str(x) for x in json.load(handle)]

    if scores.shape[0] != len(patient_ids) or scores.shape[1] != len(label_names):
        return None, None, None
    return scores, patient_ids, label_names


@dataclass
class ModelArtifacts:
    name: str
    patient_ids: List[int]
    label_names: List[str]
    pred_data: Dict[int, List[str]]
    output_subdir: Path
    predictions_jsonl: Path
    scores: Optional[np.ndarray] = None
    score_patient_ids: Optional[List[int]] = None
    score_label_names: Optional[List[str]] = None


def load_model_artifacts(
    model_cfg: Dict[str, Any],
    global_val_pids: List[int],
    evaluation_root: Optional[Path] = None,
) -> ModelArtifacts:
    """
    Load predictions from ``predictions_path`` JSONL.

    Optionally loads ``scores_path`` / ``pids_path`` / ``label_names_path`` when present
    (for ensemble dense scores or metrics_engine Recall@K).
    """
    name = model_cfg["name"]
    root = evaluation_root if evaluation_root is not None else Path("outputs/evaluation")
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_jsonl = Path(model_cfg["predictions_path"])
    if not pred_jsonl.is_file():
        raise FileNotFoundError(f"[{name}] predictions_path not found: {pred_jsonl}")

    label_names = load_labelset(model_cfg["labelset_path"])
    pred_data = load_predictions(str(pred_jsonl))
    patient_ids = list(global_val_pids)

    scores, spids, slabels = _load_optional_scores_bundle(model_cfg)

    return ModelArtifacts(
        name=name,
        patient_ids=patient_ids,
        label_names=label_names,
        pred_data=pred_data,
        output_subdir=out_dir,
        predictions_jsonl=pred_jsonl,
        scores=scores,
        score_patient_ids=spids,
        score_label_names=slabels,
    )
