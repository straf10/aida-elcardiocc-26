from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .io_utils import load_predictions
    from ..preprocessing.io_utils import load_labelset
except ImportError:
    from src.evaluation.io_utils import load_predictions
    from src.preprocessing.io_utils import load_labelset


@dataclass
class ModelArtifacts:
    name: str
    scores: Optional[np.ndarray]
    """Row order matches ``score_patient_ids`` when scores are loaded."""
    score_patient_ids: Optional[List[int]]
    """Label order matches score matrix columns (for Recall@K)."""
    score_label_names: Optional[List[str]]
    patient_ids: List[int]
    label_names: List[str]
    pred_data: Dict[int, List[str]]
    output_subdir: Path
    predictions_jsonl: Path


def _load_optional_scores_bundle(
    model_cfg: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[str]]]:
    """Load val_scores.npy + aligned pids + label column names for Recall@K (optional)."""
    scores_path = model_cfg.get("scores_path")
    pids_path = model_cfg.get("pids_path")
    label_path = model_cfg.get("label_names_path")
    if not scores_path or not Path(scores_path).exists():
        return None, None, None
    if not pids_path or not label_path:
        print(
            f"[{model_cfg.get('name', '?')}] WARN: scores_path set but pids_path/"
            f"label_names_path missing; skipping Recall@K inputs."
        )
        return None, None, None
    if not Path(pids_path).exists() or not Path(label_path).exists():
        print(
            f"[{model_cfg.get('name', '?')}] WARN: scores companion files missing; "
            "skipping Recall@K inputs."
        )
        return None, None, None

    scores = np.load(scores_path)
    with open(pids_path, "r", encoding="utf-8") as handle:
        patient_ids = [int(x) for x in json.load(handle)]
    with open(label_path, "r", encoding="utf-8") as handle:
        label_names = [str(x) for x in json.load(handle)]

    if scores.shape[0] != len(patient_ids):
        raise ValueError(
            f"Rows in scores ({scores.shape[0]}) do not match patient_ids ({len(patient_ids)})."
        )
    if scores.shape[1] != len(label_names):
        raise ValueError(
            f"Score columns ({scores.shape[1]}) do not match label names ({len(label_names)})."
        )
    return scores, patient_ids, label_names


def load_model_artifacts(
    model_cfg: Dict[str, Any],
    global_val_pids: List[int],
    evaluation_root: Optional[Path] = None,
) -> ModelArtifacts:
    """
    Load predictions from ``predictions_path`` JSONL and optional score tensors for Recall@K.
    """
    name = model_cfg["name"]
    root = evaluation_root if evaluation_root is not None else Path("outputs/evaluation")
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_jsonl = Path(model_cfg["predictions_path"])
    label_names = load_labelset(model_cfg["labelset_path"])
    pred_data = load_predictions(str(pred_jsonl))
    patient_ids = global_val_pids.copy()

    scores, score_patient_ids, score_label_names = _load_optional_scores_bundle(model_cfg)

    return ModelArtifacts(
        name=name,
        scores=scores,
        score_patient_ids=score_patient_ids,
        score_label_names=score_label_names,
        patient_ids=patient_ids,
        label_names=label_names,
        pred_data=pred_data,
        output_subdir=out_dir,
        predictions_jsonl=pred_jsonl,
    )
