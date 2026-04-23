"""Load per-model **predictions JSONL**; optionally companion score tensors for ensemble / Recall@K."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from preprocessing.io_utils import load_labelset

from .io_utils import load_predictions


def resolve_predictions_jsonl_path(model_cfg: Dict[str, Any], split: str) -> Path:
    """
    Which JSONL to load for ``load_model_artifacts``.

    - ``compare``: ``predictions_path`` (typically ``test_predictions.jsonl``; held-out test for ``compare_methods``).
    - ``val`` / ``train``: ensemble / routing on committee splits — need matching patient_ids.
      Prefer explicit ``{split}_predictions_path`` in the model config, else a sidecar file
      ``<dirname(predictions_path)>/{split}_predictions.jsonl`` (e.g. ``val_predictions.jsonl``).
    """
    if split not in ("compare", "val", "train"):
        raise ValueError(f"split must be compare|val|train, got {split!r}")
    name = str(model_cfg.get("name", "model"))
    base = Path(model_cfg["predictions_path"])
    if split == "compare":
        return base
    key = f"{split}_predictions_path"
    explicit = model_cfg.get(key)
    if isinstance(explicit, str) and explicit.strip() and Path(explicit).is_file():
        return Path(explicit)
    side = base.parent / f"{split}_predictions.jsonl"
    if side.is_file():
        return side
    raise FileNotFoundError(
        f"[{name}] need {split} predictions for ensemble (patient_ids must match {split} gold). "
        f"Either set models[].{key} in evaluation config, or create:\n  {side}\n"
        f"(test/compare file {base} is not sufficient.)"
    )


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
    *,
    predictions_split: str = "compare",
    load_scores: bool = True,
) -> ModelArtifacts:
    """
    Load predictions from JSONL (see ``resolve_predictions_jsonl_path``).

    ``predictions_split``: ``compare`` uses ``predictions_path`` (test). ``val`` / ``train`` use
    sidecar or ``{split}_predictions_path`` for ensemble alignment with that split's patient_ids.

    Optionally loads ``scores_path`` / ``pids_path`` / ``label_names_path`` when present
    (for ensemble dense scores). Scores are skipped for ``train`` by default (bundles are val-aligned).
    """
    name = model_cfg["name"]
    root = evaluation_root if evaluation_root is not None else Path("outputs/models/evaluation")
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_jsonl = resolve_predictions_jsonl_path(model_cfg, predictions_split)
    if not pred_jsonl.is_file():
        raise FileNotFoundError(f"[{name}] predictions JSONL not found: {pred_jsonl}")

    label_names = load_labelset(model_cfg["labelset_path"])
    pred_data = load_predictions(str(pred_jsonl))
    patient_ids = list(global_val_pids)

    if load_scores and predictions_split != "train":
        scores, spids, slabels = _load_optional_scores_bundle(model_cfg)
    else:
        scores, spids, slabels = None, None, None

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
