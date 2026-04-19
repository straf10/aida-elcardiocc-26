from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from src.analysis.common import ModelArtifacts, load_model_artifacts
    from src.evaluation.config_utils import get_cfg
except ImportError:
    from analysis.common import ModelArtifacts, load_model_artifacts  # type: ignore
    from evaluation.config_utils import get_cfg  # type: ignore

from src.visualisation.src.config import included_model_configs
from src.visualisation.src.cross_model_data import CrossModelBundle


def analysis_output_root(cfg: Dict[str, Any]) -> Path:
    """Resolved ``output.dir`` from analysis config (parent of per-model subfolders)."""
    return Path(get_cfg(cfg, "output.dir", "outputs/analysis"))


def load_all_model_artifacts(bundle: CrossModelBundle) -> Dict[str, ModelArtifacts]:
    """Load ``ModelArtifacts`` for each model in ``bundle`` (excludes ``EXCLUDED_MODELS``)."""
    out_root = analysis_output_root(bundle.cfg)
    model_cfgs = included_model_configs(bundle.cfg)
    return {
        m["name"]: load_model_artifacts(m, bundle.patient_ids, analysis_out_dir=out_root)
        for m in model_cfgs
    }


def pid_row_index(art: ModelArtifacts) -> Dict[int, int]:
    """Map ``patient_id`` → row index in ``art.scores`` (meaningful when ``scores`` is not ``None``)."""
    if art.scores is None:
        return {}
    pids = art.score_patient_ids if art.score_patient_ids is not None else art.patient_ids
    return {int(pid): i for i, pid in enumerate(pids)}


def load_thresholds_vector(model_cfg: Dict[str, Any], label_names: List[str]) -> np.ndarray:
    """Return per-label thresholds for a model with ``scores_path``, aligned with ``label_names``."""
    thresholds_path = model_cfg.get("thresholds_path")
    if thresholds_path and Path(thresholds_path).exists():
        if thresholds_path.endswith(".npy"):
            return np.load(thresholds_path).astype(np.float64)
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresh_data = json.load(f)
        if (
            isinstance(thresh_data, dict)
            and "thresholds" in thresh_data
            and isinstance(thresh_data["thresholds"], dict)
        ):
            thresholds_dict = thresh_data["thresholds"]
        elif isinstance(thresh_data, dict):
            thresholds_dict = thresh_data
        else:
            thresholds_dict = {}
        return np.array([float(thresholds_dict.get(l, 0.5)) for l in label_names], dtype=np.float64)
    return np.full(len(label_names), 0.5, dtype=np.float64)


def score_model_cfgs(bundle: CrossModelBundle) -> List[Dict[str, Any]]:
    """Model YAML dicts that define ``scores_path`` (for ensemble diagnostics on raw scores)."""
    return [m for m in included_model_configs(bundle.cfg) if m.get("scores_path")]


def y_true_matrix_for_artifact(
    gt_data: Dict[int, List[List[str]]],
    art: ModelArtifacts,
) -> np.ndarray:
    """Binary ground-truth matrix ``(n_docs, n_labels)`` aligned with ``art.scores`` shape when present."""
    row_pids = art.score_patient_ids if art.score_patient_ids is not None else art.patient_ids
    col_labels = art.score_label_names if art.score_label_names is not None else art.label_names
    label_to_idx = {l: i for i, l in enumerate(col_labels)}
    n_docs = len(row_pids)
    n_labels = len(col_labels)
    y_true = np.zeros((n_docs, n_labels), dtype=np.int8)
    for i, pid in enumerate(row_pids):
        for group in gt_data.get(int(pid), []):
            for code in group:
                j = label_to_idx.get(code)
                if j is not None:
                    y_true[i, j] = 1
    return y_true


def y_pred_binary_matrix(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Binary predictions from sigmoid ``scores`` and per-label ``thresholds`` (broadcastable)."""
    return (scores >= thresholds).astype(np.int8)
