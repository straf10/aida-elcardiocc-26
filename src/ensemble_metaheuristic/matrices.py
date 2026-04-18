"""Build per-model score matrices for ensemble."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

try:
    from src.analysis.common import resolve_model_paths
except ImportError:
    from ..analysis.common import resolve_model_paths


def load_thresholds_for_model(model_cfg: dict, label_names: List[str]) -> np.ndarray:
    """Load per-label thresholds aligned to label_names (defaults to 0.5)."""
    paths = resolve_model_paths(model_cfg)
    tpath = paths.get("thresholds_path", "")
    if not tpath or not Path(tpath).exists():
        return np.full(len(label_names), 0.5, dtype=np.float32)
    if tpath.endswith(".npy"):
        return np.load(tpath).astype(np.float32)
    with open(tpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "thresholds" in data and isinstance(data["thresholds"], dict):
        data = data["thresholds"]
    return np.array([float(data.get(l, 0.5)) for l in label_names], dtype=np.float32)


def build_score_matrix(
    artifacts,
    all_pids: List[int],
    all_labels: List[str],
    thresholds: np.ndarray | None = None,
) -> np.ndarray:
    """Return (n_pids x n_labels) float32 matrix for one model.

    Score-based models: scores divided by per-label thresholds so >1.0 means positive.
    Prediction-only models: binary 0/1 values.
    """
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    pid_to_row = {pid: i for i, pid in enumerate(all_pids)}
    n, m = len(all_pids), len(all_labels)
    mat = np.zeros((n, m), dtype=np.float32)

    if artifacts.scores is not None:
        model_label_to_idx = {l: i for i, l in enumerate(artifacts.label_names)}
        thr = thresholds if thresholds is not None else np.full(m, 0.5, dtype=np.float32)
        for local_i, pid in enumerate(artifacts.patient_ids):
            row = pid_to_row.get(pid)
            if row is None:
                continue
            for label, local_j in model_label_to_idx.items():
                col = label_to_idx.get(label)
                if col is not None:
                    t = thr[col]
                    mat[row, col] = artifacts.scores[local_i, local_j] / t if t > 0 else 0.0
    else:
        for pid, codes in artifacts.pred_data.items():
            row = pid_to_row.get(pid)
            if row is None:
                continue
            for code in codes:
                col = label_to_idx.get(code)
                if col is not None:
                    mat[row, col] = 1.0

    return mat
