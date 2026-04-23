"""Extract per-label stacking features from base-model score matrices.

Each base model contributes one score column per label per document (already normalised
by threshold so > 1.0 = positive for score-based models, 0/1 for binary-only models).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def extract_label_features(
    matrices: List[np.ndarray],
    label_idx: int,
    *,
    include_aggregates: bool = True,
) -> np.ndarray:
    """Build an (n_docs × n_features) feature matrix for a single label.

    Base features — one column per model: normalised score at ``label_idx``.
    Aggregate features (when ``include_aggregates``): max score, mean score, vote count
    (number of models with score ≥ 1.0).

    Parameters
    ----------
    matrices:
        List of (n_docs × n_labels) score matrices, one per base model.
    label_idx:
        Column index of the target label.
    include_aggregates:
        Whether to append max / mean / n_vote columns.

    Returns
    -------
    np.ndarray of shape (n_docs, K) or (n_docs, K + 3).
    """
    cols = [mat[:, label_idx] for mat in matrices]
    X = np.column_stack(cols).astype(np.float32)  # (n_docs, K)

    if include_aggregates:
        max_s = X.max(axis=1, keepdims=True)
        mean_s = X.mean(axis=1, keepdims=True)
        n_vote = (X >= 1.0).sum(axis=1, keepdims=True).astype(np.float32)
        X = np.concatenate([X, max_s, mean_s, n_vote], axis=1)

    return X


def build_target_matrix(
    gt_data: Dict,
    all_pids: List[int],
    all_labels: List[str],
) -> np.ndarray:
    """Build an (n_docs × n_labels) binary target matrix from ground-truth data.

    A cell is 1 if the label appears in *any* annotation group for that patient,
    0 otherwise.  This is a simplified per-code binary target (not group-level).

    Parameters
    ----------
    gt_data:
        ``Dict[patient_id, List[List[str]]]`` — annotation groups per patient.
    all_pids:
        Ordered list of patient IDs (row order).
    all_labels:
        Ordered list of label strings (column order).
    """
    label_to_idx = {lbl: j for j, lbl in enumerate(all_labels)}
    n, m = len(all_pids), len(all_labels)
    Y = np.zeros((n, m), dtype=np.float32)
    for i, pid in enumerate(all_pids):
        for grp in gt_data.get(pid, []):
            for code in grp:
                j = label_to_idx.get(code)
                if j is not None:
                    Y[i, j] = 1.0
    return Y
