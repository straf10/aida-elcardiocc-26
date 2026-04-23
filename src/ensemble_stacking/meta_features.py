"""Extract per-label stacking features from base-model score matrices.

Each base model contributes one score column per label per document (already normalised
by threshold so > 1.0 = positive for score-based models, 0/1 for binary-only models).
"""
from __future__ import annotations

from typing import Dict, List, Literal

import numpy as np

MetaFeatureMode = Literal["default", "rich", "full", "unified"]


def extract_label_features(
    matrices: List[np.ndarray],
    label_idx: int,
    *,
    include_aggregates: bool = True,
    meta_features: MetaFeatureMode = "default",
) -> np.ndarray:
    """Build an (n_docs × n_features) feature matrix for a single label.

    Base features — one column per model: normalised score at ``label_idx``.
    Aggregate features (when ``include_aggregates``): max score, mean score, vote count
    (number of models with score ≥ 1.0).

    ``meta_features``:

    * ``default`` — base columns plus aggregates only (legacy).
    * ``rich`` — adds std / min / range across models, and pairwise products of base scores
      (captures non-linear committee interactions for small ``K``).
    * ``full`` — ``rich`` plus one document-level column: mean normalised score across all
      models and labels (overall committee “temperature” for that patient).
    * ``unified`` — ``full`` plus an explicit **voting** block: fraction ≥1.0, fraction ≥0.5,
      hard-vote entropy, and top-two score margin (same for every label row of a patient;
      the meta-learner still predicts **per label**).

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
    np.ndarray of shape (n_docs, n_features).
    """
    if meta_features not in ("default", "rich", "full", "unified"):
        raise ValueError(f"meta_features must be default|rich|full|unified, got {meta_features!r}")

    use_rich = meta_features in ("rich", "full", "unified")
    use_doc_mean = meta_features in ("full", "unified")
    append_vote = meta_features == "unified"

    cols = [mat[:, label_idx] for mat in matrices]
    X_base = np.column_stack(cols).astype(np.float32)  # (n_docs, K)
    parts: List[np.ndarray] = [X_base]

    if include_aggregates:
        max_s = X_base.max(axis=1, keepdims=True)
        mean_s = X_base.mean(axis=1, keepdims=True)
        n_vote = (X_base >= 1.0).sum(axis=1, keepdims=True).astype(np.float32)
        parts.extend([max_s, mean_s, n_vote])

    if use_rich:
        std_s = X_base.std(axis=1, keepdims=True)
        min_s = X_base.min(axis=1, keepdims=True)
        range_s = (X_base.max(axis=1) - X_base.min(axis=1)).astype(np.float32)[:, np.newaxis]
        parts.extend([std_s, min_s, range_s])
        k = X_base.shape[1]
        if k >= 2:
            for i in range(k):
                for j in range(i + 1, k):
                    parts.append((X_base[:, i] * X_base[:, j])[:, np.newaxis])

    if use_doc_mean:
        stacked = np.stack([m.astype(np.float32) for m in matrices], axis=0)
        doc_mean = stacked.mean(axis=(0, 2))[:, np.newaxis].astype(np.float32)
        parts.append(doc_mean)

    out = np.concatenate(parts, axis=1)
    if append_vote:
        k = X_base.shape[1]
        eps = 1e-7
        frac_ge_1 = (X_base >= 1.0).mean(axis=1, keepdims=True).astype(np.float32)
        frac_ge_half = (X_base >= 0.5).mean(axis=1, keepdims=True).astype(np.float32)
        p = np.clip(frac_ge_1, eps, 1.0 - eps)
        vote_entropy = (
            -(p * np.log(p + eps) + (1.0 - p) * np.log(1.0 - p + eps)).astype(np.float32)
        )
        if k >= 2:
            xs = np.sort(X_base, axis=1)
            top2_margin = (xs[:, -1:] - xs[:, -2:-1]).astype(np.float32)
        else:
            top2_margin = np.zeros((X_base.shape[0], 1), dtype=np.float32)
        out = np.concatenate([out, frac_ge_1, frac_ge_half, vote_entropy, top2_margin], axis=1)

    return out


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
