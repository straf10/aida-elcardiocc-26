"""Patient-level cluster features for stacking (train-only fit, no label leakage).

Mirrors part of the metaheuristic idea (“similar patients → similar fusion”): ``KMeans``
is fit on **train** rows of concatenated base-model score matrices; val/test/blind rows
are assigned to the nearest centroid. One-hot cluster ids are appended to every
per-label meta-feature row so the meta-learner can learn interactions such as
“in cluster 2, up-weight dictionary_baseline when std across models is high”.

This does **not** use validation gold to choose clusters—only unsupervised geometry of
train scores—so it is compatible with a held-out val for threshold tuning.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sklearn.cluster import KMeans


def hstack_score_matrices(matrices: List[np.ndarray]) -> np.ndarray:
    """(n_docs × sum_j n_labels_j) — here all n_labels match, so (n_docs, K * n_labels)."""
    return np.hstack([np.asarray(m, dtype=np.float64) for m in matrices])


def fit_train_patient_clusters(
    train_matrices: List[np.ndarray],
    n_clusters: int,
    *,
    seed: int,
) -> Tuple[Optional["KMeans"], int]:
    """Fit ``KMeans`` on train score rows. Returns ``(None, 0)`` if disabled or infeasible."""
    k = int(n_clusters)
    if k < 2 or not train_matrices:
        return None, 0
    R = hstack_score_matrices(train_matrices)
    n = int(R.shape[0])
    if n < k:
        return None, 0
    from sklearn.cluster import KMeans

    km: KMeans = KMeans(n_clusters=k, random_state=int(seed), n_init=10).fit(R)
    return km, k


def patient_cluster_onehot(
    matrices: List[np.ndarray],
    kmeans: "KMeans",
    n_clusters: int,
) -> np.ndarray:
    """Shape ``(n_docs, n_clusters)`` float32 one-hot from ``kmeans.predict``."""
    R = hstack_score_matrices(matrices)
    cid = kmeans.predict(R).astype(np.int64)
    return np.eye(int(n_clusters), dtype=np.float32)[cid]
