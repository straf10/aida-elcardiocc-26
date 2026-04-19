"""
Fresh clustering on cached validation embeddings → per-cluster champion routing
(same as :mod:`per_cluster`), without using analysis ``cluster_assignments.json``.

Supports several ``sklearn`` algorithms (see ``run_embedding_cluster_sweep``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from src.evaluation.config_utils import get_cfg
    from src.evaluation.evaluator import evaluate_data
    from src.preprocessing.io_utils import load_jsonl
except ImportError:
    from ...evaluation.config_utils import get_cfg
    from ...evaluation.evaluator import evaluate_data
    from ...preprocessing.io_utils import load_jsonl

from .per_cluster import build_cluster_champion_routing, per_cluster_champion_predict


def val_pids_in_jsonl_order(val_path: str) -> List[int]:
    """Patient ids in the same row order as ``medical_clustering`` embeddings."""
    return [int(r["patient_id"]) for r in load_jsonl(val_path)]


def align_embeddings_to_ensemble_pids(
    embeddings: np.ndarray,
    val_path: str,
    all_pids: List[int],
) -> np.ndarray | None:
    """
    Reorder ``embeddings`` (rows = JSONL order) to match ``all_pids`` (ensemble matrix rows).
    Rows with unknown pid get the global mean embedding.
    """
    json_pids = val_pids_in_jsonl_order(val_path)
    if embeddings.shape[0] != len(json_pids):
        return None
    pid_to_row = {pid: i for i, pid in enumerate(json_pids)}
    n_dim = embeddings.shape[1]
    out = np.zeros((len(all_pids), n_dim), dtype=np.float64)
    centroid = embeddings.mean(axis=0)
    for i, pid in enumerate(all_pids):
        j = pid_to_row.get(pid)
        out[i] = embeddings[j] if j is not None else centroid
    return out


def _scale_features(features: np.ndarray, random_state: int) -> np.ndarray:
    """Zero-mean / unit-variance per dimension (helps GMM / spectral / ward)."""
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(features).astype(np.float64)


def embedding_cluster_labels(
    features: np.ndarray,
    n_clusters: int,
    method: str,
    random_state: int,
) -> np.ndarray:
    """
    Return shape (n_docs,) integer cluster id per row.

    ``method``: ``"kmeans"`` | ``"agglomerative"`` (Ward) | ``"gmm"`` (diag covariance) |
    ``"spectral"`` (nearest-neighbor affinity) | ``"dbscan"`` (``n_clusters`` reused as
    ``min_samples``; ``eps`` from a percentile of the k-distance graph).
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
    from sklearn.mixture import GaussianMixture

    n = features.shape[0]
    k = int(n_clusters)
    rs = int(random_state)
    m = (method or "kmeans").lower().strip()
    if m in ("dbscan",):
        if k < 2:
            raise ValueError(f"DBSCAN sweep index (min_samples) must be ≥ 2; got k={k}")
    elif k < 2 or n < k:
        raise ValueError(f"Need 2 ≤ n_clusters ≤ n_samples; got k={k}, n={n}")

    if m == "kmeans":
        km = KMeans(n_clusters=k, random_state=rs, n_init="auto")
        return km.fit_predict(features).astype(np.int32)
    if m in ("agglomerative", "ward", "hierarchical"):
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        return agg.fit_predict(features).astype(np.int32)
    if m in ("gmm", "gaussian_mixture"):
        gmm = GaussianMixture(
            n_components=k,
            random_state=rs,
            covariance_type="diag",
            max_iter=200,
        )
        gmm.fit(features)
        return gmm.predict(features).astype(np.int32)
    if m in ("spectral",):
        n_neighbors = int(min(30, max(5, n // 15)))
        sc = SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            random_state=rs,
            assign_labels="kmeans",
        )
        return sc.fit_predict(features).astype(np.int32)
    if m in ("dbscan",):
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors

        min_samples = max(2, min(k, n))
        nn_k = min(min_samples, n)
        nn = NearestNeighbors(n_neighbors=nn_k, metric="euclidean")
        nn.fit(features)
        dists, _ = nn.kneighbors(features)
        core_dist = dists[:, -1]
        pct = float(min(95, max(10, 20 + k)))
        eps = float(np.percentile(core_dist, pct))
        if not np.isfinite(eps) or eps <= 0:
            med = float(np.median(core_dist[np.isfinite(core_dist)]))
            eps = med + 1e-9
        dbs = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        return dbs.fit_predict(features).astype(np.int32)
    raise ValueError(
        f"Unknown clustering method: {method!r} "
        "(try kmeans, agglomerative, gmm, spectral, dbscan)"
    )


def assignments_from_labels(all_pids: List[int], labels: np.ndarray) -> Dict[int, int]:
    return {int(pid): int(labels[i]) for i, pid in enumerate(all_pids)}


def default_embeddings_path(cfg: dict, clustering_output_dir_fn) -> Path:
    """Resolve path to ``embeddings.npy`` from config (same defaults as analysis)."""
    explicit = get_cfg(cfg, "clustering.embeddings_cache", None)
    if explicit:
        return Path(explicit)
    return clustering_output_dir_fn(cfg) / "embeddings.npy"


def run_embedding_cluster_sweep(
    embeddings_path: Path,
    val_path: str,
    all_pids: List[int],
    names: List[str],
    per_model_preds: Dict[str, Dict[int, List[str]]],
    gt_data: Dict,
    all_labels: List[str],
    k_list: Sequence[int],
    methods: Sequence[str],
    random_state: int,
) -> List[Tuple[str, int, Dict]]:
    """
    For each (method, K), cluster scaled embeddings → champion per cluster → validation metrics.

    Returns ``(method_name, k, metrics_dict)`` rows (micro_f1, precision, recall, …).
    """
    if not embeddings_path.is_file():
        return []

    embeddings = np.load(embeddings_path)
    raw = align_embeddings_to_ensemble_pids(embeddings, val_path, all_pids)
    if raw is None:
        return []

    features = _scale_features(raw, random_state)
    results: List[Tuple[str, int, Dict]] = []
    n_docs = len(all_pids)

    for method in methods:
        mname = str(method).lower().strip()
        for k in k_list:
            k = int(k)
            if k < 2:
                continue
            # K-means / Ward / … need k ≤ n; DBSCAN only uses k as min_samples (capped in-clusterer).
            if mname not in ("dbscan",) and k > n_docs:
                continue
            try:
                labels = embedding_cluster_labels(features, k, mname, random_state)
            except Exception as exc:
                print(f"  {mname:<14} K={k:2d}  (failed: {exc})")
                continue
            ca = assignments_from_labels(all_pids, labels)
            routing, _scores = build_cluster_champion_routing(
                ca, all_pids, names, per_model_preds, gt_data, all_labels,
            )
            if not routing:
                continue
            preds = per_cluster_champion_predict(ca, all_pids, routing, per_model_preds)
            met = evaluate_data(gt_data, preds, label_space=all_labels)
            results.append((mname, k, met))

    return results


def run_embedding_kmeans_per_cluster_champion(
    embeddings_path: Path,
    val_path: str,
    all_pids: List[int],
    names: List[str],
    per_model_preds: Dict[str, Dict[int, List[str]]],
    gt_data: Dict,
    all_labels: List[str],
    k_list: List[int],
    random_state: int,
) -> List[Tuple[int, Dict]]:
    """Backward-compatible: same as ``run_embedding_cluster_sweep`` with ``methods=("kmeans",)`` only."""
    rows = run_embedding_cluster_sweep(
        embeddings_path,
        val_path,
        all_pids,
        names,
        per_model_preds,
        gt_data,
        all_labels,
        k_list,
        ("kmeans",),
        random_state,
    )
    return [(k, m) for _meth, k, m in rows]
