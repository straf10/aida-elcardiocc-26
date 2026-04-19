"""
Fresh KMeans on cached validation embeddings → same per-cluster champion routing
as :mod:`per_cluster`, without using analysis ``cluster_assignments.json``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

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


def kmeans_cluster_labels(features: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
    """Return shape (n_docs,) integer cluster id per row."""
    from sklearn.cluster import KMeans

    n = features.shape[0]
    k = int(n_clusters)
    if k < 2 or n < k:
        raise ValueError(f"KMeans needs 2 ≤ n_clusters ≤ n_samples; got k={k}, n={n}")
    km = KMeans(n_clusters=k, random_state=int(random_state), n_init="auto")
    return km.fit_predict(features).astype(np.int32)


def assignments_from_labels(all_pids: List[int], labels: np.ndarray) -> Dict[int, int]:
    return {int(pid): int(labels[i]) for i, pid in enumerate(all_pids)}


def default_embeddings_path(cfg: dict, clustering_output_dir_fn) -> Path:
    """Resolve path to ``embeddings.npy`` from config (same defaults as analysis)."""
    explicit = get_cfg(cfg, "clustering.embeddings_cache", None)
    if explicit:
        return Path(explicit)
    return clustering_output_dir_fn(cfg) / "embeddings.npy"


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
    """
    For each K in ``k_list``, KMeans on aligned embeddings → champion per cluster → metrics.

    Returns list of ``(k, metrics_dict)`` (micro_f1, precision, recall, … from ``evaluate_data``).
    """
    if not embeddings_path.is_file():
        return []

    embeddings = np.load(embeddings_path)
    features = align_embeddings_to_ensemble_pids(embeddings, val_path, all_pids)
    if features is None:
        return []

    results: List[Tuple[int, Dict]] = []
    n_docs = len(all_pids)
    for k in k_list:
        k = int(k)
        if k < 2 or k > n_docs:
            continue
        labels = kmeans_cluster_labels(features, k, random_state)
        ca = assignments_from_labels(all_pids, labels)
        routing, _scores = build_cluster_champion_routing(
            ca, all_pids, names, per_model_preds, gt_data, all_labels,
        )
        if not routing:
            continue
        preds = per_cluster_champion_predict(ca, all_pids, routing, per_model_preds)
        m = evaluate_data(gt_data, preds, label_space=all_labels)
        results.append((k, m))
    return results
