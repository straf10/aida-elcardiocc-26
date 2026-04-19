"""
Per-cluster champion routing after **unsupervised clustering** of per-document features.

Clustering features are **stacked score matrices** from the ensemble models (main pipeline and
this module's standalone CLI). Several ``sklearn`` algorithms are supported
(see ``run_cluster_sweep_from_features``).

Uses the same routing helpers as :mod:`per_cluster`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from src.evaluation.evaluator import evaluate_data
except ImportError:
    from ...evaluation.evaluator import evaluate_data

from .per_cluster import build_cluster_champion_routing, per_cluster_champion_predict


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

    ``method``: ``"kmeans"`` | ``"kmeans_cosine"`` (L2 row-normalize then KMeans) |
    ``"agglomerative"`` (Ward) | ``"gmm"`` (diag covariance) |
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

    if m in ("kmeans", "kmeans_cosine"):
        Xw = features
        if m == "kmeans_cosine":
            from sklearn.preprocessing import normalize

            Xw = normalize(features, norm="l2", axis=1, copy=True)
        # High-K routing is sensitive to local minima; more inits than sklearn's default "auto".
        km = KMeans(n_clusters=k, random_state=rs, n_init=10, max_iter=500)
        return km.fit_predict(Xw).astype(np.int32)
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
        "(try kmeans, kmeans_cosine, agglomerative, gmm, spectral, dbscan)"
    )


def assignments_from_labels(all_pids: List[int], labels: np.ndarray) -> Dict[int, int]:
    return {int(pid): int(labels[i]) for i, pid in enumerate(all_pids)}


def clustering_features_from_matrices(matrices: List[np.ndarray]) -> np.ndarray:
    """Stack model score / vote matrices horizontally → shape ``(n_docs, n_labels * n_models)``."""
    if not matrices:
        raise ValueError("matrices must be non-empty")
    return np.hstack([np.asarray(m, dtype=np.float64) for m in matrices])


def run_cluster_sweep_from_features(
    features_raw: np.ndarray,
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
    For each (method, K), cluster rows of ``features_raw`` → per-cluster champion → validation metrics.

    ``features_raw`` must be 2-D with one row per patient in ``all_pids`` order.
    Returns ``(method_name, k, metrics_dict)`` rows.
    """
    if features_raw.ndim != 2 or int(features_raw.shape[0]) != len(all_pids):
        return []

    features = _scale_features(features_raw.astype(np.float64, copy=False), int(random_state))
    results: List[Tuple[str, int, Dict]] = []
    n_docs = len(all_pids)

    for method in methods:
        mname = str(method).lower().strip()
        for k in k_list:
            k = int(k)
            if k < 2:
                continue
            if mname not in ("dbscan",) and k > n_docs:
                continue
            try:
                labels = embedding_cluster_labels(features, k, mname, random_state)
            except Exception as exc:
                print(f"  {mname:<16} K={k:3d}  (failed: {exc})")
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


def run_score_matrix_cluster_sweep(
    matrices: List[np.ndarray],
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
    Per-cluster champion sweep using **stacked ensemble score matrices** as clustering features
    (no ``embeddings.npy``).
    """
    raw = clustering_features_from_matrices(matrices)
    return run_cluster_sweep_from_features(
        raw,
        all_pids,
        names,
        per_model_preds,
        gt_data,
        all_labels,
        k_list,
        methods,
        random_state,
    )


_STANDALONE_DEFAULT_METHODS = (
    "kmeans",
    "kmeans_cosine",
    "agglomerative",
    "gmm",
    "spectral",
    "dbscan",
)


def _run_standalone_cli() -> None:
    import argparse
    from pathlib import Path

    from src.ensemble_metaheuristic.strategy_cli import (
        build_per_model_preds,
        load_validation_bundle,
        prepend_repo_root_for_strategy_file,
    )

    prepend_repo_root_for_strategy_file(Path(__file__))

    ap = argparse.ArgumentParser(
        description="Cluster patients → per-cluster champion (this module only). "
        "Uses stacked ensemble score matrices as clustering features.",
    )
    ap.add_argument("--config", default="src/analysis/analysis.yaml", help="Analysis YAML.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for clustering.")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Smaller K grid and kmeans only (smoke run).",
    )
    ap.add_argument(
        "--k-min",
        type=int,
        default=2,
        metavar="K",
        help="Minimum K (ignored with --quick).",
    )
    ap.add_argument(
        "--k-max",
        type=int,
        default=502,
        metavar="K",
        help="Maximum K inclusive (ignored with --quick).",
    )
    ap.add_argument(
        "--k-step",
        type=int,
        default=32,
        metavar="S",
        help="Step between K values (ignored with --quick).",
    )
    ap.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated clusterers (used when not --quick). Empty = default full set.",
    )
    args = ap.parse_args()

    matrices, names, is_score_model, gt_data, all_pids, all_labels, *_bundle_tail = load_validation_bundle(
        args.config,
    )
    per_model_preds = build_per_model_preds(matrices, names, is_score_model, all_pids, all_labels)

    if args.quick:
        k_list = [16, 32, 64]
        methods: Tuple[str, ...] = ("kmeans",)
    else:
        lo = max(2, int(args.k_min))
        hi = int(args.k_max)
        step = max(1, int(args.k_step))
        if hi < lo:
            raise SystemExit("--k-max must be >= --k-min")
        k_list = list(range(lo, hi + 1, step))
        if k_list and k_list[-1] < hi:
            k_list.append(hi)
        if not k_list:
            raise SystemExit("K list is empty; check --k-min, --k-max, --k-step")
        if args.methods.strip():
            methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
        else:
            methods = _STANDALONE_DEFAULT_METHODS

    print("Per-cluster champion — clustering sweep (this module only)")
    print("  Features: stacked model score matrices")
    print(f"  K sweep: {len(k_list)} value(s) from {k_list[0]} to {k_list[-1]}" + (f" step {args.k_step}" if not args.quick else ""))
    rows = run_score_matrix_cluster_sweep(
        matrices,
        all_pids,
        names,
        per_model_preds,
        gt_data,
        all_labels,
        k_list,
        methods,
        int(args.seed),
    )
    if not rows:
        print("  No results (empty features, alignment failed, or all runs failed).")
        return
    for meth, k, m in rows:
        print(
            f"  {meth:<16} K={k:3d}  micro-F1={m['micro_f1']:.4f}  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}",
        )
    best_meth, best_k, best_m = max(rows, key=lambda x: x[2]["micro_f1"])
    print(f"  Best: {best_meth} K={best_k}  micro-F1={best_m['micro_f1']:.4f}")


if __name__ == "__main__":
    _run_standalone_cli()
