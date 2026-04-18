from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    from src.analysis.metrics_engine import metrics_by_cluster
except ImportError:
    from analysis.metrics_engine import metrics_by_cluster  # type: ignore

from src.visualisation.src.cluster_context import cluster_ytick_label, load_cluster_summary
from src.visualisation.src.config import MODEL_ABBREV
from src.visualisation.src.cross_model_data import CrossModelBundle


def _cluster_ids_from_metrics(per_cluster: Dict[str, Any]) -> List[int]:
    return sorted(int(k) for k in per_cluster.keys())


def _matrices(
    bundle: CrossModelBundle,
    assignments: Dict[int, int],
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted cluster ids and F1, P, R matrices (n_clusters x n_models)."""
    f1_rows: List[List[float]] = []
    p_rows: List[List[float]] = []
    r_rows: List[List[float]] = []
    cluster_keys: List[int] | None = None

    for name in bundle.model_names:
        pc = metrics_by_cluster(
            assignments,
            bundle.gt_data,
            bundle.pred_by_model[name],
            bundle.label_names,
        )
        if cluster_keys is None:
            cluster_keys = _cluster_ids_from_metrics(pc)
        f1_row = [float(pc[str(c)]["micro_f1"]) for c in cluster_keys]
        p_row = [float(pc[str(c)]["precision"]) for c in cluster_keys]
        r_row = [float(pc[str(c)]["recall"]) for c in cluster_keys]
        f1_rows.append(f1_row)
        p_rows.append(p_row)
        r_rows.append(r_row)

    if not cluster_keys:
        return [], np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))

    f1 = np.array(f1_rows).T  # clusters x models
    p_mat = np.array(p_rows).T
    r_mat = np.array(r_rows).T
    return cluster_keys, f1, p_mat, r_mat


def plot_cluster_f1_precision_recall(
    bundle: CrossModelBundle,
    assignments: Dict[int, int],
    summary: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    cids, f1, p_mat, r_mat = _matrices(bundle, assignments)
    if not cids:
        return

    ylabels = [cluster_ytick_label(c, summary) for c in cids]
    xlabels = [MODEL_ABBREV.get(n, n) for n in bundle.model_names]

    fig, axes = plt.subplots(1, 3, figsize=(14, max(6, len(cids) * 0.35)))
    for ax, mat, title in zip(
        axes,
        [f1, p_mat, r_mat],
        ["Group micro F1", "Precision", "Recall"],
    ):
        sns.heatmap(
            mat,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            xticklabels=xlabels,
            yticklabels=ylabels,
        )
        ax.set_title(title)
        ax.set_xlabel("Model")
    fig.suptitle("Per-cluster group metrics (validation; xlm_r_base excluded)", y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cluster_f1_gap_from_best(
    bundle: CrossModelBundle,
    assignments: Dict[int, int],
    summary: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    cids, f1, _, _ = _matrices(bundle, assignments)
    if not cids or f1.size == 0:
        return

    best = f1.max(axis=1, keepdims=True)
    gap = best - f1
    ylabels = [cluster_ytick_label(c, summary) for c in cids]
    xlabels = [MODEL_ABBREV.get(n, n) for n in bundle.model_names]

    plt.figure(figsize=(max(8, len(bundle.model_names) * 2), max(6, len(cids) * 0.35)))
    sns.heatmap(
        gap,
        annot=True,
        fmt=".2f",
        cmap="Oranges",
        xticklabels=xlabels,
        yticklabels=ylabels,
        cbar_kws={"label": "F1 gap below best-in-row"},
    )
    plt.xlabel("Model")
    plt.ylabel("Cluster")
    plt.title("Distance from best group F1 in each cluster (xlm_r_base excluded)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_cluster_metrics_csv(
    bundle: CrossModelBundle,
    assignments: Dict[int, int],
    summary: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    cids, f1, p_mat, r_mat = _matrices(bundle, assignments)
    if not cids:
        return

    fieldnames = ["cluster_id", "ytick_label"]
    for n in bundle.model_names:
        ab = MODEL_ABBREV.get(n, n)
        fieldnames += [f"f1_{ab}", f"precision_{ab}", f"recall_{ab}"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, cid in enumerate(cids):
            row: Dict[str, Any] = {
                "cluster_id": cid,
                "ytick_label": cluster_ytick_label(cid, summary),
            }
            for j, n in enumerate(bundle.model_names):
                ab = MODEL_ABBREV.get(n, n)
                row[f"f1_{ab}"] = f1[i, j]
                row[f"precision_{ab}"] = p_mat[i, j]
                row[f"recall_{ab}"] = r_mat[i, j]
            w.writerow(row)


def run_cluster_metrics_plots(
    bundle: CrossModelBundle,
    assignments: Dict[int, int],
    cfg: Dict[str, Any],
    out_dir: Path,
) -> None:
    summary = load_cluster_summary(cfg)
    plot_cluster_f1_precision_recall(
        bundle,
        assignments,
        summary,
        out_dir / "cluster_micro_f1_precision_recall.png",
    )
    plot_cluster_f1_gap_from_best(
        bundle,
        assignments,
        summary,
        out_dir / "cluster_f1_gap_from_best.png",
    )
    write_cluster_metrics_csv(
        bundle,
        assignments,
        summary,
        out_dir / "cluster_metrics_summary.csv",
    )
