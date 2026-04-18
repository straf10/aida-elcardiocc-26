from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.visualisation.src.cluster_context import cluster_ytick_label, load_cluster_summary
from src.visualisation.src.config import MODEL_ABBREV
from src.visualisation.src.cross_model_data import CrossModelBundle


def _aggregate_fp_fn(
    bundle: CrossModelBundle,
    assignments: Dict[int, int],
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns sorted cluster ids, mean_fp (clusters x models), mean_fn, total_fp_plus_fn.
    """
    cids = sorted({cid for cid in assignments.values()})
    if not cids:
        return [], np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))

    idx = {c: i for i, c in enumerate(cids)}
    n_m = len(bundle.model_names)
    sum_fp: DefaultDict[Tuple[int, str], float] = defaultdict(float)
    sum_fn: DefaultDict[Tuple[int, str], float] = defaultdict(float)
    cnt: DefaultDict[Tuple[int, str], int] = defaultdict(int)

    for mi, name in enumerate(bundle.model_names):
        for row in bundle.metrics_by_model[name].get("doc_breakdown", []):
            pid = int(row["patient_id"])
            cid = assignments.get(pid)
            if cid is None:
                continue
            key = (cid, name)
            sum_fp[key] += float(row.get("fp", 0))
            sum_fn[key] += float(row.get("fn", 0))
            cnt[key] += 1

    mean_fp = np.zeros((len(cids), n_m))
    mean_fn = np.zeros((len(cids), n_m))
    mass = np.zeros((len(cids), n_m))

    for mi, name in enumerate(bundle.model_names):
        for c in cids:
            i = idx[c]
            key = (c, name)
            n = cnt[key]
            if n > 0:
                mean_fp[i, mi] = sum_fp[key] / n
                mean_fn[i, mi] = sum_fn[key] / n
            mass[i, mi] = sum_fp[key] + sum_fn[key]

    return cids, mean_fp, mean_fn, mass


def _plot_heatmap(
    data: np.ndarray,
    cids: List[int],
    summary: List[Dict[str, Any]],
    model_names: List[str],
    title: str,
    cbar_label: str,
    out_path: Path,
    vmax: float | None = None,
) -> None:
    if not cids:
        return
    ylabels = [cluster_ytick_label(c, summary) for c in cids]
    xlabels = [MODEL_ABBREV.get(n, n) for n in model_names]
    plt.figure(figsize=(max(8, len(model_names) * 2), max(6, len(cids) * 0.35)))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        xticklabels=xlabels,
        yticklabels=ylabels,
        vmin=0.0,
        vmax=vmax,
        cbar_kws={"label": cbar_label},
    )
    plt.xlabel("Model")
    plt.ylabel("Cluster")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_cluster_error_plots(
    bundle: CrossModelBundle,
    assignments: Dict[int, int],
    cfg: Dict[str, Any],
    out_dir: Path,
) -> None:
    summary = load_cluster_summary(cfg)
    cids, mean_fp, mean_fn, mass = _aggregate_fp_fn(bundle, assignments)
    if not cids:
        return

    _plot_heatmap(
        mean_fp,
        cids,
        summary,
        bundle.model_names,
        "Mean FP per document by cluster (xlm_r_base excluded)",
        "Mean FP",
        out_dir / "cluster_mean_fp.png",
    )
    _plot_heatmap(
        mean_fn,
        cids,
        summary,
        bundle.model_names,
        "Mean FN per document by cluster (xlm_r_base excluded)",
        "Mean FN",
        out_dir / "cluster_mean_fn.png",
    )
    mmax = float(mass.max()) if mass.size else 1.0
    _plot_heatmap(
        mass,
        cids,
        summary,
        bundle.model_names,
        "Total FP+FN mass by cluster (sum over docs; xlm_r_base excluded)",
        "Sum FP+FN",
        out_dir / "cluster_total_fp_fn_mass.png",
        vmax=mmax if mmax > 0 else None,
    )
