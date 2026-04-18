from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    from src.analysis.common import label_support_from_gt
    from src.analysis.label_analysis import frequency_buckets, long_tail_metrics
    from src.evaluation.evaluator import evaluate_data
except ImportError:
    from analysis.common import label_support_from_gt  # type: ignore
    from analysis.label_analysis import frequency_buckets, long_tail_metrics  # type: ignore
    from evaluation.evaluator import evaluate_data  # type: ignore

from visualisation.src.cluster_context import cluster_ytick_label, load_cluster_summary
from visualisation.src.config import MODEL_ABBREV
from visualisation.src.cross_model_data import CrossModelBundle


def _pids_in_cluster(assignments: Dict[int, int], cluster_id: int) -> List[int]:
    return [pid for pid, cid in assignments.items() if cid == cluster_id]


def run_cluster_tail_plot(
    bundle: CrossModelBundle,
    assignments: Dict[int, int],
    cfg: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Rare-bucket macro F1 per (cluster, model); buckets from full-val support."""
    if not assignments:
        return

    support = label_support_from_gt(bundle.gt_data, bundle.label_names)
    buckets = frequency_buckets(support, cfg)
    cluster_ids = sorted({cid for cid in assignments.values()})
    summary = load_cluster_summary(cfg)

    mat = np.zeros((len(cluster_ids), len(bundle.model_names)))
    for j, name in enumerate(bundle.model_names):
        preds = bundle.pred_by_model[name]
        for i, cid in enumerate(cluster_ids):
            pids = _pids_in_cluster(assignments, cid)
            sub_gt = {pid: bundle.gt_data[pid] for pid in pids if pid in bundle.gt_data}
            sub_pred = {pid: preds.get(pid, []) for pid in pids}
            metrics = evaluate_data(sub_gt, sub_pred, label_space=bundle.label_names)
            tail = long_tail_metrics(metrics.get("per_class", []), buckets)
            mat[i, j] = float(tail.get("rare", {}).get("macro_f1", 0.0))

    ylabels = [cluster_ytick_label(c, summary) for c in cluster_ids]
    xlabels = [MODEL_ABBREV.get(n, n) for n in bundle.model_names]

    plt.figure(figsize=(max(8, len(bundle.model_names) * 2), max(6, len(cluster_ids) * 0.35)))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        xticklabels=xlabels,
        yticklabels=ylabels,
        cbar_kws={"label": "Macro F1 (rare bucket)"},
    )
    plt.xlabel("Model")
    plt.ylabel("Cluster")
    plt.title(
        "Rare-label macro F1 by cluster (buckets from full val support; xlm_r_base excluded)"
    )
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "cluster_rare_macro_f1.png", dpi=300, bbox_inches="tight")
    plt.close()
