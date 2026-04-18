import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.decomposition import PCA

try:
    from .common import LONG_TAIL_BUCKET_ORDER
except ImportError:
    from src.analysis.common import LONG_TAIL_BUCKET_ORDER


def plot_confusion_heatmap(wrong_pairs_counter: Dict[str, int], top_n: int, out_path: Path):
    """Plot confusion heatmap of top-N codes."""
    if not wrong_pairs_counter:
        return
        
    code_counts = Counter()
    parsed_pairs = {}
    for pair_str, count in wrong_pairs_counter.items():
        p, t = pair_str.split("|")
        code_counts[p] += count
        code_counts[t] += count
        parsed_pairs[(p, t)] = count
        
    top_codes = [code for code, _ in code_counts.most_common(top_n)]
    
    if not top_codes:
        return
        
    matrix = np.zeros((len(top_codes), len(top_codes)))
    code_to_idx = {c: i for i, c in enumerate(top_codes)}
    
    for (p, t), count in parsed_pairs.items():
        if p in code_to_idx and t in code_to_idx:
            matrix[code_to_idx[t], code_to_idx[p]] = count
            
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=top_codes, yticklabels=top_codes)
    plt.xlabel("Predicted Code")
    plt.ylabel("True Code (Missed)")
    plt.title(f"Top {top_n} Most Confused Codes")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_cluster_map(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    error_flags: List[str] = None,
    out_path: Path = None
):
    """Plot 2D projection of clusters."""
    if embeddings is None or embeddings.shape[0] == 0:
        return
        
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2, random_state=42)
        
    proj = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    
    if error_flags is not None:
        # Plot by error type
        unique_flags = list(set(error_flags))
        colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(unique_flags)))
        
        for flag, color in zip(unique_flags, colors):
            mask = np.array([f == flag for f in error_flags])
            plt.scatter(proj[mask, 0], proj[mask, 1], label=flag, 
                        color=color, alpha=0.6, s=15)
        plt.legend(title="Status")
    else:
        # Plot by cluster
        scatter = plt.scatter(proj[:, 0], proj[:, 1], c=cluster_labels, 
                              cmap="tab20", alpha=0.6, s=15)
        plt.colorbar(scatter, label="Cluster ID")
        
    title = "UMAP Projection" if HAS_UMAP else "PCA Projection"
    plt.title(f"{title} of Medical Reports")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
    plt.close()


def plot_long_tail(buckets_metrics: Dict[str, dict], out_path: Path):
    """Plot bar chart Macro/Weighted F1 per frequency bucket."""
    if not buckets_metrics:
        return
        
    labels = list(buckets_metrics.keys())
    macro_f1 = [buckets_metrics[b]["macro_f1"] for b in labels]
    weighted_f1 = [buckets_metrics[b]["weighted_f1"] for b in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, macro_f1, width, label='Macro F1')
    ax.bar(x + width/2, weighted_f1, width, label='Weighted F1')
    
    ax.set_ylabel('F1 Score')
    ax.set_title('Performance by Label Frequency Bucket')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_models_long_tail_comparison(
    bucket_comparison: Dict[str, Dict[str, dict]],
    out_path: Path,
    model_order: Optional[List[str]] = None,
) -> None:
    """Grouped bar chart: Macro / Weighted F1 per frequency bucket across all models."""
    if not bucket_comparison:
        return

    buckets = [b for b in LONG_TAIL_BUCKET_ORDER if bucket_comparison.get(b)]
    if not buckets:
        return

    if model_order is None:
        seen: List[str] = []
        for b in buckets:
            for m in bucket_comparison[b]:
                if m not in seen:
                    seen.append(m)
        models = seen
    else:
        models = [m for m in model_order if any(m in bucket_comparison[b] for b in buckets)]
    if not models:
        return

    n_b = len(buckets)
    n_m = len(models)
    x = np.arange(n_b)
    width = min(0.8 / max(n_m, 1), 0.12)

    fig, (ax_macro, ax_weighted) = plt.subplots(
        2,
        1,
        figsize=(max(12, n_b * 3 + n_m), 10),
        constrained_layout=True,
    )

    for mi, model in enumerate(models):
        offset = (mi - (n_m - 1) / 2.0) * width
        macro_h = [
            float(bucket_comparison[b].get(model, {}).get("macro_f1", 0.0))
            for b in buckets
        ]
        w_h = [
            float(bucket_comparison[b].get(model, {}).get("weighted_f1", 0.0))
            for b in buckets
        ]
        ax_macro.bar(x + offset, macro_h, width, label=model)
        ax_weighted.bar(x + offset, w_h, width, label=model)

    for ax, title_suffix in (
        (ax_macro, "Macro F1"),
        (ax_weighted, "Weighted F1"),
    ):
        ax.set_ylabel("F1 Score")
        ax.set_title(f"Long-tail buckets — {title_suffix}")
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, rotation=0)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(n_m, 4))
        ax.set_ylim(0.0, 1.05)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
