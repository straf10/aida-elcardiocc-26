import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.decomposition import PCA


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


def plot_models_comparison(comparison_data: Dict[str, dict], out_path: Path):
    """Plot bar chart comparing Macro/Micro/Weighted F1 across models."""
    if not comparison_data:
        return
        
    models = list(comparison_data.keys())
    micro_group = [comparison_data[m].get("micro_f1_group", 0.0) for m in models]
    micro_flat = [comparison_data[m].get("micro_f1_flat", 0.0) for m in models]
    macro = [comparison_data[m].get("macro_f1", 0.0) for m in models]
    weighted = [comparison_data[m].get("weighted_f1", 0.0) for m in models]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2), 6))
    
    ax.bar(x - 1.5*width, micro_group, width, label='Micro F1 (Group)')
    ax.bar(x - 0.5*width, micro_flat, width, label='Micro F1 (Flat)')
    ax.bar(x + 0.5*width, macro, width, label='Macro F1')
    ax.bar(x + 1.5*width, weighted, width, label='Weighted F1')
    
    ax.set_ylabel('F1 Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=4)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

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
