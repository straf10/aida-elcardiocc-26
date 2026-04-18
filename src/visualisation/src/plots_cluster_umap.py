from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

try:
    import umap

    HAS_UMAP = True
except ImportError:
    from sklearn.decomposition import PCA

    HAS_UMAP = False

from src.visualisation.src.cluster_context import load_embeddings_if_aligned, validation_jsonl_pid_order
from src.visualisation.src.config import MODEL_ABBREV
from src.visualisation.src.cross_model_data import CrossModelBundle


def _fp_fn_by_pid(bundle: CrossModelBundle, model_name: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for row in bundle.metrics_by_model[model_name].get("doc_breakdown", []):
        pid = int(row["patient_id"])
        out[pid] = int(row.get("fp", 0)) + int(row.get("fn", 0))
    return out


def run_cluster_umap_grid(
    bundle: CrossModelBundle,
    cfg: Dict[str, Any],
    out_dir: Path,
) -> None:
    """2x2 UMAP/PCA of embeddings; each panel coloured by fp+fn for one model."""
    pid_order = validation_jsonl_pid_order(cfg)
    emb = load_embeddings_if_aligned(cfg, len(pid_order))
    if emb is None:
        return

    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
        title_prefix = "UMAP"
    else:
        reducer = PCA(n_components=2, random_state=42)
        title_prefix = "PCA"

    proj = reducer.fit_transform(emb)
    names = bundle.model_names[:4]
    if len(names) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()
    for ax, name in zip(axes_flat, names):
        err = _fp_fn_by_pid(bundle, name)
        colors = np.array([err.get(pid, 0) for pid in pid_order], dtype=float)
        sc = ax.scatter(
            proj[:, 0],
            proj[:, 1],
            c=colors,
            cmap="viridis",
            alpha=0.65,
            s=12,
        )
        plt.colorbar(sc, ax=ax, label="FP+FN")
        ax.set_title(f"{MODEL_ABBREV.get(name, name)} — {title_prefix}")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")

    for k in range(len(names), 4):
        axes_flat[k].set_visible(False)

    fig.suptitle(
        f"{title_prefix} of discharge embeddings (val JSONL order); colour = FP+FN per doc\n"
        "xlm_r_base excluded",
        fontsize=12,
    )
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "cluster_umap_error_grid.png", dpi=300, bbox_inches="tight")
    plt.close()
