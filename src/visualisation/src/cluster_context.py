from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from src.analysis.common import clustering_output_dir
    from src.evaluation.config_utils import get_cfg, load_config
    from src.preprocessing.io_utils import load_jsonl
except ImportError:
    from analysis.common import clustering_output_dir  # type: ignore
    from evaluation.config_utils import get_cfg, load_config  # type: ignore
    from preprocessing.io_utils import load_jsonl  # type: ignore


def load_cluster_assignments(cfg: Dict[str, Any]) -> Dict[int, int]:
    """pid -> cluster_id from clustering output."""
    path = clustering_output_dir(cfg) / "cluster_assignments.json"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def load_cluster_summary(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = clustering_output_dir(cfg) / "cluster_summary.json"
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def cluster_ytick_label(cluster_id: int, summary: List[Dict[str, Any]], max_terms: int = 2) -> str:
    for row in summary:
        if int(row.get("cluster_id", -1)) == int(cluster_id):
            terms = row.get("top_terms") or []
            tail = ", ".join(str(t) for t in terms[:max_terms])
            return f"C{cluster_id}: {tail}" if tail else f"C{cluster_id}"
    return f"C{cluster_id}"


def validation_jsonl_pid_order(cfg: Dict[str, Any]) -> List[int]:
    """Patient IDs in JSONL row order (same order as clustering embeddings)."""
    val_path = get_cfg(cfg, "data.val_path")
    records = load_jsonl(val_path)
    return [int(r["patient_id"]) for r in records]


def load_embeddings_if_aligned(cfg: Dict[str, Any], expected_n: int) -> Optional[np.ndarray]:
    """Load embeddings.npy if present and row count matches validation JSONL length."""
    cdir = clustering_output_dir(cfg)
    cache = Path(
        get_cfg(cfg, "clustering.embeddings_cache", str(cdir / "embeddings.npy"))
    )
    if not cache.is_file():
        return None
    emb = np.load(cache)
    if emb.shape[0] != expected_n:
        return None
    return emb


def cluster_context_paths(cfg: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    cdir = clustering_output_dir(cfg)
    emb = Path(get_cfg(cfg, "clustering.embeddings_cache", str(cdir / "embeddings.npy")))
    return cdir / "cluster_assignments.json", cdir / "cluster_summary.json", emb
