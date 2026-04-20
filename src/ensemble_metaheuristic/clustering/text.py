"""
Per-cluster champion using **document text embeddings** (transformer mean pooling).

Clustering + champion routing follow the train-only recipe from
``run_train_routing_champion_sweep_from_features``: scaler and clusterer fit on train
embeddings; champion per cluster from train labels; validation rows assigned to clusters;
metrics on validation predictions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    from src.evaluation.config_utils import get_cfg
    from src.ensemble_metaheuristic.clustering.embeddings import embed_texts
except ImportError:
    from ...evaluation.config_utils import get_cfg
    from .embeddings import embed_texts

from .score_matrix import run_train_routing_champion_sweep_from_features


def texts_in_pid_order(jsonl_path: str, ordered_pids: List[int]) -> List[str]:
    """Load JSONL and return ``text`` strings aligned to ``ordered_pids`` (missing → empty)."""
    try:
        from src.preprocessing.io_utils import load_jsonl, resolve_patient_id
    except ImportError:
        from ...preprocessing.io_utils import load_jsonl, resolve_patient_id

    records = load_jsonl(jsonl_path)
    pid_to_text: Dict[int, str] = {}
    for rec in records:
        pid = resolve_patient_id(rec)
        pid_to_text[pid] = str(rec.get("text", "") or "")
    return [pid_to_text.get(int(pid), "") for pid in ordered_pids]


def _cache_key_model(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")[:160]


def _load_or_embed_split(
    texts: List[str],
    cache_file: Path,
    model_name: str,
    max_length: int,
    batch_size: int,
    device,
) -> np.ndarray | None:
    if cache_file.is_file():
        arr = np.load(cache_file)
        if arr.shape[0] == len(texts) and arr.ndim == 2:
            print(f"  Loaded text embeddings cache ({arr.shape[0]} × {arr.shape[1]}) {cache_file.name}")
            return arr.astype(np.float64, copy=False)
    if not texts:
        return None
    print(
        f"  Computing text embeddings ({len(texts)} docs, device={device!s}) → {cache_file.name} …",
    )
    emb = embed_texts(texts, model_name, max_length, batch_size, device)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, emb)
    return emb.astype(np.float64, copy=False)


def run_text_embedding_cluster_sweep_train_routing(
    cfg: Dict[str, Any],
    train_jsonl_path: str,
    val_jsonl_path: str,
    train_pids: List[int],
    val_pids: List[int],
    names: List[str],
    per_train_preds: Dict[str, Dict[int, List[str]]],
    per_val_preds: Dict[str, Dict[int, List[str]]],
    train_gt: Dict[Any, Any],
    val_gt: Dict[Any, Any],
    all_labels: List[str],
    k_list: Sequence[int],
    methods: Sequence[str],
    random_state: int,
    *,
    cache_dir: Path | None = None,
) -> List[Tuple[str, int, Dict]]:
    """
    Embed train/val report texts with the clustering model from config; run the same
    train-routing champion sweep as score-matrix strategy 2t.
    """
    try:
        import torch
    except ImportError:
        print("  Skipped (PyTorch not installed).")
        return []

    model_name = str(get_cfg(cfg, "clustering.model_name", "nlpaueb/bert-base-greek-uncased-v1"))
    max_length = int(get_cfg(cfg, "clustering.max_length", 256))
    batch_size = int(get_cfg(cfg, "clustering.batch_size", 16))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = cache_dir or Path("outputs/ensemble_metaheuristic/text_cluster_cache")
    key = _cache_key_model(model_name)
    tr_path = Path(train_jsonl_path)
    va_path = Path(val_jsonl_path)
    cache_tr = base / f"{key}_train_n{len(train_pids)}.npy"
    cache_va = base / f"{key}_val_n{len(val_pids)}.npy"

    texts_tr = texts_in_pid_order(str(tr_path), train_pids)
    texts_va = texts_in_pid_order(str(va_path), val_pids)

    print(
        f"  model={model_name!r}  max_length={max_length}  batch_size={batch_size}  cache={base}",
    )
    raw_tr = _load_or_embed_split(texts_tr, cache_tr, model_name, max_length, batch_size, device)
    raw_va = _load_or_embed_split(texts_va, cache_va, model_name, max_length, batch_size, device)
    if raw_tr is None or raw_va is None:
        return []

    return run_train_routing_champion_sweep_from_features(
        raw_tr,
        raw_va,
        train_pids,
        val_pids,
        names,
        per_train_preds,
        per_val_preds,
        train_gt,
        val_gt,
        all_labels,
        k_list,
        methods,
        random_state,
        log_prefix="[text-train]",
    )
