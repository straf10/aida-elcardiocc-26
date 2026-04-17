from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import scipy.sparse as sp

try:
    from ..evaluation.config_utils import get_cfg
except ImportError:
    from src.evaluation.config_utils import get_cfg


def load_scores_bundle(cfg: Dict[str, Any]) -> Tuple[np.ndarray, List[int], List[str]]:
    """Load scores, patient IDs, and label names based on config."""
    scores_path = get_cfg(cfg, "data.scores_path", "outputs/val_scores.npy")
    pids_path = get_cfg(cfg, "data.pids_path", "outputs/val_patient_ids.json")
    labels_path = get_cfg(cfg, "data.label_names_path", "outputs/label_names.json")

    scores = np.load(scores_path)
    with open(pids_path, "r", encoding="utf-8") as handle:
        patient_ids = [int(x) for x in json.load(handle)]
    with open(labels_path, "r", encoding="utf-8") as handle:
        label_names = [str(x) for x in json.load(handle)]

    if scores.shape[0] != len(patient_ids):
        raise ValueError(f"Rows in scores ({scores.shape[0]}) do not match patient_ids ({len(patient_ids)}).")
    if scores.shape[1] != len(label_names):
        raise ValueError(f"Score columns ({scores.shape[1]}) do not match label names ({len(label_names)}).")
    
    return scores, patient_ids, label_names


def derive_predictions(
    scores: np.ndarray,
    patient_ids: List[int],
    label_names: List[str],
    cfg: Dict[str, Any]
) -> Dict[int, List[str]]:
    """Derive binary predictions using thresholds from config."""
    thresholds_path = get_cfg(cfg, "data.thresholds_path")
    
    if thresholds_path and Path(thresholds_path).exists():
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresh_data = json.load(f)
        thresholds_dict = thresh_data.get("thresholds", {})
        thresholds = np.array([thresholds_dict.get(l, 0.5) for l in label_names])
    else:
        scalar_thresh = get_cfg(cfg, "data.threshold", 0.5)
        thresholds = np.full(len(label_names), scalar_thresh)

    preds_bin = scores >= thresholds
    pred_data = {}
    for i, pid in enumerate(patient_ids):
        pred_indices = np.where(preds_bin[i])[0]
        pred_data[int(pid)] = [label_names[idx] for idx in pred_indices]
    
    return pred_data


def build_binary_matrices(
    gt_data: Dict[int, List[List[str]]],
    pred_data: Dict[int, List[str]],
    patient_ids: List[int],
    label_names: List[str]
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build scipy sparse CSR matrices for GT and Predictions (N_docs x 115)."""
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    n_docs = len(patient_ids)
    n_labels = len(label_names)
    
    y_true = sp.lil_matrix((n_docs, n_labels), dtype=np.int8)
    y_pred = sp.lil_matrix((n_docs, n_labels), dtype=np.int8)
    
    for i, pid in enumerate(patient_ids):
        # GT: mark 1 if the code appears in ANY group for the doc
        gt_groups = gt_data.get(pid, [])
        for group in gt_groups:
            for code in group:
                if code in label_to_idx:
                    y_true[i, label_to_idx[code]] = 1
                    
        # Preds
        preds = pred_data.get(pid, [])
        for code in preds:
            if code in label_to_idx:
                y_pred[i, label_to_idx[code]] = 1
                
    return y_true.tocsr(), y_pred.tocsr()


def label_support_from_gt(gt_data: Dict[int, List[List[str]]], label_names: List[str]) -> Counter:
    """Count groups containing each code (aligns with per_class_report semantics)."""
    support = Counter()
    for gt_groups in gt_data.values():
        for group in gt_groups:
            for code in set(group):
                if code in label_names:
                    support[code] += 1
    return support


def ensure_output_dir(cfg: Dict[str, Any]) -> Path:
    """Ensure and return the output directory."""
    out_dir = Path(get_cfg(cfg, "output.dir", "outputs/analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
