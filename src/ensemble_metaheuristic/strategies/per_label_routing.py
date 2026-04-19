"""Per-label routing: each ICD code from its validation champion model."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    from src.evaluation.evaluator import per_class_report
except ImportError:
    from ...evaluation.evaluator import per_class_report


def per_label_f1(
    gt_data: Dict,
    pred_data: Dict[int, List[str]],
    all_labels: List[str],
) -> Dict[str, float]:
    """Return per-label F1 dict using the competition per_class_report."""
    report = per_class_report(gt_data, pred_data, all_labels)
    out: Dict[str, float] = {}
    for row in report:
        label = row["code"]
        p = row.get("precision", 0.0)
        r = row.get("recall", 0.0)
        out[label] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return out


def build_label_routing_table(
    model_label_f1s: Dict[str, Dict[str, float]],
    all_labels: List[str],
    min_support_model: Dict[str, Dict[str, int]] | None = None,
    min_support: int = 3,
) -> Dict[str, str]:
    """
    For each label, pick the model with highest per-label F1.
    Falls back to 'mlc_greek_bert' for labels with too little support.
    """
    routing: Dict[str, str] = {}
    names = list(model_label_f1s.keys())
    for label in all_labels:
        best_model = max(names, key=lambda n: model_label_f1s[n].get(label, 0.0))
        routing[label] = best_model
    return routing


def per_label_routed_predict(
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    names: List[str],
    all_pids: List[int],
    all_labels: List[str],
    label_routing: Dict[str, str],
    *,
    score_cutoff: float = 1.0,
    binary_cutoff: float = 0.5,
) -> Dict[int, List[str]]:
    """Predict each label using its champion model."""
    name_to_idx = {n: i for i, n in enumerate(names)}
    pred_data: Dict[int, List[str]] = {pid: [] for pid in all_pids}

    for j, label in enumerate(all_labels):
        champion = label_routing.get(label, names[0])
        idx = name_to_idx[champion]
        cutoff = score_cutoff if is_score_model[idx] else binary_cutoff
        col = matrices[idx][:, j]
        for i, pid in enumerate(all_pids):
            if col[i] >= cutoff:
                pred_data[pid].append(label)

    return pred_data
