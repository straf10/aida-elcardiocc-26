"""
Alternative ensemble strategies beyond weighted voting.

per_label_routing   — each label is predicted by its per-label champion model
per_cluster_champion — each document uses the best base model on its cluster (val micro-F1)
correction_mode     — start from greek_bert, then add/remove based on other models
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

try:
    from src.evaluation.evaluator import evaluate_data, per_class_report
except ImportError:
    from ..evaluation.evaluator import evaluate_data, per_class_report


# ---------------------------------------------------------------------------
# Strategy 1: per-label routing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-cluster champion (document-level: one base model per cluster)
# ---------------------------------------------------------------------------


def build_cluster_champion_routing(
    cluster_assignments: Dict[int, int],
    all_pids: List[int],
    names: List[str],
    per_model_preds: Dict[str, Dict[int, List[str]]],
    gt_data: Dict,
    all_labels: List[str],
    default_model: str = "mlc_greek_bert",
) -> Tuple[Dict[int, str], Dict[int, float]]:
    """
    For each cluster id that appears on at least one validation patient, pick the
    base model with highest micro-F1 on patients in that cluster (same preds as per-label matrices).
    """
    cluster_ids = sorted({cluster_assignments[p] for p in all_pids if p in cluster_assignments})
    if not cluster_ids:
        return {}, {}

    routing: Dict[int, str] = {}
    scores: Dict[int, float] = {}
    for cid in cluster_ids:
        pids_in = [p for p in all_pids if cluster_assignments.get(p) == cid]
        if not pids_in:
            continue
        best_name, best_f1 = default_model, -1.0
        for name in names:
            sub_gt = {p: gt_data[p] for p in pids_in if p in gt_data}
            sub_pred = {p: per_model_preds[name].get(p, []) for p in pids_in if p in gt_data}
            f1 = evaluate_data(sub_gt, sub_pred, label_space=all_labels)["micro_f1"]
            if f1 > best_f1:
                best_f1, best_name = f1, name
        routing[cid] = best_name
        scores[cid] = best_f1
    return routing, scores


def per_cluster_champion_predict(
    cluster_assignments: Dict[int, int],
    all_pids: List[int],
    cluster_routing: Dict[int, str],
    per_model_preds: Dict[str, Dict[int, List[str]]],
    default_model: str = "mlc_greek_bert",
) -> Dict[int, List[str]]:
    """For each patient, take flat predictions from the champion model of that patient's cluster."""
    pred_data: Dict[int, List[str]] = {}
    for pid in all_pids:
        cid = cluster_assignments.get(pid)
        if cid is None:
            champ = default_model
        else:
            champ = cluster_routing.get(cid, default_model)
        pred_data[pid] = list(per_model_preds[champ].get(pid, []))
    return pred_data


# ---------------------------------------------------------------------------
# Strategy 2: correction mode
# ---------------------------------------------------------------------------

def correction_predict(
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    names: List[str],
    all_pids: List[int],
    all_labels: List[str],
    base_model: str = "mlc_greek_bert",
    *,
    add_min_votes: int = 2,        # add a code if ≥N other models predict it
    add_min_score_factor: float = 1.0,  # and score models must be >= this × threshold
    remove_if_zero_votes: bool = False, # remove base codes all others reject
) -> Dict[int, List[str]]:
    """
    Start with base_model predictions (caller often passes best individual by val F1), then:
    - ADD codes that ≥ add_min_votes other models confidently predict
    - Optionally REMOVE base codes that no other model supports
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    base_idx = name_to_idx[base_model]
    base_cutoff = 1.0 if is_score_model[base_idx] else 0.5
    other_idxs = [i for i, n in enumerate(names) if n != base_model]

    pred_data: Dict[int, List[str]] = {}

    for row_i, pid in enumerate(all_pids):
        base_set = set(
            all_labels[j] for j in np.where(matrices[base_idx][row_i] >= base_cutoff)[0]
        )

        for j, label in enumerate(all_labels):
            other_votes = 0
            for idx in other_idxs:
                cutoff = add_min_score_factor if is_score_model[idx] else 0.5
                if matrices[idx][row_i, j] >= cutoff:
                    other_votes += 1

            if label not in base_set and other_votes >= add_min_votes:
                base_set.add(label)
            elif label in base_set and remove_if_zero_votes and other_votes == 0:
                base_set.discard(label)

        pred_data[pid] = sorted(base_set)

    return pred_data


def search_correction_params(
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    names: List[str],
    all_pids: List[int],
    all_labels: List[str],
    gt_data: Dict,
    base_model: str = "mlc_greek_bert",
    *,
    extended: bool = False,
) -> Tuple[dict, float]:
    """Grid search over correction mode params, return best config and F1."""
    best_f1 = -1.0
    best_cfg: dict = {}

    if extended:
        add_votes_list = [1, 2, 3, 4, 5]
        add_factor_list = [round(x, 2) for x in np.linspace(0.55, 1.65, 13)]
    else:
        add_votes_list = [1, 2, 3]
        add_factor_list = [0.8, 1.0, 1.2, 1.5]

    n_evals = 0
    for add_votes in add_votes_list:
        for add_factor in add_factor_list:
            for remove in [False, True]:
                preds = correction_predict(
                    matrices, is_score_model, names, all_pids, all_labels,
                    base_model=base_model,
                    add_min_votes=add_votes,
                    add_min_score_factor=add_factor,
                    remove_if_zero_votes=remove,
                )
                f1 = evaluate_data(gt_data, preds, label_space=all_labels)["micro_f1"]
                n_evals += 1
                if f1 > best_f1:
                    best_f1 = f1
                    best_cfg = {
                        "add_min_votes": add_votes,
                        "add_min_score_factor": float(add_factor),
                        "remove_if_zero_votes": remove,
                    }

    best_cfg["_grid_evaluations"] = n_evals
    return best_cfg, best_f1
