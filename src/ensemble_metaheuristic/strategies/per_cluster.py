"""Per-cluster champion: one base model per document cluster (val micro-F1)."""
from __future__ import annotations

from typing import Dict, List, Tuple

from evaluation.evaluator import evaluate_data


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
