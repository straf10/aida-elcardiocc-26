"""Validation scalar for ranking (learner, threshold_mode) in stacking exports."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from evaluation.scoring import evaluate_data
from ensemble_stacking.threshold_opt import proba_to_preds, proba_to_preds_per_label


def val_pick_score(
    pick: str,
    proba: np.ndarray,
    val_gt: Dict,
    val_pids: List[int],
    all_labels: List[str],
    threshold_mode: str,
    best_t: float,
    pl_thresholds: Optional[np.ndarray],
    val_matrices: List[np.ndarray],
    stacker: object,
) -> float:
    """Scalar used to rank (learner, threshold_mode) on validation when ``--val-pick`` ≠ micro."""
    if threshold_mode == "global":
        preds = proba_to_preds(proba, val_pids, all_labels, float(best_t))
    else:
        assert pl_thresholds is not None
        preds = proba_to_preds_per_label(proba, val_pids, all_labels, pl_thresholds)
    m = evaluate_data(val_gt, preds, label_space=all_labels)
    micro = float(m["micro_f1"])
    macro = float(m.get("macro_f1_present_labels", 0.0))
    if pick == "micro":
        return micro
    if pick == "macro_present":
        return macro

    km = getattr(stacker, "_patient_kmeans", None)
    kc = int(getattr(stacker, "_patient_cluster_active", 0))
    cluster_balanced = micro
    if km is not None and kc >= 2 and val_matrices:
        from ensemble_stacking.patient_clusters import hstack_score_matrices

        cid = km.predict(hstack_score_matrices(val_matrices))
        f1s: List[float] = []
        for c in range(kc):
            idx = np.nonzero(cid == c)[0]
            if len(idx) < 1:
                continue
            sub_pids = [val_pids[i] for i in idx]
            sub_gt = {p: val_gt[p] for p in sub_pids if p in val_gt}
            sub_pred = {p: preds.get(p, []) for p in sub_pids}
            if not sub_gt:
                continue
            f1s.append(float(evaluate_data(sub_gt, sub_pred, label_space=all_labels)["micro_f1"]))
        if f1s:
            cluster_balanced = float(min(f1s))
    if pick == "cluster_min":
        return cluster_balanced
    if pick == "composite":
        return (micro + macro + cluster_balanced) / 3.0
    return micro
