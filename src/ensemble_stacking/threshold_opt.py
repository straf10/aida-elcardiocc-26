"""Threshold optimisation for stacking probability matrices.

Two strategies are provided:

* **Global** — a single threshold ``t`` applied to all labels.  The sweep
  evaluates the project's official group-level micro-F1 at each candidate
  value and picks the best.

* **Per-label** — one threshold per label, chosen to maximise each label's
  *binary* per-code F1 independently (fast vectorised sweep).  Because the
  scoring function is group-level, the resulting thresholds are an
  approximation, but in practice they are close to the optimum and much
  faster to compute than an exact joint search.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from evaluation.scoring import evaluate_data


def proba_to_preds(
    proba: np.ndarray,
    all_pids: List[int],
    all_labels: List[str],
    threshold: float,
) -> Dict[int, List[str]]:
    """Convert a probability matrix to ``{pid: [label, ...]}`` using a global threshold."""
    preds: Dict[int, List[str]] = {}
    for i, pid in enumerate(all_pids):
        preds[pid] = [all_labels[j] for j in np.where(proba[i] >= threshold)[0]]
    return preds


def proba_to_preds_per_label(
    proba: np.ndarray,
    all_pids: List[int],
    all_labels: List[str],
    thresholds: np.ndarray,
) -> Dict[int, List[str]]:
    """Convert a probability matrix to ``{pid: [label, ...]}`` using per-label thresholds."""
    preds: Dict[int, List[str]] = {}
    for i, pid in enumerate(all_pids):
        preds[pid] = [all_labels[j] for j in np.where(proba[i] >= thresholds)[0]]
    return preds


def sweep_global_threshold(
    proba: np.ndarray,
    gt_data: Dict,
    all_pids: List[int],
    all_labels: List[str],
    *,
    n_steps: int = 100,
    t_min: float = 0.05,
    t_max: float = 0.95,
) -> Tuple[float, float, Dict[int, List[str]]]:
    """Sweep a single global threshold and return ``(best_t, best_micro_f1, best_preds)``.

    Evaluation uses the official group-level micro-F1 (``evaluate_data``).
    """
    best_t, best_f1, best_preds = 0.5, -1.0, {}
    for t in np.linspace(t_min, t_max, n_steps):
        preds = proba_to_preds(proba, all_pids, all_labels, float(t))
        f1 = evaluate_data(gt_data, preds, label_space=all_labels)["micro_f1"]
        if f1 > best_f1:
            best_t, best_f1, best_preds = float(t), f1, preds
    return best_t, best_f1, best_preds


def sweep_per_label_thresholds(
    proba: np.ndarray,
    gt_data: Dict,
    all_pids: List[int],
    all_labels: List[str],
    *,
    n_steps: int = 50,
    t_min: float = 0.05,
    t_max: float = 0.95,
) -> Tuple[np.ndarray, float, Dict[int, List[str]]]:
    """Find a per-label threshold by maximising each label's binary per-code F1.

    The sweep is fully vectorised across threshold candidates for each label.
    After computing all per-label thresholds, the final predictions are scored
    with the official group-level micro-F1 via ``evaluate_data``.

    Returns
    -------
    thresholds : np.ndarray of shape (n_labels,)
    micro_f1   : float — official group-level micro-F1 of the final predictions
    preds      : Dict[int, List[str]]
    """
    label_to_idx = {lbl: j for j, lbl in enumerate(all_labels)}
    n_labels = len(all_labels)
    n_val = len(all_pids)

    # Build binary gold matrix (n_val, n_labels) — simple per-code occurrence.
    Y_gold = np.zeros((n_val, n_labels), dtype=bool)
    for i, pid in enumerate(all_pids):
        for grp in gt_data.get(pid, []):
            for code in grp:
                j = label_to_idx.get(code)
                if j is not None:
                    Y_gold[i, j] = True

    candidates = np.linspace(t_min, t_max, n_steps, dtype=np.float32)
    thresholds = np.full(n_labels, 0.5, dtype=np.float32)

    for j in range(n_labels):
        col = proba[:, j]            # (n_val,)
        gold_j = Y_gold[:, j]       # (n_val,) bool

        # Vectorise over all threshold candidates at once.
        preds_all = col[None, :] >= candidates[:, None]   # (n_steps, n_val)
        tp_all = (preds_all & gold_j[None, :]).sum(axis=1).astype(np.float32)
        fp_all = (preds_all & ~gold_j[None, :]).sum(axis=1).astype(np.float32)
        fn_all = (~preds_all & gold_j[None, :]).sum(axis=1).astype(np.float32)
        denom = 2.0 * tp_all + fp_all + fn_all
        f1_all = np.where(denom > 0, 2.0 * tp_all / denom, 0.0)
        thresholds[j] = float(candidates[int(f1_all.argmax())])

    best_preds = proba_to_preds_per_label(proba, all_pids, all_labels, thresholds)
    m = evaluate_data(gt_data, best_preds, label_space=all_labels)
    return thresholds, float(m["micro_f1"]), best_preds
