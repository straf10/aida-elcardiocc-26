"""Strategy: weighted ensemble — random search + hill climb over weights and thresholds."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from src.evaluation.evaluator import evaluate_data
except ImportError:
    from ..evaluation.evaluator import evaluate_data


def weighted_ensemble_combined_matrix(
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    weights: np.ndarray,
    model_thresholds: np.ndarray,
) -> np.ndarray:
    """Weighted sum of binary votes (score ≥ per-model threshold). Shape (n_docs, n_labels)."""
    n_docs = matrices[0].shape[0]
    n_labels = matrices[0].shape[1]
    combined = np.zeros((n_docs, n_labels), dtype=np.float64)
    mt_iter = iter(model_thresholds)
    for mat, is_score, w in zip(matrices, is_score_model, weights):
        t = next(mt_iter) if is_score else 0.5
        combined += float(w) * (mat >= t).astype(np.float64)
    return combined


def weighted_ensemble_predict(
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    weights: np.ndarray,
    model_thresholds: np.ndarray,
    global_threshold: float,
    all_pids: List[int],
    all_labels: List[str],
) -> Dict[int, List[str]]:
    """Weighted vote ensemble → flat predictions (same rule as ``score_ensemble``)."""
    combined = weighted_ensemble_combined_matrix(matrices, is_score_model, weights, model_thresholds)
    return {
        pid: [all_labels[j] for j in np.where(combined[i] >= global_threshold)[0]]
        for i, pid in enumerate(all_pids)
    }


def weighted_ensemble_predict_gated_secondary(
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    names: Sequence[str],
    weights: np.ndarray,
    model_thresholds: np.ndarray,
    global_threshold: float,
    all_pids: List[int],
    all_labels: List[str],
    *,
    secondary_names: Tuple[str, ...] = ("information_retrieval", "ner_el"),
    gate_max_base: float,
) -> Dict[int, List[str]]:
    """
    Use IR/NER votes only when the doc looks uncertain on the primary stack:
    max combined score without secondary models is below ``gate_max_base``.
    """
    name_to_i = {n: i for i, n in enumerate(names)}
    sec_idxs = [name_to_i[n] for n in secondary_names if n in name_to_i]
    w_base = weights.astype(np.float64).copy()
    for i in sec_idxs:
        w_base[i] = 0.0
    combined_base = weighted_ensemble_combined_matrix(matrices, is_score_model, w_base, model_thresholds)
    combined_full = weighted_ensemble_combined_matrix(matrices, is_score_model, weights, model_thresholds)
    pred_data: Dict[int, List[str]] = {}
    for i, pid in enumerate(all_pids):
        row = combined_full[i] if float(combined_base[i].max()) < gate_max_base else combined_base[i]
        pred_data[pid] = [all_labels[j] for j in np.where(row >= global_threshold)[0]]
    return pred_data


def weighted_ensemble_predict_top_k(
    combined: np.ndarray,
    global_threshold: float,
    all_pids: List[int],
    all_labels: List[str],
    top_k: int,
) -> Dict[int, List[str]]:
    """Keep at most ``top_k`` labels per doc among those with combined ≥ ``global_threshold`` (by score)."""
    k = max(1, int(top_k))
    pred_data: Dict[int, List[str]] = {}
    for i, pid in enumerate(all_pids):
        row = combined[i]
        cand = np.where(row >= global_threshold)[0]
        if cand.size <= k:
            pred_data[pid] = [all_labels[int(j)] for j in cand]
        else:
            sub = row[cand]
            order = np.argsort(-sub)[:k]
            picked = cand[order]
            pred_data[pid] = [all_labels[int(j)] for j in picked]
    return pred_data


def weighted_ensemble_predict_frequency_buckets(
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    weights: np.ndarray,
    model_thresholds: np.ndarray,
    base_global_threshold: float,
    all_pids: List[int],
    all_labels: List[str],
    label_support: Dict[str, int],
    *,
    support_cutoff: int,
    rare_factor: float,
    freq_factor: float,
) -> Dict[int, List[str]]:
    """Per-label global threshold = base × (rare_factor if low support else freq_factor)."""
    combined = weighted_ensemble_combined_matrix(matrices, is_score_model, weights, model_thresholds)
    thr = np.array(
        [
            base_global_threshold * (
                rare_factor if label_support.get(lbl, 0) < support_cutoff else freq_factor
            )
            for lbl in all_labels
        ],
        dtype=np.float64,
    )
    return {
        pid: [all_labels[int(j)] for j in np.where(combined[i] >= thr)[0]]
        for i, pid in enumerate(all_pids)
    }


def weighted_ensemble_predict_two_threshold(
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    weights: np.ndarray,
    model_thresholds: np.ndarray,
    all_pids: List[int],
    all_labels: List[str],
    *,
    t_high: float,
    t_low: float,
    min_votes: int,
) -> Dict[int, List[str]]:
    """
    Positive if combined ≥ t_high OR (combined ≥ t_low AND at least ``min_votes`` models vote yes).
    """
    combined = weighted_ensemble_combined_matrix(matrices, is_score_model, weights, model_thresholds)
    n_docs, n_labels = combined.shape
    votes = np.zeros((n_docs, n_labels), dtype=np.int32)
    mt_iter = iter(model_thresholds)
    for mat, is_score in zip(matrices, is_score_model):
        t = next(mt_iter) if is_score else 0.5
        votes += (mat >= t).astype(np.int32)
    mv = max(1, int(min_votes))
    mask = (combined >= t_high) | ((combined >= t_low) & (votes >= mv))
    return {
        pid: [all_labels[j] for j in np.where(mask[i])[0]]
        for i, pid in enumerate(all_pids)
    }


def score_ensemble(
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    weights: np.ndarray,
    model_thresholds: np.ndarray,
    global_threshold: float,
    gt_data: Dict,
    all_pids: List[int],
    all_labels: List[str],
) -> float:
    """Weighted vote ensemble: each model fires when its score >= its threshold.
    Predicts positive when the weighted vote sum >= global_threshold.
    Binary models (IR, NER) vote with their 0/1 values directly.
    """
    pred_data = weighted_ensemble_predict(
        matrices, is_score_model, weights, model_thresholds, global_threshold, all_pids, all_labels,
    )
    return evaluate_data(gt_data, pred_data, label_space=all_labels)["micro_f1"]


def _sample_params(
    n_models: int,
    n_score_models: int,
    rng: np.random.RandomState,
    *,
    perturb_from: Tuple[np.ndarray, np.ndarray, float] | None = None,
    step: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if perturb_from is None:
        weights = rng.uniform(0.05, 1.0, size=n_models).astype(np.float32)
        model_thresholds = rng.uniform(0.6, 1.4, size=n_score_models).astype(np.float32)
        global_threshold = float(rng.uniform(0.05, 1.2))
    else:
        pw, pt, pg = perturb_from
        weights = np.clip(pw + rng.uniform(-step, step, size=n_models), 0.01, 2.0).astype(np.float32)
        model_thresholds = np.clip(pt + rng.uniform(-step, step, size=n_score_models), 0.1, 3.0).astype(np.float32)
        global_threshold = float(np.clip(pg + rng.uniform(-step, step), 0.01, 2.5))
    return weights, model_thresholds, global_threshold


def random_search(
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    gt_data: Dict,
    all_pids: List[int],
    all_labels: List[str],
    n_iter: int,
    rng: np.random.RandomState,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    n_score = sum(is_score_model)
    best_f1 = -1.0
    best_w = np.ones(len(matrices), dtype=np.float32) / len(matrices)
    best_mt = np.ones(n_score, dtype=np.float32)
    best_gt = 0.5

    for i in range(n_iter):
        if verbose and i % 200 == 0:
            print(f"  random search {i}/{n_iter}  best={best_f1:.4f}", flush=True)
        w, mt, gt = _sample_params(len(matrices), n_score, rng)
        f1 = score_ensemble(matrices, is_score_model, w, mt, gt, gt_data, all_pids, all_labels)
        if f1 > best_f1:
            best_f1, best_w, best_mt, best_gt = f1, w.copy(), mt.copy(), gt

    return best_w, best_mt, best_gt, best_f1


def hill_climb(
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    gt_data: Dict,
    all_pids: List[int],
    all_labels: List[str],
    start_w: np.ndarray,
    start_mt: np.ndarray,
    start_gt: float,
    n_iter: int,
    rng: np.random.RandomState,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    best_w, best_mt, best_gt = start_w.copy(), start_mt.copy(), start_gt
    best_f1 = score_ensemble(matrices, is_score_model, best_w, best_mt, best_gt, gt_data, all_pids, all_labels)
    step = 0.08

    for i in range(n_iter):
        if verbose and i % 200 == 0:
            print(f"  hill climb  {i}/{n_iter}  best={best_f1:.4f}", flush=True)
        w, mt, gt = _sample_params(
            len(matrices), len(best_mt), rng,
            perturb_from=(best_w, best_mt, best_gt), step=step,
        )
        f1 = score_ensemble(matrices, is_score_model, w, mt, gt, gt_data, all_pids, all_labels)
        if f1 > best_f1:
            best_f1, best_w, best_mt, best_gt = f1, w.copy(), mt.copy(), gt

    return best_w, best_mt, best_gt, best_f1


def run_search(
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    gt_data: Dict,
    pids: List[int],
    all_labels: List[str],
    n_iter: int,
    rng: np.random.RandomState,
    *,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Random search (75%) + hill climb (25%) over weights, per-model thresholds, global threshold."""
    n_random = int(n_iter * 0.75)
    n_climb = n_iter - n_random
    best_w, best_mt, best_gt, best_f1 = random_search(
        matrices, is_score_model, gt_data, pids, all_labels, n_random, rng, verbose=verbose,
    )
    best_w, best_mt, best_gt, best_f1 = hill_climb(
        matrices, is_score_model, gt_data, pids, all_labels,
        best_w, best_mt, best_gt, n_climb, rng, verbose=verbose,
    )
    return best_w, best_mt, best_gt, best_f1
