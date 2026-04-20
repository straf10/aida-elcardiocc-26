"""Per-patient routing: kNN in score-feature space on train patients → vote best model (train labels)."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np

try:
    from src.evaluation.evaluator import evaluate_data
except ImportError:
    from ...evaluation.evaluator import evaluate_data

from .per_patient_score_routing import per_patient_routed_predict


def best_model_for_single_patient(
    pid: int,
    gt_data: Dict,
    per_model_preds: Dict[str, Dict[int, List[str]]],
    names: List[str],
    all_labels: List[str],
) -> str:
    """Which model has highest micro-F1 on this patient alone (subset evaluation)."""
    if pid not in gt_data:
        return names[0]
    sub_gt = {pid: gt_data[pid]}
    best_n, best_f1 = names[0], -1.0
    for name in names:
        sub_pred = {pid: per_model_preds[name].get(pid, [])}
        f1 = evaluate_data(sub_gt, sub_pred, label_space=all_labels)["micro_f1"]
        if f1 > best_f1:
            best_f1, best_n = f1, name
    return best_n


def build_patient_routing_knn_train(
    train_matrices: List[np.ndarray],
    val_matrices: List[np.ndarray],
    train_gt: Dict,
    train_pids: List[int],
    val_pids: List[int],
    names: List[str],
    all_labels: List[str],
    per_model_train_preds: Dict[str, Dict[int, List[str]]],
    *,
    k: int = 11,
) -> Dict[int, str]:
    """
    Scale train features, fit kNN on train rows, for each val patient vote champion model
    on each neighbor (using **train** ground truth only), majority vote → assigned model name.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    from ..clustering.score_matrix import clustering_features_from_matrices

    train_X = clustering_features_from_matrices(train_matrices)
    val_X = clustering_features_from_matrices(val_matrices)
    if train_X.shape[0] != len(train_pids) or val_X.shape[0] != len(val_pids):
        return {}

    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_X.astype(np.float64, copy=False))
    val_s = scaler.transform(val_X.astype(np.float64, copy=False))

    n_train = train_s.shape[0]
    kk = max(1, min(int(k), n_train))
    nn = NearestNeighbors(n_neighbors=kk, metric="euclidean", algorithm="auto")
    nn.fit(train_s)
    _, neigh_idx = nn.kneighbors(val_s)

    rout: Dict[int, str] = {}
    for vi, pid in enumerate(val_pids):
        votes: List[str] = []
        for j in range(neigh_idx.shape[1]):
            ti = int(neigh_idx[vi, j])
            tpid = train_pids[ti]
            votes.append(
                best_model_for_single_patient(
                    tpid, train_gt, per_model_train_preds, names, all_labels,
                ),
            )
        rout[pid] = Counter(votes).most_common(1)[0][0]
    return rout


def _run_standalone_cli() -> None:
    import argparse
    from pathlib import Path

    from src.ensemble_metaheuristic.strategy_cli import (
        build_per_model_preds,
        load_train_matrices,
        load_validation_bundle,
        prepend_repo_root_for_strategy_file,
    )

    prepend_repo_root_for_strategy_file(Path(__file__))

    ap = argparse.ArgumentParser(
        description="Per-patient kNN on train score features + train-label champion vote (this module only).",
    )
    ap.add_argument("--config", default="src/analysis/analysis.yaml", help="Analysis YAML.")
    ap.add_argument("--k", type=int, default=11, help="Number of train neighbors per val patient.")
    ap.add_argument("--sweep-steps", type=int, default=24, help="Score-cutoff sweep steps.")
    args = ap.parse_args()

    val_matrices, names, is_score_model, val_gt, val_pids, all_labels, model_cfgs, _vp = load_validation_bundle(
        args.config,
    )
    train_gt, train_pids, train_matrices, train_path = load_train_matrices(
        args.config,
        model_cfgs,
        all_labels,
    )

    per_train = build_per_model_preds(train_matrices, names, is_score_model, train_pids, all_labels)
    routing = build_patient_routing_knn_train(
        train_matrices,
        val_matrices,
        train_gt,
        train_pids,
        val_pids,
        names,
        all_labels,
        per_train,
        k=int(args.k),
    )

    print("Per-patient kNN train routing (this module only)")
    print(f"  train_path={train_path}  n_train={len(train_pids)}  k={args.k}")
    counts: Dict[str, int] = {}
    for _pid, m in routing.items():
        counts[m] = counts.get(m, 0) + 1
    print("  patients per champion model:", {n: counts.get(n, 0) for n in names})

    n_steps = max(2, int(args.sweep_steps))
    sweep_cuts = np.linspace(0.72, 1.18, n_steps)
    best_f1, best_cut, best_preds = -1.0, 1.0, {}
    for ci, cut in enumerate(sweep_cuts):
        rp = per_patient_routed_predict(
            val_matrices,
            is_score_model,
            names,
            val_pids,
            all_labels,
            routing,
            score_cutoff=float(cut),
        )
        rf = evaluate_data(val_gt, rp, label_space=all_labels)["micro_f1"]
        if rf > best_f1:
            best_f1, best_cut, best_preds = rf, float(cut), rp
        if (ci + 1) % max(1, n_steps // 8) == 0:
            print(f"    … step {ci + 1}/{n_steps}  best_micro_f1={best_f1:.4f} @ cut={best_cut:.4f}")

    m = evaluate_data(val_gt, best_preds, label_space=all_labels)
    print(f"  Best score-cutoff={best_cut:.4f}")
    print(
        f"  micro-F1={m['micro_f1']:.4f}  precision={m['precision']:.4f}  recall={m['recall']:.4f}",
    )


if __name__ == "__main__":
    _run_standalone_cli()
