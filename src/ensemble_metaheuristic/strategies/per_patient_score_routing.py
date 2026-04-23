"""Per-patient routing: pick one base model per patient from scores only (no labels)."""
from __future__ import annotations

from typing import Dict, List, Literal

import numpy as np

Policy = Literal["mean", "max", "l2"]


def per_patient_champion_from_scores(
    matrices: List[np.ndarray],
    names: List[str],
    all_pids: List[int],
    *,
    policy: Policy = "mean",
) -> Dict[int, str]:
    """
    For each patient, assign the model name with the highest heuristic score on that row.

    Policies (all use only the score matrices, not ground truth):

    - ``mean``: highest mean value across labels for that patient.
    - ``max``: highest max value across labels.
    - ``l2``: largest L2 norm of the score vector (overall activation strength).
    """
    if len(matrices) != len(names):
        raise ValueError("matrices and names length mismatch")
    rout: Dict[int, str] = {}
    for i, pid in enumerate(all_pids):
        if policy == "mean":
            keyed = [(float(np.mean(mat[i])), n) for n, mat in zip(names, matrices)]
        elif policy == "max":
            keyed = [(float(np.max(mat[i])), n) for n, mat in zip(names, matrices)]
        elif policy == "l2":
            keyed = [(float(np.linalg.norm(np.asarray(mat[i], dtype=np.float64))), n) for n, mat in zip(names, matrices)]
        else:
            raise ValueError(f"unknown policy: {policy!r}")
        rout[pid] = max(keyed, key=lambda x: x[0])[1]
    return rout


def per_patient_routed_predict(
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    names: List[str],
    all_pids: List[int],
    all_labels: List[str],
    patient_routing: Dict[int, str],
    *,
    score_cutoff: float = 1.0,
    binary_cutoff: float = 0.5,
) -> Dict[int, List[str]]:
    """Take all labels from the chosen model's row for each patient (same cutoffs as per-label routing)."""
    name_to_idx = {n: i for i, n in enumerate(names)}
    pred_data: Dict[int, List[str]] = {pid: [] for pid in all_pids}
    for i, pid in enumerate(all_pids):
        champ = patient_routing.get(pid, names[0])
        idx = name_to_idx[champ]
        cutoff = score_cutoff if is_score_model[idx] else binary_cutoff
        row = matrices[idx][i, :]
        for j, label in enumerate(all_labels):
            if float(row[j]) >= cutoff:
                pred_data[pid].append(label)
    return pred_data


def _run_standalone_cli() -> None:
    import argparse
    from pathlib import Path

    from ensemble_metaheuristic.strategy_loaders import (
        load_validation_bundle,
        prepend_repo_root_for_strategy_file,
    )

    from evaluation.evaluator import evaluate_data

    prepend_repo_root_for_strategy_file(Path(__file__))

    ap = argparse.ArgumentParser(
        description="Per-patient model choice from scores only + optional cutoff sweep (this module only).",
    )
    ap.add_argument("--config", default="src/evaluation/config.yaml", help="Evaluation YAML (config.yaml).")
    ap.add_argument(
        "--policy",
        choices=("mean", "max", "l2"),
        default="mean",
        help="Heuristic to compare models per patient.",
    )
    ap.add_argument(
        "--sweep-steps",
        type=int,
        default=24,
        help="Number of score-cutoff values between ~0.72 and ~1.18.",
    )
    args = ap.parse_args()

    matrices, names, is_score_model, gt_data, all_pids, all_labels, _mc, _vp = load_validation_bundle(
        args.config,
    )
    routing = per_patient_champion_from_scores(
        matrices, names, all_pids, policy=args.policy,
    )

    print("Per-patient score routing (this module only)")
    counts: Dict[str, int] = {}
    for _pid, m in routing.items():
        counts[m] = counts.get(m, 0) + 1
    print(f"  policy={args.policy}  patients per champion model:", {n: counts.get(n, 0) for n in names})

    n_steps = max(2, int(args.sweep_steps))
    sweep_cuts = np.linspace(0.72, 1.18, n_steps)
    best_f1, best_cut, best_preds = -1.0, 1.0, {}
    for ci, cut in enumerate(sweep_cuts):
        rp = per_patient_routed_predict(
            matrices,
            is_score_model,
            names,
            all_pids,
            all_labels,
            routing,
            score_cutoff=float(cut),
        )
        rf = evaluate_data(gt_data, rp, label_space=all_labels)["micro_f1"]
        if rf > best_f1:
            best_f1, best_cut, best_preds = rf, float(cut), rp
        if (ci + 1) % max(1, n_steps // 8) == 0:
            print(f"    … step {ci + 1}/{n_steps}  best_micro_f1={best_f1:.4f} @ cut={best_cut:.4f}")

    m = evaluate_data(gt_data, best_preds, label_space=all_labels)
    print(f"  Best score-cutoff={best_cut:.4f}")
    print(
        f"  micro-F1={m['micro_f1']:.4f}  precision={m['precision']:.4f}  recall={m['recall']:.4f}",
    )


if __name__ == "__main__":
    _run_standalone_cli()
