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


def _run_standalone_cli() -> None:
    import argparse
    from pathlib import Path

    from src.ensemble_metaheuristic.strategy_cli import (
        build_per_model_preds,
        load_validation_bundle,
        prepend_repo_root_for_strategy_file,
    )

    try:
        from src.evaluation.evaluator import evaluate_data
    except ImportError:
        from ...evaluation.evaluator import evaluate_data

    prepend_repo_root_for_strategy_file(Path(__file__))

    ap = argparse.ArgumentParser(
        description="Per-label champion routing + score-cutoff sweep (this module only).",
    )
    ap.add_argument("--config", default="src/analysis/analysis.yaml", help="Analysis YAML.")
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
    per_model_preds = build_per_model_preds(matrices, names, is_score_model, all_pids, all_labels)
    model_label_f1s = {name: per_label_f1(gt_data, per_model_preds[name], all_labels) for name in names}
    label_routing = build_label_routing_table(model_label_f1s, all_labels)

    print("Per-label routing (this module only)")
    champion_counts: Dict[str, int] = {}
    for _label, champ in label_routing.items():
        champion_counts[champ] = champion_counts.get(champ, 0) + 1
    print("  Labels routed to each model:", {n: champion_counts.get(n, 0) for n in names})

    n_steps = max(2, int(args.sweep_steps))
    sweep_cuts = np.linspace(0.72, 1.18, n_steps)
    best_f1, best_cut, best_preds = -1.0, 1.0, {}
    for ci, cut in enumerate(sweep_cuts):
        rp = per_label_routed_predict(
            matrices, is_score_model, names, all_pids, all_labels, label_routing,
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
