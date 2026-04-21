"""Shim: label-set fusion lives in ``pred_merge_*`` modules (kept for ``python -m`` / old imports)."""
from __future__ import annotations

from .pred_merge_intersection import merge_preds_intersection
from .pred_merge_k_of_n import merge_preds_k_of_n
from .pred_merge_union import merge_preds_union

__all__ = [
    "merge_preds_intersection",
    "merge_preds_k_of_n",
    "merge_preds_union",
]


def _run_standalone_cli() -> None:
    import argparse
    from pathlib import Path

    from ensemble_metaheuristic.strategy_cli import (
        build_per_model_preds,
        load_validation_bundle,
        prepend_repo_root_for_strategy_file,
    )

    from evaluation.evaluator import evaluate_data

    prepend_repo_root_for_strategy_file(Path(__file__))

    ap = argparse.ArgumentParser(
        description="Demo: OR / AND / k-of-n fusion on flat preds (this module only; toy example).",
    )
    ap.add_argument("--config", default="src/evaluation/config.yaml", help="Evaluation YAML (config.yaml).")
    ap.add_argument(
        "--mode",
        choices=("union", "intersection", "k2of3"),
        default="union",
        help="How to merge prediction dicts.",
    )
    args = ap.parse_args()

    matrices, names, is_score_model, gt_data, all_pids, all_labels, _mc, _vp = load_validation_bundle(
        args.config,
    )
    per_model_preds = build_per_model_preds(matrices, names, is_score_model, all_pids, all_labels)

    print("Combine strategies demo (this module only)")
    if args.mode == "union":
        a, b = names[0], names[1]
        merged = merge_preds_union(per_model_preds[a], per_model_preds[b], all_pids)
        title = f"OR  {a} ∪ {b}"
    elif args.mode == "intersection":
        a, b = names[0], names[1]
        merged = merge_preds_intersection(per_model_preds[a], per_model_preds[b], all_pids)
        title = f"AND {a} ∩ {b}"
    else:
        a, b, c = names[0], names[1], names[2]
        merged = merge_preds_k_of_n(
            [per_model_preds[a], per_model_preds[b], per_model_preds[c]],
            all_pids,
            2,
        )
        title = f"2-of-3  {a}, {b}, {c}"

    m = evaluate_data(gt_data, merged, label_space=all_labels)
    print(f"  {title}")
    print(
        f"  micro-F1={m['micro_f1']:.4f}  precision={m['precision']:.4f}  recall={m['recall']:.4f}",
    )


if __name__ == "__main__":
    _run_standalone_cli()
