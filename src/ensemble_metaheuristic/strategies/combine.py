"""OR / AND / k-of-n fusion of flat prediction dicts from other strategies."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List


def merge_preds_union(
    a: Dict[int, List[str]],
    b: Dict[int, List[str]],
    all_pids: List[int],
) -> Dict[int, List[str]]:
    return {pid: sorted(set(a.get(pid, [])) | set(b.get(pid, []))) for pid in all_pids}


def merge_preds_intersection(
    a: Dict[int, List[str]],
    b: Dict[int, List[str]],
    all_pids: List[int],
) -> Dict[int, List[str]]:
    return {pid: sorted(set(a.get(pid, [])) & set(b.get(pid, []))) for pid in all_pids}


def merge_preds_k_of_n(
    pred_list: List[Dict[int, List[str]]],
    all_pids: List[int],
    k: int,
) -> Dict[int, List[str]]:
    """Predict a label if it appears in at least ``k`` of the strategy outputs (per document)."""
    if not pred_list:
        return {pid: [] for pid in all_pids}
    k = min(max(int(k), 1), len(pred_list))
    out: Dict[int, List[str]] = {}
    for pid in all_pids:
        cnt: Counter[str] = Counter()
        for d in pred_list:
            for lab in d.get(pid, []):
                cnt[lab] += 1
        out[pid] = sorted(lab for lab, c in cnt.items() if c >= k)
    return out


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
        description="Demo: OR / AND / k-of-n fusion on flat preds (this module only; toy example).",
    )
    ap.add_argument("--config", default="src/analysis/analysis.yaml", help="Analysis YAML.")
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
