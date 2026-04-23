from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .config_utils import get_cfg, load_config
from .scoring import micro_f1, score_document
from .io_utils import load_ground_truth


def _micro_f1_from_preds(
    gt_items: List[Tuple[int, List[List[str]]]],
    pred_data: Dict[int, List[str]],
) -> float:
    """Group-level micro-F1 only (no per-doc breakdown or per-class report)."""
    total_tp = total_fp = total_fn = 0
    for patient_id, ground_truth_groups in gt_items:
        pred_codes = pred_data.get(patient_id, [])
        tp, fp, fn = score_document(ground_truth_groups, pred_codes)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    _, _, f1 = micro_f1(total_tp, total_fp, total_fn)
    return f1


def evaluate_thresholds(
    scores: np.ndarray,
    patient_ids: List[int],
    ground_truth_data: Dict[int, List[List[str]]],
    thresholds: np.ndarray,
    label_names: List[str],
    gt_items: Optional[List[Tuple[int, List[List[str]]]]] = None,
) -> float:
    if gt_items is None:
        gt_items = list(ground_truth_data.items())
    preds_bin = scores >= thresholds
    pred_data: Dict[int, List[str]] = {}

    for i, pid in enumerate(patient_ids):
        pred_indices = np.where(preds_bin[i])[0]
        pred_data[int(pid)] = [label_names[int(idx)] for idx in pred_indices]

    return _micro_f1_from_preds(gt_items, pred_data)


def count_label_positives_per_class(
    ground_truth_data: Dict[int, List[List[str]]], label_names: List[str]
) -> np.ndarray:
    """
    Count, per class index, how many patients have that label in any inner group
    (a label is positive for a doc if it appears in at least one list cell).
    """
    label2idx = {l: i for i, l in enumerate(label_names)}
    counts = np.zeros(len(label_names), dtype=np.int32)
    for _pid, groups in ground_truth_data.items():
        present: Set[str] = set()
        for group in groups:
            for code in group:
                if code in label2idx:
                    present.add(code)
        for code in present:
            counts[label2idx[code]] += 1
    return counts


def tune_thresholds(
    scores: np.ndarray,
    patient_ids: List[int],
    ground_truth_data: Dict[int, List[List[str]]],
    label_names: List[str],
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    passes: int = 1,
    min_pos_count: int = 0,
    min_class_gain: float = 0.0,
) -> Tuple[np.ndarray, float]:
    num_classes = scores.shape[1]
    sweep_values = np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step)
    gt_items = list(ground_truth_data.items())
    pos_counts = count_label_positives_per_class(ground_truth_data, label_names)
    n_skip_pos = 0
    if min_pos_count > 0:
        n_skip_pos = int(np.sum(pos_counts < min_pos_count))
        print(
            f"Per-class min_pos_count={min_pos_count}: {n_skip_pos}/{num_classes} "
            "classes are frozen to global t (pos count below threshold on val)."
        )

    print("Starting global threshold search...")
    best_global_t = 0.5
    best_global_f1 = 0.0

    for t in sweep_values:
        global_thresh = np.full(num_classes, t)
        f1 = evaluate_thresholds(
            scores,
            patient_ids,
            ground_truth_data,
            global_thresh,
            label_names,
            gt_items=gt_items,
        )
        if f1 > best_global_f1:
            best_global_f1 = f1
            best_global_t = t

    print(f"Best global threshold: {best_global_t:.3f} (F1: {best_global_f1:.4f})")

    print("Starting per-class greedy search...")
    best_thresholds = np.full(num_classes, best_global_t)
    current_best_f1 = best_global_f1

    n_passes = max(1, int(passes))
    for _round in range(n_passes):
        f1_before_round = current_best_f1
        for class_idx in range(num_classes):
            if min_pos_count > 0 and pos_counts[class_idx] < min_pos_count:
                continue
            best_class_t = best_thresholds[class_idx]
            best_class_f1 = current_best_f1

            for t in sweep_values:
                test_thresholds = best_thresholds.copy()
                test_thresholds[class_idx] = t
                f1 = evaluate_thresholds(
                    scores,
                    patient_ids,
                    ground_truth_data,
                    test_thresholds,
                    label_names,
                    gt_items=gt_items,
                )
                if f1 > best_class_f1:
                    best_class_f1 = f1
                    best_class_t = t

            gain = best_class_f1 - current_best_f1
            if min_class_gain <= 0.0:
                accept = best_class_f1 > current_best_f1 + 1e-12
            else:
                accept = gain >= min_class_gain and best_class_f1 > current_best_f1 - 1e-12
            if accept:
                best_thresholds[class_idx] = best_class_t
                current_best_f1 = best_class_f1

        if n_passes > 1 and current_best_f1 <= f1_before_round:
            break

    print(f"Final tuned F1: {current_best_f1:.4f}")
    changed = int(np.sum(np.abs(best_thresholds - best_global_t) > 1e-9))
    print(
        f"Per-class thresholds tuned: {changed}/{num_classes} classes changed from global"
    )
    return best_thresholds, current_best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune thresholds for MLC outputs.")
    parser.add_argument("--config", help="Optional YAML config path")
    parser.add_argument("--scores", help="Path to .npy file containing sigmoid scores (N_docs, 115)")
    parser.add_argument("--pids", help="Path to JSON list of patient_ids matching scores rows")
    parser.add_argument("--labels", help="Path to JSON list of label names")
    parser.add_argument("--ground-truth", dest="ground_truth", help="Path to ground-truth JSONL file")
    parser.add_argument("--out", help="Output JSON file for best thresholds")
    parser.add_argument("--min-threshold", type=float, help="Sweep minimum threshold")
    parser.add_argument("--max-threshold", type=float, help="Sweep maximum threshold")
    parser.add_argument("--step-threshold", type=float, help="Sweep step threshold")
    parser.add_argument(
        "--passes",
        type=int,
        help="Max coordinate-descent rounds after global search (default from config or 1)",
    )
    parser.add_argument(
        "--min-pos-count",
        type=int,
        help="Do not per-class-tune classes with fewer than this many val positives; "
        "use global t for them (default: threshold_tuning.min_pos_count or 0).",
    )
    parser.add_argument(
        "--min-class-gain",
        type=float,
        help="Min micro-F1 gain over current best to accept a per-class threshold update "
        "(default: threshold_tuning.min_class_gain or 0).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    scores_path = args.scores or get_cfg(config, "scores_path")
    pids_path = args.pids or get_cfg(config, "patient_ids_path")
    labels_path = args.labels or get_cfg(config, "label_names_path")
    ground_truth_path = args.ground_truth or get_cfg(config, "ground_truth_path")
    out_path = args.out or get_cfg(
        config, "output.thresholds_path", "outputs/models/xlm_r_large/thresholds.json"
    )

    threshold_min = (
        args.min_threshold
        if args.min_threshold is not None
        else get_cfg(config, "threshold_tuning.min", 0.05)
    )
    threshold_max = (
        args.max_threshold
        if args.max_threshold is not None
        else get_cfg(config, "threshold_tuning.max", 0.95)
    )
    threshold_step = (
        args.step_threshold
        if args.step_threshold is not None
        else get_cfg(config, "threshold_tuning.step", 0.01)
    )
    tune_passes = (
        args.passes
        if args.passes is not None
        else get_cfg(config, "threshold_tuning.passes", 1)
    )
    min_pos_count = (
        args.min_pos_count
        if args.min_pos_count is not None
        else int(get_cfg(config, "threshold_tuning.min_pos_count", 0))
    )
    min_class_gain = (
        args.min_class_gain
        if args.min_class_gain is not None
        else float(get_cfg(config, "threshold_tuning.min_class_gain", 0.0))
    )

    if not all([scores_path, pids_path, labels_path, ground_truth_path]):
        raise ValueError("Missing required inputs. Provide them via CLI flags or config file.")

    scores = np.load(scores_path)
    with open(pids_path, "r", encoding="utf-8") as handle:
        patient_ids = [int(x) for x in json.load(handle)]
    with open(labels_path, "r", encoding="utf-8") as handle:
        label_names = [str(x) for x in json.load(handle)]
    ground_truth_data = load_ground_truth(ground_truth_path)

    best_thresholds, best_f1 = tune_thresholds(
        scores=scores,
        patient_ids=patient_ids,
        ground_truth_data=ground_truth_data,
        label_names=label_names,
        threshold_min=float(threshold_min),
        threshold_max=float(threshold_max),
        threshold_step=float(threshold_step),
        passes=int(tune_passes),
        min_pos_count=min_pos_count,
        min_class_gain=float(min_class_gain),
    )

    out_dict = {
        "best_micro_f1": float(best_f1),
        "thresholds": {label: float(th) for label, th in zip(label_names, best_thresholds)},
        "sweep": {"min": threshold_min, "max": threshold_max, "step": threshold_step},
        "tuning": {
            "min_pos_count": int(min_pos_count),
            "min_class_gain": float(min_class_gain),
        },
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out_dict, handle, indent=2)

    print(f"Thresholds saved to {out_path}")
