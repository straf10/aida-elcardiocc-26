from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np

try:
    from ..evaluation.config_utils import get_cfg, load_config
    from ..evaluation.evaluator import evaluate_data
    from ..evaluation.io_utils import load_ground_truth
except ImportError:
    from src.evaluation.config_utils import get_cfg, load_config
    from src.evaluation.evaluator import evaluate_data
    from src.evaluation.io_utils import load_ground_truth


def evaluate_thresholds(
    scores: np.ndarray,
    patient_ids: List[int],
    ground_truth_data: Dict[int, List[List[str]]],
    thresholds: np.ndarray,
    label_names: List[str],
) -> float:
    preds_bin = scores >= thresholds
    pred_data: Dict[int, List[str]] = {}

    for i, pid in enumerate(patient_ids):
        pred_indices = np.where(preds_bin[i])[0]
        pred_data[int(pid)] = [label_names[idx] for idx in pred_indices]

    metrics = evaluate_data(ground_truth_data, pred_data, label_space=label_names)
    return metrics["micro_f1"]


def tune_thresholds(
    scores: np.ndarray,
    patient_ids: List[int],
    ground_truth_data: Dict[int, List[List[str]]],
    label_names: List[str],
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> Tuple[np.ndarray, float]:
    num_classes = scores.shape[1]
    sweep_values = np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step)

    print("Starting global threshold search...")
    best_global_t = 0.5
    best_global_f1 = 0.0

    for t in sweep_values:
        global_thresh = np.full(num_classes, t)
        f1 = evaluate_thresholds(scores, patient_ids, ground_truth_data, global_thresh, label_names)
        if f1 > best_global_f1:
            best_global_f1 = f1
            best_global_t = t

    print(f"Best global threshold: {best_global_t:.3f} (F1: {best_global_f1:.4f})")

    print("Starting per-class greedy search...")
    best_thresholds = np.full(num_classes, best_global_t)
    current_best_f1 = best_global_f1

    for class_idx in range(num_classes):
        best_class_t = best_thresholds[class_idx]
        best_class_f1 = current_best_f1

        for t in sweep_values:
            test_thresholds = best_thresholds.copy()
            test_thresholds[class_idx] = t
            f1 = evaluate_thresholds(
                scores, patient_ids, ground_truth_data, test_thresholds, label_names
            )
            if f1 > best_class_f1:
                best_class_f1 = f1
                best_class_t = t

        if best_class_f1 > current_best_f1:
            best_thresholds[class_idx] = best_class_t
            current_best_f1 = best_class_f1

    print(f"Final tuned F1: {current_best_f1:.4f}")
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
    args = parser.parse_args()

    config = load_config(args.config)
    scores_path = args.scores or get_cfg(config, "scores_path")
    pids_path = args.pids or get_cfg(config, "patient_ids_path")
    labels_path = args.labels or get_cfg(config, "label_names_path")
    ground_truth_path = args.ground_truth or get_cfg(config, "ground_truth_path")
    out_path = args.out or get_cfg(config, "output.thresholds_path", "outputs/thresholds.json")

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
    )

    out_dict = {
        "best_micro_f1": float(best_f1),
        "thresholds": {label: float(th) for label, th in zip(label_names, best_thresholds)},
        "sweep": {"min": threshold_min, "max": threshold_max, "step": threshold_step},
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out_dict, handle, indent=2)

    print(f"Thresholds saved to {out_path}")
