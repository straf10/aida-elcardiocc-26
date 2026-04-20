"""Average multi-seed val logits/scores and tune thresholds (Track C)."""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np

from evaluation.config_utils import get_cfg, load_config
from evaluation.io_utils import load_ground_truth
from evaluation.threshold_tune import tune_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average sigmoid scores from multiple seed runs and tune thresholds."
    )
    parser.add_argument("--config", help="Optional YAML with paths (see ensemble block)")
    parser.add_argument(
        "--scores",
        nargs="+",
        required=True,
        help="Paths to val_scores.npy (same shape, same row order per seed)",
    )
    parser.add_argument(
        "--pids",
        help="Path to val_patient_ids.json (shared across seeds)",
    )
    parser.add_argument(
        "--labels",
        help="Path to label_names.json",
    )
    parser.add_argument(
        "--ground-truth",
        dest="ground_truth",
        help="Validation JSONL for group micro-F1",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/experiments/xlm_r_large/ensemble",
        help="Directory for ensemble_scores.npy and ensemble_thresholds.json",
    )
    args = parser.parse_args()

    config = load_config(args.config) if args.config else {}

    pids_path = args.pids or get_cfg(config, "ensemble.patient_ids_path")
    labels_path = args.labels or get_cfg(config, "ensemble.label_names_path")
    gt_path = args.ground_truth or get_cfg(config, "ensemble.ground_truth_path")

    if not all([pids_path, labels_path, gt_path]):
        raise ValueError("Provide --pids, --labels, --ground-truth (or config ensemble.*).")

    stacks: List[np.ndarray] = []
    for path in args.scores:
        arr = np.load(path)
        stacks.append(arr)
    if not stacks:
        raise ValueError("No score arrays were loaded from --scores.")
    if not all(s.shape == stacks[0].shape for s in stacks):
        raise ValueError("Seed score shape mismatch across --scores inputs.")
    # Training exports `val_scores.npy` as aggregated sigmoid probabilities per patient.
    mean_scores = np.mean(np.stack(stacks, axis=0), axis=0)

    os.makedirs(args.out_dir, exist_ok=True)
    scores_out = os.path.join(args.out_dir, "ensemble_scores.npy")
    np.save(scores_out, mean_scores)

    with open(pids_path, "r", encoding="utf-8") as f:
        patient_ids = [int(x) for x in json.load(f)]
    with open(labels_path, "r", encoding="utf-8") as f:
        label_names = [str(x) for x in json.load(f)]

    ground_truth_data = load_ground_truth(gt_path)

    threshold_min = float(get_cfg(config, "threshold_tuning.min", 0.05))
    threshold_max = float(get_cfg(config, "threshold_tuning.max", 0.95))
    threshold_step = float(get_cfg(config, "threshold_tuning.step", 0.01))

    tuned_thresholds, tuned_f1 = tune_thresholds(
        scores=mean_scores,
        patient_ids=patient_ids,
        ground_truth_data=ground_truth_data,
        label_names=label_names,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
    )

    out_json = os.path.join(args.out_dir, "ensemble_thresholds.json")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "tuned_val_micro_f1_in_sample": float(tuned_f1),
                "note": "Optimistic if tuned on same val as scores; use Section 8 workflow for honest estimates.",
                "thresholds": {
                    lab: float(th) for lab, th in zip(label_names, tuned_thresholds)
                },
                "sweep": {
                    "min": threshold_min,
                    "max": threshold_max,
                    "step": threshold_step,
                },
                "source_scores": list(args.scores),
            },
            handle,
            indent=2,
        )
    print(f"Ensemble scores saved to {scores_out}")
    print(f"Ensemble thresholds saved to {out_json} | micro-F1={tuned_f1:.4f}")


if __name__ == "__main__":
    main()
