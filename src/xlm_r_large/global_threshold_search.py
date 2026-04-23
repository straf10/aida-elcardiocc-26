from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

import numpy as np

from evaluation.io_utils import load_ground_truth
from evaluation.threshold_tune import evaluate_thresholds


def _load_int_list(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as handle:
        return [int(x) for x in json.load(handle)]


def _load_str_list(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [str(x) for x in json.load(handle)]


def _ensure_same_patient_order(reference: Sequence[int], candidate: Sequence[int], path: str) -> None:
    if list(reference) != list(candidate):
        raise ValueError(
            f"Patient id ordering mismatch for {path}. All score files must share the same row order."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Global-only threshold sweep over mean val scores from multiple seed runs."
    )
    parser.add_argument(
        "--val-scores",
        nargs="+",
        required=True,
        help="Paths to val_scores.npy from one or more seed runs.",
    )
    parser.add_argument(
        "--pids",
        nargs="+",
        required=True,
        help="Paths to val_patient_ids.json (same ordering as corresponding --val-scores).",
    )
    parser.add_argument("--labels", required=True, help="Path to label_names.json")
    parser.add_argument("--ground-truth", required=True, help="Path to validation JSONL")
    parser.add_argument(
        "--out-dir",
        default="outputs/models/xlm_r_large_ensemble",
        help="Directory for ensemble_val_scores.npy and global_threshold.json",
    )
    parser.add_argument("--min", dest="t_min", type=float, default=0.40, help="Min threshold")
    parser.add_argument("--max", dest="t_max", type=float, default=0.60, help="Max threshold")
    parser.add_argument("--step", dest="t_step", type=float, default=0.01, help="Sweep step")
    args = parser.parse_args()

    if len(args.val_scores) != len(args.pids):
        raise ValueError("--val-scores and --pids must have the same number of paths.")
    if args.t_step <= 0:
        raise ValueError("--step must be positive.")
    if args.t_max < args.t_min:
        raise ValueError("--max must be >= --min.")

    score_arrays: list[np.ndarray] = []
    reference_shape: tuple[int, int] | None = None
    reference_pids: list[int] | None = None

    for idx, (score_path, pids_path) in enumerate(zip(args.val_scores, args.pids)):
        scores = np.load(score_path)
        pids = _load_int_list(pids_path)
        if scores.ndim != 2:
            raise ValueError(f"Expected 2D scores array at {score_path}, got shape {scores.shape}")
        if scores.shape[0] != len(pids):
            raise ValueError(
                f"Rows in {score_path} ({scores.shape[0]}) do not match patient ids in {pids_path} ({len(pids)})."
            )
        if reference_shape is None:
            reference_shape = scores.shape
        elif scores.shape != reference_shape:
            raise ValueError(
                f"Score shape mismatch: {score_path} has {scores.shape}, expected {reference_shape}."
            )
        if reference_pids is None:
            reference_pids = pids
        else:
            _ensure_same_patient_order(reference_pids, pids, pids_path)
        score_arrays.append(scores.astype(np.float32, copy=False))

    if not score_arrays:
        raise ValueError("No score arrays loaded.")

    label_names = _load_str_list(args.labels)
    if reference_shape is None or reference_pids is None:
        raise ValueError("Missing reference shape/pids after loading inputs.")
    if reference_shape[1] != len(label_names):
        raise ValueError(
            f"Label count mismatch: scores have {reference_shape[1]} columns, labels file has {len(label_names)} entries."
        )

    mean_scores = np.mean(np.stack(score_arrays, axis=0), axis=0)
    ground_truth_data = load_ground_truth(args.ground_truth)
    gt_items = list(ground_truth_data.items())

    sweep_values = np.arange(args.t_min, args.t_max + args.t_step / 2.0, args.t_step)
    best_t = float(sweep_values[0])
    best_f1 = -1.0
    num_labels = len(label_names)

    for t in sweep_values:
        thresholds = np.full(num_labels, float(t), dtype=np.float32)
        f1 = evaluate_thresholds(
            scores=mean_scores,
            patient_ids=reference_pids,
            ground_truth_data=ground_truth_data,
            thresholds=thresholds,
            label_names=label_names,
            gt_items=gt_items,
        )
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)

    os.makedirs(args.out_dir, exist_ok=True)
    scores_out = os.path.join(args.out_dir, "ensemble_val_scores.npy")
    meta_out = os.path.join(args.out_dir, "global_threshold.json")
    np.save(scores_out, mean_scores)
    with open(meta_out, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_t": best_t,
                "best_val_micro_f1": best_f1,
                "sweep": {"min": args.t_min, "max": args.t_max, "step": args.t_step},
                "source_val_scores": list(args.val_scores),
                "source_pid_files": list(args.pids),
                "labels_path": args.labels,
                "ground_truth_path": args.ground_truth,
            },
            handle,
            indent=2,
        )

    print(f"Saved ensemble val scores to {scores_out}")
    print(f"Best global threshold: {best_t:.4f} | micro-F1: {best_f1:.4f}")
    print(f"Saved threshold metadata to {meta_out}")


if __name__ == "__main__":
    main()
