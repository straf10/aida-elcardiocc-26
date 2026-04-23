from __future__ import annotations

import argparse
import json
import os

import numpy as np

from preprocessing.io_utils import save_jsonl
from xlm_r_large.postprocess import apply_specific_parent_child


def _load_int_list(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as handle:
        return [int(x) for x in json.load(handle)]


def _load_str_list(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [str(x) for x in json.load(handle)]


def _resolve_threshold(args: argparse.Namespace) -> float:
    if args.threshold_float is not None:
        return float(args.threshold_float)
    with open(args.threshold_json, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if "best_t" not in meta:
        raise ValueError(f"Missing 'best_t' key in {args.threshold_json}")
    return float(meta["best_t"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build submission JSONL from probability-level ensemble scores."
    )
    parser.add_argument(
        "--scores",
        nargs="+",
        required=True,
        help="Paths to test_scores.npy or blind_scores.npy from one or more seeds.",
    )
    parser.add_argument(
        "--pids",
        required=True,
        help="Path to patient ids JSON matching row order of the score arrays.",
    )
    parser.add_argument("--labels", required=True, help="Path to label_names.json")
    parser.add_argument(
        "--threshold-json",
        help="Path to global_threshold.json (contains key best_t).",
    )
    parser.add_argument(
        "--threshold-float",
        type=float,
        help="Use explicit global threshold value instead of --threshold-json.",
    )
    parser.add_argument(
        "--apply-parent-child",
        action="store_true",
        help="Apply specific parent-child postprocess rule before writing JSONL.",
    )
    parser.add_argument("--out", required=True, help="Output submission JSONL path.")
    args = parser.parse_args()

    if args.threshold_json is None and args.threshold_float is None:
        raise ValueError("Provide either --threshold-json or --threshold-float.")

    arrays: list[np.ndarray] = []
    reference_shape: tuple[int, int] | None = None
    for score_path in args.scores:
        arr = np.load(score_path)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D score array at {score_path}, got shape {arr.shape}")
        if reference_shape is None:
            reference_shape = arr.shape
        elif arr.shape != reference_shape:
            raise ValueError(
                f"Score shape mismatch: {score_path} has {arr.shape}, expected {reference_shape}."
            )
        arrays.append(arr.astype(np.float32, copy=False))
    if not arrays:
        raise ValueError("No score arrays loaded.")

    patient_ids = _load_int_list(args.pids)
    label_names = _load_str_list(args.labels)
    if reference_shape is None:
        raise ValueError("Missing reference shape after loading score arrays.")
    if reference_shape[0] != len(patient_ids):
        raise ValueError(
            f"Rows in scores ({reference_shape[0]}) do not match patient ids ({len(patient_ids)})."
        )
    if reference_shape[1] != len(label_names):
        raise ValueError(
            f"Label count mismatch: scores have {reference_shape[1]} columns, labels file has {len(label_names)}."
        )

    threshold = _resolve_threshold(args)
    mean_scores = np.mean(np.stack(arrays, axis=0), axis=0)
    preds_bin = mean_scores >= threshold

    pred_map: dict[int, list[str]] = {}
    for row_idx, patient_id in enumerate(patient_ids):
        label_indices = np.where(preds_bin[row_idx])[0]
        pred_map[int(patient_id)] = [label_names[int(idx)] for idx in label_indices]

    if args.apply_parent_child:
        pred_map = apply_specific_parent_child(pred_map)

    submission_records = []
    for patient_id in patient_ids:
        pred_codes = pred_map[int(patient_id)]
        submission_records.append(
            {
                "patient_id": int(patient_id),
                "document_level_annotations": [[code] for code in pred_codes],
            }
        )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_jsonl(submission_records, args.out)
    print(f"Submission saved to {args.out}")
    print(f"Global threshold used: {threshold:.4f}")
    print(f"Ensembled seeds: {len(arrays)}")


if __name__ == "__main__":
    main()
