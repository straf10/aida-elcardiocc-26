from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

try:
    from .config_utils import get_cfg, load_config
    from .evaluator import evaluate_file
    from .io_utils import average_pred_codes_per_doc, load_predictions
except ImportError:
    from config_utils import get_cfg, load_config
    from evaluator import evaluate_file
    from io_utils import average_pred_codes_per_doc, load_predictions


def _load_labels(path: str | None) -> List[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [str(item) for item in data]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leaderboard for prediction files.")
    parser.add_argument("--config", help="Optional YAML config path")
    parser.add_argument("--ground-truth", dest="ground_truth", help="Path to ground-truth JSONL file")
    parser.add_argument("--pred-dir", help="Directory containing prediction JSONL files")
    parser.add_argument("--labels", help="Optional JSON list of labels for macro/per-class metrics")
    parser.add_argument(
        "--per-class",
        action="store_true",
        help="Print top underperforming classes for the best system",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ground_truth_path = args.ground_truth or get_cfg(config, "ground_truth_path")
    pred_dir = args.pred_dir or get_cfg(config, "predictions_dir")
    labels_path = args.labels or get_cfg(config, "label_names_path")
    label_space = _load_labels(labels_path)

    if not ground_truth_path or not pred_dir:
        raise ValueError("ground truth path and predictions directory are required.")

    pred_files = glob.glob(os.path.join(pred_dir, "*.jsonl"))
    if not pred_files:
        print(f"No .jsonl files found in {pred_dir}")
        return

    results = []
    for pred_file in pred_files:
        filename = os.path.basename(pred_file)
        try:
            metrics = evaluate_file(ground_truth_path, pred_file, label_space=label_space or None)
            pred_data = load_predictions(pred_file)
            avg_preds = average_pred_codes_per_doc(pred_data)
            results.append(
                {
                    "system": filename,
                    "f1": metrics["micro_f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "macro_f1": metrics.get("macro_f1_present_labels", 0.0),
                    "avg_preds": avg_preds,
                    "metrics": metrics,
                }
            )
        except Exception as exc:
            print(f"Error evaluating {filename}: {exc}")

    results.sort(key=lambda row: row["f1"], reverse=True)

    print(
        f"{'System':<30} | {'MicroF1':<7} | {'MacroF1':<7} | {'Prec':<6} | "
        f"{'Rec':<6} | {'AvgPreds/Doc'}"
    )
    print("-" * 95)
    for row in results:
        print(
            f"{row['system']:<30} | {row['f1']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['avg_preds']:.2f}"
        )

    if args.per_class and label_space and results:
        best = results[0]
        per_class = best["metrics"].get("per_class", [])
        if per_class:
            underperforming = sorted(per_class, key=lambda x: (x["f1"], x["support"]))[:15]
            print("\nWorst 15 classes by F1 (best system):")
            print(f"System: {best['system']}")
            print(f"{'Code':<10} {'F1':<8} {'Support':<8} {'Precision':<10} {'Recall':<10} {'FP'}")
            for row in underperforming:
                print(
                    f"{row['code']:<10} {row['f1']:.4f}   {row['support']:<8} "
                    f"{row['precision']:.4f}     {row['recall']:.4f}     {row['fp_count']}"
                )


if __name__ == "__main__":
    main()
