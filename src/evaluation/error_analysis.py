from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from .config_utils import get_cfg, load_config
    from .evaluator import evaluate_data
    from .io_utils import load_ground_truth
except ImportError:
    from config_utils import get_cfg, load_config
    from evaluator import evaluate_data
    from io_utils import load_ground_truth


def _load_scores_inputs(scores_path: str, pids_path: str, labels_path: str):
    scores = np.load(scores_path)
    with open(pids_path, "r", encoding="utf-8") as handle:
        patient_ids = [int(x) for x in json.load(handle)]
    with open(labels_path, "r", encoding="utf-8") as handle:
        label_names = [str(x) for x in json.load(handle)]

    if scores.shape[0] != len(patient_ids):
        raise ValueError(
            f"Rows in scores ({scores.shape[0]}) do not match patient_ids ({len(patient_ids)})."
        )
    if scores.shape[1] != len(label_names):
        raise ValueError(
            f"Score columns ({scores.shape[1]}) do not match label names ({len(label_names)})."
        )
    return scores, patient_ids, label_names


def _predictions_from_threshold(
    scores: np.ndarray,
    patient_ids: List[int],
    label_names: List[str],
    threshold: float,
) -> Dict[int, List[str]]:
    preds_bin = scores >= threshold
    pred_data = {}
    for i, pid in enumerate(patient_ids):
        pred_indices = np.where(preds_bin[i])[0]
        pred_data[int(pid)] = [label_names[idx] for idx in pred_indices]
    return pred_data


def _is_range_label(code: str) -> bool:
    return "-" in code


def _build_confusion_views(metrics: Dict):
    fp_by_label = Counter()
    fn_by_label = Counter()
    wrong_pairs = Counter()
    hard_docs = []

    for row in metrics.get("doc_breakdown", []):
        missed_groups = row.get("missed_groups", [])
        wrong_codes = row.get("wrong_codes", [])

        for code in wrong_codes:
            fp_by_label[code] += 1
        for group in missed_groups:
            for code in group:
                fn_by_label[code] += 1

        for wrong_code in wrong_codes:
            for group in missed_groups:
                for missed_code in group:
                    wrong_pairs[(wrong_code, missed_code)] += 1

        hard_docs.append(
            {
                "patient_id": row.get("patient_id"),
                "tp": row.get("tp", 0),
                "fp": row.get("fp", 0),
                "fn": row.get("fn", 0),
                "wrong_codes": wrong_codes,
                "missed_groups": missed_groups,
            }
        )

    hardest_fp_docs = sorted(hard_docs, key=lambda x: x["fp"], reverse=True)[:25]
    hardest_fn_docs = sorted(hard_docs, key=lambda x: x["fn"], reverse=True)[:25]

    return {
        "fp_by_label": fp_by_label,
        "fn_by_label": fn_by_label,
        "wrong_pairs": wrong_pairs,
        "hardest_fp_docs": hardest_fp_docs,
        "hardest_fn_docs": hardest_fn_docs,
    }


def _range_vs_specific_summary(per_class_rows: List[dict], fp_by_label: Counter, fn_by_label: Counter):
    agg = defaultdict(lambda: {"support": 0, "fp": 0, "fn": 0, "labels": 0})
    for row in per_class_rows:
        code = row["code"]
        bucket = "range" if _is_range_label(code) else "specific"
        agg[bucket]["support"] += int(row.get("support", 0))
        agg[bucket]["fp"] += int(fp_by_label.get(code, 0))
        agg[bucket]["fn"] += int(fn_by_label.get(code, 0))
        agg[bucket]["labels"] += 1
    return agg


def _write_per_class_csv(rows: List[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "code",
        "support",
        "groups_hit",
        "fp_count",
        "precision",
        "recall",
        "f1",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Error analysis for multilabel ICD-10 predictions.")
    parser.add_argument("--config", help="Optional YAML config path")
    parser.add_argument("--scores", help="Path to validation scores .npy")
    parser.add_argument("--pids", help="Path to validation patient IDs JSON")
    parser.add_argument("--labels", help="Path to label names JSON")
    parser.add_argument("--ground-truth", dest="ground_truth", help="Path to ground-truth JSONL")
    parser.add_argument("--threshold", type=float, help="Decision threshold for analysis")
    parser.add_argument("--output-json", help="Output JSON report path")
    parser.add_argument("--output-csv", help="Output CSV per-class path")
    parser.add_argument("--top-k", type=int, default=25, help="Top-K rows for worst-label summaries")
    args = parser.parse_args()

    config = load_config(args.config)
    scores_path = args.scores or get_cfg(config, "output.scores_path", "outputs/val_scores.npy")
    pids_path = args.pids or get_cfg(config, "output.patient_ids_path", "outputs/val_patient_ids.json")
    labels_path = args.labels or get_cfg(config, "output.label_names_path", "outputs/label_names.json")
    ground_truth_path = args.ground_truth or get_cfg(config, "data.val_path")
    threshold = float(args.threshold if args.threshold is not None else get_cfg(config, "training.eval_threshold", 0.3))
    output_json = Path(
        args.output_json
        or get_cfg(config, "output.error_analysis_path", "outputs/xlm_r/error_analysis.json")
    )
    output_csv = Path(
        args.output_csv
        or get_cfg(config, "output.per_class_report_path", "outputs/xlm_r/per_class_report.csv")
    )
    top_k = max(1, int(args.top_k))

    if not all([scores_path, pids_path, labels_path, ground_truth_path]):
        raise ValueError("Missing required paths. Provide via CLI or config.")

    scores, patient_ids, label_names = _load_scores_inputs(
        scores_path=scores_path,
        pids_path=pids_path,
        labels_path=labels_path,
    )
    ground_truth_data = load_ground_truth(ground_truth_path)

    pred_data = _predictions_from_threshold(
        scores=scores,
        patient_ids=patient_ids,
        label_names=label_names,
        threshold=threshold,
    )
    metrics = evaluate_data(
        ground_truth_data=ground_truth_data,
        pred_data=pred_data,
        label_space=label_names,
    )

    confusion = _build_confusion_views(metrics)
    per_class_rows = metrics.get("per_class", [])

    sorted_worst_f1 = sorted(
        per_class_rows,
        key=lambda x: (x.get("f1", 0.0), -x.get("support", 0)),
    )[:top_k]
    high_fp_labels = [
        {"code": code, "fp_count": count}
        for code, count in confusion["fp_by_label"].most_common(top_k)
    ]
    high_fn_labels = [
        {"code": code, "fn_count": count}
        for code, count in confusion["fn_by_label"].most_common(top_k)
    ]
    top_wrong_pairs = [
        {"predicted_wrong": p, "missed_true": t, "count": c}
        for (p, t), c in confusion["wrong_pairs"].most_common(top_k)
    ]

    range_specific = _range_vs_specific_summary(
        per_class_rows=per_class_rows,
        fp_by_label=confusion["fp_by_label"],
        fn_by_label=confusion["fn_by_label"],
    )

    report = {
        "threshold": threshold,
        "overall": {
            "micro_f1": metrics.get("micro_f1", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "total_tp": metrics.get("total_tp", 0),
            "total_fp": metrics.get("total_fp", 0),
            "total_fn": metrics.get("total_fn", 0),
            "docs_evaluated": metrics.get("docs_evaluated", 0),
        },
        "worst_labels_by_f1": sorted_worst_f1,
        "high_fp_labels": high_fp_labels,
        "high_fn_labels": high_fn_labels,
        "top_wrong_pred_true_pairs": top_wrong_pairs,
        "range_vs_specific": range_specific,
        "hardest_docs_fp": confusion["hardest_fp_docs"],
        "hardest_docs_fn": confusion["hardest_fn_docs"],
    }

    _write_per_class_csv(per_class_rows, output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("Error analysis complete.")
    print(f"Micro-F1 @ threshold {threshold:.2f}: {report['overall']['micro_f1']:.4f}")
    print(f"Saved JSON report: {output_json}")
    print(f"Saved per-class CSV: {output_csv}")


if __name__ == "__main__":
    main()
