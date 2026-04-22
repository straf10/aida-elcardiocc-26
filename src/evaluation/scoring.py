"""Group-level ICD coding metrics (submission-style micro-F1)."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

from .io_utils import load_ground_truth, load_predictions


def score_document(ground_truth_groups: List[List[str]], pred_codes: List[str]) -> Tuple[int, int, int]:
    pred_set = set(pred_codes)
    tp = 0

    for group in ground_truth_groups:
        if pred_set.intersection(set(group)):
            tp += 1

    fn = len(ground_truth_groups) - tp
    all_ground_truth_codes = {code for group in ground_truth_groups for code in group}
    fp = len([code for code in pred_set if code not in all_ground_truth_codes])
    return tp, fp, fn


def micro_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def per_class_report(
    ground_truth_data: Dict[int, List[List[str]]],
    pred_data: Dict[int, List[str]],
    label_space: Sequence[str],
) -> List[dict]:
    support = {label: 0 for label in label_space}
    groups_hit = {label: 0 for label in label_space}
    fp_count = {label: 0 for label in label_space}

    for patient_id, ground_truth_groups in ground_truth_data.items():
        pred_set = set(pred_data.get(patient_id, []))
        all_ground_truth_codes = {c for grp in ground_truth_groups for c in grp}

        for group in ground_truth_groups:
            group_set = set(group)
            hit = bool(pred_set.intersection(group_set))
            for code in group_set:
                if code in support:
                    support[code] += 1
                    if hit:
                        groups_hit[code] += 1

        for code in pred_set:
            if code in fp_count and code not in all_ground_truth_codes:
                fp_count[code] += 1

    rows: List[dict] = []
    for code in label_space:
        s = support[code]
        gh = groups_hit[code]
        fp = fp_count[code]
        precision = gh / (gh + fp) if (gh + fp) > 0 else 0.0
        recall = gh / s if s > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "code": code,
                "support": s,
                "groups_hit": gh,
                "fp_count": fp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def evaluate_data(
    ground_truth_data: Dict[int, List[List[str]]],
    pred_data: Dict[int, List[str]],
    label_space: Sequence[str] | None = None,
) -> Dict:
    total_tp = total_fp = total_fn = 0
    doc_breakdown: List[dict] = []

    ground_truth_ids = set(ground_truth_data.keys())
    pred_ids = set(pred_data.keys())
    missing_pred_ids = sorted(ground_truth_ids - pred_ids)
    extra_pred_ids = sorted(pred_ids - ground_truth_ids)

    for patient_id, ground_truth_groups in ground_truth_data.items():
        pred_codes = pred_data.get(patient_id, [])
        pred_set = set(pred_codes)
        tp, fp, fn = score_document(ground_truth_groups, pred_codes)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        missed_groups = [
            group for group in ground_truth_groups if not pred_set.intersection(set(group))
        ]
        all_ground_truth_codes = {c for g in ground_truth_groups for c in g}
        wrong_codes = sorted([code for code in pred_set if code not in all_ground_truth_codes])

        doc_breakdown.append(
            {
                "patient_id": patient_id,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "missed_groups": missed_groups,
                "wrong_codes": wrong_codes,
            }
        )

    precision, recall, f1 = micro_f1(total_tp, total_fp, total_fn)
    result = {
        "micro_f1": f1,
        "precision": precision,
        "recall": recall,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "docs_evaluated": len(ground_truth_data),
        "missing_prediction_ids": missing_pred_ids,
        "extra_prediction_ids": extra_pred_ids,
        "doc_breakdown": doc_breakdown,
    }

    if label_space:
        class_rows = per_class_report(ground_truth_data, pred_data, label_space)
        present = [row["f1"] for row in class_rows if row["support"] > 0]
        all_rows = [row["f1"] for row in class_rows]
        result["per_class"] = class_rows
        result["macro_f1_present_labels"] = mean(present) if present else 0.0
        result["macro_f1_all_labels"] = mean(all_rows) if all_rows else 0.0

    return result


def evaluate_file(
    ground_truth_jsonl_path: str,
    pred_jsonl_path: str,
    label_space: Sequence[str] | None = None,
) -> Dict:
    """Evaluate metrics using **only** ground-truth and predictions JSONL on disk."""
    ground_truth_data = load_ground_truth(ground_truth_jsonl_path)
    pred_data = load_predictions(pred_jsonl_path)
    return evaluate_data(ground_truth_data, pred_data, label_space=label_space)


def print_metrics_short(metrics: Dict, metrics_json: str | None) -> None:
    """Compact metric print + optional JSON dump (used after IR/NER predict)."""
    print(f"Documents: {metrics['docs_evaluated']}")
    print(f"Micro-F1:  {metrics['micro_f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"TP={metrics['total_tp']} FP={metrics['total_fp']} FN={metrics['total_fn']}")
    if "macro_f1_present_labels" in metrics:
        print(f"Macro-F1 (labels with support): {metrics['macro_f1_present_labels']:.4f}")
    if metrics.get("missing_prediction_ids"):
        print(f"Missing prediction patient_ids: {len(metrics['missing_prediction_ids'])}")
    if metrics.get("extra_prediction_ids"):
        print(f"Extra prediction patient_ids:   {len(metrics['extra_prediction_ids'])}")
    if metrics_json:
        out = Path(metrics_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Wrote metrics JSON -> {out}")
