from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

try:
    from src.evaluation.evaluator import evaluate_file as evaluate_doc_file
except ImportError:
    from ..evaluation.evaluator import evaluate_file as evaluate_doc_file

from .io_utils import load_documents


def span_f1(gold: List[Tuple[int, int]], pred: List[Tuple[int, int]]) -> Dict[str, float]:
    gold_set = set(gold)
    pred_set = set(pred)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_span_file(gold_path: str, pred_debug_path: str) -> Dict[str, float]:
    gold_docs = {d.patient_id: d for d in load_documents(gold_path)}
    pred_docs = {d.patient_id: d for d in load_documents(pred_debug_path)}

    all_gold = []
    all_pred = []

    for pid, gdoc in gold_docs.items():
        pdoc = pred_docs.get(pid)
        g_spans = [(m.start, m.end) for m in gdoc.mention_level_annotations]
        p_spans = []
        if pdoc is not None:
            p_spans = [(m.start, m.end) for m in pdoc.mention_level_annotations]
        all_gold.extend([(pid, s, e) for s, e in g_spans])
        all_pred.extend([(pid, s, e) for s, e in p_spans])

    gold_flat = [(pid, s * 10_000_000 + e) for pid, s, e in all_gold]
    pred_flat = [(pid, s * 10_000_000 + e) for pid, s, e in all_pred]
    return span_f1(gold_flat, pred_flat)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NER span quality and document micro-F1.")
    parser.add_argument("--ground-truth", required=True, help="Gold JSONL with mention annotations")
    parser.add_argument("--pred-doc", required=True, help="Predicted document-level JSONL")
    parser.add_argument("--pred-debug", required=True, help="Predicted debug JSONL with mention spans")
    args = parser.parse_args()

    span_metrics = evaluate_span_file(args.ground_truth, args.pred_debug)
    doc_metrics = evaluate_doc_file(args.ground_truth, args.pred_doc)

    print("Span-level NER")
    print(f"  Precision: {span_metrics['precision']:.4f}")
    print(f"  Recall:    {span_metrics['recall']:.4f}")
    print(f"  F1:        {span_metrics['f1']:.4f}")
    print("Document-level (official)")
    print(f"  Precision: {doc_metrics['precision']:.4f}")
    print(f"  Recall:    {doc_metrics['recall']:.4f}")
    print(f"  Micro-F1:  {doc_metrics['micro_f1']:.4f}")


if __name__ == "__main__":
    main()
