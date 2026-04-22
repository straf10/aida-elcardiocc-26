"""Coverage, FP/FN, tokenization, and official evaluation reports."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from evaluation.evaluator import evaluate_data
from preprocessing.io_utils import resolve_patient_id

from .matcher import predict_codes_for_text
from .normalize import normalize_text, tokenize

if TYPE_CHECKING:
    from .config import DictionaryConfig


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def flatten_gold_groups(doc_groups):
    """
    Flatten list-of-lists annotations into a single set of codes.

    Each inner list is one clinical finding; prediction is correct if at least one
    code per group is matched (handled by the shared evaluator).
    """
    return {code for group in doc_groups for code in group}


def official_evaluation(
    records,
    matcher,
    labelset,
    *,
    config: "DictionaryConfig | None" = None,
    labelset_for_fuzzy: list[str] | None = None,
    code_desc_map: dict[str, str] | None = None,
    output_dir: Path | None = None,
):
    """
    Evaluate predictions with the official group-level evaluator
    (micro-F1, precision/recall, macro/per-class).
    """
    from preprocessing.io_utils import PROJECT_ROOT

    od = output_dir or (PROJECT_ROOT / "outputs" / "experiments" / "dictionary_baseline")
    _ensure_output_dir(od)
    ground_truth_data = {}
    pred_data = {}
    lf = labelset_for_fuzzy if labelset_for_fuzzy is not None else labelset

    for rec in records:
        patient_id = resolve_patient_id(rec)
        gold_groups = rec.get("document_level_annotations", [])
        if gold_groups is None:
            gold_groups = []

        prediction = predict_codes_for_text(
            rec.get("text", ""),
            matcher,
            config=config,
            labelset=lf,
            code_desc_map=code_desc_map,
        )
        ground_truth_data[patient_id] = gold_groups
        pred_data[patient_id] = sorted(prediction)

    metrics = evaluate_data(ground_truth_data, pred_data, label_space=labelset)
    with open(od / "official_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def coverage_report(
    records,
    matcher,
    labelset,
    *,
    config: "DictionaryConfig | None" = None,
    labelset_for_fuzzy: list[str] | None = None,
    code_desc_map: dict[str, str] | None = None,
    output_dir: Path | None = None,
):
    """
    Report how many documents and codes are covered by the dictionary.
    Also lists which codes are NOT covered (missing_codes).
    """
    from preprocessing.io_utils import PROJECT_ROOT

    od = output_dir or (PROJECT_ROOT / "outputs" / "experiments" / "dictionary_baseline")
    _ensure_output_dir(od)
    predicted_label_counter = Counter()
    covered_records = 0
    lf = labelset_for_fuzzy if labelset_for_fuzzy is not None else labelset

    for rec in records:
        pred = predict_codes_for_text(
            rec.get("text", ""),
            matcher,
            config=config,
            labelset=lf,
            code_desc_map=code_desc_map,
        )
        if pred:
            covered_records += 1
        predicted_label_counter.update(pred)

    covered_codes = set(predicted_label_counter.keys())
    missing_codes = sorted(set(labelset) - covered_codes)

    report = {
        "num_records": len(records),
        "records_with_any_prediction": covered_records,
        "record_coverage_ratio": round(covered_records / len(records), 4),
        "num_codes_in_labelset": len(labelset),
        "num_codes_covered_by_dictionary": len(covered_codes),
        "code_coverage_ratio": round(len(covered_codes) / len(labelset), 4),
        "covered_codes": sorted(covered_codes),
        "top_predicted_codes": predicted_label_counter.most_common(20),
        "missing_codes": missing_codes,
    }

    with open(od / "coverage_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def fp_fn_analysis(
    records,
    matcher,
    *,
    labelset: list[str] | None = None,
    config: "DictionaryConfig | None" = None,
    labelset_for_fuzzy: list[str] | None = None,
    code_desc_map: dict[str, str] | None = None,
    output_dir: Path | None = None,
):
    """
    Identify which codes are most over-predicted (false positives)
    and most under-predicted (false negatives).
    """
    from preprocessing.io_utils import PROJECT_ROOT

    od = output_dir or (PROJECT_ROOT / "outputs" / "experiments" / "dictionary_baseline")
    _ensure_output_dir(od)
    fp_counter = Counter()
    fn_counter = Counter()
    lf = labelset_for_fuzzy if labelset_for_fuzzy is not None else labelset

    for rec in records:
        gold = flatten_gold_groups(rec.get("document_level_annotations", []))
        pred = predict_codes_for_text(
            rec.get("text", ""),
            matcher,
            config=config,
            labelset=lf,
            code_desc_map=code_desc_map,
        )
        for c in pred - gold:
            fp_counter[c] += 1
        for c in gold - pred:
            fn_counter[c] += 1

    result = {
        "top_false_positives": fp_counter.most_common(20),
        "top_false_negatives": fn_counter.most_common(20),
    }
    with open(od / "fp_fn_analysis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return fp_counter, fn_counter


def label_frequency_report(records, *, output_dir: Path | None = None):
    """Count how many times each ICD-10 code appears in the gold labels."""
    from preprocessing.io_utils import PROJECT_ROOT

    od = output_dir or (PROJECT_ROOT / "outputs" / "experiments" / "dictionary_baseline")
    _ensure_output_dir(od)
    counter = Counter()
    for rec in records:
        counter.update(flatten_gold_groups(rec.get("document_level_annotations", [])))
    with open(od / "label_frequency.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["code", "count"])
        for code, cnt in counter.most_common():
            writer.writerow([code, cnt])
    return counter


def tokenization_report(records, output_path: str, *, output_dir: Path | None = None):
    """
    Analyze tokenization and normalization characteristics of the corpus.
    Produces a JSON report useful for the MLC team (token length distribution).
    """
    from preprocessing.io_utils import PROJECT_ROOT

    od = output_dir or (PROJECT_ROOT / "outputs" / "experiments" / "dictionary_baseline")
    _ensure_output_dir(od)
    num_docs = len(records)
    token_lengths = []
    char_lengths = []
    edge_case_docs = []

    formatting_triggers = ["\n", "/", "-", "(", ")", ":", ";"]
    abbreviation_patterns = re.compile(r"\b[A-ZΑ-Ω]{2,6}\b")

    for rec in records:
        text = rec.get("text", "")
        toks = tokenize(text)
        token_lengths.append(len(toks))
        char_lengths.append(len(text))

        has_formatting = any(x in text for x in formatting_triggers)
        abbrevs = abbreviation_patterns.findall(text)

        if has_formatting or abbrevs:
            edge_case_docs.append(
                {
                    "row_id": rec.get("_row_id"),
                    "num_tokens": len(toks),
                    "num_chars": len(text),
                    "has_formatting_chars": has_formatting,
                    "sample_abbreviations": list(set(abbrevs))[:10],
                    "sample_text": text[:200],
                }
            )

    report = {
        "num_docs": num_docs,
        "avg_tokens_per_doc": round(sum(token_lengths) / num_docs, 2) if num_docs else 0,
        "max_tokens_per_doc": max(token_lengths) if token_lengths else 0,
        "min_tokens_per_doc": min(token_lengths) if token_lengths else 0,
        "avg_chars_per_doc": round(sum(char_lengths) / num_docs, 2) if num_docs else 0,
        "docs_over_512_tokens": sum(1 for l in token_lengths if l > 512),
        "docs_over_512_tokens_pct": round(
            sum(1 for l in token_lengths if l > 512) / num_docs * 100, 2
        )
        if num_docs
        else 0,
        "num_docs_with_formatting_issues": len(edge_case_docs),
        "normalization_steps_applied": [
            "lowercase",
            "accent stripping (unicodedata NFD)",
            "punctuation → space (-, /, \\, newlines)",
            "non-alphanumeric removal",
            "whitespace collapse",
        ],
        "sample_edge_cases": edge_case_docs[:10],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Tokenization report saved: {output_path}")
    return report
