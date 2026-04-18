"""
Compare Greek BERT train.validate() gold reconstruction vs official document_level_annotations.

When any annotation group has more than one code (alternatives), validate() collapses to
one singleton group per active label, which does not match group-level evaluation in
src/evaluation/evaluator.py.
"""

from __future__ import annotations

import argparse
import json
import sys

try:
    from src.evaluation.evaluator import score_document
    from src.evaluation.io_utils import load_ground_truth
    from src.preprocessing.io_utils import load_jsonl
except ImportError:
    from ..evaluation.evaluator import score_document
    from ..evaluation.io_utils import load_ground_truth
    from ..preprocessing.io_utils import load_jsonl


def _load_label_names(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _filter_groups(groups: list, label2idx: dict[str, int]) -> list[list[str]]:
    out: list[list[str]] = []
    for group in groups or []:
        if not isinstance(group, list):
            continue
        g = [c for c in group if c in label2idx]
        if g:
            out.append(g)
    return out


def _multihot_from_groups(groups: list[list[str]], n_labels: int, label2idx: dict[str, int]) -> list[float]:
    """Same logic as CardioDataset.__getitem__ in mlc_greek_bert/train.py."""
    vec = [0.0] * n_labels
    for group in groups:
        for code in group:
            if code in label2idx:
                vec[label2idx[code]] = 1.0
    return vec


def _gold_validate_style(multihot: list[float], label_names: list[str]) -> list[list[str]]:
    """Same as validate() in mlc_greek_bert/train.py (lines 179-180)."""
    return [[label_names[j]] for j in range(len(label_names)) if multihot[j] == 1.0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Greek BERT validate() gold vs official group annotations on val data."
    )
    parser.add_argument(
        "--val-path",
        default="data/processed/validation_set.jsonl",
        help="Validation JSONL with document_level_annotations.",
    )
    parser.add_argument(
        "--labels",
        default="data/raw/Train_Set_2026/labelset.txt",
        help="Label order file (one ICD-10 code per line), same as training.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=8,
        help="Max example documents with multi-code groups to print.",
    )
    parser.add_argument(
        "--fail-on-multigroup",
        action="store_true",
        help="Exit 1 if any val document has a multi-code group (validate() cannot match official groups).",
    )
    args = parser.parse_args()

    label_names = _load_label_names(args.labels)
    label2idx = {label: i for i, label in enumerate(label_names)}
    n_labels = len(label_names)

    try:
        gt = load_ground_truth(args.val_path)
        records = load_jsonl(args.val_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(3)

    n_docs = len(records)
    multigroup_count = 0
    mismatch_checks = 0

    for record in records:
        pid = int(record["patient_id"])
        raw_groups = record.get("document_level_annotations") or []
        official = _filter_groups(raw_groups, label2idx)

        if any(len(g) > 1 for g in official):
            multigroup_count += 1

        ref = gt.get(pid)
        if ref is not None:
            ref_f = _filter_groups(ref, label2idx)
            if ref_f != official:
                mismatch_checks += 1

    print(f"Val file: {args.val_path}")
    print(f"Documents: {n_docs}")
    print(
        f"Documents with at least one multi-code group (alternatives): {multigroup_count} "
        f"({100.0 * multigroup_count / n_docs if n_docs else 0:.2f}%)"
    )
    if mismatch_checks:
        print(
            f"WARNING: load_ground_truth disagrees with filtered record annotations for {mismatch_checks} PIDs.",
            file=sys.stderr,
        )

    printed = 0
    for record in records:
        if printed >= args.examples:
            break
        raw_groups = record.get("document_level_annotations") or []
        official = _filter_groups(raw_groups, label2idx)
        if not any(len(g) > 1 for g in official):
            continue

        pid = int(record["patient_id"])
        mh = _multihot_from_groups(official, n_labels, label2idx)
        validate_style = _gold_validate_style(mh, label_names)

        tp_o, fp_o, fn_o = score_document(official, [])
        tp_v, fp_v, fn_v = score_document(validate_style, [])

        print("\n--- Example (empty predictions) ---")
        print(f"patient_id: {pid}")
        print(f"official groups:     {json.dumps(official, ensure_ascii=False)}")
        print(f"validate() reconstruction (singletons per label): {json.dumps(validate_style, ensure_ascii=False)}")
        print(f"score_document(..., pred=[]) official: tp={tp_o} fp={fp_o} fn={fn_o}")
        print(f"score_document(..., pred=[]) validate-style: tp={tp_v} fp={fp_v} fn={fn_v}")

        printed += 1

    print(
        "\nNote: validate() in mlc_greek_bert/train.py builds gold like the singleton list; "
        "for multi-code groups this differs from evaluator group semantics."
    )

    if args.fail_on_multigroup and multigroup_count > 0:
        print(
            f"\nFAIL: --fail-on-multigroup set and {multigroup_count} documents have multi-code groups.",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
