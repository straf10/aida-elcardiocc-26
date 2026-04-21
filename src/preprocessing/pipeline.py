"""
Build ``data/processed/*`` from official raw splits under ``data/raw/Train_Set_2026/``.

Reads ``train.jsonl``, ``val.jsonl``, and ``test.jsonl`` (committee schema), writes cleaned
JSONL under ``data/processed/``, and refreshes ICD frequency tables from train.
IR reads raw text from the same paths under ``data/raw/`` (see ``information_retrieval.evaluate``).
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from .cleaning import clean_text, extract_annotations, flatten_annotations
from .io_utils import (
    LABELSET_PATH,
    PROCESSED_DIR,
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    PROCESSED_VAL_PATH,
    RAW_TRAIN_FOLDER_TEST_PATH,
    RAW_TRAIN_PATH,
    RAW_VAL_PATH,
    load_jsonl,
    save_jsonl,
)


def record_from_raw_item(item: dict) -> dict:
    """Clean text, keep annotations; add ``labels_flat`` for multilabel loaders."""
    annotations = extract_annotations(item)
    labels_flat = flatten_annotations(annotations)
    record: dict = {
        "patient_id": item.get("patient_id"),
        "text": clean_text(item.get("text", "")),
        "document_level_annotations": annotations,
        "labels_flat": labels_flat,
    }
    mentions = item.get("mention_level_annotations")
    if mentions is not None:
        record["mention_level_annotations"] = mentions
    return record


def _print_eda(name: str, records: list[dict], *, train_label_counts: Counter | None = None) -> None:
    if not records:
        print(f"\n{name}: (empty)")
        return
    labels_per = [len(r["labels_flat"]) for r in records]
    text_lens = [len(r["text"]) for r in records]
    print(f"\n{name}: n={len(records)}")
    print(f"  labels/doc: min={min(labels_per)} max={max(labels_per)} avg={sum(labels_per)/len(labels_per):.2f}")
    print(f"  text len: min={min(text_lens)} max={max(text_lens)} avg={sum(text_lens)/len(text_lens):.1f}")
    if train_label_counts is not None:
        print(f"  distinct ICD in train freq table: {len(train_label_counts)}")


def write_frequency_tables(counts: Counter, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "icd10_frequencies.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(counts.most_common(), f, ensure_ascii=False, indent=2)
    csv_path = out_dir / "icd10_frequencies.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ICD10", "Count"])
        w.writerows(counts.most_common())
    print(f"Wrote {jpath} and {csv_path}")


def run_preprocessing(
    *,
    raw_train: str | Path | None = None,
    raw_val: str | Path | None = None,
    raw_test: str | Path | None = None,
    out_dir: str | Path | None = None,
    labelset_path: str | Path | None = None,
) -> None:
    """
    Load three raw splits, write cleaned JSONL under ``data/processed/``.

    Raw committee files stay in ``data/raw/Train_Set_2026/``; IR loads those for mention expansion.
    """
    rt = Path(raw_train or RAW_TRAIN_PATH)
    rv = Path(raw_val or RAW_VAL_PATH)
    rtest = Path(raw_test or RAW_TRAIN_FOLDER_TEST_PATH)
    od = Path(out_dir or PROCESSED_DIR)
    ls_path = Path(labelset_path or LABELSET_PATH)

    for p, label in ((rt, "train"), (rv, "val"), (rtest, "test")):
        if not p.exists():
            raise FileNotFoundError(f"Missing {label} JSONL: {p}")

    train_raw = load_jsonl(str(rt))
    val_raw = load_jsonl(str(rv))
    test_raw = load_jsonl(str(rtest))

    print(f"Loaded raw train={len(train_raw)} val={len(val_raw)} test={len(test_raw)}")

    train_clean = [record_from_raw_item(r) for r in train_raw]
    val_clean = [record_from_raw_item(r) for r in val_raw]
    test_clean = [record_from_raw_item(r) for r in test_raw]

    save_jsonl(train_clean, str(PROCESSED_TRAIN_PATH))
    save_jsonl(val_clean, str(PROCESSED_VAL_PATH))
    save_jsonl(test_clean, str(PROCESSED_TEST_PATH))
    print(f"Wrote cleaned -> {PROCESSED_TRAIN_PATH}, {PROCESSED_VAL_PATH}, {PROCESSED_TEST_PATH}")

    counts: Counter[str] = Counter()
    for rec in train_clean:
        counts.update(rec["labels_flat"])

    write_frequency_tables(counts, od)

    if ls_path.exists():
        labelset = {line.strip() for line in ls_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        unknown = sorted({c for c in counts if c not in labelset})
        if unknown:
            print(f"WARNING: {len(unknown)} train labels not in labelset.txt (sample): {unknown[:10]}")

    _print_eda("train (cleaned)", train_clean, train_label_counts=counts)
    _print_eda("val (cleaned)", val_clean)
    _print_eda("test (cleaned)", test_clean)


def main() -> None:
    run_preprocessing()


if __name__ == "__main__":
    main()
