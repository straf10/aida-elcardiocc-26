"""
Build ``data/processed/*`` from labeled raw splits under ``data/raw/``.

Reads ``train.jsonl``, ``val.jsonl``, and ``test.jsonl`` (committee schema), writes cleaned
JSONL under ``data/processed/``. IR loads raw train/val from the same ``data/raw/`` paths.
"""

from __future__ import annotations

from pathlib import Path

from .cleaning import clean_text, extract_annotations, flatten_annotations
from .io_utils import (
    LABELSET_PATH,
    PROCESSED_DIR,
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    PROCESSED_VAL_PATH,
    RAW_TEST_PATH,
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


def _distinct_labels(records: list[dict]) -> set[str]:
    s: set[str] = set()
    for r in records:
        s.update(r["labels_flat"])
    return s


def _print_eda(name: str, records: list[dict], *, n_distinct_train_labels: int | None = None) -> None:
    if not records:
        print(f"\n{name}: (empty)")
        return
    labels_per = [len(r["labels_flat"]) for r in records]
    text_lens = [len(r["text"]) for r in records]
    print(f"\n{name}: n={len(records)}")
    print(f"  labels/doc: min={min(labels_per)} max={max(labels_per)} avg={sum(labels_per)/len(labels_per):.2f}")
    print(f"  text len: min={min(text_lens)} max={max(text_lens)} avg={sum(text_lens)/len(text_lens):.1f}")
    if n_distinct_train_labels is not None:
        print(f"  distinct ICD codes (train): {n_distinct_train_labels}")


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
    """
    rt = Path(raw_train or RAW_TRAIN_PATH)
    rv = Path(raw_val or RAW_VAL_PATH)
    rtest = Path(raw_test or RAW_TEST_PATH)
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

    train_codes = _distinct_labels(train_clean)
    if ls_path.exists():
        labelset = {line.strip() for line in ls_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        unknown = sorted(c for c in train_codes if c not in labelset)
        if unknown:
            print(f"WARNING: {len(unknown)} train labels not in labelset.txt (sample): {unknown[:10]}")

    _print_eda("train (cleaned)", train_clean, n_distinct_train_labels=len(train_codes))
    _print_eda("val (cleaned)", val_clean)
    _print_eda("test (cleaned)", test_clean)


def main() -> None:
    run_preprocessing()


if __name__ == "__main__":
    main()
