import argparse
import json
import os
from pathlib import Path
import yaml

try:
    from src.preprocessing.io_utils import load_jsonl, resolve_patient_id, save_jsonl
    from src.training_validation.split import stratified_train_val_split
except ImportError:
    from ..preprocessing.io_utils import load_jsonl, resolve_patient_id, save_jsonl
    from .split import stratified_train_val_split


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate dataset splits.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)

    raw_path = config.get("data", {}).get("raw_path")
    cleaned_path = config.get("data", {}).get("cleaned_path")

    if not raw_path or not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data path not found: {raw_path}")
    if not cleaned_path or not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned data path not found: {cleaned_path}")

    print(f"Loading raw records from: {raw_path}")
    raw_records = load_jsonl(raw_path)

    print(f"Loading cleaned records from: {cleaned_path}")
    cleaned_records = load_jsonl(cleaned_path)

    # Ensure patient IDs map 1:1
    raw_by_id = {resolve_patient_id(r): r for r in raw_records}
    cleaned_by_id = {resolve_patient_id(r): r for r in cleaned_records}

    if set(raw_by_id.keys()) != set(cleaned_by_id.keys()):
        raise ValueError("Mismatch between patient IDs in raw and cleaned datasets.")

    test_size = config.get("split", {}).get("test_size", 0.2)
    seed = config.get("split", {}).get("seed", 42)

    print(f"Performing stratified split on cleaned records (test_size={test_size}, seed={seed})...")
    train_clean, val_clean = stratified_train_val_split(cleaned_records, test_size=test_size, seed=seed)

    train_ids = [resolve_patient_id(r) for r in train_clean]
    val_ids = [resolve_patient_id(r) for r in val_clean]

    train_raw = [raw_by_id[pid] for pid in train_ids]
    val_raw = [raw_by_id[pid] for pid in val_ids]

    train_clean_path = config.get("output", {}).get("train_clean_path")
    val_clean_path = config.get("output", {}).get("val_clean_path")
    train_raw_path = config.get("output", {}).get("train_raw_path")
    val_raw_path = config.get("output", {}).get("val_raw_path")
    assignments_path = config.get("output", {}).get("split_assignments_path")

    print(f"Saving training_set.jsonl (cleaned): {len(train_clean)} records -> {train_clean_path}")
    save_jsonl(train_clean, train_clean_path)

    print(f"Saving validation_set.jsonl (cleaned): {len(val_clean)} records -> {val_clean_path}")
    save_jsonl(val_clean, val_clean_path)

    print(f"Saving training_set_raw.jsonl (raw): {len(train_raw)} records -> {train_raw_path}")
    save_jsonl(train_raw, train_raw_path)

    print(f"Saving validation_set_raw.jsonl (raw): {len(val_raw)} records -> {val_raw_path}")
    save_jsonl(val_raw, val_raw_path)

    # Save assignments
    if assignments_path:
        os.makedirs(os.path.dirname(assignments_path), exist_ok=True)
        assignments = {
            "train": train_ids,
            "val": val_ids,
            "seed": seed
        }
        with open(assignments_path, "w", encoding="utf-8") as f:
            json.dump(assignments, f, ensure_ascii=False, indent=2)
        print(f"Saved split assignments to {assignments_path}")

    print("Done!")


if __name__ == "__main__":
    main()