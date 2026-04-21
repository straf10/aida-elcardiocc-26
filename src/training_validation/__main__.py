import argparse
import json
import os
import sys
from pathlib import Path

import yaml

_src = Path(__file__).resolve().parents[1]
if _src.name == "src" and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from preprocessing.io_utils import load_jsonl, save_jsonl
from training_validation.split import stratified_train_val_test_split


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate 80/10/10 train/val/test splits (multilabel stratified).")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)

    raw_path = config.get("data", {}).get("raw_path")
    if not raw_path or not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data path not found: {raw_path}")

    val_size = float(config.get("split", {}).get("val_size", 0.1))
    test_size = float(config.get("split", {}).get("test_size", 0.2))
    seed = int(config.get("split", {}).get("seed", 42))

    print(f"Loading records from: {raw_path}")
    records = load_jsonl(raw_path)

    train_frac = 1.0 - val_size - test_size
    print(
        f"Performing stratified train/val/test split "
        f"(train~{train_frac:.0%}, val={val_size:.0%}, test={test_size:.0%}, seed={seed})..."
    )
    train_recs, val_recs, test_recs, report = stratified_train_val_test_split(
        records, val_size=val_size, test_size=test_size, seed=seed
    )

    out_cfg = config.get("output", {}) or {}
    train_path = out_cfg.get("train_path")
    val_path = out_cfg.get("val_path")
    test_path = out_cfg.get("test_path")
    report_path = out_cfg.get("split_report_path")
    delete_original = bool(out_cfg.get("delete_original", False))

    if not train_path or not val_path or not test_path:
        raise ValueError("Config must set output.train_path, output.val_path, output.test_path")

    print(f"Saving train: {len(train_recs)} -> {train_path}")
    save_jsonl(train_recs, train_path)
    print(f"Saving val: {len(val_recs)} -> {val_path}")
    save_jsonl(val_recs, val_path)
    print(f"Saving test: {len(test_recs)} -> {test_path}")
    save_jsonl(test_recs, test_path)

    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved split report -> {report_path}")

    if delete_original and os.path.abspath(raw_path) != os.path.abspath(train_path):
        os.remove(raw_path)
        print(f"Removed original dataset: {raw_path}")

    print("Done!")


if __name__ == "__main__":
    main()
