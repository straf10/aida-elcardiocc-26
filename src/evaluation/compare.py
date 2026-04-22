"""Multi-method compare: score each model's predictions JSONL against gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from .config_utils import get_cfg, load_config
from .scoring import evaluate_file


def gather_compare_rows(args: argparse.Namespace) -> List[dict]:
    """Build one result dict per method (or error row) for compare / compare_methods."""
    from preprocessing.io_utils import LABELSET_PATH, load_labelset

    rows: list[dict] = []

    if args.config:
        cfg = load_config(args.config)
        gt_path = args.ground_truth or get_cfg(cfg, "data.val_path")
        if not gt_path or not Path(gt_path).is_file():
            raise SystemExit(f"Ground truth missing or not a file: {gt_path!r}")
        default_ls = args.labelset
        for m in get_cfg(cfg, "models", []) or []:
            name = str(m.get("name", "?"))
            pred_path = m.get("predictions_path")
            ls_path = default_ls or m.get("labelset_path") or str(LABELSET_PATH)
            if not pred_path:
                rows.append({"name": name, "error": "no predictions_path in config"})
                continue
            if not Path(pred_path).is_file():
                rows.append({"name": name, "error": f"missing file {pred_path}"})
                continue
            label_space = load_labelset(ls_path)
            metrics = evaluate_file(gt_path, pred_path, label_space=label_space)
            rows.append(
                {
                    "name": name,
                    "predictions_path": pred_path,
                    "micro_f1": metrics["micro_f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "macro_f1_present": metrics.get("macro_f1_present_labels"),
                }
            )
    elif args.ground_truth and args.pair:
        gt_path = args.ground_truth
        if not Path(gt_path).is_file():
            raise SystemExit(f"Ground truth not found: {gt_path}")
        ls_path = args.labelset or str(LABELSET_PATH)
        label_space = load_labelset(ls_path)
        for raw in args.pair:
            if ":" not in raw:
                raise SystemExit(f"--pair must be PRED.jsonl:Name, got {raw!r}")
            pred_path, _, name = raw.partition(":")
            pred_path = pred_path.strip()
            name = name.strip() or Path(pred_path).stem
            if not Path(pred_path).is_file():
                rows.append({"name": name, "error": f"missing file {pred_path}"})
                continue
            metrics = evaluate_file(gt_path, pred_path, label_space=label_space)
            rows.append(
                {
                    "name": name,
                    "predictions_path": pred_path,
                    "micro_f1": metrics["micro_f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "macro_f1_present": metrics.get("macro_f1_present_labels"),
                }
            )
    else:
        raise SystemExit("compare: pass --config, or --ground-truth with one or more --pair PRED.jsonl:Name")

    return rows


def print_compare_report(rows: List[dict]) -> None:
    """Table plus sorted micro-F1 lines (submission metric)."""
    col_w = max(22, max((len(r.get("name", "")) for r in rows), default=10) + 2)
    header = f"{'Method':<{col_w}} {'Micro-F1':>9} {'Precision':>10} {'Recall':>8} {'Macro-F1*':>10}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<{col_w}}  ERROR: {r['error']}")
        else:
            mf = r.get("macro_f1_present")
            mf_s = f"{mf:.4f}" if mf is not None else "n/a"
            print(
                f"{r['name']:<{col_w}} {r['micro_f1']:>9.4f} {r['precision']:>10.4f}"
                f" {r['recall']:>8.4f} {mf_s:>10}"
            )
    print("\n*Macro-F1 over labels with support in gold.\n")

    ok = [r for r in rows if "error" not in r]
    if ok:
        print("Micro-F1 by method (group-level, submission metric):")
        for r in sorted(ok, key=lambda x: (-float(x["micro_f1"]), str(x["name"]))):
            print(f"  {r['name']}: {r['micro_f1']:.4f}")
        print()


def run_compare(args: argparse.Namespace) -> List[dict]:
    """Evaluate every model from config or --pair list; print report; optional JSON."""
    rows = gather_compare_rows(args)
    print_compare_report(rows)
    if args.metrics_json:
        out = Path(args.metrics_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"Wrote compare table JSON -> {out}")
    return rows
