"""Multi-method compare: score each model's predictions JSONL against gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from .config_utils import get_cfg, load_config
from .scoring import evaluate_file


def _parse_splits_arg(raw: str | None) -> List[str]:
    if raw is None or not str(raw).strip():
        return ["test"]
    out: list[str] = []
    for part in str(raw).split(","):
        s = part.strip().lower()
        if s in ("test", "val", "blind"):
            out.append(s)
    return out or ["test"]


def _gold_path_for_cfg_split(cfg: dict, split: str) -> str | None:
    keys = {"test": "data.test_path", "val": "data.val_path", "blind": "data.blind_path"}
    return get_cfg(cfg, keys[split], None)


def _pred_path_for_cfg_model(m: dict, split: str) -> str | None:
    p = m.get("predictions_path")
    if not p:
        return None
    base = Path(p)
    if split == "test":
        return str(base) if base.is_file() else None
    sibling = base.parent / ("val_predictions.jsonl" if split == "val" else "blind_predictions.jsonl")
    return str(sibling) if sibling.is_file() else None


def _blind_gold_has_codes(path: str) -> bool:
    from preprocessing.io_utils import load_jsonl

    for rec in load_jsonl(path):
        for g in rec.get("document_level_annotations") or []:
            if isinstance(g, list) and any(str(c).strip() for c in g):
                return True
    return False


def gather_compare_rows(args: argparse.Namespace) -> List[dict]:
    """Build one result dict per method (or error row) for compare / compare_methods.

    With ``--config``, default ``--splits test`` scores ``models[].predictions_path`` against
    ``data.test_path``. Use ``--splits val,test,blind`` to also score sidecar ``val_predictions.jsonl`` /
    ``blind_predictions.jsonl`` when present (blind metrics only if blind JSONL has gold codes).
    """
    from preprocessing.io_utils import LABELSET_PATH, load_labelset

    rows: list[dict] = []

    if args.config:
        cfg = load_config(args.config)
        default_ls = args.labelset
        models = get_cfg(cfg, "models", []) or []

        if args.ground_truth:
            gt_path = args.ground_truth
            if not gt_path or not Path(gt_path).is_file():
                raise SystemExit(f"Ground truth missing or not a file: {gt_path!r}")
            for m in models:
                name = str(m.get("name", "?"))
                pred_path = m.get("predictions_path")
                ls_path = default_ls or m.get("labelset_path") or str(LABELSET_PATH)
                if not pred_path:
                    rows.append({"name": name, "split": "override", "error": "no predictions_path in config"})
                    continue
                if not Path(pred_path).is_file():
                    rows.append({"name": name, "split": "override", "error": f"missing file {pred_path}"})
                    continue
                label_space = load_labelset(ls_path)
                metrics = evaluate_file(gt_path, pred_path, label_space=label_space)
                rows.append(
                    {
                        "name": name,
                        "split": "override",
                        "predictions_path": pred_path,
                        "micro_f1": metrics["micro_f1"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "macro_f1_present": metrics.get("macro_f1_present_labels"),
                    }
                )
        else:
            splits = _parse_splits_arg(getattr(args, "splits", None))
            for split in splits:
                gold = _gold_path_for_cfg_split(cfg, split)
                if not gold or not Path(gold).is_file():
                    rows.append(
                        {
                            "name": "(config)",
                            "split": split,
                            "error": f"missing gold for split {split!r}: {gold!r}",
                        }
                    )
                    continue
                if split == "blind" and not _blind_gold_has_codes(gold):
                    rows.append(
                        {
                            "name": "(split blind)",
                            "split": split,
                            "error": "blind JSONL has no document_level_annotations; metrics n/a",
                        }
                    )
                    continue
                for m in models:
                    name = str(m.get("name", "?"))
                    pred_path = _pred_path_for_cfg_model(m, split)
                    ls_path = default_ls or m.get("labelset_path") or str(LABELSET_PATH)
                    if not m.get("predictions_path"):
                        rows.append({"name": name, "split": split, "error": "no predictions_path in config"})
                        continue
                    if not pred_path:
                        exp = m.get("predictions_path") if split == "test" else str(Path(m["predictions_path"]).parent / f"{split}_predictions.jsonl")
                        rows.append(
                            {
                                "name": name,
                                "split": split,
                                "error": f"missing predictions file (expected {exp!r})",
                            }
                        )
                        continue
                    label_space = load_labelset(ls_path)
                    metrics = evaluate_file(gold, pred_path, label_space=label_space)
                    rows.append(
                        {
                            "name": name,
                            "split": split,
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
                rows.append({"name": name, "split": "pair", "error": f"missing file {pred_path}"})
                continue
            metrics = evaluate_file(gt_path, pred_path, label_space=label_space)
            rows.append(
                {
                    "name": name,
                    "split": "pair",
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
    """Table plus sorted micro-F1 lines (submission metric), grouped by ``split`` when present."""
    from itertools import groupby

    def _split_key(r: dict) -> str:
        return str(r.get("split", ""))

    rows_sorted = sorted(rows, key=lambda r: (_split_key(r), str(r.get("name", ""))))
    for split, group_it in groupby(rows_sorted, key=_split_key):
        chunk = list(group_it)
        label = split if split else "default"
        print(f"\n=== Split: {label} ===")
        col_w = max(22, max((len(r.get("name", "")) for r in chunk), default=10) + 2)
        header = f"{'Method':<{col_w}} {'Micro-F1':>9} {'Precision':>10} {'Recall':>8} {'Macro-F1*':>10}"
        print(header)
        print("-" * len(header))
        for r in chunk:
            if "error" in r:
                print(f"{r['name']:<{col_w}}  ERROR: {r['error']}")
            else:
                mf = r.get("macro_f1_present")
                mf_s = f"{mf:.4f}" if mf is not None else "n/a"
                print(
                    f"{r['name']:<{col_w}} {r['micro_f1']:>9.4f} {r['precision']:>10.4f}"
                    f" {r['recall']:>8.4f} {mf_s:>10}"
                )
        print()

        ok = [r for r in chunk if "error" not in r]
        if ok:
            print("Micro-F1 by method (group-level, submission metric):")
            for r in sorted(ok, key=lambda x: (-float(x["micro_f1"]), str(x["name"]))):
                print(f"  {r['name']}: {r['micro_f1']:.4f}")
            print()
    print("*Macro-F1 over labels with support in gold.\n")


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
