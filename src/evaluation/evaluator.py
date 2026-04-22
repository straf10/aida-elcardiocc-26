"""ELCardioCC evaluation CLI: score and compare. IR predictions: ``python -m information_retrieval.predict``. NER: ``python -m ner_el.predict``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from .compare import gather_compare_rows, run_compare
from .config_utils import DEFAULT_EVAL_CONFIG, get_cfg, load_config
from .scoring import (
    evaluate_data,
    evaluate_file,
    micro_f1,
    per_class_report,
    score_document,
)

__all__ = [
    "DEFAULT_EVAL_CONFIG",
    "evaluate_data",
    "evaluate_file",
    "micro_f1",
    "per_class_report",
    "score_document",
    "gather_compare_rows",
    "run_compare",
    "main",
]


def _parse_label_space(path: str | None) -> List[str]:
    if not path:
        return []

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("label_names JSON must be a list of code strings.")
    return [str(item) for item in data]


def _cmd_score(args: argparse.Namespace) -> None:
    from preprocessing.io_utils import load_labelset

    config = load_config(args.config)
    ground_truth_path = (
        args.ground_truth
        or get_cfg(config, "ground_truth_path")
        or get_cfg(config, "data.val_path")
    )
    pred_path = args.pred or get_cfg(config, "prediction_path")
    if not ground_truth_path or not pred_path:
        raise SystemExit("score: provide --ground-truth and --pred (or set them in --config YAML).")

    label_space: Sequence[str] | None = None
    if args.labelset:
        label_space = load_labelset(args.labelset)
    elif args.labels:
        label_space = _parse_label_space(args.labels)
    else:
        labels_path = get_cfg(config, "label_names_path")
        if labels_path:
            label_space = _parse_label_space(labels_path)

    metrics = evaluate_file(ground_truth_path, pred_path, label_space=label_space)
    print(f"Evaluated {metrics['docs_evaluated']} documents.")
    print(f"Micro-F1:  {metrics['micro_f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    if "macro_f1_present_labels" in metrics:
        print(f"Macro-F1 (present labels): {metrics['macro_f1_present_labels']:.4f}")
        print(f"Macro-F1 (all labels):     {metrics['macro_f1_all_labels']:.4f}")
    print(f"TP: {metrics['total_tp']} | FP: {metrics['total_fp']} | FN: {metrics['total_fn']}")

    if args.show_missing:
        print(f"Missing prediction IDs: {len(metrics['missing_prediction_ids'])}")
        print(f"Extra prediction IDs:   {len(metrics['extra_prediction_ids'])}")

    if args.metrics_json:
        out = Path(args.metrics_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Wrote metrics JSON -> {out}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ELCardioCC evaluation CLI: score and compare (multi-method). "
        "IR: python -m information_retrieval.predict | NER: python -m ner_el.predict"
    )
    sub = parser.add_subparsers(dest="command", help="Subcommand (optional for legacy score flags).")

    p_score = sub.add_parser("score", help="Micro/macro F1 for one gold + one predictions JSONL.")
    p_score.add_argument("--config", default=None, help=f"YAML (default paths); often {DEFAULT_EVAL_CONFIG}")
    p_score.add_argument("--ground-truth", dest="ground_truth", default=None)
    p_score.add_argument("--pred", default=None)
    p_score.add_argument("--labels", help="Optional JSON list of ICD-10 labels for per-class metrics")
    p_score.add_argument("--labelset", default=None, help="labelset.txt path for macro-F1 over full label space")
    p_score.add_argument("--metrics-json", default=None)
    p_score.add_argument("--show-missing", action="store_true")

    p_cmp = sub.add_parser("compare", help="F1 table: --config (models list) or --ground-truth + --pair.")
    p_cmp.add_argument("--config", default=None, help=f"YAML with data.val_path and models[]. Default: {DEFAULT_EVAL_CONFIG}")
    p_cmp.add_argument("--ground-truth", default=None)
    p_cmp.add_argument("--pair", action="append", default=[], metavar="PRED_JSONL:NAME")
    p_cmp.add_argument("--labelset", default=None)
    p_cmp.add_argument("--metrics-json", default=None)

    return parser


def main(argv: List[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    subcommands = ("score", "compare")
    if argv and argv[0] not in subcommands and argv[0] not in ("-h", "--help"):
        argv = ["score"] + argv

    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        raise SystemExit(2)

    if args.command == "score":
        _cmd_score(args)
        return
    if args.command == "compare":
        if not args.config and not args.ground_truth and not args.pair:
            args.config = DEFAULT_EVAL_CONFIG
        run_compare(args)
        return


if __name__ == "__main__":
    main()
