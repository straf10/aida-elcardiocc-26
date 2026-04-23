"""CLI: compare all models from config and print each method's micro-F1 (submission metric)."""

from __future__ import annotations

import argparse

from .compare import run_compare
from .config_utils import DEFAULT_EVAL_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score every predictions JSONL listed in config against gold, then add any "
            "{split}_predictions.jsonl under outputs/predictions (or data.predictions_root in YAML) "
            "not already scored; print tables by split: individual models vs ensemble strategies."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"YAML with data.test_path (compare gold), models[].predictions_path (default: {DEFAULT_EVAL_CONFIG})",
    )
    parser.add_argument(
        "--ground-truth",
        dest="ground_truth",
        default=None,
        help="Gold JSONL path (overrides multi-split scoring; uses predictions_path only).",
    )
    parser.add_argument(
        "--splits",
        default="test",
        help="With --config only (no --ground-truth): comma-separated test,val,blind. "
        "Scores sidecars val_predictions.jsonl / blind_predictions.jsonl next to each model's predictions_path. "
        "Blind metrics only if data.blind_path has document_level_annotations.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="PRED_JSONL:NAME",
        help="Ad-hoc method (repeatable). Requires --ground-truth; disables --config for the list.",
    )
    parser.add_argument("--labelset", default=None, help="Override labelset path for all models.")
    parser.add_argument("--metrics-json", default=None, help="Write compare rows as JSON.")

    args = parser.parse_args()
    if not args.config and not args.ground_truth and not args.pair:
        args.config = DEFAULT_EVAL_CONFIG

    run_compare(args)


if __name__ == "__main__":
    main()
