"""CLI: compare all models from config and print each method's micro-F1 (submission metric)."""

from __future__ import annotations

import argparse

from .compare import run_compare
from .config_utils import DEFAULT_EVAL_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score every predictions JSONL listed in config against gold, "
            "print a metrics table and a micro-F1 line per method."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"YAML with data.val_path and models[].predictions_path (default: {DEFAULT_EVAL_CONFIG})",
    )
    parser.add_argument(
        "--ground-truth",
        dest="ground_truth",
        default=None,
        help="Gold JSONL path (overrides data.val_path from config when using --config).",
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
