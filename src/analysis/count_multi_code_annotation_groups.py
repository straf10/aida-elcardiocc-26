"""
Count document-level annotation groups with more than one distinct ICD code.

Uses `document_level_annotations` (list-of-lists) from the training JSONL. Each inner
list is one competition group (alternatives for one clinical entity).

If the share of multi-code groups is below ~15% of all groups, engineering aimed at
that structure is unlikely to move metrics much.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root on sys.path when running as `python src/analysis/count_multi_code_annotation_groups.py`
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.config_utils import get_cfg, load_config  # noqa: E402
from src.preprocessing.io_utils import PROJECT_ROOT, load_jsonl  # noqa: E402


def _resolve_train_path(config_path: str | None, train_path_override: str | None) -> Path:
    if train_path_override:
        p = Path(train_path_override)
        return p if p.is_absolute() else PROJECT_ROOT / p
    if not config_path:
        raise ValueError("Either --train-path or --config must be provided.")
    cfg = load_config(config_path)
    rel = get_cfg(cfg, "data.train_path")
    if not rel:
        raise ValueError("Config missing data.train_path and no --train-path given.")
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _distinct_codes(group: list) -> set[str]:
    return {str(c).strip() for c in group if c is not None and str(c).strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count annotation groups with >1 distinct code in the training set."
    )
    parser.add_argument(
        "--config",
        help="YAML config (uses data.train_path unless --train-path is set).",
    )
    parser.add_argument(
        "--train-path",
        help="Override path to training JSONL (relative paths are from repo root).",
    )
    args = parser.parse_args()

    train_path = _resolve_train_path(args.config, args.train_path)
    if not train_path.is_file():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    records = load_jsonl(str(train_path))

    total_groups = 0
    multi_code_groups = 0
    max_group_size = 0
    docs_with_any_multi = 0

    for rec in records:
        groups = rec.get("document_level_annotations") or []
        doc_has_multi = False
        for group in groups:
            if not isinstance(group, (list, tuple)):
                continue
            codes = _distinct_codes(list(group))
            if not codes:
                continue
            total_groups += 1
            n = len(codes)
            max_group_size = max(max_group_size, n)
            if n > 1:
                multi_code_groups += 1
                doc_has_multi = True
        if doc_has_multi:
            docs_with_any_multi += 1

    pct = (100.0 * multi_code_groups / total_groups) if total_groups else 0.0
    doc_pct = (100.0 * docs_with_any_multi / len(records)) if records else 0.0

    print(f"Training file: {train_path}")
    print(f"Documents: {len(records)}")
    print(f"Non-empty annotation groups (total): {total_groups}")
    print(f"Groups with >1 distinct code: {multi_code_groups}")
    print(f"Share of multi-code groups: {pct:.2f}%")
    print(f"Largest group (distinct codes): {max_group_size}")
    print(f"Documents with at least one multi-code group: {docs_with_any_multi} ({doc_pct:.2f}% of docs)")
    if total_groups and pct < 15.0:
        print(
            "\nNote: Multi-code groups are <15% of all groups; impact of group-specific "
            "engineering may be negligible."
        )


if __name__ == "__main__":
    main()
