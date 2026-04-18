"""
Verify PID disjointness between global train and validation splits.

Supports either split_assignments.json (from training_validation --config) or
two JSONL files (train + val) with PIDs resolved via resolve_patient_id.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from src.preprocessing.io_utils import load_jsonl, resolve_patient_id
except ImportError:
    from ..preprocessing.io_utils import load_jsonl, resolve_patient_id


def _load_assignments_pids(path: str) -> tuple[set[int], set[int]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    train = {int(x) for x in data["train"]}
    val = {int(x) for x in data["val"]}
    return train, val


def _pids_from_jsonl(path: str) -> list[int]:
    records = load_jsonl(path)
    return [resolve_patient_id(r) for r in records]


def _dup_flags(pids: list[int]) -> tuple[bool, dict[int, int]]:
    seen: dict[int, int] = {}
    for p in pids:
        seen[p] = seen.get(p, 0) + 1
    dups = {p: c for p, c in seen.items() if c > 1}
    return len(dups) > 0, dups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check train/val PID disjointness (global split)."
    )
    parser.add_argument(
        "--assignments",
        default="data/processed/split_assignments.json",
        help="JSON with train/val PID lists (ignored if --train-jsonl and --val-jsonl are set).",
    )
    parser.add_argument(
        "--train-jsonl",
        default=None,
        help="Training split JSONL (use with --val-jsonl instead of --assignments).",
    )
    parser.add_argument(
        "--val-jsonl",
        default=None,
        help="Validation split JSONL (use with --train-jsonl).",
    )
    parser.add_argument(
        "--show-max",
        type=int,
        default=20,
        help="Max overlapping PIDs to print (if any).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if duplicate PIDs appear within train or within val.",
    )
    args = parser.parse_args()

    if (args.train_jsonl is None) ^ (args.val_jsonl is None):
        parser.error("Provide both --train-jsonl and --val-jsonl, or neither.")

    if args.train_jsonl and args.val_jsonl:
        train_path = Path(args.train_jsonl)
        val_path = Path(args.val_jsonl)
        print(f"Mode: JSONL\n  train: {train_path}\n  val:   {val_path}\n")
        train_pids_list = _pids_from_jsonl(str(train_path))
        val_pids_list = _pids_from_jsonl(str(val_path))
        train_set = set(train_pids_list)
        val_set = set(val_pids_list)

        if args.strict:
            td, td_map = _dup_flags(train_pids_list)
            vd, vd_map = _dup_flags(val_pids_list)
            if td or vd:
                print("STRICT: Duplicate patient_id rows found:", file=sys.stderr)
                if td:
                    sample = list(td_map.items())[:10]
                    print(f"  train duplicates (pid -> count): {dict(sample)}", file=sys.stderr)
                if vd:
                    sample = list(vd_map.items())[:10]
                    print(f"  val duplicates (pid -> count): {dict(sample)}", file=sys.stderr)
                sys.exit(2)
    else:
        ap = Path(args.assignments)
        print(f"Mode: assignments file\n  {ap}\n")
        try:
            train_set, val_set = _load_assignments_pids(str(ap))
        except FileNotFoundError:
            print(f"ERROR: File not found: {ap}", file=sys.stderr)
            sys.exit(3)
        except (KeyError, json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"ERROR reading assignments: {e}", file=sys.stderr)
            sys.exit(3)

        if args.strict:
            # Re-read lists preserving order for duplicate detection — assignments store unique IDs typically
            with open(ap, encoding="utf-8") as f:
                data = json.load(f)
            train_list = [int(x) for x in data["train"]]
            val_list = [int(x) for x in data["val"]]
            td, td_map = _dup_flags(train_list)
            vd, vd_map = _dup_flags(val_list)
            if td or vd:
                print("STRICT: Duplicate PIDs in assignments:", file=sys.stderr)
                if td:
                    print(f"  train: {td_map}", file=sys.stderr)
                if vd:
                    print(f"  val: {vd_map}", file=sys.stderr)
                sys.exit(2)

    overlap = train_set & val_set
    n_overlap = len(overlap)

    print(f"Train unique PIDs: {len(train_set)}")
    print(f"Val unique PIDs:   {len(val_set)}")
    print(f"Intersection:      {n_overlap}")

    if n_overlap > 0:
        sample = sorted(overlap)
        cap = max(0, args.show_max)
        print(f"Overlapping PIDs (up to {cap}): {sample[:cap]}")
        if len(sample) > cap:
            print(f"... and {len(sample) - cap} more.")
        print("\nStatus: NOT DISJOINT (exit 1)")
        sys.exit(1)

    print("\nStatus: DISJOINT — train and val PIDs do not overlap (exit 0)")
    sys.exit(0)


if __name__ == "__main__":
    main()
