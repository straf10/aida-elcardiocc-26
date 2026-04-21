"""Validate ELCardioCC submission ZIP or directory of JSONL files (Masterplan §18)."""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Set

from preprocessing.io_utils import load_jsonl, load_labelset, resolve_patient_id

from .config_utils import get_cfg, load_config
from .io_utils import flatten_annotation_groups


@dataclass
class ValidationReport:
    """Structured result of ``validate_submission``."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    files_checked: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_expected_patient_ids(test_jsonl: str) -> Set[int]:
    """Load expected patient IDs from a reference JSONL (e.g. test set)."""
    return {resolve_patient_id(r) for r in load_jsonl(test_jsonl)}


def _iter_zip_jsonl_members(zip_path: Path) -> List[str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return sorted(name for name in zf.namelist() if name.endswith(".jsonl"))


def _validate_annotation_structure(
    annots: object,
    *,
    fname: str,
    line_no: int,
    strict_singleton: bool,
    errors: List[str],
) -> bool:
    if not isinstance(annots, list):
        errors.append(f"ERROR: {fname} line {line_no} document_level_annotations not a list")
        return False
    for j, group in enumerate(annots):
        if not isinstance(group, list) or len(group) == 0:
            errors.append(
                f"ERROR: {fname} line {line_no} group {j} not a non-empty list"
            )
            continue
        if strict_singleton and len(group) != 1:
            errors.append(
                f"ERROR: {fname} line {line_no} group {j} must be a singleton list "
                f"when require_singleton_groups is true"
            )
        for code in group:
            if not isinstance(code, str) or not code.strip():
                errors.append(
                    f"ERROR: {fname} line {line_no} group {j} contains invalid code"
                )
    return True


def _check_codes_against_labelset(
    annots: list,
    valid_codes: Set[str],
    *,
    fname: str,
    line_no: int,
    strict_codes: bool,
    errors: List[str],
    warnings: List[str],
) -> None:
    for group in annots:
        if not isinstance(group, list):
            continue
        for code in group:
            if isinstance(code, str) and code and code not in valid_codes:
                msg = (
                    f"{fname} line {line_no} code {code!r} not in training label space"
                )
                if strict_codes:
                    errors.append(f"ERROR: {msg}")
                else:
                    warnings.append(f"WARNING: {msg}")


def _validate_jsonl_lines(
    lines: Iterator[str],
    fname: str,
    expected_patient_ids: Set[int],
    valid_codes: Set[str],
    *,
    strict_codes: bool,
    require_singleton_groups: bool,
    errors: List[str],
    warnings: List[str],
    files_checked: List[str],
) -> None:
    seen_ids: Set[int] = set()
    line_no = 0
    for raw in lines:
        line_no += 1
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"ERROR: {fname} line {line_no} is not valid JSON")
            continue
        if "patient_id" not in record:
            errors.append(f"ERROR: {fname} line {line_no} missing patient_id")
            continue
        try:
            pid = int(record["patient_id"])
        except (TypeError, ValueError):
            errors.append(f"ERROR: {fname} line {line_no} patient_id not int-coercible")
            continue
        if pid in seen_ids:
            errors.append(f"ERROR: {fname} line {line_no} duplicate patient_id {pid}")
            continue
        seen_ids.add(pid)

        if "document_level_annotations" not in record:
            errors.append(f"ERROR: {fname} line {line_no} missing document_level_annotations")
            continue
        annots = record["document_level_annotations"]
        if not _validate_annotation_structure(
            annots,
            fname=fname,
            line_no=line_no,
            strict_singleton=require_singleton_groups,
            errors=errors,
        ):
            continue
        assert isinstance(annots, list)
        _check_codes_against_labelset(
            annots,
            valid_codes,
            fname=fname,
            line_no=line_no,
            strict_codes=strict_codes,
            errors=errors,
            warnings=warnings,
        )
        flatten_annotation_groups(annots)

    files_checked.append(fname)
    missing = expected_patient_ids - seen_ids
    if missing:
        sample = sorted(missing)[:5]
        errors.append(
            f"ERROR: {fname} missing {len(missing)} patient_ids "
            f"(showing up to 5): {sample}"
        )
    extra = seen_ids - expected_patient_ids
    if extra:
        warnings.append(
            f"WARNING: {fname} has {len(extra)} unexpected patient_ids "
            f"(showing up to 5): {sorted(extra)[:5]}"
        )


def validate_submission(
    submission_path: str | Path,
    expected_patient_ids: Iterable[int],
    valid_codes: Iterable[str],
    *,
    max_systems: int = 5,
    strict_codes: bool = False,
    require_singleton_groups: bool = False,
) -> ValidationReport:
    """
    Validate a submission ZIP or a directory containing ``.jsonl`` files.

    Each JSONL is checked independently (each must cover all expected patient IDs).
    """
    errors: List[str] = []
    warnings: List[str] = []
    files_checked: List[str] = []
    path = Path(submission_path)
    expected_set = set(expected_patient_ids)
    valid_set = set(valid_codes)

    if not path.exists():
        errors.append(f"ERROR: submission path does not exist: {path}")
        return ValidationReport(errors=errors, warnings=warnings, files_checked=files_checked)

    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            jsonl_members = sorted(n for n in zf.namelist() if n.endswith(".jsonl"))
            if len(jsonl_members) == 0:
                errors.append("ERROR: No .jsonl files in ZIP")
            if len(jsonl_members) > max_systems:
                errors.append(
                    f"ERROR: {len(jsonl_members)} JSONL systems in ZIP — max is {max_systems}"
                )
            for member in jsonl_members:
                with zf.open(member, "r") as raw:
                    text_stream = io.TextIOWrapper(raw, encoding="utf-8-sig")
                    lines = (ln for ln in text_stream)
                    _validate_jsonl_lines(
                        lines,
                        member,
                        expected_set,
                        valid_set,
                        strict_codes=strict_codes,
                        require_singleton_groups=require_singleton_groups,
                        errors=errors,
                        warnings=warnings,
                        files_checked=files_checked,
                    )
        return ValidationReport(errors=errors, warnings=warnings, files_checked=files_checked)

    if path.is_dir():
        jsonl_files = sorted(path.glob("*.jsonl"))
        if len(jsonl_files) == 0:
            errors.append(f"ERROR: No .jsonl files in directory: {path}")
        if len(jsonl_files) > max_systems:
            errors.append(
                f"ERROR: {len(jsonl_files)} JSONL files — max is {max_systems}"
            )
        for fp in jsonl_files:
            with open(fp, "r", encoding="utf-8-sig") as handle:
                lines = (ln for ln in handle)
                _validate_jsonl_lines(
                    lines,
                    str(fp),
                    expected_set,
                    valid_set,
                    strict_codes=strict_codes,
                    require_singleton_groups=require_singleton_groups,
                    errors=errors,
                    warnings=warnings,
                    files_checked=files_checked,
                )
        return ValidationReport(errors=errors, warnings=warnings, files_checked=files_checked)

    errors.append(
        f"ERROR: submission_path must be a .zip file or directory, got: {path}"
    )
    return ValidationReport(errors=errors, warnings=warnings, files_checked=files_checked)


def _print_report(report: ValidationReport) -> None:
    for f in report.files_checked:
        print(f"  checked: {f}")
    if report.warnings:
        for w in report.warnings:
            print(w)
    if report.errors:
        print("\n--- VALIDATION FAILED ---")
        for e in report.errors:
            print(f"  {e}")
    else:
        print("\n--- VALIDATION PASSED ---")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ELCardioCC submission ZIP or directory.")
    parser.add_argument("--config", help="Optional YAML config path")
    parser.add_argument("--submission", help="Path to .zip or directory of .jsonl files")
    parser.add_argument("--test", dest="test_jsonl", help="Reference JSONL for expected patient IDs")
    parser.add_argument("--labels", dest="labelset_path", help="labelset.txt path (valid ICD-10 codes)")
    parser.add_argument("--max-systems", type=int, help="Max JSONL files (default 5)")
    parser.add_argument(
        "--strict-codes",
        action="store_true",
        help="Treat unknown codes as errors instead of warnings",
    )
    parser.add_argument(
        "--require-singleton-groups",
        action="store_true",
        help="Require each inner list to contain exactly one code",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    submission = args.submission or get_cfg(cfg, "submission.zip_path")
    test_jsonl = args.test_jsonl or get_cfg(cfg, "submission.test_jsonl")
    labelset_path = args.labelset_path or get_cfg(cfg, "submission.labelset_path")
    max_systems = (
        args.max_systems
        if args.max_systems is not None
        else get_cfg(cfg, "submission.max_systems", 5)
    )
    strict_codes = args.strict_codes or bool(get_cfg(cfg, "submission.strict_codes", False))
    require_singleton = args.require_singleton_groups or bool(
        get_cfg(cfg, "submission.require_singleton_groups", False)
    )

    if not submission or not test_jsonl or not labelset_path:
        raise ValueError(
            "Provide --submission, --test, and --labels (or set submission.* in config)."
        )

    expected = load_expected_patient_ids(test_jsonl)
    valid_codes = load_labelset(labelset_path)
    report = validate_submission(
        submission,
        expected,
        valid_codes,
        max_systems=int(max_systems),
        strict_codes=strict_codes,
        require_singleton_groups=require_singleton,
    )
    _print_report(report)
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
