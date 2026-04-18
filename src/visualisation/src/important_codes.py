from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

Pair = Tuple[str, str]

# ICD-10-like token at start of a table cell (before Greek description in parentheses)
_CODE_TOKEN = re.compile(r"^([A-Z][A-Z0-9.-]*)(?:\s|\(|$)")
# Backtick pair: `I50 ↔ Y84` or `I21/I25 ↔ I79`
_BACKTICK_PAIR = re.compile(r"`([^`]+)`")
# Slashed codes: I21/I25
_SLASH_CODES = re.compile(r"[A-Z][A-Z0-9.-]*(?:/[A-Z][A-Z0-9.-]*)+")
# Simple code
_SINGLE_CODE = re.compile(r"\b([A-Z][A-Z0-9.-]*)\b")


def _first_code_cell(cell: str) -> str | None:
    cell = cell.strip()
    if not cell or cell.startswith("---"):
        return None
    m = _CODE_TOKEN.match(cell)
    if m:
        return m.group(1)
    return None


def _parse_confused_pair_table_lines(lines: Iterable[str]) -> Set[Pair]:
    """Parse rows like | predicted | missed | count | with optional Greek in cells."""
    pairs: Set[Pair] = set()
    for line in lines:
        line = line.strip()
        if not line.startswith("|") or "---" in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        # leading/trailing empty from split
        parts = [p for p in parts if p]
        if len(parts) < 3:
            continue
        # skip header rows
        h = parts[0].lower()
        if "predicted" in h or "ζεύγος" in h or "missed" in parts[0].lower():
            continue
        p = _first_code_cell(parts[0])
        t = _first_code_cell(parts[1])
        if p and t:
            pairs.add((p, t))
    return pairs


def _expand_slash_pair(left: str, right: str) -> Set[Pair]:
    """Handle 'I21/I25' paired with 'I79' -> two pairs."""
    out: Set[Pair] = set()
    left_parts = [x.strip() for x in left.split("/") if x.strip()]
    right_parts = [x.strip() for x in right.split("/") if x.strip()]
    if len(left_parts) > 1 and len(right_parts) == 1:
        for lp in left_parts:
            out.add((lp, right_parts[0]))
    elif len(right_parts) > 1 and len(left_parts) == 1:
        for rp in right_parts:
            out.add((left_parts[0], rp))
    elif len(left_parts) == 1 and len(right_parts) == 1:
        out.add((left_parts[0], right_parts[0]))
    else:
        for lp in left_parts:
            for rp in right_parts:
                out.add((lp, rp))
    return out


def _parse_universal_miss_cell(pair_cell: str) -> Set[Pair]:
    """Parse cells like `I50 ↔ Y84`, `I21/I25 ↔ I79`, `R07 ↔ I10/I11`."""
    pairs: Set[Pair] = set()
    text = pair_cell.strip().strip("`")
    if "↔" not in text:
        return pairs
    left, right = [s.strip() for s in text.split("↔", 1)]
    pairs |= _expand_slash_pair(left, right)
    return pairs


def parse_report_md(path: Path) -> Tuple[Set[Pair], Set[str]]:
    """Extract confused pairs from REPORT.md §6–7 and range codes mentioned in §6."""
    pairs: Set[Pair] = set()
    codes: Set[str] = set()
    if not path.is_file():
        return pairs, codes

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    m6 = re.search(r"## 6\.", text)
    m7 = re.search(r"## 7\.", text)
    if m6 and m7:
        sec6 = text[m6.start() : m7.start()]
        for m in re.finditer(r"`([A-Z][A-Z0-9.-]+)`", sec6):
            codes.add(m.group(1))

    in_pair_table = False
    in_universal = False
    buf: List[str] = []

    for line in lines:
        if "### 7." in line:
            if in_pair_table and buf:
                pairs |= _parse_confused_pair_table_lines(buf)
                buf = []
            in_pair_table = "7.3" not in line
            in_universal = "7.3" in line
            continue
        if line.startswith("## ") and "## 7" not in line:
            if in_pair_table and buf:
                pairs |= _parse_confused_pair_table_lines(buf)
                buf = []
            in_pair_table = False
            in_universal = False

        if in_universal and line.strip().startswith("|") and "↔" in line:
            parts = [p.strip() for p in line.split("|")]
            parts = [p for p in parts if p]
            if parts and not parts[0].lower().startswith("ζεύγος"):
                cell0 = parts[0]
                if "↔" in cell0:
                    pairs |= _parse_universal_miss_cell(cell0)
        elif in_pair_table and line.strip().startswith("|"):
            buf.append(line)
        elif in_pair_table and buf and not line.strip().startswith("|"):
            pairs |= _parse_confused_pair_table_lines(buf)
            buf = []

    if buf:
        pairs |= _parse_confused_pair_table_lines(buf)

    return pairs, codes


def parse_medical_summary_md(path: Path, excluded_models: Set[str] | None = None) -> Set[Pair]:
    pairs: Set[Pair] = set()
    excluded = excluded_models or set()
    if not path.is_file():
        return pairs

    lines = path.read_text(encoding="utf-8").splitlines()
    in_top10 = False
    buf: List[str] = []
    current_model: str | None = None

    def flush() -> None:
        nonlocal buf, pairs
        if current_model in excluded:
            buf = []
            return
        pairs |= _parse_confused_pair_table_lines(buf)
        buf = []

    for line in lines:
        if line.startswith("### ") and not line.startswith("#### "):
            if in_top10:
                flush()
                in_top10 = False
            m = re.match(r"^###\s+(\S+)", line)
            current_model = m.group(1) if m else None
        if "#### Top 10 Confused Pairs" in line:
            if in_top10:
                flush()
            in_top10 = True
            buf = []
            continue
        if in_top10 and line.startswith("#### ") and "Top 10" not in line:
            flush()
            in_top10 = False
            buf = []
        elif in_top10:
            if line.strip().startswith("|"):
                buf.append(line)
            elif line.strip() == "" and buf:
                flush()

    if in_top10:
        flush()
    return pairs


def load_label_analysis_artifacts(
    analysis_dir: Path,
    model_names: List[str],
    wrong_pairs_top_k: int = 30,
) -> Tuple[Set[Pair], Set[str]]:
    """Merge top_confused_pairs and top wrong_pairs_counter keys per model."""
    pairs: Set[Pair] = set()
    codes: Set[str] = set()

    for name in model_names:
        p = analysis_dir / name / "label_analysis.json"
        if not p.is_file():
            continue
        with open(p, encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        for row in data.get("top_confused_pairs", []):
            pred, missed = row.get("predicted"), row.get("missed")
            if pred and missed:
                pairs.add((str(pred), str(missed)))
                codes.add(str(pred))
                codes.add(str(missed))
        wpc = data.get("wrong_pairs_counter") or {}
        if isinstance(wpc, dict):
            ranked = sorted(wpc.items(), key=lambda kv: kv[1], reverse=True)[:wrong_pairs_top_k]
            for key, _ in ranked:
                if "|" not in key:
                    continue
                a, b = key.split("|", 1)
                pairs.add((a, b))
                codes.add(a)
                codes.add(b)

    return pairs, codes


def frequent_codes_from_gt(
    support: Counter[str],
    frequent_min_support: int,
) -> Set[str]:
    return {c for c, n in support.items() if n >= frequent_min_support}


def collect_important_pairs_and_codes(
    cfg: Dict[str, Any],
    gt_data: Dict[int, List[List[str]]],
    label_names: List[str],
    reports_dir: Path,
    analysis_dir: Path,
    model_names: List[str],
    wrong_pairs_top_k: int = 30,
) -> Tuple[Set[Pair], Set[str]]:
    """
    Union of: REPORT.md pairs + codes, medical_report_summary pairs,
    label_analysis per model, frequent-bucket code set (for axis enrichment).
    """
    try:
        from src.evaluation.config_utils import get_cfg
        from src.analysis.common import label_support_from_gt
    except ImportError:
        from evaluation.config_utils import get_cfg  # type: ignore
        from analysis.common import label_support_from_gt  # type: ignore

    freq_min = int(get_cfg(cfg, "long_tail.frequent_min_support", 50))

    support = label_support_from_gt(gt_data, label_names)
    freq_codes = frequent_codes_from_gt(support, freq_min)

    try:
        from src.visualisation.src.config import EXCLUDED_MODELS
    except ImportError:
        from config import EXCLUDED_MODELS  # type: ignore

    report_pairs, report_codes = parse_report_md(reports_dir / "REPORT.md")
    summary_pairs = parse_medical_summary_md(
        reports_dir / "medical_report_summary.md",
        excluded_models=set(EXCLUDED_MODELS),
    )
    art_pairs, art_codes = load_label_analysis_artifacts(
        analysis_dir, model_names, wrong_pairs_top_k=wrong_pairs_top_k
    )

    all_pairs: Set[Pair] = set()
    all_pairs |= report_pairs
    all_pairs |= summary_pairs
    all_pairs |= art_pairs

    all_codes: Set[str] = set()
    all_codes |= report_codes
    all_codes |= freq_codes
    all_codes |= art_codes
    for p, t in all_pairs:
        all_codes.add(p)
        all_codes.add(t)

    return all_pairs, all_codes
