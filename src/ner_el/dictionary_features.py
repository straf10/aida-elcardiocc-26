from __future__ import annotations

"""Dictionary resource utilities for the main NER->EL pipeline.

This module reads dictionary DATA files from data/external/ and does not depend on
src/ner_el/dictionary_ner_el.py (which is a separate backup pipeline).
"""

import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .types import MentionAnnotation, NERMentionPrediction

BASE_CSV = "data/external/icd10_greek_lookup.csv"
RICH_CSV = "data/external/full_dictionary.csv"


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text)


def normalize_text(text: str) -> str:
    text = _strip_accents(text.lower())
    text = re.sub(r"[^α-ωa-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_with_char_map(text: str) -> tuple[str, list[int]]:
    out_chars: List[str] = []
    norm_to_orig: List[int] = []
    prev_space = True

    for i, ch in enumerate(text):
        low = _strip_accents(ch.lower())
        candidate = low if re.match(r"[α-ωa-z0-9\s]", low) else " "
        if candidate.isspace():
            if prev_space:
                continue
            out_chars.append(" ")
            norm_to_orig.append(i)
            prev_space = True
        else:
            out_chars.append(candidate)
            norm_to_orig.append(i)
            prev_space = False

    if out_chars and out_chars[-1] == " ":
        out_chars.pop()
        norm_to_orig.pop()

    return "".join(out_chars), norm_to_orig


def load_dictionary_candidates() -> Dict[str, set]:
    mapping = defaultdict(set)

    for path, term_col, code_col in [
        (BASE_CSV, "greek_description", "code"),
        (RICH_CSV, "term", "codes_pipe_sep"),
    ]:
        p = Path(path)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                term = normalize_text(row.get(term_col, "").strip())
                if not term:
                    continue
                if code_col == "codes_pipe_sep":
                    codes = [c.strip() for c in row.get(code_col, "").split("|") if c.strip()]
                else:
                    code = row.get(code_col, "").strip()
                    codes = [code] if code else []
                for c in codes:
                    if "-" not in c:
                        mapping[term].add(c)
    return mapping


def extract_dictionary_mentions(text: str, dictionary_map: Dict[str, set], confidence: float = 0.82) -> List[NERMentionPrediction]:
    norm_text, norm_to_orig = normalize_with_char_map(text)
    mentions: List[NERMentionPrediction] = []
    seen = set()

    for term in dictionary_map.keys():
        if len(term) < 3:
            continue
        for m in re.finditer(re.escape(term), norm_text):
            s_n, e_n = m.start(), m.end()
            if s_n >= len(norm_to_orig) or e_n - 1 >= len(norm_to_orig):
                continue
            s_o = norm_to_orig[s_n]
            e_o = norm_to_orig[e_n - 1] + 1
            key = (s_o, e_o)
            if key in seen:
                continue
            seen.add(key)
            mentions.append(
                NERMentionPrediction(
                    start=s_o,
                    end=e_o,
                    text=text[s_o:e_o],
                    confidence=confidence,
                )
            )

    mentions.sort(key=lambda x: (x.start, -(x.end - x.start)))
    return mentions


def extract_dictionary_codes(text: str, dictionary_map: Dict[str, set], max_codes: int = 20) -> List[str]:
    norm = normalize_text(text)
    found = set()
    for term, codes in dictionary_map.items():
        if term and term in norm:
            found.update(codes)
    return sorted(found)[:max_codes]


def merge_gold_with_dictionary_mentions(
    text: str,
    gold_mentions: List[MentionAnnotation],
    dictionary_map: Dict[str, set],
) -> List[MentionAnnotation]:
    merged = list(gold_mentions)
    occupied = {(m.start, m.end) for m in merged}

    for dm in extract_dictionary_mentions(text, dictionary_map=dictionary_map, confidence=0.7):
        span = (dm.start, dm.end)
        if span in occupied:
            continue
        merged.append(
            MentionAnnotation(
                start=dm.start,
                end=dm.end,
                code="DICT",
                mention=dm.text,
                confidence=0.7,
                source="dictionary",
            )
        )
        occupied.add(span)

    merged.sort(key=lambda x: (x.start, x.end))
    return merged
