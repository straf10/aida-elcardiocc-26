"""Optional fuzzy matching against ICD Greek descriptions (P1; requires rapidfuzz)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .normalize import normalize_text

if TYPE_CHECKING:
    pass


def fuzzy_fallback_predictions(
    text_norm: str,
    predicted: set[str],
    labelset: list[str],
    code_desc_map: dict[str, str],
    term_code_csv_path: str,
    *,
    similarity_threshold: int = 85,
    only_missing_codes: bool = True,
) -> set[str]:
    """
    Add codes whose Greek descriptions fuzzy-match the document when they are
    globally under-covered by the term dictionary (``only_missing_codes``).

    If ``rapidfuzz`` is not installed, returns ``predicted`` unchanged and prints once.
    """
    try:
        from rapidfuzz import fuzz as rfuzz
    except ImportError:
        print("WARNING: rapidfuzz not installed; fuzzy fallback disabled")
        return predicted

    from .matcher import load_term_code_csv

    term_map = load_term_code_csv(term_code_csv_path)
    covered: set[str] = set()
    for codes in term_map.values():
        covered |= codes

    candidates = [c for c in labelset if c not in covered] if only_missing_codes else list(labelset)

    out = set(predicted)
    for code in candidates:
        desc = (code_desc_map.get(code) or "").strip()
        if not desc:
            continue
        nd = normalize_text(desc)
        if len(nd) < 4:
            continue
        if rfuzz.partial_ratio(nd, text_norm) >= similarity_threshold:
            out.add(code)
    return out
