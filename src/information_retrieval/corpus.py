"""
ICD-10 code corpus for IR: each label is one retrievable document (code + Greek text).

Uses the same paths and normalization as ``dictionary`` so queries and documents align
with dictionary-based matching preprocessing.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import NamedTuple

from dictionary.dictionary import (
    CODE_DESC_PATH,
    LABELSET_PATH,
    load_code_description_csv,
    load_labelset,
    normalize_text,
    tokenize,
)


class CodeDocument(NamedTuple):
    """One ICD-10 label as an IR document."""

    code: str
    """ICD-10 code or range string (e.g. ``I21``)."""

    raw_text: str
    """Human-readable document string (code + description) before normalization."""

    norm_text: str
    """``normalize_text(raw_text)`` for substring-consistent matching."""

    tokens: list[str]
    """Whitespace tokens after normalization."""


def default_paths() -> tuple[str, str]:
    """Return ``(labelset_path, code_description_csv_path)`` under the project root."""
    return str(LABELSET_PATH), str(CODE_DESC_PATH)


def build_code_documents(
    codes: list[str] | None = None,
    code_descriptions: dict[str, str] | None = None,
    *,
    labelset_path: str | None = None,
    code_desc_path: str | None = None,
) -> list[CodeDocument]:
    """
    Build the corpus of ICD-10 "documents" for retrieval.

    Each document is the code plus its Greek description (from lookup CSV). Codes without
    a description still appear with the code alone as the text.

    If ``codes`` is None, loads the competition labelset from ``labelset_path`` (default:
    project ``data/raw/.../labelset.txt``). If ``code_descriptions`` is None, loads from
    ``code_desc_path`` (default: ``icd10_greek_lookup.csv``).
    """
    lp = labelset_path or str(LABELSET_PATH)
    dp = code_desc_path or str(CODE_DESC_PATH)

    if codes is None:
        codes = load_labelset(lp)
    if code_descriptions is None:
        code_descriptions = load_code_description_csv(dp)

    documents: list[CodeDocument] = []
    for code in codes:
        desc = code_descriptions.get(code, "").strip()
        raw = f"{code} {desc}".strip() if desc else code
        norm = normalize_text(raw)
        documents.append(
            CodeDocument(code=code, raw_text=raw, norm_text=norm, tokens=tokenize(raw))
        )
    return documents


def mention_phrases_per_code(
    records: list[dict],
    *,
    min_phrase_len: int = 4,
    min_count: int = 2,
    max_phrases_per_code: int = 50,
) -> dict[str, list[str]]:
    """
    Mine normalized mention strings from ``mention_level_annotations``, grouped by code.

    Use **training documents only** when building phrases to avoid leaking validation/test
    labels into the retrieval index.
    """
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for rec in records:
        for m in rec.get("mention_level_annotations") or []:
            mention = (m.get("mention") or "").strip()
            code = (m.get("code") or "").strip()
            if not mention or not code:
                continue
            norm = normalize_text(mention)
            if len(norm) < min_phrase_len:
                continue
            counts[code][norm] += 1

    out: dict[str, list[str]] = {}
    for code, ctr in counts.items():
        phrases = [p for p, c in ctr.most_common(max_phrases_per_code) if c >= min_count]
        if phrases:
            out[code] = phrases
    return out


def apply_mention_expansion(
    documents: list[CodeDocument],
    phrases_by_code: dict[str, list[str]],
) -> list[CodeDocument]:
    """Append mined phrases to each document's ``raw_text`` (and re-tokenize)."""
    expanded: list[CodeDocument] = []
    for d in documents:
        extra = phrases_by_code.get(d.code)
        if not extra:
            expanded.append(d)
            continue
        raw = f"{d.raw_text} {' '.join(extra)}".strip()
        norm = normalize_text(raw)
        expanded.append(
            CodeDocument(code=d.code, raw_text=raw, norm_text=norm, tokens=tokenize(raw))
        )
    return expanded


def build_code_documents_with_mention_expansion(
    mining_records: list[dict],
    codes: list[str] | None = None,
    code_descriptions: dict[str, str] | None = None,
    *,
    labelset_path: str | None = None,
    code_desc_path: str | None = None,
    min_phrase_len: int = 4,
    min_count: int = 2,
    max_phrases_per_code: int = 50,
) -> list[CodeDocument]:
    """
    Base ICD documents plus phrases mined from ``mining_records`` mentions.

    Typical use: ``mining_records`` = training split only, then fit BM25/TF-IDF on the
    expanded corpus and evaluate on validation or test.
    """
    base = build_code_documents(
        codes=codes,
        code_descriptions=code_descriptions,
        labelset_path=labelset_path,
        code_desc_path=code_desc_path,
    )
    phrases = mention_phrases_per_code(
        mining_records,
        min_phrase_len=min_phrase_len,
        min_count=min_count,
        max_phrases_per_code=max_phrases_per_code,
    )
    return apply_mention_expansion(base, phrases)
