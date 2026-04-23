from __future__ import annotations

"""Dictionary resource utilities for the main NER->EL pipeline.

This module consumes the shared dictionary stack under src/dictionary so NER and
dictionary pipelines use the same normalization, blacklist, and matching behavior.
"""

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Iterable, List

try:
    import ahocorasick
except ImportError:  # pragma: no cover
    ahocorasick = None

from dictionary.config import load_dictionary_config
from dictionary.export import load_code_description_csv
from dictionary.matcher import load_term_code_csv, predict_codes_for_text
from dictionary.normalize import normalize_text, strip_accents
from .schemas import MentionAnnotation, NERMentionPrediction

if TYPE_CHECKING:
    from dictionary.config import DictionaryConfig


_ALNUM_SPACE_RE = re.compile(r"[α-ωa-z0-9\s]")
_MENTION_AUTOMATON_CACHE: dict[tuple[int, bool, int], object] = {}


def build_mention_automaton(dictionary_map: Dict[str, set], *, word_boundary: bool = False):
    """Build/cache Aho-Corasick automaton for dictionary mention extraction."""
    cache_key = (id(dictionary_map), bool(word_boundary), len(dictionary_map))
    cached = _MENTION_AUTOMATON_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if ahocorasick is None:
        _MENTION_AUTOMATON_CACHE[cache_key] = None
        return None
    automaton = ahocorasick.Automaton()
    for term in dictionary_map.keys():
        if len(term) < 3:
            continue
        key = f" {term} " if word_boundary else term
        automaton.add_word(key, term)
    automaton.make_automaton()
    _MENTION_AUTOMATON_CACHE[cache_key] = automaton
    return automaton


def normalize_with_char_map(text: str) -> tuple[str, list[int]]:
    out_chars: List[str] = []
    norm_to_orig: List[int] = []
    prev_space = True

    for i, ch in enumerate(text):
        low = strip_accents(ch.lower())
        candidate = low if _ALNUM_SPACE_RE.match(low) else " "
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


def load_dictionary_candidates(
    *,
    labelset: Iterable[str] | None = None,
    config_path: str | None = None,
) -> Dict[str, set]:
    """Load term->codes from shared dictionary resources.

    - Uses dictionary YAML path resolution and blacklist via `load_dictionary_config`.
    - Reuses `dictionary.matcher.load_term_code_csv` for canonical normalization/filtering.
    - Adds normalized ICD Greek descriptions as fallback terms.
    - Keeps range codes when they exist in the provided labelset.
    """
    cfg = load_dictionary_config(config_path)
    allowed_codes = set(labelset) if labelset is not None else None
    mapping = defaultdict(set)

    term_code_map = load_term_code_csv(
        cfg.paths["term_code_csv"],
        blacklist=cfg.blacklist,
    )
    for term, codes in term_code_map.items():
        if not term:
            continue
        filtered = set(codes)
        if allowed_codes is not None:
            filtered = {c for c in filtered if c in allowed_codes}
        if filtered:
            mapping[term].update(filtered)

    code_desc_map = load_code_description_csv(cfg.paths["code_description_csv"])
    for code, desc in code_desc_map.items():
        if allowed_codes is not None and code not in allowed_codes:
            continue
        term = normalize_text(desc)
        if not term:
            continue
        if term in cfg.blacklist:
            continue
        mapping[term].add(code)

    return dict(mapping)


def extract_dictionary_mentions(
    text: str,
    dictionary_map: Dict[str, set],
    confidence: float = 0.82,
    *,
    word_boundary: bool = False,
) -> List[NERMentionPrediction]:
    """Extract mention spans from dictionary term hits with optional word boundaries."""
    norm_text, norm_to_orig = normalize_with_char_map(text)
    mentions: List[NERMentionPrediction] = []
    seen = set()
    scan_text = f" {norm_text} " if word_boundary else norm_text
    mention_automaton = build_mention_automaton(dictionary_map, word_boundary=word_boundary)

    if mention_automaton is not None:
        for end_idx, term in mention_automaton.iter(scan_text):
            span_len = len(term) + (2 if word_boundary else 0)
            start_idx = int(end_idx) - span_len + 1
            if word_boundary:
                s_n = start_idx + 1
                e_n = int(end_idx)
            else:
                s_n = start_idx
                e_n = int(end_idx) + 1
            if e_n <= s_n:
                continue
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
    else:
        for term in dictionary_map.keys():
            if len(term) < 3:
                continue
            needle = f" {term} " if word_boundary else term
            for m in re.finditer(re.escape(needle), scan_text):
                if word_boundary:
                    s_n = m.start() + 1
                    e_n = m.end() - 1
                else:
                    s_n = m.start()
                    e_n = m.end()
                if e_n <= s_n:
                    continue
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


def extract_dictionary_codes(
    text: str,
    matcher,
    config: "DictionaryConfig",
    *,
    labelset: List[str] | None = None,
    code_desc_map: Dict[str, str] | None = None,
) -> List[str]:
    """Predict dictionary codes by delegating to shared dictionary matcher logic."""
    if matcher is None:
        return []
    codes = predict_codes_for_text(
        text,
        matcher,
        config=config,
        labelset=labelset,
        code_desc_map=code_desc_map,
    )
    if labelset is not None:
        allowed = set(labelset)
        codes = {c for c in codes if c in allowed}
    return sorted(codes)


def merge_gold_with_dictionary_mentions(
    text: str,
    gold_mentions: List[MentionAnnotation],
    dictionary_map: Dict[str, set],
    *,
    word_boundary: bool = False,
) -> List[MentionAnnotation]:
    merged = list(gold_mentions)
    occupied = {(m.start, m.end) for m in merged}

    for dm in extract_dictionary_mentions(
        text,
        dictionary_map=dictionary_map,
        confidence=0.7,
        word_boundary=word_boundary,
    ):
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
