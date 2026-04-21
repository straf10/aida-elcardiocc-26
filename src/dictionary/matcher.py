"""Term→code loading, Aho-Corasick automaton, and prediction."""

from __future__ import annotations

import csv
from pathlib import Path
import ahocorasick

from . import filters
from .normalize import normalize_text
from .rules import BLACKLIST_TERMS, apply_rule_based_boosts


def load_term_code_csv(csv_path: str, blacklist: set[str] | None = None) -> dict:
    """
    Load the term→code dictionary from CSV.
    Columns: term, codes_pipe_sep (e.g. "I10|I11")
    Applies blacklist filtering and text normalization on load.
    Returns: {normalized_term: set_of_codes}
    """
    bl = blacklist if blacklist is not None else BLACKLIST_TERMS
    term_code: dict[str, set[str]] = {}
    if not Path(csv_path).exists():
        print(f"WARNING: {csv_path} not found")
        return term_code
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = normalize_text(row["term"].strip())
            codes = {c.strip() for c in row["codes_pipe_sep"].split("|") if c.strip()}
            if term and term not in bl:
                term_code[term] = codes
    return term_code


def build_automaton(
    term_code_map: dict[str, set[str]],
    *,
    word_boundary: bool = False,
) -> ahocorasick.Automaton | dict[str, set[str]]:
    """
    Build an Aho-Corasick automaton from the term→codes mapping.

    If ``word_boundary`` is True, each word is stored and searched with leading/trailing
    spaces to reduce fragment false positives (P1).

    If ``term_code_map`` is empty, returns the empty dict.
    """
    if not term_code_map:
        return {}
    automaton = ahocorasick.Automaton()
    for term, codes in term_code_map.items():
        key = f" {term} " if word_boundary else term
        automaton.add_word(key, codes)
    automaton.make_automaton()
    return automaton


def _match_dictionary_only(
    full_norm: str,
    matcher: dict[str, set[str]] | ahocorasick.Automaton,
    *,
    word_boundary: bool = False,
) -> set:
    """Codes from dictionary substring (or word-boundary) matching only — no boosts."""
    predicted: set[str] = set()
    if word_boundary:
        padded = f" {full_norm} "
        if isinstance(matcher, dict):
            for term, codes in matcher.items():
                if f" {term} " in padded:
                    predicted.update(codes)
        else:
            for _, codes in matcher.iter(padded):
                predicted.update(codes)
    else:
        if isinstance(matcher, dict):
            for term, codes in matcher.items():
                if term in full_norm:
                    predicted.update(codes)
        else:
            for _, codes in matcher.iter(full_norm):
                predicted.update(codes)
    return predicted


def predict_codes_for_text(
    text: str,
    matcher: dict[str, set[str]] | ahocorasick.Automaton,
    *,
    config: "DictionaryConfig | None" = None,
    labelset: list[str] | None = None,
    code_desc_map: dict[str, str] | None = None,
) -> set:
    """
    Predict ICD-10 codes for a single document.

    When ``config`` is None, behavior matches the legacy pipeline (substring match,
    hardcoded boosts and co-occurrence).

    When ``config`` is provided, optional P1 features (word boundaries, YAML rules,
    section/negation filters, fuzzy fallback) are applied per the config.
    """
    full_norm = normalize_text(text)
    word_boundary = False
    if config is not None:
        word_boundary = bool((config.matching or {}).get("word_boundary", False))

    predicted = _match_dictionary_only(
        full_norm, matcher, word_boundary=word_boundary
    )

    if config is None:
        predicted = apply_rule_based_boosts(full_norm, predicted)
        return predicted

    boosts = config.rule_based_boosts
    cooc = config.cooccurrence_rules
    predicted = apply_rule_based_boosts(
        full_norm, predicted, boosts=boosts, cooccurrence_rules=cooc
    )

    mcfg = config.matching or {}
    sec_cfg = mcfg.get("section_filter") or {}
    if sec_cfg.get("enabled"):

        def _md(t: str) -> set:
            return _match_dictionary_only(
                t, matcher, word_boundary=word_boundary
            )

        predicted = filters.apply_section_filter(
            full_norm, predicted, sec_cfg, _md
        )

    neg_cfg = mcfg.get("negation_filter") or {}
    if neg_cfg.get("enabled"):
        predicted = filters.apply_negation_filter(full_norm, predicted, neg_cfg)

    fuzzy_cfg = config.fuzzy_fallback or {}
    if fuzzy_cfg.get("enabled") and labelset is not None and code_desc_map:
        from . import fuzzy as fuzzy_mod

        predicted = fuzzy_mod.fuzzy_fallback_predictions(
            full_norm,
            predicted=predicted,
            labelset=labelset,
            code_desc_map=code_desc_map,
            term_code_csv_path=config.paths.get("term_code_csv", ""),
            similarity_threshold=int(fuzzy_cfg.get("similarity_threshold", 85)),
            only_missing_codes=bool(fuzzy_cfg.get("only_missing_codes", True)),
        )

    return predicted
