"""
Dictionary baseline facade — re-exports public API for ``from dictionary.dictionary import …``.

Implementation lives in sibling modules (``normalize``, ``matcher``, ``rules``, etc.).
"""

from __future__ import annotations

from preprocessing.io_utils import (
    CODE_DESC_PATH,
    LABELSET_PATH,
    PROJECT_ROOT,
    TERM_CODE_CSV,
    TEST_PATH,
    TRAIN_PATH,
    load_jsonl,
    load_labelset,
    resolve_patient_id,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "dictionary_baseline"

from .build import build_mention_dictionary
from .export import export_code_lookup, export_predictions_jsonl, load_code_description_csv
from .matcher import build_automaton, load_term_code_csv, predict_codes_for_text
from .normalize import normalize_text, strip_accents, tokenize
from .reports import (
    coverage_report,
    flatten_gold_groups,
    fp_fn_analysis,
    label_frequency_report,
    official_evaluation,
    tokenization_report,
)
from .rules import BLACKLIST_TERMS, apply_cooccurrence_rules, apply_rule_based_boosts

__all__ = [
    "BLACKLIST_TERMS",
    "CODE_DESC_PATH",
    "LABELSET_PATH",
    "OUTPUT_DIR",
    "PROJECT_ROOT",
    "TERM_CODE_CSV",
    "TEST_PATH",
    "TRAIN_PATH",
    "apply_cooccurrence_rules",
    "apply_rule_based_boosts",
    "build_automaton",
    "build_mention_dictionary",
    "coverage_report",
    "export_code_lookup",
    "export_predictions_jsonl",
    "flatten_gold_groups",
    "fp_fn_analysis",
    "label_frequency_report",
    "load_code_description_csv",
    "load_jsonl",
    "load_labelset",
    "load_term_code_csv",
    "normalize_text",
    "official_evaluation",
    "predict_codes_for_text",
    "resolve_patient_id",
    "strip_accents",
    "tokenize",
    "tokenization_report",
]

if __name__ == "__main__":
    from .cli import main as _main

    _main()
