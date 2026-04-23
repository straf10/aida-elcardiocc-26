"""Load and resolve dictionary baseline configuration (YAML)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.config_utils import load_config
from preprocessing.io_utils import PROJECT_ROOT, DICTIONARY_CONFIG_PATH


def resolve_path(rel_or_abs: str) -> str:
    """Resolve a path relative to project root, or return absolute paths unchanged."""
    p = Path(rel_or_abs)
    if p.is_absolute():
        return str(p.resolve())
    return str((PROJECT_ROOT / rel_or_abs).resolve())


@dataclass
class DictionaryConfig:
    """Runtime config for dictionary matching, build, and optional P1 features."""

    paths: dict[str, str]
    matching: dict[str, Any]
    build: dict[str, Any]
    fuzzy_fallback: dict[str, Any]
    blacklist: set[str]
    cooccurrence_rules: list[dict[str, Any]] | None
    rule_based_boosts: list[dict[str, Any]] | None
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


def _default_paths() -> dict[str, str]:
    from preprocessing.io_utils import (
        TRAIN_PATH,
        TEST_PATH,
        LABELSET_PATH,
        TERM_CODE_CSV,
        TRAIN_ONLY_TERM_CODE_CSV,
        CODE_DESC_PATH,
        DICTIONARY_OUTPUT_DIR,
    )

    return {
        "train_jsonl": TRAIN_PATH,
        "test_jsonl": TEST_PATH,
        "labelset": LABELSET_PATH,
        "term_code_csv": TERM_CODE_CSV,
        "train_only_term_code": TRAIN_ONLY_TERM_CODE_CSV,
        "code_description_csv": CODE_DESC_PATH,
        "output_dir": DICTIONARY_OUTPUT_DIR,
        # Labeled split (same as ``evaluation.config.yaml`` ``data.test_path``) for ``compare_methods``.
        "compare_eval_test_jsonl": "data/processed/test.jsonl",
        "compare_predictions_jsonl": "outputs/predictions/dictionary_baseline/predictions.jsonl",
    }


def _default_blacklist_from_rules() -> set[str]:
    from .rules import BLACKLIST_TERMS

    return set(BLACKLIST_TERMS)


def load_dictionary_config(path: str | None = None) -> DictionaryConfig:
    """
    Load dictionary YAML. If ``path`` is None or missing, use built-in defaults
    matching legacy hardcoded behavior.
    """
    if not path:
        return _defaults_only_config()

    p = Path(path)
    if not p.exists():
        return _defaults_only_config()

    raw = load_config(str(p))
    if not raw:
        return _defaults_only_config()

    paths_in = raw.get("paths") or {}
    default_paths = _default_paths()
    paths: dict[str, str] = {}
    for key, default_val in default_paths.items():
        rel = paths_in.get(key, default_val)
        if isinstance(rel, str) and not Path(rel).is_absolute():
            paths[key] = resolve_path(rel)
        elif isinstance(rel, str):
            paths[key] = str(Path(rel).resolve())
        else:
            paths[key] = default_val

    matching = raw.get("matching") or {}
    build = raw.get("build") or {}
    fuzzy_fallback = raw.get("fuzzy_fallback") or {}

    bl = raw.get("blacklist")
    if isinstance(bl, list):
        blacklist = {str(x).strip() for x in bl if str(x).strip()}
    else:
        blacklist = _default_blacklist_from_rules()

    cooccurrence_rules_raw = raw.get("cooccurrence_rules")
    if isinstance(cooccurrence_rules_raw, list) and len(cooccurrence_rules_raw) > 0:
        cooccurrence_rules = cooccurrence_rules_raw
    else:
        cooccurrence_rules = None

    rule_based_boosts_raw = raw.get("rule_based_boosts")
    if isinstance(rule_based_boosts_raw, list) and len(rule_based_boosts_raw) > 0:
        rule_based_boosts = rule_based_boosts_raw
    else:
        rule_based_boosts = None

    return DictionaryConfig(
        paths=paths,
        matching=matching,
        build=build,
        fuzzy_fallback=fuzzy_fallback,
        blacklist=blacklist,
        cooccurrence_rules=cooccurrence_rules,
        rule_based_boosts=rule_based_boosts,
        raw=raw,
    )


def _defaults_only_config() -> DictionaryConfig:
    """Config object with legacy defaults (no YAML file required)."""
    return DictionaryConfig(
        paths=_default_paths(),
        matching={
            "word_boundary": False,
            "section_filter": {"enabled": False},
            "negation_filter": {"enabled": False},
        },
        build={"min_term_len": 4, "min_term_count": 1, "min_code_ratio": 0.20},
        fuzzy_fallback={"enabled": False, "similarity_threshold": 85, "only_missing_codes": True},
        blacklist=_default_blacklist_from_rules(),
        cooccurrence_rules=None,
        rule_based_boosts=None,
        raw={},
    )


def get_config_path_default() -> str:
    return DICTIONARY_CONFIG_PATH
