"""Inference-time post-processing for XLM-R large predictions (specific→specific only)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableSet, Set

# Clinical parent codes when a more specific code in the labelset is predicted.
SPECIFIC_PARENT_CHILD: Dict[str, str] = {"I11": "I10", "I22": "I21"}


def apply_specific_parent_child(
    pred_codes_by_patient: Dict[int, List[str]],
    rules: Mapping[str, str] | None = None,
) -> Dict[int, List[str]]:
    """
    If a child code is present in predictions, ensure the mapped parent is also present.
    Mutates neither the input lists nor the rules dict; returns new dict with new lists.
    """
    rules = rules or SPECIFIC_PARENT_CHILD
    out: Dict[int, List[str]] = {}
    for pid, codes in pred_codes_by_patient.items():
        s: MutableSet[str] = set(codes)
        for child, parent in rules.items():
            if child in s and parent not in s:
                s.add(parent)
        out[pid] = sorted(s)
    return out


def apply_rules_to_code_set(codes: Iterable[str], rules: Mapping[str, str] | None = None) -> List[str]:
    """Apply parent propagation to a single iterable of codes."""
    s: Set[str] = set(codes)
    rules = rules or SPECIFIC_PARENT_CHILD
    for child, parent in rules.items():
        if child in s and parent not in s:
            s.add(parent)
    return sorted(s)
