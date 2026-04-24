"""Load and apply declarative compositions over base strategy predictions (OR / AND / k-of-n)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml

from ensemble_metaheuristic.strategies import (
    merge_preds_intersection,
    merge_preds_k_of_n,
    merge_preds_union,
)

CompositionOp = Literal["union", "intersection", "k_of_n"]


@dataclass(frozen=True)
class CompositionSpec:
    slug: str
    op: CompositionOp
    inputs: List[str]
    k: Optional[int] = None


def default_compositions_path() -> Path:
    return Path(__file__).resolve().parent / "strategy_compositions.yaml"


def load_composition_specs(path: Optional[Path] = None) -> List[CompositionSpec]:
    p = path or default_compositions_path()
    if not p.is_file():
        return []
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    items = raw.get("compositions") or []
    out: List[CompositionSpec] = []
    for i, row in enumerate(items):
        if not isinstance(row, dict):
            raise ValueError(f"compositions[{i}] must be a mapping, got {type(row).__name__}")
        slug = str(row.get("slug", "")).strip()
        op = str(row.get("op", "")).strip().lower()
        inputs = row.get("inputs") or []
        if not slug:
            raise ValueError(f"compositions[{i}]: missing slug")
        if op not in ("union", "intersection", "k_of_n"):
            raise ValueError(f"compositions[{i!r} {slug!r}]: invalid op {op!r}")
        if not isinstance(inputs, list) or not all(isinstance(x, str) and str(x).strip() for x in inputs):
            raise ValueError(f"compositions[{i!r} {slug!r}]: inputs must be a list of non-empty strings")
        clean_inputs = [str(x).strip() for x in inputs]
        k = row.get("k")
        if op == "k_of_n":
            if k is None:
                raise ValueError(f"compositions[{i!r} {slug!r}]: k_of_n requires integer k")
            k = int(k)
            if k < 1 or k > len(clean_inputs):
                raise ValueError(f"compositions[{i!r} {slug!r}]: k must be in 1..len(inputs)")
        else:
            if len(clean_inputs) < 2:
                raise ValueError(f"compositions[{i!r} {slug!r}]: need at least two inputs for {op}")
            k = None
        out.append(CompositionSpec(slug=slug, op=op, inputs=clean_inputs, k=k))
    return out


def apply_composition(
    spec: CompositionSpec,
    base_preds: Dict[str, Dict[int, List[str]]],
    patient_ids: List[int],
) -> Dict[int, List[str]]:
    """Merge cached base predictions; raises ``KeyError`` if a base slug is missing."""
    preds_list = [base_preds[s] for s in spec.inputs]
    if spec.op == "union":
        a, b = preds_list[0], preds_list[1]
        merged = merge_preds_union(a, b, patient_ids)
        for rest in preds_list[2:]:
            merged = merge_preds_union(merged, rest, patient_ids)
        return merged
    if spec.op == "intersection":
        a, b = preds_list[0], preds_list[1]
        merged = merge_preds_intersection(a, b, patient_ids)
        for rest in preds_list[2:]:
            merged = merge_preds_intersection(merged, rest, patient_ids)
        return merged
    assert spec.k is not None
    return merge_preds_k_of_n(preds_list, patient_ids, int(spec.k))


def try_apply_composition(
    spec: CompositionSpec,
    base_preds: Dict[str, Dict[int, List[str]]],
    patient_ids: List[int],
) -> Optional[Dict[int, List[str]]]:
    if not all(s in base_preds for s in spec.inputs):
        return None
    try:
        return apply_composition(spec, base_preds, patient_ids)
    except Exception:
        return None
