"""Enumerate OR / AND / k-of-n over **sets of base strategies** (no per-combo Python files).

``enumerate_auto_combo_specs`` builds union, intersection, and one or more ``k_of_n`` variants per
subset. Use **presets** (``COMBO_PRESETS``) to run the grid on different subsets of base slugs, and
``k_all`` / ``extra_k`` for a wider k search (e.g. 2-of-4 vs 3-of-4 vs strict 4-of-4).
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Set, Tuple

from evaluation.evaluator import evaluate_data
from ensemble_metaheuristic.strategy_bases import PatientPreds
from ensemble_metaheuristic.strategy_compositions import CompositionSpec, try_apply_composition

# Named subsets of **base strategy slugs** (must match ``strategy_bases.BASE_STRATEGY_ORDER`` names).
# Run with ``--combo-preset NAME`` (repeatable). If none, all bases present on val are used.
COMBO_PRESETS: Dict[str, Tuple[str, ...]] = {
    "weighted_family": (
        "weighted",
        "weighted_majority_restarts",
        "weighted_top_k",
        "weighted_freq_buckets",
        "weighted_global_loose",
        "weighted_global_tight",
        "weighted_top_k_loose",
        "weighted_freq_loose",
    ),
    "committee": ("best_single_model", "committee_or", "committee_majority"),
    "routing_stack": ("per_label_routing", "correction", "weighted"),
    "k2_stack": ("weighted", "per_label_routing", "correction"),
    "diverse_mix": (
        "weighted",
        "weighted_majority_restarts",
        "committee_majority",
        "per_label_routing",
        "correction",
    ),
    "no_committee_or": (
        "best_single_model",
        "committee_majority",
        "weighted",
        "weighted_majority_restarts",
        "per_label_routing",
        "correction",
    ),
}


def combo_preset_names() -> Tuple[str, ...]:
    return tuple(sorted(COMBO_PRESETS.keys()))


def _majority_k(n_inputs: int) -> int:
    return max(1, n_inputs // 2 + 1)


def _k_values_for_n(
    n: int,
    *,
    k_all: bool,
    extra_k: Sequence[int],
) -> Tuple[int, ...]:
    """Distinct k in ``[1, n]`` for k-of-n: optionally every k, plus any ``extra_k`` that land in range."""
    vals: Set[int] = set()
    if k_all:
        vals.update(range(1, n + 1))
    else:
        vals.add(_majority_k(n))
    for k in extra_k:
        kk = int(k)
        if 1 <= kk <= n:
            vals.add(kk)
    return tuple(sorted(vals))


def enumerate_auto_combo_specs(
    base_slugs: Sequence[str],
    sizes: Tuple[int, ...] = (2, 3, 4),
    *,
    k_all: bool = False,
    extra_k: Tuple[int, ...] = (),
    max_specs: Optional[int] = None,
) -> List[CompositionSpec]:
    """
    Every unordered subset of ``sizes`` from ``base_slugs`` × ``union`` | ``intersection`` |
    ``k_of_n`` for each k in ``_k_values_for_n`` (majority only unless ``k_all`` / ``extra_k``).
    """
    slugs = tuple(sorted({str(s).strip() for s in base_slugs if str(s).strip()}))
    out: List[CompositionSpec] = []
    for n in sizes:
        if n < 2 or n > len(slugs):
            continue
        ks = _k_values_for_n(n, k_all=k_all, extra_k=extra_k)
        for combo in combinations(slugs, n):
            if max_specs is not None and len(out) >= max_specs:
                return out
            inp = list(combo)
            key = "__".join(inp)
            out.append(CompositionSpec(slug=f"ac_u__{key}", op="union", inputs=list(inp)))
            out.append(CompositionSpec(slug=f"ac_i__{key}", op="intersection", inputs=list(inp)))
            for k in ks:
                if max_specs is not None and len(out) >= max_specs:
                    return out
                out.append(CompositionSpec(slug=f"ac_k{k}__{key}", op="k_of_n", inputs=list(inp), k=int(k)))
    return out


def enumerate_combo_specs_multi_presets(
    available_slugs: Sequence[str],
    preset_names: Sequence[str],
    sizes: Tuple[int, ...],
    *,
    k_all: bool,
    extra_k: Tuple[int, ...],
    max_specs: Optional[int] = None,
) -> Tuple[List[CompositionSpec], List[str]]:
    """
    Run the grid on each preset's bases (intersected with ``available_slugs``), concatenate, dedupe by slug.

    Returns ``(specs, warnings)`` where warnings include unknown preset names or empty presets.
    """
    avail = {str(s).strip() for s in available_slugs if str(s).strip()}
    warnings: List[str] = []
    seen: Set[str] = set()
    out: List[CompositionSpec] = []

    for pname in preset_names:
        key = str(pname).strip()
        if key not in COMBO_PRESETS:
            warnings.append(f"unknown combo preset {key!r} (known: {', '.join(combo_preset_names())})")
            continue
        want = set(COMBO_PRESETS[key])
        bases = tuple(sorted(want & avail))
        if len(bases) < 2:
            warnings.append(f"preset {key!r} has fewer than 2 bases after filtering; skipped")
            continue
        chunk = enumerate_auto_combo_specs(
            bases, sizes=sizes, k_all=k_all, extra_k=extra_k, max_specs=None,
        )
        for sp in chunk:
            if sp.slug in seen:
                continue
            seen.add(sp.slug)
            out.append(sp)
            if max_specs is not None and len(out) >= max_specs:
                return out, warnings
    return out, warnings


def evaluate_combo_grid(
    specs: List[CompositionSpec],
    base_preds: Dict[str, PatientPreds],
    patient_ids: List[int],
    gt_data: Dict,
    all_labels: List[str],
) -> List[Tuple[CompositionSpec, dict]]:
    """Return ``(spec, metrics)`` for each spec that could be applied (skips missing bases)."""
    rows: List[Tuple[CompositionSpec, dict]] = []
    for spec in specs:
        merged = try_apply_composition(spec, base_preds, patient_ids)
        if merged is None:
            continue
        m = evaluate_data(gt_data, merged, label_space=all_labels)
        rows.append((spec, m))
    return rows


def top_specs_by_micro_f1(
    scored: List[Tuple[CompositionSpec, dict]],
    top_n: int,
) -> List[CompositionSpec]:
    if top_n <= 0:
        return []
    ranked = sorted(scored, key=lambda x: (-float(x[1]["micro_f1"]), x[0].slug))
    return [spec for spec, _ in ranked[:top_n]]
