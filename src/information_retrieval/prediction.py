"""
Turn ranked retrieval hits into flat ICD code predictions (better precision than fixed top-k).

Optional union with dictionary substring matching for higher recall.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from src.dictionary.dictionary import predict_codes_for_text
except ImportError:
    from ..dictionary.dictionary import predict_codes_for_text

from .types import RetrievalHit


@dataclass
class IRPredictionParams:
    """Hyperparameters for IR → code list (tune on a validation split)."""

    search_top_k: int = 80
    """How many hits to retrieve before score filtering."""

    fraction_of_top_score: float = 0.22
    """Keep hits with score ≥ this fraction of the best hit (reduces noise vs fixed top-k)."""

    max_codes: int = 12
    """Hard cap on IR codes per document."""

    min_ir_codes: int = 0
    """If 0, allow empty IR set when all scores fail the fraction cut."""

    include_dictionary: bool = True
    """Union with ``predict_codes_for_text`` when ``term_code_map`` is provided."""

    code_fraction_thresholds: dict[str, float] | None = None
    """Optional per-code relative-score thresholds overriding ``fraction_of_top_score``."""


def filter_hits_by_relative_score(
    hits: list[RetrievalHit],
    *,
    fraction_of_top: float = 0.22,
    max_codes: int = 12,
    min_ir_codes: int = 0,
    code_fraction_thresholds: dict[str, float] | None = None,
) -> list[str]:
    """
    Keep codes from the top of the ranked list until the score falls below
    ``fraction_of_top * best_score``.

    Works for BM25 (positive unbounded scores) and TF-IDF cosine (typically in [0, 1]).
    """
    if not hits:
        return []
    best = hits[0].score
    if best <= 0:
        return []

    threshold = fraction_of_top * best
    codes: list[str] = []
    for h in hits:
        per_code_fraction = None
        if code_fraction_thresholds is not None:
            per_code_fraction = code_fraction_thresholds.get(h.code)
        effective_threshold = threshold
        if per_code_fraction is not None:
            effective_threshold = max(0.0, per_code_fraction) * best
        if h.score < effective_threshold:
            if code_fraction_thresholds is None:
                break
            continue
        codes.append(h.code)
        if len(codes) >= max_codes:
            break

    if not codes and min_ir_codes > 0:
        return [hits[0].code]
    return codes


def predict_codes_from_retriever(
    text: str,
    retriever,
    params: IRPredictionParams | None = None,
    *,
    term_code_map: dict | None = None,
) -> list[str]:
    """
    Retrieve, filter by relative score, optionally union dictionary predictions.

    ``retriever`` must implement ``search(query: str, top_k: int) -> list[RetrievalHit]``.
    """
    p = params or IRPredictionParams()
    hits = retriever.search(text, top_k=p.search_top_k)
    ir_codes = filter_hits_by_relative_score(
        hits,
        fraction_of_top=p.fraction_of_top_score,
        max_codes=p.max_codes,
        min_ir_codes=p.min_ir_codes,
        code_fraction_thresholds=p.code_fraction_thresholds,
    )
    out: set[str] = set(ir_codes)
    if p.include_dictionary and term_code_map is not None:
        out |= predict_codes_for_text(text, term_code_map)
    return sorted(out)
