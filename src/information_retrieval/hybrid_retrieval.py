"""
Hybrid IR: combine term-based (BM25) and dense (sentence embedding) rankings.

Uses **Reciprocal Rank Fusion** (RRF) so scores need no cross-method calibration:

``RRF(d) = Σ 1/(k + rank_i(d))`` over channels that ranked document ``d`` (typical ``k=60``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .types import RetrievalHit

if TYPE_CHECKING:
    from .embedding_retrieval import EmbeddingCodeRetriever
    from .term_retrieval import BM25CodeRetriever


class HybridRrfRetriever:
    """
    Fuse :class:`BM25CodeRetriever` and :class:`EmbeddingCodeRetriever` with RRF.

    Both sub-retrievers must already be ``fit()`` on the same corpus (same code order
    not required; fusion keys on ICD code string).
    """

    def __init__(
        self,
        bm25: BM25CodeRetriever,
        dense: EmbeddingCodeRetriever,
        *,
        rrf_k: int = 60,
        per_channel_pool: int | None = None,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        self._bm25 = bm25
        self._dense = dense
        self.rrf_k = rrf_k
        self._per_channel_pool = per_channel_pool
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def search(self, query_text: str, top_k: int = 20) -> list[RetrievalHit]:
        n_codes = len(self._bm25._codes)
        pool = self._per_channel_pool
        if pool is None:
            pool = min(n_codes, max(top_k * 6, 80))

        bm25_hits = self._bm25.search(query_text, top_k=min(pool, n_codes))
        dense_hits = self._dense.search(query_text, top_k=min(pool, n_codes))

        rank_bm25 = {h.code: r for r, h in enumerate(bm25_hits, start=1)}
        rank_dense = {h.code: r for r, h in enumerate(dense_hits, start=1)}
        k = self.rrf_k

        fused: dict[str, float] = {}
        for code, r in rank_bm25.items():
            fused[code] = fused.get(code, 0.0) + self.bm25_weight * (1.0 / (k + r))
        for code, r in rank_dense.items():
            fused[code] = fused.get(code, 0.0) + self.dense_weight * (1.0 / (k + r))

        ordered = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
        return [RetrievalHit(code=c, score=s, document_text="") for c, s in ordered]
