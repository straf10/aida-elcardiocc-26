"""
Semantic (dense vector) retrieval: embed code+description documents and the query, rank by cosine similarity.

Zero-shot with respect to task-specific training; the encoder is typically a pretrained
multilingual sentence model. Requires ``sentence-transformers`` (optional dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .types import RetrievalHit

if TYPE_CHECKING:
    from .corpus import CodeDocument


def _require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "EmbeddingCodeRetriever needs the sentence-transformers package. "
            "Install with: pip install sentence-transformers"
        ) from e
    return SentenceTransformer


class EmbeddingCodeRetriever:
    """
    Embed ICD-10 documents and queries with a SentenceTransformer model, rank by cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        query_prefix: str = "",
        doc_prefix: str = "",
    ) -> None:
        SentenceTransformer = _require_sentence_transformers()
        self._model = SentenceTransformer(model_name)
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._codes: list[str] = []
        self._document_texts: list[str] = []
        self._doc_embeddings = None

    def fit(self, documents: list[CodeDocument]) -> EmbeddingCodeRetriever:
        self._codes = [d.code for d in documents]
        self._document_texts = [d.raw_text for d in documents]
        texts = [self._doc_prefix + t for t in self._document_texts] if self._doc_prefix else self._document_texts
        self._doc_embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return self

    def search(self, query_text: str, top_k: int = 20) -> list[RetrievalHit]:
        from sklearn.metrics.pairwise import cosine_similarity

        if self._doc_embeddings is None:
            raise RuntimeError("Call fit() before search().")
        query = self._query_prefix + query_text if self._query_prefix else query_text
        q_emb = self._model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        sims = cosine_similarity(q_emb, self._doc_embeddings).ravel()
        order = sims.argsort()[::-1][:top_k]
        return [
            RetrievalHit(
                code=self._codes[i],
                score=float(sims[i]),
                document_text=self._document_texts[i],
            )
            for i in order
        ]
