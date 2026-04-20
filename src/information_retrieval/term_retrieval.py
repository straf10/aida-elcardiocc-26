"""
Term-based retrieval: BM25 and TF–IDF + cosine (lexical / non-parametric).

Patient (or note) text is the query; ranked ICD-10 code descriptions are the hits.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

from dictionary.dictionary import tokenize

if TYPE_CHECKING:
    from .corpus import CodeDocument

from .types import RetrievalHit


def _analyzer(doc: str) -> list[str]:
    return tokenize(doc)


class TfidfCodeRetriever:
    """
    Vector-space retrieval over normalized tokens using TF–IDF weights and cosine similarity.

    Documents are fixed at fit time (typically the 115 label descriptions).
    Requires ``scikit-learn`` (imported on first ``fit()``).
    """

    def __init__(self) -> None:
        self._vectorizer = None
        self._doc_matrix = None
        self._codes: list[str] = []

    def fit(self, documents: list[CodeDocument]) -> TfidfCodeRetriever:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(analyzer=_analyzer)
        raw_docs = [d.raw_text for d in documents]
        self._codes = [d.code for d in documents]
        self._doc_matrix = self._vectorizer.fit_transform(raw_docs)
        return self

    def search(self, query_text: str, top_k: int = 20) -> list[RetrievalHit]:
        from sklearn.metrics.pairwise import cosine_similarity

        if self._doc_matrix is None or self._vectorizer is None:
            raise RuntimeError("Call fit() before search().")
        q = self._vectorizer.transform([query_text])
        sims = cosine_similarity(q, self._doc_matrix).ravel()
        order = sims.argsort()[::-1][:top_k]
        return [
            RetrievalHit(code=self._codes[i], score=float(sims[i]), document_text="")
            for i in order
        ]


class BM25CodeRetriever:
    """
    Okapi BM25 over the code+description corpus (same tokenization as the dictionary).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._codes: list[str] = []
        self._doc_freqs: list[dict[str, int]] = []
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._N: int = 0
        self._idf: dict[str, float] = {}
        self._corpus_tokens: list[list[str]] = []

    def fit(self, documents: list[CodeDocument]) -> BM25CodeRetriever:
        self._codes = [d.code for d in documents]
        self._corpus_tokens = [list(d.tokens) for d in documents]
        self._doc_freqs = []
        self._doc_lens = []
        df = defaultdict(int)
        for toks in self._corpus_tokens:
            self._doc_lens.append(len(toks))
            tf: dict[str, int] = defaultdict(int)
            for t in toks:
                tf[t] += 1
            for t in tf:
                df[t] += 1
            self._doc_freqs.append(dict(tf))

        self._N = len(self._corpus_tokens)
        self._avgdl = sum(self._doc_lens) / self._N if self._N else 0.0

        # Robertson–Walker IDF variant (smooth, positive)
        for term, dfi in df.items():
            self._idf[term] = math.log(1.0 + (self._N - dfi + 0.5) / (dfi + 0.5))

        return self

    def search(self, query_text: str, top_k: int = 20) -> list[RetrievalHit]:
        if not self._codes:
            raise RuntimeError("Call fit() before search().")

        q_tokens = tokenize(query_text)
        if not q_tokens:
            return []

        scores = [0.0] * self._N
        for qi in q_tokens:
            idf = self._idf.get(qi)
            if idf is None:
                continue
            for i, doc_tf in enumerate(self._doc_freqs):
                f = doc_tf.get(qi, 0)
                if f == 0:
                    continue
                dl = self._doc_lens[i]
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom

        indexed = sorted(range(self._N), key=lambda i: scores[i], reverse=True)
        out: list[RetrievalHit] = []
        for i in indexed[:top_k]:
            if scores[i] > 0:
                out.append(RetrievalHit(code=self._codes[i], score=scores[i], document_text=""))
        return out
