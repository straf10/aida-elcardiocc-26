"""
Information retrieval (IR) over the ELCardioCC ICD-10 label space.

The competition label set is treated as a fixed corpus of ~115 "documents" (each code or
range plus its Greek description). Free text (e.g. a discharge summary) is the query.

- **Lexical / term-based**: :class:`BM25CodeRetriever`, :class:`TfidfCodeRetriever`
  (BM25, TF–IDF + cosine; dictionary-aligned tokenization).

- **Semantic / vector-space (embeddings)**: :class:`EmbeddingCodeRetriever` (optional
  ``sentence-transformers``; zero-shot dense retrieval).

- **Hybrid IR**: :class:`HybridRrfRetriever` — Reciprocal Rank Fusion of BM25 + embeddings
  (``--retriever hybrid`` in ``evaluate.py``).

This complements **dictionary-based matching** in ``src.dictionary`` (substring/term→code
rules) with ranking-style retrieval over official code descriptions.

**Improving F1:** mine training mentions to expand each code document, filter hits with
:class:`IRPredictionParams` (relative score cut + cap), optionally union the term
dictionary; see ``evaluate.py`` for tuning helpers. Use
``python -m src.information_retrieval --retriever embedding`` for vector-space retrieval
(``sentence-transformers``); add ``--no-tune`` for a faster run without the validation grid.
``--source processed`` uses cleaned splits under
``data/processed/`` (mentions still joined from raw train JSONL by ``patient_id``).
"""

from typing import TYPE_CHECKING

from .corpus import (
    CodeDocument,
    apply_mention_expansion,
    build_code_documents,
    build_code_documents_with_mention_expansion,
    default_paths,
    mention_phrases_per_code,
)
from .evaluate import (
    evaluate_ir_on_records,
    fit_retriever,
    raw_records_by_patient_id,
    raw_rows_for_processed_split,
    tune_ir_hyperparams,
)
from .prediction import IRPredictionParams, filter_hits_by_relative_score, predict_codes_from_retriever
from .hybrid_retrieval import HybridRrfRetriever
from .term_retrieval import BM25CodeRetriever, TfidfCodeRetriever
from .types import RetrievalHit

if TYPE_CHECKING:
    from .embedding_retrieval import EmbeddingCodeRetriever

__all__ = [
    "BM25CodeRetriever",
    "CodeDocument",
    "EmbeddingCodeRetriever",
    "HybridRrfRetriever",
    "IRPredictionParams",
    "RetrievalHit",
    "TfidfCodeRetriever",
    "apply_mention_expansion",
    "build_code_documents",
    "build_code_documents_with_mention_expansion",
    "default_paths",
    "evaluate_ir_on_records",
    "filter_hits_by_relative_score",
    "fit_retriever",
    "mention_phrases_per_code",
    "predict_codes_from_retriever",
    "raw_records_by_patient_id",
    "raw_rows_for_processed_split",
    "tune_ir_hyperparams",
]


def __getattr__(name: str):
    if name == "EmbeddingCodeRetriever":
        from .embedding_retrieval import EmbeddingCodeRetriever as _E

        return _E
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
