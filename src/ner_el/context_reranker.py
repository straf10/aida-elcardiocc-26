from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from information_retrieval.embedding_retrieval import EmbeddingCodeRetriever


class ContextReranker:
    """Semantic re-ranker for mention candidates using local context windows."""

    EMBEDDINGS_FILENAME = "reranker_code_embeddings.npy"
    META_FILENAME = "reranker_meta.json"

    def __init__(
        self,
        code_desc_map: Dict[str, str],
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        window_chars: int = 200,
    ) -> None:
        self.code_desc_map = code_desc_map
        self.model_name = str(model_name)
        self.window_chars = int(window_chars)
        self._retriever = EmbeddingCodeRetriever(model_name=self.model_name)
        self._codes: List[str] = []
        self._code_to_idx: Dict[str, int] = {}
        self._embeddings: np.ndarray | None = None

    def fit(self, labelset: Iterable[str]) -> "ContextReranker":
        codes = [str(code) for code in labelset if str(code) in self.code_desc_map]
        texts = [self.code_desc_map[c] for c in codes]
        embeddings = self._retriever._model.encode(  # noqa: SLF001
            texts,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self._codes = codes
        self._code_to_idx = {c: i for i, c in enumerate(codes)}
        self._embeddings = np.asarray(embeddings, dtype=np.float32)
        return self

    def save(self, artifact_dir: str) -> None:
        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self._embeddings is None:
            raise RuntimeError("Call fit() before save().")
        np.save(out_dir / self.EMBEDDINGS_FILENAME, self._embeddings)
        with (out_dir / self.META_FILENAME).open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model_name": self.model_name,
                    "window_chars": self.window_chars,
                    "codes": self._codes,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, artifact_dir: str, code_desc_map: Dict[str, str]) -> "ContextReranker":
        out_dir = Path(artifact_dir)
        with (out_dir / cls.META_FILENAME).open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        obj = cls(
            code_desc_map=code_desc_map,
            model_name=str(meta.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2")),
            window_chars=int(meta.get("window_chars", 200)),
        )
        obj._codes = [str(c) for c in meta.get("codes", [])]
        obj._code_to_idx = {c: i for i, c in enumerate(obj._codes)}
        obj._embeddings = np.load(out_dir / cls.EMBEDDINGS_FILENAME)
        return obj

    def score_batch(
        self,
        windows: List[str],
        candidate_codes_per_window: List[List[str]],
    ) -> List[Dict[str, float]]:
        if not windows:
            return []
        if self._embeddings is None:
            raise RuntimeError("ContextReranker is not fitted.")
        query_embeddings = self._retriever._model.encode(  # noqa: SLF001
            windows,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        q = np.asarray(query_embeddings, dtype=np.float32)
        sims_all = q @ self._embeddings.T
        rows: List[Dict[str, float]] = []
        for i, candidate_codes in enumerate(candidate_codes_per_window):
            out: Dict[str, float] = {}
            for code in candidate_codes:
                idx = self._code_to_idx.get(str(code))
                out[str(code)] = float(sims_all[i, idx]) if idx is not None else -1.0
            rows.append(out)
        return rows

    def score(self, window_text: str, candidate_codes: List[str]) -> Dict[str, float]:
        if not candidate_codes:
            return {}
        return self.score_batch([window_text], [candidate_codes])[0]
