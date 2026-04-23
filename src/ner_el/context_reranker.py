from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

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
        self._device = torch.device(getattr(self._retriever._model, "device", "cpu"))  # noqa: SLF001
        self._codes: List[str] = []
        self._code_to_idx: Dict[str, int] = {}
        self._embeddings: np.ndarray | None = None
        self._embeddings_torch: torch.Tensor | None = None

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
        print(f"Reranker on device: {self._device}")
        self._codes = codes
        self._code_to_idx = {c: i for i, c in enumerate(codes)}
        self._embeddings = np.asarray(embeddings, dtype=np.float32)
        self._embeddings_torch = torch.as_tensor(self._embeddings, dtype=torch.float32, device=self._device)
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
        obj._embeddings = np.load(out_dir / cls.EMBEDDINGS_FILENAME).astype(np.float32, copy=False)
        obj._embeddings_torch = torch.as_tensor(obj._embeddings, dtype=torch.float32, device=obj._device)
        return obj

    def score_batch(
        self,
        windows: List[str],
        candidate_codes_per_window: List[List[str]],
    ) -> List[Dict[str, float]]:
        if not windows:
            return []
        if self._embeddings is None or self._embeddings_torch is None:
            raise RuntimeError("ContextReranker is not fitted.")
        query_embeddings = self._retriever._model.encode(  # noqa: SLF001
            windows,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        q = torch.as_tensor(np.asarray(query_embeddings, dtype=np.float32), dtype=torch.float32, device=self._device)
        rows: List[Dict[str, float]] = []
        for i, candidate_codes in enumerate(candidate_codes_per_window):
            out: Dict[str, float] = {}
            if not candidate_codes:
                rows.append(out)
                continue
            code_indices = [self._code_to_idx.get(str(code), -1) for code in candidate_codes]
            valid_positions = [j for j, idx in enumerate(code_indices) if idx >= 0]
            if valid_positions:
                valid_indices = torch.tensor(
                    [code_indices[j] for j in valid_positions],
                    dtype=torch.long,
                    device=self._device,
                )
                sims = torch.matmul(
                    self._embeddings_torch.index_select(0, valid_indices),
                    q[i],
                )
                sims_cpu = sims.detach().cpu().tolist()
                for pos, score in zip(valid_positions, sims_cpu):
                    out[str(candidate_codes[pos])] = float(score)
            for j, code in enumerate(candidate_codes):
                if str(code) not in out:
                    out[str(code)] = -1.0
            rows.append(out)
        return rows

    def score(self, window_text: str, candidate_codes: List[str]) -> Dict[str, float]:
        if not candidate_codes:
            return {}
        return self.score_batch([window_text], [candidate_codes])[0]
