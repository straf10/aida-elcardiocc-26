from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .context_reranker import ContextReranker
from .dictionary_features import load_dictionary_candidates, normalize_text
from .types import DocumentRecord, LinkedMention, NERMentionPrediction


LINKER_PRIOR_FILENAME = "linker_prior.json"


def build_prior_map(train_docs: List[DocumentRecord]) -> Dict[str, Counter]:
    prior = defaultdict(Counter)
    for doc in train_docs:
        for m in doc.mention_level_annotations:
            key = normalize_text(m.mention)
            if key:
                prior[key][m.code] += 1
    return prior


def _prior_to_jsonable(prior_map: Dict[str, Counter]) -> Dict[str, Dict[str, int]]:
    return {mention: dict(counter) for mention, counter in prior_map.items()}


def _prior_from_jsonable(raw: Dict[str, Dict[str, int]]) -> Dict[str, Counter]:
    prior = defaultdict(Counter)
    for mention, code_counts in raw.items():
        for code, cnt in code_counts.items():
            prior[mention][code] = int(cnt)
    return prior


def save_prior_map(prior_map: Dict[str, Counter], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "prior_map": _prior_to_jsonable(prior_map),
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_prior_map(path: str) -> Dict[str, Counter]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "prior_map" in raw:
        return _prior_from_jsonable(raw["prior_map"])
    return _prior_from_jsonable(raw)


def default_prior_artifact_path(model_dir: str) -> str:
    return str(Path(model_dir) / LINKER_PRIOR_FILENAME)


class MentionLinker:
    def __init__(
        self,
        prior_map: Dict[str, Counter],
        dictionary_map: Dict[str, set],
        reranker: ContextReranker | None = None,
        alpha: float = 0.6,
    ):
        self.prior_map = prior_map
        self.dictionary_map = dictionary_map
        self.reranker = reranker
        self.alpha = float(alpha)

    def _candidate_codes(self, mention_text: str) -> List[str]:
        key = normalize_text(mention_text)
        candidates = []
        if key in self.prior_map:
            candidates.extend([c for c, _ in self.prior_map[key].most_common()])
        if key in self.dictionary_map:
            for c in sorted(self.dictionary_map[key]):
                if c not in candidates:
                    candidates.append(c)
        return candidates

    def _prior_scores(self, mention_text: str, candidates: List[str]) -> Dict[str, float]:
        key = normalize_text(mention_text)
        counts = self.prior_map.get(key, Counter())
        values = np.asarray([float(counts.get(c, 0)) for c in candidates], dtype=np.float64)
        if values.size == 0:
            return {}
        if float(values.sum()) <= 0.0:
            values = np.full_like(values, 1.0 / max(len(candidates), 1))
        else:
            values = values / values.sum()
        return {c: float(v) for c, v in zip(candidates, values)}

    def _context_window(self, context_text: str, start: int, end: int) -> str:
        w = max(0, int(self.reranker.window_chars if self.reranker else 200))
        left = max(0, int(start) - w)
        right = min(len(context_text), int(end) + w)
        return context_text[left:right]

    def link_mentions(
        self,
        mentions: Iterable[NERMentionPrediction],
        *,
        context_text: str | None = None,
    ) -> List[LinkedMention]:
        mentions_list = list(mentions)
        if not mentions_list:
            return []

        candidates_per_mention = [self._candidate_codes(m.text) for m in mentions_list]
        semantic_rows: List[Dict[str, float]] = []
        if self.reranker is not None and context_text:
            windows = [
                self._context_window(context_text, m.start, m.end) for m in mentions_list
            ]
            semantic_rows = self.reranker.score_batch(windows, candidates_per_mention)

        linked = []
        alpha = float(np.clip(self.alpha, 0.0, 1.0))
        for i, m in enumerate(mentions_list):
            candidates = candidates_per_mention[i]
            code = candidates[0] if candidates else None
            if candidates and semantic_rows:
                prior_scores = self._prior_scores(m.text, candidates)
                semantic_scores = semantic_rows[i]
                best_code = None
                best_score = -float("inf")
                for cand in candidates:
                    fused = alpha * float(prior_scores.get(cand, 0.0)) + (1.0 - alpha) * float(
                        semantic_scores.get(cand, -1.0)
                    )
                    if fused > best_score:
                        best_score = fused
                        best_code = cand
                code = best_code
            linked.append(
                LinkedMention(
                    start=m.start,
                    end=m.end,
                    text=m.text,
                    code=code,
                    confidence=m.confidence,
                    candidates=candidates,
                )
            )
        return linked
