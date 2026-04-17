from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Dict, Iterable, List

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
    def __init__(self, prior_map: Dict[str, Counter], dictionary_map: Dict[str, set]):
        self.prior_map = prior_map
        self.dictionary_map = dictionary_map

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

    def link_mentions(self, mentions: Iterable[NERMentionPrediction]) -> List[LinkedMention]:
        linked = []
        for m in mentions:
            candidates = self._candidate_codes(m.text)
            code = candidates[0] if candidates else None
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
