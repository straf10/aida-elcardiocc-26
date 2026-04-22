from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .bio_dataset import ID2LABEL
from .types import NERMentionPrediction


def decode_mentions_from_logits(
    text: str,
    offsets: List[Tuple[int, int]],
    logits: np.ndarray,
) -> List[NERMentionPrediction]:
    pred_ids = logits.argmax(axis=-1)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)

    mentions: List[NERMentionPrediction] = []
    cur_start = None
    cur_end = None
    cur_scores = []

    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue
        tag = ID2LABEL.get(int(pred_ids[i]), "O")
        score = float(probs[i, pred_ids[i]])

        if tag == "B-MED":
            if cur_start is not None:
                mentions.append(
                    NERMentionPrediction(
                        start=cur_start,
                        end=cur_end,
                        text=text[cur_start:cur_end],
                        confidence=float(np.mean(cur_scores)) if cur_scores else 0.0,
                    )
                )
            cur_start = start
            cur_end = end
            cur_scores = [score]
        elif tag == "I-MED":
            if cur_start is None:
                cur_start = start
                cur_end = end
                cur_scores = [score]
            else:
                cur_end = end
                cur_scores.append(score)
        else:
            if cur_start is not None:
                mentions.append(
                    NERMentionPrediction(
                        start=cur_start,
                        end=cur_end,
                        text=text[cur_start:cur_end],
                        confidence=float(np.mean(cur_scores)) if cur_scores else 0.0,
                    )
                )
                cur_start = None
                cur_end = None
                cur_scores = []

    if cur_start is not None:
        mentions.append(
            NERMentionPrediction(
                start=cur_start,
                end=cur_end,
                text=text[cur_start:cur_end],
                confidence=float(np.mean(cur_scores)) if cur_scores else 0.0,
            )
        )

    return mentions
