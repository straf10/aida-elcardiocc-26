from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from .dictionary_features import merge_gold_with_dictionary_mentions
from .types import DocumentRecord


LABELS = ["O", "B-MED", "I-MED"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


@dataclass
class EncodedExample:
    patient_id: int
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    offsets: List[Tuple[int, int]]


def _prepare_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    cleaned = [(int(s), int(e)) for s, e in spans if int(e) > int(s)]
    cleaned.sort(key=lambda x: (x[0], x[1]))
    return cleaned


def _bio_labels_for_offsets(
    offsets: List[Tuple[int, int]],
    spans: List[Tuple[int, int]],
) -> List[int]:
    labels: List[int] = []
    spans = _prepare_spans(spans)
    span_idx = 0
    active_span_idx: Optional[int] = None

    for start, end in offsets:
        if start == end:
            labels.append(-100)
            continue

        while span_idx < len(spans) and spans[span_idx][1] <= start:
            if active_span_idx == span_idx:
                active_span_idx = None
            span_idx += 1

        if span_idx >= len(spans):
            labels.append(LABEL2ID["O"])
            continue

        span_start, span_end = spans[span_idx]
        if not (span_start <= start < span_end):
            labels.append(LABEL2ID["O"])
            continue

        # First token we see inside a span must open with B-MED, even if the
        # mention started in the middle of a previous token and no token starts
        # exactly at span_start.
        if active_span_idx != span_idx:
            labels.append(LABEL2ID["B-MED"])
            active_span_idx = span_idx
        else:
            labels.append(LABEL2ID["I-MED"])
    return labels


def encode_document(
    doc: DocumentRecord,
    tokenizer,
    max_length: int = 512,
    dynamic_padding: bool = True,
    dictionary_map: Optional[Dict[str, set]] = None,
    dictionary_word_boundary: bool = False,
    use_dictionary_augmentation: bool = False,
) -> EncodedExample:
    enc = tokenizer(
        doc.text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        padding=False if dynamic_padding else "max_length",
    )

    offsets = enc["offset_mapping"]
    mentions = doc.mention_level_annotations
    if use_dictionary_augmentation and dictionary_map:
        mentions = merge_gold_with_dictionary_mentions(
            doc.text,
            mentions,
            dictionary_map,
            word_boundary=dictionary_word_boundary,
        )
    spans = [(m.start, m.end) for m in mentions]
    labels = _bio_labels_for_offsets(offsets, spans)

    return EncodedExample(
        patient_id=doc.patient_id,
        input_ids=[int(v) for v in enc["input_ids"]],
        attention_mask=[int(v) for v in enc["attention_mask"]],
        labels=labels,
        offsets=offsets,
    )


class NERDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        docs: List[DocumentRecord],
        model_name: str,
        max_length: int = 512,
        dynamic_padding: bool = True,
        dictionary_map: Optional[Dict[str, set]] = None,
        dictionary_word_boundary: bool = False,
        use_dictionary_augmentation: bool = False,
    ):
        self.docs = docs
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length
        self.examples = [
            encode_document(
                d,
                self.tokenizer,
                max_length=max_length,
                dynamic_padding=dynamic_padding,
                dictionary_map=dictionary_map,
                dictionary_word_boundary=dictionary_word_boundary,
                use_dictionary_augmentation=use_dictionary_augmentation,
            )
            for d in docs
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "input_ids": ex.input_ids,
            "attention_mask": ex.attention_mask,
            "labels": ex.labels,
        }
