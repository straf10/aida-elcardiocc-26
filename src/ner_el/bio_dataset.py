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
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    offsets: List[Tuple[int, int]]


def _char_is_inside_mention(char_pos: int, spans: List[Tuple[int, int]]) -> bool:
    for s, e in spans:
        if s <= char_pos < e:
            return True
    return False


def _char_is_mention_start(char_pos: int, spans: List[Tuple[int, int]]) -> bool:
    return any(char_pos == s for s, _ in spans)


def encode_document(
    doc: DocumentRecord,
    tokenizer,
    max_length: int = 512,
    dictionary_map: Optional[Dict[str, set]] = None,
    use_dictionary_augmentation: bool = False,
) -> EncodedExample:
    enc = tokenizer(
        doc.text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offsets = enc["offset_mapping"]
    mentions = doc.mention_level_annotations
    if use_dictionary_augmentation and dictionary_map:
        mentions = merge_gold_with_dictionary_mentions(doc.text, mentions, dictionary_map)
    spans = [(m.start, m.end) for m in mentions]

    labels = []
    for start, end in offsets:
        if start == end:
            labels.append(-100)
            continue

        if _char_is_mention_start(start, spans):
            labels.append(LABEL2ID["B-MED"])
        elif _char_is_inside_mention(start, spans):
            labels.append(LABEL2ID["I-MED"])
        else:
            labels.append(LABEL2ID["O"])

    return EncodedExample(
        patient_id=doc.patient_id,
        input_ids=torch.tensor(enc["input_ids"], dtype=torch.long),
        attention_mask=torch.tensor(enc["attention_mask"], dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        offsets=offsets,
    )


class NERDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        docs: List[DocumentRecord],
        model_name: str,
        max_length: int = 512,
        dictionary_map: Optional[Dict[str, set]] = None,
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
                dictionary_map=dictionary_map,
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
