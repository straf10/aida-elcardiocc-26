from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

from .dictionary_features import extract_dictionary_codes, extract_dictionary_mentions
from .decode import decode_mentions_from_logits
from .linker import MentionLinker
from .types import DocumentRecord, LinkedMention


@dataclass
class PipelineOutput:
    patient_id: int
    doc_prediction: dict
    debug_prediction: dict


class NERELPipeline:
    def __init__(
        self,
        model,
        tokenizer_name: str,
        linker: MentionLinker,
        max_length: int = 512,
        dictionary_map: Optional[Dict[str, set]] = None,
        use_dictionary_fusion: bool = True,
        dictionary_doc_boost: bool = True,
    ):
        self.model = model
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.linker = linker
        self.max_length = max_length
        self.dictionary_map = dictionary_map or {}
        self.use_dictionary_fusion = use_dictionary_fusion
        self.dictionary_doc_boost = dictionary_doc_boost

    def _run_ner(self, text: str):
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"][0].tolist()]
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0].detach().cpu().numpy()
        return decode_mentions_from_logits(text, offsets, logits)

    @staticmethod
    def _aggregate_codes(linked_mentions: List[LinkedMention]) -> List[str]:
        seen = []
        for m in linked_mentions:
            if m.code and m.code not in seen:
                seen.append(m.code)
        return seen

    @staticmethod
    def _merge_mentions(model_mentions, dict_mentions):
        merged = list(model_mentions)
        occupied = {(m.start, m.end) for m in merged}
        for m in dict_mentions:
            span = (m.start, m.end)
            if span in occupied:
                continue
            merged.append(m)
            occupied.add(span)
        merged.sort(key=lambda x: (x.start, x.end))
        return merged

    def predict_document(self, doc: DocumentRecord) -> PipelineOutput:
        ner_mentions = self._run_ner(doc.text)
        if self.use_dictionary_fusion and self.dictionary_map:
            dict_mentions = extract_dictionary_mentions(doc.text, self.dictionary_map)
            ner_mentions = self._merge_mentions(ner_mentions, dict_mentions)

        linked_mentions = self.linker.link_mentions(ner_mentions)
        doc_codes = self._aggregate_codes(linked_mentions)
        if self.dictionary_doc_boost and self.dictionary_map:
            for code in extract_dictionary_codes(doc.text, self.dictionary_map):
                if code not in doc_codes:
                    doc_codes.append(code)

        doc_prediction = {
            "patient_id": doc.patient_id,
            "document_level_annotations": [[code] for code in doc_codes],
        }

        debug_prediction = {
            "patient_id": doc.patient_id,
            "document_level_annotations": [[code] for code in doc_codes],
            "mention_level_annotations": [
                {
                    "start": m.start,
                    "end": m.end,
                    "code": m.code,
                    "mention": m.text,
                    "confidence": m.confidence,
                    "candidates": m.candidates,
                }
                for m in linked_mentions
                if m.code is not None
            ],
        }

        return PipelineOutput(
            patient_id=doc.patient_id,
            doc_prediction=doc_prediction,
            debug_prediction=debug_prediction,
        )

    def predict_many(self, docs: List[DocumentRecord]) -> List[PipelineOutput]:
        return [self.predict_document(doc) for doc in docs]
