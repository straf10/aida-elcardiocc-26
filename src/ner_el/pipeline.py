from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

from .dictionary_features import extract_dictionary_codes, extract_dictionary_mentions
from .decode import decode_mentions_from_logits, decode_mentions_from_paths
from .linker import MentionLinker
from .schemas import DocumentRecord, LinkedMention


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
        batch_size: int = 16,
        dictionary_map: Optional[Dict[str, set]] = None,
        dictionary_matcher=None,
        dictionary_config=None,
        labelset: Optional[List[str]] = None,
        code_desc_map: Optional[Dict[str, str]] = None,
        use_dictionary_fusion: bool = True,
        dictionary_doc_boost: bool = True,
        dictionary_word_boundary: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.linker = linker
        self.max_length = max_length
        self.batch_size = max(1, int(batch_size))
        self.dictionary_map = dictionary_map or {}
        self.dictionary_matcher = dictionary_matcher
        self.dictionary_config = dictionary_config
        self.labelset = labelset
        self.code_desc_map = code_desc_map
        self.use_dictionary_fusion = use_dictionary_fusion
        self.dictionary_doc_boost = dictionary_doc_boost
        self.dictionary_word_boundary = dictionary_word_boundary

    def _run_ner(self, text: str):
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            padding=False,
            return_tensors="pt",
        )
        offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"][0].tolist()]
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)

        with torch.inference_mode():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits_gpu = out.logits[0]
        if hasattr(self.model, "crf"):
            with torch.inference_mode():
                crf_device = next(self.model.crf.parameters()).device
                mask = torch.tensor(
                    [[1 if s != e else 0 for s, e in offsets]],
                    dtype=torch.long,
                    device=crf_device,
                )
                emissions = logits_gpu.unsqueeze(0).to(crf_device)
                paths = self.model.crf.decode(emissions, mask)
            logits = logits_gpu.detach().float().cpu().numpy()
            return decode_mentions_from_paths(text, offsets, logits, paths[0])
        logits = logits_gpu.detach().float().cpu().numpy()
        return decode_mentions_from_logits(text, offsets, logits)

    def _run_ner_batch(self, texts: List[str]):
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt",
        )
        offsets_per_doc = [
            [(int(s), int(e)) for s, e in row]
            for row in enc["offset_mapping"].tolist()
        ]
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)

        with torch.inference_mode():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits_gpu = out.logits
        logits = logits_gpu.detach().float().cpu().numpy()

        if hasattr(self.model, "crf"):
            with torch.inference_mode():
                crf_device = next(self.model.crf.parameters()).device
                emissions = logits_gpu.to(crf_device)
                mask = torch.tensor(
                    [
                        [1 if s != e else 0 for s, e in offsets]
                        for offsets in offsets_per_doc
                    ],
                    dtype=torch.long,
                    device=crf_device,
                )
                paths = self.model.crf.decode(emissions, mask)
            return [
                decode_mentions_from_paths(text, offsets_per_doc[i], logits[i], paths[i])
                for i, text in enumerate(texts)
            ]
        return [
            decode_mentions_from_logits(text, offsets_per_doc[i], logits[i])
            for i, text in enumerate(texts)
        ]

    @staticmethod
    def _aggregate_codes(linked_mentions: List[LinkedMention]) -> List[str]:
        seen = []
        for m in linked_mentions:
            if m.code and m.code not in seen:
                seen.append(m.code)
        return seen

    @staticmethod
    def _merge_mentions(model_mentions, dict_mentions):
        """Merge mentions and deduplicate exact/contained overlaps."""
        merged = list(model_mentions)

        def _contains(a, b) -> bool:
            return a.start <= b.start and a.end >= b.end

        for m in dict_mentions:
            covered_by_existing = any(_contains(existing, m) for existing in merged)
            if covered_by_existing:
                continue
            # Remove shorter spans fully covered by the new span.
            merged = [existing for existing in merged if not _contains(m, existing)]
            merged.append(m)
        merged.sort(key=lambda x: (x.start, -(x.end - x.start)))
        return merged

    def predict_document(self, doc: DocumentRecord) -> PipelineOutput:
        ner_mentions = self._run_ner(doc.text)
        if self.use_dictionary_fusion and self.dictionary_map:
            dict_mentions = extract_dictionary_mentions(
                doc.text,
                self.dictionary_map,
                word_boundary=self.dictionary_word_boundary,
            )
            ner_mentions = self._merge_mentions(ner_mentions, dict_mentions)

        linked_mentions = self.linker.link_mentions(ner_mentions, context_text=doc.text)
        doc_codes = self._aggregate_codes(linked_mentions)
        if self.dictionary_doc_boost and self.dictionary_matcher and self.dictionary_config:
            for code in extract_dictionary_codes(
                doc.text,
                self.dictionary_matcher,
                self.dictionary_config,
                labelset=self.labelset,
                code_desc_map=self.code_desc_map,
            ):
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
        outputs: List[PipelineOutput] = []
        for i in range(0, len(docs), self.batch_size):
            batch_docs = docs[i : i + self.batch_size]
            ner_mentions_batch = self._run_ner_batch([d.text for d in batch_docs])
            for doc, ner_mentions in zip(batch_docs, ner_mentions_batch):
                if self.use_dictionary_fusion and self.dictionary_map:
                    dict_mentions = extract_dictionary_mentions(
                        doc.text,
                        self.dictionary_map,
                        word_boundary=self.dictionary_word_boundary,
                    )
                    ner_mentions = self._merge_mentions(ner_mentions, dict_mentions)

                linked_mentions = self.linker.link_mentions(ner_mentions, context_text=doc.text)
                doc_codes = self._aggregate_codes(linked_mentions)
                if self.dictionary_doc_boost and self.dictionary_matcher and self.dictionary_config:
                    for code in extract_dictionary_codes(
                        doc.text,
                        self.dictionary_matcher,
                        self.dictionary_config,
                        labelset=self.labelset,
                        code_desc_map=self.code_desc_map,
                    ):
                        if code not in doc_codes:
                            doc_codes.append(code)

                outputs.append(
                    PipelineOutput(
                        patient_id=doc.patient_id,
                        doc_prediction={
                            "patient_id": doc.patient_id,
                            "document_level_annotations": [[code] for code in doc_codes],
                        },
                        debug_prediction={
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
                        },
                    )
                )
        return outputs
