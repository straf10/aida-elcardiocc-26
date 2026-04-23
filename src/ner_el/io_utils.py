from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .schemas import DocumentRecord, MentionAnnotation


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: str, records: Iterable[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _parse_mentions(raw_mentions: list, text: str) -> List[MentionAnnotation]:
    parsed: List[MentionAnnotation] = []
    text_len = len(text) if text else None
    for m in raw_mentions or []:
        start = int(m["start"])
        end = int(m["end"])
        if start < 0 or end <= start:
            continue
        if text_len is not None and end > text_len:
            continue
        mention_text = str(m.get("mention") or (text[start:end] if text else ""))
        parsed.append(
            MentionAnnotation(
                start=start,
                end=end,
                code=str(m["code"]),
                mention=mention_text,
                source=str(m.get("source", "gold")),
                confidence=float(m.get("confidence", 1.0)),
            )
        )
    parsed.sort(key=lambda x: (x.start, x.end))
    return parsed


def parse_document_record(raw: dict) -> DocumentRecord:
    patient_id = int(raw["patient_id"])
    text = str(raw.get("text", ""))
    doc_groups = raw.get("document_level_annotations") or []
    labels_flat = raw.get("labels_flat") or []
    mentions = _parse_mentions(raw.get("mention_level_annotations") or [], text)

    return DocumentRecord(
        patient_id=patient_id,
        text=text,
        document_level_annotations=doc_groups,
        labels_flat=[str(c) for c in labels_flat],
        mention_level_annotations=mentions,
    )


def load_documents(path: str) -> List[DocumentRecord]:
    return [parse_document_record(rec) for rec in load_jsonl(path)]


def validate_document_schema(docs: List[DocumentRecord]) -> dict:
    invalid_spans = 0
    total_mentions = 0
    codes = set()

    for doc in docs:
        for m in doc.mention_level_annotations:
            total_mentions += 1
            if not (0 <= m.start < m.end <= len(doc.text)):
                invalid_spans += 1
            codes.add(m.code)

    mentions_per_doc = total_mentions / len(docs) if docs else 0.0
    return {
        "documents": len(docs),
        "mentions": total_mentions,
        "mentions_per_doc": mentions_per_doc,
        "unique_codes": len(codes),
        "invalid_spans": invalid_spans,
    }


def mentions_to_json(mentions: List[MentionAnnotation]) -> List[dict]:
    return [
        {
            "start": m.start,
            "end": m.end,
            "code": m.code,
            "mention": m.mention,
            "confidence": m.confidence,
            "source": m.source,
        }
        for m in mentions
    ]
