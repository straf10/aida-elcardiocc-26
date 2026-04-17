from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MentionAnnotation:
    start: int
    end: int
    code: str
    mention: str
    confidence: float = 1.0
    source: str = "gold"


@dataclass
class DocumentRecord:
    patient_id: int
    text: str
    document_level_annotations: List[List[str]] = field(default_factory=list)
    labels_flat: List[str] = field(default_factory=list)
    mention_level_annotations: List[MentionAnnotation] = field(default_factory=list)


@dataclass
class NERMentionPrediction:
    start: int
    end: int
    text: str
    confidence: float


@dataclass
class LinkedMention:
    start: int
    end: int
    text: str
    code: Optional[str]
    confidence: float
    candidates: List[str] = field(default_factory=list)
