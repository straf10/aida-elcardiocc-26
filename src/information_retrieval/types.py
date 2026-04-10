"""Shared types for information retrieval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalHit:
    """One ranked ICD-10 code from a retrieval step."""

    code: str
    score: float
    document_text: str = ""
