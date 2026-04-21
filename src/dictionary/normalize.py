"""Text normalization for dictionary matching (Greek + Latin clinical text)."""

from __future__ import annotations

import re
import unicodedata


def strip_accents(text: str) -> str:
    """Remove Greek and Latin diacritics using Unicode decomposition."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text)


def normalize_text(text: str) -> str:
    """
    Full normalization pipeline:
    1. Lowercase
    2. Strip accents
    3. Replace punctuation/separators with spaces
    4. Remove non-alphanumeric characters
    5. Collapse whitespace
    """
    text = text.lower()
    text = strip_accents(text)
    for old, new in {
        "\u2013": " ", "\u2014": " ", "-": " ", "/": " ",
        "\\": " ", "\n": " ", "\t": " ",
        "\u2018": "'", "\u0384": "", "\u2019": "'",
    }.items():
        text = text.replace(old, new)
    text = re.sub(r"[^a-zA-Zα-ωΑ-Ω0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text by whitespace splitting."""
    return normalize_text(text).split()
