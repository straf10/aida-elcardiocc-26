"""Text cleaning and annotation helpers shared by the preprocessing pipeline."""

import re


def clean_text(text: str) -> str:
    # Case is preserved intentionally: XLM-R's SentencePiece tokenizer is case-sensitive.
    # For uncased models (e.g. Greek-BERT), lowercasing is handled by the tokenizer itself.
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"[^a-zA-Z0-9\u0370-\u03ff\u1f00-\u1fff\s\-\.\,\%\/\(\)\[\]\:]",
        "",
        text,
    )
    return text.strip()


def extract_annotations(d: dict):
    return d.get("document_level_annotations", [])


def flatten_annotations(annotations) -> list[str]:
    codes: set[str] = set()
    for group in annotations:
        if isinstance(group, list):
            codes.update(str(c) for c in group if c is not None and str(c).strip())
    return sorted(codes)
