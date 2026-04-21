"""Build term→code CSV from mention-level annotations."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from .normalize import normalize_text
from .rules import BLACKLIST_TERMS


def build_mention_dictionary(
    records: list,
    output_csv: str,
    min_term_len: int = 4,
    min_term_count: int = 1,
    min_code_ratio: float = 0.20,
) -> dict:
    """
    Build term→code dictionary from ``mention_level_annotations``.
    Produces the base dictionary loaded by ``load_term_code_csv``.

    ``min_term_count`` (default 1) and ``min_code_ratio`` (default 0.20) match legacy
    behavior when left at defaults.
    """
    term_code_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for doc in records:
        for m in doc.get("mention_level_annotations") or []:
            mention_text = m.get("mention", "")
            code = m.get("code", "")
            if not mention_text or not code:
                continue
            norm_t = normalize_text(mention_text)
            if len(norm_t) >= min_term_len and norm_t not in BLACKLIST_TERMS:
                term_code_counter[norm_t][code] += 1

    term_to_codes: dict[str, set[str]] = {}
    for term, code_counts in term_code_counter.items():
        total = sum(code_counts.values())
        if total < min_term_count:
            continue
        codes = {c for c, cnt in code_counts.items() if cnt / total >= min_code_ratio}
        if codes:
            term_to_codes[term] = codes

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "codes_pipe_sep"])
        for term, codes in sorted(term_to_codes.items()):
            writer.writerow([term, "|".join(sorted(codes))])

    print(f"Dictionary saved: {len(term_to_codes)} terms -> {output_csv}")
    return term_to_codes
