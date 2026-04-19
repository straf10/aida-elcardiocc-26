import csv
import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List


def load_synonym_dict(csv_path: str) -> Dict[str, List[str]]:
    """
    Load code -> synonym terms from CSV columns: term, codes_pipe_sep.
    """
    code_to_terms: Dict[str, List[str]] = defaultdict(list)
    with open(csv_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            term = str(row.get("term", "")).strip()
            if not term:
                continue
            for code in str(row.get("codes_pipe_sep", "")).split("|"):
                code = code.strip()
                if code:
                    code_to_terms[code].append(term)
    return dict(code_to_terms)


def flatten_labels(record: dict) -> List[str]:
    labels_flat = record.get("labels_flat")
    if labels_flat:
        return [str(c) for c in labels_flat]
    return list(
        {
            str(code)
            for group in record.get("document_level_annotations", [])
            for code in group
        }
    )


def synonym_augment(
    text: str,
    record_codes: Iterable[str],
    code_to_terms: Dict[str, List[str]],
    swap_prob: float = 0.35,
) -> str:
    """
    Replace one matched term per code with a synonym term from the same code.
    """
    result = text
    for code in record_codes:
        terms = code_to_terms.get(code, [])
        if len(terms) < 2:
            continue
        for term in terms:
            if random.random() > swap_prob:
                continue
            if len(term) < 4:
                continue
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            if pattern.search(result):
                candidates = [t for t in terms if t != term and len(t) >= 4]
                if not candidates:
                    break
                replacement = random.choice(candidates)
                result = pattern.sub(replacement, result, count=1)
                break
    return result


def _parse_multipliers(multipliers: dict) -> list[tuple[int, int]]:
    """
    Parse threshold map like {"100": 4, "200": 3, "300": 2} into
    [(100,4), (200,3), (300,2)] sorted by threshold.
    """
    parsed: list[tuple[int, int]] = []
    for raw_k, raw_v in (multipliers or {}).items():
        threshold = int(str(raw_k).replace("<", "").strip())
        mult = int(raw_v)
        if threshold > 0 and mult >= 0:
            parsed.append((threshold, mult))
    parsed.sort(key=lambda x: x[0])
    return parsed


def _mult_from_freq(freq: int, parsed_multipliers: list[tuple[int, int]]) -> int:
    for threshold, mult in parsed_multipliers:
        if freq < threshold:
            return mult
    return 0


def build_augmented_dataset(
    records: list[dict],
    labelset: list[str],
    code_to_terms: Dict[str, List[str]],
    min_freq: int,
    multipliers: dict,
    swap_prob: float,
    max_real_pid: int,
) -> list[dict]:
    """
    Build train-only augmented records for rare labels.

    IMPORTANT: synonym CSV MUST be derived from the train fold only.
    """
    label_freq = Counter(code for rec in records for code in flatten_labels(rec))
    parsed_multipliers = _parse_multipliers(multipliers)

    aug_mult: Dict[str, int] = {}
    for code in labelset:
        freq = int(label_freq.get(code, 0))
        if freq < int(min_freq):
            aug_mult[code] = 0
        else:
            aug_mult[code] = _mult_from_freq(freq, parsed_multipliers)

    augmented = list(records)
    next_pid = int(max_real_pid) + 1
    for rec in records:
        rec_codes = flatten_labels(rec)
        multiplier = max((aug_mult.get(code, 0) for code in rec_codes), default=0)
        for _ in range(multiplier):
            new_text = synonym_augment(
                text=str(rec.get("text", "")),
                record_codes=rec_codes,
                code_to_terms=code_to_terms,
                swap_prob=swap_prob,
            )
            aug_record = dict(rec)
            aug_record["text"] = new_text
            aug_record["patient_id"] = next_pid
            next_pid += 1
            augmented.append(aug_record)

    random.shuffle(augmented)
    return augmented
