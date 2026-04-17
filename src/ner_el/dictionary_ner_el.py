import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

try:
    from fuzzywuzzy import fuzz
    HAS_FUZZY = True

    def partial_ratio(a: str, b: str) -> int:
        return fuzz.partial_ratio(a, b)
except ImportError:
    HAS_FUZZY = False

    def partial_ratio(a: str, b: str) -> int:
        return 0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.evaluation.evaluator import evaluate_file
except ImportError:
    evaluate_file = None

# ========================= CONFIG =========================
BASE_CSV = "data/external/icd10_greek_lookup.csv"      # original
RICH_CSV = "data/external/full_dictionary.csv"   # new rich one
TRAIN_PATH = "data/processed/validation_set.jsonl"
OUTPUT_PATH = "submissions/ner_el_pipeline_v9.jsonl"
DEBUG_OUTPUT_PATH = "outputs/ner_el/ner_el_pipeline_v10_debug.jsonl"
FUZZY_THRESHOLD = 88
MAX_CODES_PER_DOC = 8
# ========================================================

def load_merged_dictionary():
    term_to_codes = defaultdict(set)
    
    # Base dictionary
    with open(BASE_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["code"].strip()
            desc = row["greek_description"].strip()
            if code and desc and "-" not in code:
                term_to_codes[desc.lower()].add(code)
    
    # Rich dictionary
    with open(RICH_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = row["term"].strip()
            codes_str = row.get("codes_pipe_sep", "").strip()
            if term and codes_str:
                codes = [c.strip() for c in codes_str.split('|') if c.strip() and "-" not in c]
                for code in codes:
                    term_to_codes[term.lower()].add(code)
    
    print(f"✅ Merged dictionary: {len(term_to_codes)} unique terms → {len(set(c for codes in term_to_codes.values() for c in codes))} codes")
    return term_to_codes

def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text)


def normalize_text(text: str) -> str:
    text = _strip_accents(text.lower())
    text = re.sub(r"[^α-ωa-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_with_char_map(text: str) -> tuple[str, list[int]]:
    """Normalize while preserving a character index map to original text offsets."""
    out_chars = []
    norm_to_orig = []
    prev_space = True

    for i, ch in enumerate(text):
        low = _strip_accents(ch.lower())
        candidate = low if re.match(r"[α-ωa-z0-9\s]", low) else " "
        if candidate.isspace():
            if prev_space:
                continue
            out_chars.append(" ")
            norm_to_orig.append(i)
            prev_space = True
        else:
            out_chars.append(candidate)
            norm_to_orig.append(i)
            prev_space = False

    if out_chars and out_chars[-1] == " ":
        out_chars.pop()
        norm_to_orig.pop()

    return "".join(out_chars), norm_to_orig


def extract_mentions(text: str, term_to_codes: dict) -> list[dict]:
    """Weakly-supervised mention extraction from exact dictionary matches on normalized text."""
    norm_text, norm_to_orig = normalize_with_char_map(text)
    mentions = []
    seen = set()

    for term, codes in term_to_codes.items():
        if not term:
            continue
        for m in re.finditer(re.escape(term), norm_text):
            start_n, end_n = m.start(), m.end()
            if start_n >= len(norm_to_orig) or end_n - 1 >= len(norm_to_orig):
                continue

            start_o = norm_to_orig[start_n]
            end_o = norm_to_orig[end_n - 1] + 1
            key = (start_o, end_o, tuple(sorted(codes)))
            if key in seen:
                continue
            seen.add(key)
            mentions.append(
                {
                    "start_offset": start_o,
                    "end_offset": end_o,
                    "text": text[start_o:end_o],
                    "icd10_codes": sorted(codes),
                    "source": "dictionary_exact",
                }
            )

    mentions.sort(key=lambda x: (x["start_offset"], -(x["end_offset"] - x["start_offset"])))
    return mentions


def resolve_overlapping_mentions(mentions: list[dict]) -> list[dict]:
    """Longest-match-wins overlap resolution for stable BIO tags."""
    resolved = []
    for m in mentions:
        if not resolved:
            resolved.append(m)
            continue
        last = resolved[-1]
        if m["start_offset"] < last["end_offset"]:
            last_len = last["end_offset"] - last["start_offset"]
            cur_len = m["end_offset"] - m["start_offset"]
            if cur_len > last_len:
                resolved[-1] = m
            continue
        resolved.append(m)
    return resolved


def build_bio_tags(text: str, mentions: list[dict]) -> tuple[list[str], list[str]]:
    tokens = []
    token_spans = []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group(0))
        token_spans.append((m.start(), m.end()))

    tags = ["O"] * len(tokens)
    for mention in mentions:
        start, end = mention["start_offset"], mention["end_offset"]
        label = mention["icd10_codes"][0] if mention["icd10_codes"] else "ENT"
        touched = 0
        for i, (ts, te) in enumerate(token_spans):
            if te <= start or ts >= end:
                continue
            tags[i] = f"B-{label}" if touched == 0 else f"I-{label}"
            touched += 1
    return tokens, tags

def predict_codes(text: str, term_to_codes: dict, mentions: list[dict]) -> list[str]:
    norm_text = normalize_text(text)
    found = {code for m in mentions for code in m["icd10_codes"]}

    # Fuzzy fallback remains at document-level only; no reliable span for BIO creation.
    for term, codes in term_to_codes.items():
        if term in norm_text:
            found.update(codes)
            continue
        if HAS_FUZZY and len(term) > 4 and partial_ratio(term, norm_text) >= FUZZY_THRESHOLD:
            found.update(codes)

    return sorted(found)[:MAX_CODES_PER_DOC]

def main() -> None:
    print("Running v10 - dictionary NER+EL with BIO artifacts and official evaluator...")
    if not HAS_FUZZY:
        print("Warning: fuzzywuzzy is not installed; using exact-match mode only.")

    term_to_codes = load_merged_dictionary()

    predictions = []
    debug_predictions = []

    with open(TRAIN_PATH, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            mentions = extract_mentions(doc["text"], term_to_codes)
            mentions = resolve_overlapping_mentions(mentions)
            tokens, bio_tags = build_bio_tags(doc["text"], mentions)
            codes = predict_codes(doc["text"], term_to_codes, mentions)

            predictions.append(
                {
                    "patient_id": doc["patient_id"],
                    "document_level_annotations": [[code] for code in codes] if codes else [],
                }
            )

            debug_predictions.append(
                {
                    "patient_id": doc["patient_id"],
                    "document_level_annotations": [[code] for code in codes] if codes else [],
                    "mention_level_annotations": mentions,
                    "bio_tokens": tokens,
                    "bio_tags": bio_tags,
                }
            )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    Path(DEBUG_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for pred in debug_predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"Inference complete -> {OUTPUT_PATH}")
    print(f"Debug output with mentions/BIO -> {DEBUG_OUTPUT_PATH}")
    print(f"   Documents: {len(predictions)}")
    print(
        f"   Average codes per document: "
        f"{sum(len(p['document_level_annotations']) for p in predictions) / len(predictions):.2f}"
    )
    print(
        f"   Average mentions per document: "
        f"{sum(len(p['mention_level_annotations']) for p in debug_predictions) / len(debug_predictions):.2f}"
    )

    # ===================== OFFICIAL METRICS =====================
    if evaluate_file is not None and ("validation" in TRAIN_PATH or "train" in TRAIN_PATH):
        metrics = evaluate_file(TRAIN_PATH, OUTPUT_PATH)
        print("\n" + "=" * 60)
        print("Dictionary NER+EL (official evaluator)")
        print(f"   Micro Precision : {metrics['precision']:.4f}")
        print(f"   Micro Recall    : {metrics['recall']:.4f}")
        print(f"   Micro F1        : {metrics['micro_f1']:.4f}")
        print(f"   TP/FP/FN        : {metrics['total_tp']}/{metrics['total_fp']}/{metrics['total_fn']}")
        print("=" * 60)
    else:
        print("\nOfficial metrics skipped (test path or evaluator import unavailable).")

    print("\nReady: submission JSONL + debug JSONL with mention-level BIO artifacts.")


if __name__ == "__main__":
    main()