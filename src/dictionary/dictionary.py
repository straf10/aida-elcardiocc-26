import argparse
import json
import re
import csv
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict

from evaluation.evaluator import evaluate_data
from preprocessing.io_utils import (
    load_jsonl,
    load_labelset,
    resolve_patient_id,
    TRAIN_PATH,
    LABELSET_PATH,
    TERM_CODE_CSV,
    CODE_DESC_PATH,
    PROJECT_ROOT,
)

# =========================================================
# CONFIG
# =========================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "dictionary_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# BLACKLIST
#
# Terms removed from the dictionary because they cause too many
# false positives. These are either:
#   - Very short / ambiguous abbreviations that appear everywhere
#   - Echo grading abbreviations (e.g. "MR 2/4" does not mean mitral disease)
#   - Substrings of unrelated words (fragment matching problem)
#   - Clinical terms with precision < 0.30 on training data
#
# Each entry was identified through per-term precision analysis.
# =========================================================
BLACKLIST_TERMS = {
    # Short / ambiguous abbreviations
    "πε", "τr", "tr", "vt", "vf", "af", "as", "κμ", "πκμ", "αυ", "ay",
    # Echo grading abbreviations ("mr 2/4", "ar 1/4" etc. ≠ disease codes)
    "mr", "ar",
    # Fragment substrings of longer Greek words
    "ονα",   # fragment of "μονάδα" → fires N17 on 1230 docs (prec=0.02)
    "απο",   # fragment → J81
    "οπο",   # fragment → J81
    "ολλ",   # fragment → C00-C97
    "aor",   # fragment → I35
    "σελ",   # → M32 but matches abbreviation "ΣΕΛ" everywhere
    # Ambiguous clinical terms with very low precision
    "ανιουσα αορτη",            # → I71: 232 hits, only 9 gold I71 docs (prec=0.04)
    "μικρη mr",                 # "minimal MR" in echo report ≠ mitral regurgitation disease
    "ανεπαρκεια τριγλωχινας",  # → I07: 102 hits, only 21 gold I07 docs
    "ανεπαρκεια τριγλωχινας βαλβιδας",
    "τριγλωχινα",              # → I07: 284 hits, prec=0.04
    # Discovered in second-round precision analysis
    "κα",   # → I50: appears in 2500/2500 docs (fragment of "καρδιακη", "καθε" etc.)
    "απ",   # → I10: "ΑΠ" = blood pressure abbreviation, appears everywhere
    "εμ",   # → I21: fragment of "επεμβάσεις", "εμβόλιο" etc.
    "πεμ",  # → I21: fragment of "επεμβάσεις" (prec=0.34)
    "α υ",  # → I10: spaced abbreviation, prec=0.48 only
    "κ α",  # → I50: spaced fragment, prec=0.42
    "ka",   # → I50: latin fragment, prec=0.32
    "acs",  # → I21: prec=0.29, matches unrelated contexts
}

# =========================================================
# TEXT NORMALIZATION
#
# All text (both dictionary terms and document text) goes through
# the same normalization pipeline before matching. This ensures
# consistent comparison regardless of accents, punctuation, or casing.
# =========================================================

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
        "–": " ", "—": " ", "-": " ", "/": " ",
        "\\": " ", "\n": " ", "\t": " ",
        "'": "'", "΄": "", "'": "'",
    }.items():
        text = text.replace(old, new)
    text = re.sub(r"[^a-zA-Zα-ωΑ-Ω0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str):
    """Tokenize normalized text by whitespace splitting."""
    return normalize_text(text).split()

# =========================================================
# LOADERS
# =========================================================

def load_term_code_csv(csv_path: str) -> dict:
    """
    Load the term→code dictionary from CSV.
    Columns: term, codes_pipe_sep (e.g. "I10|I11")
    Applies blacklist filtering and text normalization on load.
    Returns: {normalized_term: set_of_codes}
    """
    term_code = {}
    if not Path(csv_path).exists():
        print(f"WARNING: {csv_path} not found")
        return term_code
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = normalize_text(row["term"].strip())
            codes = {c.strip() for c in row["codes_pipe_sep"].split("|") if c.strip()}
            if term and term not in BLACKLIST_TERMS:
                term_code[term] = codes
    return term_code

# =========================================================
# GOLD LABEL HELPERS
#
# The competition uses a "list of lists" annotation format.
# Each document has multiple annotation groups, where each group
# contains one or more alternative codes for the same clinical finding.
# A prediction is correct if it matches at least one code per group.
# =========================================================

def flatten_gold_groups(doc_groups):
    """Flatten list-of-lists annotations into a single set of codes."""
    return {code for group in doc_groups for code in group}

def group_satisfied(group, predicted_codes):
    """Return True if at least one code in the group is predicted."""
    return any(code in predicted_codes for code in group)

def relaxed_group_recall(gold_groups, predicted_codes):
    """
    Compute the fraction of annotation groups that are satisfied.
    This closely mirrors the official competition evaluation logic.
    """
    if not gold_groups:
        return 0.0
    hits = sum(1 for g in gold_groups if group_satisfied(g, predicted_codes))
    return hits / len(gold_groups)

# =========================================================
# CO-OCCURRENCE RULES
#
# Derived from training data statistics. When one code is predicted,
# its highly co-occurring partner is automatically added.
#
#   I10 ↔ I11 : always appear together (1207/1207 docs, 100%)
#   I22 → I21 : every I22 (recent MI) always has I21 (389/389, 100%)
#   I25 → Z95 : strong co-occurrence (75% of I25 docs also have Z95)
# =========================================================

def apply_cooccurrence_rules(predicted: set) -> set:
    if "I10" in predicted: predicted.add("I11")
    if "I11" in predicted: predicted.add("I10")
    if "I22" in predicted: predicted.add("I21")
    if "I25" in predicted: predicted.add("Z95")
    return predicted

# =========================================================
# RULE-BASED BOOSTS
#
# Hard-coded clinical patterns that reliably indicate specific codes.
# These catch cases where dictionary term matching alone is insufficient,
# e.g. procedure mentions, device implantations, specific syndromes.
# =========================================================

def apply_rule_based_boosts(text_norm: str, predicted: set) -> set:
    # Procedures / Devices
    if any(x in text_norm for x in ["pci", "ppci", "ptca", "αγγειοπλαστικη"]):
        predicted.add("Y84")   # medical procedure as cause of complication
    if any(x in text_norm for x in ["stent", "pacemaker", "ppm", "βηματοδοτη"]):
        predicted.add("Z95")   # presence of cardiac device
    if "tavi" in text_norm:
        predicted.update({"Y84", "Z95", "I35"})   # TAVI → procedure + device + aortic valve disease
    if "cabg" in text_norm:
        predicted.update({"Y84", "Z95", "I25"})   # CABG → procedure + device + chronic ischemic heart disease

    # Cardiovascular conditions
    if "μυοκαρδιτιδα" in text_norm:
        predicted.add("I41")   # myocarditis
    if "περικαρδιτιδα" in text_norm:
        predicted.add("I30")   # acute pericarditis
    if "πνευμονικη εμβολη" in text_norm:
        predicted.add("I26")   # pulmonary embolism
    if any(x in text_norm for x in ["κοιλιακη ταχυκαρδια", "κολπικη ταχυκαρδια"]):
        predicted.add("I47")   # paroxysmal tachycardia
    if any(x in text_norm for x in ["κολποκοιλιακος αποκλεισμος", "mobitz ii"]):
        predicted.add("I44")   # AV block
    if any(x in text_norm for x in ["οσς", "οξυ στεφανιαιο συνδρομο", "acs"]):
        predicted.update({"I20", "I21", "I22", "I24"})   # acute coronary syndrome
    if any(x in text_norm for x in ["aov", "στενωση aov"]):
        predicted.add("I35")   # aortic valve disease
    if "ανευρυσμα ανιουσας αορτης" in text_norm:
        predicted.add("I71")   # aortic aneurysm
    if "ισχαιμικο αεε" in text_norm:
        predicted.add("I64")   # stroke
    if any(x in text_norm for x in ["καρδιακη ανακοπη", "αναταχθεισα ανακοπη"]):
        predicted.add("I46")   # cardiac arrest

    # Renal / Respiratory
    if any(x in text_norm for x in ["οξεια νεφρικη βλαβη", "οξεια νεφρικη ανεπαρκεια"]):
        predicted.add("N17")   # acute kidney injury
    if any(x in text_norm for x in ["αιμοκαθαρση", "τεχνητο νεφρο"]):
        predicted.add("Z99")   # dependence on dialysis
    if "λοιμωξη αναπνευστικου" in text_norm:
        predicted.add("J22")   # lower respiratory infection
    if "αναπνευστικη ανεπαρκεια" in text_norm:
        predicted.add("J96")   # respiratory failure

    # Other
    if "ασκιτης" in text_norm:
        predicted.add("R18")   # ascites
    if "αμυλοειδωσ" in text_norm:
        predicted.add("E85")   # amyloidosis
    if "takotsubo" in text_norm:
        predicted.add("I43")   # takotsubo cardiomyopathy

    predicted = apply_cooccurrence_rules(predicted)
    return predicted

# =========================================================
# PREDICTION
#
# We search the FULL document text (not just the diagnosis section)
# because many codes (hypertension, diabetes, device history) are
# mentioned in the "Medical History" or "Physical Examination" sections.
#
# Matching is substring-based on normalized text. All dictionary terms
# have been normalized using the same pipeline (see load_term_code_csv).
# =========================================================

def predict_codes_for_text(text: str, term_code_map: dict) -> set:
    """
    Predict ICD-10 codes for a single document.
    Steps:
      1. Normalize the full document text
      2. Check each dictionary term against the normalized text (substring match)
      3. Apply rule-based boosts for hard clinical patterns
      4. Apply co-occurrence rules
    Returns: set of predicted ICD-10 codes
    """
    full_norm = normalize_text(text)
    predicted = set()

    for term, codes in term_code_map.items():
        if term in full_norm:
            predicted.update(codes)

    predicted = apply_rule_based_boosts(full_norm, predicted)
    return predicted

def predict_all(records: list[dict], term_code_map: dict) -> dict[int, list[str]]:
    """
    Standard channel interface: predict codes for a list of records.
    Returns: {patient_id: [code1, code2, ...]}
    """
    pred_data = {}
    for rec in records:
        patient_id = resolve_patient_id(rec)
        prediction = predict_codes_for_text(rec.get("text", ""), term_code_map)
        pred_data[patient_id] = sorted(prediction)
    return pred_data

# =========================================================
# EVALUATION
# =========================================================

def baseline_evaluation(records, term_code_map):
    """
    Evaluate predictions against gold labels on a set of records.
    Computes flat micro-F1 (precision/recall/F1) and relaxed group recall.

    Note: flat_f1 is an internal proxy metric. The official competition
    evaluation uses list-of-lists matching (see relaxed_group_recall).
    """
    total_tp = total_fp = total_fn = 0
    relaxed_recalls = []

    for rec in records:
        text        = rec.get("text", "")
        gold_groups = rec.get("document_level_annotations", [])
        gold_flat   = flatten_gold_groups(gold_groups)

        pred = predict_codes_for_text(text, term_code_map)

        total_tp += len(pred & gold_flat)
        total_fp += len(pred - gold_flat)
        total_fn += len(gold_flat - pred)

        rr = relaxed_group_recall(gold_groups, pred)
        relaxed_recalls.append(rr)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    avg_rr    = sum(relaxed_recalls) / len(relaxed_recalls) if relaxed_recalls else 0.0

    metrics = {
        "flat_precision":           round(precision, 4),
        "flat_recall":              round(recall, 4),
        "flat_f1":                  round(f1, 4),
        "avg_relaxed_group_recall": round(avg_rr, 4),
        "num_records":              len(records),
    }

    with open(OUTPUT_DIR / "baseline_metrics_train.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics

def official_evaluation(records, term_code_map, labelset):
    """
    Evaluate predictions with the official group-level evaluator
    used by evaluation (micro-F1, precision/recall, macro/per-class).
    """
    ground_truth_data = {}
    pred_data = {}

    for rec in records:
        patient_id = resolve_patient_id(rec)
        gold_groups = rec.get("document_level_annotations", [])
        if gold_groups is None:
            gold_groups = []

        prediction = predict_codes_for_text(rec.get("text", ""), term_code_map)
        ground_truth_data[patient_id] = gold_groups
        pred_data[patient_id] = sorted(prediction)

    metrics = evaluate_data(ground_truth_data, pred_data, label_space=labelset)
    with open(OUTPUT_DIR / "official_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics

def coverage_report(records, term_code_map, labelset):
    """
    Report how many documents and codes are covered by the dictionary.
    Also lists which codes are NOT covered (missing_codes) — useful for
    identifying gaps that other workstreams (NER+EL, MLC) need to fill.
    """
    predicted_label_counter = Counter()
    covered_records = 0

    for rec in records:
        pred = predict_codes_for_text(rec.get("text", ""), term_code_map)
        if pred:
            covered_records += 1
        predicted_label_counter.update(pred)

    covered_codes = set(predicted_label_counter.keys())
    missing_codes = sorted(set(labelset) - covered_codes)

    report = {
        "num_records":                      len(records),
        "records_with_any_prediction":      covered_records,
        "record_coverage_ratio":            round(covered_records / len(records), 4),
        "num_codes_in_labelset":            len(labelset),
        "num_codes_covered_by_dictionary":  len(covered_codes),
        "code_coverage_ratio":              round(len(covered_codes) / len(labelset), 4),
        "top_predicted_codes":              predicted_label_counter.most_common(20),
        "missing_codes":                    missing_codes,
    }

    with open(OUTPUT_DIR / "coverage_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report

def fp_fn_analysis(records, term_code_map):
    """
    Identify which codes are most over-predicted (false positives)
    and most under-predicted (false negatives).
    Useful for refining the blacklist and rule-based boosts.
    """
    fp_counter = Counter()
    fn_counter = Counter()
    for rec in records:
        gold = flatten_gold_groups(rec.get("document_level_annotations", []))
        pred = predict_codes_for_text(rec.get("text", ""), term_code_map)
        for c in pred - gold: fp_counter[c] += 1
        for c in gold - pred: fn_counter[c] += 1

    result = {
        "top_false_positives": fp_counter.most_common(20),
        "top_false_negatives": fn_counter.most_common(20),
    }
    with open(OUTPUT_DIR / "fp_fn_analysis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return fp_counter, fn_counter

def label_frequency_report(records):
    """Count how many times each ICD-10 code appears in the gold labels."""
    counter = Counter()
    for rec in records:
        counter.update(flatten_gold_groups(rec.get("document_level_annotations", [])))
    with open(OUTPUT_DIR / "label_frequency.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["code", "count"])
        for code, cnt in counter.most_common():
            writer.writerow([code, cnt])
    return counter

# =========================================================
# DICTIONARY BUILDER
#
# Automatically extracts term→code pairs from mention-level annotations.
# The 1,453 documents with mention annotations contain ~12,000 text spans
# with their corresponding ICD-10 codes — this is gold-standard data.
#
# Each term is mapped to codes that appear in ≥20% of its occurrences,
# to handle cases where the same text span maps to multiple codes.
#
# Run once to generate glikeria_full_dictionary.csv.
# If the CSV already exists, this step is skipped.
# =========================================================

def build_mention_dictionary(records, output_csv: str, min_term_len=4):
    """
    Build term→code dictionary from mention_level_annotations.
    Produces the base dictionary loaded by load_term_code_csv().
    """
    term_code_counter = defaultdict(Counter)

    for doc in records:
        for m in doc.get("mention_level_annotations", []):
            mention_text = m.get("mention", "")
            code = m.get("code", "")
            if not mention_text or not code:
                continue
            norm_t = normalize_text(mention_text)
            if len(norm_t) >= min_term_len and norm_t not in BLACKLIST_TERMS:
                term_code_counter[norm_t][code] += 1

    # Keep codes with >= 20% frequency per term
    term_to_codes = {}
    for term, code_counts in term_code_counter.items():
        total = sum(code_counts.values())
        codes = {c for c, cnt in code_counts.items() if cnt / total >= 0.20}
        if codes:
            term_to_codes[term] = codes

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "codes_pipe_sep"])
        for term, codes in sorted(term_to_codes.items()):
            writer.writerow([term, "|".join(sorted(codes))])

    print(f"Dictionary saved: {len(term_to_codes)} terms -> {output_csv}")
    return term_to_codes

# =========================================================
# TOKENIZATION / NORMALIZATION REPORT
#
# Analyzes document characteristics after normalization.
# Key findings from training data:
#   - Average ~280 tokens/doc (well within 512-token transformer limit)
#   - Only 2.36% of docs exceed 512 tokens
#   - All docs contain formatting characters (newlines, slashes, dashes)
#   - Extensive use of medical abbreviations (ΑΠ, ΗΚΓ, PCI, CABG etc.)
# =========================================================

def tokenization_report(records, output_path: str):
    """
    Analyze tokenization and normalization characteristics of the corpus.
    Produces a JSON report useful for the MLC team (token length distribution)
    and for documenting preprocessing decisions.
    """
    num_docs = len(records)
    token_lengths = []
    char_lengths = []
    edge_case_docs = []

    formatting_triggers = ["\n", "/", "-", "(", ")", ":", ";"]
    abbreviation_patterns = re.compile(r'\b[A-ZΑ-Ω]{2,6}\b')

    for rec in records:
        text = rec.get("text", "")
        toks = tokenize(text)
        token_lengths.append(len(toks))
        char_lengths.append(len(text))

        has_formatting = any(x in text for x in formatting_triggers)
        abbrevs = abbreviation_patterns.findall(text)

        if has_formatting or abbrevs:
            edge_case_docs.append({
                "row_id":                rec.get("_row_id"),
                "num_tokens":            len(toks),
                "num_chars":             len(text),
                "has_formatting_chars":  has_formatting,
                "sample_abbreviations":  list(set(abbrevs))[:10],
                "sample_text":           text[:200],
            })

    report = {
        "num_docs":                        num_docs,
        "avg_tokens_per_doc":              round(sum(token_lengths) / num_docs, 2) if num_docs else 0,
        "max_tokens_per_doc":              max(token_lengths) if token_lengths else 0,
        "min_tokens_per_doc":              min(token_lengths) if token_lengths else 0,
        "avg_chars_per_doc":               round(sum(char_lengths) / num_docs, 2) if num_docs else 0,
        "docs_over_512_tokens":            sum(1 for l in token_lengths if l > 512),
        "docs_over_512_tokens_pct":        round(sum(1 for l in token_lengths if l > 512) / num_docs * 100, 2),
        "num_docs_with_formatting_issues": len(edge_case_docs),
        "normalization_steps_applied": [
            "lowercase",
            "accent stripping (unicodedata NFD)",
            "punctuation → space (-, /, \\, newlines)",
            "non-alphanumeric removal",
            "whitespace collapse",
        ],
        "sample_edge_cases": edge_case_docs[:10],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Tokenization report saved: {output_path}")
    return report

# =========================================================
# ICD-10 CODE → GREEK DESCRIPTION LOOKUP
#
# Maps each ICD-10 code to its official Greek description.
# Used by:
#   - Panagiotis: for building LLM few-shot prompts
#   - Stelios: as reference for the entity linking module
# =========================================================

def load_code_description_csv(csv_path: str) -> dict:
    """Load code→greek_description mapping from CSV."""
    code_desc = {}
    if not Path(csv_path).exists():
        print(f"WARNING: {csv_path} not found — skipping code lookup")
        return code_desc
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["code"].strip()
            desc = row["greek_description"].strip()
            if code and desc:
                code_desc[code] = desc
    return code_desc

def export_code_lookup(code_desc_map: dict, output_path: str):
    """Export code→description mapping as JSON for easy use by other workstreams."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(code_desc_map, f, ensure_ascii=False, indent=2)
    print(f"Code lookup saved: {output_path}")

# =========================================================
# SUBMISSION FORMAT
#
# Produces JSONL in the exact format required by the competition:
#   {"patient_id": 123, "document_level_annotations": [["I10"], ["I25"], ...]}
#
# Each predicted code is wrapped in its own list (single-element group).
# Supports both 'patient_id' (2026 format) and 'id' (2025 format) fields.
# =========================================================

def export_predictions_jsonl(records, term_code_map, output_path: str):
    """Export predictions to JSONL submission format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            pred = predict_codes_for_text(rec.get("text", ""), term_code_map)
            doc_annotations = [[code] for code in sorted(pred)]
            pid = resolve_patient_id(rec)
            line = {
                "patient_id": pid,
                "document_level_annotations": doc_annotations,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"Predictions saved: {output_path}")

# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-from",
        dest="build_from",
        default=None,
        help="Optional JSONL source used only to build a term-code CSV and exit.",
    )
    parser.add_argument(
        "--out",
        dest="out_csv",
        default=None,
        help="Output CSV path for --build-from mode.",
    )
    args = parser.parse_args()

    if args.build_from or args.out_csv:
        if not args.build_from or not args.out_csv:
            raise ValueError("--build-from and --out must be provided together.")
        build_records = load_jsonl(args.build_from)
        build_mention_dictionary(build_records, args.out_csv)
        print(f"Train-only dictionary written to {args.out_csv}")
        return

    print("Loading data...")
    records  = load_jsonl(TRAIN_PATH)
    labelset = load_labelset(LABELSET_PATH)
    print(f"Loaded {len(records)} records, {len(labelset)} labels")

    # Build dictionary from mention annotations if not already present
    if not Path(TERM_CODE_CSV).exists():
        print(f"\n{TERM_CODE_CSV} not found -> building from mention_level_annotations...")
        build_mention_dictionary(records, TERM_CODE_CSV)

    # Load dictionary (blacklist is applied inside load_term_code_csv)
    print("\nLoading term-code dictionary...")
    term_code_map = load_term_code_csv(TERM_CODE_CSV)
    print(f"Dictionary size: {len(term_code_map)} terms (after blacklist)")

    # Tokenization / normalization analysis
    print("\nTokenization / normalization report...")
    tok_report = tokenization_report(records, str(OUTPUT_DIR / "tokenization_report.json"))
    print(f"  Avg tokens/doc: {tok_report['avg_tokens_per_doc']}")
    print(f"  Docs over 512 tokens: {tok_report['docs_over_512_tokens']} ({tok_report['docs_over_512_tokens_pct']}%)")

    # Label frequency across training set
    print("\nLabel frequency report...")
    label_frequency_report(records)

    # Dictionary coverage analysis
    print("Coverage report...")
    cov = coverage_report(records, term_code_map, labelset)
    print(f"  Records with predictions: {cov['records_with_any_prediction']}/{cov['num_records']}")
    print(f"  Codes covered: {cov['num_codes_covered_by_dictionary']}/{cov['num_codes_in_labelset']}")
    print(f"  Missing codes: {cov['missing_codes']}")

    # Evaluate on full training set (self-evaluation)
    has_gold = any(rec.get("document_level_annotations") for rec in records)
    if has_gold:
        print("\nBaseline evaluation (full training set)...")
        metrics = baseline_evaluation(records, term_code_map)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

        print("\nOfficial evaluation (group-level metrics)...")
        official_metrics = official_evaluation(records, term_code_map, labelset)
        print(f"  Micro-F1:  {official_metrics['micro_f1']:.4f}")
        print(f"  Precision: {official_metrics['precision']:.4f}")
        print(f"  Recall:    {official_metrics['recall']:.4f}")
        if "macro_f1_present_labels" in official_metrics:
            print(f"  Macro-F1 (present labels): {official_metrics['macro_f1_present_labels']:.4f}")
            print(f"  Macro-F1 (all labels):     {official_metrics['macro_f1_all_labels']:.4f}")

        print("\nFP/FN analysis...")
        fp_c, fn_c = fp_fn_analysis(records, term_code_map)
        print(f"  Top FP: {fp_c.most_common(5)}")
        print(f"  Top FN: {fn_c.most_common(5)}")

    # Load and export ICD-10 code → Greek description lookup
    print("\nCode description lookup...")
    code_desc_map = load_code_description_csv(CODE_DESC_PATH)
    if code_desc_map:
        export_code_lookup(code_desc_map, str(OUTPUT_DIR / "icd10_lookup.json"))
        print(f"  Loaded {len(code_desc_map)} code descriptions")

    # Export predictions for training set (for ensemble — Stanimeros)
    export_predictions_jsonl(records, term_code_map,
                             str(OUTPUT_DIR / "dictionary_predictions_train.jsonl"))

    # Export predictions for test set if available
    TEST_PATH = str(PROJECT_ROOT / "data" / "raw" / "Test_Set_2026" / "test_set.jsonl")
    if Path(TEST_PATH).exists():
        print(f"\nTest set found: {TEST_PATH}")
        test_records = load_jsonl(TEST_PATH)
        print(f"Loaded {len(test_records)} test records")
        export_predictions_jsonl(test_records, term_code_map,
                                 str(OUTPUT_DIR / "dictionary_predictions_test.jsonl"))
    else:
        print(f"\nTest set not found ({TEST_PATH}) — skipping.")

    print(f"\nDone. Outputs in: {OUTPUT_DIR.resolve()}")
    print("\nNext steps:")
    print("  1. Send dictionary_predictions_train.jsonl to Stanimeros")
    print("  2. Send dictionary_predictions_test.jsonl to Stanimeros and Strafiotis")
    print("  3. Send full_dictionary.csv to Stelios")
    print("  4. Send icd10_lookup.json to Panagiotis")

if __name__ == "__main__":
    main()