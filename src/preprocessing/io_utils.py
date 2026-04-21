import json
from pathlib import Path
from typing import Dict, Iterable, List

# =========================================================
# PATH CONSTANTS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TRAIN_PATH = str(PROJECT_ROOT / "data" / "raw" / "Train_Set_2026" / "train_dataset.jsonl")
PROCESSED_TRAIN_PATH = str(PROJECT_ROOT / "data" / "processed" / "training_set.jsonl")
PROCESSED_VAL_PATH = str(PROJECT_ROOT / "data" / "processed" / "validation_set.jsonl")
RAW_TRAIN_SPLIT_PATH = str(PROJECT_ROOT / "data" / "processed" / "training_set_raw.jsonl")
RAW_VAL_SPLIT_PATH = str(PROJECT_ROOT / "data" / "processed" / "validation_set_raw.jsonl")

LABELSET_PATH = str(PROJECT_ROOT / "data" / "raw" / "Train_Set_2026" / "labelset.txt")
TERM_CODE_CSV = str(PROJECT_ROOT / "data" / "external" / "full_dictionary.csv")
TRAIN_ONLY_TERM_CODE_CSV = str(PROJECT_ROOT / "data" / "external" / "full_dictionary.train_only.csv")
CODE_DESC_PATH = str(PROJECT_ROOT / "data" / "external" / "icd10_greek_lookup.csv")
TEST_PATH = str(PROJECT_ROOT / "data" / "raw" / "Test_Set_2026" / "test_set.jsonl")
DICTIONARY_CONFIG_PATH = str(PROJECT_ROOT / "src" / "dictionary" / "dictionary.yaml")
DICTIONARY_OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "experiments" / "dictionary_baseline")

# =========================================================
# I/O UTILITIES
# =========================================================

def load_jsonl(path: str) -> List[dict]:
    """Load a JSONL file into a list of dicts. Adds '_row_id' for tracking."""
    records: List[dict] = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for i, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "_row_id" not in data:
                data["_row_id"] = i
            records.append(data)
    return records

def save_jsonl(records: Iterable[dict], path: str) -> None:
    """Save an iterable of dicts to a JSONL file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_labelset(path: str) -> List[str]:
    """Load the list of valid ICD-10 codes from a text file (one per line)."""
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            code = line.strip()
            if code:
                labels.append(code)
    return labels


def resolve_patient_id(rec: dict) -> int:
    """Stable patient key from a JSONL record (competition / internal field variants)."""
    raw = rec.get("patient_id") or rec.get("id") or rec.get("doc_id") or rec.get("_row_id")
    return int(raw)


# Re-export load_term_code_csv from dictionary since it owns the blacklist/normalization logic
# (Removed to avoid circular imports)
