import json
import re
import os
import numpy as np
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import requests
import csv

# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(
    BASE_DIR,
    "..", "..", "data", "raw", "Train_Set_2026", "train_dataset.jsonl"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "data", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n==============================")
print("CLEANING + EDA PIPELINE STARTED")
print("==============================\n")

print("FILE:", __file__)
print("DATA_PATH:", DATA_PATH)
print("OUTPUT:", OUTPUT_DIR)

# ==========================================
# OPTIONAL: GitHub fallback
# ==========================================

GITHUB_URL = "https://raw.githubusercontent.com/straf10/ELCardioCC/main/data/raw/Train_Set_2026/train_dataset.jsonl"

if not os.path.exists(DATA_PATH):
    print("\nLocal file not found -> downloading from GitHub...")

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    try:
        r = requests.get(GITHUB_URL)
        r.raise_for_status()

        with open(DATA_PATH, "wb") as f:
            f.write(r.content)

        print("Download successful")

    except Exception as e:
        print("Download failed:", e)
        exit()

# ==========================================
# FUNCTIONS
# ==========================================

def clean_text(text):
    # Case is preserved intentionally: XLM-R's SentencePiece tokenizer is case-sensitive.
    # For uncased models (e.g. Greek-BERT), lowercasing is handled by the tokenizer itself.
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"[^a-zA-Z0-9\u0370-\u03ff\u1f00-\u1fff\s\-\.\,\%\/\(\)\[\]\:]",
        "",
        text
    )
    return text.strip()


def extract_annotations(d):
    return d.get("document_level_annotations", [])

def flatten_annotations(annotations):
    codes = set()
    for group in annotations:
        if isinstance(group, list):
            codes.update(group)
    return sorted(list(codes))

# ==========================================
# LOAD DATA
# ==========================================

print("\nLOADING DATA...\n")

processed_data = []
all_labels = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    lines = [l for l in f if l.strip()]

print("Total lines:", len(lines))

for i, line in enumerate(lines):
    item = json.loads(line)

    annotations = extract_annotations(item)
    labels_flat = flatten_annotations(annotations)
    all_labels.extend(labels_flat)

    record = {
        "patient_id": item.get("patient_id"),
        "text": clean_text(item.get("text", "")),
        "document_level_annotations": annotations,
        "labels_flat": labels_flat,
    }

    mention_annotations = item.get("mention_level_annotations")
    if mention_annotations is not None:
        record["mention_level_annotations"] = mention_annotations

    processed_data.append(record)

    if i < 2:
        print("\n--- SAMPLE ---")
        print("ID:", item.get("patient_id"))
        print("TEXT:", item.get("text", "")[:200])
        print("LABELS (FLAT):", labels_flat)

print("\n==============================")
print("TOTAL SAMPLES:", len(processed_data))
print("==============================\n")

# ==========================================
# LABEL FREQUENCIES
# ==========================================

counts = Counter(all_labels)

print("\nALL ICD-10 FREQUENCIES")
print("==============================")

for code, count in counts.most_common():
    print(f"{code}: {count}")

print("==============================")
print("Unique ICD-10 codes:", len(counts))

# ==========================================
# SAVE FREQUENCIES
# ==========================================

freq_json = os.path.join(OUTPUT_DIR, "icd10_frequencies.json")

with open(freq_json, "w", encoding="utf-8") as f:
    json.dump(counts.most_common(), f, ensure_ascii=False, indent=2)

print("\nSaved JSON:", freq_json)

freq_csv = os.path.join(OUTPUT_DIR, "icd10_frequencies.csv")

with open(freq_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ICD10", "Count"])

    for code, count in counts.most_common():
        writer.writerow([code, count])

print("Saved CSV:", freq_csv)

# ==========================================
# EDA SECTION
# ==========================================

print("\n==============================")
print("EDA ANALYSIS")
print("==============================\n")

labels_per_sample = [len(x["labels_flat"]) for x in processed_data]
text_lengths = [len(x["text"]) for x in processed_data]

print("Dataset size:", len(processed_data))

print("\nLabels per sample")
print("Min:", min(labels_per_sample))
print("Max:", max(labels_per_sample))
print("Avg:", sum(labels_per_sample) / len(labels_per_sample))

print("\nText length")
print("Min:", min(text_lengths))
print("Max:", max(text_lengths))
print("Avg:", sum(text_lengths) / len(text_lengths))

empty_texts = sum(1 for x in processed_data if len(x["text"].strip()) == 0)
print("\nEmpty texts:", empty_texts)

print("\nTOP 20 ICD-10")
for code, count in counts.most_common(20):
    print(code, count)

print("\nTOP 10 % DISTRIBUTION")
total_labels = sum(counts.values())

for code, count in counts.most_common(10):
    print(f"{code}: {(count/total_labels)*100:.2f}%")

# ==========================================
# TRAIN / VALID SPLIT
# ==========================================

def ensure_train_coverage(data):
    """
    Identify samples that must go to train to ensure val labels are covered.
    Returns: (forced_train_indices, singleton_labels)
    """
    from collections import defaultdict
    
    # Build label -> sample indices mapping
    label_to_samples = defaultdict(set)
    for idx, sample in enumerate(data):
        for label in sample["labels_flat"]:
            label_to_samples[label].add(idx)
    
    # Singleton labels (count=1) must go to train only
    singleton_labels = {lbl for lbl, idxs in label_to_samples.items() if len(idxs) == 1}
    
    forced_train = set()
    for label in singleton_labels:
        forced_train.update(label_to_samples[label])
    
    # Rare labels (count 2-5): ensure at least 1 in train
    rare_labels = {lbl for lbl, idxs in label_to_samples.items() if 2 <= len(idxs) <= 5}
    for label in rare_labels:
        sample_idxs = list(label_to_samples[label])
        if not any(i in forced_train for i in sample_idxs):
            forced_train.add(sample_idxs[0])
    
    return forced_train, singleton_labels

def validate_split(train_data_list, val_data_list):
    """Ensure every validation label exists in training."""
    train_labels = set()
    for sample in train_data_list:
        train_labels.update(sample["labels_flat"])
    
    val_labels = set()
    for sample in val_data_list:
        val_labels.update(sample["labels_flat"])
    
    val_only = val_labels - train_labels
    if val_only:
        raise ValueError(f"Validation contains labels not in train: {sorted(val_only)}")
    
    print(f"Split validation passed. Train-only labels: {len(train_labels - val_labels)}")

# Identify samples that must go to train
forced_train_idx, singleton_labels = ensure_train_coverage(processed_data)

if singleton_labels:
    print(f"\nWARNING: Found {len(singleton_labels)} singleton labels (count=1).")
    print(f"These will be forced to the training set to prevent validation-only labels.")

# Separate forced train samples from the rest
forced_train_data = [processed_data[i] for i in forced_train_idx]
pool_data = [processed_data[i] for i in range(len(processed_data)) if i not in forced_train_idx]

# Perform stratified split on the remaining pool
mlb = MultiLabelBinarizer()
labels_list = [x["labels_flat"] for x in pool_data]

if labels_list:
    Y = mlb.fit_transform(labels_list)
    X = np.arange(len(pool_data)).reshape(-1, 1)

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    train_data = []
    val_data = []
    for train_idx, val_idx in msss.split(X, Y):
        train_data = [pool_data[i] for i in train_idx]
        val_data = [pool_data[i] for i in val_idx]
else:
    train_data = []
    val_data = []

# Merge forced train samples back into train_data
train_data.extend(forced_train_data)

# Shuffle the final train_data to mix in the forced samples
np.random.seed(42)
np.random.shuffle(train_data)

print("\nSPLIT")
print("Train:", len(train_data))
print("Val:", len(val_data))

# Validate the split
validate_split(train_data, val_data)

# ==========================================
# SAVE DATASETS
# ==========================================

def save_jsonl(data, name):
    path = os.path.join(OUTPUT_DIR, name)

    print("\nSaving:", path)

    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Done:", name)


save_jsonl(train_data, "training_set.jsonl")
save_jsonl(val_data, "validation_set.jsonl")

print("\nPIPELINE COMPLETED SUCCESSFULLY")
print("==============================\n")