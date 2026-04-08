import json
import re
import os
from collections import Counter
from sklearn.model_selection import train_test_split
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
print("🚀 CLEANING SCRIPT STARTED")
print("==============================\n")

print("📌 FILE RUNNING:", __file__)
print("📌 BASE_DIR:", BASE_DIR)
print("📌 DATA_PATH:", DATA_PATH)
print("📌 OUTPUT_DIR:", OUTPUT_DIR)

print("\n==============================")

# ==========================================
# OPTIONAL: GitHub fallback download
# ==========================================

GITHUB_URL = "https://raw.githubusercontent.com/straf10/ELCardioCC/main/data/raw/Train_Set_2026/train_dataset.jsonl"

if not os.path.exists(DATA_PATH):
    print("⚠️ Local file not found.")
    print("📥 Downloading from GitHub...")

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    try:
        r = requests.get(GITHUB_URL)
        r.raise_for_status()

        with open(DATA_PATH, "wb") as f:
            f.write(r.content)

        print("✔ Download successful")

    except Exception as e:
        print("❌ Download failed:")
        print(e)
        exit()

# ==========================================
# FUNCTIONS
# ==========================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"[^a-zA-Z0-9\u0370-\u03ff\s\-\.\,\%\/\(\)\[\]\:]",
        "",
        text
    )
    return text.strip()


def extract_labels(d):
    codes = set()
    annotations = d.get("document_level_annotations", [])

    for group in annotations:
        if isinstance(group, list):
            codes.update(group)

    return sorted(list(codes))

# ==========================================
# LOAD DATA
# ==========================================

print("\n📂 LOADING DATA...\n")

processed_data = []
all_labels = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    lines = [l for l in f if l.strip()]

print("📊 Total raw lines:", len(lines))

for i, line in enumerate(lines):
    item = json.loads(line)

    labels = extract_labels(item)
    all_labels.extend(labels)

    processed_data.append({
        "patient_id": item.get("patient_id"),
        "text": clean_text(item.get("text", "")),
        "labels": labels
    })

    # preview πρώτα 2 samples
    if i < 2:
        print("\n--- SAMPLE PREVIEW ---")
        print("ID:", item.get("patient_id"))
        print("TEXT:", item.get("text", "")[:200])
        print("LABELS:", labels)

print("\n==============================")
print("✅ TOTAL SAMPLES LOADED:", len(processed_data))
print("==============================\n")

# ==========================================
# LABEL STATISTICS (ALL CODES)
# ==========================================

counts = Counter(all_labels)

print("\n📊 ALL ICD-10 FREQUENCIES:")
print("====================================")

for code, count in counts.most_common():
    print(f"{code}: {count}")

print("====================================")
print("TOTAL UNIQUE ICD-10 CODES:", len(counts))

# ==========================================
# SAVE FREQUENCIES (JSON + CSV)
# ==========================================

freq_json_path = os.path.join(OUTPUT_DIR, "icd10_frequencies.json")

with open(freq_json_path, "w", encoding="utf-8") as f:
    json.dump(counts.most_common(), f, ensure_ascii=False, indent=2)

print("\n✔ Saved JSON frequencies:", freq_json_path)

# CSV export
freq_csv_path = os.path.join(OUTPUT_DIR, "icd10_frequencies.csv")

with open(freq_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ICD10", "Count"])

    for code, count in counts.most_common():
        writer.writerow([code, count])

print("✔ Saved CSV frequencies:", freq_csv_path)

# ==========================================
# TRAIN / VALIDATION SPLIT
# ==========================================

train_data, val_data = train_test_split(
    processed_data,
    test_size=0.2,
    random_state=42
)

print("\n📦 SPLIT DONE")
print("Train samples:", len(train_data))
print("Validation samples:", len(val_data))

# ==========================================
# SAVE DATASETS
# ==========================================

def save_jsonl(data, filename):
    path = os.path.join(OUTPUT_DIR, filename)

    print("\n💾 SAVING:", path)

    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("✔ Saved:", filename)


save_jsonl(train_data, "training_set.jsonl")
save_jsonl(val_data, "validation_set.jsonl")

print("\n🎉 ALL DONE SUCCESSFULLY")
print("==============================\n")