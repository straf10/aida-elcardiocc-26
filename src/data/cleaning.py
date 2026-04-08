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
print("🚀 CLEANING + EDA PIPELINE STARTED")
print("==============================\n")

print("📌 FILE:", __file__)
print("📌 DATA_PATH:", DATA_PATH)
print("📌 OUTPUT:", OUTPUT_DIR)

# ==========================================
# OPTIONAL: GitHub fallback
# ==========================================

GITHUB_URL = "https://raw.githubusercontent.com/straf10/ELCardioCC/main/data/raw/Train_Set_2026/train_dataset.jsonl"

if not os.path.exists(DATA_PATH):
    print("\n⚠️ Local file not found → downloading from GitHub...")

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    try:
        r = requests.get(GITHUB_URL)
        r.raise_for_status()

        with open(DATA_PATH, "wb") as f:
            f.write(r.content)

        print("✔ Download successful")

    except Exception as e:
        print("❌ Download failed:", e)
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

print("📊 Total lines:", len(lines))

for i, line in enumerate(lines):
    item = json.loads(line)

    labels = extract_labels(item)
    all_labels.extend(labels)

    processed_data.append({
        "patient_id": item.get("patient_id"),
        "text": clean_text(item.get("text", "")),
        "labels": labels
    })

    if i < 2:
        print("\n--- SAMPLE ---")
        print("ID:", item.get("patient_id"))
        print("TEXT:", item.get("text", "")[:200])
        print("LABELS:", labels)

print("\n==============================")
print("✅ TOTAL SAMPLES:", len(processed_data))
print("==============================\n")

# ==========================================
# LABEL FREQUENCIES
# ==========================================

counts = Counter(all_labels)

print("\n📊 ALL ICD-10 FREQUENCIES")
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

print("\n✔ Saved JSON:", freq_json)

freq_csv = os.path.join(OUTPUT_DIR, "icd10_frequencies.csv")

with open(freq_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ICD10", "Count"])

    for code, count in counts.most_common():
        writer.writerow([code, count])

print("✔ Saved CSV:", freq_csv)

# ==========================================
# EDA SECTION
# ==========================================

print("\n==============================")
print("📊 EDA ANALYSIS")
print("==============================\n")

labels_per_sample = [len(x["labels"]) for x in processed_data]
text_lengths = [len(x["text"]) for x in processed_data]

print("📦 Dataset size:", len(processed_data))

print("\n🏷 Labels per sample")
print("Min:", min(labels_per_sample))
print("Max:", max(labels_per_sample))
print("Avg:", sum(labels_per_sample) / len(labels_per_sample))

print("\n📝 Text length")
print("Min:", min(text_lengths))
print("Max:", max(text_lengths))
print("Avg:", sum(text_lengths) / len(text_lengths))

empty_texts = sum(1 for x in processed_data if len(x["text"].strip()) == 0)
print("\n⚠️ Empty texts:", empty_texts)

print("\n📊 TOP 20 ICD-10")
for code, count in counts.most_common(20):
    print(code, count)

print("\n📊 TOP 10 % DISTRIBUTION")
total_labels = sum(counts.values())

for code, count in counts.most_common(10):
    print(f"{code}: {(count/total_labels)*100:.2f}%")

# ==========================================
# TRAIN / VALID SPLIT
# ==========================================

train_data, val_data = train_test_split(
    processed_data,
    test_size=0.2,
    random_state=42
)

print("\n📦 SPLIT")
print("Train:", len(train_data))
print("Val:", len(val_data))

# ==========================================
# SAVE DATASETS
# ==========================================

def save_jsonl(data, name):
    path = os.path.join(OUTPUT_DIR, name)

    print("\n💾 Saving:", path)

    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("✔ Done:", name)


save_jsonl(train_data, "training_set.jsonl")
save_jsonl(val_data, "validation_set.jsonl")

print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY")
print("==============================\n")