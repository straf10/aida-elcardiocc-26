import json
import re
import os
from collections import Counter
import requests
import csv


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


def save_jsonl(data, path):
    print("\nSaving:", path)
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Done:", path)


def main():
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

    counts = Counter(all_labels)

    print("\nALL ICD-10 FREQUENCIES")
    print("==============================")

    for code, count in counts.most_common():
        print(f"{code}: {count}")

    print("==============================")
    print("Unique ICD-10 codes:", len(counts))

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

    cleaned_jsonl = os.path.join(OUTPUT_DIR, "cleaned.jsonl")
    save_jsonl(processed_data, cleaned_jsonl)

    print("\nPIPELINE COMPLETED SUCCESSFULLY")
    print("==============================\n")


if __name__ == "__main__":
    main()
