"""
Inference script — generates a submission-ready JSONL from a trained MLC model.
Owner: Vasiliki

Usage:
    # With default threshold (0.6):
    python -m mlc_greek_bert.predict \
        --config src/mlc_greek_bert/mlc_greek_bert.yaml \
        --checkpoint checkpoints/mlc_greek_bert/best_model.pt \
        --input data/processed/test.jsonl \
        --output submissions/mlc_greek_bert.jsonl

    # With tuned per-class thresholds from Strafiotis:
    python -m mlc_greek_bert.predict \
        --config src/mlc_greek_bert/mlc_greek_bert.yaml \
        --checkpoint checkpoints/mlc_greek_bert/best_model.pt \
        --input data/processed/test.jsonl \
        --output submissions/mlc_greek_bert_tuned.jsonl \
        --thresholds checkpoints/mlc_greek_bert/best_thresholds.json

Output format (per line):
    {"patient_id": 1234, "document_level_annotations": [["I21.0"], ["I10"], ["I50.0"]]}
"""

import json
import argparse
import numpy as np
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlc_greek_bert.model import MLCModel
from mlc_greek_bert.train import CardioDataset


def predict(config: dict, checkpoint_path: str, input_path: str, output_path: str, thresholds_path: str = None, export_scores: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load label names from checkpoint dir (saved during training)
    ckpt_dir = Path(checkpoint_path).parent
    with open(ckpt_dir / "labels.json", "r", encoding="utf-8") as f:
        label_names = json.load(f)

    # Load per-class thresholds (from Strafiotis's threshold_tune.py) or default 0.6
    if thresholds_path:
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresh_dict = json.load(f)
        thresholds = np.array([thresh_dict[label] for label in label_names])
        print(f"Loaded per-class thresholds from {thresholds_path}")
    else:
        thresholds = np.full(len(label_names), 0.6)
        print("Using default threshold 0.6 for all classes")

    # Tokenizer + dataset
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    dataset = CardioDataset(input_path, label_names, tokenizer, config["model"]["max_length"])
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"] * 2, shuffle=False, num_workers=8)

    # Load model
    model = MLCModel(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        dropout=0.0,          # No dropout at inference
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Run inference
    all_scores = []
    all_pids = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)
            all_pids.extend(batch["patient_id"].tolist())

    all_scores = np.vstack(all_scores)

    # Write submission JSONL
    # Each predicted code goes in its own single-element group: [["I10"], ["I21.0"]]
    # This satisfies the submission format while being evaluated correctly.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, pid in enumerate(all_pids):
            pred_codes = [
                label_names[j]
                for j, score in enumerate(all_scores[i])
                if score >= thresholds[j]
            ]

            # Wrap each predicted code as its own group
            annotations = [[code] for code in pred_codes]

            # Safety: if model predicts nothing, emit empty list (will score as all FN)
            record = {
                "patient_id": pid,
                "document_level_annotations": annotations,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} predictions to {output_path}")
    print(f"Avg codes per doc: {np.mean([(all_scores[i] >= thresholds).sum() for i in range(len(all_pids))]):.2f}")

    if export_scores:
        scores_path = config.get("output", {}).get("scores_path", "checkpoints/mlc_greek_bert/val_scores.npy")
        pids_path = config.get("output", {}).get("pids_path", "checkpoints/mlc_greek_bert/val_pids.json")
        label_names_path = config.get("output", {}).get("label_names_path", "checkpoints/mlc_greek_bert/label_names.json")

        Path(scores_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(scores_path, all_scores)
        with open(pids_path, "w", encoding="utf-8") as f:
            json.dump(all_pids, f)
        with open(label_names_path, "w", encoding="utf-8") as f:
            json.dump(label_names, f)
        print(f"Exported score artifacts to {Path(scores_path).parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--checkpoint",
        default="outputs/models/greek_bert/best_model.pt",
        help="Path to best_model.pt",
    )
    parser.add_argument(
        "--input",
        default="data/processed/validation_set.jsonl",
        help="Path to input JSONL (test or val)",
    )
    parser.add_argument(
        "--output",
        default="outputs/predictions/mlc_greek_bert/predictions.jsonl",
        help="Path to output predictions JSONL",
    )
    parser.add_argument(
        "--thresholds",
        default="outputs/models/greek_bert/best_thresholds.json",
        help="Path to per-class thresholds JSON (optional; uses 0.6 per class if missing)",
    )
    parser.add_argument("--export-scores", action="store_true", help="Export score artifacts for analysis")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    thresh_path = args.thresholds
    if thresh_path and not Path(thresh_path).is_file():
        print(f"WARN: thresholds not found at {thresh_path}, using default 0.6 per class")
        thresh_path = None

    predict(config, args.checkpoint, args.input, args.output, thresh_path, args.export_scores)