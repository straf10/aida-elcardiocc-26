import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from transformers import AutoModel, AutoTokenizer, logging as transformers_logging

transformers_logging.set_verbosity_error()


def load_data(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_labelset(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def load_label_descriptions(csv_path, labelset):
    desc = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            desc[row["code"]] = row["greek_description"]
    return [desc.get(l, l) for l in labelset]


def load_thresholds_vector(path: str, labelset: list) -> np.ndarray:
    """Load per-class thresholds from JSON (``thresholds`` dict) written by ``train.py``."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "thresholds" not in data:
        raise ValueError(f"Expected JSON with top-level 'thresholds' dict at {path}")
    th_map = data["thresholds"]
    missing = [lab for lab in labelset if lab not in th_map]
    if missing:
        raise ValueError(
            f"Thresholds file {path} is missing {len(missing)} label(s) required by labelset "
            f"(copy the thresholds.json from the same training run): {missing[:15]}"
            + (" ..." if len(missing) > 15 else "")
        )
    return np.array([float(th_map[lab]) for lab in labelset], dtype=np.float64)


class MedicalModelWithDescriptions(nn.Module):
    def __init__(self, model_name, num_labels, label_descriptions, tokenizer, device):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropouts = nn.ModuleList([nn.Dropout(0.1 * (i + 1)) for i in range(5)])
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Precompute description embeddings
        temp_encoder = AutoModel.from_pretrained(model_name).to(device)
        temp_encoder.eval()
        all_cls = []
        with torch.no_grad():
            for desc in label_descriptions:
                enc = tokenizer(
                    desc, max_length=64, truncation=True, padding="max_length", return_tensors="pt"
                ).to(device)
                out = temp_encoder(**enc)
                all_cls.append(out.last_hidden_state[:, 0, :].cpu())
        del temp_encoder
        torch.cuda.empty_cache()
        self.register_buffer("desc_emb", nn.functional.normalize(torch.cat(all_cls, dim=0), dim=-1))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        doc_cls = out.last_hidden_state[:, 0, :]
        base_logits = sum(self.classifier(dp(doc_cls)) for dp in self.dropouts) / len(self.dropouts)
        doc_cls_norm = nn.functional.normalize(doc_cls, dim=-1)
        return base_logits + self.alpha * (doc_cls_norm @ self.desc_emb.T)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    records = load_data(args.data)
    labelset = load_labelset(args.labels)
    label_descs = load_label_descriptions(args.desc_csv, labelset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    thresholds_path = args.thresholds
    if not thresholds_path:
        out_cfg = config.get("output") or {}
        thresholds_path = out_cfg.get("thresholds_path") or os.path.join(args.model_dir, "thresholds.json")

    if not os.path.isfile(thresholds_path):
        raise FileNotFoundError(
            f"Thresholds file not found: {thresholds_path}\n"
            "Copy thresholds.json from your training outputs into the model directory, "
            "or pass --thresholds PATH."
        )
    thresholds = load_thresholds_vector(thresholds_path, labelset)
    print(f"Loaded per-class thresholds from {thresholds_path}")

    with open(os.path.join(args.model_dir, "icd_hierarchy.json"), "r", encoding="utf-8") as f:
        icd_hierarchy = json.load(f)

    l2i = {l: i for i, l in enumerate(labelset)}

    models = []
    for fold in range(args.folds):
        pt_path = os.path.join(args.model_dir, f"model_fold_{fold}.pt")
        if os.path.exists(pt_path):
            print(f"Loading {pt_path}...")
            model = MedicalModelWithDescriptions(
                "xlm-roberta-base", len(labelset), label_descs, tokenizer, device
            ).to(device)
            model.load_state_dict(torch.load(pt_path, map_location=device))
            model.eval()
            models.append(model)

    print(f"Successfully loaded {len(models)} models.")

    predictions = []
    print("Generating predictions...")

    with torch.no_grad():
        for record in records:
            enc = tokenizer(
                record["text"],
                max_length=args.max_len,
                truncation=True,
                truncation_side="left",
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            ensemble_probs = np.zeros(len(labelset))
            for model in models:
                logits = model(enc["input_ids"], enc["attention_mask"])
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                ensemble_probs += probs
            ensemble_probs /= len(models)

            for child, parent in icd_hierarchy.items():
                if child in l2i and parent in l2i:
                    ensemble_probs[l2i[parent]] = max(ensemble_probs[l2i[parent]], ensemble_probs[l2i[child]])

            pred_indices = np.where(ensemble_probs >= thresholds)[0]
            pred_codes = [labelset[idx] for idx in pred_indices]

            formatted_preds = [[code] for code in pred_codes]

            predictions.append(
                {
                    "patient_id": record["patient_id"],
                    "document_level_annotations": formatted_preds,
                }
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Saved {len(predictions)} predictions to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Optional YAML; uses output.thresholds_path if --thresholds omitted")
    # Default split for inference only; xlm_r_base/train.py still uses train+val from YAML/CLI.
    p.add_argument("--data", default="data/processed/test.jsonl")
    p.add_argument("--labels", default="data/raw/labelset.txt")
    p.add_argument("--desc_csv", default="data/external/icd10_greek_lookup.csv")
    p.add_argument("--model_dir", default="outputs/models/xlm_base")
    p.add_argument(
        "--thresholds",
        default=None,
        help="Path to thresholds JSON (from train.py). Default: output.thresholds_path from --config or <model_dir>/thresholds.json",
    )
    p.add_argument("--out", default="outputs/predictions/xlm_r_base/predictions.jsonl")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max_len", type=int, default=512)
    main(p.parse_args())
