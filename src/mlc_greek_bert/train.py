"""
Training script for Greek-BERT Multi-Label Classifier.
Owner: Vasiliki
Track: Greek-BERT (nlpaueb/bert-base-greek-uncased-v1)

Usage:
    python -m src.mlc_greek_bert.train --config src/mlc_greek_bert/mlc_greek_bert.yaml

W&B logs every run automatically. Check the dashboard for val_micro_f1.
"""

import json
import random
import argparse
import numpy as np
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import wandb
import yaml

import unicodedata

def strip_accents_and_lowercase(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).lower()

# Add project root to path so we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.mlc_greek_bert.model import MLCModel
from src.evaluation.evaluator import score_document, micro_f1


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic GPU behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # (Optional but strong) enforce determinism
    # torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ── Dataset ───────────────────────────────────────────────────────────────────

class CardioDataset(Dataset):
    """
    Loads ELCardioCC JSONL data.
    Each item:  text (str), label_vector (FloatTensor of shape 115,), patient_id (int)
    """

    def __init__(self, jsonl_path: str, label_names: list, tokenizer, max_length: int):
        self.label2idx = {label: i for i, label in enumerate(label_names)}
        self.num_labels = len(label_names)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self.records.append(record)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        text = strip_accents_and_lowercase(record["text"])
        #text = record["text"]
        patient_id = record["patient_id"]

        # Build multi-hot label vector
        label_vector = torch.zeros(self.num_labels, dtype=torch.float32)
        for group in record.get("document_level_annotations", []):
            for code in group:
                if code in self.label2idx:
                    label_vector[self.label2idx[code]] = 1.0

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label_vector,
            "patient_id": patient_id,
        }


# ── Class weights for imbalance ───────────────────────────────────────────────

def compute_pos_weights(jsonl_path: str, label_names: list) -> torch.Tensor:
    """
    Computes per-class pos_weight = (N - pos) / pos for BCEWithLogitsLoss.
    Rare codes get higher weights so the model pays more attention to them.
    """
    label2idx = {label: i for i, label in enumerate(label_names)}
    counts = np.zeros(len(label_names))
    total = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1
            seen = set()
            for group in record.get("document_level_annotations", []):
                for code in group:
                    if code in label2idx and code not in seen:
                        counts[label2idx[code]] += 1
                        seen.add(code)

    # Avoid division by zero for codes with 0 occurrences
    pos_weight = np.where(counts > 0, (total - counts) / np.maximum(counts, 1), total)
    return torch.tensor(pos_weight, dtype=torch.float32)


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model, val_loader, label_names, device, criterion):
    """
    Runs inference on validation set, returns micro-F1 + raw sigmoid scores.
    Uses threshold 0.5 here — proper per-class tuning is done by threshold_tune.py.
    """
    model.eval()
    all_scores = []
    all_pids = []
    gold_data = {}
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            pids = batch["patient_id"]

            logits = model(input_ids, attention_mask)

            # validation loss
            val_loss = criterion(logits, labels.to(device))
            total_val_loss += val_loss.item()

            scores = torch.sigmoid(logits).cpu().numpy()

            all_scores.append(scores)
            all_pids.extend(pids.tolist())

            # Reconstruct gold groups from label vectors for micro-F1 calculation
            for i, pid in enumerate(pids.tolist()):
                gold_groups = [[label_names[j]] for j in range(len(label_names)) if labels[i][j] == 1.0]
                gold_data[pid] = gold_groups

    all_scores = np.vstack(all_scores)
    avg_val_loss = total_val_loss / len(val_loader)   

    # Compute micro-F1 at threshold 0.5
    total_tp, total_fp, total_fn = 0, 0, 0
    for i, pid in enumerate(all_pids):
        pred_codes = [label_names[j] for j, s in enumerate(all_scores[i]) if s >= 0.6]
        tp, fp, fn = score_document(gold_data[pid], pred_codes)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    p, r, f1 = micro_f1(total_tp, total_fp, total_fn)
    return f1, p, r, all_scores, all_pids, avg_val_loss


# ── Main training loop ────────────────────────────────────────────────────────

def train(config: dict):
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generator for deterministic DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(config["training"]["seed"])
    
    # Load label names
   # with open(config["data"]["labels_path"], "r", encoding="utf-8") as f:
   #     label_names = json.load(f)
   # assert len(label_names) == config["model"]["num_labels"], \
   #     f"Expected {config['model']['num_labels']} labels, got {len(label_names)}"


    # Build label names from training data directly
    labels_path = "data/raw/Train_Set_2026/labelset.txt"

    with open(labels_path, "r", encoding="utf-8") as f:
        label_names = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(label_names)} labels from labelset.txt")

    #all_codes = set()
    #with open(config["data"]["train_path"], "r", encoding="utf-8") as f:
    #    for line in f:
    #        line = line.strip()
    #        if not line:
    #            continue
    #        record = json.loads(line)
    #        for group in record.get("document_level_annotations", []):
    #            for code in group:
    #                all_codes.add(code)
    #label_names = sorted(all_codes)

    #print(f"Found {len(label_names)} unique labels in training data")
    assert len(label_names) == config["model"]["num_labels"], \
        f"Expected {config['model']['num_labels']} labels, got {len(label_names)}"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    # Datasets & loaders
    train_dataset = CardioDataset(
        config["data"]["train_path"], label_names, tokenizer, config["model"]["max_length"]
    )
    val_dataset = CardioDataset(
        config["data"]["val_path"], label_names, tokenizer, config["model"]["max_length"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Model
    model = MLCModel(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        dropout=config["model"]["dropout"],
    ).to(device)

    # Loss — class-weighted BCE for imbalance
    if config["training"]["use_class_weights"]:
        pos_weights = compute_pos_weights(config["data"]["train_path"], label_names).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        print("Using class-weighted BCEWithLogitsLoss")
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    #total_steps = (len(train_loader) // config["training"]["grad_accum_steps"]) * config["training"]["epochs"]
    total_steps = math.ceil(len(train_loader) / config["training"]["grad_accum_steps"]) * config["training"]["epochs"]
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # W&B init
    wandb.init(
        project=config["wandb"]["project"],
        name=config["wandb"]["run_name"],
        tags=config["wandb"]["tags"],
        config={
            "model": config["model"]["name"],
            "learning_rate": config["training"]["learning_rate"],
            "batch_size": config["training"]["batch_size"],
            "max_length": config["model"]["max_length"],
            "epochs": config["training"]["epochs"],
            "use_class_weights": config["training"]["use_class_weights"],
            "grad_accum_steps": config["training"]["grad_accum_steps"],
        },
    )

    # Output dir
    ckpt_dir = Path(config["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    best_epoch = 0
    global_step = 0

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        # track train F1
        train_tp, train_fp, train_fn = 0, 0, 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = loss / config["training"]["grad_accum_steps"]
            loss.backward()
            total_loss += loss.item() * config["training"]["grad_accum_steps"]

            # compute train F1 on-the-fly
            with torch.no_grad():
                scores = torch.sigmoid(logits).cpu().numpy()
                preds = (scores >= 0.7)

                for i in range(len(labels)):
                    pred_codes = [label_names[j] for j in range(len(label_names)) if preds[i][j]]
                    gold_groups = [[label_names[j]] for j in range(len(label_names)) if labels[i][j].item() == 1.0]

                    tp, fp, fn = score_document(gold_groups, pred_codes)
                    train_tp += tp
                    train_fp += fp
                    train_fn += fn

            if (step + 1) % config["training"]["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
        
        
        # flush remaining gradients at end of epoch 
        if len(train_loader) % config["training"]["grad_accum_steps"] != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)

        # compute train F1
        _, _, train_f1 = micro_f1(train_tp, train_fp, train_fn)


        # Validate every epoch
        val_f1, val_p, val_r, val_scores, val_pids, avg_val_loss  = validate(model, val_loader, label_names, device, criterion)

        print(
            f"Epoch {epoch}/{config['training']['epochs']} | "
            f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | "
            f"P: {val_p:.4f} | R: {val_r:.4f}"
        )

        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
            "train_micro_f1": train_f1,
            "val_micro_f1": val_f1,
            "val_precision": val_p,
            "val_recall": val_r,
            "lr": scheduler.get_last_lr()[0],
        })

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

            # Save val scores for threshold tuning (Strafiotis needs these)
            np.save(config["output"]["scores_path"], val_scores)
            with open(config["output"]["pids_path"], "w", encoding="utf-8") as f:
                json.dump(val_pids, f)

            print(f"  ✓ New best model saved (F1={best_f1:.4f})")

    print(f"\nTraining complete. Best val F1: {best_f1:.4f} at epoch {best_epoch}")
    wandb.summary["best_val_micro_f1"] = best_f1
    wandb.finish()

    # Save label names alongside checkpoints so predict_mlc.py can load them
    with open(ckpt_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_names, f)

    return best_f1


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)