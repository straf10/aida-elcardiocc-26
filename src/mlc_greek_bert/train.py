"""
Training script for Greek-BERT Multi-Label Classifier.
Owner: Vasiliki
Track: Greek-BERT (nlpaueb/bert-base-greek-uncased-v1)

Usage:
    python -m mlc_greek_bert.train --config src/mlc_greek_bert/mlc_greek_bert.yaml

W&B logs every run automatically. Check the dashboard for val_micro_f1.
"""

import os
import sys

_REPO_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

import json
import random
import argparse
import numpy as np
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import wandb
import yaml

import unicodedata
from typing import Dict, List

def strip_accents_and_lowercase(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).lower()

from mlc_greek_bert.model import MLCModel
from evaluation.evaluator import evaluate_data
from evaluation.io_utils import load_ground_truth
from preprocessing.io_utils import LABELSET_PATH, load_labelset


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

def validate(
    model,
    loader,
    label_names: List[str],
    device,
    criterion,
    ground_truth_data: Dict[int, List[List[str]]],
    eval_threshold: float = 0.6,
):
    """
    Runs inference on validation set, returns micro-F1 + raw sigmoid scores.
    Uses eval_threshold here — proper per-class tuning is done by threshold_tune.py.
    Gold labels come from ground_truth_data (list-of-lists per patient), not from multi-hot.
    """
    model.eval()
    all_scores: List[np.ndarray] = []
    all_pids: List[int] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pids = batch["patient_id"]

            logits = model(input_ids, attention_mask)
            total_loss += criterion(logits, labels).item()

            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)
            all_pids.extend(pids.tolist())

    all_scores = np.vstack(all_scores)
    avg_loss = total_loss / len(loader)

    pred_data: Dict[int, List[str]] = {
        int(pid): [label_names[j] for j, s in enumerate(all_scores[i]) if s >= eval_threshold]
        for i, pid in enumerate(all_pids)
    }

    metrics = evaluate_data(ground_truth_data, pred_data, label_space=label_names)
    return (
        metrics["micro_f1"],
        metrics["precision"],
        metrics["recall"],
        all_scores,
        all_pids,
        avg_loss,
    )


# ── Main training loop ────────────────────────────────────────────────────────

def train(config: dict):
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generator for deterministic DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(config["training"]["seed"])
    
    label_names = load_labelset(LABELSET_PATH)
    print(f"Loaded {len(label_names)} labels from {LABELSET_PATH}")
    assert len(label_names) == config["model"]["num_labels"], \
        f"Expected {config['model']['num_labels']} labels, got {len(label_names)}"

    num_workers = config["training"].get("num_workers", 0)

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
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Ground truth dicts for evaluate_data
    val_ground_truth = load_ground_truth(config["data"]["val_path"])
    train_ground_truth = load_ground_truth(config["data"]["train_path"])

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

    # W&B init — skip if sweep agent already called init
    if wandb.run is None:
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
                "dropout": config["model"]["dropout"],
                "warmup_ratio": config["training"]["warmup_ratio"],
                "weight_decay": config["training"]["weight_decay"],
            },
        )
    else:
        wandb.config.update({
            "model": config["model"]["name"],
            "learning_rate": config["training"]["learning_rate"],
            "batch_size": config["training"]["batch_size"],
            "max_length": config["model"]["max_length"],
            "epochs": config["training"]["epochs"],
            "dropout": config["model"]["dropout"],
        })

    # Output dir
    ckpt_dir = Path(config["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    labels_path = Path(config["output"]["labels_path"])
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(label_names, f)

    best_f1 = 0.0
    best_epoch = 0
    global_step = 0

    eval_threshold = 0.6

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        train_pred_buffer: Dict[int, List[str]] = {}

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pids = batch["patient_id"].tolist()

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = loss / config["training"]["grad_accum_steps"]
            loss.backward()
            total_loss += loss.item() * config["training"]["grad_accum_steps"]

            # Accumulate train predictions for end-of-epoch evaluate_data (group-level micro-F1)
            with torch.no_grad():
                scores = torch.sigmoid(logits).cpu().numpy()
                preds = scores >= eval_threshold

                for i in range(len(pids)):
                    pid = int(pids[i])
                    pred_codes = [label_names[j] for j in range(len(label_names)) if preds[i][j]]
                    train_pred_buffer[pid] = pred_codes

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

        train_metrics = evaluate_data(
            train_ground_truth, train_pred_buffer, label_space=label_names
        )
        train_f1 = train_metrics["micro_f1"]

        # Validate every epoch
        val_f1, val_p, val_r, val_scores, val_pids, avg_val_loss = validate(
            model,
            val_loader,
            label_names,
            device,
            criterion,
            ground_truth_data=val_ground_truth,
            eval_threshold=eval_threshold,
        )

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

 
    # ── Final test-set evaluation (skipped during sweeps) ────────────────────
    is_sweep = wandb.run is not None and wandb.run.sweep_id is not None
    if not is_sweep and config["data"].get("test_path"):
        print("\nEvaluating best model on held-out test set...")
        model.load_state_dict(torch.load(ckpt_dir / "best_model.pt", map_location=device))
 
        test_ground_truth = load_ground_truth(config["data"]["test_path"])
        test_dataset = CardioDataset(
            config["data"]["test_path"], label_names, tokenizer, config["model"]["max_length"]
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["training"]["batch_size"] * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        test_f1, test_p, test_r, _, _, test_loss = validate(
            model, test_loader, label_names, device, criterion,
            ground_truth_data=test_ground_truth,
            eval_threshold=eval_threshold,
        )
        print(
            f"Test set results — "
            f"F1: {test_f1:.4f} | P: {test_p:.4f} | R: {test_r:.4f} | Loss: {test_loss:.4f}"
        )
        wandb.summary["test_micro_f1"]  = test_f1
        wandb.summary["test_precision"] = test_p
        wandb.summary["test_recall"]    = test_r
    # ─────────────────────────────────────────────────────────────────────────

    wandb.finish()
 
    return best_f1


# ── Entry point ───────────────────────────────────────────────────────────────


def train_for_sweep(config: dict):
    """Called by the W&B sweep agent. Overrides config with sweep-chosen values."""
    with wandb.init():
        sweep_cfg = wandb.config
 
        if "learning_rate" in sweep_cfg:
            config["training"]["learning_rate"] = sweep_cfg["learning_rate"]
        if "dropout" in sweep_cfg:
            config["model"]["dropout"] = sweep_cfg["dropout"]
        if "batch_size" in sweep_cfg:
            config["training"]["batch_size"] = sweep_cfg["batch_size"]
        if "warmup_ratio" in sweep_cfg:
            config["training"]["warmup_ratio"] = sweep_cfg["warmup_ratio"]
        if "weight_decay" in sweep_cfg:
            config["training"]["weight_decay"] = sweep_cfg["weight_decay"]
        if "epochs" in sweep_cfg:
            config["training"]["epochs"] = sweep_cfg["epochs"]
 
        lr = config["training"]["learning_rate"]
        bs = config["training"]["batch_size"]
        do = config["model"]["dropout"]
        config["wandb"]["run_name"] = f"sweep-lr{lr:.0e}-bs{bs}-do{do}"
 
        train(config)
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--sweep", action="store_true", help="Run as W&B sweep agent")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.sweep:
        train_for_sweep(config)
    else:
        train(config)