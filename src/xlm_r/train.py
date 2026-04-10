import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

try:
    from src.data.dataset import ELCardioDataset
    from src.evaluation.config_utils import get_cfg, load_config
    from src.evaluation.evaluator import evaluate_data
    from src.evaluation.io_utils import load_ground_truth
    from src.training.device_utils import get_device, use_amp_fp16
except ImportError:
    from ..data.dataset import ELCardioDataset
    from ..evaluation.config_utils import get_cfg, load_config
    from ..evaluation.evaluator import evaluate_data
    from ..evaluation.io_utils import load_ground_truth
    from ..training.device_utils import get_device, use_amp_fp16

from .chunk_aggregate import aggregate_scores_by_patient
from .model import build_model, compute_pos_weights


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optimizer_steps_per_epoch(num_batches: int, grad_accum: int) -> int:
    n = 0
    for step in range(num_batches):
        if (step + 1) % grad_accum == 0 or (step + 1) == num_batches:
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Train XLM-R MLC model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--device",
        help="Explicit device to use (e.g., 'cpu', 'cuda', 'mps')",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    model_name = get_cfg(config, "model.name", "xlm-roberta-large")
    num_labels = get_cfg(config, "model.num_labels", 115)

    train_path = get_cfg(config, "data.train_path")
    val_path = get_cfg(config, "data.val_path")
    labelset_path = get_cfg(config, "data.labelset_path")
    frequencies_path = get_cfg(config, "data.frequencies_path")
    max_length = get_cfg(config, "data.max_length", 512)
    sliding_window = get_cfg(config, "data.sliding_window", False)
    stride = get_cfg(config, "data.stride", 256)

    epochs = get_cfg(config, "training.epochs", 10)
    batch_size = get_cfg(config, "training.batch_size", 8)
    grad_accum = get_cfg(config, "training.gradient_accumulation_steps", 4)
    lr = get_cfg(config, "training.learning_rate", 1e-5)
    weight_decay = get_cfg(config, "training.weight_decay", 0.01)
    warmup_ratio = get_cfg(config, "training.warmup_ratio", 0.0)
    fp16 = get_cfg(config, "training.fp16", True)
    seed = get_cfg(config, "training.seed", 42)
    loss_type = get_cfg(config, "training.loss", "bce_weighted")
    focal_gamma = get_cfg(config, "training.focal_gamma", 2.0)
    max_grad_norm = get_cfg(config, "training.max_grad_norm", 1.0)

    checkpoint_dir = get_cfg(config, "output.checkpoint_dir", "outputs/checkpoints")
    scores_path = get_cfg(config, "output.scores_path", "outputs/val_scores.npy")
    pids_path = get_cfg(config, "output.patient_ids_path", "outputs/val_patient_ids.json")
    label_names_path = get_cfg(config, "output.label_names_path", "outputs/label_names.json")
    log_dir = get_cfg(config, "output.log_dir", None)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(scores_path), exist_ok=True)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    set_seed(seed)

    device = get_device(args.device)
    use_amp = use_amp_fp16(device, fp16)
    print(f"Using device: {device} | AMP (fp16): {use_amp}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading datasets...")
    train_dataset = ELCardioDataset(
        train_path,
        labelset_path,
        tokenizer,
        max_length=max_length,
        sliding_window=sliding_window,
        stride=stride,
        is_training=True,
    )
    val_dataset = ELCardioDataset(
        val_path,
        labelset_path,
        tokenizer,
        max_length=max_length,
        sliding_window=sliding_window,
        stride=stride,
        is_training=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Building model...")
    model = build_model(num_labels=num_labels, model_name=model_name)
    model.to(device)

    if loss_type == "bce_weighted":
        pos_weights = compute_pos_weights(
            train_dataset.labels, frequencies_path, len(train_dataset.records)
        )
        pos_weights = pos_weights.to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    elif loss_type == "focal":

        def focal_loss(logits, targets, gamma=focal_gamma):
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            pt = torch.exp(-bce_loss)
            f_loss = ((1 - pt) ** gamma) * bce_loss
            return f_loss.mean()

        criterion = focal_loss
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = _optimizer_steps_per_epoch(len(train_loader), grad_accum)
    total_scheduler_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_scheduler_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_scheduler_steps,
    )

    ground_truth_data = load_ground_truth(val_path)

    best_f1 = 0.0

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * grad_accum
            progress_bar.set_postfix({"loss": total_loss / (step + 1)})

        model.eval()
        pid_to_logits = {}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pids = batch["patient_id"].tolist()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits.cpu().numpy()

                for i, pid in enumerate(pids):
                    if pid not in pid_to_logits:
                        pid_to_logits[pid] = []
                    pid_to_logits[pid].append(logits[i])

        unique_pids, aggregated_scores = aggregate_scores_by_patient(pid_to_logits)

        preds_bin = aggregated_scores >= 0.5
        pred_data = {}
        for i, pid in enumerate(unique_pids):
            pred_indices = np.where(preds_bin[i])[0]
            pred_data[pid] = [train_dataset.labels[idx] for idx in pred_indices]

        metrics = evaluate_data(
            ground_truth_data, pred_data, label_space=train_dataset.labels
        )
        val_f1 = metrics["micro_f1"]
        print(f"Epoch {epoch + 1} - Val Micro-F1 (thresh=0.5): {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"New best F1! Saving model to {checkpoint_dir}")
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            np.save(scores_path, aggregated_scores)
            with open(pids_path, "w", encoding="utf-8") as f:
                json.dump(unique_pids, f)
            with open(label_names_path, "w", encoding="utf-8") as f:
                json.dump(train_dataset.labels, f)

    print(f"Training complete. Best Val F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
