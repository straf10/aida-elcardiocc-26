import argparse
import json
import os
import random

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

try:
    from src.data.dataset import ELCardioDataset
    from src.evaluation.config_utils import get_cfg, load_config
    from src.evaluation.evaluator import evaluate_data
    from src.evaluation.io_utils import load_ground_truth
    from src.training.device_utils import get_device, use_amp_fp16
    from src.training.dotenv_util import load_dotenv_if_present
except ImportError:
    from ..data.dataset import ELCardioDataset
    from ..evaluation.config_utils import get_cfg, load_config
    from ..evaluation.evaluator import evaluate_data
    from ..evaluation.io_utils import load_ground_truth
    from ..training.device_utils import get_device, use_amp_fp16
    from ..training.dotenv_util import load_dotenv_if_present

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

    load_dotenv_if_present()

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
    scheduler_type = get_cfg(config, "training.scheduler", "linear")
    eval_threshold = get_cfg(config, "training.eval_threshold", 0.15)
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

    wb_enabled  = get_cfg(config, "wandb.enabled", False)
    wb_project  = get_cfg(config, "wandb.project", "elcardiocc-2026")
    wb_entity   = get_cfg(config, "wandb.entity", None)
    wb_run_name = get_cfg(config, "wandb.run_name", None)
    wb_notes    = get_cfg(config, "wandb.notes", "")
    wb_tags     = get_cfg(config, "wandb.tags", [])

    if wb_enabled:
        wandb.init(
            project=wb_project,
            entity=wb_entity,
            name=wb_run_name,
            notes=wb_notes,
            tags=wb_tags,
            config={
                "model_name": model_name,
                "learning_rate": lr,
                "batch_size": batch_size,
                "effective_batch_size": batch_size * grad_accum,
                "max_length": max_length,
                "num_epochs": epochs,
                "loss_function": loss_type,
                "focal_gamma": focal_gamma if loss_type == "focal" else None,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "scheduler": scheduler_type,
                "eval_threshold": eval_threshold,
                "sliding_window": sliding_window,
                "stride": stride if sliding_window else None,
                "seed": seed,
                "fp16": fp16,
                "device": str(device),
            },
        )

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

    # Apply Layer-wise Learning Rate Decay (LLRD)
    head_lr = lr * 5.0
    embedding_lr = lr * 0.1
    middle_lr = lr * 0.5
    top_lr = lr * 1.0

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    
    if hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    else:
        num_layers = 24  # default fallback

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        weight_decay_val = 0.0 if any(nd in n for nd in no_decay) else weight_decay
        
        if "classifier" in n:
            param_lr = head_lr
        elif "roberta.embeddings" in n:
            param_lr = embedding_lr
        elif "roberta.encoder.layer" in n:
            try:
                layer_num = int(n.split("roberta.encoder.layer.")[1].split(".")[0])
                if layer_num < num_layers // 3:
                    param_lr = embedding_lr
                elif layer_num < (2 * num_layers) // 3:
                    param_lr = middle_lr
                else:
                    param_lr = top_lr
            except ValueError:
                param_lr = top_lr
        else:
            param_lr = top_lr
            
        optimizer_grouped_parameters.append({
            "params": [p],
            "lr": param_lr,
            "weight_decay": weight_decay_val
        })

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = _optimizer_steps_per_epoch(len(train_loader), grad_accum)
    total_scheduler_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_scheduler_steps * warmup_ratio)
    
    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_scheduler_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_scheduler_steps,
        )

    ground_truth_data = load_ground_truth(val_path)

    best_f1 = 0.0
    global_step = 0

    print("Starting training...")
    if wb_enabled and loss_type == "bce_weighted":
        wandb.log({
            "pos_weights/max": pos_weights.max().item(),
            "pos_weights/mean": pos_weights.mean().item()
        })
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
                global_step += 1
                scaler.unscale_(optimizer)
                grad_norm = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)**0.5
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                if wb_enabled:
                    wandb.log({
                        "train/loss_step": loss.item() * grad_accum,
                        "train/grad_norm": grad_norm.item()
                    }, step=global_step)

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

        preds_bin = aggregated_scores >= eval_threshold
        pred_data = {}
        for i, pid in enumerate(unique_pids):
            pred_indices = np.where(preds_bin[i])[0]
            pred_data[pid] = [train_dataset.labels[idx] for idx in pred_indices]

        metrics = evaluate_data(
            ground_truth_data, pred_data, label_space=train_dataset.labels
        )
        val_f1 = metrics["micro_f1"]
        print(f"Epoch {epoch + 1} - Val Micro-F1 (thresh={eval_threshold}): {val_f1:.4f}")

        if wb_enabled:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss_epoch": total_loss / len(train_loader),
                "val/micro_f1": val_f1,
                "val/precision": metrics["precision"],
                "val/recall": metrics["recall"],
                "lr": scheduler.get_last_lr()[0],
            }
            
            # Log per-threshold sweep
            for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
                t_preds_bin = aggregated_scores >= t
                t_pred_data = {}
                for i, pid in enumerate(unique_pids):
                    t_pred_indices = np.where(t_preds_bin[i])[0]
                    t_pred_data[pid] = [train_dataset.labels[idx] for idx in t_pred_indices]
                t_metrics = evaluate_data(ground_truth_data, t_pred_data, label_space=train_dataset.labels)
                log_dict[f"val/f1_thresh_{t}"] = t_metrics["micro_f1"]

            if device.type == "cuda":
                log_dict["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
                log_dict["gpu/memory_reserved_gb"]  = torch.cuda.memory_reserved(device) / 1e9
            wandb.log(log_dict, step=global_step)

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

            if wb_enabled and "per_class" in metrics:
                rows = [[r["code"], r["support"], round(r["f1"], 4),
                         round(r["precision"], 4), round(r["recall"], 4)]
                        for r in metrics["per_class"]]
                table = wandb.Table(
                    columns=["code", "support", "f1", "precision", "recall"],
                    data=rows,
                )
                wandb.log({"per_class_f1": table}, step=global_step)
            
            if wb_enabled:
                artifact = wandb.Artifact(
                    name="model-best",
                    type="model",
                    metadata={"val_micro_f1": best_f1, "epoch": epoch + 1},
                )
                artifact.add_dir(checkpoint_dir)
                wandb.log_artifact(artifact)

    print(f"Training complete. Best Val F1: {best_f1:.4f}")

    if wb_enabled:
        wandb.summary["best_val_micro_f1"] = best_f1
        wandb.finish()


if __name__ == "__main__":
    main()
