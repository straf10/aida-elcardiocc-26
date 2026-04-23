from __future__ import annotations

import os
import sys

_REPO_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

import argparse
import json
import random
import uuid
from collections import Counter
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from evaluation.config_utils import get_cfg, load_config
from evaluation.evaluator import evaluate_data
from evaluation.io_utils import load_ground_truth
from evaluation.threshold_tune import tune_thresholds
from preprocessing.dataset import ELCardioDataset
from preprocessing.io_utils import load_jsonl
from split_data.device_utils import get_device, use_amp_fp16
from split_data.dotenv_util import load_dotenv_if_present

from xlm_r_large.chunk_aggregate import aggregate_scores_by_patient
from xlm_r_large.model import build_model


def _safe_model_slug(model_name: str, max_len: int = 100) -> str:
    return model_name.replace("/", "-").replace(".", "-")[:max_len]


def _bump_run_counter(counter_path: str) -> int:
    """Atomically increment a local run counter (best-effort; not multi-process safe)."""
    path = os.path.abspath(counter_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    n = 1
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as handle:
                n = int(handle.read().strip()) + 1
        except ValueError:
            n = 1
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(n))
    return n


def make_wandb_run_name(
    model_name: str,
    explicit: str | None,
    *,
    style: str = "random",
    counter_path: str | None = None,
) -> str:
    """
    Build a W&B run name: fixed ``explicit`` (unless null/empty/auto), else
    ``{model}-{counter}`` or ``{model}-{random_hex}`` per ``style``.
    """
    safe = _safe_model_slug(model_name)
    if explicit is not None:
        s = str(explicit).strip()
        if s and s.lower() != "auto":
            return s
    style_l = str(style or "random").lower()
    if style_l == "counter":
        path = counter_path or "outputs/experiments/xlm_r_large/wandb_run_counter.txt"
        n = _bump_run_counter(path)
        return f"{safe}-{n}"
    return f"{safe}-{uuid.uuid4().hex[:8]}"


def set_seed(seed: int) -> None:
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


def _cosine_with_warmup_lambda(
    current_step: int,
    num_warmup_steps: int,
    num_training_steps: int,
) -> float:
    if num_training_steps <= 0:
        return 1.0
    if current_step < num_warmup_steps:
        return float(current_step) / max(1, num_warmup_steps)
    progress = float(current_step - num_warmup_steps) / max(
        1, num_training_steps - num_warmup_steps
    )
    return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))


def _asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip: float = 0.05,
) -> torch.Tensor:
    probs_pos = torch.sigmoid(logits)
    probs_neg = 1.0 - probs_pos
    if clip > 0:
        probs_neg = (probs_neg + clip).clamp(max=1.0)
    loss_pos = targets * torch.log(probs_pos.clamp(min=1e-8))
    loss_neg = (1.0 - targets) * torch.log(probs_neg.clamp(min=1e-8))
    loss = -loss_pos * (1.0 - probs_pos).pow(gamma_pos) - loss_neg * probs_pos.pow(
        gamma_neg
    )
    return loss.mean()


def _zlpr_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pos_mask = targets.bool()
    neg_mask = ~pos_mask
    neg_logits = logits.masked_fill(pos_mask, float("-inf"))
    pos_logits = (-logits).masked_fill(neg_mask, float("-inf"))
    zeros = torch.zeros_like(pos_logits[..., :1])
    loss_pos = torch.logsumexp(torch.cat([zeros, pos_logits], dim=-1), dim=-1)
    loss_neg = torch.logsumexp(torch.cat([zeros, neg_logits], dim=-1), dim=-1)
    return (loss_pos + loss_neg).mean()


def _no_upload_settings() -> wandb.Settings:
    try:
        return wandb.Settings(
            save_code=False,
            disable_job_creation=True,
        )
    except TypeError:
        return wandb.Settings(save_code=False)


def _compute_pos_weight(
    train_records: list[dict],
    labelset: list[str],
    cap: float = 20.0,
) -> torch.Tensor:
    counts: Counter[str] = Counter()
    for record in train_records:
        for code in record.get("labels_flat", []):
            counts[str(code)] += 1
    n = len(train_records)
    values = []
    for label in labelset:
        pos = counts.get(str(label), 0)
        values.append((n - pos) / max(pos, 1))
    return torch.tensor(np.clip(np.array(values, dtype=np.float32), 1.0, float(cap)))


def _freeze_bottom_layers(model: torch.nn.Module, freeze_layers: int) -> None:
    if int(freeze_layers) <= 0:
        return
    for name, param in model.named_parameters():
        if (
            "roberta.embeddings" in name
            or "xlm_roberta.embeddings" in name
            or "backbone.embeddings" in name
        ):
            param.requires_grad = False
            continue

        layer_num = None
        for marker in (
            "roberta.encoder.layer.",
            "xlm_roberta.encoder.layer.",
            "backbone.encoder.layer.",
        ):
            if marker in name:
                try:
                    layer_num = int(name.split(marker)[1].split(".")[0])
                except ValueError:
                    layer_num = None
                break
        if layer_num is not None and layer_num < int(freeze_layers):
            param.requires_grad = False


def _build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": float(weight_decay)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(lr),
    )


def _get_prefetch_factor(num_workers: int) -> int | None:
    return 4 if int(num_workers) > 0 else None


def run(
    config: dict[str, Any],
    wandb_init_kwargs: dict[str, Any] | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    model_name = get_cfg(config, "model.name", "xlm-roberta-large")
    num_labels = int(get_cfg(config, "model.num_labels", 115))

    train_path = get_cfg(config, "data.train_path")
    val_path = get_cfg(config, "data.val_path")
    labelset_path = get_cfg(config, "data.labelset_path")
    max_length = int(get_cfg(config, "data.max_length", 512))
    sliding_window = bool(get_cfg(config, "data.sliding_window", False))
    stride = int(get_cfg(config, "data.stride", 256))
    chunk_strategy = get_cfg(config, "data.chunk_strategy", "random")
    truncation_side = get_cfg(config, "data.truncation_side", "right")
    chunk_aggregation = str(get_cfg(config, "data.chunk_aggregation", "mean_max"))
    chunk_aggregation_alpha = float(get_cfg(config, "data.chunk_aggregation_alpha", 0.5))

    epochs = int(get_cfg(config, "training.epochs", 30))
    batch_size = int(get_cfg(config, "training.batch_size", 2))
    eval_batch_size = int(get_cfg(config, "training.eval_batch_size", 16))
    grad_accum = int(get_cfg(config, "training.gradient_accumulation_steps", 16))
    lr = float(get_cfg(config, "training.learning_rate", 1e-5))
    weight_decay = float(get_cfg(config, "training.weight_decay", 0.05))
    warmup_ratio = float(get_cfg(config, "training.warmup_ratio", 0.10))
    eval_threshold = float(get_cfg(config, "training.eval_threshold", 0.5))
    eval_thresholds = [float(t) for t in get_cfg(config, "training.eval_thresholds", [0.5])]
    if eval_threshold not in eval_thresholds:
        eval_thresholds.append(eval_threshold)
    fp16 = bool(get_cfg(config, "training.fp16", True))
    seed = int(get_cfg(config, "training.seed", 42))
    loss_type = str(get_cfg(config, "training.loss", "asl"))
    asl_gamma_neg = float(get_cfg(config, "training.asl_gamma_neg", 4.0))
    asl_gamma_pos = float(get_cfg(config, "training.asl_gamma_pos", 1.0))
    asl_clip = float(get_cfg(config, "training.asl_clip", 0.05))
    pos_weight_cap = float(get_cfg(config, "training.pos_weight_cap", 20.0))
    max_grad_norm = float(get_cfg(config, "training.max_grad_norm", 1.0))
    early_stopping_patience = int(get_cfg(config, "training.early_stopping_patience", 7))
    freeze_layers = int(get_cfg(config, "training.freeze_layers", 0))
    classifier_dropout = float(get_cfg(config, "training.classifier_dropout", 0.3))
    auto_threshold_tuning = bool(get_cfg(config, "training.auto_threshold_tuning", False))
    gradient_checkpointing = bool(get_cfg(config, "training.gradient_checkpointing", False))
    compile_model = bool(get_cfg(config, "training.compile_model", False))
    num_workers = int(get_cfg(config, "training.num_workers", 4))
    pin_memory = bool(get_cfg(config, "training.pin_memory", True))

    threshold_min = float(get_cfg(config, "threshold_tuning.min", 0.05))
    threshold_max = float(get_cfg(config, "threshold_tuning.max", 0.95))
    threshold_step = float(get_cfg(config, "threshold_tuning.step", 0.01))

    checkpoint_dir = get_cfg(
        config,
        "output.checkpoint_dir",
        "outputs/experiments/xlm_r_large/checkpoints",
    )
    scores_path = get_cfg(
        config,
        "output.scores_path",
        "outputs/experiments/xlm_r_large/val_scores.npy",
    )
    pids_path = get_cfg(
        config,
        "output.patient_ids_path",
        "outputs/experiments/xlm_r_large/val_patient_ids.json",
    )
    label_names_path = get_cfg(
        config,
        "output.label_names_path",
        "outputs/experiments/xlm_r_large/label_names.json",
    )
    thresholds_path = get_cfg(
        config,
        "output.thresholds_path",
        "outputs/models/xlm_r_large/thresholds.json",
    )
    log_dir = get_cfg(config, "output.log_dir", None)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(scores_path), exist_ok=True)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Maximize GPU throughput where possible.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    set_seed(seed)

    device = get_device(device_override)
    use_amp = use_amp_fp16(device, fp16)
    use_bf16 = bool(use_amp and device.type == "cuda" and torch.cuda.is_bf16_supported())
    print(f"Using device: {device} | AMP: {use_amp} | BF16: {use_bf16}")

    wb_enabled = bool(get_cfg(config, "wandb.enabled", False))
    wb_project = get_cfg(config, "wandb.project", "elcardiocc-2026")
    wb_entity = get_cfg(config, "wandb.entity", None)
    wb_run_name = get_cfg(config, "wandb.run_name", None)
    wb_run_id_style = get_cfg(config, "wandb.run_id_style", "random")
    wb_run_counter_path = get_cfg(
        config,
        "wandb.run_counter_path",
        "outputs/experiments/xlm_r_large/wandb_run_counter.txt",
    )
    wb_anonymous = get_cfg(config, "wandb.anonymous", None)
    wb_notes = get_cfg(config, "wandb.notes", "")
    wb_tags = get_cfg(config, "wandb.tags", [])
    wb_disable_upload = bool(get_cfg(config, "wandb.disable_upload", True))
    wb_save_code = bool(get_cfg(config, "wandb.save_code", False))

    if wb_enabled:
        init_kwargs = dict(
            project=wb_project,
            entity=wb_entity,
            name=make_wandb_run_name(
                model_name,
                wb_run_name,
                style=str(wb_run_id_style),
                counter_path=str(wb_run_counter_path),
            ),
            notes=wb_notes,
            tags=wb_tags,
            config={
                "model_name": model_name,
                "learning_rate": lr,
                "batch_size": batch_size,
                "eval_batch_size": eval_batch_size,
                "effective_batch_size": batch_size * grad_accum,
                "max_length": max_length,
                "epochs": epochs,
                "loss": loss_type,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "eval_threshold": eval_threshold,
                "sliding_window": sliding_window,
                "stride": stride if sliding_window else None,
                "seed": seed,
                "fp16": fp16,
                "bf16": use_bf16,
                "freeze_layers": freeze_layers,
                "classifier_dropout": classifier_dropout,
                "train_path": train_path,
                "val_path": val_path,
            },
            save_code=wb_save_code,
        )
        if wb_anonymous is not None:
            init_kwargs["anonymous"] = wb_anonymous
        if wb_disable_upload:
            init_kwargs["settings"] = _no_upload_settings()
        if wandb_init_kwargs:
            init_kwargs.update(wandb_init_kwargs)
        wandb.init(**init_kwargs)

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
        chunk_strategy=chunk_strategy,
        truncation_side=truncation_side,
    )
    val_dataset = ELCardioDataset(
        val_path,
        labelset_path,
        tokenizer,
        max_length=max_length,
        sliding_window=sliding_window,
        stride=stride,
        is_training=False,
        chunk_strategy=chunk_strategy,
        truncation_side=truncation_side,
    )
    prefetch_factor = _get_prefetch_factor(num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=prefetch_factor,
    )

    print("Building model...")
    model = build_model(
        num_labels=num_labels,
        model_name=model_name,
        classifier_dropout=classifier_dropout,
    )
    _freeze_bottom_layers(model, freeze_layers)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    model.to(device)
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead", dynamic=False)

    if loss_type == "asl":
        criterion = lambda logits, targets: _asymmetric_loss(
            logits,
            targets,
            gamma_neg=asl_gamma_neg,
            gamma_pos=asl_gamma_pos,
            clip=asl_clip,
        )
    elif loss_type == "bce_pos_weight":
        train_records = load_jsonl(train_path)
        pos_weight = _compute_pos_weight(
            train_records,
            train_dataset.labels,
            cap=pos_weight_cap,
        ).to(device)
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion = lambda logits, targets: bce(logits, targets)
    elif loss_type == "zlpr":
        criterion = lambda logits, targets: _zlpr_loss(logits, targets)
    else:
        raise ValueError(
            f"Unsupported loss '{loss_type}'. Use 'asl', 'bce_pos_weight', or 'zlpr'."
        )

    optimizer = _build_optimizer(model=model, lr=lr, weight_decay=weight_decay)

    steps_per_epoch = _optimizer_steps_per_epoch(len(train_loader), grad_accum)
    total_scheduler_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_scheduler_steps * warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_with_warmup_lambda(
            current_step=step,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_scheduler_steps,
        ),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not use_bf16))
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    ground_truth_data = load_ground_truth(val_path)
    best_f1 = 0.0
    best_epoch = 0
    best_scores = None
    best_unique_pids = None
    global_step = 0
    epochs_without_improvement = 0

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            autocast_ctx = (
                torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
                loss = loss / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                global_step += 1
                grad_norm = torch.tensor(0.0, device=device)
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if wb_enabled:
                    wandb.log(
                        {
                            "train/loss_step": loss.item() * grad_accum,
                            "train/grad_norm": float(grad_norm.item()),
                            "lr": scheduler.get_last_lr()[0],
                        },
                        step=global_step,
                    )

            total_loss += loss.item() * grad_accum
            progress_bar.set_postfix({"loss": total_loss / (step + 1)})

        model.eval()
        pid_to_logits: dict[int, list[np.ndarray]] = {}
        val_total_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                pids = batch["patient_id"].tolist()

                autocast_ctx = (
                    torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with autocast_ctx:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    val_total_loss += criterion(logits, labels).item()
                    val_steps += 1
                    logits = logits.detach().float().cpu().numpy()

                for i, pid in enumerate(pids):
                    pid_to_logits.setdefault(int(pid), []).append(logits[i])

        unique_pids, aggregated_scores = aggregate_scores_by_patient(
            pid_to_logits,
            strategy=chunk_aggregation,
            temperature=1.0,
            alpha=chunk_aggregation_alpha,
        )

        sweep_max_f1 = 0.0
        sweep_argmax_t = eval_threshold
        threshold_metrics = {}
        for threshold in eval_thresholds:
            pred_data = {}
            preds_bin = aggregated_scores >= threshold
            for i, pid in enumerate(unique_pids):
                pred_idx = np.where(preds_bin[i])[0]
                pred_data[int(pid)] = [train_dataset.labels[idx] for idx in pred_idx]
            metrics = evaluate_data(
                ground_truth_data,
                pred_data,
                label_space=train_dataset.labels,
            )
            threshold_metrics[threshold] = metrics
            if metrics["micro_f1"] > sweep_max_f1:
                sweep_max_f1 = metrics["micro_f1"]
                sweep_argmax_t = threshold

        metrics = threshold_metrics[eval_threshold]
        val_f1 = metrics["micro_f1"]
        print(
            f"Epoch {epoch + 1} - Primary Val Micro-F1: {val_f1:.4f} "
            f"(Sweep Max: {sweep_max_f1:.4f} at thresh={sweep_argmax_t:.2f})"
        )

        if wb_enabled:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss_epoch": total_loss / max(1, len(train_loader)),
                "val/loss_epoch": val_total_loss / max(1, val_steps),
                "val/micro_f1_primary": val_f1,
                "val/precision": metrics["precision"],
                "val/recall": metrics["recall"],
                "val/sweep_max_f1": sweep_max_f1,
                "val/sweep_argmax_t": sweep_argmax_t,
            }
            for threshold, t_metrics in threshold_metrics.items():
                log_dict[f"val/f1_thresh_{threshold}"] = t_metrics["micro_f1"]
            if device.type == "cuda":
                log_dict["gpu/memory_allocated_gb"] = (
                    torch.cuda.memory_allocated(device) / 1e9
                )
                log_dict["gpu/memory_reserved_gb"] = (
                    torch.cuda.memory_reserved(device) / 1e9
                )
            wandb.log(log_dict, step=global_step)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_scores = aggregated_scores.copy()
            best_unique_pids = list(unique_pids)
            print(f"New best F1! Saving model to {checkpoint_dir}")
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            np.save(scores_path, aggregated_scores)
            with open(pids_path, "w", encoding="utf-8") as handle:
                json.dump(unique_pids, handle)
            with open(label_names_path, "w", encoding="utf-8") as handle:
                json.dump(train_dataset.labels, handle)

            if wb_enabled and "per_class" in metrics:
                rows = [
                    [
                        row["code"],
                        row["support"],
                        round(row["f1"], 4),
                        round(row["precision"], 4),
                        round(row["recall"], 4),
                    ]
                    for row in metrics["per_class"]
                ]
                table = wandb.Table(
                    columns=["code", "support", "f1", "precision", "recall"],
                    data=rows,
                )
                wandb.log({"per_class_f1": table}, step=global_step)
        else:
            epochs_without_improvement += 1
            if (
                early_stopping_patience > 0
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(
                    f"Early stopping triggered after {epoch + 1} epochs without improvement."
                )
                break

    tuned_f1 = None
    if auto_threshold_tuning and best_scores is not None and best_unique_pids is not None:
        print("Running automatic per-label threshold tuning...")
        tuned_thresholds, tuned_f1 = tune_thresholds(
            scores=best_scores,
            patient_ids=best_unique_pids,
            ground_truth_data=ground_truth_data,
            label_names=train_dataset.labels,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_step=threshold_step,
        )
        os.makedirs(os.path.dirname(thresholds_path), exist_ok=True)
        with open(thresholds_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "tuned_val_micro_f1_in_sample": float(tuned_f1),
                    "note": "F1 is optimistic; thresholds and F1 were selected on the same val split.",
                    "thresholds": {
                        label: float(th)
                        for label, th in zip(train_dataset.labels, tuned_thresholds)
                    },
                    "sweep": {
                        "min": threshold_min,
                        "max": threshold_max,
                        "step": threshold_step,
                    },
                },
                handle,
                indent=2,
            )
        print(
            f"Threshold tuning complete. In-sample tuned F1={float(tuned_f1):.4f} (optimistic)."
        )

    print(f"Training complete. Best Val F1: {best_f1:.4f}")
    if wb_enabled and wandb.run is not None:
        wandb.run.summary["best_val_micro_f1"] = float(best_f1)
        wandb.run.summary["best_epoch"] = int(best_epoch)
        if tuned_f1 is not None:
            wandb.run.summary["val/tuned_f1_in_sample"] = float(tuned_f1)
        wandb.finish()

    return {
        "best_val_micro_f1": float(best_f1),
        "best_epoch": int(best_epoch),
        "tuned_f1": None if tuned_f1 is None else float(tuned_f1),
        "checkpoint_dir": checkpoint_dir,
        "thresholds_path": thresholds_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XLM-R large model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--device",
        help="Explicit device to use (e.g., 'cpu', 'cuda', 'mps')",
    )
    args = parser.parse_args()

    load_dotenv_if_present()
    config = load_config(args.config)
    run(config=config, device_override=args.device)


if __name__ == "__main__":
    main()
