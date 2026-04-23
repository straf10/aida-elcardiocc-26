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
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import wandb
import yaml

import unicodedata

from mlc_greek_bert.model import MLCModel
from evaluation.evaluator import evaluate_data
from evaluation.io_utils import load_ground_truth
from preprocessing.io_utils import LABELSET_PATH, load_labelset
from split_data.device_utils import use_amp_fp16


def strip_accents_and_lowercase(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    ).lower()


# ── Reproducibility ──────────────────────────────────────────────────────────


def set_seed(seed: int, *, allow_cudnn_benchmark: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if allow_cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ── Dataset ─────────────────────────────────────────────────────────────────


class CardioDataset(Dataset):
    """
    Loads ELCardioCC JSONL data with one-time tokenization and variable-length
    per-example input_ids (padded to batch max in collate).
    Each item: input_ids (list[int]), label_vector (FloatTensor 115,), patient_id (int)
    """

    def __init__(self, jsonl_path: str, label_names: list, tokenizer, max_length: int):
        self.label2idx = {label: i for i, label in enumerate(label_names)}
        self.num_labels = len(label_names)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records: list[dict] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

        texts: list[str] = []
        self._labels: list[torch.Tensor] = []
        self._patient_ids: list[int] = []

        for record in self.records:
            text = strip_accents_and_lowercase(record["text"])
            texts.append(text)
            pid = int(record["patient_id"])
            self._patient_ids.append(pid)

            label_vector = torch.zeros(self.num_labels, dtype=torch.float32)
            for group in record.get("document_level_annotations", []):
                for code in group:
                    if code in self.label2idx:
                        label_vector[self.label2idx[code]] = 1.0
            self._labels.append(label_vector)

        enc = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_token_type_ids=False,
        )
        self._input_ids: list[list[int]] = enc["input_ids"]

    def __len__(self) -> int:
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> dict[str, Union[list[int], torch.Tensor, int]]:
        return {
            "input_ids": self._input_ids[idx],
            "labels": self._labels[idx],
            "patient_id": self._patient_ids[idx],
        }


def build_collate_fn(pad_token_id: int):
    """Dynamic padding to the longest sequence in the batch."""

    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for x in batch:
            ids: list[int] = x["input_ids"]
            l = len(ids)
            pad_len = max_len - l
            input_ids.append(ids + [pad_token_id] * pad_len)
            attention_mask.append([1] * l + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.stack([x["labels"] for x in batch], dim=0).float(),
            "patient_id": torch.tensor([x["patient_id"] for x in batch], dtype=torch.long),
        }

    return collate


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

    pos_weight = np.where(
        counts > 0, (total - counts) / np.maximum(counts, 1), float(total)
    )
    return torch.tensor(pos_weight, dtype=torch.float32)


def _asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip: float = 0.05,
) -> torch.Tensor:
    """Asymmetric loss for multi-label (ported from xlm_r_large/train.py)."""
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


def build_criterion(
    config: dict, label_names: list, train_path: str, device: torch.device
) -> Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    loss_type = str(config["training"].get("loss", "bce_pos_weight")).lower()
    if loss_type == "asl":
        g_neg = float(config["training"].get("asl_gamma_neg", 4.0))
        g_pos = float(config["training"].get("asl_gamma_pos", 1.0))
        clip = float(config["training"].get("asl_clip", 0.05))

        def _criterion(logits, targets):
            return _asymmetric_loss(
                logits, targets, gamma_neg=g_neg, gamma_pos=g_pos, clip=clip
            )

        print("Using Asymmetric Loss (ASL)")
        return _criterion

    if config["training"].get("use_class_weights", True):
        cap = float(config["training"].get("pos_weight_cap", 20.0))
        pw = compute_pos_weights(train_path, label_names)
        pw = torch.clamp(pw, min=1.0, max=cap).to(device=device, dtype=torch.float32)
        print(f"Using class-weighted BCEWithLogitsLoss (pos_weight_cap={cap})")
        return nn.BCEWithLogitsLoss(pos_weight=pw).to(device)
    return nn.BCEWithLogitsLoss().to(device)


def build_optimizer(
    model: nn.Module, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    no_decay_substrings = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in name for nd in no_decay_substrings):
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


# ── Validation ────────────────────────────────────────────────────────────────


def _eval_loss(
    criterion: Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Loss on float32 logits (stable for BCE with pos_weight and ASL)."""
    logits = logits.float()
    if isinstance(criterion, nn.Module):
        return criterion(logits, labels)
    return criterion(logits, labels)


def validate(
    model: nn.Module,
    loader: DataLoader,
    label_names: List[str],
    device: torch.device,
    criterion: Union[nn.Module, Callable],
    ground_truth_data: Dict[int, List[List[str]]],
    eval_thresholds: List[float],
    eval_threshold_primary: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[
    float,
    float,
    float,
    np.ndarray,
    List,
    float,
    Dict[float, Dict[str, float]],
    float,
    float,
]:
    """
    Validation with threshold sweep. Reported val F1 uses eval_threshold_primary.
    Also returns sweep_max_f1 and sweep_argmax_t (argmax over eval_thresholds).
    """
    model.eval()
    all_scores: List[np.ndarray] = []
    all_pids: List = []
    total_loss = 0.0
    n_steps = 0
    nb = device.type == "cuda"

    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(
                device, non_blocking=nb, memory_format=torch.contiguous_format
            )
            attention_mask = batch["attention_mask"].to(
                device, non_blocking=nb, memory_format=torch.contiguous_format
            )
            labels = batch["labels"].to(device, non_blocking=nb, dtype=torch.float32)
            pids = batch["patient_id"]

            autocast_ctx = (
                torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast_ctx:
                logits = model(input_ids, attention_mask)
            total_loss += float(_eval_loss(criterion, logits, labels).item())
            n_steps += 1
            scores = torch.sigmoid(logits.float()).cpu().numpy()
            all_scores.append(scores)
            all_pids.extend(pids.tolist())

    all_scores_arr = np.vstack(all_scores)
    avg_loss = total_loss / max(1, n_steps)

    threshold_metrics: Dict[float, Dict[str, float]] = {}
    for t in eval_thresholds:
        pred_data: Dict[int, List[str]] = {
            int(pid): [label_names[j] for j, s in enumerate(all_scores_arr[i]) if s >= t]
            for i, pid in enumerate(all_pids)
        }
        m = evaluate_data(ground_truth_data, pred_data, label_space=label_names)
        threshold_metrics[float(t)] = {
            "micro_f1": float(m["micro_f1"]),
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
        }

    t_primary = float(eval_threshold_primary)
    if t_primary not in threshold_metrics and eval_thresholds:
        closest = min(eval_thresholds, key=lambda x: abs(float(x) - t_primary))
        t_key = float(closest)
    else:
        t_key = t_primary

    primary = threshold_metrics.get(t_key, next(iter(threshold_metrics.values())))

    sweep_max_f1 = 0.0
    sweep_argmax_t = t_primary
    for t, m in threshold_metrics.items():
        if m["micro_f1"] > sweep_max_f1:
            sweep_max_f1 = m["micro_f1"]
            sweep_argmax_t = t

    return (
        float(primary["micro_f1"]),
        float(primary["precision"]),
        float(primary["recall"]),
        all_scores_arr,
        all_pids,
        float(avg_loss),
        threshold_metrics,
        float(sweep_max_f1),
        float(sweep_argmax_t),
    )


# ── Main training loop ────────────────────────────────────────────────────────


def _make_wandb_threshold_key(t: float) -> str:
    s = f"{t:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "_")


def train(config: dict) -> float:
    set_seed(int(config["training"]["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = bool(config["training"].get("fp16", True))
    use_amp = use_amp_fp16(device, fp16)
    use_bf16 = bool(
        use_amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not use_bf16))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    pin_memory = bool(config["training"].get("pin_memory", True))
    nb = device.type == "cuda" and pin_memory
    print(
        f"Using device: {device} | AMP: {use_amp} | "
        f"BF16: {use_bf16} | pin_memory: {pin_memory}"
    )
    save_last_only = bool(config["training"].get("save_last_only", False))

    g = torch.Generator()
    g.manual_seed(int(config["training"]["seed"]))

    label_names = load_labelset(LABELSET_PATH)
    print(f"Loaded {len(label_names)} labels from {LABELSET_PATH}")
    assert len(label_names) == config["model"]["num_labels"], (
        f"Expected {config['model']['num_labels']} labels, got {len(label_names)}"
    )

    num_workers = int(config["training"].get("num_workers", 0))
    prefetch = 4 if num_workers > 0 else None
    persistent = num_workers > 0

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0
    collate = build_collate_fn(pad_id)

    train_dataset = CardioDataset(
        config["data"]["train_path"],
        label_names,
        tokenizer,
        int(config["model"]["max_length"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        collate_fn=collate,
    )

    val_loader = None
    val_ground_truth = None
    if not save_last_only:
        val_path = config["data"].get("val_path")
        if not val_path:
            raise ValueError("data.val_path is required unless training.save_last_only=true")
        val_dataset = CardioDataset(
            str(val_path), label_names, tokenizer, int(config["model"]["max_length"])
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config["training"]["batch_size"]) * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
            collate_fn=collate,
        )
        val_ground_truth = load_ground_truth(str(val_path))

    train_ground_truth = load_ground_truth(str(config["data"]["train_path"]))

    model = MLCModel(
        model_name=str(config["model"]["name"]),
        num_labels=int(config["model"]["num_labels"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)

    criterion = build_criterion(
        config, label_names, str(config["data"]["train_path"]), device
    )
    optimizer = build_optimizer(
        model,
        float(config["training"]["learning_rate"]),
        float(config["training"]["weight_decay"]),
    )

    grad_accum = int(config["training"]["grad_accum_steps"])
    max_epochs = int(config["training"]["epochs"])
    total_steps = math.ceil(len(train_loader) / max(1, grad_accum)) * max_epochs
    warmup_steps = int(total_steps * float(config["training"]["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, max(1, total_steps)
    )

    eval_threshold = float(config["training"].get("eval_threshold", 0.5))
    eval_thresholds = [
        float(t) for t in config["training"].get("eval_thresholds", [eval_threshold])
    ]
    if eval_threshold not in eval_thresholds:
        eval_thresholds = sorted(set(eval_thresholds + [eval_threshold]))
    es_patience = int(config["training"].get("early_stopping_patience", 5))

    loss_name = str(config["training"].get("loss", "bce_pos_weight"))
    run_cfg: Dict[str, Any] = {
        "model": str(config["model"]["name"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "batch_size": int(config["training"]["batch_size"]),
        "max_length": int(config["model"]["max_length"]),
        "epochs": max_epochs,
        "use_class_weights": bool(config["training"].get("use_class_weights", True)),
        "pos_weight_cap": float(config["training"].get("pos_weight_cap", 20.0)),
        "loss": loss_name,
        "grad_accum_steps": grad_accum,
        "dropout": float(config["model"]["dropout"]),
        "warmup_ratio": float(config["training"]["warmup_ratio"]),
        "weight_decay": float(config["training"]["weight_decay"]),
        "fp16": fp16,
        "bf16": use_bf16,
        "eval_threshold": eval_threshold,
        "eval_thresholds": eval_thresholds,
        "early_stopping_patience": es_patience,
    }
    if wandb.run is None:
        wandb.init(
            project=str(config["wandb"]["project"]),
            name=str(config["wandb"]["run_name"]),
            tags=list(config["wandb"].get("tags", [])),
            config=run_cfg,
        )
    else:
        wandb.config.update(
            {k: v for k, v in run_cfg.items() if k not in ("model", "epochs")}
        )

    ckpt_dir = Path(str(config["output"]["checkpoint_dir"]))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    labels_path = Path(str(config["output"]["labels_path"]))
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(label_names, f)

    best_f1 = 0.0
    best_epoch = 0
    global_step = 0
    epochs_wo = 0
    max_gn = float(config["training"]["max_grad_norm"])

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_tr_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        train_pred_buffer: Dict[int, List[str]] = {}

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(
                device, non_blocking=nb, memory_format=torch.contiguous_format
            )
            attention_mask = batch["attention_mask"].to(
                device, non_blocking=nb, memory_format=torch.contiguous_format
            )
            labels = batch["labels"].to(device, non_blocking=nb, dtype=torch.float32)
            pids = batch["patient_id"].tolist()

            autocast_ctx = (
                torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast_ctx:
                logits = model(input_ids, attention_mask)
                loss = _eval_loss(criterion, logits, labels) / max(1, grad_accum)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_tr_loss += float((loss * grad_accum).item())

            with torch.inference_mode():
                scores = torch.sigmoid(logits.float()).cpu().numpy()
                preds = scores >= eval_threshold
                for i in range(len(pids)):
                    pid = int(pids[i])
                    train_pred_buffer[pid] = [
                        label_names[j] for j in range(len(label_names)) if preds[i, j]
                    ]

            should_step = (step + 1) % grad_accum == 0 or (step + 1) == len(
                train_loader
            )
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_gn
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_gn
                    )
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        avg_tr_loss = total_tr_loss / max(1, len(train_loader))
        train_m = evaluate_data(
            train_ground_truth, train_pred_buffer, label_space=label_names
        )
        train_f1 = float(train_m["micro_f1"])

        if save_last_only:
            best_f1 = train_f1
            best_epoch = epoch
            print(
                f"Epoch {epoch}/{max_epochs} | "
                f"Train Loss: {avg_tr_loss:.4f} | Train F1: {train_f1:.4f}"
            )
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": avg_tr_loss,
                    "train_micro_f1": train_f1,
                    "lr": scheduler.get_last_lr()[0],
                }
            )
            continue

        assert val_loader is not None and val_ground_truth is not None
        (
            val_f1,
            val_p,
            val_r,
            val_scores,
            val_pids,
            val_loss,
            th_metrics,
            sweep_max,
            sweep_arg,
        ) = validate(
            model,
            val_loader,
            label_names,
            device,
            criterion,
            val_ground_truth,
            eval_thresholds,
            eval_threshold,
            use_amp,
            amp_dtype,
        )
        log_dict: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": avg_tr_loss,
            "val_loss": val_loss,
            "train_micro_f1": train_f1,
            "val_micro_f1": val_f1,
            "val_precision": val_p,
            "val_recall": val_r,
            "lr": scheduler.get_last_lr()[0],
            "val/sweep_max_f1": sweep_max,
            "val/sweep_argmax_t": sweep_arg,
        }
        for t, m in th_metrics.items():
            log_dict[f"val/f1_thresh_{_make_wandb_threshold_key(t)}"] = m["micro_f1"]
        print(
            f"Epoch {epoch}/{max_epochs} | "
            f"Train Loss: {avg_tr_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} (primary t={eval_threshold}) | "
            f"Val sweep max: {sweep_max:.4f} @ t={sweep_arg} | P: {val_p:.4f} R: {val_r:.4f}"
        )
        wandb.log(log_dict)

        if val_f1 > best_f1 + 1e-9:
            best_f1 = val_f1
            best_epoch = epoch
            epochs_wo = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            np.save(str(config["output"]["scores_path"]), val_scores)
            with open(
                str(config["output"]["pids_path"]), "w", encoding="utf-8"
            ) as handle:
                json.dump(val_pids, handle)
            print(f"  New best model saved (F1={best_f1:.4f} @ t={eval_threshold})")
        else:
            epochs_wo += 1
            if es_patience > 0 and epochs_wo >= es_patience:
                print(f"Early stopping (patience={es_patience}) at epoch {epoch}.")
                break

    if save_last_only:
        final_ckpt = ckpt_dir / "final_model.pt"
        torch.save(model.state_dict(), final_ckpt)
        print(
            f"\nTraining complete. Final model saved to {final_ckpt} (epoch={best_epoch})"
        )
        wandb.summary["final_train_micro_f1"] = float(best_f1)
    else:
        print(
            f"\nTraining complete. Best val F1: {best_f1:.4f} at epoch {best_epoch}"
        )
        wandb.summary["best_val_micro_f1"] = float(best_f1)

    is_sweep = wandb.run is not None and wandb.run.sweep_id is not None
    if not is_sweep and config["data"].get("test_path"):
        name = "final_model.pt" if save_last_only else "best_model.pt"
        print(f"\nEvaluating {name} on held-out test set...")
        ckpt_p = ckpt_dir / name
        try:
            state = torch.load(ckpt_p, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt_p, map_location=device)
        model.load_state_dict(state)
        test_gt = load_ground_truth(str(config["data"]["test_path"]))
        test_ds = CardioDataset(
            str(config["data"]["test_path"]),
            label_names,
            tokenizer,
            int(config["model"]["max_length"]),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=int(config["training"]["batch_size"]) * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
            collate_fn=collate,
        )
        t_f1, t_p, t_r, _, _, t_loss, _, _, _ = validate(
            model,
            test_loader,
            label_names,
            device,
            criterion,
            test_gt,
            [eval_threshold],
            eval_threshold,
            use_amp,
            amp_dtype,
        )
        print(
            f"Test set — F1: {t_f1:.4f} | P: {t_p:.4f} | R: {t_r:.4f} | Loss: {t_loss:.4f}"
        )
        wandb.summary["test_micro_f1"] = t_f1
        wandb.summary["test_precision"] = t_p
        wandb.summary["test_recall"] = t_r

    wandb.finish()
    return float(best_f1)


# ── Entry point ───────────────────────────────────────────────────────────────


def train_for_sweep(config: dict) -> None:
    with wandb.init():
        sc = wandb.config
        t = config["training"]
        m = config["model"]
        if "learning_rate" in sc:
            t["learning_rate"] = sc["learning_rate"]
        if "dropout" in sc:
            m["dropout"] = sc["dropout"]
        if "batch_size" in sc:
            t["batch_size"] = sc["batch_size"]
        if "grad_accum_steps" in sc:
            t["grad_accum_steps"] = sc["grad_accum_steps"]
        if "warmup_ratio" in sc:
            t["warmup_ratio"] = sc["warmup_ratio"]
        if "weight_decay" in sc:
            t["weight_decay"] = sc["weight_decay"]
        if "epochs" in sc:
            t["epochs"] = sc["epochs"]
        if "loss" in sc:
            t["loss"] = sc["loss"]
        lr = t["learning_rate"]
        bs = t["batch_size"]
        do = m["dropout"]
        ls = t.get("loss", "bce_pos_weight")
        config["wandb"]["run_name"] = f"sweep-lr{lr:.0e}-bs{bs}-do{do}-{ls}"
        train(config)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    ap.add_argument("--sweep", action="store_true", help="Run as W&B sweep agent")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.sweep:
        train_for_sweep(cfg)
    else:
        train(cfg)
