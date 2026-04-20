import argparse
import hashlib
import json
import os
import random
import uuid
from pathlib import Path

import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from src.preprocessing.augmentation import (
        build_augmented_dataset,
        load_synonym_dict,
    )
    from src.preprocessing.io_utils import (
        load_jsonl,
        load_labelset,
        resolve_patient_id,
        save_jsonl,
    )
    from src.preprocessing.dataset import ELCardioDataset
    from src.evaluation.config_utils import get_cfg, load_config
    from src.evaluation.evaluator import evaluate_data
    from src.evaluation.io_utils import load_ground_truth
    from src.evaluation.threshold_tune import tune_thresholds
    from src.training_validation.device_utils import get_device, use_amp_fp16
    from src.training_validation.dotenv_util import load_dotenv_if_present
except ImportError:
    from ..preprocessing.augmentation import (
        build_augmented_dataset,
        load_synonym_dict,
    )
    from ..preprocessing.io_utils import (
        load_jsonl,
        load_labelset,
        resolve_patient_id,
        save_jsonl,
    )
    from ..preprocessing.dataset import ELCardioDataset
    from ..evaluation.config_utils import get_cfg, load_config
    from ..evaluation.evaluator import evaluate_data
    from ..evaluation.io_utils import load_ground_truth
    from ..evaluation.threshold_tune import tune_thresholds
    from ..training_validation.device_utils import get_device, use_amp_fp16
    from ..training_validation.dotenv_util import load_dotenv_if_present

from .chunk_aggregate import aggregate_scores_by_patient
from .model import (
    DescResidualWrapper,
    build_model,
    load_label_descriptions_from_csv,
    rebake_description_embeddings,
)


def _wandb_run_name(explicit: str | None, model_name: str) -> str:
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    safe = model_name.replace("/", "-").replace(".", "-")[:100]
    return f"{safe}-{uuid.uuid4().hex[:8]}"


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


def _cosine_with_warmup_floor_lambda(
    current_step: int,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float,
) -> float:
    min_lr_ratio = min(max(float(min_lr_ratio), 0.0), 1.0)
    if num_training_steps <= 0:
        return 1.0
    if current_step < num_warmup_steps:
        return max(min_lr_ratio, float(current_step) / max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / max(
        1, num_training_steps - num_warmup_steps
    )
    cosine = 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


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


def _asl_scalar(
    logit: torch.Tensor,
    target: torch.Tensor,
    gamma_neg: float,
    gamma_pos: float,
    clip: float,
) -> torch.Tensor:
    """Single-logit ASL (same formula as _asymmetric_loss one dimension)."""
    probs_pos = torch.sigmoid(logit)
    probs_neg = 1.0 - probs_pos
    if clip > 0:
        probs_neg = (probs_neg + clip).clamp(max=1.0)
    loss_pos = target * torch.log(probs_pos.clamp(min=1e-8))
    loss_neg = (1.0 - target) * torch.log(probs_neg.clamp(min=1e-8))
    loss = -loss_pos * (1.0 - probs_pos).pow(gamma_pos) - loss_neg * probs_pos.pow(
        gamma_neg
    )
    return loss


def _group_wise_asl_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    groups_batch: list,
    gamma_neg: float,
    gamma_pos: float,
    clip: float,
    group_temp: float,
) -> torch.Tensor:
    """
    OR-pool logits within each gold group; ASL on group logits with target 1.
    Per-label ASL on labels not in any gold group (targets from tensor).
    """
    device = logits.device
    B, C = logits.shape
    losses = []
    for b in range(B):
        groups = groups_batch[b]
        lb = logits[b]
        tb = targets[b]
        if not groups:
            losses.append(
                _asymmetric_loss(
                    lb.unsqueeze(0),
                    tb.unsqueeze(0),
                    gamma_neg=gamma_neg,
                    gamma_pos=gamma_pos,
                    clip=clip,
                )
            )
            continue
        in_union = set()
        for g in groups:
            in_union.update(g)
        terms = []
        for g in groups:
            if len(g) == 0:
                continue
            idx = torch.tensor(g, device=device, dtype=torch.long)
            sub = lb[idx]
            z = group_temp * torch.logsumexp(sub / group_temp, dim=0)
            terms.append(_asl_scalar(z, torch.ones((), device=device), gamma_neg, gamma_pos, clip))
        neg_idx = [j for j in range(C) if j not in in_union]
        for j in neg_idx:
            terms.append(
                _asl_scalar(lb[j], tb[j], gamma_neg, gamma_pos, clip)
            )
        if not terms:
            losses.append(torch.tensor(0.0, device=device))
        else:
            losses.append(torch.stack(terms).mean())
    return torch.stack(losses).mean()


def collate_fn_group_asl(batch: list) -> dict:
    """Stack tensors; keep per-example gold groups as Python lists."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    groups = [b["groups"] for b in batch]
    patient_id = [b["patient_id"] for b in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "groups": groups,
        "patient_id": patient_id,
    }


def _model_config(model: torch.nn.Module):
    """Config for LLRD / layer count (unwrap DescResidualWrapper)."""
    if isinstance(model, DescResidualWrapper):
        return model.inner.config
    return model.config


def _freeze_bottom_layers(model, freeze_layers: int):
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


def _file_sha8(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:8]


def _normalize_multipliers(multipliers_cfg) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for raw_k, raw_v in (multipliers_cfg or {}).items():
        key = str(raw_k).replace("<", "").strip()
        if not key.isdigit():
            continue
        val = int(raw_v)
        if val < 0:
            continue
        normalized[str(int(key))] = val
    return dict(sorted(normalized.items(), key=lambda x: int(x[0])))


def _build_augmented_train_file(
    train_path: str,
    val_path: str,
    labelset_path: str,
    synonym_csv: str,
    min_freq: int,
    multipliers: dict,
    swap_prob: float,
    seed: int,
    cache_dir: str = "data/processed/_augmented",
) -> tuple[str, dict]:
    syn_name = Path(synonym_csv).name.lower()
    if "train_only" not in syn_name:
        raise ValueError(
            f"Synonym CSV must be train-only to avoid leakage. Got: {synonym_csv}"
        )

    train_sha8 = _file_sha8(train_path)
    val_sha8 = _file_sha8(val_path)
    syn_sha8 = _file_sha8(synonym_csv)
    multipliers_norm = _normalize_multipliers(multipliers)
    multipliers_sha8 = hashlib.sha256(
        json.dumps(multipliers_norm, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]

    cache_name = (
        f"training_set.aug_{train_sha8}_{syn_sha8}_{multipliers_sha8}_"
        f"min{int(min_freq)}_p{float(swap_prob):.2f}_seed{int(seed)}.jsonl"
    )
    cache_path = Path(cache_dir) / cache_name
    if cache_path.exists():
        return str(cache_path), {
            "train_sha8": train_sha8,
            "val_sha8": val_sha8,
            "synonym_csv_sha8": syn_sha8,
            "multipliers_sha8": multipliers_sha8,
            "cached": True,
        }

    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)
    labelset = load_labelset(labelset_path)
    code_to_terms = load_synonym_dict(synonym_csv)
    max_real_pid = max(
        resolve_patient_id(rec) for rec in train_records + val_records
    )
    random.seed(seed)
    augmented_records = build_augmented_dataset(
        records=train_records,
        labelset=labelset,
        code_to_terms=code_to_terms,
        min_freq=int(min_freq),
        multipliers=multipliers_norm,
        swap_prob=float(swap_prob),
        max_real_pid=max_real_pid,
    )
    save_jsonl(augmented_records, str(cache_path))

    return str(cache_path), {
        "train_sha8": train_sha8,
        "val_sha8": val_sha8,
        "synonym_csv_sha8": syn_sha8,
        "multipliers_sha8": multipliers_sha8,
        "cached": False,
    }


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
    max_length = get_cfg(config, "data.max_length", 512)
    sliding_window = get_cfg(config, "data.sliding_window", False)
    stride = get_cfg(config, "data.stride", 256)
    chunk_strategy = get_cfg(config, "data.chunk_strategy", "random")
    truncation_side = get_cfg(config, "data.truncation_side", "right")

    epochs = get_cfg(config, "training.epochs", 10)
    batch_size = get_cfg(config, "training.batch_size", 8)
    grad_accum = get_cfg(config, "training.gradient_accumulation_steps", 4)
    lr = get_cfg(config, "training.learning_rate", 1e-5)
    weight_decay = get_cfg(config, "training.weight_decay", 0.01)
    warmup_ratio = get_cfg(config, "training.warmup_ratio", 0.0)
    scheduler_type = get_cfg(config, "training.scheduler", "cosine")
    eta_min_ratio = get_cfg(config, "training.eta_min_ratio", 0.0)
    eval_threshold = get_cfg(config, "training.eval_threshold", 0.15)
    primary_eval_threshold = float(get_cfg(config, "training.primary_eval_threshold", eval_threshold))
    eval_thresholds = get_cfg(
        config,
        "training.eval_thresholds",
        [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    )
    eval_thresholds = [float(t) for t in eval_thresholds]
    if primary_eval_threshold not in eval_thresholds:
        eval_thresholds.append(primary_eval_threshold)
    fp16 = get_cfg(config, "training.fp16", True)
    seed = get_cfg(config, "training.seed", 42)
    loss_type = get_cfg(config, "training.loss", "asl")
    asl_gamma_neg = get_cfg(config, "training.asl_gamma_neg", 4.0)
    asl_gamma_pos = get_cfg(config, "training.asl_gamma_pos", 1.0)
    asl_clip = get_cfg(config, "training.asl_clip", 0.05)
    group_asl_temperature = float(get_cfg(config, "training.group_asl_temperature", 1.0))
    max_grad_norm = get_cfg(config, "training.max_grad_norm", 1.0)
    early_stopping_patience = get_cfg(config, "training.early_stopping_patience", 3)
    freeze_layers = get_cfg(config, "training.freeze_layers", 0)
    classifier_dropout = get_cfg(config, "training.classifier_dropout", 0.3)
    pooling_strategy = get_cfg(config, "training.pooling_strategy", "cls")
    multi_sample_dropout_samples = get_cfg(
        config, "training.multi_sample_dropout_samples", 1
    )
    aggregation_strategy = get_cfg(config, "training.aggregation_strategy", "max")
    aggregation_temperature = get_cfg(config, "training.aggregation_temperature", 1.0)
    swa_start_epoch = int(get_cfg(config, "training.swa_start_epoch", 0))
    swa_save_if_best = bool(get_cfg(config, "training.swa_save_if_best", True))
    rebake_desc_after_swa = bool(get_cfg(config, "training.rebake_desc_after_swa", False))
    auto_threshold_tuning = bool(get_cfg(config, "training.auto_threshold_tuning", False))
    threshold_min = float(get_cfg(config, "threshold_tuning.min", 0.05))
    threshold_max = float(get_cfg(config, "threshold_tuning.max", 0.95))
    threshold_step = float(get_cfg(config, "threshold_tuning.step", 0.01))

    checkpoint_dir = get_cfg(config, "output.checkpoint_dir", "outputs/experiments/xlm_r_large/checkpoints")
    scores_path = get_cfg(config, "output.scores_path", "outputs/experiments/xlm_r_large/val_scores.npy")
    pids_path = get_cfg(config, "output.patient_ids_path", "outputs/experiments/xlm_r_large/val_patient_ids.json")
    label_names_path = get_cfg(config, "output.label_names_path", "outputs/experiments/xlm_r_large/label_names.json")
    thresholds_path = get_cfg(config, "output.thresholds_path", "outputs/experiments/xlm_r_large/thresholds.json")
    log_dir = get_cfg(config, "output.log_dir", None)

    label_description_csv = get_cfg(config, "data.label_description_csv", None)
    init_classifier_from_descriptions = bool(
        get_cfg(config, "training.init_classifier_from_descriptions", False)
    )
    desc_init_scale = float(get_cfg(config, "training.desc_init_scale", 0.02))
    use_desc_residual = bool(get_cfg(config, "training.use_desc_residual", False))
    desc_residual_alpha_init = float(
        get_cfg(config, "training.desc_residual_alpha_init", 0.1)
    )

    label_descriptions = None
    if label_description_csv and (init_classifier_from_descriptions or use_desc_residual):
        label_descriptions = load_label_descriptions_from_csv(
            label_description_csv, load_labelset(labelset_path)
        )

    return_groups = loss_type == "group_asl"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(scores_path), exist_ok=True)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    set_seed(seed)

    augment_enabled = bool(get_cfg(config, "data.augment_rare_codes.enabled", False))
    data_integrity = {
        "train_path": train_path,
        "val_path": val_path,
        "train_sha8": _file_sha8(train_path),
        "val_sha8": _file_sha8(val_path),
        "augmented": False,
    }
    if augment_enabled:
        synonym_csv = get_cfg(
            config,
            "data.augment_rare_codes.synonym_csv",
            "data/external/full_dictionary.train_only.csv",
        )
        min_freq = int(get_cfg(config, "data.augment_rare_codes.min_freq", 30))
        multipliers_cfg = get_cfg(config, "data.augment_rare_codes.multipliers", {})
        swap_prob = float(get_cfg(config, "data.augment_rare_codes.swap_prob", 0.35))
        augmented_train_path, aug_meta = _build_augmented_train_file(
            train_path=train_path,
            val_path=val_path,
            labelset_path=labelset_path,
            synonym_csv=synonym_csv,
            min_freq=min_freq,
            multipliers=multipliers_cfg,
            swap_prob=swap_prob,
            seed=seed,
        )
        if os.path.abspath(augmented_train_path) == os.path.abspath(val_path):
            raise ValueError("Augmented train path must not equal validation path.")
        train_path = augmented_train_path
        data_integrity = {
            "train_path": train_path,
            "val_path": val_path,
            "train_sha8": aug_meta["train_sha8"],
            "val_sha8": aug_meta["val_sha8"],
            "augmented": True,
            "synonym_csv": synonym_csv,
            "synonym_csv_sha8": aug_meta["synonym_csv_sha8"],
            "multipliers_sha8": aug_meta["multipliers_sha8"],
            "augment_cache_hit": aug_meta["cached"],
        }
        print(f"Augmentation enabled. Using cached/created train file: {train_path}")
        if "train_only" not in Path(synonym_csv).name.lower():
            raise ValueError(
                f"augment_rare_codes requires a train-only synonym CSV. Got: {synonym_csv}"
            )

    device = get_device(args.device)
    use_amp = use_amp_fp16(device, fp16)
    print(f"Using device: {device} | AMP (fp16): {use_amp}")

    wb_enabled  = get_cfg(config, "wandb.enabled", False)
    wb_log_model_artifact = get_cfg(config, "wandb.log_model_artifact", False)
    wb_project  = get_cfg(config, "wandb.project", "elcardiocc-2026")
    wb_entity   = get_cfg(config, "wandb.entity", None)
    wb_run_name = get_cfg(config, "wandb.run_name", None)
    wb_anonymous = get_cfg(config, "wandb.anonymous", None)
    wb_notes    = get_cfg(config, "wandb.notes", "")
    wb_tags     = get_cfg(config, "wandb.tags", [])

    if wb_enabled:
        init_kwargs = dict(
            project=wb_project,
            entity=wb_entity,
            name=_wandb_run_name(wb_run_name, model_name),
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
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "scheduler": scheduler_type,
                "eta_min_ratio": eta_min_ratio,
                "eval_threshold": eval_threshold,
                "sliding_window": sliding_window,
                "stride": stride if sliding_window else None,
                "seed": seed,
                "fp16": fp16,
                "device": str(device),
                "freeze_layers": freeze_layers,
                "classifier_dropout": classifier_dropout,
                "pooling_strategy": pooling_strategy,
                "multi_sample_dropout_samples": multi_sample_dropout_samples,
                "aggregation_strategy": aggregation_strategy,
                "swa_start_epoch": swa_start_epoch,
                "train_path": train_path,
                "val_path": val_path,
                "train_sha8": data_integrity["train_sha8"],
                "val_sha8": data_integrity["val_sha8"],
            },
        )
        if wb_anonymous is not None:
            init_kwargs["anonymous"] = wb_anonymous
        wandb.init(**init_kwargs)
        wandb.run.summary["data_integrity"] = data_integrity

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
        return_groups=return_groups,
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
        return_groups=False,
    )

    train_collate = collate_fn_group_asl if return_groups else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Building model...")
    model = build_model(
        num_labels=num_labels,
        model_name=model_name,
        classifier_dropout=classifier_dropout,
        pooling_strategy=pooling_strategy,
        multi_sample_dropout_samples=multi_sample_dropout_samples,
        init_classifier_from_descriptions=init_classifier_from_descriptions,
        label_descriptions=label_descriptions,
        desc_init_scale=desc_init_scale,
        use_desc_residual=use_desc_residual,
        desc_residual_alpha_init=desc_residual_alpha_init,
    )
    _freeze_bottom_layers(model, freeze_layers)
    model.to(device)

    if loss_type == "asl":

        def asl_loss(logits, targets):
            return _asymmetric_loss(
                logits,
                targets,
                gamma_neg=asl_gamma_neg,
                gamma_pos=asl_gamma_pos,
                clip=asl_clip,
            )

        criterion = asl_loss
    elif loss_type == "group_asl":

        def group_asl_loss(logits, targets, groups_batch):
            return _group_wise_asl_loss(
                logits,
                targets,
                groups_batch,
                gamma_neg=asl_gamma_neg,
                gamma_pos=asl_gamma_pos,
                clip=asl_clip,
                group_temp=group_asl_temperature,
            )

        criterion = group_asl_loss
    else:
        raise ValueError(f"Unsupported loss '{loss_type}'. Use 'asl' or 'group_asl'.")

    # Apply Layer-wise Learning Rate Decay (LLRD)
    head_mult = get_cfg(config, "training.llrd_head_multiplier", 2.0)
    embedding_mult = get_cfg(config, "training.llrd_embedding_multiplier", 0.5)
    middle_mult = get_cfg(config, "training.llrd_middle_multiplier", 0.8)
    
    head_lr = lr * head_mult
    embedding_lr = lr * embedding_mult
    middle_lr = lr * middle_mult
    top_lr = lr * 1.0

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    
    _cfg = _model_config(model)
    if hasattr(_cfg, "num_hidden_layers"):
        num_layers = _cfg.num_hidden_layers
    else:
        num_layers = 24  # default fallback

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        weight_decay_val = 0.0 if any(nd in n for nd in no_decay) else weight_decay
        
        if "classifier" in n:
            param_lr = head_lr
        elif (
            "roberta.embeddings" in n
            or "xlm_roberta.embeddings" in n
            or "backbone.embeddings" in n
        ):
            param_lr = embedding_lr
        elif (
            "roberta.encoder.layer" in n
            or "xlm_roberta.encoder.layer" in n
            or "backbone.encoder.layer" in n
        ):
            try:
                layer_num = None
                for marker in (
                    "roberta.encoder.layer.",
                    "xlm_roberta.encoder.layer.",
                    "backbone.encoder.layer.",
                ):
                    if marker in n:
                        layer_num = int(n.split(marker)[1].split(".")[0])
                        break
                if layer_num is None:
                    raise ValueError("Could not parse layer number")
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
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _cosine_with_warmup_floor_lambda(
                current_step=step,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_scheduler_steps,
                min_lr_ratio=eta_min_ratio,
            ),
        )
    else:
        raise ValueError(
            f"Unsupported scheduler '{scheduler_type}'. Use 'cosine'."
        )

    ground_truth_data = load_ground_truth(val_path)

    best_f1 = 0.0
    best_scores = None
    best_unique_pids = None
    best_metrics = None
    global_step = 0
    epochs_without_improvement = 0
    swa_state = None
    swa_count = 0

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
                if loss_type == "group_asl":
                    groups_batch = batch["groups"]
                    loss = criterion(logits, labels, groups_batch)
                else:
                    loss = criterion(logits, labels)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                global_step += 1
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_grad_norm,
                )
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

        unique_pids, aggregated_scores = aggregate_scores_by_patient(
            pid_to_logits,
            strategy=aggregation_strategy,
            temperature=aggregation_temperature,
        )

        if epoch == 0:
            multi_doc_patients = sum(1 for pid in unique_pids if pid in ground_truth_data and len(ground_truth_data[pid]) > 1)
            print(f"Aggregation info: {len(unique_pids)} patients total, {multi_doc_patients} patients have multiple documents pooled together.")

        sweep_max_f1 = 0.0
        sweep_argmax_t = eval_threshold

        threshold_metrics = {}
        for t in eval_thresholds:
            t_preds_bin = aggregated_scores >= t
            t_pred_data = {}
            for i, pid in enumerate(unique_pids):
                t_pred_indices = np.where(t_preds_bin[i])[0]
                t_pred_data[pid] = [train_dataset.labels[idx] for idx in t_pred_indices]
            
            t_metrics = evaluate_data(ground_truth_data, t_pred_data, label_space=train_dataset.labels)
            threshold_metrics[t] = t_metrics
            if t_metrics["micro_f1"] > sweep_max_f1:
                sweep_max_f1 = t_metrics["micro_f1"]
                sweep_argmax_t = t

        primary_metrics = threshold_metrics[primary_eval_threshold]
        val_f1 = primary_metrics["micro_f1"]
        metrics = primary_metrics
        print(f"Epoch {epoch + 1} - Primary Val Micro-F1: {val_f1:.4f} (Sweep Max: {sweep_max_f1:.4f} at thresh={sweep_argmax_t:.2f})")

        if swa_start_epoch > 0 and (epoch + 1) >= swa_start_epoch:
            current_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            if swa_state is None:
                swa_state = current_state
            else:
                for key, prev in swa_state.items():
                    if prev.is_floating_point():
                        swa_state[key] = (prev * swa_count + current_state[key]) / (swa_count + 1)
                    else:
                        swa_state[key] = current_state[key]
            swa_count += 1

        if wb_enabled:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss_epoch": total_loss / len(train_loader),
                "val/micro_f1_primary": val_f1,
                "val/precision": metrics["precision"],
                "val/recall": metrics["recall"],
                "lr": scheduler.get_last_lr()[0],
                "val/sweep_max_f1": sweep_max_f1,
                "val/sweep_argmax_t": sweep_argmax_t,
            }
            
            # Log per-threshold sweep
            for t, t_metrics in threshold_metrics.items():
                log_dict[f"val/f1_thresh_{t}"] = t_metrics["micro_f1"]

            if device.type == "cuda":
                log_dict["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
                log_dict["gpu/memory_reserved_gb"]  = torch.cuda.memory_reserved(device) / 1e9
            wandb.log(log_dict, step=global_step)

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            best_scores = aggregated_scores.copy()
            best_unique_pids = list(unique_pids)
            best_metrics = metrics
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
            
            if wb_enabled and wb_log_model_artifact:
                artifact = wandb.Artifact(
                    name="model-best",
                    type="model",
                    metadata={"val_micro_f1": best_f1, "epoch": epoch + 1},
                )
                artifact.add_dir(checkpoint_dir)
                wandb.log_artifact(artifact)
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs without improvement.")
                break

    if swa_state is not None and swa_count > 0:
        print(f"Evaluating SWA model built from {swa_count} epochs...")
        swa_model = build_model(
            num_labels=num_labels,
            model_name=model_name,
            classifier_dropout=classifier_dropout,
            pooling_strategy=pooling_strategy,
            multi_sample_dropout_samples=multi_sample_dropout_samples,
            init_classifier_from_descriptions=False,
            label_descriptions=label_descriptions,
            desc_init_scale=desc_init_scale,
            use_desc_residual=use_desc_residual,
            desc_residual_alpha_init=desc_residual_alpha_init,
        )
        swa_model.load_state_dict(swa_state, strict=True)
        swa_model.to(device)
        if (
            rebake_desc_after_swa
            and isinstance(swa_model, DescResidualWrapper)
            and label_descriptions
        ):
            print("Re-baking description embeddings with SWA backbone...")
            rebake_description_embeddings(
                swa_model, label_descriptions, tokenizer, device
            )
        swa_model.eval()

        pid_to_logits = {}
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="SWA [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pids = batch["patient_id"].tolist()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = swa_model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits.detach().cpu().numpy()

                for i, pid in enumerate(pids):
                    pid_to_logits.setdefault(pid, []).append(logits[i])

        swa_pids, swa_scores = aggregate_scores_by_patient(
            pid_to_logits,
            strategy=aggregation_strategy,
            temperature=aggregation_temperature,
        )
        swa_sweep_max_f1 = 0.0
        swa_sweep_argmax_t = eval_threshold
        threshold_metrics = {}
        for t in eval_thresholds:
            preds_bin = swa_scores >= t
            pred_data = {}
            for i, pid in enumerate(swa_pids):
                pred_indices = np.where(preds_bin[i])[0]
                pred_data[pid] = [train_dataset.labels[idx] for idx in pred_indices]
            t_metrics = evaluate_data(
                ground_truth_data, pred_data, label_space=train_dataset.labels
            )
            threshold_metrics[t] = t_metrics
            if t_metrics["micro_f1"] > swa_sweep_max_f1:
                swa_sweep_max_f1 = t_metrics["micro_f1"]
                swa_sweep_argmax_t = t

        swa_primary_metrics = threshold_metrics[primary_eval_threshold]
        swa_best_f1 = swa_primary_metrics["micro_f1"]
        swa_metrics = swa_primary_metrics

        print(
            f"SWA Primary Val Micro-F1: {swa_best_f1:.4f} (Sweep Max: {swa_sweep_max_f1:.4f} at thresh={swa_sweep_argmax_t:.2f})"
        )
        if swa_save_if_best and swa_best_f1 > best_f1:
            best_f1 = swa_best_f1
            best_scores = swa_scores.copy()
            best_unique_pids = list(swa_pids)
            best_metrics = swa_metrics
            print("SWA model outperformed best checkpoint. Saving SWA weights.")
            swa_model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            np.save(scores_path, swa_scores)
            with open(pids_path, "w", encoding="utf-8") as f:
                json.dump(swa_pids, f)
            with open(label_names_path, "w", encoding="utf-8") as f:
                json.dump(train_dataset.labels, f)

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
            f"Threshold tuning complete. In-sample tuned F1={tuned_f1:.4f} (optimistic)."
        )

    print(f"Training complete. Best Val F1: {best_f1:.4f}")

    if wb_enabled:
        wandb.summary["best_val_micro_f1"] = best_f1
        if auto_threshold_tuning and 'tuned_f1' in locals():
            wandb.summary["val/tuned_f1_in_sample"] = tuned_f1
        wandb.finish()


if __name__ == "__main__":
    main()
