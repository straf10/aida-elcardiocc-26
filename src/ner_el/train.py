from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
from transformers import Trainer, TrainingArguments

from .bio_dataset import ID2LABEL, LABEL2ID, NERDataset
from .config import parse_train_args
from .dictionary_features import load_dictionary_candidates
from .io_utils import load_documents, validate_document_schema
from .linker import build_prior_map, default_prior_artifact_path, save_prior_map
from .model import build_ner_model


def token_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    mask = labels != -100
    y_true = labels[mask]
    y_pred = preds[mask]

    tp = int(((y_pred != LABEL2ID["O"]) & (y_true != LABEL2ID["O"])).sum())
    fp = int(((y_pred != LABEL2ID["O"]) & (y_true == LABEL2ID["O"])).sum())
    fn = int(((y_pred == LABEL2ID["O"]) & (y_true != LABEL2ID["O"])).sum())

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }


def _export_checkpoint(source_dir: str, export_dir: str) -> None:
    source_path = Path(source_dir)
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    for item in source_path.iterdir():
        destination = export_path / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def main() -> None:
    cfg = parse_train_args()
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.export_dir).mkdir(parents=True, exist_ok=True)

    train_docs = load_documents(cfg.train_path)
    val_docs = load_documents(cfg.val_path)
    dictionary_map = load_dictionary_candidates() if cfg.use_dictionary_augmentation else None

    print("Train schema:", validate_document_schema(train_docs))
    print("Val schema:", validate_document_schema(val_docs))

    train_ds = NERDataset(
        train_docs,
        model_name=cfg.model_name,
        max_length=cfg.max_length,
        dictionary_map=dictionary_map,
        use_dictionary_augmentation=cfg.use_dictionary_augmentation,
    )
    val_ds = NERDataset(
        val_docs,
        model_name=cfg.model_name,
        max_length=cfg.max_length,
        dictionary_map=dictionary_map,
        use_dictionary_augmentation=cfg.use_dictionary_augmentation,
    )

    model = build_ner_model(cfg.model_name)

    common_args = dict(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="token_f1",
        greater_is_better=True,
        report_to=[],
    )

    try:
        args = TrainingArguments(
            evaluation_strategy="epoch",
            save_strategy="epoch",
            **common_args,
        )
    except TypeError:
        # transformers>=5 renamed evaluation_strategy -> eval_strategy
        args = TrainingArguments(
            eval_strategy="epoch",
            save_strategy="epoch",
            **common_args,
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=token_metrics,
    )

    trainer.train()
    best_dir = os.path.join(cfg.output_dir, "best")
    trainer.save_model(best_dir)
    train_ds.tokenizer.save_pretrained(best_dir)

    prior_map = build_prior_map(train_docs)
    prior_path = default_prior_artifact_path(best_dir)
    save_prior_map(prior_map, prior_path)
    print(f"Saved linker prior artifact: {prior_path}")

    _export_checkpoint(best_dir, cfg.export_dir)
    print(f"Exported best checkpoint to: {cfg.export_dir}")

    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)


if __name__ == "__main__":
    main()
