from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from dictionary.config import get_config_path_default, load_dictionary_config
from dictionary.export import load_code_description_csv
from dictionary.matcher import build_automaton
from evaluation.scoring import evaluate_data
from preprocessing.io_utils import LABELSET_PATH, load_labelset
from .bio_dataset import LABEL2ID, NERDataset
from .config import parse_train_args
from .decode import decode_mentions_from_logits
from .dictionary_features import (
    extract_dictionary_codes,
    extract_dictionary_mentions,
    load_dictionary_candidates,
)
from .io_utils import load_documents, validate_document_schema
from .linker import MentionLinker, build_prior_map, default_prior_artifact_path, save_prior_map
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


def _merge_mentions_containment(model_mentions, dict_mentions):
    merged = list(model_mentions)

    def _contains(a, b) -> bool:
        return a.start <= b.start and a.end >= b.end

    for mention in dict_mentions:
        if any(_contains(existing, mention) for existing in merged):
            continue
        merged = [existing for existing in merged if not _contains(mention, existing)]
        merged.append(mention)
    merged.sort(key=lambda x: (x.start, -(x.end - x.start)))
    return merged


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            ignore_index=-100,
        )
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def build_compute_metrics(
    *,
    val_docs,
    val_ds: NERDataset,
    linker: MentionLinker,
    dictionary_map: Dict[str, set],
    dictionary_matcher,
    dictionary_config,
    labelset: List[str],
    code_desc_map: Dict[str, str],
    use_dictionary_fusion: bool,
    dictionary_doc_boost: bool,
    dictionary_word_boundary: bool,
):
    ground_truth = {doc.patient_id: doc.document_level_annotations for doc in val_docs}

    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        metrics = token_metrics(eval_pred)
        predictions = {}
        span_tp = 0
        span_fp = 0
        span_fn = 0

        for idx, doc in enumerate(val_docs):
            offsets = val_ds.examples[idx].offsets
            doc_logits = logits[idx][: len(offsets)]
            mentions = decode_mentions_from_logits(doc.text, offsets, doc_logits)
            pred_spans = {(m.start, m.end) for m in mentions}
            gold_spans = {(m.start, m.end) for m in doc.mention_level_annotations}
            span_tp += len(pred_spans & gold_spans)
            span_fp += len(pred_spans - gold_spans)
            span_fn += len(gold_spans - pred_spans)

            if use_dictionary_fusion and dictionary_map:
                dict_mentions = extract_dictionary_mentions(
                    doc.text,
                    dictionary_map,
                    word_boundary=dictionary_word_boundary,
                )
                mentions = _merge_mentions_containment(mentions, dict_mentions)

            linked_mentions = linker.link_mentions(mentions)
            doc_codes: List[str] = []
            for mention in linked_mentions:
                if mention.code and mention.code not in doc_codes:
                    doc_codes.append(mention.code)

            if dictionary_doc_boost and dictionary_matcher and dictionary_config:
                boost_codes = extract_dictionary_codes(
                    doc.text,
                    dictionary_matcher,
                    dictionary_config,
                    labelset=labelset,
                    code_desc_map=code_desc_map,
                )
                for code in boost_codes:
                    if code not in doc_codes:
                        doc_codes.append(code)

            predictions[doc.patient_id] = doc_codes

        official = evaluate_data(ground_truth, predictions, label_space=labelset)
        span_precision = span_tp / (span_tp + span_fp) if (span_tp + span_fp) > 0 else 0.0
        span_recall = span_tp / (span_tp + span_fn) if (span_tp + span_fn) > 0 else 0.0
        span_f1 = (
            2 * span_precision * span_recall / (span_precision + span_recall)
            if (span_precision + span_recall) > 0
            else 0.0
        )
        metrics.update(
            {
                "micro_f1": float(official["micro_f1"]),
                "precision": float(official["precision"]),
                "recall": float(official["recall"]),
                "span_precision": span_precision,
                "span_recall": span_recall,
                "span_f1": span_f1,
            }
        )
        return metrics

    return compute_metrics


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
    labelset = load_labelset(LABELSET_PATH)
    dictionary_cfg = load_dictionary_config(get_config_path_default())
    dictionary_word_boundary = bool((dictionary_cfg.matching or {}).get("word_boundary", False))
    dictionary_map = load_dictionary_candidates(
        labelset=labelset,
        config_path=get_config_path_default(),
    )
    dictionary_matcher = build_automaton(
        dictionary_map,
        word_boundary=dictionary_word_boundary,
    )
    code_desc_map = load_code_description_csv(dictionary_cfg.paths["code_description_csv"])

    print("Train schema:", validate_document_schema(train_docs))
    print("Val schema:", validate_document_schema(val_docs))

    train_ds = NERDataset(
        train_docs,
        model_name=cfg.model_name,
        max_length=cfg.max_length,
        dynamic_padding=cfg.dynamic_padding,
        dictionary_map=dictionary_map if cfg.use_dictionary_augmentation else None,
        dictionary_word_boundary=dictionary_word_boundary,
        use_dictionary_augmentation=cfg.use_dictionary_augmentation,
    )
    val_ds = NERDataset(
        val_docs,
        model_name=cfg.model_name,
        max_length=cfg.max_length,
        dynamic_padding=cfg.dynamic_padding,
        dictionary_map=dictionary_map if cfg.use_dictionary_augmentation else None,
        dictionary_word_boundary=dictionary_word_boundary,
        use_dictionary_augmentation=cfg.use_dictionary_augmentation,
    )

    model = build_ner_model(cfg.model_name)
    prior_map = build_prior_map(train_docs)
    linker = MentionLinker(prior_map=prior_map, dictionary_map=dictionary_map)
    compute_metrics = build_compute_metrics(
        val_docs=val_docs,
        val_ds=val_ds,
        linker=linker,
        dictionary_map=dictionary_map,
        dictionary_matcher=dictionary_matcher,
        dictionary_config=dictionary_cfg,
        labelset=labelset,
        code_desc_map=code_desc_map,
        use_dictionary_fusion=cfg.use_dictionary_augmentation,
        dictionary_doc_boost=cfg.dictionary_doc_boost,
        dictionary_word_boundary=dictionary_word_boundary,
    )

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
        metric_for_best_model=cfg.metric_for_best_model,
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

    data_collator = (
        DataCollatorForTokenClassification(tokenizer=train_ds.tokenizer, padding=True)
        if cfg.dynamic_padding
        else None
    )

    trainer_cls = WeightedTrainer if cfg.use_class_weights else Trainer
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    if cfg.use_class_weights:
        trainer_kwargs["class_weights"] = torch.tensor(cfg.class_weights, dtype=torch.float)

    trainer = trainer_cls(**trainer_kwargs)

    trainer.train()
    best_dir = os.path.join(cfg.output_dir, "best")
    trainer.save_model(best_dir)
    train_ds.tokenizer.save_pretrained(best_dir)

    prior_path = default_prior_artifact_path(best_dir)
    save_prior_map(prior_map, prior_path)
    print(f"Saved linker prior artifact: {prior_path}")

    _export_checkpoint(best_dir, cfg.export_dir)
    print(f"Exported best checkpoint to: {cfg.export_dir}")

    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)


if __name__ == "__main__":
    main()
