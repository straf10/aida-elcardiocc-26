from __future__ import annotations

import os
import sys

_REPO_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    DataCollatorForTokenClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from dictionary.config import get_config_path_default, load_dictionary_config
from dictionary.export import load_code_description_csv
from dictionary.matcher import build_automaton
from evaluation.scoring import evaluate_data, print_metrics_short
from preprocessing.io_utils import LABELSET_PATH, load_labelset
from ner_el.bio_dataset import ID2LABEL, LABEL2ID, NERDataset
from ner_el.config import parse_train_args
from ner_el.context_reranker import ContextReranker
from ner_el.decode import decode_mentions_from_logits, decode_mentions_from_paths
from ner_el.dictionary_features import (
    extract_dictionary_codes,
    extract_dictionary_mentions,
    load_dictionary_candidates,
)
from ner_el.io_utils import load_documents, validate_document_schema
from ner_el.linker import MentionLinker, build_prior_map, default_prior_artifact_path, save_prior_map
from ner_el.model import build_ner_model, build_ner_model_with_crf
from ner_el.partial_crf import PartialCRF


class MetricEarlyStoppingCallback(TrainerCallback):
    """Stop training when eval metric plateaus, independent of HF best-model loader."""

    def __init__(self, metric_name: str, patience: int, greater_is_better: bool = True) -> None:
        self.metric_name = str(metric_name)
        self.patience = int(max(0, patience))
        self.greater_is_better = bool(greater_is_better)
        self.best_metric: float | None = None
        self.bad_epochs = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.patience <= 0:
            return control
        metrics = metrics or {}
        raw = metrics.get(self.metric_name)
        if raw is None and not self.metric_name.startswith("eval_"):
            raw = metrics.get(f"eval_{self.metric_name}")
        if raw is None:
            return control
        current = float(raw)
        improved = (
            self.best_metric is None
            or (current > self.best_metric if self.greater_is_better else current < self.best_metric)
        )
        if improved:
            self.best_metric = current
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                print(
                    f"Early stopping triggered after {self.bad_epochs} non-improving evals "
                    f"on {self.metric_name}."
                )
                control.should_training_stop = True
        return control


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


def _build_bio_sequences(logits, labels, id2label: Dict[int, str]) -> Tuple[List[List[str]], List[List[str]]]:
    preds = np.asarray(logits).argmax(axis=-1)
    labels = np.asarray(labels)
    mask = labels != -100
    y_true_seqs: List[List[str]] = []
    y_pred_seqs: List[List[str]] = []
    for i in range(labels.shape[0]):
        m = mask[i]
        y_true_seqs.append([id2label[int(t)] for t in labels[i][m]])
        y_pred_seqs.append([id2label[int(t)] for t in preds[i][m]])
    return y_true_seqs, y_pred_seqs


def _seqeval_metrics(y_true_seqs: List[List[str]], y_pred_seqs: List[List[str]]) -> Dict[str, float]:
    try:
        from seqeval.metrics import precision_score, recall_score, f1_score
        from seqeval.scheme import IOB2
    except ImportError:
        return {
            "seqeval_precision": 0.0,
            "seqeval_recall": 0.0,
            "seqeval_f1": 0.0,
            "seqeval_available": 0.0,
        }
    return {
        "seqeval_precision": float(
            precision_score(
                y_true_seqs,
                y_pred_seqs,
                mode="strict",
                scheme=IOB2,
                zero_division=0,
            )
        ),
        "seqeval_recall": float(
            recall_score(
                y_true_seqs,
                y_pred_seqs,
                mode="strict",
                scheme=IOB2,
                zero_division=0,
            )
        ),
        "seqeval_f1": float(
            f1_score(
                y_true_seqs,
                y_pred_seqs,
                mode="strict",
                scheme=IOB2,
                zero_division=0,
            )
        ),
        "seqeval_available": 1.0,
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
    def __init__(
        self,
        *args,
        class_weights: torch.Tensor,
        use_partial_crf: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_partial_crf = bool(use_partial_crf)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        allow_mask = inputs.get("allow_mask")
        model_inputs = {
            k: v
            for k, v in inputs.items()
            if k not in {"labels", "allow_mask", "partial_annotation"}
        }
        outputs = model(**model_inputs)
        logits = outputs.logits
        if self.use_partial_crf:
            if not hasattr(model, "crf"):
                raise ValueError("use_partial_crf=True but model has no CRF head.")
            if allow_mask is None:
                raise ValueError("Missing allow_mask for Partial CRF loss.")
            loss = model.crf(
                logits,
                allow_mask=allow_mask.to(logits.device).bool(),
                attention_mask=model_inputs["attention_mask"].to(logits.device),
            )
        else:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device),
                ignore_index=-100,
            )
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class PartialLabelCollator:
    def __init__(self, tokenizer, num_labels: int):
        self.tokenizer = tokenizer
        self.num_labels = int(num_labels)

    def __call__(self, features):
        pad_id = int(self.tokenizer.pad_token_id or 0)
        input_ids = [
            torch.tensor(list(f["input_ids"]), dtype=torch.long)
            for f in features
        ]
        attention_mask = [
            torch.tensor(list(f["attention_mask"]), dtype=torch.long)
            for f in features
        ]
        labels = [
            torch.tensor(list(f["labels"]), dtype=torch.long)
            for f in features
        ]

        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

        batch_size, max_len = input_ids_padded.shape
        allow_mask = torch.zeros((batch_size, max_len, self.num_labels), dtype=torch.bool)
        partial_annotation = torch.zeros(batch_size, dtype=torch.long)
        for i, f in enumerate(features):
            rows = f.get("allow_mask", [])
            if rows:
                cur = torch.tensor(rows, dtype=torch.bool)
                allow_mask[i, : cur.shape[0], :] = cur
            partial_annotation[i] = int(f.get("partial_annotation", 0))
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "allow_mask": allow_mask,
            "partial_annotation": partial_annotation,
        }


def _resolve_best_checkpoint(output_dir: str, trainer: Trainer) -> str:
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    if best_ckpt and Path(best_ckpt).exists():
        return str(best_ckpt)
    checkpoints = sorted(
        (p for p in Path(output_dir).glob("checkpoint-*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if checkpoints:
        return str(checkpoints[-1])
    return output_dir


def _decode_mentions_with_crf_batch(
    *,
    docs,
    val_ds: NERDataset,
    logits: np.ndarray,
    crf_module: PartialCRF,
):
    offsets_per_doc = [val_ds.examples[idx].offsets for idx in range(len(docs))]
    max_len = max((len(offsets) for offsets in offsets_per_doc), default=0)
    if max_len == 0:
        return [[] for _ in docs]

    num_labels = int(np.asarray(logits).shape[-1])
    crf_device = next(crf_module.parameters()).device
    emissions = torch.zeros((len(docs), max_len, num_labels), dtype=torch.float32, device=crf_device)
    mask = torch.zeros((len(docs), max_len), dtype=torch.long, device=crf_device)
    for idx, offsets in enumerate(offsets_per_doc):
        seq_len = len(offsets)
        if seq_len == 0:
            continue
        doc_logits = np.asarray(logits[idx], dtype=np.float32)[:seq_len]
        emissions[idx, :seq_len, :] = torch.as_tensor(doc_logits, dtype=torch.float32, device=crf_device)
        mask[idx, :seq_len] = torch.as_tensor(
            [1 if s != e else 0 for s, e in offsets],
            dtype=torch.long,
            device=crf_device,
        )

    with torch.no_grad():
        paths = crf_module.decode(emissions, mask)

    mentions_by_doc = []
    for idx, doc in enumerate(docs):
        offsets = offsets_per_doc[idx]
        seq_len = len(offsets)
        doc_logits = np.asarray(logits[idx])[:seq_len]
        path = paths[idx][:seq_len]
        mentions_by_doc.append(
            decode_mentions_from_paths(
                doc.text,
                offsets,
                doc_logits,
                path,
            )
        )
    return mentions_by_doc


def _load_checkpoint_state_dict(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    ckpt_dir = Path(checkpoint_dir)
    safetensor_path = ckpt_dir / "model.safetensors"
    if safetensor_path.exists():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("safetensors is required to load model.safetensors checkpoints.") from exc
        return load_file(str(safetensor_path))
    bin_path = ckpt_dir / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No model weights found in checkpoint: {checkpoint_dir}")


def _reload_model_from_trainer_checkpoint(cfg, checkpoint_dir: str):
    model = build_ner_model_with_crf(cfg.model_name) if cfg.use_partial_crf else build_ner_model(cfg.model_name)
    state_dict = _load_checkpoint_state_dict(checkpoint_dir)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Reload warning: missing keys when loading checkpoint ({len(missing)}).")
    if unexpected:
        print(f"Reload warning: unexpected keys when loading checkpoint ({len(unexpected)}).")
    return model


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
    use_partial_crf: bool,
    crf_module: PartialCRF | None,
):
    ground_truth = {doc.patient_id: doc.document_level_annotations for doc in val_docs}

    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        metrics = token_metrics(eval_pred)
        logits_np = np.asarray(logits)
        labels_np = np.asarray(labels)
        y_true_seqs, y_pred_seqs = _build_bio_sequences(logits_np, labels_np, ID2LABEL)
        metrics.update(_seqeval_metrics(y_true_seqs, y_pred_seqs))
        predictions = {}
        span_tp = 0
        span_fp = 0
        span_fn = 0
        crf_mentions = None
        if use_partial_crf and crf_module is not None:
            crf_mentions = _decode_mentions_with_crf_batch(
                docs=val_docs,
                val_ds=val_ds,
                logits=np.asarray(logits),
                crf_module=crf_module,
            )

        for idx, doc in enumerate(val_docs):
            offsets = val_ds.examples[idx].offsets
            doc_logits = logits[idx][: len(offsets)]
            if crf_mentions is not None:
                mentions = crf_mentions[idx]
            else:
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

            linked_mentions = linker.link_mentions(mentions, context_text=doc.text)
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


def _run_final_inference(
    *,
    trainer: Trainer,
    val_docs,
    val_ds: NERDataset,
    linker: MentionLinker,
    dictionary_map,
    dictionary_matcher,
    dictionary_config,
    labelset: List[str],
    code_desc_map: Dict[str, str],
    use_dictionary_fusion: bool,
    dictionary_doc_boost: bool,
    dictionary_word_boundary: bool,
    use_partial_crf: bool,
    crf_module: PartialCRF | None,
) -> Tuple[Dict, Dict[str, float], List[Dict]]:
    """Run a single forward pass over the validation set with the (loaded
    best) model and return official doc-level metrics, auxiliary span metrics
    and per-document debug records."""
    predictions_output = trainer.predict(val_ds)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids

    mask = labels != -100
    y_true = labels[mask]
    y_pred = np.asarray(logits).argmax(axis=-1)[mask]
    tp = int(((y_pred != LABEL2ID["O"]) & (y_true != LABEL2ID["O"])).sum())
    fp = int(((y_pred != LABEL2ID["O"]) & (y_true == LABEL2ID["O"])).sum())
    fn = int(((y_pred == LABEL2ID["O"]) & (y_true != LABEL2ID["O"])).sum())
    tok_precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    tok_recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    tok_f1 = (
        2 * tok_precision * tok_recall / (tok_precision + tok_recall)
        if (tok_precision + tok_recall) > 0
        else 0.0
    )

    ground_truth = {doc.patient_id: doc.document_level_annotations for doc in val_docs}
    predictions: Dict[int, List[str]] = {}
    debug_records: List[Dict] = []

    span_tp = span_fp = span_fn = 0
    crf_mentions = None
    if use_partial_crf and crf_module is not None:
        crf_mentions = _decode_mentions_with_crf_batch(
            docs=val_docs,
            val_ds=val_ds,
            logits=np.asarray(logits),
            crf_module=crf_module,
        )

    for idx, doc in enumerate(val_docs):
        offsets = val_ds.examples[idx].offsets
        doc_logits = np.asarray(logits[idx])[: len(offsets)]
        if crf_mentions is not None:
            mentions = crf_mentions[idx]
        else:
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

        linked_mentions = linker.link_mentions(mentions, context_text=doc.text)
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
        debug_records.append(
            {
                "patient_id": doc.patient_id,
                "text": doc.text,
                "mention_level_annotations": [
                    {
                        "start": m.start,
                        "end": m.end,
                        "text": m.text,
                        "code": m.code,
                        "confidence": m.confidence,
                    }
                    for m in linked_mentions
                ],
                "document_level_annotations": doc_codes,
            }
        )

    span_precision = span_tp / (span_tp + span_fp) if (span_tp + span_fp) > 0 else 0.0
    span_recall = span_tp / (span_tp + span_fn) if (span_tp + span_fn) > 0 else 0.0
    span_f1 = (
        2 * span_precision * span_recall / (span_precision + span_recall)
        if (span_precision + span_recall) > 0
        else 0.0
    )

    official = evaluate_data(ground_truth, predictions, label_space=labelset)
    logits_np = np.asarray(logits)
    labels_np = np.asarray(labels)
    y_true_seqs, y_pred_seqs = _build_bio_sequences(logits_np, labels_np, ID2LABEL)
    seqeval_block = _seqeval_metrics(y_true_seqs, y_pred_seqs)
    aux_metrics = {
        "token_precision": tok_precision,
        "token_recall": tok_recall,
        "token_f1": tok_f1,
        "span_precision": span_precision,
        "span_recall": span_recall,
        "span_f1": span_f1,
        "span_tp": span_tp,
        "span_fp": span_fp,
        "span_fn": span_fn,
        **seqeval_block,
    }
    return official, aux_metrics, debug_records


def _print_training_summary(
    *,
    cfg,
    train_metrics: Dict,
    aux_metrics: Dict[str, float],
    official: Dict,
    metrics_json_path: str,
    debug_path: str,
) -> None:
    def _fmt(value, digits: int = 4) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    banner = "=" * 66
    run_name = Path(cfg.output_dir).name or cfg.output_dir
    print(banner)
    print(f"NER TRAINING SUMMARY :: {run_name}")
    print(banner)
    if train_metrics.get("train_runtime") is not None:
        print(
            f"Epochs: {cfg.epochs} | train_loss: {_fmt(train_metrics.get('train_loss'))} | "
            f"runtime: {_fmt(train_metrics.get('train_runtime'), 1)}s"
        )
        print(
            f"Samples/s: {_fmt(train_metrics.get('train_samples_per_second'), 2)} | "
            f"Steps/s: {_fmt(train_metrics.get('train_steps_per_second'), 3)}"
        )
    else:
        print("Evaluation-only run (no training stats).")
    print("")
    print("-- Token-level (auxiliary) --")
    print(
        f"  Precision={_fmt(aux_metrics['token_precision'])}  "
        f"Recall={_fmt(aux_metrics['token_recall'])}  "
        f"F1={_fmt(aux_metrics['token_f1'])}"
    )
    print("")
    print("-- Seqeval (entity-level, strict IOB2) --")
    if aux_metrics.get("seqeval_available", 1.0) < 0.5:
        print("  (seqeval not installed; pip install seqeval)")
    else:
        print(
            f"  Precision={_fmt(aux_metrics.get('seqeval_precision', 0.0))}  "
            f"Recall={_fmt(aux_metrics.get('seqeval_recall', 0.0))}  "
            f"F1={_fmt(aux_metrics.get('seqeval_f1', 0.0))}"
        )
    print("")
    print("-- Span-level (auxiliary, exact match) --")
    span_note = ""
    if aux_metrics["span_f1"] < 0.05:
        span_note = (
            "  !! low (<0.05): gold uses char spans vs pred uses tokenizer offsets—"
            "often not a checkpoint bug; prefer seqeval entity F1 above"
        )
    print(
        f"  Precision={_fmt(aux_metrics['span_precision'])}  "
        f"Recall={_fmt(aux_metrics['span_recall'])}  "
        f"F1={_fmt(aux_metrics['span_f1'])}"
        f"  [TP={aux_metrics['span_tp']} FP={aux_metrics['span_fp']} FN={aux_metrics['span_fn']}]"
        f"{span_note}"
    )
    print("")
    print("-- Document-level (official, evaluation.scoring) --")
    print_metrics_short(official, metrics_json=metrics_json_path)
    if debug_path:
        print(f"Wrote debug JSONL -> {debug_path}")
    print(banner)


def _write_debug_jsonl(records: Sequence[Dict], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


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
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
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
        use_partial_crf=cfg.use_partial_crf,
        partial_all=cfg.partial_all,
    )
    val_ds = NERDataset(
        val_docs,
        model_name=cfg.model_name,
        max_length=cfg.max_length,
        dynamic_padding=cfg.dynamic_padding,
        dictionary_map=dictionary_map if cfg.use_dictionary_augmentation else None,
        dictionary_word_boundary=dictionary_word_boundary,
        use_dictionary_augmentation=cfg.use_dictionary_augmentation,
        use_partial_crf=cfg.use_partial_crf,
        partial_all=cfg.partial_all,
    )

    model = (
        build_ner_model_with_crf(cfg.model_name)
        if cfg.use_partial_crf
        else build_ner_model(cfg.model_name)
    )
    prior_map = build_prior_map(train_docs)
    reranker = None
    if cfg.use_reranker:
        artifact_meta = Path(cfg.reranker_artifact_dir) / ContextReranker.META_FILENAME
        artifact_emb = Path(cfg.reranker_artifact_dir) / ContextReranker.EMBEDDINGS_FILENAME
        if artifact_meta.exists() and artifact_emb.exists():
            reranker = ContextReranker.load(
                artifact_dir=cfg.reranker_artifact_dir,
                code_desc_map=code_desc_map,
            )
        else:
            reranker = ContextReranker(
                code_desc_map=code_desc_map,
                model_name=cfg.reranker_model,
                window_chars=cfg.reranker_window_chars,
            ).fit(labelset)
            reranker.save(cfg.reranker_artifact_dir)
    linker = MentionLinker(
        prior_map=prior_map,
        dictionary_map=dictionary_map,
        reranker=reranker,
        alpha=cfg.reranker_alpha,
    )
    compute_metrics = build_compute_metrics(
        val_docs=val_docs,
        val_ds=val_ds,
        linker=linker,
        dictionary_map=dictionary_map,
        dictionary_matcher=dictionary_matcher,
        dictionary_config=dictionary_cfg,
        labelset=labelset,
        code_desc_map=code_desc_map,
        use_dictionary_fusion=cfg.use_dictionary_fusion,
        dictionary_doc_boost=cfg.dictionary_doc_boost,
        dictionary_word_boundary=dictionary_word_boundary,
        use_partial_crf=cfg.use_partial_crf,
        crf_module=(
            None
            if not cfg.use_partial_crf or not hasattr(model, "crf")
            else model.crf
        ),
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
        load_best_model_at_end=False,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=True,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=max(0, int(cfg.dataloader_num_workers)),
    )
    if cfg.bf16 is not None:
        common_args["bf16"] = bool(cfg.bf16)
    if cfg.fp16 is not None:
        common_args["fp16"] = bool(cfg.fp16)
    if cfg.save_total_limit and cfg.save_total_limit > 0:
        common_args["save_total_limit"] = int(cfg.save_total_limit)

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

    if cfg.use_partial_crf:
        data_collator = PartialLabelCollator(train_ds.tokenizer, num_labels=len(LABEL2ID))
    else:
        data_collator = (
            DataCollatorForTokenClassification(tokenizer=train_ds.tokenizer, padding=True)
            if cfg.dynamic_padding
            else None
        )

    trainer_cls = WeightedTrainer if (cfg.use_class_weights or cfg.use_partial_crf) else Trainer
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    if cfg.use_class_weights or cfg.use_partial_crf:
        trainer_kwargs["class_weights"] = torch.tensor(cfg.class_weights, dtype=torch.float)
        trainer_kwargs["use_partial_crf"] = cfg.use_partial_crf

    trainer = trainer_cls(**trainer_kwargs)
    if cfg.early_stopping_patience > 0:
        trainer.add_callback(
            MetricEarlyStoppingCallback(
                metric_name=cfg.metric_for_best_model,
                patience=cfg.early_stopping_patience,
                greater_is_better=True,
            )
        )

    train_output = trainer.train()
    best_ckpt = _resolve_best_checkpoint(cfg.output_dir, trainer)
    best_dir = os.path.join(cfg.output_dir, "best")
    reloaded_model = _reload_model_from_trainer_checkpoint(cfg, best_ckpt)
    if cfg.use_partial_crf and hasattr(reloaded_model, "crf"):
        crf_abs_sum = float(torch.sum(torch.abs(reloaded_model.crf.transitions)).item())
        print(f"Reloaded CRF head from best checkpoint (|transitions| sum={crf_abs_sum:.4f}).")
    trainer.model = reloaded_model.to(trainer.args.device)
    trainer.save_model(best_dir)
    train_ds.tokenizer.save_pretrained(best_dir)
    print(
        f"Best checkpoint: {best_ckpt} | metric={trainer.state.best_metric} "
        f"| global_step={trainer.state.global_step}"
    )

    prior_path = default_prior_artifact_path(best_dir)
    save_prior_map(prior_map, prior_path)
    print(f"Saved linker prior artifact: {prior_path}")

    _export_checkpoint(best_dir, cfg.export_dir)
    print(f"Exported best checkpoint to: {cfg.export_dir}")

    run_name = Path(cfg.output_dir).name or "ner_el"
    default_metrics_path = Path(cfg.output_dir).parent / f"{run_name}_val_metrics.json"
    default_debug_path = Path(cfg.output_dir).parent / f"{run_name}_val_debug.jsonl"
    metrics_json_path = str(cfg.metrics_json_path or default_metrics_path)
    debug_jsonl_path = str(default_debug_path)

    official, aux_metrics, debug_records = _run_final_inference(
        trainer=trainer,
        val_docs=val_docs,
        val_ds=val_ds,
        linker=linker,
        dictionary_map=dictionary_map,
        dictionary_matcher=dictionary_matcher,
        dictionary_config=dictionary_cfg,
        labelset=labelset,
        code_desc_map=code_desc_map,
        use_dictionary_fusion=cfg.use_dictionary_fusion,
        dictionary_doc_boost=cfg.dictionary_doc_boost,
        dictionary_word_boundary=dictionary_word_boundary,
        use_partial_crf=cfg.use_partial_crf,
        crf_module=(
            None
            if not cfg.use_partial_crf or not hasattr(trainer.model, "crf")
            else trainer.model.crf
        ),
    )
    _write_debug_jsonl(debug_records, debug_jsonl_path)

    train_metrics: Dict = getattr(train_output, "metrics", {}) or {}
    _print_training_summary(
        cfg=cfg,
        train_metrics=train_metrics,
        aux_metrics=aux_metrics,
        official=official,
        metrics_json_path=metrics_json_path,
        debug_path=debug_jsonl_path,
    )


if __name__ == "__main__":
    main()
