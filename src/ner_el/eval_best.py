"""Evaluate a trained NER+EL checkpoint (e.g. ``outputs/models/ner_el``) on val only.

Loads tokenizer + weights from ``model_dir`` (same tree as after training export),
runs ``_run_final_inference`` (token / seqeval / span-exact / document-level metrics).

Example (repo root, ``src`` on ``PYTHONPATH``)::

    $env:PYTHONPATH='src'
    python -m ner_el.eval_best --config src/ner_el/ner_el.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

import torch
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from dictionary.config import get_config_path_default, load_dictionary_config
from dictionary.export import load_code_description_csv
from dictionary.matcher import build_automaton
from evaluation.config_utils import get_cfg, load_config
from preprocessing.io_utils import LABELSET_PATH, load_labelset

from ner_el.bio_dataset import LABEL2ID, NERDataset
from ner_el.context_reranker import ContextReranker
from ner_el.dictionary_features import load_dictionary_candidates
from ner_el.io_utils import load_documents
from ner_el.linker import MentionLinker, default_prior_artifact_path, load_prior_map
from ner_el.model import load_ner_model_for_inference
from ner_el.train import (
    PartialLabelCollator,
    WeightedTrainer,
    _print_training_summary,
    _run_final_inference,
    _write_debug_jsonl,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ner_el best/export checkpoint on validation JSONL.")
    p.add_argument("--config", default="src/ner_el/ner_el.yaml", help="YAML with data/training/output/linker blocks.")
    p.add_argument(
        "--model-dir",
        default=None,
        help="Checkpoint dir (tokenizer + weights + optional partial_crf.pt). Defaults to output.export_dir in YAML.",
    )
    p.add_argument("--val-path", default=None, help="Defaults to data.val_path in YAML.")
    p.add_argument("--metrics-json", default=None, help="Where to write combined official + auxiliary metrics JSON.")
    p.add_argument("--debug-jsonl", default=None, help="Per-document debug JSONL (same shape as training).")
    p.add_argument("--batch-size", type=int, default=None, help="Defaults to training.eval_batch_size in YAML.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg_file = load_config(args.config)

    model_dir = args.model_dir or get_cfg(cfg_file, "output.export_dir", "outputs/models/ner_el")
    val_path = args.val_path or get_cfg(cfg_file, "data.val_path", "data/processed/val.jsonl")
    max_length = int(get_cfg(cfg_file, "training.max_length", 512))
    use_partial_crf = bool(get_cfg(cfg_file, "training.use_partial_crf", True))
    partial_all = bool(get_cfg(cfg_file, "training.partial_all", False))
    dynamic_padding = bool(get_cfg(cfg_file, "training.dynamic_padding", True))
    use_dict_fusion = bool(
        get_cfg(
            cfg_file,
            "training.use_dictionary_fusion",
            get_cfg(cfg_file, "prediction.use_dictionary_fusion", True),
        )
    )
    dict_doc_boost = bool(get_cfg(cfg_file, "training.dictionary_doc_boost", True))
    use_reranker = bool(get_cfg(cfg_file, "linker.use_reranker", True))
    reranker_alpha = float(get_cfg(cfg_file, "linker.alpha", 0.6))
    reranker_dir = str(get_cfg(cfg_file, "linker.artifact_dir", model_dir))
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else get_cfg(cfg_file, "training.eval_batch_size", 8)
    )

    exp_dir = Path(str(get_cfg(cfg_file, "output.output_dir", "outputs/experiments/ner_el/greek_bert_ner")))
    metrics_json = args.metrics_json or str(exp_dir.parent / "eval_only_metrics.json")
    debug_path = args.debug_jsonl or str(exp_dir.parent / "eval_only_debug.jsonl")

    if not Path(model_dir).exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not Path(val_path).exists():
        raise FileNotFoundError(f"Validation JSONL not found: {val_path}")

    val_docs = load_documents(val_path)
    labelset = load_labelset(LABELSET_PATH)
    dict_cfg = load_dictionary_config(get_config_path_default())
    dict_wb = bool((dict_cfg.matching or {}).get("word_boundary", False))
    dict_map = load_dictionary_candidates(labelset=labelset, config_path=get_config_path_default())
    dict_automaton = build_automaton(dict_map, word_boundary=dict_wb)
    code_desc_map = load_code_description_csv(dict_cfg.paths["code_description_csv"])

    val_ds = NERDataset(
        val_docs,
        model_name=str(Path(model_dir)),
        max_length=max_length,
        dynamic_padding=dynamic_padding,
        dictionary_map=None,
        dictionary_word_boundary=dict_wb,
        use_dictionary_augmentation=False,
        use_partial_crf=use_partial_crf,
        partial_all=partial_all,
    )

    model = load_ner_model_for_inference(model_dir, use_partial_crf=use_partial_crf)

    prior_path = default_prior_artifact_path(model_dir)
    prior_map = load_prior_map(prior_path) if Path(prior_path).exists() else {}

    reranker = None
    if use_reranker:
        meta = Path(reranker_dir) / ContextReranker.META_FILENAME
        emb = Path(reranker_dir) / ContextReranker.EMBEDDINGS_FILENAME
        if meta.exists() and emb.exists():
            reranker = ContextReranker.load(artifact_dir=str(reranker_dir), code_desc_map=code_desc_map)

    linker = MentionLinker(
        prior_map=prior_map,
        dictionary_map=dict_map,
        reranker=reranker,
        alpha=reranker_alpha,
    )

    if use_partial_crf:
        collator = PartialLabelCollator(val_ds.tokenizer, num_labels=len(LABEL2ID))
    elif dynamic_padding:
        collator = DataCollatorForTokenClassification(tokenizer=val_ds.tokenizer, padding=True)
    else:
        collator = None

    out_eval = str(Path(model_dir) / "_tmp_eval")
    common_args = dict(
        output_dir=out_eval,
        per_device_eval_batch_size=batch_size,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,
    )
    try:
        targs = TrainingArguments(evaluation_strategy="no", **common_args)
    except TypeError:
        targs = TrainingArguments(eval_strategy="no", **common_args)

    trainer_cls = WeightedTrainer if use_partial_crf else Trainer
    trainer_kwargs: dict = dict(model=model, args=targs, data_collator=collator)
    if use_partial_crf:
        trainer_kwargs["class_weights"] = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float)
        trainer_kwargs["use_partial_crf"] = True
    trainer = trainer_cls(**trainer_kwargs)

    crf_mod = trainer.model.crf if hasattr(trainer.model, "crf") else None
    official, aux_metrics, debug_records = _run_final_inference(
        trainer=trainer,
        val_docs=val_docs,
        val_ds=val_ds,
        linker=linker,
        dictionary_map=dict_map,
        dictionary_matcher=dict_automaton,
        dictionary_config=dict_cfg,
        labelset=labelset,
        code_desc_map=code_desc_map,
        use_dictionary_fusion=use_dict_fusion,
        dictionary_doc_boost=dict_doc_boost,
        dictionary_word_boundary=dict_wb,
        use_partial_crf=use_partial_crf,
        crf_module=crf_mod,
    )

    Path(metrics_json).parent.mkdir(parents=True, exist_ok=True)
    payload = {"official": official, "auxiliary": aux_metrics}
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_debug_jsonl(debug_records, debug_path)

    class _CfgShim:
        output_dir = str(exp_dir)
        epochs = 0

    _print_training_summary(
        cfg=_CfgShim(),
        train_metrics={},
        aux_metrics=aux_metrics,
        official=official,
        metrics_json_path=metrics_json,
        debug_path=debug_path,
    )


if __name__ == "__main__":
    main()
