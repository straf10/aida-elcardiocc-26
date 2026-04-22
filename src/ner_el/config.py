from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TrainConfig:
    train_path: str = "data/processed/train.jsonl"
    val_path: str = "data/processed/val.jsonl"
    model_name: str = "nlpaueb/bert-base-greek-uncased-v1"
    output_dir: str = "outputs/experiments/ner_el/greek_bert_ner"
    export_dir: str = "outputs/models/ner_el"
    max_length: int = 512
    epochs: int = 3
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    use_dictionary_augmentation: bool = False


@dataclass
class PredictConfig:
    """CLI inference defaults only. ``TrainConfig`` / ``ner_el.train`` still use train+val paths."""

    model_dir: str = "outputs/models/ner_el"
    tokenizer_name: str = "nlpaueb/bert-base-greek-uncased-v1"
    input_path: str = "data/processed/test.jsonl"
    train_path_for_linker: Optional[str] = None
    output_doc_path: str = "outputs/predictions/ner_el/predictions.jsonl"
    output_debug_path: str = "outputs/experiments/ner_el/ner_el_main_debug.jsonl"
    max_length: int = 512
    use_dictionary_fusion: bool = True
    dictionary_doc_boost: bool = True

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PredictConfig":
        return cls(
            model_dir=str(raw.get("model_dir", cls.model_dir)),
            tokenizer_name=str(raw.get("tokenizer_name", cls.tokenizer_name)),
            input_path=str(raw.get("input_path", cls.input_path)),
            train_path_for_linker=raw.get("train_path_for_linker"),
            output_doc_path=str(raw.get("output_doc_path", cls.output_doc_path)),
            output_debug_path=str(raw.get("output_debug_path", cls.output_debug_path)),
            max_length=int(raw.get("max_length", cls.max_length)),
            use_dictionary_fusion=bool(raw.get("use_dictionary_fusion", cls.use_dictionary_fusion)),
            dictionary_doc_boost=bool(raw.get("dictionary_doc_boost", cls.dictionary_doc_boost)),
        )

    def validate_for_cli(self) -> None:
        model_path = Path(self.model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        if not Path(self.input_path).exists():
            raise FileNotFoundError(f"Input path not found: {self.input_path}")
        if self.train_path_for_linker and not Path(self.train_path_for_linker).exists():
            raise FileNotFoundError(f"Fallback linker train path not found: {self.train_path_for_linker}")


def parse_train_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train BIO NER model")
    parser.add_argument("--train-path", default=TrainConfig.train_path)
    parser.add_argument("--val-path", default=TrainConfig.val_path)
    parser.add_argument("--model-name", default=TrainConfig.model_name)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--export-dir", default=TrainConfig.export_dir)
    parser.add_argument("--max-length", type=int, default=TrainConfig.max_length)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--train-batch-size", type=int, default=TrainConfig.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=TrainConfig.eval_batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--use-dictionary-augmentation", action="store_true")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def parse_predict_args() -> PredictConfig:
    parser = argparse.ArgumentParser(description="Run NER+EL inference")
    parser.add_argument("--model-dir", default=PredictConfig.model_dir)
    parser.add_argument("--tokenizer-name", default=PredictConfig.tokenizer_name)
    parser.add_argument("--input-path", default=PredictConfig.input_path)
    parser.add_argument("--train-path-for-linker", default=PredictConfig.train_path_for_linker)
    parser.add_argument("--output-doc-path", default=PredictConfig.output_doc_path)
    parser.add_argument("--output-debug-path", default=PredictConfig.output_debug_path)
    parser.add_argument("--max-length", type=int, default=PredictConfig.max_length)
    parser.add_argument("--no-dictionary-fusion", action="store_true")
    parser.add_argument("--no-dictionary-doc-boost", action="store_true")
    args = parser.parse_args()
    cfg = PredictConfig(
        model_dir=args.model_dir,
        tokenizer_name=args.tokenizer_name,
        input_path=args.input_path,
        train_path_for_linker=args.train_path_for_linker,
        output_doc_path=args.output_doc_path,
        output_debug_path=args.output_debug_path,
        max_length=args.max_length,
    )
    cfg.use_dictionary_fusion = not args.no_dictionary_fusion
    cfg.dictionary_doc_boost = not args.no_dictionary_doc_boost
    return cfg
