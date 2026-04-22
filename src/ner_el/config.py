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
    dictionary_doc_boost: bool = True
    dynamic_padding: bool = True
    metric_for_best_model: str = "micro_f1"
    use_class_weights: bool = False
    class_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    save_total_limit: int = 1
    metrics_json_path: Optional[str] = None


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
    def _parse_class_weights(raw: str) -> tuple[float, float, float]:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("class weights must have exactly 3 comma-separated values")
        try:
            values = tuple(float(p) for p in parts)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("class weights must be numeric values") from exc
        return values  # type: ignore[return-value]

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
    parser.add_argument("--no-dictionary-doc-boost", action="store_true")
    parser.add_argument("--no-dynamic-padding", action="store_true")
    parser.add_argument("--metric-for-best-model", default=TrainConfig.metric_for_best_model)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument(
        "--class-weights",
        type=_parse_class_weights,
        default=TrainConfig.class_weights,
        help="Comma-separated weights for labels O,B-MED,I-MED",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=TrainConfig.save_total_limit,
        help="Max number of epoch checkpoints to keep on disk (0 disables pruning).",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Optional explicit path for the final metrics JSON artifact.",
    )
    args = parser.parse_args()
    cfg = TrainConfig(
        train_path=args.train_path,
        val_path=args.val_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        export_dir=args.export_dir,
        max_length=args.max_length,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_dictionary_augmentation=args.use_dictionary_augmentation,
        metric_for_best_model=args.metric_for_best_model,
        use_class_weights=args.use_class_weights,
        class_weights=args.class_weights,
    )
    cfg.dictionary_doc_boost = not args.no_dictionary_doc_boost
    cfg.dynamic_padding = not args.no_dynamic_padding
    cfg.save_total_limit = int(args.save_total_limit)
    cfg.metrics_json_path = args.metrics_json
    return cfg


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
