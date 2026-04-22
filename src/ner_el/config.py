from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from evaluation.config_utils import get_cfg, load_config


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
    use_partial_crf: bool = False
    partial_all: bool = False
    use_reranker: bool = False
    reranker_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    reranker_alpha: float = 0.6
    reranker_window_chars: int = 200
    reranker_artifact_dir: str = "outputs/models/ner_el"


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
    use_partial_crf: bool = False
    use_dictionary_fusion: bool = True
    dictionary_doc_boost: bool = True
    use_reranker: bool = False
    reranker_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    reranker_alpha: float = 0.6
    reranker_window_chars: int = 200
    reranker_artifact_dir: str = "outputs/models/ner_el"

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PredictConfig":
        model_block = raw.get("model", {}) if isinstance(raw.get("model"), dict) else {}
        prediction_block = raw.get("prediction", {}) if isinstance(raw.get("prediction"), dict) else {}
        linker_block = raw.get("linker", {}) if isinstance(raw.get("linker"), dict) else {}
        training_block = raw.get("training", {}) if isinstance(raw.get("training"), dict) else {}
        return cls(
            model_dir=str(raw.get("model_dir", prediction_block.get("model_dir", cls.model_dir))),
            tokenizer_name=str(raw.get("tokenizer_name", model_block.get("name", cls.tokenizer_name))),
            input_path=str(raw.get("input_path", prediction_block.get("input_path", cls.input_path))),
            train_path_for_linker=raw.get(
                "train_path_for_linker",
                prediction_block.get("train_path_for_linker"),
            ),
            output_doc_path=str(
                raw.get("output_doc_path", prediction_block.get("output_doc_path", cls.output_doc_path))
            ),
            output_debug_path=str(
                raw.get("output_debug_path", prediction_block.get("output_debug_path", cls.output_debug_path))
            ),
            max_length=int(raw.get("max_length", training_block.get("max_length", cls.max_length))),
            use_partial_crf=bool(
                raw.get("use_partial_crf", training_block.get("use_partial_crf", cls.use_partial_crf))
            ),
            use_dictionary_fusion=bool(
                raw.get(
                    "use_dictionary_fusion",
                    prediction_block.get("use_dictionary_fusion", cls.use_dictionary_fusion),
                )
            ),
            dictionary_doc_boost=bool(
                raw.get(
                    "dictionary_doc_boost",
                    prediction_block.get("dictionary_doc_boost", cls.dictionary_doc_boost),
                )
            ),
            use_reranker=bool(raw.get("use_reranker", linker_block.get("use_reranker", cls.use_reranker))),
            reranker_model=str(
                raw.get("reranker_model", linker_block.get("reranker_model", cls.reranker_model))
            ),
            reranker_alpha=float(
                raw.get("reranker_alpha", linker_block.get("alpha", cls.reranker_alpha))
            ),
            reranker_window_chars=int(
                raw.get(
                    "reranker_window_chars",
                    linker_block.get("window_chars", cls.reranker_window_chars),
                )
            ),
            reranker_artifact_dir=str(
                raw.get(
                    "reranker_artifact_dir",
                    linker_block.get("artifact_dir", cls.reranker_artifact_dir),
                )
            ),
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
    parser.add_argument("--config", default=None, help="Optional YAML config path")
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--val-path", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--use-dictionary-augmentation", dest="use_dictionary_augmentation", action="store_true")
    parser.add_argument("--no-dictionary-augmentation", dest="use_dictionary_augmentation", action="store_false")
    parser.add_argument("--dictionary-doc-boost", dest="dictionary_doc_boost", action="store_true")
    parser.add_argument("--no-dictionary-doc-boost", dest="dictionary_doc_boost", action="store_false")
    parser.add_argument("--dynamic-padding", dest="dynamic_padding", action="store_true")
    parser.add_argument("--no-dynamic-padding", dest="dynamic_padding", action="store_false")
    parser.add_argument("--metric-for-best-model", default=None)
    parser.add_argument("--use-class-weights", dest="use_class_weights", action="store_true")
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")
    parser.add_argument("--use-partial-crf", dest="use_partial_crf", action="store_true")
    parser.add_argument("--no-partial-crf", dest="use_partial_crf", action="store_false")
    parser.add_argument("--partial-all", dest="partial_all", action="store_true")
    parser.add_argument("--no-partial-all", dest="partial_all", action="store_false")
    parser.add_argument("--use-reranker", dest="use_reranker", action="store_true")
    parser.add_argument("--no-reranker", dest="use_reranker", action="store_false")
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--reranker-alpha", type=float, default=None)
    parser.add_argument("--reranker-window-chars", type=int, default=None)
    parser.add_argument("--reranker-artifact-dir", default=None)
    parser.add_argument(
        "--class-weights",
        type=_parse_class_weights,
        default=None,
        help="Comma-separated weights for labels O,B-MED,I-MED",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help="Max number of epoch checkpoints to keep on disk (0 disables pruning).",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Optional explicit path for the final metrics JSON artifact.",
    )
    parser.set_defaults(
        use_dictionary_augmentation=None,
        dictionary_doc_boost=None,
        dynamic_padding=None,
        use_class_weights=None,
        use_partial_crf=None,
        partial_all=None,
        use_reranker=None,
    )
    args = parser.parse_args()
    cfg_file = load_config(args.config)

    def pick(cli_value: Any, dotted_key: str, default: Any) -> Any:
        if cli_value is not None:
            return cli_value
        return get_cfg(cfg_file, dotted_key, default)

    cfg = TrainConfig(
        train_path=str(pick(args.train_path, "data.train_path", TrainConfig.train_path)),
        val_path=str(pick(args.val_path, "data.val_path", TrainConfig.val_path)),
        model_name=str(pick(args.model_name, "model.name", TrainConfig.model_name)),
        output_dir=str(pick(args.output_dir, "output.output_dir", TrainConfig.output_dir)),
        export_dir=str(pick(args.export_dir, "output.export_dir", TrainConfig.export_dir)),
        max_length=int(pick(args.max_length, "training.max_length", TrainConfig.max_length)),
        epochs=int(pick(args.epochs, "training.epochs", TrainConfig.epochs)),
        train_batch_size=int(
            pick(args.train_batch_size, "training.train_batch_size", TrainConfig.train_batch_size)
        ),
        eval_batch_size=int(
            pick(args.eval_batch_size, "training.eval_batch_size", TrainConfig.eval_batch_size)
        ),
        learning_rate=float(
            pick(args.learning_rate, "training.learning_rate", TrainConfig.learning_rate)
        ),
        weight_decay=float(
            pick(args.weight_decay, "training.weight_decay", TrainConfig.weight_decay)
        ),
        use_dictionary_augmentation=bool(
            pick(
                args.use_dictionary_augmentation,
                "training.use_dictionary_augmentation",
                TrainConfig.use_dictionary_augmentation,
            )
        ),
        dictionary_doc_boost=bool(
            pick(
                args.dictionary_doc_boost,
                "training.dictionary_doc_boost",
                TrainConfig.dictionary_doc_boost,
            )
        ),
        dynamic_padding=bool(
            pick(args.dynamic_padding, "training.dynamic_padding", TrainConfig.dynamic_padding)
        ),
        metric_for_best_model=str(
            pick(
                args.metric_for_best_model,
                "training.metric_for_best_model",
                TrainConfig.metric_for_best_model,
            )
        ),
        use_class_weights=bool(
            pick(args.use_class_weights, "training.use_class_weights", TrainConfig.use_class_weights)
        ),
        class_weights=pick(args.class_weights, "training.class_weights", TrainConfig.class_weights),
        save_total_limit=int(
            pick(args.save_total_limit, "training.save_total_limit", TrainConfig.save_total_limit)
        ),
        metrics_json_path=pick(args.metrics_json, "output.metrics_json_path", None),
        use_partial_crf=bool(
            pick(args.use_partial_crf, "training.use_partial_crf", TrainConfig.use_partial_crf)
        ),
        partial_all=bool(
            pick(args.partial_all, "training.partial_all", TrainConfig.partial_all)
        ),
        use_reranker=bool(
            pick(args.use_reranker, "linker.use_reranker", TrainConfig.use_reranker)
        ),
        reranker_model=str(
            pick(args.reranker_model, "linker.reranker_model", TrainConfig.reranker_model)
        ),
        reranker_alpha=float(
            pick(args.reranker_alpha, "linker.alpha", TrainConfig.reranker_alpha)
        ),
        reranker_window_chars=int(
            pick(
                args.reranker_window_chars,
                "linker.window_chars",
                TrainConfig.reranker_window_chars,
            )
        ),
        reranker_artifact_dir=str(
            pick(
                args.reranker_artifact_dir,
                "linker.artifact_dir",
                TrainConfig.reranker_artifact_dir,
            )
        ),
    )
    return cfg


def parse_predict_args() -> PredictConfig:
    parser = argparse.ArgumentParser(description="Run NER+EL inference")
    parser.add_argument("--config", default=None, help="Optional YAML config path")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--train-path-for-linker", default=None)
    parser.add_argument("--output-doc-path", default=None)
    parser.add_argument("--output-debug-path", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--use-partial-crf", dest="use_partial_crf", action="store_true")
    parser.add_argument("--no-partial-crf", dest="use_partial_crf", action="store_false")
    parser.add_argument("--use-dictionary-fusion", dest="use_dictionary_fusion", action="store_true")
    parser.add_argument("--no-dictionary-fusion", dest="use_dictionary_fusion", action="store_false")
    parser.add_argument("--dictionary-doc-boost", dest="dictionary_doc_boost", action="store_true")
    parser.add_argument("--no-dictionary-doc-boost", dest="dictionary_doc_boost", action="store_false")
    parser.add_argument("--use-reranker", dest="use_reranker", action="store_true")
    parser.add_argument("--no-reranker", dest="use_reranker", action="store_false")
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--reranker-alpha", type=float, default=None)
    parser.add_argument("--reranker-window-chars", type=int, default=None)
    parser.add_argument("--reranker-artifact-dir", default=None)
    parser.set_defaults(
        use_partial_crf=None,
        use_dictionary_fusion=None,
        dictionary_doc_boost=None,
        use_reranker=None,
    )
    args = parser.parse_args()
    cfg_file = load_config(args.config)
    cfg = PredictConfig.from_dict(cfg_file)

    if args.model_dir is not None:
        cfg.model_dir = args.model_dir
    if args.tokenizer_name is not None:
        cfg.tokenizer_name = args.tokenizer_name
    if args.input_path is not None:
        cfg.input_path = args.input_path
    if args.train_path_for_linker is not None:
        cfg.train_path_for_linker = args.train_path_for_linker
    if args.output_doc_path is not None:
        cfg.output_doc_path = args.output_doc_path
    if args.output_debug_path is not None:
        cfg.output_debug_path = args.output_debug_path
    if args.max_length is not None:
        cfg.max_length = args.max_length
    if args.use_partial_crf is not None:
        cfg.use_partial_crf = bool(args.use_partial_crf)
    if args.use_dictionary_fusion is not None:
        cfg.use_dictionary_fusion = bool(args.use_dictionary_fusion)
    if args.dictionary_doc_boost is not None:
        cfg.dictionary_doc_boost = bool(args.dictionary_doc_boost)
    if args.use_reranker is not None:
        cfg.use_reranker = bool(args.use_reranker)
    if args.reranker_model is not None:
        cfg.reranker_model = args.reranker_model
    if args.reranker_alpha is not None:
        cfg.reranker_alpha = float(args.reranker_alpha)
    if args.reranker_window_chars is not None:
        cfg.reranker_window_chars = int(args.reranker_window_chars)
    if args.reranker_artifact_dir is not None:
        cfg.reranker_artifact_dir = args.reranker_artifact_dir
    return cfg
