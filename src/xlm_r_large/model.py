"""Model helpers for XLM-R large training and inference."""

from __future__ import annotations

from transformers import AutoConfig, AutoModelForSequenceClassification


def _load_sequence_classifier_sdpa_fallback(
    model_name: str, *, local_files_only: bool = False, **kwargs
):
    """Load with SDPA when supported; else retry without (older transformers / backends)."""
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            attn_implementation="sdpa",
            **kwargs,
        )
    except (TypeError, ValueError):
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            **kwargs,
        )


def build_model(
    num_labels: int = 115,
    model_name: str = "xlm-roberta-large",
    classifier_dropout: float = 0.3,
):
    """Build a plain HF sequence classifier for multi-label ICD coding."""
    return _load_sequence_classifier_sdpa_fallback(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        classifier_dropout=float(classifier_dropout),
    )


def load_model_for_inference(
    checkpoint_dir: str,
    num_labels: int = 115,
    local_files_only: bool = True,
):
    """Load a trained checkpoint for inference."""
    config = AutoConfig.from_pretrained(
        checkpoint_dir,
        local_files_only=local_files_only,
    )
    classifier_dropout = float(getattr(config, "classifier_dropout", 0.3) or 0.3)
    return _load_sequence_classifier_sdpa_fallback(
        checkpoint_dir,
        local_files_only=local_files_only,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        classifier_dropout=classifier_dropout,
    )
