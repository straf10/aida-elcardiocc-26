from __future__ import annotations

from transformers import AutoModelForTokenClassification

from .bio_dataset import ID2LABEL, LABEL2ID


def build_ner_model(model_name: str):
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
