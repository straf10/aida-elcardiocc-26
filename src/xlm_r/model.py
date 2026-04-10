import json

import torch
from transformers import AutoModelForSequenceClassification


def build_model(num_labels=115, model_name="xlm-roberta-large"):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )
    return model


def compute_pos_weights(labelset, frequencies_path, num_train_docs):
    with open(frequencies_path, "r", encoding="utf-8") as f:
        freqs = json.load(f)

    freq_dict = {code: count for code, count in freqs}

    pos_weights = torch.zeros(len(labelset), dtype=torch.float)
    for i, code in enumerate(labelset):
        pos_count = freq_dict.get(code, 0)
        if pos_count > 0:
            pos_weights[i] = (num_train_docs - pos_count) / pos_count
        else:
            pos_weights[i] = 1.0

    return pos_weights


def load_model_for_inference(checkpoint_dir, num_labels=115):
    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )
