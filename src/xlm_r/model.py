import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class MultiSampleDropoutHead(nn.Module):
    """
    RoBERTa-style classification head with multi-sample dropout.
    Keeps dense/out_proj parameter names compatible with standard checkpoints.
    """

    def __init__(self, hidden_size, num_labels, dropout_rate=0.3, num_samples=5):
        super().__init__()
        self.num_samples = max(1, int(num_samples))
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(self.num_samples)]
        )
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        # HF sequence classifiers pass full sequence output [B, T, H].
        if features.dim() == 3:
            x = features[:, 0, :]  # CLS token
        else:
            x = features
        x = torch.tanh(self.dense(x))
        logits = torch.stack([self.out_proj(dp(x)) for dp in self.dropouts], dim=0)
        return logits.mean(dim=0)


class XLMRMeanPoolClassifier(nn.Module):
    """
    Mean-pooling classifier wrapper for XLM-R style backbones.
    """

    HEAD_STATE_FILENAME = "mean_pool_head.pt"
    MODEL_META_FILENAME = "custom_model_meta.json"

    def __init__(
        self,
        backbone,
        num_labels,
        dropout=0.3,
        multi_sample_dropout_samples=1,
    ):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.num_labels = int(num_labels)
        self.config = self.backbone.config
        self.config.num_labels = self.num_labels
        self.config.problem_type = "multi_label_classification"
        self.config.pooling_strategy = "mean"
        self.config.classifier_dropout = float(dropout)
        self.config.multi_sample_dropout_samples = int(multi_sample_dropout_samples)

        self.classifier = nn.Linear(self.backbone.config.hidden_size, self.num_labels)
        self.ms_dropouts = nn.ModuleList(
            [
                nn.Dropout(dropout)
                for _ in range(max(1, int(multi_sample_dropout_samples)))
            ]
        )

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        denom = torch.clamp(mask_expanded.sum(dim=1), min=1e-8)
        pooled = (token_embeddings * mask_expanded).sum(dim=1) / denom
        logits = torch.stack(
            [self.classifier(dp(self.dropout(pooled))) for dp in self.ms_dropouts],
            dim=0,
        ).mean(dim=0)
        return SequenceClassifierOutput(logits=logits)

    def save_pretrained(self, save_directory):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(str(save_dir))
        torch.save(
            {
                "classifier": self.classifier.state_dict(),
                "dropout_p": self.dropout.p,
                "num_labels": self.num_labels,
                "multi_sample_dropout_samples": len(self.ms_dropouts),
            },
            save_dir / self.HEAD_STATE_FILENAME,
        )
        with open(save_dir / self.MODEL_META_FILENAME, "w", encoding="utf-8") as handle:
            json.dump({"pooling_strategy": "mean"}, handle, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        num_labels=115,
        classifier_dropout=0.3,
        multi_sample_dropout_samples=1,
    ):
        backbone = AutoModel.from_pretrained(model_name_or_path)
        model = cls(
            backbone=backbone,
            num_labels=num_labels,
            dropout=classifier_dropout,
            multi_sample_dropout_samples=multi_sample_dropout_samples,
        )
        head_path = Path(model_name_or_path) / cls.HEAD_STATE_FILENAME
        if head_path.exists():
            payload = torch.load(head_path, map_location="cpu")
            model.classifier.load_state_dict(payload.get("classifier", {}))
        return model


def _build_hf_sequence_classifier(
    model_name,
    num_labels,
    classifier_dropout=0.3,
    multi_sample_dropout_samples=1,
):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        classifier_dropout=classifier_dropout,
    )
    if int(multi_sample_dropout_samples) > 1:
        original_head = model.classifier
        hidden_size = model.config.hidden_size
        new_head = MultiSampleDropoutHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout_rate=classifier_dropout,
            num_samples=multi_sample_dropout_samples,
        )
        if hasattr(original_head, "dense") and hasattr(original_head, "out_proj"):
            new_head.dense.load_state_dict(original_head.dense.state_dict())
            new_head.out_proj.load_state_dict(original_head.out_proj.state_dict())
        model.classifier = new_head
    model.config.pooling_strategy = "cls"
    model.config.multi_sample_dropout_samples = int(multi_sample_dropout_samples)
    return model


def build_model(
    num_labels=115,
    model_name="xlm-roberta-large",
    classifier_dropout=0.3,
    pooling_strategy="cls",
    multi_sample_dropout_samples=1,
):
    if pooling_strategy == "mean":
        backbone = AutoModel.from_pretrained(model_name)
        return XLMRMeanPoolClassifier(
            backbone=backbone,
            num_labels=num_labels,
            dropout=classifier_dropout,
            multi_sample_dropout_samples=multi_sample_dropout_samples,
        )
    return _build_hf_sequence_classifier(
        model_name=model_name,
        num_labels=num_labels,
        classifier_dropout=classifier_dropout,
        multi_sample_dropout_samples=multi_sample_dropout_samples,
    )


def compute_pos_weights(labelset, frequencies_path, num_train_docs):
    with open(frequencies_path, "r", encoding="utf-8") as f:
        freqs = json.load(f)

    freq_dict = {code: count for code, count in freqs}

    pos_weights = torch.zeros(len(labelset), dtype=torch.float)
    MAX_POS_WEIGHT = 50.0
    for i, code in enumerate(labelset):
        pos_count = freq_dict.get(code, 0)
        if pos_count > 0:
            pos_weights[i] = min((num_train_docs - pos_count) / pos_count, MAX_POS_WEIGHT)
        else:
            pos_weights[i] = 1.0

    return pos_weights


def load_model_for_inference(checkpoint_dir, num_labels=115):
    config = AutoConfig.from_pretrained(checkpoint_dir)
    pooling_strategy = getattr(config, "pooling_strategy", "cls")
    classifier_dropout = float(getattr(config, "classifier_dropout", 0.3))
    multi_sample_dropout_samples = int(
        getattr(config, "multi_sample_dropout_samples", 1)
    )

    if pooling_strategy == "mean":
        return XLMRMeanPoolClassifier.from_pretrained(
            checkpoint_dir,
            num_labels=num_labels,
            classifier_dropout=classifier_dropout,
            multi_sample_dropout_samples=multi_sample_dropout_samples,
        )
    return _build_hf_sequence_classifier(
        model_name=checkpoint_dir,
        num_labels=num_labels,
        classifier_dropout=classifier_dropout,
        multi_sample_dropout_samples=multi_sample_dropout_samples,
    )
