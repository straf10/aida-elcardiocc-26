"""
Multi-Label Classification model for ELCardioCC 2026.
Owner: Vasiliki
Track: Greek-BERT (nlpaueb/bert-base-greek-uncased-v1)
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class MLCModel(nn.Module):
    """
    Greek-BERT with a linear classification head for multi-label ICD-10 coding.
    Outputs raw logits (115,) — apply sigmoid externally for probabilities.
    """

    def __init__(self, model_name: str, num_labels: int = 115, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits  # shape: (batch_size, num_labels)