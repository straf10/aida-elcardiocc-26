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

    def __init__(
        self,
        model_name: str,
        num_labels: int = 115,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        pooling: str = "cls",
        head: str = "linear",
        head_hidden_dim: int | None = None,
    ):
        super().__init__()
        try:
            self.encoder = AutoModel.from_pretrained(
                model_name, attn_implementation="sdpa"
            )
        except (TypeError, ValueError, RuntimeError, OSError):
            self.encoder = AutoModel.from_pretrained(model_name)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.pooling = str(pooling).lower()
        self.head = str(head).lower()
        if self.pooling not in {"cls", "mean", "mean_cls"}:
            raise ValueError(
                f"Unsupported pooling '{pooling}'. Use 'cls', 'mean', or 'mean_cls'."
            )
        if self.head not in {"linear", "mlp"}:
            raise ValueError(f"Unsupported head '{head}'. Use 'linear' or 'mlp'.")

        feat_dim = hidden_size * 2 if self.pooling == "mean_cls" else hidden_size
        if self.head == "linear":
            # Keep a plain linear layer for backward-compatible state_dict keys.
            self.classifier = nn.Linear(feat_dim, num_labels)
        else:
            inner_dim = int(head_hidden_dim or hidden_size)
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, inner_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, num_labels),
            )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs.last_hidden_state
        cls_vec = hidden[:, 0, :]
        if self.pooling == "cls":
            pooled = cls_vec
        else:
            mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            mean_vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            if self.pooling == "mean":
                pooled = mean_vec
            else:
                pooled = torch.cat([cls_vec, mean_vec], dim=-1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits  # shape: (batch_size, num_labels)