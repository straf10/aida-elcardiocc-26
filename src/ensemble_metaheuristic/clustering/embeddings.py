"""Mean-pooled transformer embeddings for raw texts (ensemble text-cluster strategy)."""
from __future__ import annotations

from typing import List

import numpy as np


class _TextDataset:
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int):
        return {key: val[idx] for key, val in self.encodings.items()}


def embed_texts(
    texts: List[str],
    model_name: str,
    max_length: int,
    batch_size: int,
    device,
) -> np.ndarray:
    """Mean-pooled last hidden states (mask-aware) using a HuggingFace ``AutoModel``."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    dataset = _TextDataset(texts, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            chunks.append(mean_pooled.cpu().numpy())

    return np.vstack(chunks).astype(np.float64)
