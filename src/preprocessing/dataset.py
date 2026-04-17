import random
import torch
from torch.utils.data import Dataset

from .io_utils import load_jsonl, load_labelset

class ELCardioDataset(Dataset):
    def __init__(self, jsonl_path, labelset_path, tokenizer, max_length=512,
                 sliding_window=False, stride=256, is_training=False, chunk_strategy="random"):
        self.records = load_jsonl(jsonl_path)
        self.labels = load_labelset(labelset_path)
        self.label2idx = {l: i for i, l in enumerate(self.labels)}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sliding_window = sliding_window
        self.stride = stride
        self.is_training = is_training
        self.chunk_strategy = chunk_strategy

        self.chunks = []
        self.doc_to_chunks = []

        cls_token = self.tokenizer.cls_token_id
        sep_token = self.tokenizer.sep_token_id
        pad_token = self.tokenizer.pad_token_id

        for doc_idx, record in enumerate(self.records):
            text = record.get("text", "")
            patient_id = record.get("patient_id", -1)

            label_vector = torch.zeros(len(self.labels), dtype=torch.float)
            for code in record.get("labels_flat", []):
                if code in self.label2idx:
                    label_vector[self.label2idx[code]] = 1.0

            tokens = self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=False
            )["input_ids"]

            max_seq_len = self.max_length - 2

            doc_chunk_indices = []

            if not self.sliding_window or len(tokens) <= max_seq_len:
                chunk_tokens = tokens[:max_seq_len]
                input_ids = [cls_token] + chunk_tokens + [sep_token]
                attention_mask = [1] * len(input_ids)

                padding_length = self.max_length - len(input_ids)
                input_ids = input_ids + [pad_token] * padding_length
                attention_mask = attention_mask + [0] * padding_length

                chunk_idx = len(self.chunks)
                self.chunks.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": label_vector,
                    "patient_id": patient_id,
                    "doc_idx": doc_idx
                })
                doc_chunk_indices.append(chunk_idx)
            else:
                for i in range(0, len(tokens), self.stride):
                    chunk_tokens = tokens[i:i + max_seq_len]
                    input_ids = [cls_token] + chunk_tokens + [sep_token]
                    attention_mask = [1] * len(input_ids)

                    padding_length = self.max_length - len(input_ids)
                    input_ids = input_ids + [pad_token] * padding_length
                    attention_mask = attention_mask + [0] * padding_length

                    chunk_idx = len(self.chunks)
                    self.chunks.append({
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                        "labels": label_vector,
                        "patient_id": patient_id,
                        "doc_idx": doc_idx
                    })
                    doc_chunk_indices.append(chunk_idx)

                    if i + max_seq_len >= len(tokens):
                        break

            self.doc_to_chunks.append(doc_chunk_indices)

    def __len__(self):
        if self.sliding_window and self.is_training and self.chunk_strategy == "random":
            return len(self.records)
        return len(self.chunks)

    def __getitem__(self, idx):
        if self.sliding_window and self.is_training and self.chunk_strategy == "random":
            chunk_indices = self.doc_to_chunks[idx]
            sampled_chunk_idx = random.choice(chunk_indices)
            return self.chunks[sampled_chunk_idx]
        return self.chunks[idx]
