import random
import torch
from torch.utils.data import Dataset

from .io_utils import load_jsonl, load_labelset


class ELCardioDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        labelset_path,
        tokenizer,
        max_length=512,
        sliding_window=False,
        stride=256,
        is_training=False,
        chunk_strategy="random",
        truncation_side="right",
        return_groups=False,
    ):
        self.records = load_jsonl(jsonl_path)
        self.labels = load_labelset(labelset_path)
        self.label2idx = {l: i for i, l in enumerate(self.labels)}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sliding_window = sliding_window
        self.stride = stride
        self.is_training = is_training
        self.chunk_strategy = chunk_strategy
        self.truncation_side = (
            truncation_side if truncation_side in ("left", "right") else "right"
        )
        self.return_groups = return_groups

        self.doc_to_chunks: list[list[int]] = []
        n_labels = len(self.labels)

        cls_token = self.tokenizer.cls_token_id
        sep_token = self.tokenizer.sep_token_id
        pad_token = self.tokenizer.pad_token_id
        if cls_token is None or sep_token is None or pad_token is None:
            raise ValueError("Tokenizer must define cls_token, sep_token, and pad_token.")

        texts = [r.get("text", "") for r in self.records]
        enc = self.tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
        )
        all_token_ids = enc["input_ids"]

        max_seq_len = self.max_length - 2

        # Pass 1: count total chunks
        n_chunks = 0
        doc_chunk_counts: list[int] = []
        for record in self.records:
            doc_idx = len(doc_chunk_counts)
            tokens = all_token_ids[doc_idx]
            m = len(tokens)
            if not self.sliding_window or m <= max_seq_len:
                doc_chunk_counts.append(1)
                n_chunks += 1
            else:
                c = 0
                for i in range(0, m, self.stride):
                    c += 1
                    if i + max_seq_len >= m:
                        break
                doc_chunk_counts.append(c)
                n_chunks += c

        self._input_ids = torch.full(
            (n_chunks, self.max_length), int(pad_token), dtype=torch.long
        )
        self._attention_mask = torch.zeros(
            (n_chunks, self.max_length), dtype=torch.long
        )
        self._labels = torch.zeros((n_chunks, n_labels), dtype=torch.float)
        self._patient_id = torch.empty((n_chunks,), dtype=torch.long)
        self._doc_idx = torch.empty((n_chunks,), dtype=torch.long)
        self._groups_per_chunk: list | None = [] if self.return_groups else None

        row = 0
        for doc_idx, record in enumerate(self.records):
            tokens = all_token_ids[doc_idx]
            patient_id = int(record.get("patient_id", -1))

            label_vector = torch.zeros(n_labels, dtype=torch.float)
            for code in record.get("labels_flat", []):
                if code in self.label2idx:
                    label_vector[self.label2idx[code]] = 1.0

            groups_for_doc = (
                self._groups_from_record(record) if self.return_groups else None
            )
            m = len(tokens)
            doc_chunk_indices: list[int] = []

            if not self.sliding_window or m <= max_seq_len:
                if m <= max_seq_len:
                    chunk_tokens = tokens
                elif self.truncation_side == "left":
                    chunk_tokens = tokens[-max_seq_len:]
                else:
                    chunk_tokens = tokens[:max_seq_len]
                self._write_chunk_row(
                    row,
                    chunk_tokens,
                    label_vector,
                    patient_id,
                    doc_idx,
                    groups_for_doc,
                    cls_token,
                    sep_token,
                    pad_token,
                )
                doc_chunk_indices.append(row)
                row += 1
            else:
                for i in range(0, m, self.stride):
                    chunk_tokens = tokens[i : i + max_seq_len]
                    self._write_chunk_row(
                        row,
                        chunk_tokens,
                        label_vector,
                        patient_id,
                        doc_idx,
                        groups_for_doc,
                        cls_token,
                        sep_token,
                        pad_token,
                    )
                    doc_chunk_indices.append(row)
                    row += 1
                    if i + max_seq_len >= m:
                        break

            self.doc_to_chunks.append(doc_chunk_indices)

        if row != n_chunks:
            raise RuntimeError(
                f"Chunk count mismatch: expected {n_chunks} rows, wrote {row}."
            )

    def _write_chunk_row(
        self,
        row: int,
        chunk_tokens: list,
        label_vector: torch.Tensor,
        patient_id: int,
        doc_idx: int,
        groups_for_doc: list | None,
        cls_token: int,
        sep_token: int,
        pad_token: int,
    ) -> None:
        input_ids = [cls_token] + list(chunk_tokens) + [sep_token]
        attention = [1] * len(input_ids)
        pad_len = self.max_length - len(input_ids)
        input_ids = input_ids + [pad_token] * pad_len
        attention = attention + [0] * pad_len
        self._input_ids[row] = torch.tensor(input_ids, dtype=torch.long)
        self._attention_mask[row] = torch.tensor(attention, dtype=torch.long)
        self._labels[row] = label_vector
        self._patient_id[row] = patient_id
        self._doc_idx[row] = doc_idx
        if self._groups_per_chunk is not None:
            self._groups_per_chunk.append(groups_for_doc)

    def _row_dict(self, chunk_idx: int) -> dict:
        out: dict = {
            "input_ids": self._input_ids[chunk_idx].clone(),
            "attention_mask": self._attention_mask[chunk_idx].clone(),
            "labels": self._labels[chunk_idx].clone(),
            "patient_id": int(self._patient_id[chunk_idx].item()),
            "doc_idx": int(self._doc_idx[chunk_idx].item()),
        }
        if self._groups_per_chunk is not None:
            out["groups"] = self._groups_per_chunk[chunk_idx]
        return out

    def _groups_from_record(self, record):
        """List of gold synonym groups as lists of label indices (OR-eval semantics)."""
        ann = record.get("document_level_annotations")
        if ann:
            out = []
            for g in ann:
                idxs = [self.label2idx[c] for c in g if c in self.label2idx]
                if idxs:
                    out.append(idxs)
            return out
        lf = record.get("labels_flat") or []
        singletons = [[self.label2idx[c]] for c in lf if c in self.label2idx]
        return singletons

    def __len__(self):
        if self.sliding_window and self.is_training and self.chunk_strategy == "random":
            return len(self.records)
        return self._input_ids.shape[0]

    def __getitem__(self, idx):
        if self.sliding_window and self.is_training and self.chunk_strategy == "random":
            chunk_indices = self.doc_to_chunks[idx]
            sampled = random.choice(chunk_indices)
            return self._row_dict(sampled)
        return self._row_dict(idx)
