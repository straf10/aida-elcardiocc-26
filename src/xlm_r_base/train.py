import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from collections import Counter, defaultdict
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, logging as transformers_logging

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from evaluation.config_utils import get_cfg, load_config
from evaluation.evaluator import evaluate_data
from evaluation.threshold_tune import tune_thresholds as tune_thresholds_group

transformers_logging.set_verbosity_error()
random.seed(42)
np.random.seed(42)


def load_data(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_labelset(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def load_label_descriptions(csv_path, labelset):
    desc = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            desc[row["code"]] = row["greek_description"]
    return [desc.get(l, l) for l in labelset]


def load_synonym_dict(csv_path):
    code_to_terms = defaultdict(list)
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            term = row["term"].strip()
            for code in row["codes_pipe_sep"].split("|"):
                code = code.strip()
                if term:
                    code_to_terms[code].append(term)
    return code_to_terms


def flatten_labels(r):
    return list({c for g in r["document_level_annotations"] for c in g})


def synonym_augment(text: str, rec_codes: list, code_to_terms: dict, p: float = 0.35) -> str:
    result = text
    for code in rec_codes:
        terms = code_to_terms.get(code, [])
        if len(terms) < 2:
            continue
        for term in terms:
            if random.random() > p:
                continue
            if len(term) < 4:
                continue
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            if pattern.search(result):
                replacement = random.choice([t for t in terms if t != term and len(t) >= 4])
                if replacement:
                    result = pattern.sub(replacement, result, count=1)
                break
    return result


def build_augmented_dataset(records: list, labelset: list, code_to_terms: dict) -> list:
    label_freq = Counter(c for r in records for g in r["document_level_annotations"] for c in g)
    aug_mult = {}
    for code in labelset:
        freq = label_freq.get(code, 0)
        if freq < 30:
            aug_mult[code] = 0
        elif freq < 100:
            aug_mult[code] = 4
        elif freq < 200:
            aug_mult[code] = 3
        elif freq < 300:
            aug_mult[code] = 2
        else:
            aug_mult[code] = 0

    augmented = list(records)
    for record in records:
        rec_codes = flatten_labels(record)
        multiplier = max((aug_mult.get(c, 0) for c in rec_codes), default=0)
        for _ in range(multiplier):
            new_text = synonym_augment(record["text"], rec_codes, code_to_terms)
            augmented.append(
                {
                    "patient_id": f"{record['patient_id']}_aug",
                    "document_level_annotations": record["document_level_annotations"],
                    "mention_level_annotations": record.get("mention_level_annotations", []),
                    "text": new_text,
                }
            )
    random.shuffle(augmented)
    return augmented


def build_cooccurrence_rules(records, min_cooccur=40, min_prob=0.45):
    freq = Counter(c for r in records for g in r["document_level_annotations"] for c in g)
    pair_freq = Counter()
    for r in records:
        codes = sorted(set(c for g in r["document_level_annotations"] for c in g))
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                pair_freq[(codes[i], codes[j])] += 1
    rules = {}
    for (a, b), co in pair_freq.items():
        if co >= min_cooccur:
            if co / max(freq[a], 1) >= min_prob:
                rules.setdefault(a, []).append((b, round(co / freq[a], 3)))
            if co / max(freq[b], 1) >= min_prob:
                rules.setdefault(b, []).append((a, round(co / freq[b], 3)))
    return rules


class ClinicalDataset(Dataset):
    def __init__(self, records, tokenizer, mlb, max_len=512):
        self.records = records
        self.tokenizer = tokenizer
        self.labels = mlb.transform([flatten_labels(r) for r in records]).astype(np.float32)
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.records[idx]["text"],
            max_length=self.max_len,
            truncation=True,
            truncation_side="left",
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


class MedicalModelWithDescriptions(nn.Module):
    def __init__(self, model_name: str, num_labels: int, label_descriptions: list, tokenizer, device):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        H = self.encoder.config.hidden_size
        self.dropouts = nn.ModuleList([nn.Dropout(0.1 * (i + 1)) for i in range(5)])
        self.classifier = nn.Linear(H, num_labels)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("desc_emb", self._encode_descriptions(label_descriptions, tokenizer, device, model_name))

    @torch.no_grad()
    def _encode_descriptions(self, descriptions, tokenizer, device, model_name):
        temp_encoder = AutoModel.from_pretrained(model_name).to(device)
        temp_encoder.eval()
        all_cls = []
        for desc in descriptions:
            enc = tokenizer(desc, max_length=64, truncation=True, padding="max_length", return_tensors="pt").to(device)
            out = temp_encoder(**enc)
            all_cls.append(out.last_hidden_state[:, 0, :].cpu())
        del temp_encoder
        torch.cuda.empty_cache()
        emb = torch.cat(all_cls, dim=0)
        return nn.functional.normalize(emb, dim=-1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        doc_cls = out.last_hidden_state[:, 0, :]
        base_logits = sum(self.classifier(dp(doc_cls)) for dp in self.dropouts) / len(self.dropouts)
        doc_cls_norm = nn.functional.normalize(doc_cls, dim=-1)
        desc_sim = doc_cls_norm @ self.desc_emb.T
        return base_logits + self.alpha * desc_sim


def compute_pos_weights(train_records, labelset, cap=20.0):
    n = len(train_records)
    l2i = {l: i for i, l in enumerate(labelset)}
    counts = np.zeros(len(labelset), dtype=np.float32)
    for r in train_records:
        for code in flatten_labels(r):
            if code in l2i:
                counts[l2i[code]] += 1
    pw = (n - counts) / np.maximum(counts, 1.0)
    return torch.tensor(np.clip(pw, 1.0, cap), dtype=torch.float)


def _proba_to_pred_data(proba: np.ndarray, val_pids: list, labelset: list, thresholds: np.ndarray) -> dict:
    """Map sigmoid matrix to flat code lists per patient (order matches val_pids)."""
    pred_data = {}
    n_labels = len(labelset)
    for i, pid in enumerate(val_pids):
        pred_codes = [labelset[j] for j in range(n_labels) if proba[i, j] >= thresholds[j]]
        pred_data[int(pid)] = pred_codes
    return pred_data


@torch.no_grad()
def evaluate(model, loader, device, val_recs: list, labelset: list, thresholds=None):
    """
    Validation micro-F1 using group-level ``evaluate_data`` / ``score_document``
    (same as shared task metric). Default threshold 0.5 per label for checkpointing.
    """
    n_labels = len(labelset)
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        lg = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        all_logits.append(lg.cpu())
        all_labels.append(batch["labels"])
    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_labels).numpy().astype(int)
    proba = torch.sigmoid(torch.tensor(logits)).numpy()
    if thresholds is None:
        thresholds = np.full(n_labels, 0.5)
    y_pred = (proba >= thresholds).astype(int)

    val_pids = [int(r["patient_id"]) for r in val_recs]
    ground_truth_data = {pid: r["document_level_annotations"] for pid, r in zip(val_pids, val_recs)}
    pred_data = _proba_to_pred_data(proba, val_pids, labelset, thresholds)

    metrics = evaluate_data(ground_truth_data, pred_data, label_space=labelset)
    micro_f1 = metrics["micro_f1"]
    avg_p = y_pred.sum(axis=1).mean()
    return micro_f1, proba, y_true, avg_p


ICD_HIERARCHY = {"I11": "I10", "I22": "I21"}


def apply_icd_hierarchy(y_pred: np.ndarray, labelset: list) -> np.ndarray:
    l2i = {l: i for i, l in enumerate(labelset)}
    result = y_pred.copy()
    for child, parent in ICD_HIERARCHY.items():
        if child not in l2i or parent not in l2i:
            continue
        ci, pi = l2i[child], l2i[parent]
        mask = (result[:, ci] == 1) & (result[:, pi] == 0)
        result[mask, pi] = 1
    return result


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    config = load_config(args.config) if args.config else {}
    threshold_min = (
        args.threshold_min
        if args.threshold_min is not None
        else float(get_cfg(config, "threshold_tuning.min", 0.05))
    )
    threshold_max = (
        args.threshold_max
        if args.threshold_max is not None
        else float(get_cfg(config, "threshold_tuning.max", 0.95))
    )
    threshold_step = (
        args.threshold_step
        if args.threshold_step is not None
        else float(get_cfg(config, "threshold_tuning.step", 0.01))
    )

    records = load_data(args.data)
    labelset = load_labelset(args.labels)
    mlb = MultiLabelBinarizer(classes=labelset).fit([labelset])
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    label_descs = load_label_descriptions(args.desc_csv, labelset)
    code_to_terms = load_synonym_dict(args.syn_csv)

    with open(os.path.join(args.out, "label_names.json"), "w", encoding="utf-8") as f:
        json.dump(labelset, f)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(records), 1):
        print(f"\n=== FOLD {fold}/{args.folds} ===")
        train_recs = [records[i] for i in train_idx]
        val_recs = [records[i] for i in val_idx]

        aug_train = build_augmented_dataset(train_recs, labelset, code_to_terms)

        train_ds = ClinicalDataset(aug_train, tokenizer, mlb, args.max_len)
        val_ds = ClinicalDataset(val_recs, tokenizer, mlb, args.max_len)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False, num_workers=2, pin_memory=True)

        model = MedicalModelWithDescriptions("xlm-roberta-base", len(labelset), label_descs, tokenizer, device).to(device)

        pw = compute_pos_weights(aug_train, labelset, cap=20.0).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        best_f1, best_proba, patience_ctr = 0.0, None, 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            tot_loss = 0.0
            t0 = time.time()
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                loss = loss_fn(logits, batch["labels"].to(device))
                (loss / args.grad_acc).backward()
                tot_loss += loss.item()

                if (step + 1) % args.grad_acc == 0 or (step + 1) == len(train_loader):
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            f1, proba, _, _ = evaluate(model, val_loader, device, val_recs, labelset, thresholds=None)
            print(
                f"  Fold {fold} Ep {epoch:2d} | loss={tot_loss/len(train_loader):.4f} | "
                f"group_micro_F1={f1:.4f} | {time.time()-t0:.0f}s"
            )

            if f1 > best_f1:
                best_f1, best_proba, patience_ctr = f1, proba.copy(), 0
                torch.save(model.state_dict(), os.path.join(args.out, f"model_fold_{fold-1}.pt"))
                print(f"    ★ Saved (group_micro_F1={best_f1:.4f}) | alpha={model.alpha.item():.4f}")
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    break

        val_pids = [int(r["patient_id"]) for r in val_recs]
        gt_val = {pid: r["document_level_annotations"] for pid, r in zip(val_pids, val_recs)}

        tuned_vec, tuned_f1 = tune_thresholds_group(
            scores=best_proba,
            patient_ids=val_pids,
            ground_truth_data=gt_val,
            label_names=labelset,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_step=threshold_step,
        )

        y_pred_raw = (best_proba >= tuned_vec).astype(int)
        y_pred_hier = apply_icd_hierarchy(y_pred_raw, labelset)
        pred_hier = {}
        for i, pid in enumerate(val_pids):
            pred_hier[int(pid)] = [labelset[j] for j in range(len(labelset)) if y_pred_hier[i, j]]
        f1_hier_metrics = evaluate_data(gt_val, pred_hier, label_space=labelset)
        f1_hier = f1_hier_metrics["micro_f1"]

        fold_thresh_path = os.path.join(args.out, f"thresholds_fold_{fold}.json")
        fold_payload = {
            "fold": fold,
            "best_val_group_micro_f1": float(best_f1),
            "tuned_val_micro_f1_in_sample": float(tuned_f1),
            "micro_f1_with_hierarchy": float(f1_hier),
            "note": "F1 is optimistic; thresholds selected on same val split.",
            "thresholds": {lab: float(tuned_vec[i]) for i, lab in enumerate(labelset)},
            "sweep": {"min": threshold_min, "max": threshold_max, "step": threshold_step},
        }
        with open(fold_thresh_path, "w", encoding="utf-8") as f:
            json.dump(fold_payload, f, indent=2)

        results.append(
            {
                "fold": fold,
                "thresholds": tuned_vec,
                "best_f1": best_f1,
                "f1_tuned": f1_hier,
                "tuned_f1_before_hier": tuned_f1,
            }
        )

    mean_vec = np.mean([r["thresholds"] for r in results], axis=0)
    final_path = os.path.join(args.out, "thresholds.json")
    final_payload = {
        "thresholds": {lab: float(mean_vec[i]) for i, lab in enumerate(labelset)},
        "per_fold_files": [f"thresholds_fold_{r['fold']}.json" for r in results],
        "mean_f1_tuned_with_hierarchy": float(np.mean([r["f1_tuned"] for r in results])),
        "mean_best_val_group_micro_f1": float(np.mean([r["best_f1"] for r in results])),
        "note": "Mean per-class thresholds across folds; in-sample tuning is optimistic.",
        "sweep": {"min": threshold_min, "max": threshold_max, "step": threshold_step},
    }
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    alt_thresholds = get_cfg(config, "output.thresholds_path")
    if alt_thresholds and os.path.abspath(alt_thresholds) != os.path.abspath(final_path):
        os.makedirs(os.path.dirname(alt_thresholds) or ".", exist_ok=True)
        with open(alt_thresholds, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2)

    tokenizer.save_pretrained(args.out)
    with open(os.path.join(args.out, "icd_hierarchy.json"), "w") as f:
        json.dump(ICD_HIERARCHY, f)

    # -------------------------------------------------------------------------
    # Μετά τον fold loop — rules για inference από ΟΛΟΚΛΗΡΟ το dataset
    # -------------------------------------------------------------------------
    print("\nBuilding final co-occurrence rules from the training dataset for inference...", flush=True)
    co_rules_full = build_cooccurrence_rules(records)
    with open(os.path.join(args.out, "cooccurrence_rules.json"), "w", encoding="utf-8") as f:
        json.dump(co_rules_full, f, ensure_ascii=False, indent=2)

    print(f"\nMean group_micro_F1 (tuned + hierarchy): {np.mean([r['f1_tuned'] for r in results]):.4f}")
    print(f"Saved thresholds to {final_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Optional YAML (threshold_tuning, output.thresholds_path)")
    p.add_argument("--data", default="training_set.jsonl")
    p.add_argument("--labels", default="labelset.txt")
    p.add_argument("--desc_csv", default="icd10_greek_lookup.csv")
    p.add_argument("--syn_csv", default="full_dictionary.csv")
    p.add_argument("--out", default="model_v15_base_fixed")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--grad_acc", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--threshold-min", dest="threshold_min", type=float, default=None)
    p.add_argument("--threshold-max", dest="threshold_max", type=float, default=None)
    p.add_argument("--threshold-step", dest="threshold_step", type=float, default=None)
    main(p.parse_args())
