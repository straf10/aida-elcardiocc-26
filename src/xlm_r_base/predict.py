import argparse
import json
import os
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer

try:
    from src.evaluation.config_utils import get_cfg, load_config
    from src.preprocessing.io_utils import load_jsonl
    from src.xlm_r_base.train import (
        MedicalModelWithDescriptions,
        load_labelset,
        load_label_descriptions,
    )
except ImportError:
    from ..evaluation.config_utils import get_cfg, load_config
    from ..preprocessing.io_utils import load_jsonl
    from .train import (
        MedicalModelWithDescriptions,
        load_labelset,
        load_label_descriptions,
    )


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, records, tokenizer, max_len):
        self.records = records
        self.tokenizer = tokenizer
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
            "patient_id": self.records[idx].get("patient_id", -1),
        }


def main():
    parser = argparse.ArgumentParser(description="XLM-R Base MLC inference")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--split", required=True, choices=["val", "test"], help="Which split to predict on")
    parser.add_argument("--fold", type=int, default=0, help="Which fold model to load")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load paths
    checkpoint_dir = get_cfg(config, "output.checkpoint_dir", "outputs/model_v15_base")
    scores_path = get_cfg(config, "output.scores_path", f"{checkpoint_dir}/val_scores.npy")
    pids_path = get_cfg(config, "output.patient_ids_path", f"{checkpoint_dir}/val_patient_ids.json")
    label_names_path = get_cfg(config, "output.label_names_path", f"{checkpoint_dir}/label_names.json")
    thresholds_out_path = get_cfg(config, "output.thresholds_path", f"{checkpoint_dir}/thresholds.json")
    
    labelset_path = get_cfg(config, "data.labelset_path", "data/raw/Train_Set_2026/labelset.txt")
    desc_csv = get_cfg(config, "data.desc_csv", "data/external/icd10_greek_lookup.csv")
    max_length = get_cfg(config, "data.max_length", 512)
    batch_size = get_cfg(config, "training.batch_size", 16)
    
    if args.split == "val":
        data_path = get_cfg(config, "data.val_path", "data/processed/validation_set.jsonl")
    else:
        data_path = get_cfg(config, "data.test_path", "data/raw/Test_Set_2026/test_set.jsonl")

    # Load resources
    print(f"Loading data from {data_path}...")
    records = load_jsonl(data_path)
    labelset = load_labelset(labelset_path)
    label_descs = load_label_descriptions(desc_csv, labelset)

    print(f"Loading tokenizer from {checkpoint_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    
    dataset = SimpleDataset(records, tokenizer, max_length)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"Building model (fold {args.fold})...")
    model = MedicalModelWithDescriptions(
        "xlm-roberta-base", len(labelset), label_descs, tokenizer, device
    ).to(device)
    
    pt_path = os.path.join(checkpoint_dir, f"model_fold_{args.fold}.pt")
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()

    all_logits = []
    all_pids = []

    print("Running inference...")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pids = batch["patient_id"]
            
            logits = model(input_ids, attention_mask)
            all_logits.append(logits.cpu())
            all_pids.extend(pids.tolist())

    scores = torch.sigmoid(torch.cat(all_logits, dim=0)).numpy()
    
    if args.split == "val":
        print("Exporting validation artifacts...")
        os.makedirs(os.path.dirname(scores_path), exist_ok=True)
        np.save(scores_path, scores)
        
        with open(pids_path, "w", encoding="utf-8") as f:
            json.dump(all_pids, f)
            
        with open(label_names_path, "w", encoding="utf-8") as f:
            json.dump(labelset, f)
            
        # Export thresholds if available from training
        avg_thresh_path = os.path.join(checkpoint_dir, "avg_thresholds.npy")
        if os.path.exists(avg_thresh_path):
            thresholds_array = np.load(avg_thresh_path)
            thresh_dict = {
                "best_micro_f1": 0.0,
                "thresholds": {label: float(th) for label, th in zip(labelset, thresholds_array)}
            }
            with open(thresholds_out_path, "w", encoding="utf-8") as f:
                json.dump(thresh_dict, f, indent=2)
                
    print("Done.")

if __name__ == "__main__":
    main()
