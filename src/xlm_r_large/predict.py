import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from src.preprocessing.dataset import ELCardioDataset
    from src.preprocessing.io_utils import save_jsonl
    from src.evaluation.config_utils import get_cfg, load_config
    from src.evaluation.evaluator import evaluate_data
    from src.evaluation.io_utils import load_ground_truth
    from src.training_validation.device_utils import get_device, use_amp_fp16
except ImportError:
    from ..preprocessing.dataset import ELCardioDataset
    from ..preprocessing.io_utils import save_jsonl
    from ..evaluation.config_utils import get_cfg, load_config
    from ..evaluation.evaluator import evaluate_data
    from ..evaluation.io_utils import load_ground_truth
    from ..training_validation.device_utils import get_device, use_amp_fp16

from .chunk_aggregate import aggregate_scores_by_patient
from .model import load_model_for_inference
from .postprocess import apply_specific_parent_child


def main():
    parser = argparse.ArgumentParser(description="XLM-R MLC inference")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--split",
        required=True,
        choices=["val", "test"],
        help="Which split to predict on",
    )
    parser.add_argument(
        "--thresholds",
        help="Path to tuned thresholds JSON (required for test split)",
    )
    parser.add_argument(
        "--device",
        help="Explicit device to use (e.g., 'cpu', 'cuda', 'mps')",
    )
    parser.add_argument(
        "--apply-parent-child",
        action="store_true",
        help="Add specific→specific parent codes (e.g. I11→I10) after thresholding.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    num_labels = get_cfg(config, "model.num_labels", 115)

    if args.split == "val":
        data_path = get_cfg(config, "data.val_path")
    else:
        data_path = get_cfg(
            config, "data.test_path", "data/raw/Test_Set_2026/test_set.jsonl"
        )

    labelset_path = get_cfg(config, "data.labelset_path")
    max_length = get_cfg(config, "data.max_length", 512)
    sliding_window = get_cfg(config, "data.sliding_window", False)
    stride = get_cfg(config, "data.stride", 256)
    truncation_side = get_cfg(config, "data.truncation_side", "right")
    batch_size = get_cfg(config, "training.batch_size", 8)
    fp16 = get_cfg(config, "training.fp16", True)
    aggregation_strategy = get_cfg(config, "training.aggregation_strategy", "max")
    aggregation_temperature = get_cfg(config, "training.aggregation_temperature", 1.0)

    checkpoint_dir = get_cfg(config, "output.checkpoint_dir", "outputs/experiments/xlm_r_large/checkpoints")
    scores_path = get_cfg(config, "output.scores_path", "outputs/experiments/xlm_r_large/val_scores.npy")
    pids_path = get_cfg(config, "output.patient_ids_path", "outputs/experiments/xlm_r_large/val_patient_ids.json")
    label_names_path = get_cfg(config, "output.label_names_path", "outputs/experiments/xlm_r_large/label_names.json")

    device = get_device(args.device)
    use_amp = use_amp_fp16(device, fp16)
    print(f"Using device: {device} | AMP (fp16): {use_amp}")

    print(f"Loading model from {checkpoint_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = load_model_for_inference(checkpoint_dir, num_labels=num_labels)
    model.to(device)
    model.eval()

    print(f"Loading dataset from {data_path}...")
    dataset = ELCardioDataset(
        data_path,
        labelset_path,
        tokenizer,
        max_length=max_length,
        sliding_window=sliding_window,
        stride=stride,
        is_training=False,
        truncation_side=truncation_side,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    pid_to_logits = {}

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pids = batch["patient_id"].tolist()

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.cpu().numpy()

            for i, pid in enumerate(pids):
                if pid not in pid_to_logits:
                    pid_to_logits[pid] = []
                pid_to_logits[pid].append(logits[i])

    unique_pids, aggregated_scores = aggregate_scores_by_patient(
        pid_to_logits,
        strategy=aggregation_strategy,
        temperature=aggregation_temperature,
    )

    if args.split == "val":
        print("Exporting validation artifacts for threshold tuning...")
        os.makedirs(os.path.dirname(scores_path), exist_ok=True)
        np.save(scores_path, aggregated_scores)
        with open(pids_path, "w", encoding="utf-8") as f:
            json.dump(unique_pids, f)
        with open(label_names_path, "w", encoding="utf-8") as f:
            json.dump(dataset.labels, f)

        print("Evaluating with threshold 0.5 for reference...")
        preds_bin = aggregated_scores >= 0.5
        pred_data = {}
        for i, pid in enumerate(unique_pids):
            pred_indices = np.where(preds_bin[i])[0]
            pred_data[pid] = [dataset.labels[idx] for idx in pred_indices]

        ground_truth_data = load_ground_truth(data_path)
        metrics = evaluate_data(
            ground_truth_data, pred_data, label_space=dataset.labels
        )
        print(f"Val Micro-F1 (thresh=0.5): {metrics['micro_f1']:.4f}")

    elif args.split == "test":
        if not args.thresholds:
            raise ValueError("--thresholds is required for test split")

        print(f"Loading tuned thresholds from {args.thresholds}...")
        with open(args.thresholds, "r", encoding="utf-8") as f:
            thresh_data = json.load(f)

        thresholds_dict = thresh_data.get("thresholds", {})
        thresholds = np.array([thresholds_dict.get(l, 0.5) for l in dataset.labels])

        print("Applying thresholds and generating submission JSONL...")
        preds_bin = aggregated_scores >= thresholds

        submission_records = []
        pred_map = {}
        for i, pid in enumerate(unique_pids):
            pred_indices = np.where(preds_bin[i])[0]
            pred_codes = [dataset.labels[idx] for idx in pred_indices]
            pred_map[int(pid)] = pred_codes
        if args.apply_parent_child:
            pred_map = apply_specific_parent_child(pred_map)
        for pid in unique_pids:
            pred_codes = pred_map[int(pid)]
            doc_annotations = [[code] for code in pred_codes]
            submission_records.append(
                {
                    "patient_id": pid,
                    "document_level_annotations": doc_annotations,
                }
            )

        out_path = os.path.join(os.path.dirname(checkpoint_dir), "test_predictions.jsonl")
        save_jsonl(submission_records, out_path)
        print(f"Submission saved to {out_path}")


if __name__ == "__main__":
    main()
