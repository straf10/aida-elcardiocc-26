import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from preprocessing.dataset import ELCardioDataset
from preprocessing.io_utils import PROCESSED_TEST_PATH, save_jsonl
from evaluation.config_utils import get_cfg, load_config
from evaluation.evaluator import evaluate_data
from evaluation.io_utils import load_ground_truth
from split_data.device_utils import get_device, use_amp_fp16

from .chunk_aggregate import aggregate_scores_by_patient
from .model import load_model_for_inference
from .postprocess import apply_specific_parent_child


def _as_repo_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


def _resolve_checkpoint_dir(config: dict) -> Path:
    """
    Training YAML may keep output.checkpoint_dir under experiments; copied weights usually live
    under outputs/models/xlm_large. Try config path first, then the directory containing
    output.thresholds_path, then outputs/models/xlm_large.
    """
    ck_cfg = get_cfg(config, "output.checkpoint_dir", "outputs/experiments/xlm_r_large/checkpoints")
    thr_cfg = get_cfg(config, "output.thresholds_path", "outputs/models/xlm_large/thresholds.json")
    candidates: list[Path] = []
    seen: set[Path] = set()
    for rel in (ck_cfg, str(Path(thr_cfg).parent), "outputs/models/xlm_large"):
        cand = _as_repo_path(rel)
        if cand not in seen:
            seen.add(cand)
            candidates.append(cand)
    for cand in candidates:
        if cand.is_dir() and (cand / "config.json").is_file():
            if cand != _as_repo_path(ck_cfg):
                print(
                    f"Note: output.checkpoint_dir ({ck_cfg}) has no HF save; loading weights from {cand}",
                    flush=True,
                )
            return cand
    raise SystemExit(
        "No HuggingFace checkpoint found (need a directory containing config.json). Tried:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\nCopy your trained save into outputs/models/xlm_large (same layout as training export), "
        "or set output.checkpoint_dir in the YAML to that path."
    )


def _threshold_vector_from_json(thr_path: str, labels: list) -> np.ndarray:
    """Load per-label thresholds; fail if file missing keys (no fabricated defaults)."""
    with open(thr_path, "r", encoding="utf-8") as f:
        thresh_data = json.load(f)
    if not isinstance(thresh_data, dict):
        raise ValueError(f"Expected a JSON object at {thr_path}")
    thresholds_dict = thresh_data.get("thresholds", thresh_data)
    if not isinstance(thresholds_dict, dict):
        raise ValueError(f"Expected top-level 'thresholds' object at {thr_path}")
    missing = [lab for lab in labels if lab not in thresholds_dict]
    if missing:
        raise ValueError(
            f"Thresholds file {thr_path} is missing {len(missing)} label(s): {missing[:15]}"
            + (" ..." if len(missing) > 15 else "")
        )
    return np.array([float(thresholds_dict[lab]) for lab in labels], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="XLM-R MLC inference")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    # Default test for inference only; training reads data.train_path / data.val_path from YAML.
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Which split to predict on (default: test, aligned with compare / run_predictions).",
    )
    parser.add_argument(
        "--thresholds",
        help="Path to tuned thresholds JSON (defaults to output.thresholds_path in config for val and test)",
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
        data_path = get_cfg(config, "data.test_path", PROCESSED_TEST_PATH)

    labelset_path = get_cfg(config, "data.labelset_path")
    max_length = get_cfg(config, "data.max_length", 512)
    sliding_window = get_cfg(config, "data.sliding_window", False)
    stride = get_cfg(config, "data.stride", 256)
    truncation_side = get_cfg(config, "data.truncation_side", "right")
    batch_size = get_cfg(config, "training.batch_size", 8)
    fp16 = get_cfg(config, "training.fp16", True)

    ckpt_path = _resolve_checkpoint_dir(config)
    checkpoint_dir = str(ckpt_path)

    scores_path = get_cfg(config, "output.scores_path", "outputs/experiments/xlm_r_large/val_scores.npy")
    pids_path = get_cfg(config, "output.patient_ids_path", "outputs/experiments/xlm_r_large/val_patient_ids.json")
    label_names_path = get_cfg(config, "output.label_names_path", "outputs/experiments/xlm_r_large/label_names.json")

    device = get_device(args.device)
    use_amp = use_amp_fp16(device, fp16)
    print(f"Using device: {device} | AMP (fp16): {use_amp}")

    print(f"Loading model from {checkpoint_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
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
                logits = outputs.logits.float().cpu().numpy()

            for i, pid in enumerate(pids):
                if pid not in pid_to_logits:
                    pid_to_logits[pid] = []
                pid_to_logits[pid].append(logits[i])

    unique_pids, aggregated_scores = aggregate_scores_by_patient(
        pid_to_logits,
        strategy="max",
        temperature=1.0,
    )

    thresholds_default = get_cfg(
        config, "output.thresholds_path", "outputs/models/xlm_large/thresholds.json"
    )
    val_predictions_path = get_cfg(
        config,
        "output.val_predictions_path",
        "outputs/predictions/xlm_r_large/predictions.jsonl",
    )

    if args.split == "val":
        print("Exporting validation artifacts for threshold tuning...")
        os.makedirs(os.path.dirname(scores_path), exist_ok=True)
        np.save(scores_path, aggregated_scores)
        with open(pids_path, "w", encoding="utf-8") as f:
            json.dump(unique_pids, f)
        with open(label_names_path, "w", encoding="utf-8") as f:
            json.dump(dataset.labels, f)

        thr_path = args.thresholds or thresholds_default
        if not os.path.isfile(thr_path):
            raise FileNotFoundError(
                f"Thresholds not found: {thr_path}. Tune thresholds or pass --thresholds."
            )
        print(f"Loading tuned thresholds from {thr_path}...")
        thresholds = _threshold_vector_from_json(thr_path, dataset.labels)

        print("Applying thresholds and writing validation predictions JSONL...")
        preds_bin = aggregated_scores >= thresholds
        pred_map: dict[int, list[str]] = {}
        for i, pid in enumerate(unique_pids):
            pred_indices = np.where(preds_bin[i])[0]
            pred_codes = [dataset.labels[idx] for idx in pred_indices]
            pred_map[int(pid)] = pred_codes
        if args.apply_parent_child:
            pred_map = apply_specific_parent_child(pred_map)

        submission_records = []
        for pid in unique_pids:
            pred_codes = pred_map[int(pid)]
            doc_annotations = [[code] for code in pred_codes]
            submission_records.append(
                {
                    "patient_id": pid,
                    "document_level_annotations": doc_annotations,
                }
            )
        os.makedirs(os.path.dirname(val_predictions_path), exist_ok=True)
        save_jsonl(submission_records, val_predictions_path)
        print(f"Validation predictions saved to {val_predictions_path}")

        ground_truth_data = load_ground_truth(data_path)
        metrics = evaluate_data(
            ground_truth_data, pred_map, label_space=dataset.labels
        )
        print(f"Val Micro-F1 (tuned thresholds): {metrics['micro_f1']:.4f}")

    elif args.split == "test":
        thr_path = args.thresholds or thresholds_default
        if not thr_path or not os.path.isfile(thr_path):
            raise ValueError(
                "--thresholds is required for test split (or set output.thresholds_path in config)"
            )

        print(f"Loading tuned thresholds from {thr_path}...")
        thresholds = _threshold_vector_from_json(thr_path, dataset.labels)

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

        out_path = val_predictions_path
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        save_jsonl(submission_records, out_path)
        print(f"Submission saved to {out_path}")


if __name__ == "__main__":
    main()
