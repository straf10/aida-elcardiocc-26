import os
import sys

_REPO_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from preprocessing.dataset import ELCardioDataset
from preprocessing.io_utils import (
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    RAW_SUBMISSION_TEST_PATH,
    save_jsonl,
)
from evaluation.config_utils import get_cfg, load_config
from evaluation.evaluator import evaluate_data
from evaluation.io_utils import load_ground_truth
from split_data.device_utils import get_device, use_amp_fp16

from xlm_r_large.chunk_aggregate import aggregate_scores_by_patient_torch
from xlm_r_large.model import load_model_for_inference
from xlm_r_large.postprocess import apply_specific_parent_child


def _as_repo_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


def _resolve_checkpoint_dir(config: dict) -> Path:
    """
    Prefer ``output.checkpoint_dir`` from YAML; weights usually live under
    ``outputs/models/xlm_r_large`` (legacy: ``outputs/models/xlm_large``). Try config path first,
    then the directory containing ``output.thresholds_path``, then those model dirs.
    """
    ck_cfg = get_cfg(config, "output.checkpoint_dir", "outputs/models/xlm_r_large")
    thr_cfg = get_cfg(config, "output.thresholds_path", "outputs/models/xlm_r_large/thresholds.json")
    candidates: list[Path] = []
    seen: set[Path] = set()
    for rel in (
        ck_cfg,
        str(Path(thr_cfg).parent),
        "outputs/models/xlm_r_large",
        "outputs/models/xlm_large",
    ):
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
        + "\nCopy your trained save into outputs/models/xlm_r_large (same layout as training export), "
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
        choices=["val", "test", "blind", "train"],
        help="Which split: val, test (labeled), train (processed train), or blind (submission).",
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
        help="Add specific->specific parent codes (e.g. I11->I10) after thresholding.",
    )
    parser.add_argument(
        "--export-scores",
        action="store_true",
        help=(
            "Export aggregated sigmoid scores (+ pids + labels) for test/blind "
            "so probability-level ensembling can run offline."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)

    num_labels = get_cfg(config, "model.num_labels", 115)

    if args.split == "val":
        data_path = get_cfg(config, "data.val_path")
    elif args.split == "blind":
        data_path = get_cfg(config, "data.blind_path", RAW_SUBMISSION_TEST_PATH)
    elif args.split == "train":
        data_path = get_cfg(config, "data.train_path", PROCESSED_TRAIN_PATH)
    else:
        data_path = get_cfg(config, "data.test_path", PROCESSED_TEST_PATH)

    labelset_path = get_cfg(config, "data.labelset_path")
    max_length = int(get_cfg(config, "data.max_length", 512))
    sliding_window = bool(get_cfg(config, "data.sliding_window", False))
    stride = int(get_cfg(config, "data.stride", 256))
    truncation_side = str(get_cfg(config, "data.truncation_side", "right"))
    chunk_aggregation = str(get_cfg(config, "data.chunk_aggregation", "mean_max"))
    chunk_aggregation_alpha = float(get_cfg(config, "data.chunk_aggregation_alpha", 0.5))
    batch_size = int(
        get_cfg(
            config,
            "training.eval_batch_size",
            get_cfg(config, "training.batch_size", 8),
        )
    )
    num_workers = int(get_cfg(config, "training.num_workers", 4))
    pin_memory = bool(get_cfg(config, "training.pin_memory", True))
    fp16 = bool(get_cfg(config, "training.fp16", True))
    eval_threshold = float(get_cfg(config, "training.eval_threshold", 0.5))

    ckpt_path = _resolve_checkpoint_dir(config)
    checkpoint_dir = str(ckpt_path)

    scores_path = get_cfg(config, "output.scores_path", "outputs/models/xlm_r_large/val_scores.npy")
    pids_path = get_cfg(config, "output.patient_ids_path", "outputs/models/xlm_r_large/val_patient_ids.json")
    label_names_path = get_cfg(config, "output.label_names_path", "outputs/models/xlm_r_large/label_names.json")
    test_scores_path = get_cfg(
        config,
        "output.test_scores_path",
        scores_path.replace("val_scores", "test_scores"),
    )
    blind_scores_path = get_cfg(
        config,
        "output.blind_scores_path",
        scores_path.replace("val_scores", "blind_scores"),
    )
    test_pids_path = pids_path.replace("val_patient_ids", "test_patient_ids")
    blind_pids_path = pids_path.replace("val_patient_ids", "blind_patient_ids")

    device = get_device(args.device)
    use_amp = use_amp_fp16(device, fp16)
    use_bf16 = bool(use_amp and device.type == "cuda" and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Using device: {device} | AMP (fp16): {use_amp} | BF16: {use_bf16}")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

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
    prefetch_factor = 4 if num_workers > 0 else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=prefetch_factor,
    )

    logits_list: list[torch.Tensor] = []
    pids_list: list[torch.Tensor] = []

    print("Running inference...")
    with torch.inference_mode():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            pids = batch["patient_id"]
            if not isinstance(pids, torch.Tensor):
                pids = torch.as_tensor(pids, device=device, dtype=torch.long)
            else:
                pids = pids.to(device, non_blocking=True, dtype=torch.long)

            autocast_ctx = (
                torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.float()
            logits_list.append(logits)
            pids_list.append(pids)

    if logits_list:
        all_logits = torch.cat(logits_list, dim=0)
        all_pids = torch.cat(pids_list, dim=0)
    else:
        all_logits = torch.empty(
            (0, int(num_labels)), device=device, dtype=torch.float32
        )
        all_pids = torch.empty((0,), device=device, dtype=torch.long)
    unique_pids, aggregated_scores = aggregate_scores_by_patient_torch(
        all_logits,
        all_pids,
        strategy=chunk_aggregation,
        temperature=1.0,
        alpha=chunk_aggregation_alpha,
    )
    if args.export_scores and args.split in {"test", "blind"}:
        if args.split == "test":
            export_scores_path = test_scores_path
            export_pids_path = test_pids_path
        else:
            export_scores_path = blind_scores_path
            export_pids_path = blind_pids_path
        for out_path in (export_scores_path, export_pids_path, label_names_path):
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        np.save(export_scores_path, aggregated_scores)
        with open(export_pids_path, "w", encoding="utf-8") as f:
            json.dump(unique_pids, f)
        with open(label_names_path, "w", encoding="utf-8") as f:
            json.dump(dataset.labels, f)
        print(f"[export-scores] Aggregated scores saved to {export_scores_path}")

    thresholds_default = get_cfg(
        config, "output.thresholds_path", "outputs/models/xlm_r_large/thresholds.json"
    )
    val_predictions_path = get_cfg(
        config,
        "output.val_predictions_path",
        "outputs/predictions/xlm_r_large/val_predictions.jsonl",
    )
    test_predictions_path = get_cfg(
        config,
        "output.test_predictions_path",
        "outputs/predictions/xlm_r_large/test_predictions.jsonl",
    )
    blind_predictions_path = get_cfg(
        config,
        "output.blind_predictions_path",
        "outputs/predictions/xlm_r_large/blind_predictions.jsonl",
    )
    train_predictions_path = get_cfg(
        config,
        "output.train_predictions_path",
        "outputs/predictions/xlm_r_large/train_predictions.jsonl",
    )

    if args.split == "val":
        print("Exporting validation artifacts for threshold tuning...")
        for _path in (scores_path, pids_path, label_names_path):
            _d = os.path.dirname(_path)
            if _d:
                os.makedirs(_d, exist_ok=True)
        np.save(scores_path, aggregated_scores)
        with open(pids_path, "w", encoding="utf-8") as f:
            json.dump(unique_pids, f)
        with open(label_names_path, "w", encoding="utf-8") as f:
            json.dump(dataset.labels, f)

        thr_path = args.thresholds or thresholds_default
        if os.path.isfile(thr_path):
            print(f"Loading tuned thresholds from {thr_path}...")
            thresholds = _threshold_vector_from_json(thr_path, dataset.labels)
        else:
            print(
                f"Thresholds not found at {thr_path}; using global eval_threshold={eval_threshold:.2f}."
            )
            thresholds = np.full(len(dataset.labels), eval_threshold, dtype=np.float64)

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
        if thr_path and os.path.isfile(thr_path):
            print(f"Loading tuned thresholds from {thr_path}...")
            thresholds = _threshold_vector_from_json(thr_path, dataset.labels)
        else:
            print(
                f"No thresholds file found; using global eval_threshold={eval_threshold:.2f} for test."
            )
            thresholds = np.full(len(dataset.labels), eval_threshold, dtype=np.float64)

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

        out_path = test_predictions_path
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        save_jsonl(submission_records, out_path)
        print(f"Test predictions saved to {out_path}")

    elif args.split == "train":
        thr_path = args.thresholds or thresholds_default
        if thr_path and os.path.isfile(thr_path):
            print(f"Loading tuned thresholds from {thr_path}...")
            thresholds = _threshold_vector_from_json(thr_path, dataset.labels)
        else:
            print(
                f"No thresholds file found; using global eval_threshold={eval_threshold:.2f} for train."
            )
            thresholds = np.full(len(dataset.labels), eval_threshold, dtype=np.float64)

        print("Applying thresholds and generating train split JSONL...")
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

        out_path = train_predictions_path
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        save_jsonl(submission_records, out_path)
        print(f"Train predictions saved to {out_path}")

    elif args.split == "blind":
        thr_path = args.thresholds or thresholds_default
        if thr_path and os.path.isfile(thr_path):
            print(f"Loading tuned thresholds from {thr_path}...")
            thresholds = _threshold_vector_from_json(thr_path, dataset.labels)
        else:
            print(
                f"No thresholds file found; using global eval_threshold={eval_threshold:.2f} for blind."
            )
            thresholds = np.full(len(dataset.labels), eval_threshold, dtype=np.float64)

        print("Applying thresholds and generating blind submission JSONL...")
        preds_bin = aggregated_scores >= thresholds

        submission_records = []
        for i, pid in enumerate(unique_pids):
            pred_indices = np.where(preds_bin[i])[0]
            pred_codes = [dataset.labels[idx] for idx in pred_indices]
            doc_annotations = [[code] for code in pred_codes]
            submission_records.append(
                {
                    "patient_id": pid,
                    "document_level_annotations": doc_annotations,
                }
            )

        out_path = blind_predictions_path
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        save_jsonl(submission_records, out_path)
        print(f"Blind predictions saved to {out_path}")


if __name__ == "__main__":
    main()
