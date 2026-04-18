from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import scipy.sparse as sp

try:
    from ..evaluation.config_utils import get_cfg, load_config
    from ..evaluation.io_utils import load_predictions
    from ..preprocessing.io_utils import load_labelset
except ImportError:
    from src.evaluation.config_utils import get_cfg, load_config
    from src.evaluation.io_utils import load_predictions
    from src.preprocessing.io_utils import load_labelset


@dataclass
class ModelArtifacts:
    name: str
    type: str  # "scores" or "predictions_only"
    scores: Optional[np.ndarray]
    patient_ids: List[int]
    label_names: List[str]
    pred_data: Dict[int, List[str]]
    output_subdir: Path


def resolve_model_paths(model_cfg: Dict[str, Any]) -> Dict[str, str]:
    """Resolve artifacts paths from a model's config."""
    paths = {}
    if "paths" in model_cfg:
        paths = model_cfg["paths"].copy()
        if "val_path" not in paths:
            paths["val_path"] = "data/processed/validation_set.jsonl"
        return paths
        
    if "train_config" in model_cfg:
        tcfg = load_config(model_cfg["train_config"])
        paths["scores_path"] = get_cfg(tcfg, "output.scores_path")
        paths["pids_path"] = get_cfg(tcfg, "output.patient_ids_path", get_cfg(tcfg, "output.pids_path"))
        paths["label_names_path"] = get_cfg(tcfg, "output.label_names_path")
        paths["thresholds_path"] = get_cfg(tcfg, "output.thresholds_path")
        paths["val_path"] = get_cfg(tcfg, "data.val_path", "data/processed/validation_set.jsonl")
    return paths


def ensure_model_artifacts(model_cfg: Dict[str, Any]) -> None:
    """Run predict.py if artifacts are missing for models."""
    mtype = model_cfg.get("type", "scores")
    
    if mtype == "scores":
        paths = resolve_model_paths(model_cfg)
        scores_path = paths.get("scores_path")
        
        if not scores_path or not Path(scores_path).exists():
            print(f"[{model_cfg['name']}] Artifacts missing. Running inference...")
            if "predict_module" not in model_cfg:
                print(f"[{model_cfg['name']}] No predict_module specified, cannot run inference.")
                return
                
            cmd = ["python", "-m", model_cfg["predict_module"]]
            if "predict_args" in model_cfg:
                cmd.extend(model_cfg["predict_args"])
            if "train_config" in model_cfg:
                cmd.extend(["--config", model_cfg["train_config"]])
            
            # Support for fold argument in xlm_r_base
            if "fold" in model_cfg:
                cmd.extend(["--fold", str(model_cfg["fold"])])
                
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        else:
            print(f"[{model_cfg['name']}] Found existing artifacts at {scores_path}")
            
    elif mtype == "predictions_only":
        pred_path = model_cfg.get("predictions_path")
        if not pred_path or not Path(pred_path).exists():
            print(f"[{model_cfg['name']}] Artifacts missing. Running inference...")
            if "predict_module" not in model_cfg:
                print(f"[{model_cfg['name']}] No predict_module specified, cannot run inference.")
                return
                
            cmd = ["python", "-m", model_cfg["predict_module"]]
            if "predict_args" in model_cfg:
                cmd.extend(model_cfg["predict_args"])
                
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        else:
            print(f"[{model_cfg['name']}] Found existing artifacts at {pred_path}")


def load_model_artifacts(
    model_cfg: Dict[str, Any],
    global_val_pids: List[int],
    analysis_out_dir: Optional[Path] = None,
) -> ModelArtifacts:
    """Load model outputs (either scores or predictions_only)."""
    mtype = model_cfg.get("type", "scores")
    name = model_cfg["name"]
    root = analysis_out_dir if analysis_out_dir is not None else Path("outputs/analysis")
    # Per-model derived artifacts (plots/json) live under the analysis output dir
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if mtype == "scores":
        paths = resolve_model_paths(model_cfg)
        scores = np.load(paths["scores_path"])
        with open(paths["pids_path"], "r", encoding="utf-8") as handle:
            patient_ids = [int(x) for x in json.load(handle)]
        with open(paths["label_names_path"], "r", encoding="utf-8") as handle:
            label_names = [str(x) for x in json.load(handle)]
            
        # Derive predictions
        thresholds_path = paths.get("thresholds_path")
        if thresholds_path and Path(thresholds_path).exists():
            if thresholds_path.endswith(".npy"):
                thresholds = np.load(thresholds_path)
            else:
                with open(thresholds_path, "r", encoding="utf-8") as f:
                    thresh_data = json.load(f)
                # Accept both shapes:
                #   wrapped: {"best_micro_f1": ..., "thresholds": {code: t, ...}, ...}
                #   flat:    {code: t, ...}
                if (
                    isinstance(thresh_data, dict)
                    and "thresholds" in thresh_data
                    and isinstance(thresh_data["thresholds"], dict)
                ):
                    thresholds_dict = thresh_data["thresholds"]
                elif isinstance(thresh_data, dict):
                    thresholds_dict = thresh_data
                else:
                    thresholds_dict = {}
                missing = [l for l in label_names if l not in thresholds_dict]
                if missing:
                    preview = ", ".join(missing[:5])
                    suffix = "..." if len(missing) > 5 else ""
                    print(
                        f"[{model_cfg.get('name', '?')}] WARN: {len(missing)}/"
                        f"{len(label_names)} labels missing in {thresholds_path}, "
                        f"defaulting to 0.5 for: {preview}{suffix}"
                    )
                thresholds = np.array(
                    [float(thresholds_dict.get(l, 0.5)) for l in label_names]
                )
        else:
            thresholds = np.full(len(label_names), 0.5)

        preds_bin = scores >= thresholds
        pred_data = {}
        for i, pid in enumerate(patient_ids):
            pred_indices = np.where(preds_bin[i])[0]
            pred_data[int(pid)] = [label_names[idx] for idx in pred_indices]
            
        return ModelArtifacts(name, mtype, scores, patient_ids, label_names, pred_data, out_dir)
        
    elif mtype == "predictions_only":
        pred_path = model_cfg["predictions_path"]
        labelset_path = model_cfg["labelset_path"]
        
        pred_data = {int(pid): list(codes) for pid, codes in load_predictions(pred_path).items()}
        
        label_names = load_labelset(labelset_path)
        
        # Use the global patient_ids ordering so build_binary_matrices aligns with GT
        patient_ids = global_val_pids.copy()
        
        return ModelArtifacts(name, mtype, None, patient_ids, label_names, pred_data, out_dir)
        
    else:
        raise ValueError(f"Unknown model type {mtype}")


def build_binary_matrices(
    gt_data: Dict[int, List[List[str]]],
    pred_data: Dict[int, List[str]],
    patient_ids: List[int],
    label_names: List[str]
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build scipy sparse CSR matrices for GT and Predictions (N_docs x 115)."""
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    n_docs = len(patient_ids)
    n_labels = len(label_names)
    
    y_true = sp.lil_matrix((n_docs, n_labels), dtype=np.int8)
    y_pred = sp.lil_matrix((n_docs, n_labels), dtype=np.int8)
    
    for i, pid in enumerate(patient_ids):
        # GT: mark 1 if the code appears in ANY group for the doc
        gt_groups = gt_data.get(pid, [])
        for group in gt_groups:
            for code in group:
                if code in label_to_idx:
                    y_true[i, label_to_idx[code]] = 1
                    
        # Preds
        preds = pred_data.get(pid, [])
        for code in preds:
            if code in label_to_idx:
                y_pred[i, label_to_idx[code]] = 1
                
    return y_true.tocsr(), y_pred.tocsr()


def label_support_from_gt(gt_data: Dict[int, List[List[str]]], label_names: List[str]) -> Counter:
    """Count groups containing each code (aligns with per_class_report semantics)."""
    support = Counter()
    for gt_groups in gt_data.values():
        for group in gt_groups:
            for code in set(group):
                if code in label_names:
                    support[code] += 1
    return support


def ensure_output_dir(cfg: Dict[str, Any]) -> Path:
    """Ensure and return the output directory."""
    out_dir = Path(get_cfg(cfg, "output.dir", "outputs/analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def clustering_output_dir(cfg: Dict[str, Any]) -> Path:
    """Directory for global clustering artifacts (embeddings cache, assignments, summary, UMAP plot)."""
    explicit = get_cfg(cfg, "clustering.dir", None)
    if explicit:
        p = Path(explicit)
    else:
        p = Path(get_cfg(cfg, "output.dir", "outputs/analysis")) / "clustering"
    p.mkdir(parents=True, exist_ok=True)
    return p
