import argparse
import json
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support

try:
    from ..evaluation.config_utils import load_config, get_cfg
    from ..evaluation.evaluator import evaluate_data
    from ..evaluation.io_utils import load_ground_truth
    from .common import load_model_artifacts, build_binary_matrices, ensure_output_dir
except ImportError:
    from src.evaluation.config_utils import load_config, get_cfg
    from src.evaluation.evaluator import evaluate_data
    from src.evaluation.io_utils import load_ground_truth
    from src.analysis.common import load_model_artifacts, build_binary_matrices, ensure_output_dir


def compute_flat_metrics(
    gt_data: Dict[int, List[List[str]]],
    pred_data: Dict[int, List[str]],
    patient_ids: List[int],
    label_names: List[str]
) -> Dict[str, Any]:
    """Compute flat matrix metrics using sklearn (ignores group semantics)."""
    y_true, y_pred = build_binary_matrices(gt_data, pred_data, patient_ids, label_names)
    
    averages = {}
    for avg in ["micro", "macro", "weighted"]:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        averages[f"{avg}_precision"] = float(p)
        averages[f"{avg}_recall"] = float(r)
        averages[f"{avg}_f1"] = float(f1)
        
    report = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )
    
    return {
        "averages": averages,
        "classification_report": report
    }


def recall_at_k(
    scores: np.ndarray,
    gt_data: Dict[int, List[List[str]]],
    patient_ids: List[int],
    label_names: List[str],
    ks: List[int]
) -> Dict[str, float]:
    """Compute Recall@K (fraction of true codes whose score is within the top-K)."""
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    
    recalls = {k: [] for k in ks}
    
    for i, pid in enumerate(patient_ids):
        gt_groups = gt_data.get(pid, [])
        if not gt_groups:
            continue
            
        # Get set of all true codes for this doc
        true_codes = {code for group in gt_groups for code in group if code in label_to_idx}
        if not true_codes:
            continue
            
        true_indices = {label_to_idx[code] for code in true_codes}
        
        # Sort indices by score descending
        doc_scores = scores[i]
        top_indices = np.argsort(doc_scores)[::-1]
        
        for k in ks:
            top_k_indices = set(top_indices[:k])
            hits = len(true_indices.intersection(top_k_indices))
            recalls[k].append(hits / len(true_indices))
            
    return {f"recall_at_{k}": float(np.mean(recalls[k])) if recalls[k] else 0.0 for k in ks}


def metrics_by_cluster(
    cluster_assignments: Dict[int, int],
    gt_data: Dict[int, List[List[str]]],
    pred_data: Dict[int, List[str]],
    label_space: List[str]
) -> Dict[str, Any]:
    """Compute group-level metrics per cluster."""
    cluster_gts = defaultdict(dict)
    cluster_preds = defaultdict(dict)
    
    for pid, gt_groups in gt_data.items():
        if pid in cluster_assignments:
            cid = cluster_assignments[pid]
            cluster_gts[cid][pid] = gt_groups
            cluster_preds[cid][pid] = pred_data.get(pid, [])
            
    cluster_metrics = {}
    for cid in cluster_gts.keys():
        metrics = evaluate_data(cluster_gts[cid], cluster_preds[cid], label_space=label_space)
        cluster_metrics[str(cid)] = {
            "micro_f1": float(metrics.get("micro_f1", 0.0)),
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "docs_evaluated": metrics.get("docs_evaluated", 0)
        }
        
    return cluster_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/analysis/analysis.yaml")
    parser.add_argument("--model", required=True, help="Which model to analyze from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    val_path = get_cfg(cfg, "data.val_path")
    out_dir = ensure_output_dir(cfg)
    
    model_cfgs = {m["name"]: m for m in get_cfg(cfg, "models", [])}
    if args.model not in model_cfgs:
        raise ValueError(f"Model {args.model} not found in analysis.yaml models list.")
        
    model_cfg = model_cfgs[args.model]
    
    print(f"Loading data for metrics engine ({args.model})...")
    gt_data = load_ground_truth(val_path)
    global_pids = list(gt_data.keys())
    
    from .common import load_model_artifacts
    artifacts = load_model_artifacts(model_cfg, global_pids, analysis_out_dir=out_dir)
    
    print("Computing group-level metrics (evaluator.py)...")
    group_metrics = evaluate_data(gt_data, artifacts.pred_data, label_space=artifacts.label_names)
    group_summary = {
        "micro_f1": float(group_metrics.get("micro_f1", 0.0)),
        "precision": float(group_metrics.get("precision", 0.0)),
        "recall": float(group_metrics.get("recall", 0.0)),
        "macro_f1_present_labels": float(group_metrics.get("macro_f1_present_labels", 0.0)),
        "macro_f1_all_labels": float(group_metrics.get("macro_f1_all_labels", 0.0)),
    }
    
    print("Computing flat metrics (sklearn)...")
    flat_metrics = compute_flat_metrics(gt_data, artifacts.pred_data, artifacts.patient_ids, artifacts.label_names)
    
    print("Computing Recall@K...")
    ks = get_cfg(cfg, "top_k", [3, 5, 10])
    if artifacts.scores is not None:
        top_k_metrics = recall_at_k(artifacts.scores, gt_data, artifacts.patient_ids, artifacts.label_names, ks)
    else:
        top_k_metrics = {f"recall_at_{k}": "N/A" for k in ks}
    
    # Try to load cluster assignments
    cluster_path = out_dir / "cluster_assignments.json"
    cluster_metrics = {}
    if cluster_path.exists():
        print("Computing per-cluster metrics...")
        with open(cluster_path, "r", encoding="utf-8") as f:
            cluster_assignments = {int(k): int(v) for k, v in json.load(f).items()}
        cluster_metrics = metrics_by_cluster(cluster_assignments, gt_data, artifacts.pred_data, artifacts.label_names)
        
    report = {
        "group_level": group_summary,
        "flat_metrics": flat_metrics["averages"],
        "top_k": top_k_metrics,
        "per_cluster": cluster_metrics
    }
    
    with open(artifacts.output_subdir / "metrics_engine.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"Metrics engine complete for {args.model}.")


if __name__ == "__main__":
    main()
