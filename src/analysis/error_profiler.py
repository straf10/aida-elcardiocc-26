import argparse
import json
import statistics
from typing import Dict, List, Any

try:
    from ..preprocessing.io_utils import load_jsonl
    from ..dictionary.dictionary import normalize_term
    from ..evaluation.config_utils import load_config, get_cfg
    from ..evaluation.evaluator import evaluate_data
    from ..evaluation.io_utils import load_ground_truth, load_predictions
    from .common import load_scores_bundle, derive_predictions, ensure_output_dir, load_model_artifacts
except ImportError:
    from src.preprocessing.io_utils import load_jsonl
    from src.dictionary.dictionary import normalize_term
    from src.evaluation.config_utils import load_config, get_cfg
    from src.evaluation.evaluator import evaluate_data
    from src.evaluation.io_utils import load_ground_truth, load_predictions
    from src.analysis.common import load_scores_bundle, derive_predictions, ensure_output_dir, load_model_artifacts


def keyword_hard_cases(records: List[dict], doc_breakdown: List[dict], keywords: List[str]) -> Dict[str, dict]:
    """Find hard cases based on clinical keywords."""
    norm_keywords = [normalize_term(k) for k in keywords]
    pid_to_text = {int(r["patient_id"]): r.get("text", "") for r in records}
    
    breakdown_by_pid = {row["patient_id"]: row for row in doc_breakdown}
    
    keyword_stats = {}
    
    for kw, norm_kw in zip(keywords, norm_keywords):
        subset_pids = []
        for pid, text in pid_to_text.items():
            if norm_kw in normalize_term(text):
                subset_pids.append(pid)
                
        if not subset_pids:
            continue
            
        subset_breakdown = [breakdown_by_pid[pid] for pid in subset_pids if pid in breakdown_by_pid]
        
        if not subset_breakdown:
            continue
            
        mean_fn = statistics.mean(row["fn"] for row in subset_breakdown)
        mean_fp = statistics.mean(row["fp"] for row in subset_breakdown)
        
        worst_docs = sorted(subset_breakdown, key=lambda x: (x["fn"], x["fp"]), reverse=True)[:5]
        
        keyword_stats[kw] = {
            "n_docs": len(subset_pids),
            "mean_fn": mean_fn,
            "mean_fp": mean_fp,
            "worst_pids": [row["patient_id"] for row in worst_docs]
        }
        
    return keyword_stats


def length_split_analysis(
    records: List[dict],
    gt_data: Dict[int, List[List[str]]],
    pred_data: Dict[int, List[str]],
    labels: List[str],
    split_method: str
) -> Dict[str, Any]:
    """Split docs by length and compute metrics on halves."""
    pid_to_len = {int(r["patient_id"]): len(r.get("text", "")) for r in records}
    
    if not pid_to_len:
        return {}
        
    lens = list(pid_to_len.values())
    if split_method == "median":
        threshold = statistics.median(lens)
    else:
        try:
            threshold = int(split_method)
        except ValueError:
            threshold = statistics.median(lens)
            
    short_pids = {pid for pid, l in pid_to_len.items() if l <= threshold}
    long_pids = {pid for pid, l in pid_to_len.items() if l > threshold}
    
    short_gt = {pid: gt_data[pid] for pid in short_pids if pid in gt_data}
    short_pred = {pid: pred_data.get(pid, []) for pid in short_pids}
    
    long_gt = {pid: gt_data[pid] for pid in long_pids if pid in gt_data}
    long_pred = {pid: pred_data.get(pid, []) for pid in long_pids}
    
    short_metrics = evaluate_data(short_gt, short_pred, label_space=labels)
    long_metrics = evaluate_data(long_gt, long_pred, label_space=labels)
    
    return {
        "threshold_chars": threshold,
        "short_docs": {
            "n": len(short_gt),
            "micro_f1": float(short_metrics.get("micro_f1", 0.0)),
            "precision": float(short_metrics.get("precision", 0.0)),
            "recall": float(short_metrics.get("recall", 0.0))
        },
        "long_docs": {
            "n": len(long_gt),
            "micro_f1": float(long_metrics.get("micro_f1", 0.0)),
            "precision": float(long_metrics.get("precision", 0.0)),
            "recall": float(long_metrics.get("recall", 0.0))
        }
    }


def compare_systems(
    records: List[dict],
    gt_data: Dict[int, List[List[str]]],
    pred_sources: Dict[str, str],
    labels: List[str]
) -> Dict[str, Dict[str, float]]:
    """Compare multiple prediction sources."""
    results = {}
    for name, path in pred_sources.items():
        try:
            sys_preds = load_predictions(path)
            metrics = evaluate_data(gt_data, sys_preds, label_space=labels)
            results[name] = {
                "micro_f1": float(metrics.get("micro_f1", 0.0)),
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0))
            }
        except Exception as e:
            print(f"Skipping {name} ({path}): {e}")
    return results


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
    
    print(f"Loading data for error profiling ({args.model})...")
    records = load_jsonl(val_path)
    gt_data = load_ground_truth(val_path)
    global_pids = list(gt_data.keys())
    
    from .common import load_model_artifacts
    artifacts = load_model_artifacts(model_cfg, global_pids)
    
    metrics = evaluate_data(gt_data, artifacts.pred_data, label_space=artifacts.label_names)
    doc_breakdown = metrics.get("doc_breakdown", [])
    
    keywords = get_cfg(cfg, "keywords", [])
    kw_stats = keyword_hard_cases(records, doc_breakdown, keywords)
    
    split_method = str(get_cfg(cfg, "length_split", "median"))
    length_stats = length_split_analysis(records, gt_data, artifacts.pred_data, artifacts.label_names, split_method)
    
    report = {
        "keyword_hard_cases": kw_stats,
        "length_analysis": length_stats
    }
    
    with open(artifacts.output_subdir / "error_profiler.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"Error profiling complete for {args.model}.")


if __name__ == "__main__":
    main()
