import argparse
import json
from collections import Counter
from typing import Dict, List, Any

try:
    from ..evaluation.config_utils import load_config, get_cfg
    from ..evaluation.evaluator import evaluate_data
    from ..evaluation.io_utils import load_ground_truth
    from .common import ensure_output_dir, load_model_artifacts, label_support_from_gt
    from .error_analysis import build_confusion_views, range_vs_specific_summary
except ImportError:
    from src.evaluation.config_utils import load_config, get_cfg
    from src.evaluation.evaluator import evaluate_data
    from src.evaluation.io_utils import load_ground_truth
    from src.analysis.common import ensure_output_dir, load_model_artifacts, label_support_from_gt
    from src.analysis.error_analysis import build_confusion_views, range_vs_specific_summary


def frequency_buckets(support_counter: Counter, cfg: Dict[str, Any]) -> Dict[str, str]:
    """Bucket codes into frequent, medium, rare based on config."""
    freq_min = get_cfg(cfg, "long_tail.frequent_min_support", 50)
    rare_max = get_cfg(cfg, "long_tail.rare_max_support", 10)
    
    buckets = {}
    for code, count in support_counter.items():
        if count >= freq_min:
            buckets[code] = "frequent"
        elif count <= rare_max:
            buckets[code] = "rare"
        else:
            buckets[code] = "medium"
    return buckets


def long_tail_metrics(per_class_rows: List[dict], buckets: Dict[str, str]) -> Dict[str, dict]:
    """Compute Macro and Weighted F1 per frequency bucket."""
    bucket_stats = {
        "frequent": {"f1s": [], "supports": [], "precisions": [], "recalls": []},
        "medium": {"f1s": [], "supports": [], "precisions": [], "recalls": []},
        "rare": {"f1s": [], "supports": [], "precisions": [], "recalls": []},
    }
    
    for row in per_class_rows:
        code = row["code"]
        b = buckets.get(code, "medium")
        bucket_stats[b]["f1s"].append(row.get("f1", 0.0))
        bucket_stats[b]["supports"].append(row.get("support", 0))
        bucket_stats[b]["precisions"].append(row.get("precision", 0.0))
        bucket_stats[b]["recalls"].append(row.get("recall", 0.0))
        
    results = {}
    for b, stats in bucket_stats.items():
        total_support = sum(stats["supports"])
        if not stats["f1s"]:
            results[b] = {
                "macro_f1": 0.0, "weighted_f1": 0.0,
                "mean_precision": 0.0, "mean_recall": 0.0,
                "n_labels": 0, "total_support": 0
            }
            continue
            
        macro_f1 = sum(stats["f1s"]) / len(stats["f1s"])
        mean_p = sum(stats["precisions"]) / len(stats["precisions"])
        mean_r = sum(stats["recalls"]) / len(stats["recalls"])
        
        weighted_f1 = 0.0
        if total_support > 0:
            weighted_f1 = sum(f * s for f, s in zip(stats["f1s"], stats["supports"])) / total_support
            
        results[b] = {
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "mean_precision": mean_p,
            "mean_recall": mean_r,
            "n_labels": len(stats["f1s"]),
            "total_support": total_support
        }
        
    return results


def top_confused_pairs(metrics: Dict, k: int = 10) -> List[dict]:
    """Get the top K most confused (predicted -> missed) code pairs."""
    confusion = build_confusion_views(metrics)
    top_pairs = [
        {"predicted": p, "missed": t, "count": c}
        for (p, t), c in confusion["wrong_pairs"].most_common(k)
    ]
    return top_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/analysis/analysis.yaml")
    parser.add_argument("--model", required=True, help="Which model to analyze from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    val_path = get_cfg(cfg, "data.val_path")
    
    model_cfgs = {m["name"]: m for m in get_cfg(cfg, "models", [])}
    if args.model not in model_cfgs:
        raise ValueError(f"Model {args.model} not found in analysis.yaml models list.")
        
    model_cfg = model_cfgs[args.model]
    
    print(f"Loading data for label analysis ({args.model})...")
    # Load GT to get global pid ordering for predictions_only models
    gt_data = load_ground_truth(val_path)
    global_pids = list(gt_data.keys())
    
    out_root = ensure_output_dir(cfg)
    artifacts = load_model_artifacts(model_cfg, global_pids, analysis_out_dir=out_root)
    
    print("Evaluating predictions...")
    metrics = evaluate_data(gt_data, artifacts.pred_data, label_space=artifacts.label_names)
    
    # Label analysis
    support_counter = label_support_from_gt(gt_data, artifacts.label_names)
    buckets = frequency_buckets(support_counter, cfg)
    
    tail_metrics = long_tail_metrics(metrics.get("per_class", []), buckets)
    
    top_k_pairs = get_cfg(cfg, "top_n_confused_pairs", 10)
    confused_pairs = top_confused_pairs(metrics, k=top_k_pairs)
    
    confusion = build_confusion_views(metrics)
    range_specific = range_vs_specific_summary(
        per_class_rows=metrics.get("per_class", []),
        fp_by_label=confusion["fp_by_label"],
        fn_by_label=confusion["fn_by_label"]
    )
    
    report = {
        "long_tail": tail_metrics,
        "top_confused_pairs": confused_pairs,
        "range_vs_specific": range_specific,
        "wrong_pairs_counter": {f"{k[0]}|{k[1]}": v for k, v in confusion["wrong_pairs"].items()} # Serializing Counter
    }
    
    with open(artifacts.output_subdir / "label_analysis.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"Label analysis complete for {args.model}.")


if __name__ == "__main__":
    main()
