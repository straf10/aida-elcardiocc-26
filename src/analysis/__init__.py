import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any

try:
    from ..evaluation.config_utils import load_config, get_cfg
    from .common import (
        clustering_output_dir,
        collect_long_tail_comparison,
        ensure_output_dir,
        ensure_model_artifacts,
    )
    from .visualizer import (
        plot_confusion_heatmap,
        plot_long_tail,
        plot_cluster_map,
        plot_models_long_tail_comparison,
    )
except ImportError:
    from src.evaluation.config_utils import load_config, get_cfg
    from src.analysis.common import (
        clustering_output_dir,
        collect_long_tail_comparison,
        ensure_output_dir,
        ensure_model_artifacts,
    )
    from src.analysis.visualizer import (
        plot_confusion_heatmap,
        plot_long_tail,
        plot_cluster_map,
        plot_models_long_tail_comparison,
    )


def generate_report_md(out_dir: Path, cfg: Dict[str, Any], comparison_data: Dict[str, dict]):
    """Generate combined Markdown summary from all models' artifacts."""
    md_lines = ["# Medical Report Summary\n"]
    
    # 1. Cross-Model Comparison Table
    md_lines.append("## 1. Cross-Model Comparison\n")
    md_lines.append("| Model | Micro-F1 (Group) | Micro-F1 (Flat) | Macro-F1 | Weighted-F1 | Recall@3 | Recall@5 |")
    md_lines.append("|---|---|---|---|---|---|---|")
    
    for model_name, metrics in comparison_data.items():
        micro_g = metrics.get("micro_f1_group", 0.0)
        micro_f = metrics.get("micro_f1_flat", 0.0)
        macro = metrics.get("macro_f1", 0.0)
        weighted = metrics.get("weighted_f1", 0.0)
        r3 = metrics.get("recall_at_3", "N/A")
        r5 = metrics.get("recall_at_5", "N/A")
        
        r3_str = f"{r3:.4f}" if isinstance(r3, float) else r3
        r5_str = f"{r5:.4f}" if isinstance(r5, float) else r5
        
        md_lines.append(f"| **{model_name}** | {micro_g:.4f} | {micro_f:.4f} | {macro:.4f} | {weighted:.4f} | {r3_str} | {r5_str} |")
    md_lines.append("\n")

    if (out_dir / "models_comparison_buckets.png").exists():
        md_lines.append("### Long-tail (frequency bucket) comparison across models\n")
        md_lines.append(
            "![Model comparison by frequency bucket](models_comparison_buckets.png)\n"
        )

    # 2. Per-Model Sections
    md_lines.append("## 2. Per-Model Details\n")
    
    models = get_cfg(cfg, "models", [])
    for model_cfg in models:
        name = model_cfg["name"]
        model_dir = out_dir / name
        if not model_dir.exists():
            continue
            
        md_lines.append(f"### {name}\n")
        
        # Long-tail metrics
        label_path = model_dir / "label_analysis.json"
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                l_data = json.load(f)
                
            md_lines.append("#### Long-Tail Performance\n")
            md_lines.append("| Bucket | N Labels | Support | Macro F1 | Weighted F1 |")
            md_lines.append("|---|---|---|---|---|")
            for bucket, stats in l_data.get("long_tail", {}).items():
                md_lines.append(
                    f"| {bucket} | {stats.get('n_labels', 0)} | {stats.get('total_support', 0)} | "
                    f"{stats.get('macro_f1', 0.0):.4f} | {stats.get('weighted_f1', 0.0):.4f} |"
                )
            md_lines.append("\n")
            
            md_lines.append("#### Top 10 Confused Pairs\n")
            md_lines.append("| Predicted (Wrong) | Missed (True) | Count |")
            md_lines.append("|---|---|---|")
            for pair in l_data.get("top_confused_pairs", []):
                md_lines.append(f"| {pair.get('predicted')} | {pair.get('missed')} | {pair.get('count')} |")
            md_lines.append("\n")
            
            if (model_dir / "confusion_heatmap.png").exists():
                md_lines.append(f"![Confusion Heatmap]({name}/confusion_heatmap.png)\n")
            if (model_dir / "long_tail.png").exists():
                md_lines.append(f"![Long Tail Analysis]({name}/long_tail.png)\n")
                
        # Hard Cases & Short/Long
        prof_path = model_dir / "error_profiler.json"
        if prof_path.exists():
            with open(prof_path, "r", encoding="utf-8") as f:
                p_data = json.load(f)
                
            md_lines.append("#### Keyword Hard Cases\n")
            md_lines.append("| Keyword | N Docs | Mean FP | Mean FN | Worst PIDs |")
            md_lines.append("|---|---|---|---|---|")
            for kw, stats in p_data.get("keyword_hard_cases", {}).items():
                pids = ", ".join(map(str, stats.get("worst_pids", [])))
                md_lines.append(
                    f"| {kw} | {stats.get('n_docs', 0)} | {stats.get('mean_fp', 0.0):.2f} | "
                    f"{stats.get('mean_fn', 0.0):.2f} | {pids} |"
                )
            md_lines.append("\n")
            
            md_lines.append("#### Short vs Long Reports\n")
            length_stats = p_data.get("length_analysis", {})
            if length_stats:
                threshold = length_stats.get("threshold_chars", 0)
                short = length_stats.get("short_docs", {})
                long_docs = length_stats.get("long_docs", {})
                md_lines.append(f"Split Threshold: {threshold} chars\n")
                md_lines.append("| Split | N Docs | Micro-F1 | Precision | Recall |")
                md_lines.append("|---|---|---|---|---|")
                md_lines.append(
                    f"| Short | {short.get('n', 0)} | {short.get('micro_f1', 0.0):.4f} | "
                    f"{short.get('precision', 0.0):.4f} | {short.get('recall', 0.0):.4f} |"
                )
                md_lines.append(
                    f"| Long | {long_docs.get('n', 0)} | {long_docs.get('micro_f1', 0.0):.4f} | "
                    f"{long_docs.get('precision', 0.0):.4f} | {long_docs.get('recall', 0.0):.4f} |"
                )
            md_lines.append("\n")
            
        md_lines.append("---\n")

    # 3. Clustering (Global)
    md_lines.append("## 3. Medical Clusters (Global Validation Set)\n")
    cluster_dir = clustering_output_dir(cfg)
    cluster_path = cluster_dir / "cluster_summary.json"
    if cluster_path.exists():
        with open(cluster_path, "r", encoding="utf-8") as f:
            c_data = json.load(f)
            
        md_lines.append("| Cluster ID | Size | Mean Len | Top Terms |")
        md_lines.append("|---|---|---|---|")
        for cluster in c_data:
            cid = str(cluster.get("cluster_id"))
            terms = ", ".join(cluster.get("top_terms", []))
            md_lines.append(
                f"| {cid} | {cluster.get('size', 0)} | {cluster.get('mean_doc_len', 0.0):.0f} | {terms} |"
            )
        md_lines.append("\n")
        
    if (cluster_dir / "cluster_map.png").exists():
        md_lines.append("![Global Cluster Map](clustering/cluster_map.png)\n")
        
    report_path = out_dir / "medical_report_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    print(f"Medical Report Summary generated: {report_path}")


def run_full_analysis(config_path: str):
    """Orchestrate the full multi-model medical analysis pipeline."""
    cfg = load_config(config_path)
    out_dir = ensure_output_dir(cfg)
    
    # 1. Run global clustering (once)
    print("\n--- Running Global Clustering ---")
    try:
        subprocess.run(
            ["python", "-m", "src.analysis.medical_clustering", "--config", config_path],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running medical_clustering.py: {e}")
        
    # Generate Cluster Map
    cdir = clustering_output_dir(cfg)
    embeddings_path = Path(
        get_cfg(cfg, "clustering.embeddings_cache", str(cdir / "embeddings.npy"))
    )
    cluster_path = cdir / "cluster_assignments.json"
    if embeddings_path.exists() and cluster_path.exists():
        import numpy as np
        embeddings = np.load(embeddings_path)
        with open(cluster_path, "r", encoding="utf-8") as f:
            c_data = json.load(f)
            # Reorder labels to match embeddings order (assuming keys are 0...N-1 pids or indices)
            labels = np.array([c_data[k] for k in sorted(c_data.keys(), key=int)])
            
        plot_cluster_map(
            embeddings,
            labels,
            out_path=cdir / "cluster_map.png"
        )

    # 2. Per-Model Analysis
    models = get_cfg(cfg, "models", [])
    comparison_data = {}
    
    for model_cfg in models:
        name = model_cfg["name"]
        print(f"\n=== Analyzing Model: {name} ===")
        
        # Ensure artifacts exist (runs inference if type=scores and missing)
        ensure_model_artifacts(model_cfg)
        
        # Run submodules for this model
        for script in ["label_analysis", "metrics_engine", "error_profiler"]:
            print(f"--- Running {script} for {name} ---")
            try:
                subprocess.run(
                    ["python", "-m", f"src.analysis.{script}", "--config", config_path, "--model", name],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error running {script}.py for {name}: {e}")
                continue
                
        model_dir = out_dir / name
        
        # Extract comparison metrics
        metrics_path = model_dir / "metrics_engine.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                m_data = json.load(f)
                comparison_data[name] = {
                    "micro_f1_group": m_data.get("group_level", {}).get("micro_f1", 0.0),
                    "micro_f1_flat": m_data.get("flat_metrics", {}).get("micro_f1", 0.0),
                    "macro_f1": m_data.get("flat_metrics", {}).get("macro_f1", 0.0),
                    "weighted_f1": m_data.get("flat_metrics", {}).get("weighted_f1", 0.0),
                    "recall_at_3": m_data.get("top_k", {}).get("recall_at_3", "N/A"),
                    "recall_at_5": m_data.get("top_k", {}).get("recall_at_5", "N/A"),
                    "recall_at_10": m_data.get("top_k", {}).get("recall_at_10", "N/A"),
                }
                
        # Visualizations for this model
        label_path = model_dir / "label_analysis.json"
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                l_data = json.load(f)
                
            top_n = get_cfg(cfg, "top_n_heatmap_codes", 20)
            plot_confusion_heatmap(
                l_data.get("wrong_pairs_counter", {}),
                top_n,
                model_dir / "confusion_heatmap.png"
            )
            
            plot_long_tail(
                l_data.get("long_tail", {}),
                model_dir / "long_tail.png"
            )

    # 3. Dump models_comparison.json and long-tail bucket comparison chart
    if comparison_data:
        with open(out_dir / "models_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison_data, f, indent=2)

    model_names = [m["name"] for m in models]
    bucket_comparison = collect_long_tail_comparison(out_dir, model_names)
    if bucket_comparison:
        with open(out_dir / "models_bucket_comparison.json", "w", encoding="utf-8") as f:
            json.dump(bucket_comparison, f, indent=2)
        plot_models_long_tail_comparison(
            bucket_comparison,
            out_dir / "models_comparison_buckets.png",
            model_order=model_names,
        )

    # 4. Generate Combined Report
    print("\n--- Generating Combined Report ---")
    generate_report_md(out_dir, cfg, comparison_data)
    print("\nAll multi-model analysis complete.")


def main():
    parser = argparse.ArgumentParser(description="Medical Text Classification Multi-Model Analysis")
    parser.add_argument("--config", default="src/analysis/analysis.yaml", help="Path to config YAML")
    args = parser.parse_args()
    run_full_analysis(args.config)


if __name__ == "__main__":
    main()
