import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any

try:
    from ..evaluation.config_utils import load_config, get_cfg
    from .common import ensure_output_dir
    from .visualizer import plot_confusion_heatmap, plot_long_tail, plot_cluster_map
except ImportError:
    from src.evaluation.config_utils import load_config, get_cfg
    from src.analysis.common import ensure_output_dir
    from src.analysis.visualizer import plot_confusion_heatmap, plot_long_tail, plot_cluster_map


def generate_report_md(out_dir: Path, cfg: Dict[str, Any]):
    """Generate Markdown summary from all JSON artifacts."""
    md_lines = ["# Medical Report Summary\n"]
    
    # 1. Overview
    metrics_path = out_dir / "metrics_engine.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
            
        md_lines.append("## 1. Overview\n")
        
        md_lines.append("### Group-Level Metrics")
        group = m.get("group_level", {})
        md_lines.append(f"- **Micro-F1**: {group.get('micro_f1', 0.0):.4f}")
        md_lines.append(f"- **Precision**: {group.get('precision', 0.0):.4f}")
        md_lines.append(f"- **Recall**: {group.get('recall', 0.0):.4f}")
        md_lines.append(f"- **Macro-F1 (present)**: {group.get('macro_f1_present_labels', 0.0):.4f}")
        md_lines.append("")
        
        md_lines.append("### Flat Metrics (sklearn)")
        flat = m.get("flat_metrics", {})
        md_lines.append(f"- **Micro-F1**: {flat.get('micro_f1', 0.0):.4f}")
        md_lines.append(f"- **Macro-F1**: {flat.get('macro_f1', 0.0):.4f}")
        md_lines.append(f"- **Weighted-F1**: {flat.get('weighted_f1', 0.0):.4f}")
        md_lines.append("")
        
        md_lines.append("### Recall@K")
        for k, v in m.get("top_k", {}).items():
            md_lines.append(f"- **{k}**: {v:.4f}")
        md_lines.append("\n")
        
    # 2. Long-tail & Pairs
    label_path = out_dir / "label_analysis.json"
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            l_data = json.load(f)
            
        md_lines.append("## 2. Long-Tail Analysis\n")
        md_lines.append("| Bucket | N Labels | Support | Macro F1 | Weighted F1 |")
        md_lines.append("|---|---|---|---|---|")
        
        for bucket, stats in l_data.get("long_tail", {}).items():
            md_lines.append(
                f"| {bucket} | {stats.get('n_labels', 0)} | {stats.get('total_support', 0)} | "
                f"{stats.get('macro_f1', 0.0):.4f} | {stats.get('weighted_f1', 0.0):.4f} |"
            )
            
        md_lines.append("\n## 3. Top Confused Pairs\n")
        md_lines.append("| Predicted (Wrong) | Missed (True) | Count |")
        md_lines.append("|---|---|---|")
        for pair in l_data.get("top_confused_pairs", []):
            md_lines.append(f"| {pair.get('predicted')} | {pair.get('missed')} | {pair.get('count')} |")
            
        md_lines.append("\n")
        
    # 4. Clusters
    cluster_path = out_dir / "cluster_summary.json"
    if cluster_path.exists() and metrics_path.exists():
        with open(cluster_path, "r", encoding="utf-8") as f:
            c_data = json.load(f)
        
        with open(metrics_path, "r", encoding="utf-8") as f:
            m_data = json.load(f)
            
        per_cluster = m_data.get("per_cluster", {})
            
        md_lines.append("## 4. Medical Clusters\n")
        md_lines.append("| Cluster ID | Size | Mean Len | Micro-F1 | Top Terms |")
        md_lines.append("|---|---|---|---|---|")
        
        for cluster in c_data:
            cid = str(cluster.get("cluster_id"))
            f1 = per_cluster.get(cid, {}).get("micro_f1", 0.0)
            terms = ", ".join(cluster.get("top_terms", []))
            md_lines.append(
                f"| {cid} | {cluster.get('size', 0)} | {cluster.get('mean_doc_len', 0.0):.0f} | "
                f"{f1:.4f} | {terms} |"
            )
            
        md_lines.append("\n")
        
    # 5. Error Profiler
    profiler_path = out_dir / "error_profiler.json"
    if profiler_path.exists():
        with open(profiler_path, "r", encoding="utf-8") as f:
            p_data = json.load(f)
            
        md_lines.append("## 5. Clinical Hard Cases\n")
        md_lines.append("### By Keyword\n")
        md_lines.append("| Keyword | N Docs | Mean FP | Mean FN | Worst PIDs |")
        md_lines.append("|---|---|---|---|---|")
        for kw, stats in p_data.get("keyword_hard_cases", {}).items():
            pids = ", ".join(map(str, stats.get("worst_pids", [])))
            md_lines.append(
                f"| {kw} | {stats.get('n_docs', 0)} | {stats.get('mean_fp', 0.0):.2f} | "
                f"{stats.get('mean_fn', 0.0):.2f} | {pids} |"
            )
            
        md_lines.append("\n### Short vs Long Reports\n")
        length_stats = p_data.get("length_analysis", {})
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
        
        sys_comp = p_data.get("system_comparison", {})
        if sys_comp:
            md_lines.append("\n### System Comparison\n")
            md_lines.append("| System | Micro-F1 | Precision | Recall |")
            md_lines.append("|---|---|---|---|")
            for sys_name, sys_stats in sys_comp.items():
                md_lines.append(
                    f"| {sys_name} | {sys_stats.get('micro_f1', 0.0):.4f} | "
                    f"{sys_stats.get('precision', 0.0):.4f} | {sys_stats.get('recall', 0.0):.4f} |"
                )
                
        md_lines.append("\n")
        
    # 6. Visualizations
    md_lines.append("## 6. Visualizations\n")
    if (out_dir / "confusion_heatmap.png").exists():
        md_lines.append("![Confusion Heatmap](confusion_heatmap.png)\n")
    if (out_dir / "cluster_map.png").exists():
        md_lines.append("![Cluster Map](cluster_map.png)\n")
    if (out_dir / "long_tail.png").exists():
        md_lines.append("![Long Tail Analysis](long_tail.png)\n")

    report_path = out_dir / "medical_report_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    print(f"Medical Report Summary generated: {report_path}")


def run_full_analysis(config_path: str):
    """Orchestrate the full medical analysis pipeline."""
    cfg = load_config(config_path)
    out_dir = ensure_output_dir(cfg)
    
    scripts = [
        "medical_clustering",
        "label_analysis",
        "metrics_engine",
        "error_profiler"
    ]
    
    # Run submodules via subprocess to keep isolation
    for script in scripts:
        print(f"\n--- Running {script}.py ---")
        try:
            subprocess.run(
                ["python", "-m", f"src.analysis.{script}", "--config", config_path],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}.py: {e}")
            return
            
    # Generate Visualizations
    print("\n--- Generating Visualizations ---")
    label_path = out_dir / "label_analysis.json"
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            l_data = json.load(f)
            
        top_n = get_cfg(cfg, "top_n_heatmap_codes", 20)
        plot_confusion_heatmap(
            l_data.get("wrong_pairs_counter", {}),
            top_n,
            out_dir / "confusion_heatmap.png"
        )
        
        plot_long_tail(
            l_data.get("long_tail", {}),
            out_dir / "long_tail.png"
        )
        
    embeddings_path = Path(get_cfg(cfg, "clustering.embeddings_cache", "outputs/analysis/embeddings.npy"))
    cluster_path = out_dir / "cluster_assignments.json"
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
            out_path=out_dir / "cluster_map.png"
        )
            
    # Generate Report
    print("\n--- Generating Report ---")
    generate_report_md(out_dir, cfg)
    print("\nAll analysis complete.")


def main():
    parser = argparse.ArgumentParser(description="Medical Text Classification Analysis")
    parser.add_argument("--config", default="src/analysis/analysis.yaml", help="Path to config YAML")
    args = parser.parse_args()
    run_full_analysis(args.config)


if __name__ == "__main__":
    main()
