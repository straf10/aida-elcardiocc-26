"""Compare micro-F1 across all methods defined in analysis.yaml.

For each model, ``load_model_artifacts`` writes ``outputs/predictions/<model>/predictions.jsonl``
and metrics are computed with ``evaluate_from_prediction_files`` (ground truth + that JSONL only).

Usage:
    python -m src.analysis.compare_methods
    python -m src.analysis.compare_methods --config src/analysis/analysis.yaml
"""

from __future__ import annotations

import argparse

try:
    from ..evaluation.config_utils import load_config, get_cfg
    from ..evaluation.evaluator import evaluate_from_prediction_files
    from ..evaluation.io_utils import load_ground_truth
    from .common import ensure_model_artifacts, load_model_artifacts, ensure_output_dir
except ImportError:
    from src.evaluation.config_utils import load_config, get_cfg
    from src.evaluation.evaluator import evaluate_from_prediction_files
    from src.evaluation.io_utils import load_ground_truth
    from src.analysis.common import ensure_model_artifacts, load_model_artifacts, ensure_output_dir


def main():
    parser = argparse.ArgumentParser(description="Compare F1 across all methods.")
    parser.add_argument("--config", default="src/analysis/analysis.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    val_path = get_cfg(cfg, "data.val_path")
    out_dir = ensure_output_dir(cfg)

    gt_data = load_ground_truth(val_path)
    global_pids = list(gt_data.keys())

    results = []
    for model_cfg in get_cfg(cfg, "models", []):
        name = model_cfg["name"]
        try:
            ensure_model_artifacts(model_cfg)
            artifacts = load_model_artifacts(model_cfg, global_pids, analysis_out_dir=out_dir)
            metrics = evaluate_from_prediction_files(
                val_path,
                str(artifacts.predictions_jsonl),
                label_space=artifacts.label_names,
            )
            results.append({
                "name": name,
                "micro_f1": metrics["micro_f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "macro_f1": metrics.get("macro_f1_present_labels", 0.0),
            })
        except Exception as exc:
            results.append({"name": name, "error": str(exc)})

    col_w = 22
    header = f"{'Method':<{col_w}} {'Micro-F1':>9} {'Precision':>10} {'Recall':>8} {'Macro-F1':>10}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['name']:<{col_w}}  ERROR: {r['error']}")
        else:
            print(
                f"{r['name']:<{col_w}} {r['micro_f1']:>9.4f} {r['precision']:>10.4f}"
                f" {r['recall']:>8.4f} {r['macro_f1']:>10.4f}"
            )
    print()


if __name__ == "__main__":
    main()
