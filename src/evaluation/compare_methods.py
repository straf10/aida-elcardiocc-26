"""Compare micro-F1 across all methods defined in experiment.yaml.

Metrics use ``evaluate_from_prediction_files`` on ``data.val_path`` and each model's
``predictions_path`` (submission-format JSONL). If a file is missing, ``ensure_model_artifacts``
runs the model's ``predict_module`` from ``experiment.yaml``.

Usage (from repo root; either form works):

    python -m evaluation.compare_methods
    python src/evaluation/compare_methods.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.config_utils import get_cfg, load_config
from evaluation.evaluator import evaluate_from_prediction_files
from evaluation.inference import ensure_model_artifacts, ensure_output_dir
from evaluation.io_utils import load_ground_truth
from evaluation.model_artifacts import load_model_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare F1 across all methods.")
    parser.add_argument("--config", default="src/evaluation/experiment.yaml")
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
            artifacts = load_model_artifacts(model_cfg, global_pids, evaluation_root=out_dir)
            metrics = evaluate_from_prediction_files(
                val_path,
                str(artifacts.predictions_jsonl),
                label_space=artifacts.label_names,
            )
            results.append(
                {
                    "name": name,
                    "micro_f1": metrics["micro_f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "macro_f1": metrics.get("macro_f1_present_labels", 0.0),
                }
            )
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
