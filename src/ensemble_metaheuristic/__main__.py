"""
Metaheuristic ensemble: weighted voting over IR, NER-EL, Greek BERT, XLM-R-Large.

Each model contributes a (n_docs x n_labels) score matrix:
  - score-based models : scores normalised by per-label threshold (>1.0 = positive)
  - prediction-only    : binary 0/1 votes

Search strategy: random search (phase 1) + hill climbing (phase 2) over
per-model weights, per-model activation thresholds, and one global threshold.

Usage:
    python -m src.ensemble_metaheuristic
    python -m src.ensemble_metaheuristic --n-iter 4000 --seed 0
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np

try:
    from src.analysis.common import load_model_artifacts
    from src.evaluation.config_utils import load_config, get_cfg
    from src.evaluation.evaluator import evaluate_data
    from src.evaluation.io_utils import load_ground_truth
except ImportError:
    from ..analysis.common import load_model_artifacts
    from ..evaluation.config_utils import load_config, get_cfg
    from ..evaluation.evaluator import evaluate_data
    from ..evaluation.io_utils import load_ground_truth

from .matrices import build_score_matrix, load_thresholds_for_model
from .search import run_search
from .strategies import (
    per_label_f1,
    build_label_routing_table,
    per_label_routed_predict,
    search_correction_params,
    correction_predict,
)

ANALYSIS_CFG = "src/analysis/analysis.yaml"
ENSEMBLE_MODELS = ["xlm_r_large", "mlc_greek_bert", "information_retrieval", "ner_el"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Metaheuristic ensemble search")
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=ANALYSIS_CFG)
    args = parser.parse_args()

    cfg = load_config(args.config)
    val_path = get_cfg(cfg, "data.val_path")
    gt_data = load_ground_truth(val_path)
    all_pids = list(gt_data.keys())
    model_cfgs = {m["name"]: m for m in get_cfg(cfg, "models", [])}

    print("Loading model artifacts...")
    artifacts_list = []
    for name in ENSEMBLE_MODELS:
        arts = load_model_artifacts(model_cfgs[name], all_pids)
        artifacts_list.append((name, arts))
        print(f"  {name}: {'scores' if arts.scores is not None else 'binary predictions'}")

    canonical_arts = next(a for n, a in artifacts_list if n == "xlm_r_large")
    all_labels = canonical_arts.label_names

    print(f"\nBuilding score matrices ({len(all_pids)} docs x {len(all_labels)} labels)...")
    matrices, names, is_score_model = [], [], []
    for name, arts in artifacts_list:
        thr = load_thresholds_for_model(model_cfgs[name], all_labels) if arts.scores is not None else None
        matrices.append(build_score_matrix(arts, all_pids, all_labels, thr))
        names.append(name)
        is_score_model.append(arts.scores is not None)

    print("\nIndividual model micro-F1:")
    for name, arts, mat in zip(names, [a for _, a in artifacts_list], matrices):
        cutoff = 1.0 if arts.scores is not None else 0.5
        preds = {pid: [all_labels[j] for j in np.where(mat[i] >= cutoff)[0]] for i, pid in enumerate(all_pids)}
        f1 = evaluate_data(gt_data, preds, label_space=all_labels)["micro_f1"]
        print(f"  {name}: {f1:.4f}")

    # --- Strategy 1: per-label routing ---
    print("\n--- Strategy 1: per-label routing ---")
    per_model_preds = {
        name: {
            pid: [all_labels[j] for j in np.where(mat[i] >= (1.0 if is_score else 0.5))[0]]
            for i, pid in enumerate(all_pids)
        }
        for name, mat, is_score in zip(names, matrices, is_score_model)
    }
    model_label_f1s = {name: per_label_f1(gt_data, preds, all_labels) for name, preds in per_model_preds.items()}
    label_routing = build_label_routing_table(model_label_f1s, all_labels)

    champion_counts = {}
    for label, champ in label_routing.items():
        champion_counts[champ] = champion_counts.get(champ, 0) + 1
    print("  Labels routed to each model:", {n: champion_counts.get(n, 0) for n in names})

    routed_preds = per_label_routed_predict(matrices, is_score_model, names, all_pids, all_labels, label_routing)
    m1 = evaluate_data(gt_data, routed_preds, label_space=all_labels)
    print(f"  Micro-F1={m1['micro_f1']:.4f}  Precision={m1['precision']:.4f}  Recall={m1['recall']:.4f}")

    # --- Strategy 2: correction mode (grid search) ---
    print("\n--- Strategy 2: correction mode (grid search over add/remove params) ---")
    best_cfg, best_f1_corr = search_correction_params(
        matrices, is_score_model, names, all_pids, all_labels, gt_data,
    )
    corr_preds = correction_predict(
        matrices, is_score_model, names, all_pids, all_labels, **best_cfg,
    )
    m2 = evaluate_data(gt_data, corr_preds, label_space=all_labels)
    print(f"  Best config: {best_cfg}")
    print(f"  Micro-F1={m2['micro_f1']:.4f}  Precision={m2['precision']:.4f}  Recall={m2['recall']:.4f}")

    # --- Strategy 3: weighted search (baseline) ---
    rng = np.random.RandomState(args.seed)
    print(f"\n--- Strategy 3: weighted search ({args.n_iter} iters) ---")
    best_w, best_mt, best_gt, best_f1 = run_search(
        matrices, is_score_model, gt_data, all_pids, all_labels, args.n_iter, rng,
    )
    print(f"\n=== Results ===")
    print(f"  Per-label routing : {m1['micro_f1']:.4f}")
    print(f"  Correction mode   : {m2['micro_f1']:.4f}")
    print(f"  Weighted search   : {best_f1:.4f}")
    print(f"  greek_bert alone  : {next(f for n,f in zip(names,[evaluate_data(gt_data,per_model_preds[n],label_space=all_labels)['micro_f1'] for n in names]) if n=='mlc_greek_bert'):.4f}")
    print(f"\nWeighted search params — threshold={best_gt:.4f}")
    mt_iter = iter(best_mt)
    for name, is_score, w in zip(names, is_score_model, best_w):
        mt = next(mt_iter) if is_score else 0.5
        suffix = f"  act_thr={mt:.4f}" if is_score else "  act_thr=0.5 (binary)"
        print(f"  {name:<25} weight={w:.4f}{suffix}")


if __name__ == "__main__":
    main()
