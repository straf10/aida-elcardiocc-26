"""
Metaheuristic ensemble: weighted voting over XLM-R-Large, Greek-BERT, XLM-R-Base,
information retrieval, and NER-EL.

Each model contributes a (n_docs x n_labels) score matrix:
  - score-based models : scores normalised by per-label threshold (>1.0 = positive)
  - prediction-only    : binary 0/1 votes

Runs four strategies on the validation split (in order):
  1. Weighted search — random search + hill climb over weights/thresholds
  2. Per-cluster champion — best base model per document cluster (needs cluster_assignments.json)
  3. Per-label routing — each ICD code from its best model (optional score-cutoff sweep)
  4. Correction mode — best individual model (by val micro-F1) + grid over add/remove rules

Usage:
    python -m src.ensemble_metaheuristic
    python -m src.ensemble_metaheuristic --n-iter 10000 --seed 0
    python -m src.ensemble_metaheuristic --cluster-assignments outputs/analysis/clustering/cluster_assignments.json
    python -m src.ensemble_metaheuristic --routing-sweep-steps 80 --correction-extended
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from src.analysis.common import clustering_output_dir, load_model_artifacts
    from src.evaluation.config_utils import load_config, get_cfg
    from src.evaluation.evaluator import evaluate_data
    from src.evaluation.io_utils import load_ground_truth
except ImportError:
    from ..analysis.common import clustering_output_dir, load_model_artifacts
    from ..evaluation.config_utils import load_config, get_cfg
    from ..evaluation.evaluator import evaluate_data
    from ..evaluation.io_utils import load_ground_truth

from .matrices import build_score_matrix, load_thresholds_for_model
from .search import run_search
from .strategies import (
    build_cluster_champion_routing,
    build_label_routing_table,
    correction_predict,
    per_cluster_champion_predict,
    per_label_f1,
    per_label_routed_predict,
    search_correction_params,
)

ANALYSIS_CFG = "src/analysis/analysis.yaml"
ENSEMBLE_MODELS = [
    "xlm_r_large",
    "mlc_greek_bert",
    "xlm_r_base",
    "information_retrieval",
    "ner_el",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Metaheuristic ensemble search")
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10000,
        help="Weighted search: total iterations (75%% random, 25%% hill climb).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for weighted search.")
    parser.add_argument(
        "--weighted-restarts",
        type=int,
        default=1,
        help="Run weighted search this many times with seeds seed, seed+1, ...; keep best F1.",
    )
    parser.add_argument(
        "--routing-sweep-steps",
        type=int,
        default=0,
        help="If > 0, try this many score-cutoff values for per-label routing (default 1.0 only).",
    )
    parser.add_argument(
        "--correction-extended",
        action="store_true",
        help="Larger correction-mode grid (more add_votes × add_factor combinations).",
    )
    parser.add_argument(
        "--cluster-assignments",
        type=str,
        default=None,
        help="Path to cluster_assignments.json; default is clustering dir from config.",
    )
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
    individual_micro_f1: Dict[str, float] = {}
    for name, arts, mat in zip(names, [a for _, a in artifacts_list], matrices):
        cutoff = 1.0 if arts.scores is not None else 0.5
        preds = {pid: [all_labels[j] for j in np.where(mat[i] >= cutoff)[0]] for i, pid in enumerate(all_pids)}
        f1 = evaluate_data(gt_data, preds, label_space=all_labels)["micro_f1"]
        individual_micro_f1[name] = f1
        print(f"  {name}: {f1:.4f}")
    # Tie-break: earlier model in `names` wins (stable, matches ENSEMBLE_MODELS order).
    best_single_name = min(names, key=lambda n: (-individual_micro_f1[n], names.index(n)))
    best_single_f1 = individual_micro_f1[best_single_name]

    per_model_preds: Dict[str, Dict[int, List[str]]] = {
        name: {
            pid: [all_labels[j] for j in np.where(mat[i] >= (1.0 if is_score else 0.5))[0]]
            for i, pid in enumerate(all_pids)
        }
        for name, mat, is_score in zip(names, matrices, is_score_model)
    }

    # --- Strategy 1: weighted search ---
    print(
        f"\n--- Strategy 1: weighted search "
        f"({args.n_iter} iters × {args.weighted_restarts} restart(s)) ---"
    )
    best_w = best_mt = best_gt = best_f1 = None
    best_restart = 0
    for r in range(max(1, int(args.weighted_restarts))):
        rng = np.random.RandomState(int(args.seed) + r)
        print(f"  Restart {r + 1}/{max(1, int(args.weighted_restarts))}  seed={int(args.seed) + r}")
        w, mt, gt, f1 = run_search(
            matrices, is_score_model, gt_data, all_pids, all_labels, args.n_iter, rng,
        )
        if best_f1 is None or f1 > best_f1:
            best_w, best_mt, best_gt, best_f1 = w, mt, gt, f1
            best_restart = r
    assert best_f1 is not None
    print(f"  Best restart: {best_restart + 1} (seed {int(args.seed) + best_restart})")
    print(f"  Micro-F1={best_f1:.4f}")

    # --- Strategy 2: per-cluster champion ---
    cluster_path = (
        Path(args.cluster_assignments)
        if args.cluster_assignments
        else clustering_output_dir(cfg) / "cluster_assignments.json"
    )
    cluster_assignments: Dict[int, int] = {}
    print("\n--- Strategy 2: per-cluster champion ---")
    if cluster_path.is_file():
        cluster_assignments = {
            int(k): int(v)
            for k, v in json.loads(cluster_path.read_text(encoding="utf-8")).items()
        }
    cluster_routing, cluster_scores = (
        build_cluster_champion_routing(
            cluster_assignments, all_pids, names, per_model_preds, gt_data, all_labels,
        )
        if cluster_assignments
        else ({}, {})
    )
    m_per_cluster = None
    if not cluster_routing:
        if not cluster_path.is_file():
            print(f"  Skipped (missing {cluster_path})")
        else:
            print("  Skipped (no cluster id covers any validation patient)")
    else:
        for cid in sorted(cluster_routing):
            print(
                f"  cluster {cid}: {cluster_routing[cid]} "
                f"(subset micro-F1={cluster_scores[cid]:.4f})",
            )
        pc_preds = per_cluster_champion_predict(
            cluster_assignments, all_pids, cluster_routing, per_model_preds,
        )
        m_per_cluster = evaluate_data(gt_data, pc_preds, label_space=all_labels)
        print(
            f"  Micro-F1={m_per_cluster['micro_f1']:.4f}  "
            f"Precision={m_per_cluster['precision']:.4f}  "
            f"Recall={m_per_cluster['recall']:.4f}",
        )

    # --- Strategy 3: per-label routing ---
    print("\n--- Strategy 3: per-label routing ---")
    model_label_f1s = {name: per_label_f1(gt_data, preds, all_labels) for name, preds in per_model_preds.items()}
    label_routing = build_label_routing_table(model_label_f1s, all_labels)

    champion_counts = {}
    for label, champ in label_routing.items():
        champion_counts[champ] = champion_counts.get(champ, 0) + 1
    print("  Labels routed to each model:", {n: champion_counts.get(n, 0) for n in names})

    if args.routing_sweep_steps and args.routing_sweep_steps > 0:
        print(f"  Threshold sweep: {args.routing_sweep_steps} score-cutoff values …")
        sweep_cuts = np.linspace(0.72, 1.18, int(args.routing_sweep_steps))
        best_r_f1, best_r_cut, best_r_preds = -1.0, 1.0, {}
        for ci, cut in enumerate(sweep_cuts):
            rp = per_label_routed_predict(
                matrices, is_score_model, names, all_pids, all_labels, label_routing,
                score_cutoff=float(cut),
            )
            rf = evaluate_data(gt_data, rp, label_space=all_labels)["micro_f1"]
            if rf > best_r_f1:
                best_r_f1, best_r_cut, best_r_preds = rf, float(cut), rp
            if (ci + 1) % max(1, len(sweep_cuts) // 8) == 0:
                print(f"    … step {ci + 1}/{len(sweep_cuts)}  best_micro_f1={best_r_f1:.4f} @ cut={best_r_cut:.4f}")
        routed_preds = best_r_preds
        m_per_label = evaluate_data(gt_data, routed_preds, label_space=all_labels)
        print(f"  Best score-cutoff={best_r_cut:.4f} (sweep)")
    else:
        routed_preds = per_label_routed_predict(
            matrices, is_score_model, names, all_pids, all_labels, label_routing,
        )
        m_per_label = evaluate_data(gt_data, routed_preds, label_space=all_labels)
    print(
        f"  Micro-F1={m_per_label['micro_f1']:.4f}  "
        f"Precision={m_per_label['precision']:.4f}  "
        f"Recall={m_per_label['recall']:.4f}",
    )

    # --- Strategy 4: correction mode (grid search) ---
    print(
        "\n--- Strategy 4: correction mode (grid search over add/remove params) ---\n"
        f"  Base model (best individual): {best_single_name}  micro-F1={best_single_f1:.4f}",
    )
    best_cfg, _ = search_correction_params(
        matrices,
        is_score_model,
        names,
        all_pids,
        all_labels,
        gt_data,
        base_model=best_single_name,
        extended=bool(args.correction_extended),
    )
    n_corr_grid = best_cfg.pop("_grid_evaluations", None)
    if n_corr_grid is not None:
        print(f"  Grid evaluations: {n_corr_grid}")
    corr_preds = correction_predict(
        matrices,
        is_score_model,
        names,
        all_pids,
        all_labels,
        base_model=best_single_name,
        **best_cfg,
    )
    m_correction = evaluate_data(gt_data, corr_preds, label_space=all_labels)
    print(f"  Best config: {best_cfg}")
    print(
        f"  Micro-F1={m_correction['micro_f1']:.4f}  "
        f"Precision={m_correction['precision']:.4f}  "
        f"Recall={m_correction['recall']:.4f}",
    )

    print("\n=== Results (validation micro-F1) ===")
    print(f"  Weighted search     : {best_f1:.4f}")
    print(
        "  Per-cluster champion  : "
        + (
            f"{m_per_cluster['micro_f1']:.4f}"
            if m_per_cluster is not None
            else "(skipped)"
        ),
    )
    print(f"  Per-label routing     : {m_per_label['micro_f1']:.4f}")
    print(f"  Correction mode       : {m_correction['micro_f1']:.4f}")
    print(f"  Best single model ({best_single_name})  : {best_single_f1:.4f}")
    print(f"\nWeighted search params — threshold={best_gt:.4f}")
    mt_iter = iter(best_mt)
    for name, is_score, w in zip(names, is_score_model, best_w):
        mt = next(mt_iter) if is_score else 0.5
        suffix = f"  act_thr={mt:.4f}" if is_score else "  act_thr=0.5 (binary)"
        print(f"  {name:<25} weight={w:.4f}{suffix}")


if __name__ == "__main__":
    main()
