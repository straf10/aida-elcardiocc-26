"""
Metaheuristic ensemble: weighted voting over XLM-R-Large, Greek-BERT, XLM-R-Base,
information retrieval, and NER-EL.

Each model contributes a (n_docs x n_labels) score matrix:
  - score-based models : scores normalised by per-label threshold (>1.0 = positive)
  - prediction-only    : binary 0/1 votes

``python -m src.ensemble_metaheuristic`` runs the full validation pipeline:

  - **Weighted search** — classic and/or VNS (``--weighted-search``; default ``both`` uses
    ``WEIGHTED_RESTARTS`` seeds; best micro-F1 drives fusion + majority vote over restarts).
  - **Per-cluster sweeps** (heavy): validation clustering, train-only score routing, optional Greek-BERT
    text routing. **Off by default**; use ``--cluster-sweeps``. Caches text runs under
    ``outputs/ensemble_metaheuristic/text_cluster_cache/``.
  - **Per-patient routing** (score-only, then kNN-on-train when data exist).
  - **Per-label champion routing** (cutoff sweep) and **per-label champion + non-champion vote** (cut ×
    ``min_other_votes`` grid) — see ``strategies.per_label_routing`` and ``strategies.per_label_champion_plus_vote``.
  - **Correction** (grid) + label-set fusion + rule-based extras.

To run **one** strategy end-to-end, use the matching module, for example
``python -m src.ensemble_metaheuristic.strategies.weighted_strategy`` or
``python -m src.ensemble_metaheuristic.strategies.per_label_routing`` (see ``--help`` on each).

Classic search is ``strategies.weighted_strategy`` (``run_search``); VNS is ``strategies.weighted_vns_strategy``
(``run_vns_search``). ``strategies.weighted_search`` is a re-export shim for older imports.

CLI: ``--config``, ``--n-iter``, ``--seed``, ``--weighted-search classic|vns|both``, ``--cluster-sweeps``.
Tune constants in this file if needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from src.evaluation.config_utils import load_config, get_cfg
    from src.evaluation.evaluator import evaluate_data
    from src.evaluation.io_utils import load_ground_truth
    from src.evaluation.model_artifacts import load_model_artifacts
except ImportError:
    from ..evaluation.config_utils import load_config, get_cfg
    from ..evaluation.evaluator import evaluate_data
    from ..evaluation.io_utils import load_ground_truth
    from ..evaluation.model_artifacts import load_model_artifacts

from .matrices import build_score_matrix, load_thresholds_for_model
from .strategy_cli import build_per_model_preds, load_train_matrices
from .strategies import (
    build_label_routing_table,
    build_patient_routing_knn_train,
    correction_predict,
    merge_preds_intersection,
    merge_preds_k_of_n,
    merge_preds_union,
    per_label_champion_plus_other_vote_predict,
    per_label_f1,
    per_label_routed_predict,
    per_patient_champion_from_scores,
    per_patient_routed_predict,
    run_score_matrix_cluster_sweep,
    run_score_matrix_cluster_sweep_train_routing,
    run_text_embedding_cluster_sweep_train_routing,
    run_search,
    run_vns_search,
    search_correction_params,
    weighted_ensemble_combined_matrix,
    weighted_ensemble_predict,
    weighted_ensemble_predict_frequency_buckets,
    weighted_ensemble_predict_gated_secondary,
    weighted_ensemble_predict_top_k,
    weighted_ensemble_predict_two_threshold,
)

EXPERIMENT_CFG = "src/evaluation/experiment.yaml"
ENSEMBLE_MODELS = [
    "xlm_r_large",
    "mlc_greek_bert",
    "xlm_r_base",
    "information_retrieval",
    "ner_el",
]

WEIGHTED_RESTARTS = 2
ROUTING_SWEEP_STEPS = 24
# K = 2, 6, …, 62, 64 (step 4 from 2; 64 appended so the sweep reaches 64). Methods with k > n_docs skip.
EMBEDDING_K_LIST = sorted(set(range(2, 65, 4)) | {64})
EMBEDDING_CLUSTER_METHODS = (
    "kmeans",
    "kmeans_cosine",
    "agglomerative",
    "gmm",
    "spectral",
    "dbscan",
)
PATIENT_SCORE_ROUTING_POLICY = "mean"
PATIENT_KNN_K = 11
# Text-embedding cluster sweep (train-only routing): skip spectral/dbscan on large train for speed.
TEXT_CLUSTER_METHODS = ("kmeans", "kmeans_cosine", "agglomerative", "gmm")


def _label_doc_frequency(gt_data: Dict, all_labels: List[str]) -> Dict[str, int]:
    """How many validation documents contain each code in any diagnosis group."""
    out = {lbl: 0 for lbl in all_labels}
    for groups in gt_data.values():
        for c in {x for grp in groups for x in grp}:
            if c in out:
                out[c] += 1
    return out


def _run_weighted_restarts(
    optimizer: str,
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    gt_data: Dict,
    all_pids: List[int],
    all_labels: List[str],
    n_iter: int,
    base_seed: int,
    n_restarts: int,
    *,
    verbose_first_restart: bool,
) -> Tuple[np.ndarray, np.ndarray, float, float, List[Dict[int, List[str]]]]:
    """Run ``run_search`` (classic) or ``run_vns_search`` (vns) for ``n_restarts`` seeds."""
    best_w = best_mt = best_gt = best_f1 = None
    restart_preds: List[Dict[int, List[str]]] = []
    nr = max(1, int(n_restarts))
    for r in range(nr):
        rng = np.random.RandomState(int(base_seed) + r)
        print(f"  Restart {r + 1}/{nr}  seed={int(base_seed) + r}")
        if optimizer == "classic":
            w, mt, gt, f1 = run_search(
                matrices,
                is_score_model,
                gt_data,
                all_pids,
                all_labels,
                n_iter,
                rng,
                verbose=verbose_first_restart and r == 0,
            )
        else:
            w, mt, gt, f1 = run_vns_search(
                matrices,
                is_score_model,
                gt_data,
                all_pids,
                all_labels,
                n_iter,
                rng,
                verbose=verbose_first_restart and r == 0,
            )
        restart_preds.append(
            weighted_ensemble_predict(matrices, is_score_model, w, mt, gt, all_pids, all_labels),
        )
        if best_f1 is None or f1 > best_f1:
            best_w, best_mt, best_gt, best_f1 = w, mt, gt, f1
    assert best_w is not None and best_f1 is not None
    return best_w, best_mt, best_gt, float(best_f1), restart_preds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensemble metaheuristic: full validation pipeline (edit WEIGHTED_RESTARTS / "
        "ROUTING_SWEEP_STEPS / EMBEDDING_* in __main__.py to tune).",
    )
    parser.add_argument("--config", default=EXPERIMENT_CFG, help="Path to experiment YAML (models, data paths).")
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10000,
        help="Weighted search budget per restart: classic = that many evals (75%% random + 25%% hill "
        "climb); vns ≈ that many evals (init random + shake/local loop).",
    )
    parser.add_argument(
        "--weighted-search",
        choices=("classic", "vns", "both"),
        default="both",
        help="Weighted optimizers: 'classic' (random + hill climb), 'vns' (Variable Neighborhood Search), "
        "or 'both' (run each; higher micro-F1 sets weights for combinations; restarts pooled for majority vote).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed for weighted search restarts and KMeans.",
    )
    parser.add_argument(
        "--cluster-sweeps",
        action="store_true",
        help="Run per-cluster sweeps: validation routing, train score-matrix routing, and text-embedding "
        "routing. Skipped by default (slow / noisy).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    val_path = str(get_cfg(cfg, "data.val_path"))
    train_jsonl_path = str(get_cfg(cfg, "data.train_path", "data/processed/training_set.jsonl"))
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
    best_single_name = min(names, key=lambda n: (-individual_micro_f1[n], names.index(n)))
    best_single_f1 = individual_micro_f1[best_single_name]

    per_model_preds: Dict[str, Dict[int, List[str]]] = {
        name: {
            pid: [all_labels[j] for j in np.where(mat[i] >= (1.0 if is_score else 0.5))[0]]
            for i, pid in enumerate(all_pids)
        }
        for name, mat, is_score in zip(names, matrices, is_score_model)
    }

    train_bundle: tuple | None = None
    train_load_error: str | None = None
    try:
        tr_gt, tr_pids, tr_mats, tr_path = load_train_matrices(
            args.config,
            model_cfgs,
            all_labels,
        )
        tr_preds = build_per_model_preds(
            tr_mats, names, is_score_model, tr_pids, all_labels,
        )
        train_bundle = (tr_gt, tr_pids, tr_mats, tr_path, tr_preds)
    except FileNotFoundError as exc:
        train_load_error = str(exc)

    # --- Weighted search (classic and/or VNS) ---
    print(
        f"\n--- Weighted search ({args.weighted_search}) "
        f"({args.n_iter} iters × {WEIGHTED_RESTARTS} restart(s) per optimizer) ---",
    )
    fusion_from: str = str(args.weighted_search)
    restart_weighted_preds: List[Dict[int, List[str]]] = []
    best_w: np.ndarray | None = None
    best_mt: np.ndarray | None = None
    best_gt: float | None = None
    best_f1: float | None = None
    classic_f1: float | None = None
    vns_f1: float | None = None
    bc_w = bc_mt = bc_gt = bv_w = bv_mt = bv_gt = None

    if args.weighted_search in ("classic", "both"):
        print("\n  --- Weighted search — classic (random + hill climb) ---")
        bc_w, bc_mt, bc_gt, classic_f1, rp_c = _run_weighted_restarts(
            "classic",
            matrices,
            is_score_model,
            gt_data,
            all_pids,
            all_labels,
            args.n_iter,
            int(args.seed),
            WEIGHTED_RESTARTS,
            verbose_first_restart=True,
        )
        print(f"  Micro-F1 (best restart)={classic_f1:.4f}")

    if args.weighted_search in ("vns", "both"):
        print("\n  --- Weighted search — VNS ---")
        bv_w, bv_mt, bv_gt, vns_f1, rp_v = _run_weighted_restarts(
            "vns",
            matrices,
            is_score_model,
            gt_data,
            all_pids,
            all_labels,
            args.n_iter,
            int(args.seed),
            WEIGHTED_RESTARTS,
            verbose_first_restart=True,
        )
        print(f"  Micro-F1 (best restart)={vns_f1:.4f}")

    if args.weighted_search == "classic":
        assert bc_w is not None and classic_f1 is not None
        best_w, best_mt, best_gt, best_f1 = bc_w, bc_mt, bc_gt, classic_f1
        restart_weighted_preds = rp_c
    elif args.weighted_search == "vns":
        assert bv_w is not None and vns_f1 is not None
        best_w, best_mt, best_gt, best_f1 = bv_w, bv_mt, bv_gt, vns_f1
        restart_weighted_preds = rp_v
    else:
        assert bc_w is not None and bv_w is not None
        assert classic_f1 is not None and vns_f1 is not None
        restart_weighted_preds = rp_c + rp_v
        if vns_f1 > classic_f1:
            best_w, best_mt, best_gt, best_f1 = bv_w, bv_mt, bv_gt, vns_f1
            fusion_from = "vns"
        else:
            best_w, best_mt, best_gt, best_f1 = bc_w, bc_mt, bc_gt, classic_f1
            fusion_from = "classic"
        print(
            f"\n  Using {fusion_from} weights for downstream combinations "
            f"(classic={classic_f1:.4f}, vns={vns_f1:.4f})",
        )

    assert best_w is not None and best_mt is not None and best_gt is not None and best_f1 is not None

    embed_cluster_rows: List[tuple] = []
    embed_cluster_train_rows: List[tuple] = []
    embed_cluster_text_rows: List[tuple] = []
    if not args.cluster_sweeps:
        print(
            "\n--- Per-cluster sweeps ---\n"
            "  Skipped (pass --cluster-sweeps for validation routing, train score routing, and text routing).",
        )
    else:
        print(
            "\n--- Per-cluster — validation clustering (champion from val labels; optimistic) ---\n"
            f"  methods={list(EMBEDDING_CLUSTER_METHODS)}, "
            f"K={EMBEDDING_K_LIST[0]}…{EMBEDDING_K_LIST[-1]} n={len(EMBEDDING_K_LIST)} values",
        )
        embed_cluster_rows = run_score_matrix_cluster_sweep(
            matrices,
            all_pids,
            names,
            per_model_preds,
            gt_data,
            all_labels,
            EMBEDDING_K_LIST,
            EMBEDDING_CLUSTER_METHODS,
            int(args.seed),
        )
        if not embed_cluster_rows:
            print("  Skipped (no clustering results; check matrices / methods / K range).")
        else:
            for meth, k, m in embed_cluster_rows:
                print(
                    f"  {meth:<16} K={k:3d}  micro-F1={m['micro_f1']:.4f}  "
                    f"P={m['precision']:.4f}  R={m['recall']:.4f}",
                )

        print(
            "\n--- Per-cluster — train-only routing (cluster + champion on train; score val once) ---",
        )
        if train_bundle is None:
            print(f"  Skipped ({train_load_error or 'training data unavailable'}).")
        else:
            tr_gt, tr_pids, tr_mats, tr_path, tr_preds = train_bundle
            print(f"  train_path={tr_path}  n_train={len(tr_pids)}")
            embed_cluster_train_rows = run_score_matrix_cluster_sweep_train_routing(
                tr_mats,
                matrices,
                tr_pids,
                all_pids,
                names,
                tr_preds,
                per_model_preds,
                tr_gt,
                gt_data,
                all_labels,
                EMBEDDING_K_LIST,
                EMBEDDING_CLUSTER_METHODS,
                int(args.seed),
            )
            if not embed_cluster_train_rows:
                print("  No results (clustering failures or empty routing on train).")
            else:
                for meth, k, m in embed_cluster_train_rows:
                    print(
                        f"  {meth:<16} K={k:3d}  micro-F1={m['micro_f1']:.4f}  "
                        f"P={m['precision']:.4f}  R={m['recall']:.4f}",
                    )

        print(
            "\n--- Per-cluster — text embeddings (train-only routing) ---\n"
            f"  methods={list(TEXT_CLUSTER_METHODS)}, "
            f"K={EMBEDDING_K_LIST[0]}…{EMBEDDING_K_LIST[-1]} n={len(EMBEDDING_K_LIST)} values",
        )
        if train_bundle is None:
            print(f"  Skipped ({train_load_error or 'training data unavailable'}).")
        else:
            tr_gt, tr_pids, _tr_mats, _tr_path, tr_preds = train_bundle
            try:
                embed_cluster_text_rows = run_text_embedding_cluster_sweep_train_routing(
                    cfg,
                    train_jsonl_path,
                    val_path,
                    tr_pids,
                    all_pids,
                    names,
                    tr_preds,
                    per_model_preds,
                    tr_gt,
                    gt_data,
                    all_labels,
                    EMBEDDING_K_LIST,
                    TEXT_CLUSTER_METHODS,
                    int(args.seed),
                )
            except Exception as exc:
                print(f"  Failed: {exc}")
                embed_cluster_text_rows = []
            if not embed_cluster_text_rows:
                print(
                    "  No results (PyTorch/transformers missing, embedding error, or empty routing).",
                )
            else:
                for meth, k, m in embed_cluster_text_rows:
                    print(
                        f"  {meth:<16} K={k:3d}  micro-F1={m['micro_f1']:.4f}  "
                        f"P={m['precision']:.4f}  R={m['recall']:.4f}",
                    )

    m_pp_score = None
    m_pp_knn = None

    print(
        "\n--- Per-patient routing — score-only "
        f"(policy={PATIENT_SCORE_ROUTING_POLICY}) ---",
    )
    pr_score = per_patient_champion_from_scores(
        matrices, names, all_pids, policy=PATIENT_SCORE_ROUTING_POLICY,
    )
    sweep_pp = np.linspace(0.72, 1.18, int(ROUTING_SWEEP_STEPS))
    best_pp_s_f1, best_pp_s_cut, best_pp_s_preds = -1.0, 1.0, {}
    for cut in sweep_pp:
        rp = per_patient_routed_predict(
            matrices,
            is_score_model,
            names,
            all_pids,
            all_labels,
            pr_score,
            score_cutoff=float(cut),
        )
        rf = evaluate_data(gt_data, rp, label_space=all_labels)["micro_f1"]
        if rf > best_pp_s_f1:
            best_pp_s_f1, best_pp_s_cut, best_pp_s_preds = rf, float(cut), rp
    m_pp_score = evaluate_data(gt_data, best_pp_s_preds, label_space=all_labels)
    print(f"  Best score-cutoff={best_pp_s_cut:.4f}")
    print(
        f"  Micro-F1={m_pp_score['micro_f1']:.4f}  "
        f"Precision={m_pp_score['precision']:.4f}  Recall={m_pp_score['recall']:.4f}",
    )

    print(f"\n--- Per-patient routing — kNN on train (k={PATIENT_KNN_K}) ---")
    if train_bundle is None:
        print(f"  Skipped ({train_load_error or 'training data unavailable'}).")
    else:
        train_gt, train_pids, train_mats, train_path_used, per_train_preds = train_bundle
        pr_knn = build_patient_routing_knn_train(
            train_mats,
            matrices,
            train_gt,
            train_pids,
            all_pids,
            names,
            all_labels,
            per_train_preds,
            k=PATIENT_KNN_K,
        )
        if not pr_knn:
            print("  Skipped (empty routing).")
        else:
            print(f"  train_path={train_path_used}  n_train={len(train_pids)}")
            best_pp_k_f1, best_pp_k_cut, best_pp_k_preds = -1.0, 1.0, {}
            for cut in sweep_pp:
                rp = per_patient_routed_predict(
                    matrices,
                    is_score_model,
                    names,
                    all_pids,
                    all_labels,
                    pr_knn,
                    score_cutoff=float(cut),
                )
                rf = evaluate_data(gt_data, rp, label_space=all_labels)["micro_f1"]
                if rf > best_pp_k_f1:
                    best_pp_k_f1, best_pp_k_cut, best_pp_k_preds = rf, float(cut), rp
            m_pp_knn = evaluate_data(gt_data, best_pp_k_preds, label_space=all_labels)
            print(f"  Best score-cutoff={best_pp_k_cut:.4f}")
            print(
                f"  Micro-F1={m_pp_knn['micro_f1']:.4f}  "
                f"Precision={m_pp_knn['precision']:.4f}  Recall={m_pp_knn['recall']:.4f}",
            )

    # --- Per-label champion routing ---
    print("\n--- Per-label champion routing ---")
    model_label_f1s = {name: per_label_f1(gt_data, preds, all_labels) for name, preds in per_model_preds.items()}
    label_routing = build_label_routing_table(model_label_f1s, all_labels)

    champion_counts = {}
    for label, champ in label_routing.items():
        champion_counts[champ] = champion_counts.get(champ, 0) + 1
    print("  Labels routed to each model:", {n: champion_counts.get(n, 0) for n in names})

    print(f"  Threshold sweep: {ROUTING_SWEEP_STEPS} score-cutoff values …")
    sweep_cuts = np.linspace(0.72, 1.18, int(ROUTING_SWEEP_STEPS))
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
    print(
        f"  Micro-F1={m_per_label['micro_f1']:.4f}  "
        f"Precision={m_per_label['precision']:.4f}  "
        f"Recall={m_per_label['recall']:.4f}",
    )

    print(
        "\n--- Per-label champion + non-champion vote (OR; sweep cut × min_other) ---\n"
        "  Base = champion fires; else label if ≥min_other other models fire.",
    )
    best_pv_f1, best_pv_cut, best_pv_min_o, plus_vote_preds = -1.0, 1.0, 0, {}
    max_others = max(0, len(names) - 1)
    min_other_grid = tuple(range(1, min(5, max_others + 1))) if max_others > 0 else (0,)
    for cut in sweep_cuts:
        for min_o in min_other_grid:
            pv = per_label_champion_plus_other_vote_predict(
                matrices,
                is_score_model,
                names,
                all_pids,
                all_labels,
                label_routing,
                score_cutoff=float(cut),
                min_other_votes=int(min_o),
            )
            rf = evaluate_data(gt_data, pv, label_space=all_labels)["micro_f1"]
            if rf > best_pv_f1:
                best_pv_f1, best_pv_cut, best_pv_min_o, plus_vote_preds = rf, float(cut), int(min_o), pv
    m_per_label_plus = evaluate_data(gt_data, plus_vote_preds, label_space=all_labels)
    print(
        f"  Best score-cutoff={best_pv_cut:.4f}  min_other_votes={best_pv_min_o}  "
        f"(grid {len(sweep_cuts)}×{len(min_other_grid)})",
    )
    print(
        f"  Micro-F1={m_per_label_plus['micro_f1']:.4f}  "
        f"Precision={m_per_label_plus['precision']:.4f}  "
        f"Recall={m_per_label_plus['recall']:.4f}",
    )

    # --- Correction mode ---
    print(
        "\n--- Correction mode (grid search over add/remove params) ---\n"
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
        extended=False,
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

    extra_rows: List[tuple] = []
    print("\n--- Post-search rules (no stacking / no NN) ---")
    label_support = _label_doc_frequency(gt_data, all_labels)

    if len(restart_weighted_preds) >= 2:
        k_maj = len(restart_weighted_preds) // 2 + 1
        p_mv = merge_preds_k_of_n(restart_weighted_preds, all_pids, k_maj)
        m_mv = evaluate_data(gt_data, p_mv, label_space=all_labels)
        extra_rows.append((f"Majority vote ({len(restart_weighted_preds)} weighted restarts, k={k_maj})", m_mv))
        print(
            f"  Majority vote ({len(restart_weighted_preds)} restarts, k={k_maj}): "
            f"micro-F1={m_mv['micro_f1']:.4f}  P={m_mv['precision']:.4f}  R={m_mv['recall']:.4f}",
        )

    best_g_f1, best_g_gate = -1.0, 0.0
    for gate in np.linspace(0.35, 1.25, num=7):
        p_g = weighted_ensemble_predict_gated_secondary(
            matrices,
            is_score_model,
            names,
            best_w,
            best_mt,
            best_gt,
            all_pids,
            all_labels,
            gate_max_base=float(gate),
        )
        f1g = evaluate_data(gt_data, p_g, label_space=all_labels)["micro_f1"]
        if f1g > best_g_f1:
            best_g_f1, best_g_gate = f1g, float(gate)
    p_g_best = weighted_ensemble_predict_gated_secondary(
        matrices,
        is_score_model,
        names,
        best_w,
        best_mt,
        best_gt,
        all_pids,
        all_labels,
        gate_max_base=best_g_gate,
    )
    m_g = evaluate_data(gt_data, p_g_best, label_space=all_labels)
    extra_rows.append((f"Gated IR/NER (best gate_max_base={best_g_gate:.3f})", m_g))
    print(
        f"  Gated IR/NER (grid, best gate_max_base={best_g_gate:.3f}): "
        f"micro-F1={m_g['micro_f1']:.4f}  P={m_g['precision']:.4f}  R={m_g['recall']:.4f}",
    )

    comb_best = weighted_ensemble_combined_matrix(matrices, is_score_model, best_w, best_mt)
    best_k_f1, best_k = -1.0, 8
    for k in (6, 10, 14, 18):
        p_k = weighted_ensemble_predict_top_k(comb_best, best_gt, all_pids, all_labels, k)
        f1k = evaluate_data(gt_data, p_k, label_space=all_labels)["micro_f1"]
        if f1k > best_k_f1:
            best_k_f1, best_k = f1k, k
    p_kbest = weighted_ensemble_predict_top_k(comb_best, best_gt, all_pids, all_labels, best_k)
    m_k = evaluate_data(gt_data, p_kbest, label_space=all_labels)
    extra_rows.append((f"Top-K after threshold (K={best_k})", m_k))
    print(
        f"  Top-K prune (K∈{{6,10,14,18}}, best K={best_k}): "
        f"micro-F1={m_k['micro_f1']:.4f}  P={m_k['precision']:.4f}  R={m_k['recall']:.4f}",
    )

    p_b = weighted_ensemble_predict_frequency_buckets(
        matrices,
        is_score_model,
        best_w,
        best_mt,
        best_gt,
        all_pids,
        all_labels,
        label_support,
        support_cutoff=25,
        rare_factor=1.08,
        freq_factor=0.97,
    )
    m_b = evaluate_data(gt_data, p_b, label_space=all_labels)
    extra_rows.append(("Rare/freq buckets (cutoff=25, rare×1.08 / freq×0.97)", m_b))
    print(
        f"  Rare/freq threshold buckets: "
        f"micro-F1={m_b['micro_f1']:.4f}  P={m_b['precision']:.4f}  R={m_b['recall']:.4f}",
    )

    p_2t = weighted_ensemble_predict_two_threshold(
        matrices,
        is_score_model,
        best_w,
        best_mt,
        all_pids,
        all_labels,
        t_high=float(best_gt),
        t_low=float(best_gt) * 0.72,
        min_votes=3,
    )
    m_2t = evaluate_data(gt_data, p_2t, label_space=all_labels)
    extra_rows.append(("Two-threshold (t_high=gt, t_low=0.72×gt, min_votes=3)", m_2t))
    print(
        f"  Two-threshold + min votes: "
        f"micro-F1={m_2t['micro_f1']:.4f}  P={m_2t['precision']:.4f}  R={m_2t['recall']:.4f}",
    )

    weighted_preds = weighted_ensemble_predict(
        matrices, is_score_model, best_w, best_mt, best_gt, all_pids, all_labels,
    )
    print("\n--- Label-set fusion ---")
    combo_rows = []
    for title, merged in (
        ("OR  per-label ∪ weighted", merge_preds_union(routed_preds, weighted_preds, all_pids)),
        ("AND per-label ∩ weighted", merge_preds_intersection(routed_preds, weighted_preds, all_pids)),
        ("OR  per-label+vote ∪ weighted", merge_preds_union(plus_vote_preds, weighted_preds, all_pids)),
        ("AND per-label+vote ∩ weighted", merge_preds_intersection(plus_vote_preds, weighted_preds, all_pids)),
        ("OR  per-label ∪ correction", merge_preds_union(routed_preds, corr_preds, all_pids)),
        ("AND per-label ∩ correction", merge_preds_intersection(routed_preds, corr_preds, all_pids)),
        ("OR  weighted ∪ correction", merge_preds_union(weighted_preds, corr_preds, all_pids)),
        ("AND weighted ∩ correction", merge_preds_intersection(weighted_preds, corr_preds, all_pids)),
        (
            "2-of-3 weighted, per-label, correction",
            merge_preds_k_of_n([weighted_preds, routed_preds, corr_preds], all_pids, 2),
        ),
        (
            "3-of-3 weighted, per-label, correction",
            merge_preds_k_of_n([weighted_preds, routed_preds, corr_preds], all_pids, 3),
        ),
    ):
        m = evaluate_data(gt_data, merged, label_space=all_labels)
        combo_rows.append((title, m))
        print(
            f"  {title}: micro-F1={m['micro_f1']:.4f}  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}",
        )

    print("\n=== Results (validation micro-F1) ===")
    if args.weighted_search == "both":
        assert classic_f1 is not None and vns_f1 is not None
        print(f"  Weighted search (classic) : {classic_f1:.4f}")
        print(f"  Weighted search (vns)     : {vns_f1:.4f}")
        print(f"  Weighted search (combinations ← {fusion_from}) : {best_f1:.4f}")
    else:
        print(f"  Weighted search ({args.weighted_search}) : {best_f1:.4f}")
    if embed_cluster_rows:
        best_meth, best_k, best_em = max(embed_cluster_rows, key=lambda x: x[2]["micro_f1"])
        print(
            f"  Per-cluster (val routing; optimistic) best ({best_meth}, K={best_k}) "
            f": {best_em['micro_f1']:.4f}",
        )
    elif args.cluster_sweeps:
        print("  Per-cluster (val routing) : (no successful runs)")
    else:
        print("  Per-cluster (val routing) : (not run — use --cluster-sweeps)")
    if embed_cluster_train_rows:
        best_tm, best_tk, best_te = max(embed_cluster_train_rows, key=lambda x: x[2]["micro_f1"])
        print(
            f"  Per-cluster (train routing) best ({best_tm}, K={best_tk}) "
            f": {best_te['micro_f1']:.4f}",
        )
    elif not args.cluster_sweeps:
        print("  Per-cluster (train routing) : (not run — use --cluster-sweeps)")
    elif train_bundle is not None:
        print("  Per-cluster (train routing) : (no successful runs)")
    else:
        print("  Per-cluster (train routing) : (skipped — no train data)")
    if embed_cluster_text_rows:
        best_txt_m, best_txt_k, best_txt_e = max(embed_cluster_text_rows, key=lambda x: x[2]["micro_f1"])
        print(
            f"  Per-cluster (text embed, train routing) best ({best_txt_m}, K={best_txt_k}) "
            f": {best_txt_e['micro_f1']:.4f}",
        )
    elif not args.cluster_sweeps:
        print("  Per-cluster (text embed, train routing) : (not run — use --cluster-sweeps)")
    elif train_bundle is not None:
        print("  Per-cluster (text embed, train routing) : (no successful runs)")
    if m_pp_score is not None:
        print(
            f"  Per-patient score routing ({PATIENT_SCORE_ROUTING_POLICY}) : "
            f"{m_pp_score['micro_f1']:.4f}",
        )
    if m_pp_knn is not None:
        print(f"  Per-patient kNN train (k={PATIENT_KNN_K})     : {m_pp_knn['micro_f1']:.4f}")
    print(f"  Per-label routing     : {m_per_label['micro_f1']:.4f}")
    print(
        f"  Per-label + other vote: {m_per_label_plus['micro_f1']:.4f}  "
        f"(best cut={best_pv_cut:.4f}, min_other={best_pv_min_o})",
    )
    print(f"  Correction mode       : {m_correction['micro_f1']:.4f}")
    print(f"  Best single model ({best_single_name})  : {best_single_f1:.4f}")
    best_combo_title, best_combo_m = max(combo_rows, key=lambda row: row[1]["micro_f1"])
    print(f"  Best combination ({best_combo_title}) : {best_combo_m['micro_f1']:.4f}")
    if extra_rows:
        best_x_title, best_x_m = max(extra_rows, key=lambda row: row[1]["micro_f1"])
        print(f"  Best extra rule strategy ({best_x_title}) : {best_x_m['micro_f1']:.4f}")
    def _print_weighted_param_block(header: str, w: np.ndarray, mt: np.ndarray, gt: float) -> None:
        print(f"\n{header} — threshold={gt:.4f}")
        mt_iter = iter(mt)
        for name, is_score, weight in zip(names, is_score_model, w):
            mthr = next(mt_iter) if is_score else 0.5
            suffix = f"  act_thr={mthr:.4f}" if is_score else "  act_thr=0.5 (binary)"
            print(f"  {name:<25} weight={weight:.4f}{suffix}")

    if args.weighted_search == "both":
        assert bc_w is not None and bv_w is not None and bc_mt is not None and bv_mt is not None
        _print_weighted_param_block("Weighted search (classic) params", bc_w, bc_mt, float(bc_gt))
        _print_weighted_param_block("Weighted search (VNS) params", bv_w, bv_mt, float(bv_gt))
        print(f"\nCombinations use the {fusion_from} run (higher micro-F1).")
    else:
        _print_weighted_param_block(f"Weighted search ({args.weighted_search}) params", best_w, best_mt, float(best_gt))


if __name__ == "__main__":
    main()
