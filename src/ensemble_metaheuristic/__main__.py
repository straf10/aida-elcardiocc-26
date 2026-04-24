"""
Metaheuristic ensemble: weighted voting over XLM-R-Large, Greek-BERT, XLM-R-Base,
information retrieval, dictionary baseline, and NER-EL.

Each model contributes a (n_docs x n_labels) score matrix:
  - score-based models : scores normalised by per-label threshold (>1.0 = positive)
  - prediction-only    : binary 0/1 votes

``python -m ensemble_metaheuristic`` runs the full validation pipeline (scores **val** micro-F1).
Each model needs **validation** predictions whose ``patient_id`` values match ``data.val_path``: either set
``val_predictions_path`` on that model in ``evaluation/config.yaml``, or place a sidecar file next to
the test predictions: ``outputs/predictions/<name>/val_predictions.jsonl``. Using **test** predictions
(``test_predictions.jsonl``) as if they were validation yields **0.0000** for every model.
If a model's val (or train) prediction file is missing, a **WARNING** is printed and that model is **skipped**;
the run continues with the remaining models (fails only if **none** could be loaded).

  - **Weighted search** — classic and/or VNS (``--weighted-search``; default ``both`` uses
    ``WEIGHTED_RESTARTS`` seeds; best micro-F1 drives fusion + majority vote over restarts).
  - **Per-cluster sweeps** (heavy): validation clustering, train-only score routing, optional Greek-BERT
    text routing. **Off by default**; use ``--cluster-sweeps``. Caches text runs under
    ``outputs/ensemble_metaheuristic/text_cluster_cache/``.
  - **Auxiliary global thresholds**: after the main weighted search, a small val sweep picks looser /
    tighter ``global_threshold`` values (same weights) for extra base exports.
  - **Per-label routing** (val champion per code + score-cutoff sweep) and **correction** (val grid) as
    export bases — different from weighted fusion, so k-of-n / YAML merges can match stronger test runs.
  - Label-set **compositions** in ``strategy_compositions.yaml``, plus an **automatic grid** (module
    ``combination_grid``): unions, intersections, and k-of-n over base-slug subsets (sizes
    ``--combo-sizes``; optional ``--combo-preset`` subsets; ``--combo-k-all`` / ``--combo-k-extra`` for
    k sweep; ``--combo-max-specs`` cap; val-ranked; top ``--combo-grid-top`` exported).

To run **one** strategy end-to-end, use the matching module, for example
``python -m ensemble_metaheuristic.strategies.weighted_strategy`` (see ``--help`` on each).

Classic search is ``strategies.weighted_strategy`` (``run_search``); VNS is ``strategies.weighted_vns_strategy``
(``run_vns_search``). ``strategies.weighted_search`` is a re-export shim for older imports.

CLI: ``--config``, ``--n-iter``, ``--seed``, ``--weighted-search classic|vns|both``, ``--cluster-sweeps``,
``--export-dir``, ``--no-export-predictions``, ``--no-combo-grid``, ``--combo-sizes``, ``--combo-preset``,
``--combo-k-all``, ``--combo-k-extra``, ``--combo-max-specs``, ``--combo-grid-top N``.

Subset / ablation sweeps on val (weighted search only): ``python -m ensemble_metaheuristic.weighted_subset_sweep``.
Leave-one-out **test-drag** flags (val-tuned weighted fusion): ``python -m ensemble_metaheuristic.committee_member_harm``.

By default the run writes one folder per strategy under ``--export-dir/<slug>/`` with
``test_predictions.jsonl`` and ``blind_predictions.jsonl`` (blind empty codes if a strategy fails on blind).
See ``manifest.json`` in that directory. Point ``ensemble_metaheuristic`` in ``evaluation/config.yaml`` at the best slug under
``outputs/predictions/ensemble_metaheuristic/<slug>/test_predictions.jsonl`` for ``compare_methods``.

Tune constants in this file if needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_src = Path(__file__).resolve().parents[1]
if _src.name == "src" and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np

from evaluation.config_utils import load_config, get_cfg
from evaluation.evaluator import evaluate_data
from evaluation.io_utils import load_ground_truth
from ensemble_metaheuristic.matrices import build_score_matrix, load_thresholds_for_model
from ensemble_metaheuristic.strategy_loaders import (
    build_per_model_preds,
    canonical_ensemble_label_arts,
    gather_ensemble_artifacts,
    load_train_matrices,
)
from ensemble_metaheuristic.strategies import (
    build_label_routing_table,
    correction_predict,
    merge_preds_k_of_n,
    per_label_f1,
    per_label_routed_predict,
    run_score_matrix_cluster_sweep,
    run_score_matrix_cluster_sweep_train_routing,
    run_text_embedding_cluster_sweep_train_routing,
    run_search,
    run_vns_search,
    search_correction_params,
    weighted_ensemble_combined_matrix,
    weighted_ensemble_predict,
    weighted_ensemble_predict_frequency_buckets,
    weighted_ensemble_predict_top_k,
)

EXPERIMENT_CFG = "src/evaluation/config.yaml"


def _parse_csv_ints(s: str) -> Tuple[int, ...]:
    t = str(s).strip()
    if not t:
        return ()
    return tuple(int(p.strip()) for p in t.split(",") if p.strip())


WEIGHTED_RESTARTS = 2
ROUTING_SWEEP_LABEL = 24
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
) -> Tuple[np.ndarray, np.ndarray, float, float, List[Dict[int, List[str]]], List[Tuple[np.ndarray, np.ndarray, float]]]:
    """Run ``run_search`` (classic) or ``run_vns_search`` (vns) for ``n_restarts`` seeds."""
    best_w = best_mt = best_gt = best_f1 = None
    restart_preds: List[Dict[int, List[str]]] = []
    restart_wmt: List[Tuple[np.ndarray, np.ndarray, float]] = []
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
        restart_wmt.append((w.copy(), mt.copy(), float(gt)))
        if best_f1 is None or f1 > best_f1:
            best_w, best_mt, best_gt, best_f1 = w, mt, gt, f1
    assert best_w is not None and best_f1 is not None
    return best_w, best_mt, best_gt, float(best_f1), restart_preds, restart_wmt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensemble metaheuristic: full validation pipeline (edit WEIGHTED_RESTARTS / "
        "EMBEDDING_* in __main__.py to tune).",
    )
    parser.add_argument("--config", default=EXPERIMENT_CFG, help="Path to evaluation config.yaml (models, data paths).")
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
    parser.add_argument(
        "--export-dir",
        default="outputs/predictions/ensemble_metaheuristic",
        help="Directory for val_predictions.jsonl, test_predictions.jsonl, blind_predictions.jsonl (when inputs exist).",
    )
    parser.add_argument(
        "--no-export-predictions",
        action="store_true",
        help="Do not write ensemble JSONL outputs after the run.",
    )
    parser.add_argument(
        "--no-combo-grid",
        action="store_true",
        help="Skip automatic union / intersection / k-of-n over base strategies (see --combo-sizes).",
    )
    parser.add_argument(
        "--combo-sizes",
        default="2,3,4",
        help="Comma-separated subset sizes for the auto combo grid (e.g. 2,3,4,5). Each n uses C(|bases|,n) subsets.",
    )
    parser.add_argument(
        "--combo-preset",
        action="append",
        default=[],
        metavar="NAME",
        help="Repeatable: run the grid on a named subset of bases (see ensemble_metaheuristic.combination_grid "
        "COMBO_PRESETS). Concatenate and dedupe; if omitted, all bases present on val are used.",
    )
    parser.add_argument(
        "--combo-k-all",
        action="store_true",
        help="For k-of-n, evaluate every k from 1 through n (wide search; many specs when n is large).",
    )
    parser.add_argument(
        "--combo-k-extra",
        default="",
        metavar="K_LIST",
        help="Comma-separated extra k values (each used only when 1<=k<=n). Majority k is always included unless "
        "--combo-k-all is set.",
    )
    parser.add_argument(
        "--combo-max-specs",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of combo specs (deterministic order); omit for no cap.",
    )
    parser.add_argument(
        "--combo-grid-top",
        type=int,
        default=25,
        help="Export the best N auto-grid combos by validation micro-F1 (0 = evaluate and print only).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    val_path = str(get_cfg(cfg, "data.val_path"))
    train_jsonl_path = str(get_cfg(cfg, "data.train_path", "data/processed/train.jsonl"))
    gt_data = load_ground_truth(val_path)
    all_pids = list(gt_data.keys())
    model_cfgs = {m["name"]: m for m in get_cfg(cfg, "models", [])}

    print("Loading model artifacts (missing val prediction files → WARNING and skip)...")
    artifacts_list = gather_ensemble_artifacts(model_cfgs, all_pids, "val")
    if not artifacts_list:
        print(
            "ERROR: no ensemble models had validation predictions. "
            "Run: PYTHONPATH=src python -m evaluation.run_predictions",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    for name, arts in artifacts_list:
        print(f"  {name}: {'scores' if arts.scores is not None else 'binary predictions'}")

    all_labels = canonical_ensemble_label_arts(artifacts_list).label_names
    ensemble_model_names = [n for n, _ in artifacts_list]

    print(f"\nBuilding score matrices ({len(all_pids)} docs x {len(all_labels)} labels)...")
    matrices, names, is_score_model = [], [], []
    for name, arts in artifacts_list:
        thr = load_thresholds_for_model(model_cfgs[name], all_labels) if arts.scores is not None else None
        matrices.append(build_score_matrix(arts, all_pids, all_labels, thr))
        names.append(name)
        is_score_model.append(arts.scores is not None)

    print("\nIndividual model micro-F1:")
    individual_micro_f1: Dict[str, float] = {}
    for name, arts in artifacts_list:
        preds = {pid: list(arts.pred_data.get(pid, [])) for pid in all_pids}
        f1 = evaluate_data(gt_data, preds, label_space=arts.label_names)["micro_f1"]
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
            model_names=ensemble_model_names,
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
    restart_triples: List[Tuple[np.ndarray, np.ndarray, float]] = []
    best_w: np.ndarray | None = None
    best_mt: np.ndarray | None = None
    best_gt: float | None = None
    best_f1: float | None = None
    classic_f1: float | None = None
    vns_f1: float | None = None
    bc_w = bc_mt = bc_gt = bv_w = bv_mt = bv_gt = None
    rp_c = triples_c = None
    rp_v = triples_v = None

    if args.weighted_search in ("classic", "both"):
        print("\n  --- Weighted search — classic (random + hill climb) ---")
        bc_w, bc_mt, bc_gt, classic_f1, rp_c, triples_c = _run_weighted_restarts(
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
        bv_w, bv_mt, bv_gt, vns_f1, rp_v, triples_v = _run_weighted_restarts(
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
        assert bc_w is not None and classic_f1 is not None and rp_c is not None and triples_c is not None
        best_w, best_mt, best_gt, best_f1 = bc_w, bc_mt, bc_gt, classic_f1
        restart_weighted_preds = rp_c
        restart_triples = triples_c
    elif args.weighted_search == "vns":
        assert bv_w is not None and vns_f1 is not None and rp_v is not None and triples_v is not None
        best_w, best_mt, best_gt, best_f1 = bv_w, bv_mt, bv_gt, vns_f1
        restart_weighted_preds = rp_v
        restart_triples = triples_v
    else:
        assert bc_w is not None and bv_w is not None
        assert classic_f1 is not None and vns_f1 is not None
        assert rp_c is not None and triples_c is not None and rp_v is not None and triples_v is not None
        restart_weighted_preds = rp_c + rp_v
        restart_triples = triples_c + triples_v
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

    print("\n--- Auxiliary global thresholds (validation; same ensemble weights) ---")
    gt0 = float(best_gt)
    best_loose_gt, best_loose_f1 = gt0, -1.0
    for mul in np.linspace(0.80, 0.99, 15):
        g = gt0 * float(mul)
        preds = weighted_ensemble_predict(
            matrices, is_score_model, best_w, best_mt, float(g), all_pids, all_labels,
        )
        f1 = evaluate_data(gt_data, preds, label_space=all_labels)["micro_f1"]
        if f1 > best_loose_f1:
            best_loose_f1, best_loose_gt = f1, float(g)
    best_tight_gt, best_tight_f1 = gt0, -1.0
    for mul in np.linspace(1.01, 1.15, 15):
        g = gt0 * float(mul)
        preds = weighted_ensemble_predict(
            matrices, is_score_model, best_w, best_mt, float(g), all_pids, all_labels,
        )
        f1 = evaluate_data(gt_data, preds, label_space=all_labels)["micro_f1"]
        if f1 > best_tight_f1:
            best_tight_f1, best_tight_gt = f1, float(g)
    print(f"  Loose best: global_thr={best_loose_gt:.4f}  micro-F1={best_loose_f1:.4f}")
    print(f"  Tight best: global_thr={best_tight_gt:.4f}  micro-F1={best_tight_f1:.4f}")

    print("\n--- Per-label routing (val champion per code; export base) ---")
    model_label_f1s = {name: per_label_f1(gt_data, preds, all_labels) for name, preds in per_model_preds.items()}
    label_routing = build_label_routing_table(model_label_f1s, all_labels)
    sweep_cuts_pl = np.linspace(0.72, 1.18, int(ROUTING_SWEEP_LABEL))
    best_pl_f1, best_r_cut, best_pl_preds = -1.0, 1.0, {}
    for cut in sweep_cuts_pl:
        rp = per_label_routed_predict(
            matrices,
            is_score_model,
            names,
            all_pids,
            all_labels,
            label_routing,
            score_cutoff=float(cut),
        )
        rf = evaluate_data(gt_data, rp, label_space=all_labels)["micro_f1"]
        if rf > best_pl_f1:
            best_pl_f1, best_r_cut, best_pl_preds = rf, float(cut), rp
    m_per_label_base = evaluate_data(gt_data, best_pl_preds, label_space=all_labels)
    print(
        f"  best score-cutoff={best_r_cut:.4f}  micro-F1={m_per_label_base['micro_f1']:.4f}  "
        f"P={m_per_label_base['precision']:.4f}  R={m_per_label_base['recall']:.4f}",
    )

    print("\n--- Correction (grid on val; export base) ---")
    best_corr_cfg, _ = search_correction_params(
        matrices,
        is_score_model,
        names,
        all_pids,
        all_labels,
        gt_data,
        base_model=best_single_name,
        extended=False,
    )
    best_corr_cfg.pop("_grid_evaluations", None)
    _ck = ("add_min_votes", "add_min_score_factor", "remove_if_zero_votes")
    correction_export_kw = {k: best_corr_cfg[k] for k in _ck if k in best_corr_cfg}
    corr_preds_val = correction_predict(
        matrices,
        is_score_model,
        names,
        all_pids,
        all_labels,
        base_model=best_single_name,
        **correction_export_kw,
    )
    m_correction_base = evaluate_data(gt_data, corr_preds_val, label_space=all_labels)
    print(f"  best params: {correction_export_kw}")
    print(
        f"  micro-F1={m_correction_base['micro_f1']:.4f}  "
        f"P={m_correction_base['precision']:.4f}  R={m_correction_base['recall']:.4f}",
    )

    from ensemble_metaheuristic.export_strategies import StrategyExportContext
    from ensemble_metaheuristic.strategy_bases import BASE_STRATEGY_ORDER, build_base_strategy_functions
    from ensemble_metaheuristic.strategy_compositions import load_composition_specs, try_apply_composition

    restart_triples_export = restart_triples if len(restart_triples) >= 2 else None
    export_ctx = StrategyExportContext(
        config_path=str(args.config),
        model_cfgs=model_cfgs,
        names=list(names),
        is_score_model=list(is_score_model),
        all_labels=all_labels,
        export_root=Path(args.export_dir),
        best_w=best_w,
        best_mt=best_mt,
        best_gt=float(best_gt),
        fusion_label=str(fusion_from),
        restart_triples=restart_triples_export,
        best_k=int(best_k),
        label_support=label_support,
        weighted_aux_gt_loose=float(best_loose_gt),
        weighted_aux_gt_tight=float(best_tight_gt),
        best_single_name=str(best_single_name),
        label_routing=label_routing,
        best_r_cut=float(best_r_cut),
        correction_export_kw=dict(correction_export_kw),
    )
    base_registry = build_base_strategy_functions(export_ctx)
    base_val_preds: Dict[str, Dict[int, List[str]]] = {}
    for slug in BASE_STRATEGY_ORDER:
        if slug not in base_registry:
            continue
        try:
            base_val_preds[slug] = base_registry[slug](matrices, all_pids)
        except Exception as exc:
            print(f"  [skip base {slug}] {exc!r}", flush=True)

    print("\n--- Base strategies (validation micro-F1) ---")
    for slug in BASE_STRATEGY_ORDER:
        if slug not in base_val_preds:
            continue
        m = evaluate_data(gt_data, base_val_preds[slug], label_space=all_labels)
        print(
            f"  {slug:<40} micro-F1={m['micro_f1']:.4f}  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}",
        )

    composition_rows: List[Tuple[str, dict]] = []
    specs = load_composition_specs()
    print("\n--- Composed strategies (strategy_compositions.yaml) ---")
    if not specs:
        print("  (no compositions file or empty list — edit strategy_compositions.yaml)")
    for spec in specs:
        merged = try_apply_composition(spec, base_val_preds, all_pids)
        if merged is None:
            miss = [x for x in spec.inputs if x not in base_val_preds]
            print(f"  {spec.slug}: skipped (missing bases: {miss})")
            continue
        m = evaluate_data(gt_data, merged, label_space=all_labels)
        composition_rows.append((spec.slug, m))
        op_s = spec.op if spec.op != "k_of_n" else f"k_of_n k={spec.k}"
        print(
            f"  {spec.slug:<40} op={op_s}  inputs={spec.inputs}  "
            f"micro-F1={m['micro_f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}",
        )

    auto_combo_rows: List[Tuple[str, dict]] = []
    export_ctx.auto_export_specs = None
    if not args.no_combo_grid:
        from ensemble_metaheuristic.combination_grid import (
            enumerate_auto_combo_specs,
            enumerate_combo_specs_multi_presets,
            evaluate_combo_grid,
            top_specs_by_micro_f1,
        )

        grid_bases = tuple(s for s in BASE_STRATEGY_ORDER if s in base_val_preds)
        combo_sizes = tuple(n for n in _parse_csv_ints(args.combo_sizes) if n >= 2)
        if not combo_sizes:
            combo_sizes = (2, 3, 4)
        extra_k = _parse_csv_ints(args.combo_k_extra)
        cap = args.combo_max_specs
        presets = tuple(str(p).strip() for p in (args.combo_preset or []) if str(p).strip())
        k_all = bool(args.combo_k_all)
        if presets:
            auto_specs, combo_warn = enumerate_combo_specs_multi_presets(
                grid_bases,
                presets,
                sizes=combo_sizes,
                k_all=k_all,
                extra_k=extra_k,
                max_specs=cap,
            )
            for w in combo_warn:
                print(f"  WARNING: {w}", flush=True)
        else:
            auto_specs = enumerate_auto_combo_specs(
                grid_bases,
                sizes=combo_sizes,
                k_all=k_all,
                extra_k=extra_k,
                max_specs=cap,
            )
        auto_scored = evaluate_combo_grid(auto_specs, base_val_preds, all_pids, gt_data, all_labels)
        sz_s = ",".join(str(x) for x in combo_sizes)
        k_desc = "k=1..n" if k_all else ("k=majority" + (f" + extra {list(extra_k)}" if extra_k else ""))
        preset_s = f"preset={list(presets)}" if presets else "all val bases"
        cap_s = f" cap={cap}" if cap is not None else ""
        print(
            f"\n--- Auto combination grid (sizes [{sz_s}]; {k_desc}; {preset_s}{cap_s}; "
            f"n_specs={len(auto_specs)} evaluated={len(auto_scored)}) ---",
        )
        auto_scored.sort(key=lambda x: (-float(x[1]["micro_f1"]), x[0].slug))
        auto_combo_rows = [(spec.slug, m) for spec, m in auto_scored]
        for spec, m in auto_scored[:35]:
            op_s = spec.op if spec.op != "k_of_n" else f"k_of_n k={spec.k}"
            print(
                f"  {spec.slug:<52} op={op_s}  micro-F1={m['micro_f1']:.4f}  "
                f"P={m['precision']:.4f}  R={m['recall']:.4f}",
            )
        if args.combo_grid_top > 0 and auto_scored:
            export_ctx.auto_export_specs = top_specs_by_micro_f1(auto_scored, int(args.combo_grid_top))
            print(
                f"\n  (exporting top {len(export_ctx.auto_export_specs)} auto combos by val micro-F1 "
                f"under ac_* slugs — use --combo-grid-top 0 to skip)",
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
    print(f"  Best single model ({best_single_name})  : {best_single_f1:.4f}")
    print(f"  Per-label routing (export base) : {m_per_label_base['micro_f1']:.4f}")
    print(f"  Correction (export base)        : {m_correction_base['micro_f1']:.4f}")
    yaml_comp_slugs = {r[0] for r in composition_rows}
    all_comp_rows = list(composition_rows) + list(auto_combo_rows)
    if all_comp_rows:
        best_comp_slug, best_comp_m = max(all_comp_rows, key=lambda row: row[1]["micro_f1"])
        src = "YAML" if best_comp_slug in yaml_comp_slugs else "auto-grid"
        print(f"  Best composition ({src}: {best_comp_slug}) : {best_comp_m['micro_f1']:.4f}")
    else:
        print("  Best composition : (none evaluated)")
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

    if not args.no_export_predictions:
        from ensemble_metaheuristic.export_strategies import export_all_strategy_subfolders

        manifest = export_all_strategy_subfolders(export_ctx)
        print(
            f"\n--- Ensemble JSONL export (per-strategy subfolders under {args.export_dir}; "
            "test_predictions.jsonl + blind_predictions.jsonl each) ---",
        )
        print(f"  manifest: {manifest.get('export_root', '')}/manifest.json")
        print(f"  bases: {', '.join(manifest.get('base_strategies', []))}")
        print(f"  composed: {', '.join(manifest.get('composed_strategies', []))}")
        au = manifest.get("auto_composed_strategies") or []
        if au:
            print(f"  auto_composed ({len(au)}): {', '.join(au)}")
        print(f"  all folders: {', '.join(manifest.get('strategies', []))}")


if __name__ == "__main__":
    main()
