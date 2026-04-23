"""Stacking ensemble: meta-learner on top of base model scores.

Each base model's (n_docs × n_labels) score matrix is used as input features
for a meta-classifier.  Four learner families can be compared in a single run
(``--meta-learner all``):

  * logistic_regression  — fast, interpretable, well-regularised; CPU.
  * random_forest        — captures non-linear interactions; CPU, multi-core.
  * gradient_boosting    — HistGradientBoostingClassifier; strongest CPU option.
  * pytorch_mlp          — shared MLP trained on GPU (CUDA / MPS / CPU fallback).
                           Use ``--device cuda`` to force GPU.

After predicting stacking probabilities on the validation set, thresholds are
optimised to maximise micro-F1:

  * global    — one threshold for all labels (uses official group-level metric).
  * per_label — per-label thresholds via vectorised binary-F1 sweep (fast
                approximation; evaluated with the official metric afterwards).

Usage
-----
PYTHONPATH=src python -m ensemble_stacking \\
    [--config src/evaluation/config.yaml] \\
    [--meta-learner logistic_regression|random_forest|gradient_boosting|pytorch_mlp|all] \\
    [--device auto|cuda|mps|cpu] \\
    [--threshold-mode global|per_label|both] \\
    [--n-threshold-steps 100] \\
    [--seed 42] \\
    [--export-dir outputs/predictions/ensemble_stacking] \\
    [--no-export-predictions]

GPU quick-start
---------------
# Auto-detect GPU:
PYTHONPATH=src python -m ensemble_stacking --meta-learner pytorch_mlp

# Force CUDA:
PYTHONPATH=src python -m ensemble_stacking --meta-learner pytorch_mlp --device cuda

# Long GPU run (phase-1 + val early stopping, then train+val refit; auto-picks CUDA if available):
PYTHONPATH=src python -m ensemble_stacking --meta-learner pytorch_mlp --mlp-long-run --device cuda

Train longer (meta-learners)
----------------------------
# e.g. more MLP epochs + wider hidden stack + deeper trees for RF/HGB:
PYTHONPATH=src python -m ensemble_stacking --device cuda \\
    --mlp-epochs 150 --mlp-hidden 128,64,32 \\
    --rf-n-estimators 300 --hgb-max-iter 250

# MLP: long run + val early stopping (restores best val BCE weights):
PYTHONPATH=src python -m ensemble_stacking --device cuda --meta-learner pytorch_mlp \\
    --mlp-epochs 200 --mlp-early-stop-patience 15 --mlp-hidden 128,64,32

Richer stacking (no new base models)
------------------------------------
Try ``--meta-features rich`` or ``full`` (pairwise score products, disagreement stats, optional
doc-level committee mean). For ``pytorch_mlp``, ``--mlp-label-emb 16`` adds a per-code embedding
so fusion is label-specific. Tune ``--logreg-c`` for per-label logistic regression.

**Patient clustering (like metaheuristic routing cues):** ``--patient-clusters-k K`` fits
``KMeans`` on **train** rows of concatenated score matrices and appends cluster one-hot to
every meta-feature row so the stacker can learn cluster-dependent fusion (no val labels used
to build clusters).

**Single “unified” meta vector:** ``--stacking-unified`` sets ``--meta-features unified`` (full
+ explicit voting statistics) and, unless you set ``--patient-clusters-k`` yourself, defaults
clusters to 8 — **one** meta-learner sees clusters, per-label score rows, and voting columns
together.  ``--val-pick`` chooses the best threshold/learner on val using micro-F1, mean
per-label macro-F1, worst cluster micro-F1, or a composite of those (same exported model).

Higher F1 (committee)
---------------------
``logistic_regression`` often beats MLP here. The largest gain is usually **another strong base
model** (e.g. set ``use_in_ensemble: true`` for ``xlm_r_large`` when train/val/test JSONL exist).

Requirements
------------
Each base model in ``evaluation/config.yaml`` must have both val **and** train
prediction JSONL files on disk.  Val files are needed to evaluate; train files
are needed to fit the meta-learners.

  outputs/predictions/<name>/val_predictions.jsonl
  outputs/predictions/<name>/train_predictions.jsonl

(Or set ``val_predictions_path`` / ``train_predictions_path`` per model entry.)

``evaluation.run_predictions`` writes train/val/test (and blind when raw blind exists) for each track.
If train JSONL is missing, run:

  PYTHONPATH=src python -m evaluation.run_predictions

Notes
-----
Train matrices are built from binary JSONL predictions (0/1 per code per doc).
Val matrices are built from continuous normalised scores where available (the
score-based models xlm_r_large, mlc_greek_bert), otherwise 0/1.  This means
the meta-learner trains on binary features and predicts on partially-continuous
features; in practice this works well because the learned decision boundary
generalises.  Test / blind predictions use binary features (JSONL only) which
matches the training distribution.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_src = Path(__file__).resolve().parents[1]
if _src.name == "src" and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np

from evaluation.config_utils import get_cfg, load_config
from evaluation.io_utils import load_ground_truth, save_predictions_jsonl
from evaluation.scoring import evaluate_data
from ensemble_metaheuristic.matrices import build_score_matrix, load_thresholds_for_model
from ensemble_metaheuristic.strategy_loaders import (
    canonical_ensemble_label_arts,
    gather_ensemble_artifacts,
)
from ensemble_stacking.meta_learners import LEARNER_NAMES, PyTorchMLPStacker, make_stacker
from ensemble_stacking.threshold_opt import (
    proba_to_preds,
    proba_to_preds_per_label,
    sweep_global_threshold,
    sweep_per_label_thresholds,
)
from ensemble_stacking.val_pick import val_pick_score

EXPERIMENT_CFG = "src/evaluation/config.yaml"


def _load_matrices_for_split(
    model_cfgs: Dict,
    names: List[str],
    is_score_model: List[bool],
    pids: List[int],
    all_labels: List[str],
    split: str,
) -> List[np.ndarray]:
    """Build one score matrix per model for a given split (train / compare / blind)."""
    from evaluation.model_artifacts import load_model_artifacts

    matrices: List[np.ndarray] = []
    for name, is_score in zip(names, is_score_model):
        arts = load_model_artifacts(
            model_cfgs[name],
            pids,
            predictions_split=split,
            load_scores=False,
        )
        thr = load_thresholds_for_model(model_cfgs[name], all_labels) if is_score else None
        matrices.append(build_score_matrix(arts, pids, all_labels, thr))
    return matrices


# ---------------------------------------------------------------------------
# Result: (learner, threshold_mode, micro_f1, best_t, stacker, proba,
#          per_label_thresholds_or_None, val_pick_score)
# ---------------------------------------------------------------------------
_Result = Tuple[
    str, str, float, float,
    object,          # stacker (PerLabelStackingEnsemble or PyTorchMLPStacker)
    np.ndarray,
    Optional[np.ndarray],
    float,           # val_pick_score (equals micro_f1 when --val-pick micro)
]


def _parse_mlp_hidden(s: str) -> Tuple[int, ...]:
    parts = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not parts:
        raise ValueError("--mlp-hidden must list at least one positive int, e.g. 64,32")
    for p in parts:
        if p < 1:
            raise ValueError("--mlp-hidden values must be positive")
    return tuple(parts)


def _run_learner(
    learner_name: str,
    threshold_mode: str,
    train_matrices: List[np.ndarray],
    train_gt: Dict,
    train_pids: List[int],
    val_matrices: List[np.ndarray],
    val_gt: Dict,
    val_pids: List[int],
    all_labels: List[str],
    seed: int,
    n_threshold_steps: int,
    device: str,
    stacker_train_kw: Dict[str, Any],
    val_pick: str,
) -> List[_Result]:
    """Train one stacker and sweep thresholds; return one result per threshold mode."""
    print(f"\n--- Stacking: {learner_name} ---")
    stacker = make_stacker(learner_name, seed=seed, device=device, **stacker_train_kw)
    print(f"  Fitting meta-learner on train ({len(train_pids)} docs)…")
    if isinstance(stacker, PyTorchMLPStacker):
        stacker.fit(
            train_matrices,
            train_gt,
            train_pids,
            all_labels,
            val_matrices=val_matrices,
            val_gt=val_gt,
            val_pids=val_pids,
        )
    else:
        stacker.fit(train_matrices, train_gt, train_pids, all_labels)

    print(f"  Predicting stacking probabilities on val ({len(val_pids)} docs)…")
    proba = stacker.predict_proba(val_matrices, val_pids, all_labels)

    results: List[_Result] = []

    if threshold_mode in ("global", "both"):
        print(f"  Sweeping global threshold ({n_threshold_steps} steps)…")
        best_t, best_f1, _ = sweep_global_threshold(
            proba, val_gt, val_pids, all_labels, n_steps=n_threshold_steps,
        )
        print(f"  Global  threshold={best_t:.4f}  micro-F1={best_f1:.4f}")
        pick_g = val_pick_score(
            val_pick, proba, val_gt, val_pids, all_labels,
            "global", best_t, None, val_matrices, stacker,
        )
        results.append((learner_name, "global", best_f1, best_t, stacker, proba, None, pick_g))

    if threshold_mode in ("per_label", "both"):
        print(f"  Sweeping per-label thresholds ({n_threshold_steps} steps each)…")
        pl_thresholds, pl_f1, _ = sweep_per_label_thresholds(
            proba, val_gt, val_pids, all_labels, n_steps=n_threshold_steps,
        )
        print(f"  Per-label thresholds  micro-F1={pl_f1:.4f}")
        pick_pl = val_pick_score(
            val_pick, proba, val_gt, val_pids, all_labels,
            "per_label", 0.0, pl_thresholds, val_matrices, stacker,
        )
        results.append(
            (learner_name, "per_label", pl_f1, 0.0, stacker, proba, pl_thresholds, pick_pl),
        )

    return results


def _apply_threshold(
    proba: np.ndarray,
    pids: List[int],
    all_labels: List[str],
    threshold_mode: str,
    best_t: float,
    pl_thresholds: Optional[np.ndarray],
) -> Dict[int, List[str]]:
    if threshold_mode == "global":
        return proba_to_preds(proba, pids, all_labels, best_t)
    assert pl_thresholds is not None
    return proba_to_preds_per_label(proba, pids, all_labels, pl_thresholds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stacking ensemble: meta-learner on base model scores. "
            "Default: rank candidates by val micro-F1; use --val-pick for other val objectives."
        ),
    )
    parser.add_argument(
        "--config",
        default=EXPERIMENT_CFG,
        help="Path to evaluation/config.yaml.",
    )
    parser.add_argument(
        "--meta-learner",
        choices=list(LEARNER_NAMES) + ["all"],
        default="all",
        help=(
            "Meta-learner type. "
            "'all' sweeps all four options and picks the best by val micro-F1."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "PyTorch device for pytorch_mlp (ignored for sklearn learners). "
            "'auto' picks CUDA → MPS → CPU. Examples: cuda, cuda:0, mps, cpu."
        ),
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("global", "per_label", "both"),
        default="both",
        help="Threshold strategy applied to stacking probabilities.",
    )
    parser.add_argument(
        "--n-threshold-steps",
        type=int,
        default=100,
        help="Number of threshold candidates in the sweep (global and per-label).",
    )
    parser.add_argument(
        "--meta-features",
        choices=("default", "rich", "full", "unified"),
        default="default",
        help=(
            "Meta-inputs: default (K + max/mean/vote); rich (+ std/min/range + pairwise products); "
            "full (+ doc mean activation); unified (= full + vote entropy, frac≥1, frac≥0.5, top2 margin)."
        ),
    )
    parser.add_argument(
        "--stacking-unified",
        action="store_true",
        help=(
            "Shortcut: set --meta-features unified and, if patient-clusters-k<2, default it to 8. "
            "One model uses cluster one-hot + rich per-label features + voting block."
        ),
    )
    parser.add_argument(
        "--val-pick",
        choices=("micro", "macro_present", "cluster_min", "composite"),
        default="micro",
        help=(
            "Which validation scalar ranks (learner, threshold) for export. "
            "micro=group micro-F1; macro_present=mean per-code F1 (labels with support); "
            "cluster_min=min cluster micro-F1 (needs train-fitted patient clusters); "
            "composite=average of micro, macro_present, cluster_min."
        ),
    )
    parser.add_argument(
        "--mlp-label-emb",
        type=int,
        default=0,
        metavar="D",
        help="pytorch_mlp only: if D>0, concatenate a learned Embedding(n_labels, D) per row.",
    )
    parser.add_argument(
        "--logreg-c",
        type=float,
        default=1.0,
        help="Inverse L2 strength for sklearn logistic_regression meta-learner (default: 1.0).",
    )
    parser.add_argument(
        "--patient-clusters-k",
        type=int,
        default=0,
        metavar="K",
        help=(
            "If K≥2, fit KMeans on train patients (concatenated base score rows) and append "
            "a K-dimensional cluster one-hot to every meta-feature row. 0 disables."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--export-dir",
        default="outputs/predictions/ensemble_stacking",
        help="Root directory for exported JSONL prediction files.",
    )
    parser.add_argument(
        "--no-export-predictions",
        action="store_true",
        help="Skip writing JSONL outputs after the run.",
    )
    train = parser.add_argument_group(
        "Meta-learner training budget (defaults match previous hard-coded values).",
    )
    train.add_argument(
        "--mlp-epochs",
        type=int,
        default=50,
        help="PyTorch MLP: optimization epochs (default: 50).",
    )
    train.add_argument(
        "--mlp-lr",
        type=float,
        default=1e-3,
        help="PyTorch MLP: Adam learning rate (default: 1e-3).",
    )
    train.add_argument(
        "--mlp-batch-size",
        type=int,
        default=512,
        help="PyTorch MLP: DataLoader batch size (default: 512).",
    )
    train.add_argument(
        "--mlp-hidden",
        default="64,32",
        metavar="H1,H2,...",
        help="PyTorch MLP: comma-separated hidden layer widths (default: 64,32).",
    )
    train.add_argument(
        "--logreg-max-iter",
        type=int,
        default=1000,
        help="LogisticRegression max_iter per label (default: 1000).",
    )
    train.add_argument(
        "--rf-n-estimators",
        type=int,
        default=100,
        help="RandomForest n_estimators per label (default: 100).",
    )
    train.add_argument(
        "--rf-max-depth",
        type=int,
        default=6,
        help="RandomForest max_depth per label (default: 6).",
    )
    train.add_argument(
        "--hgb-max-iter",
        type=int,
        default=100,
        help="HistGradientBoosting max_iter per label (default: 100).",
    )
    train.add_argument(
        "--hgb-max-depth",
        type=int,
        default=4,
        help="HistGradientBoosting max_depth per label (default: 4).",
    )
    train.add_argument(
        "--mlp-early-stop-patience",
        type=int,
        default=0,
        help=(
            "PyTorch MLP only: if >0, each epoch evaluate val BCE (same loss as train) and stop "
            "when it fails to improve for this many epochs; restore best val weights. "
            "Typical: 10–20 with --mlp-epochs 150."
        ),
    )
    train.add_argument(
        "--mlp-refit-trainval-epochs",
        type=int,
        default=0,
        help=(
            "PyTorch MLP only: after phase-1 (and optional val early-stop), run this many extra "
            "epochs on **train+val** patients (BCE with val labels). 0 disables. Use with GPU for "
            "a longer, stronger fit."
        ),
    )
    train.add_argument(
        "--mlp-refit-lr",
        type=float,
        default=None,
        help="PyTorch MLP phase-2 Adam LR; default 0.25× phase-1 --mlp-lr when omitted.",
    )
    train.add_argument(
        "--mlp-long-run",
        action="store_true",
        help=(
            "Preset for pytorch_mlp: ~240 phase-1 epochs, patience 30, hidden 128,64,32, batch 256, "
            "label_emb 32, 100 phase-2 train+val refit epochs, refit LR 2e-4. "
            "If --device auto and CUDA exists, selects cuda. Expect several minutes on GPU."
        ),
    )
    args = parser.parse_args()
    if getattr(args, "stacking_unified", False):
        args.meta_features = "unified"
        if int(args.patient_clusters_k) < 2:
            args.patient_clusters_k = 8
    if getattr(args, "mlp_long_run", False):
        try:
            import torch

            if str(args.device) == "auto" and torch.cuda.is_available():
                args.device = "cuda"
                print(
                    "[ensemble_stacking] --mlp-long-run: using CUDA (--device was auto)",
                    flush=True,
                )
        except ImportError:
            pass
        args.mlp_epochs = max(int(args.mlp_epochs), 240)
        args.mlp_early_stop_patience = max(int(args.mlp_early_stop_patience), 30)
        args.mlp_hidden = "128,64,32"
        args.mlp_batch_size = min(int(args.mlp_batch_size), 256)
        args.mlp_label_emb = max(int(args.mlp_label_emb), 32)
        args.mlp_refit_trainval_epochs = max(int(args.mlp_refit_trainval_epochs), 100)
        if args.mlp_refit_lr is None:
            args.mlp_refit_lr = 2e-4
        print(
            "[ensemble_stacking] --mlp-long-run: "
            f"phase1 ≤{args.mlp_epochs} epochs (val early-stop patience={args.mlp_early_stop_patience}), "
            f"phase2 train+val {args.mlp_refit_trainval_epochs} epochs, hidden=128,64,32",
            flush=True,
        )
    try:
        mlp_hidden_dims = _parse_mlp_hidden(args.mlp_hidden)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    stacker_train_kw: Dict[str, Any] = {
        "mlp_epochs": args.mlp_epochs,
        "mlp_lr": args.mlp_lr,
        "mlp_batch_size": args.mlp_batch_size,
        "mlp_hidden_dims": mlp_hidden_dims,
        "meta_features": args.meta_features,
        "mlp_label_emb_dim": int(args.mlp_label_emb),
        "logreg_c": float(args.logreg_c),
        "logreg_max_iter": args.logreg_max_iter,
        "rf_n_estimators": args.rf_n_estimators,
        "rf_max_depth": args.rf_max_depth,
        "hgb_max_iter": args.hgb_max_iter,
        "hgb_max_depth": args.hgb_max_depth,
        "mlp_early_stop_patience": args.mlp_early_stop_patience,
        "patient_cluster_k": int(args.patient_clusters_k),
        "mlp_refit_trainval_epochs": int(args.mlp_refit_trainval_epochs),
        "mlp_refit_lr": args.mlp_refit_lr,
    }

    # -----------------------------------------------------------------------
    # Load ground truth and model configs
    # -----------------------------------------------------------------------
    cfg = load_config(args.config)
    val_path = str(get_cfg(cfg, "data.val_path"))
    train_path = str(get_cfg(cfg, "data.train_path", "data/processed/train.jsonl"))

    val_gt = load_ground_truth(val_path)
    train_gt = load_ground_truth(train_path)
    val_pids = list(val_gt.keys())
    train_pids = list(train_gt.keys())
    model_cfgs = {m["name"]: m for m in get_cfg(cfg, "models", [])}

    # -----------------------------------------------------------------------
    # Load val model artifacts (continuous scores where available)
    # -----------------------------------------------------------------------
    print("Loading val model artifacts…")
    artifacts_list = gather_ensemble_artifacts(model_cfgs, val_pids, "val")
    if not artifacts_list:
        print(
            "ERROR: no ensemble models had validation predictions on disk.\n"
            "Generate them with: PYTHONPATH=src python -m evaluation.run_predictions",
            file=sys.stderr,
        )
        sys.exit(1)

    all_labels = canonical_ensemble_label_arts(artifacts_list).label_names
    names = [n for n, _ in artifacts_list]
    is_score_model = [arts.scores is not None for _, arts in artifacts_list]

    print(
        f"  models={names}\n"
        f"  labels={len(all_labels)}  val_docs={len(val_pids)}  train_docs={len(train_pids)}\n"
        f"  meta_features={args.meta_features}  logreg_c={args.logreg_c}  "
        f"mlp_label_emb={args.mlp_label_emb}  patient_clusters_k={args.patient_clusters_k}",
    )

    # -----------------------------------------------------------------------
    # Build val score matrices
    # -----------------------------------------------------------------------
    print("\nBuilding val score matrices…")
    val_matrices: List[np.ndarray] = []
    for name, arts in artifacts_list:
        thr = load_thresholds_for_model(model_cfgs[name], all_labels) if arts.scores is not None else None
        val_matrices.append(build_score_matrix(arts, val_pids, all_labels, thr))

    # -----------------------------------------------------------------------
    # Build train score matrices (required for fitting meta-learners)
    # -----------------------------------------------------------------------
    print("Building train score matrices…")
    try:
        train_matrices = _load_matrices_for_split(
            model_cfgs, names, is_score_model, train_pids, all_labels, "train",
        )
    except FileNotFoundError as exc:
        print(
            f"ERROR: {exc}\n"
            "Stacking fits the meta-learner on the **train** split; each base model needs "
            "``outputs/predictions/<name>/train_predictions.jsonl`` (same patient_ids as ``data.train_path``). "
            "Val/test JSONL alone are not enough.\n"
            "Generate train (and refresh val/test) with:\n"
            "  PYTHONPATH=src python -m evaluation.run_predictions",
            file=sys.stderr,
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Individual model baselines
    # -----------------------------------------------------------------------
    print("\nIndividual model micro-F1 (val):")
    for name, arts in artifacts_list:
        preds = {pid: list(arts.pred_data.get(pid, [])) for pid in val_pids}
        f1 = evaluate_data(val_gt, preds, label_space=arts.label_names)["micro_f1"]
        print(f"  {name:<25}: {f1:.4f}")

    # -----------------------------------------------------------------------
    # Meta-learner sweep
    # -----------------------------------------------------------------------
    learners_to_try = list(LEARNER_NAMES) if args.meta_learner == "all" else [args.meta_learner]
    all_results: List[_Result] = []

    for learner_name in learners_to_try:
        results = _run_learner(
            learner_name=learner_name,
            threshold_mode=args.threshold_mode,
            train_matrices=train_matrices,
            train_gt=train_gt,
            train_pids=train_pids,
            val_matrices=val_matrices,
            val_gt=val_gt,
            val_pids=val_pids,
            all_labels=all_labels,
            seed=args.seed,
            n_threshold_steps=args.n_threshold_steps,
            device=args.device,
            stacker_train_kw=stacker_train_kw,
            val_pick=str(args.val_pick),
        )
        all_results.extend(results)

    if not all_results:
        print("No results produced.", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    pick_key = lambda r: -r[7]
    title = "validation micro-F1"
    if args.val_pick != "micro":
        title = f"validation (--val-pick={args.val_pick}; micro-F1 shown second)"
    print(f"\n=== Results ({title}) ===")
    for row in sorted(all_results, key=pick_key):
        learner, thr_mode, f1, best_t, _, _, _, pscore = row
        thr_str = f"  threshold={best_t:.4f}" if thr_mode == "global" else "  (per-label)"
        extra = f"  pick={pscore:.4f}" if args.val_pick != "micro" else ""
        print(f"  {learner:<25}  {thr_mode:<12}  micro={f1:.4f}{extra}{thr_str}")

    best: _Result = max(all_results, key=lambda r: r[7])
    (
        best_learner,
        best_thr_mode,
        best_f1,
        best_t,
        best_stacker,
        best_proba,
        best_pl_thrs,
        best_pick,
    ) = best
    print(
        f"\nBest (by --val-pick={args.val_pick}): {best_learner} / {best_thr_mode}  "
        f"pick={best_pick:.4f}  micro-F1={best_f1:.4f}",
    )

    if args.no_export_predictions:
        return

    # -----------------------------------------------------------------------
    # Export predictions
    # -----------------------------------------------------------------------
    export_root = Path(args.export_dir)
    export_root.mkdir(parents=True, exist_ok=True)
    slug = f"{best_learner}_{best_thr_mode}"
    if args.meta_features != "default":
        slug = f"{slug}__mf_{args.meta_features}"
    if int(args.mlp_label_emb) > 0:
        slug = f"{slug}_leb{int(args.mlp_label_emb)}"
    if float(args.logreg_c) != 1.0 and best_learner == "logistic_regression":
        slug = f"{slug}_c{float(args.logreg_c):g}".replace(".", "p")
    if int(args.patient_clusters_k) >= 2:
        slug = f"{slug}_pk{int(args.patient_clusters_k)}"
    if args.val_pick != "micro":
        slug = f"{slug}_vp_{args.val_pick}"
    out_dir = export_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # Val predictions
    val_preds = _apply_threshold(best_proba, val_pids, all_labels, best_thr_mode, best_t, best_pl_thrs)
    save_predictions_jsonl(val_preds, out_dir / "val_predictions.jsonl")
    print(f"\n--- Stacking JSONL export (slug={slug}) ---")
    print(f"  val   → {out_dir / 'val_predictions.jsonl'}")

    def _export_split(split_name: str, cfg_key: str, pred_split: str) -> None:
        split_path = str(get_cfg(cfg, cfg_key, ""))
        if not split_path or not Path(split_path).is_file():
            print(f"  {split_name:<6} → skipped ({cfg_key} not found in config or file missing)")
            return
        split_gt = load_ground_truth(split_path)
        split_pids = list(split_gt.keys())
        try:
            split_mats = _load_matrices_for_split(
                model_cfgs, names, is_score_model, split_pids, all_labels, pred_split,
            )
        except FileNotFoundError as exc:
            print(f"  {split_name:<6} → skipped ({exc})")
            return
        split_proba = best_stacker.predict_proba(split_mats, split_pids, all_labels)
        split_preds = _apply_threshold(
            split_proba, split_pids, all_labels, best_thr_mode, best_t, best_pl_thrs,
        )
        out_path = out_dir / f"{split_name}_predictions.jsonl"
        save_predictions_jsonl(split_preds, out_path)
        print(f"  {split_name:<6} → {out_path}")

    _export_split("test", "data.test_path", "compare")
    _export_split("blind", "data.blind_path", "blind")

    # Write manifest
    manifest = {
        "export_root": str(export_root.resolve()),
        "best_learner": best_learner,
        "best_threshold_mode": best_thr_mode,
        "best_threshold": best_t if best_thr_mode == "global" else None,
        "val_micro_f1": best_f1,
        "val_pick": args.val_pick,
        "val_pick_score": best_pick,
        "meta_features": args.meta_features,
        "logreg_c": float(args.logreg_c),
        "mlp_label_emb": int(args.mlp_label_emb),
        "patient_clusters_k": int(args.patient_clusters_k),
        "mlp_refit_trainval_epochs": int(args.mlp_refit_trainval_epochs),
        "mlp_refit_lr": args.mlp_refit_lr,
        "mlp_long_run": bool(getattr(args, "mlp_long_run", False)),
        "models": names,
        "slug": slug,
        "strategies": [slug],
        "all_results": [
            {
                "learner": l,
                "threshold_mode": tm,
                "micro_f1": round(f1, 6),
                "val_pick_score": round(ps, 6),
                "threshold": round(t, 6) if tm == "global" else None,
            }
            for l, tm, f1, t, _, _, _, ps in sorted(all_results, key=pick_key)
        ],
    }
    manifest_path = export_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"\n  manifest → {manifest_path}")


if __name__ == "__main__":
    main()
