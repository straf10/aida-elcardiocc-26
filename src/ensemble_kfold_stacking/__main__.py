"""K-fold stacking: meta-learner fit on train with **out-of-fold** predictions (no leakage).

For each fold, the stacker is trained only on ``K-1`` train folds and predicts probabilities
on the held-out fold.  Those rows fill an **OOF** matrix on all train patients.  We report
micro-F1 on train gold after a threshold sweep on OOF (honest train-side check).

The **exported** model is a **fresh** stacker refit on **all** train patients (same API as
``ensemble_stacking``), with thresholds chosen on the real **validation** split — so test /
blind exports match production stacking behaviour, with an extra OOF diagnostic first.

Writes under ``--export-dir`` (default ``outputs/predictions/ensemble_kfold_stacking``), not
``ensemble_stacking/``.

Usage
-----
PYTHONPATH=src python -m ensemble_kfold_stacking \\
    --k 5 \\
    [--meta-learner logistic_regression|...|all] \\
    [--export-dir outputs/predictions/ensemble_kfold_stacking]

Most ``--meta-features``, ``--patient-clusters-k``, ``--logreg-c``, ``--mlp-*`` flags match
``ensemble_stacking`` (see that module's ``--help``).
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
from sklearn.model_selection import KFold

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

EXPERIMENT_CFG = "src/evaluation/config.yaml"


def _parse_mlp_hidden(s: str) -> Tuple[int, ...]:
    parts = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not parts:
        raise ValueError("--mlp-hidden must list positive ints")
    for p in parts:
        if p < 1:
            raise ValueError("--mlp-hidden values must be positive")
    return tuple(parts)


def _load_matrices_for_split(
    model_cfgs: Dict,
    names: List[str],
    is_score_model: List[bool],
    pids: List[int],
    all_labels: List[str],
    split: str,
) -> List[np.ndarray]:
    from evaluation.model_artifacts import load_model_artifacts

    out: List[np.ndarray] = []
    for name, is_score in zip(names, is_score_model):
        arts = load_model_artifacts(
            model_cfgs[name],
            pids,
            predictions_split=split,
            load_scores=False,
        )
        thr = load_thresholds_for_model(model_cfgs[name], all_labels) if is_score else None
        out.append(build_score_matrix(arts, pids, all_labels, thr))
    return out


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


def _run_kfold_oof(
    learner_name: str,
    train_matrices: List[np.ndarray],
    train_gt: Dict,
    train_pids: List[int],
    all_labels: List[str],
    k: int,
    seed: int,
    device: str,
    stacker_kw: Dict[str, Any],
) -> Tuple[np.ndarray, float]:
    """Return (oof_proba shape (n_train, n_labels), oof_micro_f1 after per_label sweep on train)."""
    n_train = len(train_pids)
    n_labels = len(all_labels)
    oof = np.zeros((n_train, n_labels), dtype=np.float32)
    idx = np.arange(n_train)
    kf = KFold(n_splits=int(k), shuffle=True, random_state=int(seed))

    for fold, (tr_i, ho_i) in enumerate(kf.split(idx)):
        tr_pids = [train_pids[i] for i in tr_i]
        ho_pids = [train_pids[i] for i in ho_i]
        tr_gt = {p: train_gt[p] for p in tr_pids}
        tr_mats = [m[tr_i].copy() for m in train_matrices]
        ho_mats = [m[ho_i].copy() for m in train_matrices]

        stacker = make_stacker(
            learner_name,
            seed=seed + 17 * fold,
            device=device,
            **stacker_kw,
        )
        print(f"  [kfold] fold {fold + 1}/{k}  train={len(tr_pids)}  holdout={len(ho_pids)}", flush=True)
        if isinstance(stacker, PyTorchMLPStacker):
            stacker.fit(
                tr_mats,
                tr_gt,
                tr_pids,
                all_labels,
                val_matrices=None,
                val_gt=None,
                val_pids=None,
            )
        else:
            stacker.fit(tr_mats, tr_gt, tr_pids, all_labels)

        proba = stacker.predict_proba(ho_mats, ho_pids, all_labels)
        oof[ho_i] = proba.astype(np.float32, copy=False)

    pl_th, oof_f1, _ = sweep_per_label_thresholds(
        oof, train_gt, train_pids, all_labels, n_steps=100,
    )
    print(f"  [kfold] OOF train micro-F1 (per_label sweep)={oof_f1:.4f}", flush=True)
    return oof, float(oof_f1)


def main() -> None:
    parser = argparse.ArgumentParser(description="K-fold OOF stacking diagnostics + refit export.")
    parser.add_argument("--config", default=EXPERIMENT_CFG)
    parser.add_argument("--k", type=int, default=5, help="KFold splits on train patients (>=2).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--meta-learner",
        choices=list(LEARNER_NAMES) + ["all"],
        default="logistic_regression",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--export-dir",
        default="outputs/predictions/ensemble_kfold_stacking",
        help="Separate from ensemble_stacking/ to avoid manifest races.",
    )
    parser.add_argument("--threshold-mode", choices=("global", "per_label", "both"), default="both")
    parser.add_argument("--n-threshold-steps", type=int, default=100)
    parser.add_argument("--no-export-predictions", action="store_true")
    parser.add_argument("--meta-features", choices=("default", "rich", "full", "unified"), default="default")
    parser.add_argument("--stacking-unified", action="store_true")
    parser.add_argument("--patient-clusters-k", type=int, default=0)
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--logreg-max-iter", type=int, default=1000)
    parser.add_argument("--rf-n-estimators", type=int, default=100)
    parser.add_argument("--rf-max-depth", type=int, default=6)
    parser.add_argument("--hgb-max-iter", type=int, default=100)
    parser.add_argument("--hgb-max-depth", type=int, default=4)
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-batch-size", type=int, default=512)
    parser.add_argument("--mlp-hidden", default="64,32")
    parser.add_argument("--mlp-label-emb", type=int, default=0)
    parser.add_argument("--mlp-early-stop-patience", type=int, default=0)
    parser.add_argument("--mlp-refit-trainval-epochs", type=int, default=0)
    parser.add_argument("--mlp-refit-lr", type=float, default=None)
    args = parser.parse_args()

    if int(args.k) < 2:
        raise SystemExit("--k must be >= 2")

    if getattr(args, "stacking_unified", False):
        args.meta_features = "unified"
        if int(args.patient_clusters_k) < 2:
            args.patient_clusters_k = 8

    try:
        mlp_hidden_dims = _parse_mlp_hidden(args.mlp_hidden)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    stacker_kw: Dict[str, Any] = {
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

    cfg = load_config(args.config)
    val_path = str(get_cfg(cfg, "data.val_path"))
    train_path = str(get_cfg(cfg, "data.train_path", "data/processed/train.jsonl"))
    val_gt = load_ground_truth(val_path)
    train_gt = load_ground_truth(train_path)
    val_pids = list(val_gt.keys())
    train_pids = list(train_gt.keys())
    model_cfgs = {m["name"]: m for m in get_cfg(cfg, "models", [])}

    print("Loading val artifacts…")
    artifacts_list = gather_ensemble_artifacts(model_cfgs, val_pids, "val")
    if not artifacts_list:
        print("ERROR: no val predictions for ensemble.", file=sys.stderr)
        sys.exit(1)
    all_labels = canonical_ensemble_label_arts(artifacts_list).label_names
    names = [n for n, _ in artifacts_list]
    is_score_model = [arts.scores is not None for _, arts in artifacts_list]

    val_matrices: List[np.ndarray] = []
    for name, arts in artifacts_list:
        thr = load_thresholds_for_model(model_cfgs[name], all_labels) if arts.scores is not None else None
        val_matrices.append(build_score_matrix(arts, val_pids, all_labels, thr))

    try:
        train_matrices = _load_matrices_for_split(
            model_cfgs, names, is_score_model, train_pids, all_labels, "train",
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    learners = list(LEARNER_NAMES) if args.meta_learner == "all" else [args.meta_learner]
    best_learner: Optional[str] = None
    best_oof = -1.0

    print(f"\n=== K-fold OOF on train (K={args.k}) ===")
    for ln in learners:
        print(f"\n--- OOF: {ln} ---", flush=True)
        _, oof_f1 = _run_kfold_oof(
            ln,
            train_matrices,
            train_gt,
            train_pids,
            all_labels,
            int(args.k),
            int(args.seed),
            str(args.device),
            stacker_kw,
        )
        if oof_f1 > best_oof:
            best_oof, best_learner = oof_f1, ln

    assert best_learner is not None
    print(f"\nBest meta-learner by OOF train micro-F1: {best_learner}  ({best_oof:.4f})")

    print(f"\n=== Refit {best_learner} on full train → val thresholds ===", flush=True)
    final = make_stacker(
        best_learner,
        seed=int(args.seed),
        device=str(args.device),
        **stacker_kw,
    )
    if isinstance(final, PyTorchMLPStacker):
        final.fit(
            train_matrices,
            train_gt,
            train_pids,
            all_labels,
            val_matrices=val_matrices,
            val_gt=val_gt,
            val_pids=val_pids,
        )
    else:
        final.fit(train_matrices, train_gt, train_pids, all_labels)

    val_proba = final.predict_proba(val_matrices, val_pids, all_labels)
    results: List[Tuple[str, float, float, Optional[np.ndarray]]] = []
    if args.threshold_mode in ("global", "both"):
        bt, bf, _ = sweep_global_threshold(
            val_proba, val_gt, val_pids, all_labels, n_steps=args.n_threshold_steps,
        )
        print(f"  val global  micro-F1={bf:.4f}  t={bt:.4f}")
        results.append(("global", bf, bt, None))
    if args.threshold_mode in ("per_label", "both"):
        pl_th, pf, _ = sweep_per_label_thresholds(
            val_proba, val_gt, val_pids, all_labels, n_steps=args.n_threshold_steps,
        )
        print(f"  val per_label  micro-F1={pf:.4f}")
        results.append(("per_label", pf, 0.0, pl_th))

    prefer_pl = 0.02
    if args.threshold_mode == "both" and len(results) == 2:
        g = next(r for r in results if r[0] == "global")
        pl = next(r for r in results if r[0] == "per_label")
        if g[1] - pl[1] <= prefer_pl:
            best_mode, best_f1, best_t, best_pl = pl[0], pl[1], pl[2], pl[3]
            print(f"\nExport: {best_mode} (tie-break vs global)  val micro-F1={best_f1:.4f}")
        else:
            best_mode, best_f1, best_t, best_pl = max(results, key=lambda r: r[1])
            print(f"\nExport: {best_mode}  val micro-F1={best_f1:.4f}")
    else:
        best_mode, best_f1, best_t, best_pl = max(results, key=lambda r: r[1])
        print(f"\nExport: {best_mode}  val micro-F1={best_f1:.4f}")

    if args.no_export_predictions:
        return

    root = Path(args.export_dir)
    root.mkdir(parents=True, exist_ok=True)
    slug = f"kfold{args.k}_{best_learner}_{best_mode}__mf_{args.meta_features}"
    if int(args.patient_clusters_k) >= 2:
        slug = f"{slug}_pk{int(args.patient_clusters_k)}"
    out_dir = root / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    val_preds = _apply_threshold(val_proba, val_pids, all_labels, best_mode, best_t, best_pl)
    save_predictions_jsonl(val_preds, out_dir / "val_predictions.jsonl")
    print(f"\n--- K-fold stacking export (slug={slug}) ---\n  val → {out_dir / 'val_predictions.jsonl'}")

    def _export(name: str, cfg_key: str, pred_split: str) -> None:
        pth = str(get_cfg(cfg, cfg_key, ""))
        if not pth or not Path(pth).is_file():
            print(f"  {name:<6} skipped")
            return
        gt = load_ground_truth(pth)
        pids = list(gt.keys())
        try:
            mats = _load_matrices_for_split(model_cfgs, names, is_score_model, pids, all_labels, pred_split)
        except FileNotFoundError as exc:
            print(f"  {name:<6} skipped ({exc})")
            return
        proba = final.predict_proba(mats, pids, all_labels)
        preds = _apply_threshold(proba, pids, all_labels, best_mode, best_t, best_pl)
        outp = out_dir / f"{name}_predictions.jsonl"
        save_predictions_jsonl(preds, outp)
        print(f"  {name:<6} → {outp}")

    _export("test", "data.test_path", "compare")
    _export("blind", "data.blind_path", "blind")

    manifest = {
        "export_root": str(root.resolve()),
        "method": "ensemble_kfold_stacking",
        "k": int(args.k),
        "best_learner": best_learner,
        "oof_train_micro_f1": best_oof,
        "best_threshold_mode": best_mode,
        "val_micro_f1": best_f1,
        "meta_features": args.meta_features,
        "patient_clusters_k": int(args.patient_clusters_k),
        "models": names,
        "slug": slug,
    }
    with open(root / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"  manifest → {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
