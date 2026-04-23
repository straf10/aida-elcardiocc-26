"""Committee grid MLP — multi-label fusion **without** stacking.

Each training example is one **document** (any patient id, including unseen blind ids):
input tensor ``(n_models, n_labels)`` = stacked, threshold-normalised scores from the
committee.  A single NN maps that grid to ``n_labels`` logits so **labels interact**
inside the net.  No patient-index embedding, no KNN over train patients, no
per-patient parameter table — only functions of scores for the current document.

Contrast with ``ensemble_stacking`` (per-label meta-features / (doc,label) MLP rows).

Usage
-----
PYTHONPATH=src python -m ensemble_committee_mlp \\
    [--config src/evaluation/config.yaml] \\
    [--arch flatten|conv] \\
    [--hidden 512,256] \\
    [--epochs 200] [--device cuda] [--batch-size 64] \\
    [--early-stop-patience 20] \\
    [--threshold-mode global|per_label|both] (default: per_label; global is often poor on test) \\
    [--export-dir outputs/predictions/ensemble_committee_mlp]

Add a ``models[]`` row pointing at ``…/test_predictions.jsonl`` under ``--export-dir``
for ``compare_methods``.
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.config_utils import get_cfg, load_config
from evaluation.io_utils import load_ground_truth, save_predictions_jsonl
from evaluation.scoring import evaluate_data
from ensemble_committee_mlp.grid_net import (
    CommitteeConvGridMLP,
    CommitteeFlattenMLP,
    apply_zscore_grid,
    resolve_torch_device,
    stack_model_grid,
    train_zscore_grid,
)
from ensemble_metaheuristic.matrices import build_score_matrix, load_thresholds_for_model
from ensemble_metaheuristic.strategy_loaders import (
    canonical_ensemble_label_arts,
    gather_ensemble_artifacts,
)
from ensemble_stacking.meta_features import build_target_matrix
from ensemble_stacking.threshold_opt import (
    proba_to_preds,
    proba_to_preds_per_label,
    sweep_global_threshold,
    sweep_per_label_thresholds,
)

EXPERIMENT_CFG = "src/evaluation/config.yaml"


def _load_matrices_for_split(
    model_cfgs: Dict,
    names: List[str],
    is_score_model: List[bool],
    pids: List[int],
    all_labels: List[str],
    split: str,
) -> List[np.ndarray]:
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


def _parse_hidden(s: str) -> Tuple[int, ...]:
    parts = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not parts:
        raise ValueError("--hidden must list positive ints, e.g. 512,256")
    for p in parts:
        if p < 1:
            raise ValueError("--hidden values must be positive")
    return tuple(parts)


def _train_grid_mlp(
    model: nn.Module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    Y_val: Optional[np.ndarray],
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    early_stop_patience: int,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    model.to(device)
    n_pos = float(Y_train.sum())
    n_neg = float(Y_train.size - Y_train.sum())
    pos_w = n_neg / n_pos if n_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_w], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=1e-4)

    Xt = torch.from_numpy(X_train).to(device)
    Yt = torch.from_numpy(Y_train.astype(np.float32)).to(device)
    loader = DataLoader(
        TensorDataset(Xt, Yt),
        batch_size=int(batch_size),
        shuffle=True,
    )

    use_es = early_stop_patience > 0 and X_val is not None and Y_val is not None
    if use_es:
        print(
            f"    [committee_mlp] val early-stop patience={early_stop_patience}",
            flush=True,
        )
    best_v = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stall = 0
    log_every = max(1, epochs // 25)

    for epoch in range(int(epochs)):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tot += loss.item() * len(xb)
        if epoch == 0 or (epoch + 1) % log_every == 0 or epoch + 1 == epochs:
            print(
                f"    [committee_mlp] epoch {epoch + 1:4d}/{epochs}  train_bce={tot / len(Xt):.4f}",
                flush=True,
            )
        if use_es:
            model.eval()
            with torch.no_grad():
                xv = torch.from_numpy(X_val).to(device)
                yv = torch.from_numpy(Y_val.astype(np.float32)).to(device)
                vloss = float(criterion(model(xv), yv).item())
            model.train()
            if vloss < best_v - 1e-7:
                best_v = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stall = 0
            else:
                stall += 1
                if stall >= early_stop_patience:
                    print(
                        f"    [committee_mlp] early stop at {epoch + 1}  best_val_bce={best_v:.4f}",
                        flush=True,
                    )
                    break

    if use_es and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"    [committee_mlp] restored best val BCE weights ({best_v:.4f})", flush=True)


@torch.no_grad()
def _predict_proba_grid(
    model: nn.Module,
    matrices: List[np.ndarray],
    device: torch.device,
    mu: np.ndarray,
    sig: np.ndarray,
) -> np.ndarray:
    model.eval()
    X = apply_zscore_grid(stack_model_grid(matrices), mu, sig)
    xt = torch.from_numpy(X).to(device)
    logits = model(xt)
    return torch.sigmoid(logits).cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Committee grid MLP: one NN per document over (models × labels) scores.",
    )
    parser.add_argument("--config", default=EXPERIMENT_CFG)
    parser.add_argument(
        "--arch",
        choices=("flatten", "conv"),
        default="flatten",
        help="flatten: MLP on vec(models*labels); conv: Conv1d over labels then MLP head.",
    )
    parser.add_argument(
        "--hidden",
        default="512,256",
        metavar="H,...",
        help="flatten: comma MLP hidden widths; conv: ignored for conv body, last MLP uses --conv-mlp-hidden.",
    )
    parser.add_argument(
        "--conv-mlp-hidden",
        type=int,
        default=256,
        help="conv arch: hidden size before final linear.",
    )
    parser.add_argument("--conv-channels", type=int, default=64, help="conv arch: channel width.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument(
        "--threshold-mode",
        choices=("global", "per_label", "both"),
        default="per_label",
        help="Default per_label: one threshold per code. ``global`` is often much worse on test.",
    )
    parser.add_argument("--n-threshold-steps", type=int, default=100)
    parser.add_argument(
        "--export-dir",
        default="outputs/predictions/ensemble_committee_mlp",
    )
    parser.add_argument("--no-export-predictions", action="store_true")
    args = parser.parse_args()

    try:
        hidden_dims = _parse_hidden(args.hidden)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    cfg = load_config(args.config)
    val_path = str(get_cfg(cfg, "data.val_path"))
    train_path = str(get_cfg(cfg, "data.train_path", "data/processed/train.jsonl"))
    val_gt = load_ground_truth(val_path)
    train_gt = load_ground_truth(train_path)
    val_pids = list(val_gt.keys())
    train_pids = list(train_gt.keys())
    model_cfgs = {m["name"]: m for m in get_cfg(cfg, "models", [])}

    print("Loading val committee artifacts…")
    artifacts_list = gather_ensemble_artifacts(model_cfgs, val_pids, "val")
    if not artifacts_list:
        print("ERROR: no ensemble models with val predictions.", file=sys.stderr)
        sys.exit(1)
    all_labels = canonical_ensemble_label_arts(artifacts_list).label_names
    names = [n for n, _ in artifacts_list]
    is_score_model = [arts.scores is not None for _, arts in artifacts_list]
    n_models = len(names)
    n_labels = len(all_labels)
    print(f"  models={names}  n_labels={n_labels}  train={len(train_pids)}  val={len(val_pids)}")

    val_matrices: List[np.ndarray] = []
    for name, arts in artifacts_list:
        thr = load_thresholds_for_model(model_cfgs[name], all_labels) if arts.scores is not None else None
        val_matrices.append(build_score_matrix(arts, val_pids, all_labels, thr))

    print("\nIndividual val micro-F1:")
    for name, arts in artifacts_list:
        preds = {pid: list(arts.pred_data.get(pid, [])) for pid in val_pids}
        f1 = evaluate_data(val_gt, preds, label_space=arts.label_names)["micro_f1"]
        print(f"  {name:<25} {f1:.4f}")

    try:
        train_matrices = _load_matrices_for_split(
            model_cfgs, names, is_score_model, train_pids, all_labels, "train",
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    X_train = stack_model_grid(train_matrices)
    Y_train = build_target_matrix(train_gt, train_pids, all_labels)
    X_val = stack_model_grid(val_matrices)
    Y_val = build_target_matrix(val_gt, val_pids, all_labels)
    X_train, grid_mu, grid_sig = train_zscore_grid(X_train)
    X_val = apply_zscore_grid(X_val, grid_mu, grid_sig)
    print("  grid: train Z-score per (model,label) from train split only", flush=True)

    dev = resolve_torch_device(str(args.device))
    print(f"\n[committee_mlp] arch={args.arch}  device={dev}  grid_shape=({n_models}, {n_labels})")

    if args.arch == "flatten":
        model: nn.Module = CommitteeFlattenMLP(n_models, n_labels, hidden_dims)
    else:
        model = CommitteeConvGridMLP(
            n_models,
            n_labels,
            conv_channels=args.conv_channels,
            mlp_hidden=args.conv_mlp_hidden,
        )

    _train_grid_mlp(
        model,
        X_train,
        Y_train,
        X_val,
        Y_val,
        device=dev,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        early_stop_patience=int(args.early_stop_patience),
        seed=int(args.seed),
    )

    val_proba = _predict_proba_grid(model, val_matrices, dev, grid_mu, grid_sig)
    results: List[Tuple[str, float, float, Optional[np.ndarray]]] = []

    if args.threshold_mode in ("global", "both"):
        best_t, best_f1, _ = sweep_global_threshold(
            val_proba, val_gt, val_pids, all_labels, n_steps=args.n_threshold_steps,
        )
        print(f"\n  global  threshold={best_t:.4f}  micro-F1={best_f1:.4f}")
        results.append(("global", best_f1, best_t, None))

    if args.threshold_mode in ("per_label", "both"):
        pl_th, pl_f1, _ = sweep_per_label_thresholds(
            val_proba, val_gt, val_pids, all_labels, n_steps=args.n_threshold_steps,
        )
        print(f"  per_label  micro-F1={pl_f1:.4f}")
        results.append(("per_label", pl_f1, 0.0, pl_th))

    # If val micro is almost tied, prefer per_label (global is unstable on rare labels / test).
    prefer_pl_margin = 0.02
    if args.threshold_mode == "both" and len(results) == 2:
        g = next(r for r in results if r[0] == "global")
        pl = next(r for r in results if r[0] == "per_label")
        if g[1] - pl[1] <= prefer_pl_margin:
            best_mode, best_f1, best_t, best_pl = pl[0], pl[1], pl[2], pl[3]
            print(
                f"\nBest threshold mode: {best_mode}  val micro-F1={best_f1:.4f}  "
                f"(tie-break: global only +{g[1] - pl[1]:.4f} on val → per_label)",
                flush=True,
            )
        else:
            best_mode, best_f1, best_t, best_pl = max(results, key=lambda r: r[1])
            print(f"\nBest threshold mode: {best_mode}  val micro-F1={best_f1:.4f}")
    else:
        best_mode, best_f1, best_t, best_pl = max(results, key=lambda r: r[1])
        print(f"\nBest threshold mode: {best_mode}  val micro-F1={best_f1:.4f}")

    if args.no_export_predictions:
        return

    export_root = Path(args.export_dir)
    export_root.mkdir(parents=True, exist_ok=True)
    slug = f"committee_{args.arch}_{best_mode}"
    out_dir = export_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    def _apply_thr(proba: np.ndarray, pids: List[int], mode: str) -> Dict[int, List[str]]:
        if mode == "global":
            return proba_to_preds(proba, pids, all_labels, float(best_t))
        assert best_pl is not None
        return proba_to_preds_per_label(proba, pids, all_labels, best_pl)

    val_preds = _apply_thr(val_proba, val_pids, best_mode)
    save_predictions_jsonl(val_preds, out_dir / "val_predictions.jsonl")
    print(f"\n--- Export slug={slug} ---\n  val → {out_dir / 'val_predictions.jsonl'}")

    def _export_split(name: str, cfg_key: str, pred_split: str) -> None:
        pth = str(get_cfg(cfg, cfg_key, ""))
        if not pth or not Path(pth).is_file():
            print(f"  {name:<6} skipped ({cfg_key})")
            return
        gt = load_ground_truth(pth)
        pids = list(gt.keys())
        try:
            mats = _load_matrices_for_split(model_cfgs, names, is_score_model, pids, all_labels, pred_split)
        except FileNotFoundError as exc:
            print(f"  {name:<6} skipped ({exc})")
            return
        proba = _predict_proba_grid(model, mats, dev, grid_mu, grid_sig)
        preds = _apply_thr(proba, pids, best_mode)
        outp = out_dir / f"{name}_predictions.jsonl"
        save_predictions_jsonl(preds, outp)
        print(f"  {name:<6} → {outp}")

    _export_split("test", "data.test_path", "compare")
    _export_split("blind", "data.blind_path", "blind")

    manifest = {
        "export_root": str(export_root.resolve()),
        "method": "ensemble_committee_mlp",
        "arch": args.arch,
        "best_threshold_mode": best_mode,
        "best_threshold": best_t if best_mode == "global" else None,
        "val_micro_f1": best_f1,
        "hidden": args.hidden if args.arch == "flatten" else None,
        "conv_channels": args.conv_channels if args.arch == "conv" else None,
        "conv_mlp_hidden": args.conv_mlp_hidden if args.arch == "conv" else None,
        "models": names,
        "slug": slug,
    }
    with open(export_root / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"  manifest → {export_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
