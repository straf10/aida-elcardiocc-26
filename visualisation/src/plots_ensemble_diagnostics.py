from __future__ import annotations

"""Ensemble diagnostics: calibration, FN intersections, TP/FP score shapes, error correlation, length vs F1.

Calibration and TP/FP histograms require ``type: scores`` models (sigmoid probabilities). Models with
``predictions_only`` are skipped for those plots but included in FN intersections, correlation, and length plots.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import cohen_kappa_score

try:
    from src.preprocessing.io_utils import load_jsonl, resolve_patient_id
except ImportError:
    from preprocessing.io_utils import load_jsonl, resolve_patient_id  # type: ignore

from visualisation.src.config import MODEL_ABBREV
from visualisation.src.cross_model_data import CrossModelBundle
from visualisation.src.ensemble_artifacts import (
    ModelArtifacts,
    load_thresholds_vector,
    score_model_cfgs,
    y_pred_binary_matrix,
    y_true_matrix_for_artifact,
)

def _plot_fn_intersection_matrix(
    counter: Counter[Tuple[str, ...]],
    out_path: Path,
    title: str,
) -> None:
    """Matplotlib bar + dot matrix for FN intersection patterns (UpSet-style; robust vs upsetplot/pandas CoW)."""
    if not counter:
        return
    patterns = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    n = min(len(patterns), 45)
    patterns = patterns[:n]
    all_tags = sorted({a for k in counter for a in k})
    cols = np.arange(n)
    counts = np.array([c for _, c in patterns], dtype=float)
    memb_list = [set(m) for m, _ in patterns]

    fig, (ax_bar, ax_mat) = plt.subplots(
        2,
        1,
        figsize=(max(10, 0.35 * n), 7),
        height_ratios=[1, 1.6],
        sharex=True,
        layout="constrained",
    )
    ax_bar.bar(cols, counts, color="steelblue", width=0.8)
    ax_bar.set_ylabel("Count")
    ax_bar.set_title(title)

    n_tags = len(all_tags)
    for j in range(n):
        for yi, tag in enumerate(all_tags):
            filled = tag in memb_list[j]
            ax_mat.scatter(
                j,
                yi,
                s=100,
                c="0.15" if filled else "0.85",
                edgecolors="0.3",
                linewidths=0.5,
                zorder=2,
            )
    ax_mat.set_yticks(range(n_tags))
    ax_mat.set_yticklabels(all_tags)
    ax_mat.set_xticks(cols)
    labels = [",".join(sorted(m)) if m else "—" for m, _ in patterns]
    ax_mat.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax_mat.set_xlabel("Missed-by pattern (which models miss this GT group)")
    ax_mat.set_ylabel("Model")
    ax_mat.set_ylim(-0.5, n_tags - 0.5)
    ax_mat.invert_yaxis()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _abbrev(name: str) -> str:
    return MODEL_ABBREV.get(name, name[:3])


def plot_calibration_curves(
    bundle: CrossModelBundle,
    artifacts_by_name: Dict[str, ModelArtifacts],
    out_path: Path,
    n_bins: int = 10,
) -> None:
    """Reliability diagrams for score-based models (soft-voting calibration). Skips predictions_only."""
    score_cfgs = {m["name"]: m for m in score_model_cfgs(bundle)}
    models_with_scores = [n for n in bundle.model_names if n in score_cfgs and artifacts_by_name[n].scores is not None]
    if not models_with_scores:
        print("[ensemble] Calibration: no models with score matrices; skipping.")
        return

    n = len(models_with_scores)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for ax, name in zip(axes[0], models_with_scores):
        art = artifacts_by_name[name]
        assert art.scores is not None
        cfg = score_cfgs[name]
        thr = load_thresholds_vector(cfg, art.label_names)
        y_true = y_true_matrix_for_artifact(bundle.gt_data, art)
        y_prob = art.scores.astype(np.float64).ravel()
        y_bin = y_true.astype(np.int32).ravel()
        prob_pred, prob_true = calibration_curve(y_bin, y_prob, n_bins=n_bins, strategy="uniform")
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax.plot(prob_pred, prob_true, "s-", label=_abbrev(name))
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives (empirical)")
        ax.set_title(f"Calibration — {_abbrev(name)}")
        ax.legend(loc="lower right")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Διαγράμματα βαθμονόμησης / Calibration (readiness for soft voting)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ensemble] Wrote calibration plot to {out_path.resolve()}")


def plot_fn_upset(bundle: CrossModelBundle, out_path: Path) -> None:
    """UpSet of which models miss each GT group (document-level FN pattern)."""
    counter: Counter[Tuple[str, ...]] = Counter()
    for pid in bundle.patient_ids:
        for group in bundle.gt_data.get(pid, []):
            if not group:
                continue
            missed: List[str] = []
            gset = set(group)
            for m in bundle.model_names:
                pred = set(bundle.pred_by_model[m].get(pid, []))
                if not pred.intersection(gset):
                    missed.append(m)
            if not missed:
                continue
            key = tuple(sorted(_abbrev(m) for m in missed))
            counter[key] += 1

    if not counter:
        print("[ensemble] FN UpSet: no missed groups; skipping.")
        return

    title = "False negatives on GT groups — intersection of who misses each entity"
    _plot_fn_intersection_matrix(counter, out_path, title)
    print(f"[ensemble] Wrote FN intersection plot (UpSet-style) to {out_path.resolve()}")


def _top_codes_by_support(gt_data: Dict[int, List[List[str]]], label_names: List[str], k: int) -> List[str]:
    sup: Counter[str] = Counter()
    for groups in gt_data.values():
        for group in groups:
            for c in group:
                if c in label_names:
                    sup[c] += 1
    ordered = [c for c, _ in sup.most_common()]
    rest = [c for c in label_names if c not in ordered]
    out = ordered[:k]
    if len(out) < k:
        out.extend(rest[: k - len(out)])
    return out[:k]


def plot_tp_fp_score_histograms(
    bundle: CrossModelBundle,
    artifacts_by_name: Dict[str, ModelArtifacts],
    out_path: Path,
    top_classes: int = 10,
) -> None:
    """Per score model: TP vs FP score distributions for top-supported ICD codes."""
    score_cfgs = {m["name"]: m for m in score_model_cfgs(bundle)}
    models_with_scores = [n for n in bundle.model_names if n in score_cfgs and artifacts_by_name[n].scores is not None]
    if not models_with_scores:
        print("[ensemble] TP/FP histograms: no score models; skipping.")
        return

    first = artifacts_by_name[models_with_scores[0]]
    codes = _top_codes_by_support(bundle.gt_data, first.label_names, top_classes)
    nrows = len(models_with_scores)
    ncols = len(codes)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows), squeeze=False)
    for ri, name in enumerate(models_with_scores):
        art = artifacts_by_name[name]
        assert art.scores is not None
        cfg = score_cfgs[name]
        thr = load_thresholds_vector(cfg, art.label_names)
        y_true = y_true_matrix_for_artifact(bundle.gt_data, art)
        y_pred = y_pred_binary_matrix(art.scores, thr)
        idx_map = {c: i for i, c in enumerate(art.label_names)}
        for ci, code in enumerate(codes):
            ax = axes[ri][ci]
            j = idx_map.get(code)
            if j is None:
                ax.set_visible(False)
                continue
            tp_scores = art.scores[:, j][(y_true[:, j] == 1) & (y_pred[:, j] == 1)]
            fp_scores = art.scores[:, j][(y_true[:, j] == 0) & (y_pred[:, j] == 1)]
            if tp_scores.size == 0 and fp_scores.size == 0:
                ax.text(0.5, 0.5, "no TP/FP", ha="center", va="center", transform=ax.transAxes)
            else:
                bins = np.linspace(0, 1, 21)
                ax.hist(tp_scores, bins=bins, alpha=0.55, label="TP", color="C0", density=True)
                ax.hist(fp_scores, bins=bins, alpha=0.55, label="FP", color="C3", density=True)
                ax.legend(fontsize=7)
            ax.set_xlim(0, 1)
            if ri == nrows - 1:
                ax.set_xlabel(code, fontsize=8)
            if ci == 0:
                ax.set_ylabel(_abbrev(name), fontsize=9)
    fig.suptitle("Score distributions: TP vs FP (top GT-support codes)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ensemble] Wrote TP/FP histograms to {out_path.resolve()}")


def _doc_error_vectors(bundle: CrossModelBundle) -> Tuple[np.ndarray, List[str]]:
    """Shape (n_models, n_docs), binary 1 = (fp+fn) > 0 for that document."""
    names = bundle.model_names
    pids = bundle.patient_ids
    pid_pos = {pid: i for i, pid in enumerate(pids)}
    mat = np.zeros((len(names), len(pids)), dtype=np.int8)
    for mi, m in enumerate(names):
        for row in bundle.metrics_by_model[m]["doc_breakdown"]:
            pid = int(row["patient_id"])
            j = pid_pos.get(pid)
            if j is None:
                continue
            fp = int(row.get("fp", 0))
            fn = int(row.get("fn", 0))
            mat[mi, j] = 1 if (fp + fn) > 0 else 0
    return mat, names


def plot_error_correlation_heatmaps(bundle: CrossModelBundle, out_path: Path) -> None:
    """Pearson correlation and Cohen's kappa on per-document binary error indicators."""
    mat, names = _doc_error_vectors(bundle)
    labels = [_abbrev(n) for n in names]
    n = len(names)
    x = mat.astype(np.float64)
    pearson = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = x[i], x[j]
            sa, sb = float(np.std(a)), float(np.std(b))
            if sa < 1e-12 or sb < 1e-12:
                r = 0.0
            else:
                r = float(np.corrcoef(a, b)[0, 1])
                if not np.isfinite(r):
                    r = 0.0
            pearson[i, j] = pearson[j, i] = r
    kappa = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                kappa[i, j] = 1.0
            else:
                kappa[i, j] = float(cohen_kappa_score(mat[i], mat[j]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    sns.heatmap(
        pearson,
        ax=ax1,
        xticklabels=labels,
        yticklabels=labels,
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        square=True,
        annot=True,
        fmt=".2f",
    )
    ax1.set_title("Pearson r (binary doc error)")
    sns.heatmap(
        kappa,
        ax=ax2,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cmap="viridis",
        square=True,
        annot=True,
        fmt=".2f",
    )
    ax2.set_title("Cohen's κ (pairwise doc error)")
    fig.suptitle("Error correlation across models", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ensemble] Wrote error correlation heatmaps to {out_path.resolve()}")


def _word_counts_by_pid(val_jsonl_path: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for rec in load_jsonl(val_jsonl_path):
        pid = resolve_patient_id(rec)
        text = rec.get("text") or ""
        out[pid] = len(str(text).split())
    return out


def _per_doc_f1(tp: int, fp: int, fn: int) -> float:
    """Per-document micro-F1 on group-level tp/fp/fn. Perfect empty: tp=fp=fn=0 → 1.0."""
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    den = 2 * tp + fp + fn
    return (2.0 * tp / den) if den > 0 else 0.0


def plot_length_vs_performance(
    bundle: CrossModelBundle,
    out_path: Path,
) -> None:
    """Scatter: word count vs per-doc F1 and vs error count (fp+fn), one column per metric."""
    try:
        from src.evaluation.config_utils import get_cfg
    except ImportError:
        from evaluation.config_utils import get_cfg  # type: ignore

    val_path = get_cfg(bundle.cfg, "data.val_path")
    wc = _word_counts_by_pid(str(val_path))
    pids = bundle.patient_ids
    x = np.array([wc.get(pid, 0) for pid in pids], dtype=np.float64)

    n_models = len(bundle.model_names)
    fig, axes = plt.subplots(n_models, 2, figsize=(10, 2.8 * n_models), squeeze=False)
    for mi, m in enumerate(bundle.model_names):
        by_pid = {int(r["patient_id"]): r for r in bundle.metrics_by_model[m]["doc_breakdown"]}
        f1s: List[float] = []
        errs: List[int] = []
        for pid in pids:
            row = by_pid.get(pid, {"tp": 0, "fp": 0, "fn": 0})
            tp = int(row.get("tp", 0))
            fp = int(row.get("fp", 0))
            fn = int(row.get("fn", 0))
            f1s.append(_per_doc_f1(tp, fp, fn))
            errs.append(fp + fn)
        y_f1 = np.array(f1s, dtype=np.float64)
        y_err = np.array(errs, dtype=np.float64)
        ax0 = axes[mi][0]
        ax1 = axes[mi][1]
        ax0.scatter(x, y_f1, alpha=0.35, s=12, c="C0")
        ax0.set_ylabel(_abbrev(m))
        ax0.set_xlabel("Word count")
        ax0.set_ylim(-0.05, 1.05)
        ax0.set_title("Per-doc F1 vs length" if mi == 0 else "")
        ax1.scatter(x, y_err, alpha=0.35, s=12, c="C1")
        ax1.set_xlabel("Word count")
        ax1.set_ylabel(_abbrev(m))
        ax1.set_title("FP+FN vs length" if mi == 0 else "")
    fig.suptitle("Performance vs sequence length (validation)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ensemble] Wrote length vs performance to {out_path.resolve()}")


def run_ensemble_diagnostics(
    bundle: CrossModelBundle,
    artifacts_by_name: Dict[str, ModelArtifacts],
    out_dir: Path,
    top_classes: int = 10,
    skip_upset: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_calibration_curves(bundle, artifacts_by_name, out_dir / "ensemble_calibration.png")
    if not skip_upset:
        plot_fn_upset(bundle, out_dir / "ensemble_fn_upset.png")
    plot_tp_fp_score_histograms(
        bundle,
        artifacts_by_name,
        out_dir / "ensemble_tp_fp_score_hists.png",
        top_classes=top_classes,
    )
    plot_error_correlation_heatmaps(bundle, out_dir / "ensemble_error_correlation.png")
    plot_length_vs_performance(bundle, out_dir / "ensemble_length_vs_performance.png")
