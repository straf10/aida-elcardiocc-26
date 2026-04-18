from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from visualisation.src.config import MODEL_ABBREV
from visualisation.src.cross_model_data import CrossModelBundle, DocWrongEdge, top_pairs_subset

Pair = Tuple[str, str]


def _edges_for_pair(bundle: CrossModelBundle, pair: Pair) -> List[DocWrongEdge]:
    p, t = pair
    out: List[DocWrongEdge] = []
    for name in bundle.model_names:
        for e in bundle.doc_edges_by_model.get(name, []):
            if e.predicted == p and e.missed == t:
                out.append(e)
    return out


def _cell_rescue_stats(bundle: CrossModelBundle, pair: Pair) -> Tuple[float, str]:
    """Return (rescue_rate, annotation string of unique rescuer abbrevs)."""
    edges = _edges_for_pair(bundle, pair)
    if not edges:
        return 0.0, ""
    n_rescued = 0
    rescuers: Set[str] = set()
    for e in edges:
        rs = bundle.rescuers_for_edge(e)
        if rs:
            n_rescued += 1
        for m in rs:
            rescuers.add(MODEL_ABBREV.get(m, m[:3]))
    rate = n_rescued / len(edges)
    ann = ",".join(sorted(rescuers)) if rescuers else "-"
    return rate, ann


def _matrix_for_pairs(bundle: CrossModelBundle, pairs: List[Pair]) -> Tuple[np.ndarray, List[str]]:
    """Rows = true/missed (T), cols = predicted (P), values = pooled counts."""
    codes: Set[str] = set()
    for p, t in pairs:
        codes.add(p)
        codes.add(t)
    top_codes = sorted(codes)
    n = len(top_codes)
    idx = {c: i for i, c in enumerate(top_codes)}
    mat = np.zeros((n, n))
    for pr, tr in pairs:
        if pr in idx and tr in idx:
            mat[idx[tr], idx[pr]] = float(bundle.pooled_wrong_pairs[(pr, tr)])
    return mat, top_codes


def plot_pooled_annotated(
    bundle: CrossModelBundle,
    pairs: List[Pair],
    out_path: Path,
) -> None:
    mat, labels = _matrix_for_pairs(bundle, pairs)
    if mat.size == 0:
        return

    annot = np.empty(mat.shape, dtype=object)
    for i, ti in enumerate(labels):
        for j, pj in enumerate(labels):
            v = mat[i, j]
            if v <= 0:
                annot[i, j] = ""
                continue
            _, rtxt = _cell_rescue_stats(bundle, (pj, ti))
            if len(rtxt) > 12:
                rtxt = rtxt[:10] + "…"
            annot[i, j] = f"{v:.0f}\n{rtxt}" if rtxt else f"{v:.0f}"

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        mat,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Pooled wrong-pair count"},
    )
    plt.xlabel("Predicted code (FP)")
    plt.ylabel("True code (missed)")
    plt.title(
        "Pooled confusion (4 models; xlm_r_base excluded)\n"
        "Cell = sum of wrong-pair counts; second line = other models predicting true code on same doc"
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_rescue_rate(
    bundle: CrossModelBundle,
    pairs: List[Pair],
    out_path: Path,
) -> None:
    mat, labels = _matrix_for_pairs(bundle, pairs)
    if mat.size == 0:
        return

    rate_mat = np.zeros_like(mat)
    for i, ti in enumerate(labels):
        for j, pj in enumerate(labels):
            if mat[i, j] <= 0:
                continue
            rate, _ = _cell_rescue_stats(bundle, (pj, ti))
            rate_mat[i, j] = rate

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        rate_mat,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Rescue rate"},
    )
    plt.xlabel("Predicted code (FP)")
    plt.ylabel("True code (missed)")
    plt.title(
        "Cross-model rescue rate\n"
        "Fraction of (doc, model) wrong-pair events where ≥1 other model predicts the missed true code"
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_error_pair_by_model(
    bundle: CrossModelBundle,
    pairs: List[Pair],
    out_path: Path,
) -> None:
    rows = [f"{p} → {t}" for p, t in pairs]
    cols = bundle.model_names
    data = np.zeros((len(pairs), len(cols)))
    for i, pair in enumerate(pairs):
        for j, name in enumerate(cols):
            data[i, j] = float(bundle.wrong_pairs_by_model[name].get(pair, 0))

    plt.figure(figsize=(max(8, len(cols) * 2), max(6, len(rows) * 0.35)))
    sns.heatmap(
        data,
        annot=True,
        fmt=".0f",
        cmap="Purples",
        xticklabels=[MODEL_ABBREV.get(c, c) for c in cols],
        yticklabels=rows,
    )
    plt.xlabel("Model")
    plt.ylabel("(predicted → missed)")
    plt.title("Wrong-pair counts by model (xlm_r_base excluded)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_cross_confusion_plots(
    bundle: CrossModelBundle,
    important_pairs: Set[Pair],
    out_dir: Path,
    top_n: int,
) -> List[Pair]:
    pairs = top_pairs_subset(bundle.pooled_wrong_pairs, important_pairs, top_n)
    plot_pooled_annotated(bundle, pairs, out_dir / "pooled_confusion_annotated.png")
    plot_rescue_rate(bundle, pairs, out_dir / "pooled_confusion_rescue_rate.png")
    plot_error_pair_by_model(bundle, pairs, out_dir / "error_pair_by_model.png")
    return pairs
