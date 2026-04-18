from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.visualisation.src.config import MODEL_ABBREV
from src.visualisation.src.cross_model_data import CrossModelBundle, per_class_fn


def _per_class_rows_by_code(bundle: CrossModelBundle) -> Dict[str, Dict[str, dict]]:
    """code -> model_name -> per_class row."""
    out: Dict[str, Dict[str, dict]] = {}
    for name in bundle.model_names:
        metrics = bundle.metrics_by_model[name]
        for row in metrics.get("per_class", []):
            code = row["code"]
            out.setdefault(code, {})[name] = row
    return out


def build_code_rescue_table(bundle: CrossModelBundle) -> List[dict]:
    """
    One row per label with support>0 where at least one model has FN>0
    (FN = support - groups_hit at group level for that code).
    """
    by_code = _per_class_rows_by_code(bundle)
    rows_out: List[dict] = []

    for code in bundle.label_names:
        per_m = by_code.get(code, {})
        if not per_m:
            continue
        support = max(int(r.get("support", 0)) for r in per_m.values())
        if support <= 0:
            continue
        fn_by: Dict[str, int] = {}
        recall_by: Dict[str, float] = {}
        for name in bundle.model_names:
            row = per_m.get(name, {})
            fn_by[name] = per_class_fn(row)
            recall_by[name] = float(row.get("recall", 0.0)) if row else 0.0
        total_fn = sum(fn_by.values())
        if not any(fn_by.get(n, 0) > 0 for n in bundle.model_names):
            continue
        perfect = [n for n in bundle.model_names if fn_by.get(n, 0) == 0]
        rescuer_txt = "+".join(MODEL_ABBREV.get(n, n[:3]) for n in sorted(perfect)) if perfect else "none"
        rows_out.append(
            {
                "code": code,
                "support": support,
                "total_fn_sum": total_fn,
                "rescuers_fn0": rescuer_txt,
                **{f"fn_{n}": fn_by.get(n, 0) for n in bundle.model_names},
                **{f"recall_{n}": recall_by.get(n, 0.0) for n in bundle.model_names},
            }
        )

    rows_out.sort(key=lambda r: (-r["total_fn_sum"], -r["support"], r["code"]))
    return rows_out


def plot_code_rescue(bundle: CrossModelBundle, out_png: Path, out_csv: Path) -> None:
    table = build_code_rescue_table(bundle)
    if not table:
        return

    codes = [r["code"] for r in table]
    cols = bundle.model_names
    recall_mat = np.array([[r[f"recall_{n}"] for n in cols] for r in table], dtype=float)

    h = max(8, min(40, len(codes) * 0.22))
    plt.figure(figsize=(10, h))
    sns.heatmap(
        recall_mat,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        xticklabels=[MODEL_ABBREV.get(n, n) for n in cols],
        yticklabels=codes,
        cbar_kws={"label": "Recall (group-level)"},
    )
    plt.xlabel("Model")
    plt.ylabel("True code (any model has missed groups)")
    plt.title(
        "Per-code recall for labels missed by ≥1 model\n"
        "(xlm_r_base excluded). CSV lists models with FN=0 as rescuers_fn0."
    )
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()

    fieldnames = (
        ["code", "support", "total_fn_sum", "rescuers_fn0"]
        + [f"fn_{n}" for n in cols]
        + [f"recall_{n}" for n in cols]
    )
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in table:
            w.writerow(r)
