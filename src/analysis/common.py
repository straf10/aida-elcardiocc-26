from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Matches label_analysis.long_tail_metrics bucket keys (frequency-based).
LONG_TAIL_BUCKET_ORDER: Tuple[str, ...] = ("frequent", "medium", "rare")

import numpy as np
import scipy.sparse as sp

try:
    from ..evaluation.config_utils import clustering_output_dir, get_cfg, load_config
    from ..evaluation.model_artifacts import ModelArtifacts, load_model_artifacts
except ImportError:
    from src.evaluation.config_utils import clustering_output_dir, get_cfg, load_config
    from src.evaluation.model_artifacts import ModelArtifacts, load_model_artifacts


def _is_range_label(code: str) -> bool:
    return "-" in code


def build_confusion_views(metrics: Dict) -> Dict:
    """Aggregate FP/FN counts and wrong (pred, missed) pairs from evaluator ``doc_breakdown``."""
    fp_by_label = Counter()
    fn_by_label = Counter()
    wrong_pairs = Counter()
    hard_docs = []

    for row in metrics.get("doc_breakdown", []):
        missed_groups = row.get("missed_groups", [])
        wrong_codes = row.get("wrong_codes", [])

        for code in wrong_codes:
            fp_by_label[code] += 1
        for group in missed_groups:
            for code in group:
                fn_by_label[code] += 1

        for wrong_code in wrong_codes:
            for group in missed_groups:
                for missed_code in group:
                    wrong_pairs[(wrong_code, missed_code)] += 1

        hard_docs.append(
            {
                "patient_id": row.get("patient_id"),
                "tp": row.get("tp", 0),
                "fp": row.get("fp", 0),
                "fn": row.get("fn", 0),
                "wrong_codes": wrong_codes,
                "missed_groups": missed_groups,
            }
        )

    hardest_fp_docs = sorted(hard_docs, key=lambda x: x["fp"], reverse=True)[:25]
    hardest_fn_docs = sorted(hard_docs, key=lambda x: x["fn"], reverse=True)[:25]

    return {
        "fp_by_label": fp_by_label,
        "fn_by_label": fn_by_label,
        "wrong_pairs": wrong_pairs,
        "hardest_fp_docs": hardest_fp_docs,
        "hardest_fn_docs": hardest_fn_docs,
    }


def range_vs_specific_summary(
    per_class_rows: List[dict], fp_by_label: Counter, fn_by_label: Counter
) -> Dict:
    agg = defaultdict(lambda: {"support": 0, "fp": 0, "fn": 0, "labels": 0})
    for row in per_class_rows:
        code = row["code"]
        bucket = "range" if _is_range_label(code) else "specific"
        agg[bucket]["support"] += int(row.get("support", 0))
        agg[bucket]["fp"] += int(fp_by_label.get(code, 0))
        agg[bucket]["fn"] += int(fn_by_label.get(code, 0))
        agg[bucket]["labels"] += 1
    return agg


def ensure_model_artifacts(model_cfg: Dict[str, Any]) -> None:
    """Run predict module if the predictions JSONL is missing."""
    pred_path = model_cfg.get("predictions_path")
    if pred_path and Path(pred_path).exists():
        print(f"[{model_cfg['name']}] Found existing predictions at {pred_path}")
        return
    print(f"[{model_cfg['name']}] Predictions missing. Running inference...")
    if "predict_module" not in model_cfg:
        print(f"[{model_cfg['name']}] No predict_module specified, cannot run inference.")
        return
    cmd = ["python", "-m", model_cfg["predict_module"]]
    if "predict_args" in model_cfg:
        cmd.extend(model_cfg["predict_args"])
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def build_binary_matrices(
    gt_data: Dict[int, List[List[str]]],
    pred_data: Dict[int, List[str]],
    patient_ids: List[int],
    label_names: List[str],
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build scipy sparse CSR matrices for GT and Predictions (N_docs x 115)."""
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    n_docs = len(patient_ids)
    n_labels = len(label_names)

    y_true = sp.lil_matrix((n_docs, n_labels), dtype=np.int8)
    y_pred = sp.lil_matrix((n_docs, n_labels), dtype=np.int8)

    for i, pid in enumerate(patient_ids):
        gt_groups = gt_data.get(pid, [])
        for group in gt_groups:
            for code in group:
                if code in label_to_idx:
                    y_true[i, label_to_idx[code]] = 1

        preds = pred_data.get(pid, [])
        for code in preds:
            if code in label_to_idx:
                y_pred[i, label_to_idx[code]] = 1

    return y_true.tocsr(), y_pred.tocsr()


def label_support_from_gt(gt_data: Dict[int, List[List[str]]], label_names: List[str]) -> Counter:
    """Count groups containing each code (aligns with per_class_report semantics)."""
    support = Counter()
    for gt_groups in gt_data.values():
        for group in gt_groups:
            for code in set(group):
                if code in label_names:
                    support[code] += 1
    return support


def ensure_output_dir(cfg: Dict[str, Any]) -> Path:
    """Ensure and return the output directory."""
    out_dir = Path(get_cfg(cfg, "output.dir", "outputs/analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def analysis_reports_dir(cfg: Dict[str, Any]) -> Path:
    """Subdirectory for Markdown reports (under output.dir)."""
    base = Path(get_cfg(cfg, "output.dir", "outputs/analysis"))
    sub = get_cfg(cfg, "output.reports_subdir", "reports")
    p = base / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def analysis_summary_dir(cfg: Dict[str, Any]) -> Path:
    """Subdirectory for cross-model JSON and figures (under output.dir)."""
    base = Path(get_cfg(cfg, "output.dir", "outputs/analysis"))
    sub = get_cfg(cfg, "output.summary_subdir", "summary")
    p = base / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def collect_long_tail_comparison(
    out_dir: Path,
    model_names: List[str],
) -> Dict[str, Dict[str, dict]]:
    """Merge per-model label_analysis.json ``long_tail`` blocks for cross-model plots.

    Returns mapping: bucket_name -> model_name -> {macro_f1, weighted_f1, ...}.
    Omits models with missing or invalid ``label_analysis.json``.
    """
    result: Dict[str, Dict[str, dict]] = {b: {} for b in LONG_TAIL_BUCKET_ORDER}

    for model_name in model_names:
        path = out_dir / model_name / "label_analysis.json"
        if not path.exists():
            print(
                f"[collect_long_tail_comparison] WARN: missing {path}, skipping model {model_name}"
            )
            continue
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        lt = data.get("long_tail")
        if not isinstance(lt, dict):
            print(
                f"[collect_long_tail_comparison] WARN: no long_tail in {path}, skipping model {model_name}"
            )
            continue
        for b in LONG_TAIL_BUCKET_ORDER:
            row = lt.get(b)
            if not isinstance(row, dict):
                continue
            result[b][model_name] = {
                "macro_f1": float(row.get("macro_f1", 0.0)),
                "weighted_f1": float(row.get("weighted_f1", 0.0)),
                "n_labels": int(row.get("n_labels", 0)),
                "total_support": int(row.get("total_support", 0)),
            }

    non_empty = {k: v for k, v in result.items() if v}
    return non_empty
