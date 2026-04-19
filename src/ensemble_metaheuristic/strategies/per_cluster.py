"""Per-cluster champion: one base model per document cluster (val micro-F1)."""
from __future__ import annotations

from typing import Dict, List, Tuple

try:
    from src.evaluation.evaluator import evaluate_data
except ImportError:
    from ...evaluation.evaluator import evaluate_data


def build_cluster_champion_routing(
    cluster_assignments: Dict[int, int],
    all_pids: List[int],
    names: List[str],
    per_model_preds: Dict[str, Dict[int, List[str]]],
    gt_data: Dict,
    all_labels: List[str],
    default_model: str = "mlc_greek_bert",
) -> Tuple[Dict[int, str], Dict[int, float]]:
    """
    For each cluster id that appears on at least one validation patient, pick the
    base model with highest micro-F1 on patients in that cluster (same preds as per-label matrices).
    """
    cluster_ids = sorted({cluster_assignments[p] for p in all_pids if p in cluster_assignments})
    if not cluster_ids:
        return {}, {}

    routing: Dict[int, str] = {}
    scores: Dict[int, float] = {}
    for cid in cluster_ids:
        pids_in = [p for p in all_pids if cluster_assignments.get(p) == cid]
        if not pids_in:
            continue
        best_name, best_f1 = default_model, -1.0
        for name in names:
            sub_gt = {p: gt_data[p] for p in pids_in if p in gt_data}
            sub_pred = {p: per_model_preds[name].get(p, []) for p in pids_in if p in gt_data}
            f1 = evaluate_data(sub_gt, sub_pred, label_space=all_labels)["micro_f1"]
            if f1 > best_f1:
                best_f1, best_name = f1, name
        routing[cid] = best_name
        scores[cid] = best_f1
    return routing, scores


def per_cluster_champion_predict(
    cluster_assignments: Dict[int, int],
    all_pids: List[int],
    cluster_routing: Dict[int, str],
    per_model_preds: Dict[str, Dict[int, List[str]]],
    default_model: str = "mlc_greek_bert",
) -> Dict[int, List[str]]:
    """For each patient, take flat predictions from the champion model of that patient's cluster."""
    pred_data: Dict[int, List[str]] = {}
    for pid in all_pids:
        cid = cluster_assignments.get(pid)
        if cid is None:
            champ = default_model
        else:
            champ = cluster_routing.get(cid, default_model)
        pred_data[pid] = list(per_model_preds[champ].get(pid, []))
    return pred_data


def _run_standalone_cli() -> None:
    import argparse
    import json
    from pathlib import Path

    from src.ensemble_metaheuristic.strategy_cli import (
        build_per_model_preds,
        load_validation_bundle,
        prepend_repo_root_for_strategy_file,
    )

    try:
        from src.evaluation.config_utils import clustering_output_dir, load_config
        from src.evaluation.evaluator import evaluate_data
    except ImportError:
        from ...evaluation.config_utils import clustering_output_dir, load_config
        from ...evaluation.evaluator import evaluate_data

    prepend_repo_root_for_strategy_file(Path(__file__))

    ap = argparse.ArgumentParser(
        description="Per-cluster champion from analysis cluster_assignments.json (this module only).",
    )
    ap.add_argument("--config", default="src/analysis/analysis.yaml", help="Analysis YAML.")
    ap.add_argument(
        "--cluster-json",
        type=str,
        default="",
        help="Path to cluster_assignments.json (default: from config clustering output dir).",
    )
    args = ap.parse_args()

    matrices, names, is_score_model, gt_data, all_pids, all_labels, _mc, _vp = load_validation_bundle(
        args.config,
    )
    cfg = load_config(args.config)
    cluster_path = (
        Path(args.cluster_json)
        if str(args.cluster_json).strip()
        else clustering_output_dir(cfg) / "cluster_assignments.json"
    )

    print("Per-cluster champion (this module only)")
    print(f"  Cluster file: {cluster_path}")
    if not cluster_path.is_file():
        print("  Skipped (file missing).")
        return

    cluster_assignments = {int(k): int(v) for k, v in json.loads(cluster_path.read_text(encoding="utf-8")).items()}
    per_model_preds = build_per_model_preds(matrices, names, is_score_model, all_pids, all_labels)
    cluster_routing, cluster_scores = build_cluster_champion_routing(
        cluster_assignments, all_pids, names, per_model_preds, gt_data, all_labels,
    )
    if not cluster_routing:
        print("  Skipped (no cluster id covers any validation patient).")
        return

    for cid in sorted(cluster_routing):
        print(
            f"  cluster {cid}: {cluster_routing[cid]}  (subset micro-F1={cluster_scores[cid]:.4f})",
        )
    preds = per_cluster_champion_predict(
        cluster_assignments, all_pids, cluster_routing, per_model_preds,
    )
    m = evaluate_data(gt_data, preds, label_space=all_labels)
    print(
        f"  micro-F1={m['micro_f1']:.4f}  precision={m['precision']:.4f}  recall={m['recall']:.4f}",
    )


if __name__ == "__main__":
    _run_standalone_cli()
