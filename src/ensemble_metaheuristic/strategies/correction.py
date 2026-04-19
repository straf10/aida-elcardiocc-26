"""Correction mode: start from a base model, add/remove codes by other-model consensus."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

try:
    from src.evaluation.evaluator import evaluate_data
except ImportError:
    from ...evaluation.evaluator import evaluate_data


def correction_predict(
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    names: List[str],
    all_pids: List[int],
    all_labels: List[str],
    base_model: str = "mlc_greek_bert",
    *,
    add_min_votes: int = 2,
    add_min_score_factor: float = 1.0,
    remove_if_zero_votes: bool = False,
) -> Dict[int, List[str]]:
    """
    Start with base_model predictions (caller often passes best individual by val F1), then:
    - ADD codes that ≥ add_min_votes other models confidently predict
    - Optionally REMOVE base codes that no other model supports
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    base_idx = name_to_idx[base_model]
    base_cutoff = 1.0 if is_score_model[base_idx] else 0.5
    other_idxs = [i for i, n in enumerate(names) if n != base_model]

    pred_data: Dict[int, List[str]] = {}

    for row_i, pid in enumerate(all_pids):
        base_set = set(
            all_labels[j] for j in np.where(matrices[base_idx][row_i] >= base_cutoff)[0]
        )

        for j, label in enumerate(all_labels):
            other_votes = 0
            for idx in other_idxs:
                cutoff = add_min_score_factor if is_score_model[idx] else 0.5
                if matrices[idx][row_i, j] >= cutoff:
                    other_votes += 1

            if label not in base_set and other_votes >= add_min_votes:
                base_set.add(label)
            elif label in base_set and remove_if_zero_votes and other_votes == 0:
                base_set.discard(label)

        pred_data[pid] = sorted(base_set)

    return pred_data


def search_correction_params(
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    names: List[str],
    all_pids: List[int],
    all_labels: List[str],
    gt_data: Dict,
    base_model: str = "mlc_greek_bert",
    *,
    extended: bool = False,
) -> Tuple[dict, float]:
    """Grid search over correction mode params, return best config and F1."""
    best_f1 = -1.0
    best_cfg: dict = {}

    if extended:
        add_votes_list = [1, 2, 3, 4, 5]
        add_factor_list = [round(x, 2) for x in np.linspace(0.55, 1.65, 13)]
    else:
        add_votes_list = [1, 2, 3]
        add_factor_list = [0.8, 1.0, 1.2, 1.5]

    n_evals = 0
    for add_votes in add_votes_list:
        for add_factor in add_factor_list:
            for remove in [False, True]:
                preds = correction_predict(
                    matrices, is_score_model, names, all_pids, all_labels,
                    base_model=base_model,
                    add_min_votes=add_votes,
                    add_min_score_factor=add_factor,
                    remove_if_zero_votes=remove,
                )
                f1 = evaluate_data(gt_data, preds, label_space=all_labels)["micro_f1"]
                n_evals += 1
                if f1 > best_f1:
                    best_f1 = f1
                    best_cfg = {
                        "add_min_votes": add_votes,
                        "add_min_score_factor": float(add_factor),
                        "remove_if_zero_votes": remove,
                    }

    best_cfg["_grid_evaluations"] = n_evals
    return best_cfg, best_f1


def _run_standalone_cli() -> None:
    import argparse
    from pathlib import Path

    from src.ensemble_metaheuristic.strategy_cli import (
        load_validation_bundle,
        prepend_repo_root_for_strategy_file,
    )

    try:
        from src.evaluation.evaluator import evaluate_data
    except ImportError:
        from ...evaluation.evaluator import evaluate_data

    prepend_repo_root_for_strategy_file(Path(__file__))

    ap = argparse.ArgumentParser(description="Correction-mode grid search (this module only).")
    ap.add_argument("--config", default="src/analysis/analysis.yaml", help="Analysis YAML.")
    ap.add_argument(
        "--extended",
        action="store_true",
        help="Larger search grid (same as search_correction_params extended=True).",
    )
    args = ap.parse_args()

    matrices, names, is_score_model, gt_data, all_pids, all_labels, _mc, _vp = load_validation_bundle(
        args.config,
    )

    individual_micro_f1: Dict[str, float] = {}
    for name, mat, is_score in zip(names, matrices, is_score_model):
        cutoff = 1.0 if is_score else 0.5
        preds = {pid: [all_labels[j] for j in np.where(mat[i] >= cutoff)[0]] for i, pid in enumerate(all_pids)}
        individual_micro_f1[name] = evaluate_data(gt_data, preds, label_space=all_labels)["micro_f1"]
    best_single_name = min(names, key=lambda n: (-individual_micro_f1[n], names.index(n)))

    print("Correction mode (this module only)")
    print(f"  Base model (best single on val): {best_single_name}  micro-F1={individual_micro_f1[best_single_name]:.4f}")
    best_cfg, best_f1 = search_correction_params(
        matrices,
        is_score_model,
        names,
        all_pids,
        all_labels,
        gt_data,
        base_model=best_single_name,
        extended=bool(args.extended),
    )
    n_grid = best_cfg.pop("_grid_evaluations", None)
    if n_grid is not None:
        print(f"  Grid evaluations: {n_grid}")
    preds = correction_predict(
        matrices,
        is_score_model,
        names,
        all_pids,
        all_labels,
        base_model=best_single_name,
        **best_cfg,
    )
    m = evaluate_data(gt_data, preds, label_space=all_labels)
    print(f"  Best config: {best_cfg}")
    print(
        f"  micro-F1={m['micro_f1']:.4f}  precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
        f"(search best F1={best_f1:.4f})",
    )


if __name__ == "__main__":
    _run_standalone_cli()
