"""Paired bootstrap + McNemar significance test: best ensemble vs. best single model.

Run: python -m evaluation.significance_test [--config src/evaluation/config.yaml] [--n-boot 10000] [--seed 42]

Reuses the same score matrices / weighted-search machinery as ``ensemble_metaheuristic``
so the ensemble side is the actual weighted-fusion strategy, evaluated on validation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

_src = Path(__file__).resolve().parents[1]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np

from evaluation.config_utils import load_config, get_cfg
from evaluation.scoring import evaluate_data, score_document
from evaluation.io_utils import load_ground_truth
from ensemble_metaheuristic.matrices import build_score_matrix, load_thresholds_for_model
from ensemble_metaheuristic.strategy_loaders import gather_ensemble_artifacts, canonical_ensemble_label_arts
from ensemble_metaheuristic.strategies import run_search, weighted_ensemble_predict


def bootstrap_ci(pids: List[int], gt_data: Dict, predA: Dict, predB: Dict, n_boot: int, rng: np.random.RandomState):
    """Paired bootstrap over documents. Returns (mean_diff, ci_lo, ci_hi, p_value) for F1(A) - F1(B)."""
    n = len(pids)
    diffs = np.empty(n_boot, dtype=np.float64)
    pids_arr = np.array(pids, dtype=object)
    for b in range(n_boot):
        sample = pids_arr[rng.randint(0, n, size=n)]
        tpA = fpA = fnA = tpB = fpB = fnB = 0
        for pid in sample:
            gt = gt_data[pid]
            t, f, n_ = score_document(gt, predA.get(pid, []))
            tpA += t; fpA += f; fnA += n_
            t, f, n_ = score_document(gt, predB.get(pid, []))
            tpB += t; fpB += f; fnB += n_
        f1A = 2 * tpA / (2 * tpA + fpA + fnA) if (2 * tpA + fpA + fnA) > 0 else 0.0
        f1B = 2 * tpB / (2 * tpB + fpB + fnB) if (2 * tpB + fpB + fnB) > 0 else 0.0
        diffs[b] = f1A - f1B
    mean_diff = float(diffs.mean())
    ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return mean_diff, ci_lo, ci_hi, float(p_value), diffs


def mcnemar_on_groups(pids: List[int], gt_data: Dict, predA: Dict, predB: Dict):
    """Paired McNemar test over (document, gold-group) pairs: does A cover the group vs. does B."""
    b = c = both = neither = 0
    for pid in pids:
        setA = set(predA.get(pid, []))
        setB = set(predB.get(pid, []))
        for group in gt_data[pid]:
            gset = set(group)
            hitA = bool(setA & gset)
            hitB = bool(setB & gset)
            if hitA and not hitB:
                b += 1
            elif hitB and not hitA:
                c += 1
            elif hitA and hitB:
                both += 1
            else:
                neither += 1
    # Exact McNemar (binomial) when b + c small; chi-square with continuity correction otherwise.
    n_disc = b + c
    if n_disc == 0:
        return b, c, both, neither, float("nan")
    if n_disc < 25:
        from math import comb
        k = min(b, c)
        p_value = 2 * sum(comb(n_disc, i) * 0.5 ** n_disc for i in range(0, k + 1))
        p_value = min(p_value, 1.0)
    else:
        chi2 = (abs(b - c) - 1) ** 2 / n_disc
        # chi-square(1) survival function without scipy: use complementary error function relation
        import math
        p_value = math.erfc(math.sqrt(chi2 / 2))
    return b, c, both, neither, float(p_value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="src/evaluation/config.yaml")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-iter", type=int, default=10000, help="Weighted-search budget (match ensemble_metaheuristic default).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    val_path = str(get_cfg(cfg, "data.val_path"))
    gt_data = load_ground_truth(val_path)
    all_pids = list(gt_data.keys())
    model_cfgs = {m["name"]: m for m in get_cfg(cfg, "models", [])}

    print("Loading model artifacts...")
    artifacts_list = gather_ensemble_artifacts(model_cfgs, all_pids, "val")
    all_labels = canonical_ensemble_label_arts(artifacts_list).label_names

    matrices, is_score_model = [], []
    for name, arts in artifacts_list:
        thr = load_thresholds_for_model(model_cfgs[name], all_labels) if arts.scores is not None else None
        matrices.append(build_score_matrix(arts, all_pids, all_labels, thr))
        is_score_model.append(arts.scores is not None)

    print("Individual model micro-F1:")
    individual_f1 = {}
    single_preds = {}
    for name, arts in artifacts_list:
        preds = {pid: list(arts.pred_data.get(pid, [])) for pid in all_pids}
        f1 = evaluate_data(gt_data, preds, label_space=arts.label_names)["micro_f1"]
        individual_f1[name] = f1
        single_preds[name] = preds
        print(f"  {name}: {f1:.4f}")
    best_single_name = max(individual_f1, key=individual_f1.get)
    best_single_f1 = individual_f1[best_single_name]
    print(f"Best single model: {best_single_name} ({best_single_f1:.4f})")

    print(f"\nRunning weighted search (classic, n_iter={args.n_iter}, seed={args.seed}) for ensemble predictions...")
    rng = np.random.RandomState(args.seed)
    w, mt, gt_thr, f1 = run_search(matrices, is_score_model, gt_data, all_pids, all_labels, args.n_iter, rng, verbose=False)
    ensemble_preds = weighted_ensemble_predict(matrices, is_score_model, w, mt, gt_thr, all_pids, all_labels)
    ensemble_f1 = evaluate_data(gt_data, ensemble_preds, label_space=all_labels)["micro_f1"]
    print(f"Weighted ensemble micro-F1: {ensemble_f1:.4f}")

    print(f"\n--- Paired bootstrap (B={args.n_boot}) : weighted ensemble vs. {best_single_name} ---")
    boot_rng = np.random.RandomState(args.seed)
    mean_diff, lo, hi, p_boot, _ = bootstrap_ci(all_pids, gt_data, ensemble_preds, single_preds[best_single_name], args.n_boot, boot_rng)
    print(f"Mean F1 diff (ensemble - {best_single_name}): {mean_diff:+.4f}")
    print(f"95% CI: [{lo:+.4f}, {hi:+.4f}]")
    print(f"Bootstrap two-sided p-value: {p_boot:.4g}")

    print(f"\n--- McNemar test (paired, per gold-group) : weighted ensemble vs. {best_single_name} ---")
    b, c, both, neither, p_mc = mcnemar_on_groups(all_pids, gt_data, ensemble_preds, single_preds[best_single_name])
    print(f"Groups covered by ensemble only: {b}")
    print(f"Groups covered by {best_single_name} only: {c}")
    print(f"Groups covered by both: {both} | neither: {neither}")
    print(f"McNemar p-value: {p_mc:.4g}")


if __name__ == "__main__":
    main()
