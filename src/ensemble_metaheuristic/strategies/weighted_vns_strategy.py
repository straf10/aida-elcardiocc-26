"""Variable Neighborhood Search over weighted ensemble weights and thresholds."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .weighted_strategy import _sample_params, random_search, score_ensemble


def _shake_params_k(
    w: np.ndarray,
    mt: np.ndarray,
    gt: float,
    k: int,
    rng: np.random.RandomState,
    *,
    base_step: float = 0.07,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Shaking neighborhood N_k: perturbation radius grows with k (classic Basic VNS on R^d).
    k is 1-based; larger k explores farther from the incumbent.
    """
    kk = max(1, int(k))
    step = float(base_step) * float(kk)
    n_models = int(w.shape[0])
    n_score = int(mt.shape[0])
    nw = np.clip(w + rng.uniform(-step, step, size=n_models), 0.01, 2.0).astype(np.float32)
    nmt = np.clip(mt + rng.uniform(-step, step, size=n_score), 0.1, 3.0).astype(np.float32)
    ngt = float(np.clip(gt + rng.uniform(-step, step), 0.01, 2.5))
    return nw, nmt, ngt


def _shake_params_partition(
    w: np.ndarray,
    mt: np.ndarray,
    gt: float,
    k: int,
    rng: np.random.RandomState,
    *,
    step: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Alternate shake neighborhoods (analogous to intra- vs inter-route moves):
    k≡1 mod 4 weights only, 2 model thresholds, 3 global threshold, 0 all with full step.
    """
    kk = max(1, int(k))
    mode = kk % 4
    n_models = int(w.shape[0])
    n_score = int(mt.shape[0])
    nw, nmt, ngt = w.astype(np.float32).copy(), mt.astype(np.float32).copy(), float(gt)
    if mode == 1:
        nw = np.clip(w + rng.uniform(-step, step, size=n_models), 0.01, 2.0).astype(np.float32)
    elif mode == 2:
        nmt = np.clip(mt + rng.uniform(-step, step, size=n_score), 0.1, 3.0).astype(np.float32)
    elif mode == 3:
        ngt = float(np.clip(gt + rng.uniform(-step, step), 0.01, 2.5))
    else:
        nw = np.clip(w + rng.uniform(-step, step, size=n_models), 0.01, 2.0).astype(np.float32)
        nmt = np.clip(mt + rng.uniform(-step, step, size=n_score), 0.1, 3.0).astype(np.float32)
        ngt = float(np.clip(gt + rng.uniform(-step, step), 0.01, 2.5))
    return nw, nmt, ngt


def _local_search_from(
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    gt_data: Dict,
    all_pids: List[int],
    all_labels: List[str],
    start_w: np.ndarray,
    start_mt: np.ndarray,
    start_gt: float,
    max_evals: int,
    rng: np.random.RandomState,
    *,
    step: float = 0.06,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    """
    Stochastic first-improvement hill climb: ``max_evals`` objective evaluations
    (including the initial score at the start point).
    """
    if max_evals <= 0:
        return start_w.copy(), start_mt.copy(), float(start_gt), -1.0, 0
    n_models = len(matrices)
    n_score = len(start_mt)
    best_w, best_mt, best_gt = start_w.copy(), start_mt.copy(), float(start_gt)
    best_f1 = score_ensemble(
        matrices, is_score_model, best_w, best_mt, best_gt, gt_data, all_pids, all_labels,
    )
    used = 1
    it = 0
    while used < max_evals:
        if verbose and it % 200 == 0 and it > 0:
            print(f"    VNS local  {it}  best={best_f1:.4f}  evals={used}/{max_evals}", flush=True)
        w, mt, gt = _sample_params(
            n_models, n_score, rng,
            perturb_from=(best_w, best_mt, best_gt), step=step,
        )
        f1 = score_ensemble(matrices, is_score_model, w, mt, gt, gt_data, all_pids, all_labels)
        used += 1
        it += 1
        if f1 > best_f1:
            best_w, best_mt, best_gt, best_f1 = w.copy(), mt.copy(), gt, f1
    return best_w, best_mt, best_gt, best_f1, used


def run_vns_search(
    matrices: List[np.ndarray],
    is_score_model: List[bool],
    gt_data: Dict,
    pids: List[int],
    all_labels: List[str],
    n_iter: int,
    rng: np.random.RandomState,
    *,
    k_max: int = 5,
    init_random_frac: float = 0.22,
    local_eval_cap: int | None = None,
    shake_mode: str = "radius",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Variable Neighborhood Search on ensemble weights / thresholds / global gate.

    - **Shaking**: random perturbation of the incumbent; strength grows with ``k`` (``radius``)
      or alternates parameter groups (``partition``).
    - **Local search**: short hill-climb trajectory from the shaken point (same move operator
      as ``hill_climb``, smaller step).
    - **Neighborhood change**: if the local optimum does not beat the incumbent, ``k`` is
      increased up to ``k_max``; on improvement, ``k`` resets to 1.

    Total objective evaluations is approximately ``n_iter`` (initialization uses
    ``max(1, round(init_random_frac * n_iter))`` random samples; the remainder drives VNS).
    """
    k_max = max(1, int(k_max))
    if n_iter < 2:
        w, mt, gt = _sample_params(len(matrices), sum(is_score_model), rng)
        f1 = score_ensemble(matrices, is_score_model, w, mt, gt, gt_data, pids, all_labels)
        return w, mt, gt, f1

    n_random = max(1, int(round(float(init_random_frac) * float(n_iter))))
    n_random = min(n_random, n_iter - 1)
    best_w, best_mt, best_gt, best_f1 = random_search(
        matrices, is_score_model, gt_data, pids, all_labels, n_random, rng, verbose=verbose,
    )

    cur_w, cur_mt, cur_gt = best_w.copy(), best_mt.copy(), float(best_gt)
    cur_f1 = float(best_f1)

    budget_left = n_iter - n_random
    cap = local_eval_cap if local_eval_cap is not None else max(24, min(160, n_iter // 12))
    shake_fn = _shake_params_k if shake_mode == "radius" else _shake_params_partition
    if shake_mode not in ("radius", "partition"):
        raise ValueError("shake_mode must be 'radius' or 'partition'")

    outer = 0
    while budget_left > 0:
        if verbose and outer % 50 == 0:
            print(
                f"  VNS outer {outer}  incumbent={cur_f1:.4f}  global_best={best_f1:.4f}  "
                f"budget_left={budget_left}",
                flush=True,
            )
        k = 1
        improved_cycle = False
        while k <= k_max and budget_left > 0:
            sw, smt, sgt = shake_fn(cur_w, cur_mt, cur_gt, k, rng)

            loc_budget = min(cap, budget_left)
            lw, lmt, lgt, lf1, used = _local_search_from(
                matrices,
                is_score_model,
                gt_data,
                pids,
                all_labels,
                sw,
                smt,
                sgt,
                loc_budget,
                rng,
                verbose=False,
            )
            budget_left -= used

            if lf1 > best_f1:
                best_w, best_mt, best_gt, best_f1 = lw.copy(), lmt.copy(), lgt, lf1

            if lf1 > cur_f1:
                cur_w, cur_mt, cur_gt, cur_f1 = lw.copy(), lmt.copy(), lgt, lf1
                k = 1
                improved_cycle = True
            else:
                k += 1

        if not improved_cycle and budget_left > 0:
            w2, mt2, gt2 = _sample_params(len(matrices), len(cur_mt), rng, perturb_from=None)
            cur_w, cur_mt, cur_gt = w2, mt2, gt2
            cur_f1 = score_ensemble(
                matrices, is_score_model, cur_w, cur_mt, cur_gt, gt_data, pids, all_labels,
            )
            budget_left -= 1
            if cur_f1 > best_f1:
                best_w, best_mt, best_gt, best_f1 = cur_w.copy(), cur_mt.copy(), cur_gt, cur_f1
        outer += 1

    return best_w, best_mt, best_gt, best_f1


def _run_standalone_cli() -> None:
    import argparse
    from pathlib import Path

    from src.ensemble_metaheuristic.strategy_cli import (
        load_validation_bundle,
        prepend_repo_root_for_strategy_file,
    )

    prepend_repo_root_for_strategy_file(Path(__file__))

    parser = argparse.ArgumentParser(
        description="Variable Neighborhood Search on weighted ensemble parameters (this module only).",
    )
    parser.add_argument("--config", default="src/analysis/analysis.yaml", help="Analysis YAML.")
    parser.add_argument("--n-iter", type=int, default=10_000, help="Approx. objective evaluations per restart.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument("--restarts", type=int, default=2, help="Number of independent VNS restarts.")
    parser.add_argument("--k-max", type=int, default=5, help="Maximum shake neighborhood index.")
    parser.add_argument(
        "--shake-mode",
        choices=("radius", "partition"),
        default="radius",
        help="Shake neighborhood family.",
    )
    args = parser.parse_args()

    matrices, names, is_score_model, gt_data, all_pids, all_labels, _mc, _vp = load_validation_bundle(
        args.config,
    )

    print("Weighted VNS (this module only)")
    best_w = best_mt = best_gt = best_f1 = None
    best_r = 0
    for r in range(max(1, int(args.restarts))):
        rng = np.random.RandomState(int(args.seed) + r)
        print(f"  Restart {r + 1}/{max(1, int(args.restarts))}  seed={int(args.seed) + r}")
        w, mt, gt, f1 = run_vns_search(
            matrices,
            is_score_model,
            gt_data,
            all_pids,
            all_labels,
            int(args.n_iter),
            rng,
            k_max=int(args.k_max),
            shake_mode=str(args.shake_mode),
            verbose=(r == 0),
        )
        if best_f1 is None or f1 > best_f1:
            best_w, best_mt, best_gt, best_f1, best_r = w, mt, gt, f1, r
    assert best_w is not None and best_f1 is not None
    print(f"  Best restart: {best_r + 1}  micro-F1={best_f1:.4f}")
    mt_iter = iter(best_mt)
    for name, is_score, w in zip(names, is_score_model, best_w):
        mt = next(mt_iter) if is_score else 0.5
        suffix = f"  act_thr={mt:.4f}" if is_score else "  act_thr=0.5 (binary)"
        print(f"  {name:<25} weight={w:.4f}{suffix}")
    print(f"  global_threshold={best_gt:.4f}")


if __name__ == "__main__":
    _run_standalone_cli()
