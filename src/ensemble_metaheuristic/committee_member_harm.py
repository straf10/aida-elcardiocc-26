"""
Detect committee members that **look fine on val** but **hurt test** under weighted fusion.

For each model ``m`` in the pool, we compare:

1. **Full pool**: tune weights on **val** → report val / test micro-F1 (test = frozen val weights).
2. **Leave ``m`` out**: tune again on val on ``pool \\ {m}`` → same metrics.

If ``test(pool \\ {m}) - test(full)`` exceeds a small threshold, ``m`` is flagged: the committee **without**
``m`` achieves better **test** F1 after a **fresh** val-only retune — a sign ``m`` may be **dragging**
test performance even when val scores stay high (same pitfall as adding ``xlm_r_large`` in your runs).

This is **exploratory** (many implicit comparisons if you tweak thresholds); use a held-out split or
one-shot test evaluation for final claims.

Examples::

    PYTHONPATH=src python -m ensemble_metaheuristic.committee_member_harm
    PYTHONPATH=src python -m ensemble_metaheuristic.committee_member_harm --min-test-gain 0.01 --csv outputs/committee_harm.csv

Uses the same loaders as ``weighted_subset_sweep`` (including ``use_in_ensemble: false``).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_src = Path(__file__).resolve().parents[1]
if _src.name == "src" and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ensemble_metaheuristic.strategy_loaders import load_validation_bundle
from ensemble_metaheuristic.weighted_subset_sweep import (
    _filter_pool,
    _parse_name_list,
    _slice_matrices,
    _subset_indices,
    _test_micro_f1_frozen_weights,
    _tune_val_best_params,
)


def _run_one_committee(
    matrices: List[np.ndarray],
    names: Sequence[str],
    is_score_model: List[bool],
    subset: Sequence[str],
    gt_data: Dict,
    all_pids: List[int],
    all_labels: List[str],
    model_cfgs: Dict[str, Any],
    test_gt: Dict,
    test_pids: List[int],
    *,
    n_iter: int,
    base_seed: int,
    n_restarts: int,
    weighted_search: str,
    verbose: bool,
) -> Tuple[float, float, Tuple[str, ...]]:
    idxs = _subset_indices(names, subset)
    mats, ism = _slice_matrices(matrices, is_score_model, idxs)
    sub_names = tuple(names[i] for i in idxs)
    c, v, b, w, mt, gt = _tune_val_best_params(
        mats,
        ism,
        gt_data,
        all_pids,
        all_labels,
        n_iter=n_iter,
        base_seed=base_seed,
        n_restarts=n_restarts,
        weighted_search=weighted_search,
        verbose=verbose,
    )
    tf = _test_micro_f1_frozen_weights(
        model_cfgs,
        sub_names,
        ism,
        all_labels,
        test_gt,
        test_pids,
        w,
        mt,
        gt,
    )
    return float(b), float(tf), sub_names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-one-out weighted ensemble: flag models whose removal improves test F1 "
        "(val-tuned weights, separate retune per row).",
    )
    parser.add_argument("--config", default="src/evaluation/config.yaml", help="Evaluation YAML.")
    parser.add_argument("--only", default=None, help="Comma-separated names; restrict pool.")
    parser.add_argument("--exclude", default=None, help="Comma-separated names removed from pool.")
    parser.add_argument(
        "--min-test-gain",
        type=float,
        default=0.005,
        help="Flag ``left_out`` when test(without m) - test(full) exceeds this (default 0.005).",
    )
    parser.add_argument(
        "--min-val-drop",
        type=float,
        default=None,
        help="Optional: also require val(full) - val(without m) >= this to flag 'val looked better with m'.",
    )
    parser.add_argument(
        "--weighted-search",
        choices=("classic", "vns", "both"),
        default="both",
        help="Weighted optimizers (same as ensemble_metaheuristic).",
    )
    parser.add_argument("--n-iter", type=int, default=4000, help="Search budget per restart.")
    parser.add_argument("--restarts", type=int, default=2, help="Restarts per optimizer.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed (full pool).")
    parser.add_argument("--quiet", action="store_true", help="Less optimizer logging.")
    parser.add_argument("--csv", default=None, help="Write one row per left-out model.")
    args = parser.parse_args()

    from evaluation.config_utils import get_cfg, load_config
    from evaluation.io_utils import load_ground_truth

    cfg = load_config(args.config)
    tp = str(get_cfg(cfg, "data.test_path", "") or "")
    if not tp or not Path(tp).is_file():
        raise SystemExit(f"Need data.test_path pointing to an existing file; got {tp!r}")

    matrices, names, is_score_model, gt_data, all_pids, all_labels, model_cfgs, _vp = load_validation_bundle(
        args.config,
    )
    test_gt = load_ground_truth(tp)
    test_pids = list(test_gt.keys())

    pool = _filter_pool(names, only=_parse_name_list(args.only), exclude=_parse_name_list(args.exclude))
    if len(pool) < 2:
        raise SystemExit("Need at least two models in the pool for leave-one-out harm detection.")

    wsearch = args.weighted_search
    verbose = not args.quiet

    print(f"Committee ({len(names)} loaded): {', '.join(names)}")
    print(f"Pool ({len(pool)}): {', '.join(pool)}")
    print(f"weighted-search={wsearch} n_iter={args.n_iter} restarts={args.restarts} seed={args.seed}\n")

    val_full, test_full, _ = _run_one_committee(
        matrices,
        names,
        is_score_model,
        pool,
        gt_data,
        all_pids,
        all_labels,
        model_cfgs,
        test_gt,
        test_pids,
        n_iter=int(args.n_iter),
        base_seed=int(args.seed),
        n_restarts=int(args.restarts),
        weighted_search=wsearch,
        verbose=verbose,
    )
    print(f"Full pool  val micro-F1 (tuned): {val_full:.4f}")
    print(f"Full pool  test micro-F1 (frozen val weights): {test_full:.4f}\n")

    rows: List[Dict[str, object]] = []
    flagged: List[str] = []

    for j, left_out in enumerate(pool):
        sub = tuple(m for m in pool if m != left_out)
        seed_lo = int(args.seed) + 101 * (j + 1)
        val_lo, test_lo, _ = _run_one_committee(
            matrices,
            names,
            is_score_model,
            sub,
            gt_data,
            all_pids,
            all_labels,
            model_cfgs,
            test_gt,
            test_pids,
            n_iter=int(args.n_iter),
            base_seed=seed_lo,
            n_restarts=int(args.restarts),
            weighted_search=wsearch,
            verbose=verbose,
        )
        gain_test = test_lo - test_full
        gain_val = val_lo - val_full
        flag = gain_test > float(args.min_test_gain)
        if args.min_val_drop is not None:
            flag = flag and (val_full - val_lo >= float(args.min_val_drop))

        rows.append(
            {
                "left_out": left_out,
                "val_without": round(val_lo, 6),
                "test_without": round(test_lo, 6),
                "gain_val": round(gain_val, 6),
                "gain_test": round(gain_test, 6),
                "flag_test_drag": flag,
            },
        )
        if flag:
            flagged.append(left_out)

    rows.sort(key=lambda r: float(r["gain_test"]), reverse=True)

    print("Leave-one-out (columns = metrics for committee **without** ``left_out``):")
    print(
        f"{'left_out':<28} {'val_wo':>8} {'test_wo':>8} {'Δval':>8} {'Δtest':>8}  "
        f"{'flag':>5}  (Δtest = test_wo - full {test_full:.4f})",
    )
    for r in rows:
        fl = "YES" if r["flag_test_drag"] else ""
        print(
            f"{str(r['left_out']):<28} {float(r['val_without']):>8.4f} {float(r['test_without']):>8.4f} "
            f"{float(r['gain_val']):>+8.4f} {float(r['gain_test']):>+8.4f}  {fl:>5}",
        )

    print("\n--- Summary ---")
    if flagged:
        print(
            "Models whose **removal** improves **test** F1 (after separate val retune), "
            f"gain_test > {args.min_test_gain}:",
        )
        for n in flagged:
            print(f"  - {n}")
    else:
        print(f"No model exceeded --min-test-gain={args.min_test_gain} on test improvement when left out.")

    if args.csv:
        outp = Path(args.csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "left_out",
            "val_full",
            "test_full",
            "val_without",
            "test_without",
            "gain_val",
            "gain_test",
            "flag_test_drag",
        ]
        with open(outp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in sorted(rows, key=lambda x: str(x["left_out"])):
                w.writerow(
                    {
                        "left_out": r["left_out"],
                        "val_full": round(val_full, 6),
                        "test_full": round(test_full, 6),
                        "val_without": r["val_without"],
                        "test_without": r["test_without"],
                        "gain_val": r["gain_val"],
                        "gain_test": r["gain_test"],
                        "flag_test_drag": bool(r["flag_test_drag"]),
                    },
                )
        print(f"\nWrote CSV: {outp.resolve()}")


if __name__ == "__main__":
    main()
