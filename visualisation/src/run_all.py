from __future__ import annotations

import argparse
from pathlib import Path

from visualisation.src.config import (
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_CLUSTER_OUT,
    DEFAULT_CONFIG,
    DEFAULT_ENSEMBLE_OUT,
    DEFAULT_OUT_DIR,
    DEFAULT_REPORTS_DIR,
)
from visualisation.src.cross_model_data import load_cross_model_bundle, top_pairs_subset
from visualisation.src.important_codes import collect_important_pairs_and_codes
from visualisation.src.plots_code_rescue import plot_code_rescue
from visualisation.src.plots_cross_confusion import run_cross_confusion_plots
from visualisation.src.plots_ensemble_diagnostics import run_ensemble_diagnostics
from visualisation.src.ensemble_artifacts import load_all_model_artifacts
from visualisation.src.run_cluster import run_all_cluster_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model confusion and rescue visualisations.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--top-pairs", type=int, default=20, help="Top N (P,T) pairs by pooled count for heatmaps.")
    parser.add_argument("--wrong-pairs-json-top", type=int, default=30, help="Top entries from label_analysis wrong_pairs_counter.")
    parser.add_argument(
        "--ensemble-out",
        type=Path,
        default=DEFAULT_ENSEMBLE_OUT,
        help="Output directory for ensemble diagnostic plots (calibration, UpSet, ...).",
    )
    parser.add_argument("--top-classes", type=int, default=10, help="Top ICD codes by GT support for TP/FP histogram grid.")
    parser.add_argument("--skip-upset", action="store_true", help="Skip FN UpSet plot (e.g. if upsetplot unavailable).")
    args = parser.parse_args()

    bundle = load_cross_model_bundle(args.config)
    important_pairs, _ = collect_important_pairs_and_codes(
        bundle.cfg,
        bundle.gt_data,
        bundle.label_names,
        args.reports_dir,
        args.analysis_dir,
        bundle.model_names,
        wrong_pairs_top_k=args.wrong_pairs_json_top,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chosen = run_cross_confusion_plots(bundle, important_pairs, args.out_dir, args.top_pairs)
    print(f"Wrote cross-model plots ({len(chosen)} pairs) to {args.out_dir.resolve()}")

    plot_code_rescue(
        bundle,
        args.out_dir / "code_fn_rescuers.png",
        args.out_dir / "code_fn_rescuers.csv",
    )
    print(f"Wrote code rescue plot and CSV to {args.out_dir.resolve()}")

    run_all_cluster_plots(bundle.cfg, bundle, DEFAULT_CLUSTER_OUT)

    arts = load_all_model_artifacts(bundle)
    args.ensemble_out.mkdir(parents=True, exist_ok=True)
    run_ensemble_diagnostics(
        bundle,
        arts,
        args.ensemble_out,
        top_classes=args.top_classes,
        skip_upset=args.skip_upset,
    )
    print(f"Wrote ensemble diagnostic plots to {args.ensemble_out.resolve()}")


if __name__ == "__main__":
    main()
