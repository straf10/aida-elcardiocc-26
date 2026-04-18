from __future__ import annotations

import argparse
from pathlib import Path
from visualisation.src.cluster_context import cluster_context_paths, load_cluster_assignments
from visualisation.src.config import DEFAULT_CLUSTER_OUT, DEFAULT_CONFIG
from visualisation.src.cross_model_data import CrossModelBundle, load_cross_model_bundle
from visualisation.src.plots_cluster_errors import run_cluster_error_plots
from visualisation.src.plots_cluster_metrics import run_cluster_metrics_plots
from visualisation.src.plots_cluster_tail import run_cluster_tail_plot
from visualisation.src.plots_cluster_umap import run_cluster_umap_grid


def run_all_cluster_plots(cfg: dict, bundle: CrossModelBundle, out_dir: Path) -> bool:
    """
    Generate all cluster-level figures into out_dir.
    Returns False if cluster assignments are missing.
    """
    assignments = load_cluster_assignments(cfg)
    if not assignments:
        print("[cluster] No cluster_assignments.json — skipping cluster plots.")
        return False

    a_path, _, _ = cluster_context_paths(cfg)
    print(f"[cluster] Using assignments from {a_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cluster_metrics_plots(bundle, assignments, cfg, out_dir)
    run_cluster_error_plots(bundle, assignments, cfg, out_dir)
    run_cluster_tail_plot(bundle, assignments, cfg, out_dir)
    run_cluster_umap_grid(bundle, cfg, out_dir)

    print(f"[cluster] Wrote cluster visualisations to {out_dir.resolve()}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster-level model visualisations.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cluster-out", type=Path, default=DEFAULT_CLUSTER_OUT)
    args = parser.parse_args()

    bundle = load_cross_model_bundle(args.config)
    run_all_cluster_plots(bundle.cfg, bundle, args.cluster_out)


if __name__ == "__main__":
    main()
