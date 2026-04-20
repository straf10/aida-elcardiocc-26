"""Run predictors when prediction JSONL is missing; ensure output dirs from config."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from .config_utils import get_cfg


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
    cmd = [sys.executable, "-m", model_cfg["predict_module"]]
    if "predict_args" in model_cfg:
        cmd.extend(model_cfg["predict_args"])
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ensure_output_dir(cfg: Dict[str, Any]) -> Path:
    """Ensure and return the output directory from config (default ``outputs/evaluation``)."""
    out_dir = Path(get_cfg(cfg, "output.dir", "outputs/evaluation"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
