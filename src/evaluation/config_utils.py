from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping at the root.")
    return data


def get_cfg(config: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def ensure_evaluation_output_dir(cfg: Dict[str, Any]) -> Path:
    """Create ``output.dir`` from config (default ``outputs/evaluation``)."""
    out_dir = Path(get_cfg(cfg, "output.dir", "outputs/evaluation"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
