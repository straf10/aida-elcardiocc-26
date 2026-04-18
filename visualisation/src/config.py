from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

# ``visualisation/`` package root (parent of this ``src/`` dir)
_VIZ_PKG_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = _VIZ_PKG_ROOT.parent

DEFAULT_CONFIG = REPO_ROOT / "src" / "analysis" / "analysis.yaml"
DEFAULT_REPORTS_DIR = REPO_ROOT / "outputs" / "analysis" / "reports"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"
DEFAULT_OUT_DIR = _VIZ_PKG_ROOT / "out"
DEFAULT_CLUSTER_OUT = _VIZ_PKG_ROOT / "cluster_out"
DEFAULT_ENSEMBLE_OUT = _VIZ_PKG_ROOT / "out" / "ensemble"

EXCLUDED_MODELS = frozenset({"xlm_r_base"})

# Short names for plot annotations
MODEL_ABBREV: Dict[str, str] = {
    "mlc_greek_bert": "GB",
    "xlm_r_large": "XL",
    "information_retrieval": "IR",
    "ner_el": "NE",
}


def included_model_configs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [m for m in cfg.get("models", []) if m.get("name") not in EXCLUDED_MODELS]
