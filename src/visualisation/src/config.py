from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

# ``src/visualisation/`` package root (parent of this nested ``src/`` dir)
_VIZ_PKG_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = _VIZ_PKG_ROOT.parent.parent

DEFAULT_CONFIG = REPO_ROOT / "src" / "analysis" / "analysis.yaml"
DEFAULT_REPORTS_DIR = REPO_ROOT / "outputs" / "analysis" / "reports"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "plots"
DEFAULT_CLUSTER_OUT = REPO_ROOT / "outputs" / "plots" / "cluster"
DEFAULT_ENSEMBLE_OUT = REPO_ROOT / "outputs" / "plots" / "ensemble"

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
