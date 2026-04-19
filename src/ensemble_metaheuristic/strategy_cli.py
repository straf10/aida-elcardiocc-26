"""Load validation matrices and ground truth for standalone ``python -m ...strategies.*`` runs."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ENSEMBLE_MODELS = (
    "xlm_r_large",
    "mlc_greek_bert",
    "xlm_r_base",
    "information_retrieval",
    "ner_el",
)


def prepend_repo_root_for_strategy_file(strategy_file: Path) -> Path:
    """
    Ensure the repository root (parent of ``src``) is on ``sys.path`` so ``import src...`` works
    when executing ``python src/ensemble_metaheuristic/strategies/foo.py``.
    """
    root = strategy_file.resolve().parents[3]
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def load_validation_bundle(
    config_path: str,
) -> Tuple[
    List[np.ndarray],
    List[str],
    List[bool],
    Dict[Any, Any],
    List[int],
    List[str],
    Dict[str, Any],
    str,
]:
    try:
        from src.analysis.common import load_model_artifacts
        from src.evaluation.config_utils import get_cfg, load_config
        from src.evaluation.io_utils import load_ground_truth
        from src.ensemble_metaheuristic.matrices import build_score_matrix, load_thresholds_for_model
    except ImportError:
        from ..analysis.common import load_model_artifacts
        from ..evaluation.config_utils import get_cfg, load_config
        from ..evaluation.io_utils import load_ground_truth
        from .matrices import build_score_matrix, load_thresholds_for_model

    cfg = load_config(config_path)
    val_path = get_cfg(cfg, "data.val_path")
    gt_data = load_ground_truth(val_path)
    all_pids = list(gt_data.keys())
    model_cfgs = {m["name"]: m for m in get_cfg(cfg, "models", [])}

    artifacts_list = []
    for name in ENSEMBLE_MODELS:
        arts = load_model_artifacts(model_cfgs[name], all_pids)
        artifacts_list.append((name, arts))

    canonical_arts = next(a for n, a in artifacts_list if n == "xlm_r_large")
    all_labels = canonical_arts.label_names

    matrices: List[np.ndarray] = []
    names: List[str] = []
    is_score_model: List[bool] = []
    for name, arts in artifacts_list:
        thr = load_thresholds_for_model(model_cfgs[name], all_labels) if arts.scores is not None else None
        matrices.append(build_score_matrix(arts, all_pids, all_labels, thr))
        names.append(name)
        is_score_model.append(arts.scores is not None)

    return matrices, names, is_score_model, gt_data, all_pids, all_labels, model_cfgs, val_path


def build_per_model_preds(
    matrices: List[np.ndarray],
    names: List[str],
    is_score_model: List[bool],
    all_pids: List[int],
    all_labels: List[str],
) -> Dict[str, Dict[int, List[str]]]:
    return {
        name: {
            pid: [all_labels[j] for j in np.where(mat[i] >= (1.0 if is_score else 0.5))[0]]
            for i, pid in enumerate(all_pids)
        }
        for name, mat, is_score in zip(names, matrices, is_score_model)
    }
