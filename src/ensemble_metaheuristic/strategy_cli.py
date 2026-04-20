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
    Put the ``src`` directory on ``sys.path`` so sibling packages (``evaluation``, …) resolve
    when executing ``python src/ensemble_metaheuristic/strategies/foo.py`` without ``pip install -e .``.
    """
    root = strategy_file.resolve().parents[3]
    src_dir = str(root / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
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
    from evaluation.config_utils import get_cfg, load_config
    from evaluation.io_utils import load_ground_truth
    from evaluation.model_artifacts import load_model_artifacts
    from ensemble_metaheuristic.matrices import build_score_matrix, load_thresholds_for_model

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


def load_train_validation_matrices(
    config_path: str,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[str],
    List[bool],
    Dict[Any, Any],
    Dict[Any, Any],
    List[int],
    List[int],
    List[str],
    Dict[str, Any],
    str,
    str,
]:
    """
    Load validation + training score matrices (same models / label space as ``load_validation_bundle``).

    Returns
    -------
    val_matrices, train_matrices, names, is_score_model, val_gt, train_gt, val_pids, train_pids,
    all_labels, model_cfgs, val_path, train_path
    """
    from evaluation.config_utils import get_cfg, load_config
    from evaluation.io_utils import load_ground_truth
    from evaluation.model_artifacts import load_model_artifacts
    from ensemble_metaheuristic.matrices import build_score_matrix, load_thresholds_for_model

    cfg = load_config(config_path)
    val_path = get_cfg(cfg, "data.val_path")
    train_path = get_cfg(cfg, "data.train_path", "data/processed/training_set.jsonl")
    val_gt = load_ground_truth(val_path)
    train_gt = load_ground_truth(train_path)
    val_pids = list(val_gt.keys())
    train_pids = list(train_gt.keys())
    model_cfgs = {m["name"]: m for m in get_cfg(cfg, "models", [])}

    artifacts_val = []
    artifacts_train = []
    for name in ENSEMBLE_MODELS:
        artifacts_val.append((name, load_model_artifacts(model_cfgs[name], val_pids)))
        artifacts_train.append((name, load_model_artifacts(model_cfgs[name], train_pids)))

    canonical_arts = next(a for n, a in artifacts_val if n == "xlm_r_large")
    all_labels = canonical_arts.label_names

    val_matrices: List[np.ndarray] = []
    train_matrices: List[np.ndarray] = []
    names: List[str] = []
    is_score_model: List[bool] = []
    for (name, arts_v), (name_t, arts_tr) in zip(artifacts_val, artifacts_train):
        assert name == name_t
        thr = load_thresholds_for_model(model_cfgs[name], all_labels) if arts_v.scores is not None else None
        val_matrices.append(build_score_matrix(arts_v, val_pids, all_labels, thr))
        train_matrices.append(build_score_matrix(arts_tr, train_pids, all_labels, thr))
        names.append(name)
        is_score_model.append(arts_v.scores is not None)

    return (
        val_matrices,
        train_matrices,
        names,
        is_score_model,
        val_gt,
        train_gt,
        val_pids,
        train_pids,
        all_labels,
        model_cfgs,
        val_path,
        train_path,
    )


def load_train_matrices(
    config_path: str,
    model_cfgs: Dict[str, Any],
    all_labels: List[str],
) -> Tuple[Dict[Any, Any], List[int], List[np.ndarray], str]:
    """Load train ground truth and one score matrix per ensemble model (same label order as validation)."""
    from evaluation.config_utils import get_cfg, load_config
    from evaluation.io_utils import load_ground_truth
    from evaluation.model_artifacts import load_model_artifacts
    from ensemble_metaheuristic.matrices import build_score_matrix, load_thresholds_for_model

    cfg = load_config(config_path)
    train_path = str(get_cfg(cfg, "data.train_path", "data/processed/training_set.jsonl"))
    train_gt = load_ground_truth(train_path)
    train_pids = list(train_gt.keys())
    train_matrices: List[np.ndarray] = []
    for name in ENSEMBLE_MODELS:
        arts = load_model_artifacts(model_cfgs[name], train_pids)
        thr = load_thresholds_for_model(model_cfgs[name], all_labels) if arts.scores is not None else None
        train_matrices.append(build_score_matrix(arts, train_pids, all_labels, thr))
    return train_gt, train_pids, train_matrices, train_path


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
