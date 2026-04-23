"""Load validation matrices and ground truth for standalone ``python -m ...strategies.*`` runs."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ENSEMBLE_MODELS = (
    "xlm_r_large",
    "mlc_greek_bert",
    "xlm_r_base",
    "information_retrieval",
    "dictionary_baseline",
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


def gather_ensemble_artifacts(
    model_cfgs: Dict[str, Any],
    pids: List[int],
    split: str,
) -> List[Tuple[str, Any]]:
    """
    Load ``ENSEMBLE_MODELS`` in order. Skip a model if it is missing from ``model_cfgs`` or if
    predictions for ``split`` are not on disk (``FileNotFoundError``).
    """
    from evaluation.model_artifacts import load_model_artifacts

    loaded: List[Tuple[str, Any]] = []
    for name in ENSEMBLE_MODELS:
        if name not in model_cfgs:
            print(
                f"[ensemble] WARNING: model {name!r} is not listed under ``models`` in the evaluation config — skipping.",
                flush=True,
            )
            continue
        try:
            arts = load_model_artifacts(model_cfgs[name], pids, predictions_split=split)
            loaded.append((name, arts))
        except FileNotFoundError as exc:
            print(f"[ensemble] WARNING: skipping {name!r} — {exc}", flush=True)
    return loaded


def canonical_ensemble_label_arts(
    artifacts_list: List[Tuple[str, Any]],
    prefer: str = "xlm_r_large",
) -> Any:
    """Label order for matrices: prefer ``xlm_r_large`` if present, else the first loaded model."""
    if not artifacts_list:
        raise ValueError("canonical_ensemble_label_arts: empty artifacts_list")
    for n, a in artifacts_list:
        if n == prefer:
            return a
    return artifacts_list[0][1]


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
    from ensemble_metaheuristic.matrices import build_score_matrix, load_thresholds_for_model

    cfg = load_config(config_path)
    val_path = get_cfg(cfg, "data.val_path")
    gt_data = load_ground_truth(val_path)
    all_pids = list(gt_data.keys())
    model_cfgs = {m["name"]: m for m in get_cfg(cfg, "models", [])}

    artifacts_list = gather_ensemble_artifacts(model_cfgs, all_pids, "val")
    if not artifacts_list:
        raise FileNotFoundError(
            "No ensemble models had validation predictions on disk. "
            "Generate them with: PYTHONPATH=src python -m evaluation.run_predictions"
        )

    all_labels = canonical_ensemble_label_arts(artifacts_list).label_names

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
    train_path = get_cfg(cfg, "data.train_path", "data/processed/train.jsonl")
    val_gt = load_ground_truth(val_path)
    train_gt = load_ground_truth(train_path)
    val_pids = list(val_gt.keys())
    train_pids = list(train_gt.keys())
    model_cfgs = {m["name"]: m for m in get_cfg(cfg, "models", [])}

    paired: List[Tuple[str, Any, Any]] = []
    for name in ENSEMBLE_MODELS:
        if name not in model_cfgs:
            print(
                f"[ensemble] WARNING: model {name!r} is not listed under ``models`` in the evaluation config — skipping.",
                flush=True,
            )
            continue
        try:
            arts_v = load_model_artifacts(model_cfgs[name], val_pids, predictions_split="val")
        except FileNotFoundError as exc:
            print(f"[ensemble] WARNING: skipping {name!r} (val) — {exc}", flush=True)
            continue
        try:
            arts_tr = load_model_artifacts(model_cfgs[name], train_pids, predictions_split="train")
        except FileNotFoundError as exc:
            print(
                f"[ensemble] WARNING: skipping {name!r} (train predictions missing after val ok) — {exc}",
                flush=True,
            )
            continue
        paired.append((name, arts_v, arts_tr))

    if not paired:
        raise FileNotFoundError(
            "No ensemble models had both validation and train prediction files. "
            "Generate them with: PYTHONPATH=src python -m evaluation.run_predictions"
        )

    all_labels = canonical_ensemble_label_arts([(n, av) for n, av, _at in paired]).label_names

    val_matrices: List[np.ndarray] = []
    train_matrices: List[np.ndarray] = []
    names: List[str] = []
    is_score_model: List[bool] = []
    for name, arts_v, arts_tr in paired:
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
    model_names: Optional[Sequence[str]] = None,
) -> Tuple[Dict[Any, Any], List[int], List[np.ndarray], str]:
    """
    Load train ground truth and one score matrix per ensemble model (same label order as validation).

    ``model_names``: subset to load (same order as validation matrices). Default: all ``ENSEMBLE_MODELS``.
    """
    from evaluation.config_utils import get_cfg, load_config
    from evaluation.io_utils import load_ground_truth
    from evaluation.model_artifacts import load_model_artifacts
    from ensemble_metaheuristic.matrices import build_score_matrix, load_thresholds_for_model

    cfg = load_config(config_path)
    train_path = str(get_cfg(cfg, "data.train_path", "data/processed/train.jsonl"))
    train_gt = load_ground_truth(train_path)
    train_pids = list(train_gt.keys())
    names_to_load = list(model_names) if model_names is not None else list(ENSEMBLE_MODELS)
    train_matrices: List[np.ndarray] = []
    for name in names_to_load:
        arts = load_model_artifacts(model_cfgs[name], train_pids, predictions_split="train")
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
