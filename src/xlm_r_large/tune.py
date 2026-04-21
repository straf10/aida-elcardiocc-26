from __future__ import annotations

import argparse
import copy
import gc
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
import torch
import wandb

from evaluation.config_utils import get_cfg, load_config
from split_data.dotenv_util import load_dotenv_if_present

from . import train
from .tuning_utils import (
    dump_optuna_importance,
    get_wandb_no_upload_settings,
    log_summary_to_wandb_parent,
    retain_top_k_checkpoints,
    write_best_json,
    write_results_csv,
)
def _set_nested(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = cfg
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _suggest(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name, spec in search_space.items():
        t = str(spec.get("type", "")).lower()
        if t == "categorical":
            out[name] = trial.suggest_categorical(name, spec["choices"])
        elif t == "uniform":
            out[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
        elif t == "loguniform":
            out[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=True,
            )
        else:
            raise ValueError(f"Unsupported search-space type '{t}' for '{name}'.")
    return out


def _apply_overrides(
    cfg: dict[str, Any],
    suggested: dict[str, Any],
) -> dict[str, Any]:
    trial_cfg = copy.deepcopy(cfg)
    key_map = {
        "learning_rate": "training.learning_rate",
        "weight_decay": "training.weight_decay",
        "warmup_ratio": "training.warmup_ratio",
        "asl_gamma_neg": "training.asl_gamma_neg",
        "freeze_layers": "training.freeze_layers",
        "classifier_dropout": "training.classifier_dropout",
        "loss": "training.loss",
        "truncation_side": "data.truncation_side",
    }
    for key, value in suggested.items():
        if key in key_map:
            _set_nested(trial_cfg, key_map[key], value)
    return trial_cfg


def _build_sampler(tune_cfg: dict[str, Any]) -> optuna.samplers.BaseSampler:
    sampler = str(get_cfg(tune_cfg, "tuning.sampler", "tpe")).lower()
    seed = int(get_cfg(tune_cfg, "tuning.seed", 42))
    if sampler == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if sampler == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if sampler == "grid":
        search_space = get_cfg(tune_cfg, "tuning.search_space", {})
        grid: dict[str, list[Any]] = {}
        for name, spec in search_space.items():
            if str(spec.get("type", "")).lower() != "categorical":
                raise ValueError("Grid sampler requires categorical choices for all params.")
            grid[name] = list(spec["choices"])
        return optuna.samplers.GridSampler(grid)
    raise ValueError(f"Unsupported sampler '{sampler}'.")


def _build_pruner(tune_cfg: dict[str, Any]) -> optuna.pruners.BasePruner:
    p_type = str(get_cfg(tune_cfg, "tuning.pruner.type", "median")).lower()
    if p_type == "none":
        return optuna.pruners.NopPruner()
    if p_type == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(get_cfg(tune_cfg, "tuning.pruner.n_startup_trials", 5)),
            n_warmup_steps=int(get_cfg(tune_cfg, "tuning.pruner.n_warmup_steps", 6)),
        )
    if p_type == "hyperband":
        return optuna.pruners.HyperbandPruner()
    raise ValueError(f"Unsupported pruner '{p_type}'.")


def _make_trial_paths(base_dir: str, trial_number: int) -> dict[str, str]:
    trial_dir = os.path.join(base_dir, f"trial_{trial_number:03d}")
    return {
        "trial_dir": trial_dir,
        "checkpoint_dir": os.path.join(trial_dir, "checkpoints"),
        "scores_path": os.path.join(trial_dir, "val_scores.npy"),
        "patient_ids_path": os.path.join(trial_dir, "val_patient_ids.json"),
        "label_names_path": os.path.join(trial_dir, "label_names.json"),
        "thresholds_path": os.path.join(trial_dir, "thresholds.json"),
        "log_dir": os.path.join(trial_dir, "logs"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna tuning for XLM-R large")
    parser.add_argument("--config", required=True, help="Path to tune YAML config")
    parser.add_argument("--n-trials", type=int, help="Override number of trials")
    parser.add_argument("--study-name", help="Reuse an existing study name")
    parser.add_argument("--storage", help="Optuna storage URL")
    parser.add_argument("--device", help="Explicit device for train.run")
    args = parser.parse_args()

    load_dotenv_if_present()
    tune_cfg = load_config(args.config)
    base_cfg_path = get_cfg(tune_cfg, "base_config")
    if not base_cfg_path:
        raise ValueError("tune.yaml must define 'base_config'.")
    base_cfg = load_config(base_cfg_path)

    run_id = args.study_name or get_cfg(tune_cfg, "tuning.study_name")
    if not run_id:
        run_id = f"xlm-r-large-tune-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    tuning_root = get_cfg(
        tune_cfg,
        "output.tuning_root",
        "outputs/experiments/xlm_r_large/tuning",
    )
    run_dir = os.path.join(tuning_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    sampler = _build_sampler(tune_cfg)
    pruner = _build_pruner(tune_cfg)
    direction = str(get_cfg(tune_cfg, "tuning.direction", "maximize"))
    n_trials = int(args.n_trials or get_cfg(tune_cfg, "tuning.n_trials", 25))
    storage = args.storage or f"sqlite:///{Path(run_dir) / 'study.db'}"

    study = optuna.create_study(
        study_name=run_id,
        storage=storage,
        load_if_exists=True,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )

    wb_enabled = bool(get_cfg(base_cfg, "wandb.enabled", False)) and bool(
        get_cfg(tune_cfg, "wandb.enabled", True)
    )
    parent_run = None
    if wb_enabled:
        parent_run = wandb.init(
            project=get_cfg(base_cfg, "wandb.project", "elcardiocc-2026"),
            entity=get_cfg(base_cfg, "wandb.entity"),
            name=f"{run_id}-parent",
            group=run_id,
            job_type=get_cfg(tune_cfg, "wandb.parent_job_type", "study"),
            config={"run_id": run_id, "tuning_config": tune_cfg},
            save_code=False,
            settings=get_wandb_no_upload_settings(),
        )

    def objective(trial: optuna.Trial) -> float:
        search_space = get_cfg(tune_cfg, "tuning.search_space", {})
        suggested = _suggest(trial, search_space)
        trial_cfg = _apply_overrides(base_cfg, suggested)

        val_path = str(get_cfg(trial_cfg, "data.val_path", ""))
        if "test.jsonl" in val_path.replace("\\", "/"):
            raise ValueError("Validation path cannot point to test split.")

        paths = _make_trial_paths(run_dir, trial.number)
        _set_nested(trial_cfg, "output.checkpoint_dir", paths["checkpoint_dir"])
        _set_nested(trial_cfg, "output.scores_path", paths["scores_path"])
        _set_nested(trial_cfg, "output.patient_ids_path", paths["patient_ids_path"])
        _set_nested(trial_cfg, "output.label_names_path", paths["label_names_path"])
        _set_nested(trial_cfg, "output.thresholds_path", paths["thresholds_path"])
        _set_nested(trial_cfg, "output.log_dir", paths["log_dir"])

        trial.set_user_attr("trial_dir", paths["trial_dir"])
        trial.set_user_attr("checkpoint_dir", paths["checkpoint_dir"])
        trial.set_user_attr("thresholds_path", paths["thresholds_path"])

        trial_wandb_kwargs = None
        if wb_enabled:
            trial_model = get_cfg(trial_cfg, "model.name", "xlm-roberta-large")
            trial_wandb_kwargs = {
                "project": get_cfg(base_cfg, "wandb.project", "elcardiocc-2026"),
                "entity": get_cfg(base_cfg, "wandb.entity"),
                "name": train.make_wandb_run_name(
                    trial_model,
                    None,
                    trial_number=trial.number,
                ),
                "group": run_id,
                "job_type": get_cfg(tune_cfg, "wandb.trial_job_type", "trial"),
                "reinit": True,
                "save_code": False,
                "settings": get_wandb_no_upload_settings(),
            }

        result = None
        try:
            result = train.run(
                config=trial_cfg,
                wandb_init_kwargs=trial_wandb_kwargs,
                optuna_trial=trial,
                device_override=args.device,
            )
            trial.set_user_attr("best_epoch", result.get("best_epoch"))
            trial.set_user_attr("best_val_micro_f1", result.get("best_val_micro_f1"))
            return float(result["best_val_micro_f1"])
        finally:
            del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    study.optimize(
        objective,
        n_trials=n_trials,
        gc_after_trial=True,
        catch=(RuntimeError,),
        show_progress_bar=True,
    )

    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete_trials:
        raise RuntimeError("No completed trials. Check logs for repeated failures.")

    results_csv = os.path.join(run_dir, get_cfg(tune_cfg, "output.results_csv", "results.csv"))
    best_json = os.path.join(run_dir, get_cfg(tune_cfg, "output.best_json", "best.json"))
    importance_png = os.path.join(run_dir, "optuna_importance.png")

    write_results_csv(study, results_csv)
    write_best_json(study, best_json)
    keep_k = int(get_cfg(tune_cfg, "tuning.checkpoints.k", 3))
    keep_ids = retain_top_k_checkpoints(study, keep_k)
    dump_optuna_importance(study, importance_png)

    print(f"Study complete. Best trial: {study.best_trial.number}")
    print(f"Best micro-F1: {study.best_value:.4f}")
    print(f"Retained checkpoint trials: {keep_ids}")
    print(f"Results CSV: {results_csv}")
    print(f"Best JSON: {best_json}")

    if parent_run is not None:
        log_summary_to_wandb_parent(
            parent_run=parent_run,
            study=study,
            importance_path=importance_png,
        )
        parent_run.finish()


if __name__ == "__main__":
    main()
