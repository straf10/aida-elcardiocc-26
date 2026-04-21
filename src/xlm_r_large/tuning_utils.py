from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
import optuna
import wandb


def get_wandb_no_upload_settings() -> wandb.Settings:
    """Disable W&B code/job uploads while keeping metric logging."""
    try:
        return wandb.Settings(
            save_code=False,
            disable_job_creation=True,
        )
    except TypeError:
        return wandb.Settings(save_code=False)


def write_results_csv(study: optuna.Study, path: str) -> None:
    """Persist all trial results to a CSV table."""
    rows = []
    all_keys: set[str] = set()
    for trial in study.trials:
        base = {
            "trial": trial.number,
            "state": str(trial.state),
            "value": trial.value,
            "duration_sec": (
                None
                if trial.duration is None
                else round(float(trial.duration.total_seconds()), 3)
            ),
        }
        for key, value in trial.params.items():
            base[f"param.{key}"] = value
        for key, value in trial.user_attrs.items():
            base[f"user.{key}"] = value
        rows.append(base)
        all_keys.update(base.keys())

    fieldnames = sorted(all_keys)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_best_json(study: optuna.Study, path: str) -> None:
    """Write best-trial metadata for downstream inference use."""
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete_trials:
        raise RuntimeError("No completed trials available; cannot write best.json.")
    best = study.best_trial
    payload = {
        "trial_number": best.number,
        "best_val_micro_f1": best.value,
        "params": best.params,
        "checkpoint_dir": best.user_attrs.get("checkpoint_dir"),
        "thresholds_path": best.user_attrs.get("thresholds_path"),
        "run_dir": best.user_attrs.get("trial_dir"),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def retain_top_k_checkpoints(study: optuna.Study, k: int) -> list[int]:
    """Keep only top-k completed-trial checkpoint dirs by value."""
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    complete_trials.sort(key=lambda trial: float(trial.value), reverse=True)
    keep_ids = {trial.number for trial in complete_trials[: max(0, int(k))]}

    for trial in study.trials:
        trial_dir = trial.user_attrs.get("trial_dir")
        checkpoint_dir = trial.user_attrs.get("checkpoint_dir")
        if not trial_dir or not checkpoint_dir:
            continue
        checkpoint_path = Path(checkpoint_dir)
        if trial.number in keep_ids:
            continue
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path, ignore_errors=True)
        # Also remove tokenizer files that are copied beside checkpoints by HF.
        trial_path = Path(trial_dir)
        for filename in [
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "sentencepiece.bpe.model",
        ]:
            maybe = trial_path / "checkpoints" / filename
            if maybe.exists():
                maybe.unlink(missing_ok=True)
    return sorted(keep_ids)


def dump_optuna_importance(study: optuna.Study, out_path: str) -> None:
    """Dump parameter-importance chart as PNG if available."""
    try:
        from optuna.visualization.matplotlib import plot_param_importances
    except Exception:
        return
    fig = plot_param_importances(study).figure
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")


def log_summary_to_wandb_parent(
    parent_run: wandb.sdk.wandb_run.Run,
    study: optuna.Study,
    importance_path: str,
) -> None:
    """Log results table and summary metrics to parent W&B run."""
    table = wandb.Table(columns=["trial", "state", "value", "params", "user_attrs"])
    for trial in study.trials:
        table.add_data(
            trial.number,
            str(trial.state),
            trial.value,
            json.dumps(trial.params, ensure_ascii=False),
            json.dumps(trial.user_attrs, ensure_ascii=False),
        )
    parent_run.log({"study/results_table": table})
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if complete_trials:
        parent_run.summary["study/best_trial_number"] = study.best_trial.number
        parent_run.summary["study/best_val_micro_f1"] = study.best_value
    if os.path.isfile(importance_path):
        parent_run.log({"study/optuna_param_importance": wandb.Image(importance_path)})
