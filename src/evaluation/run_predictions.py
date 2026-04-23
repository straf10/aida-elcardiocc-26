#!/usr/bin/env python3
"""
Run built-in prediction pipelines for **val**, **labeled test**, and **blind** JSONL outputs.

Writes under ``outputs/predictions/<method>/``:

- ``val_predictions.jsonl``
- ``test_predictions.jsonl`` (path used by ``evaluation.config.yaml`` / ``compare_methods``)
- ``blind_predictions.jsonl`` (skipped if ``data/raw/blind_test.jsonl`` is missing)

If one subprocess errors, a message is printed and **remaining commands still run**.
Exit code **1** if any command failed, **0** if all succeeded.

From the repository root::

    PYTHONPATH=src python -m evaluation.run_predictions

Options::

    PYTHONPATH=src python -m evaluation.run_predictions --skip xlm_base,ner,dictionary
    PYTHONPATH=src python -m evaluation.run_predictions --ner-model-dir /path/to/ner_el

``PYTHONPATH`` must include ``src``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
_SRC_ROOT = _EVAL_DIR.parent
_REPO_ROOT = _SRC_ROOT.parent

PROCESSED_VAL = "data/processed/val.jsonl"
PROCESSED_TEST = "data/processed/test.jsonl"
RAW_BLIND = "data/raw/blind_test.jsonl"


def _env_with_src(src: Path) -> dict[str, str]:
    sep = os.pathsep
    prev = os.environ.get("PYTHONPATH", "")
    merged = str(src) + (sep + prev if prev else "")
    return {**os.environ, "PYTHONPATH": merged}


def _run(name: str, cwd: Path, env: dict[str, str], cmd: list[str]) -> bool:
    print(f"\n{'=' * 60}\n[{name}]\n{' '.join(cmd)}\n{'=' * 60}", flush=True)
    try:
        rc = subprocess.run(cmd, cwd=str(cwd), env=env).returncode
    except OSError as exc:
        print(f"[{name}] ERROR: could not run subprocess ({exc!r}) (continuing).\n", flush=True)
        return False
    if rc != 0:
        print(f"[{name}] ERROR: subprocess exited with code {rc} (continuing).\n", flush=True)
        return False
    return True


def _default_ner_model_dir(repo: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    for rel in ("outputs/models/ner_el", "outputs/models/NER_EL"):
        p = repo / rel
        if p.is_dir():
            return p
    return repo / "outputs/models/ner_el"


def _first_existing_threshold(repo: Path, *rels: str) -> str:
    for rel in rels:
        p = repo / rel
        if p.is_file():
            return rel
    return rels[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run val / test / blind prediction JSONL generators for each track.",
    )
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated: mlc, xlm_large, xlm_base, ir, dictionary, ner",
    )
    parser.add_argument(
        "--ner-model-dir",
        default=None,
        help="NER+EL HuggingFace save dir (default: first existing of outputs/models/ner_el, outputs/models/NER_EL)",
    )
    parser.add_argument(
        "--xlm-base-folds",
        type=int,
        default=5,
        help="``--folds`` for xlm_r_base.predict (match checkpoint count on disk).",
    )
    args = parser.parse_args()
    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}

    if not _SRC_ROOT.is_dir():
        raise SystemExit(f"Expected src/ at {_SRC_ROOT}")
    env = _env_with_src(_SRC_ROOT)
    py = sys.executable
    repo = _REPO_ROOT

    ner_dir = _default_ner_model_dir(repo, Path(args.ner_model_dir) if args.ner_model_dir else None)

    thr_xlm_large = _first_existing_threshold(
        repo,
        "outputs/models/xlm_r_large/thresholds.json",
        "outputs/models/xlm_large/thresholds.json",
    )

    failures: list[str] = []
    blind_exists = (repo / RAW_BLIND).is_file()
    if not blind_exists:
        print(
            f"\n[run_predictions] Note: {RAW_BLIND} not found — blind prediction steps are skipped.\n",
            flush=True,
        )

    def run_group(group_name: str, commands: list[list[str]]) -> None:
        for i, cmd in enumerate(commands):
            label = f"{group_name}[{i + 1}/{len(commands)}]"
            if not _run(label, repo, env, cmd):
                failures.append(label)

    # --- MLC Greek BERT ---
    if "mlc" not in skip:
        base = [py, "-m", "mlc_greek_bert.predict", "--config", "src/mlc_greek_bert/mlc_greek_bert.yaml"]
        mlc_cmds = [
            base + ["--input", PROCESSED_VAL, "--output", "outputs/predictions/mlc_greek_bert/val_predictions.jsonl"],
            base + ["--input", PROCESSED_TEST, "--output", "outputs/predictions/mlc_greek_bert/test_predictions.jsonl"],
        ]
        if blind_exists:
            mlc_cmds.append(
                base + ["--input", RAW_BLIND, "--output", "outputs/predictions/mlc_greek_bert/blind_predictions.jsonl"],
            )
        run_group("mlc_greek_bert", mlc_cmds)

    # --- XLM-R Large ---
    if "xlm_large" not in skip:
        base = [
            py,
            "-m",
            "xlm_r_large.predict",
            "--config",
            "src/xlm_r_large/xlm_r.yaml",
            "--thresholds",
            thr_xlm_large,
        ]
        xl_cmds = [base + ["--split", "val"], base + ["--split", "test"]]
        if blind_exists:
            xl_cmds.append(base + ["--split", "blind"])
        run_group("xlm_r_large", xl_cmds)

    # --- XLM-R Base ---
    if "xlm_base" not in skip:
        base = [
            py,
            "-m",
            "xlm_r_base.predict",
            "--config",
            "src/xlm_r_base/xlm_r_base.yaml",
            "--folds",
            str(int(args.xlm_base_folds)),
        ]
        xb_cmds = [
            base + ["--data", PROCESSED_VAL, "--out", "outputs/predictions/xlm_r_base/val_predictions.jsonl"],
            base + ["--data", PROCESSED_TEST, "--out", "outputs/predictions/xlm_r_base/test_predictions.jsonl"],
        ]
        if blind_exists:
            xb_cmds.append(
                base + ["--data", RAW_BLIND, "--out", "outputs/predictions/xlm_r_base/blind_predictions.jsonl"],
            )
        run_group("xlm_r_base", xb_cmds)

    # --- Information retrieval (one tuned fit, three writes) ---
    if "ir" not in skip:
        cmd = [
            py,
            "-m",
            "information_retrieval.predict",
            "--tune",
            "--retriever",
            "hybrid",
            "--embedding-model",
            "intfloat/multilingual-e5-base",
            "--hybrid-rrf-k",
            "30",
            "--hybrid-bm25-weight",
            "1.0",
            "--hybrid-dense-weight",
            "0.4",
            "--export-standard-splits-dir",
            "outputs/predictions/information_retrieval",
        ]
        if not _run("information_retrieval", repo, env, cmd):
            failures.append("information_retrieval")

    # --- Dictionary baseline ---
    if "dictionary" not in skip:
        cmd = [py, "-m", "dictionary.cli", "--config", "src/dictionary/dictionary.yaml"]
        if not _run("dictionary_baseline", repo, env, cmd):
            failures.append("dictionary_baseline")

    # --- NER+EL ---
    if "ner" not in skip:
        if ner_dir.is_dir():
            base = [py, "-m", "ner_el.predict", "--model-dir", str(ner_dir)]
            ner_cmds = [
                base
                + [
                    "--input-path",
                    PROCESSED_VAL,
                    "--output-doc-path",
                    "outputs/predictions/ner_el/val_predictions.jsonl",
                    "--output-debug-path",
                    "outputs/predictions/ner_el/val_predictions.debug.jsonl",
                ],
                base
                + [
                    "--input-path",
                    PROCESSED_TEST,
                    "--output-doc-path",
                    "outputs/predictions/ner_el/test_predictions.jsonl",
                    "--output-debug-path",
                    "outputs/predictions/ner_el/test_predictions.debug.jsonl",
                ],
            ]
            if blind_exists:
                ner_cmds.append(
                    base
                    + [
                        "--input-path",
                        RAW_BLIND,
                        "--output-doc-path",
                        "outputs/predictions/ner_el/blind_predictions.jsonl",
                        "--output-debug-path",
                        "outputs/predictions/ner_el/blind_predictions.debug.jsonl",
                    ],
                )
            run_group("ner_el", ner_cmds)
        else:
            print(
                f"\n[ner] SKIP: no model dir at {ner_dir}\n",
                flush=True,
            )

    print("\nAll prediction steps finished.\n", flush=True)
    if failures:
        print(
            "WARNING: One or more commands did not complete successfully: "
            + ", ".join(failures)
            + "\n",
            flush=True,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
