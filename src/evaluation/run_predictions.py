#!/usr/bin/env python3
"""
Run all built-in prediction pipelines on the default labeled test split (JSONL out).

From the repository root::

    PYTHONPATH=src python -m evaluation.run_predictions

Options::

    PYTHONPATH=src python -m evaluation.run_predictions --skip xlm_base,ner
    PYTHONPATH=src python -m evaluation.run_predictions --ner-model-dir /path/to/NER_EL

``PYTHONPATH`` must include ``src`` (this module does not modify ``sys.path`` for child processes
beyond setting ``PYTHONPATH`` for subprocesses the same way).
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


def _env_with_src(src: Path) -> dict[str, str]:
    sep = os.pathsep
    prev = os.environ.get("PYTHONPATH", "")
    merged = str(src) + (sep + prev if prev else "")
    return {**os.environ, "PYTHONPATH": merged}


def _run(name: str, cwd: Path, env: dict[str, str], cmd: list[str]) -> None:
    print(f"\n{'=' * 60}\n[{name}]\n{' '.join(cmd)}\n{'=' * 60}", flush=True)
    rc = subprocess.run(cmd, cwd=str(cwd), env=env).returncode
    if rc != 0:
        raise SystemExit(f"[{name}] failed with exit code {rc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all prediction JSONL generators.")
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated: mlc, xlm_large, xlm_base, ir, ner",
    )
    parser.add_argument(
        "--ner-model-dir",
        default=None,
        help="NER+EL HuggingFace save dir (default: outputs/models/NER_EL under repo root)",
    )
    args = parser.parse_args()
    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}

    if not _SRC_ROOT.is_dir():
        raise SystemExit(f"Expected src/ at {_SRC_ROOT}")
    env = _env_with_src(_SRC_ROOT)
    py = sys.executable

    ner_dir = Path(args.ner_model_dir) if args.ner_model_dir else _REPO_ROOT / "outputs" / "models" / "NER_EL"

    steps: list[tuple[str, list[str]]] = []

    if "mlc" not in skip:
        steps.append(
            (
                "mlc_greek_bert",
                [
                    py,
                    "-m",
                    "mlc_greek_bert.predict",
                    "--config",
                    "src/mlc_greek_bert/mlc_greek_bert.yaml",
                ],
            )
        )

    if "xlm_large" not in skip:
        steps.append(
            (
                "xlm_r_large",
                [
                    py,
                    "-m",
                    "xlm_r_large.predict",
                    "--config",
                    "src/xlm_r_large/xlm_r.yaml",
                    "--split",
                    "test",
                    "--thresholds",
                    "outputs/models/xlm_large/thresholds.json",
                ],
            )
        )

    if "xlm_base" not in skip:
        steps.append(
            (
                "xlm_r_base",
                [
                    py,
                    "-m",
                    "xlm_r_base.predict",
                    "--config",
                    "src/xlm_r_base/xlm_r_base.yaml",
                    "--data",
                    "data/processed/test.jsonl",
                    "--out",
                    "outputs/predictions/xlm_r_base/predictions.jsonl",
                ],
            )
        )

    if "ir" not in skip:
        steps.append(
            (
                "information_retrieval",
                [py, "-m", "information_retrieval.predict"],
            )
        )

    if "ner" not in skip:
        if not ner_dir.is_dir():
            print(f"\n[ner] SKIP: model dir not found: {ner_dir}\n", flush=True)
        else:
            steps.append(
                (
                    "ner_el",
                    [
                        py,
                        "-m",
                        "ner_el.predict",
                        "--model-dir",
                        str(ner_dir),
                        "--input-path",
                        "data/processed/test.jsonl",
                        "--output-doc-path",
                        "outputs/predictions/ner_el/predictions.jsonl",
                        "--output-debug-path",
                        "outputs/predictions/ner_el/predictions.debug.jsonl",
                    ],
                )
            )

    if not steps:
        print("Nothing to run (all steps skipped).", flush=True)
        return

    for name, cmd in steps:
        _run(name, _REPO_ROOT, env, cmd)

    print("\nAll prediction steps finished.\n", flush=True)


if __name__ == "__main__":
    main()
