#!/usr/bin/env python3
"""
Run all built-in prediction pipelines on the default labeled test split (JSONL out).
If one method errors (missing checkpoints, failed subprocess, copy error), a message is printed
and **remaining methods still run**. Exit code is **1** if any step failed, **0** if all succeeded.

From the repository root::

    PYTHONPATH=src python -m evaluation.run_predictions

If a bundled file exists, it is **copied** to the usual ``outputs/predictions/...`` path and
inference is **skipped** for that step:

- ``outputs/models/greek_bert/predictions.jsonl`` → ``outputs/predictions/mlc_greek_bert/predictions.jsonl``
- ``outputs/models/xlm_large/predictions.jsonl`` → ``outputs/predictions/xlm_r_large/predictions.jsonl``
- ``outputs/models/xlm_base/predictions.jsonl`` → ``outputs/predictions/xlm_r_base/predictions.jsonl``
- ``outputs/models/ir/predictions.jsonl`` → ``outputs/predictions/information_retrieval/predictions.jsonl``
- NER: ``outputs/models/ner/predictions.jsonl``, then ``outputs/models/NER_EL/predictions.jsonl``,
  then ``<ner-model-dir>/predictions.jsonl`` → ``outputs/predictions/ner_el/predictions.jsonl``

Options::

    PYTHONPATH=src python -m evaluation.run_predictions --skip xlm_base,ner
    PYTHONPATH=src python -m evaluation.run_predictions --ner-model-dir /path/to/NER_EL

``PYTHONPATH`` must include ``src`` (this module does not modify ``sys.path`` for child processes
beyond setting ``PYTHONPATH`` for subprocesses the same way).
"""

from __future__ import annotations

import argparse
import os
import shutil
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


def _try_copy_bundled_predictions(repo: Path, name: str, src: Path | None, dst_rel: str) -> bool:
    """
    If ``src`` exists (any absolute or relative path), copy to ``repo / dst_rel`` and return True.
    ``dst_rel`` is repo-relative.
    """
    if src is None or not src.is_file():
        return False
    dst = repo / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    try:
        src_disp = str(src.resolve().relative_to(repo.resolve()))
    except ValueError:
        src_disp = str(src)
    print(
        f"\n{'=' * 60}\n[{name}] bundled predictions (inference skipped)\n"
        f"  {src_disp}\n  -> {dst_rel}\n{'=' * 60}",
        flush=True,
    )
    return True


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

    # (step_name, subprocess_cmd or None, bundled_src Path or None, predictions_dst_rel under repo)
    # If bundled_src exists, copy to dst and skip inference.
    steps: list[tuple[str, list[str] | None, Path | None, str]] = []

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
                _REPO_ROOT / "outputs/models/greek_bert/predictions.jsonl",
                "outputs/predictions/mlc_greek_bert/predictions.jsonl",
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
                    "--thresholds",
                    "outputs/models/xlm_large/thresholds.json",
                ],
                _REPO_ROOT / "outputs/models/xlm_large/predictions.jsonl",
                "outputs/predictions/xlm_r_large/predictions.jsonl",
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
                _REPO_ROOT / "outputs/models/xlm_base/predictions.jsonl",
                "outputs/predictions/xlm_r_base/predictions.jsonl",
            )
        )

    if "ir" not in skip:
        steps.append(
            (
                "information_retrieval",
                [py, "-m", "information_retrieval.predict"],
                _REPO_ROOT / "outputs/models/ir/predictions.jsonl",
                "outputs/predictions/information_retrieval/predictions.jsonl",
            )
        )

    ner_dst_rel = "outputs/predictions/ner_el/predictions.jsonl"
    if "ner" not in skip:
        ner_cmd: list[str] | None = None
        if ner_dir.is_dir():
            ner_cmd = [
                py,
                "-m",
                "ner_el.predict",
                "--model-dir",
                str(ner_dir),
                "--input-path",
                "data/processed/test.jsonl",
                "--output-doc-path",
                ner_dst_rel,
                "--output-debug-path",
                "outputs/predictions/ner_el/predictions.debug.jsonl",
            ]
        ner_bundled_candidates = [
            _REPO_ROOT / "outputs/models/ner/predictions.jsonl",
            _REPO_ROOT / "outputs/models/NER_EL/predictions.jsonl",
        ]
        if ner_dir.is_dir():
            ner_bundled_candidates.append(ner_dir / "predictions.jsonl")
        ner_bundle: Path | None = next((p for p in ner_bundled_candidates if p.is_file()), None)
        if ner_bundle is not None or ner_cmd is not None:
            steps.append(("ner_el", ner_cmd, ner_bundle, ner_dst_rel))
        else:
            print(
                f"\n[ner] SKIP: no bundled predictions under outputs/models/ner|NER_EL and "
                f"no model dir: {ner_dir}\n",
                flush=True,
            )

    if not steps:
        print("Nothing to run (all steps skipped).", flush=True)
        return

    failures: list[str] = []
    for name, cmd, bundled_src, dst_rel in steps:
        try:
            if _try_copy_bundled_predictions(_REPO_ROOT, name, bundled_src, dst_rel):
                continue
            if cmd is None:
                print(
                    f"[{name}] Bundled predictions not found; inference not available — skipped.\n",
                    flush=True,
                )
                continue
            if not _run(name, _REPO_ROOT, env, cmd):
                failures.append(name)
        except Exception as exc:
            print(f"[{name}] ERROR: {exc!r} (continuing).\n", flush=True)
            failures.append(name)

    print("\nAll prediction steps finished.\n", flush=True)
    if failures:
        print(
            "WARNING: One or more methods did not complete successfully: "
            + ", ".join(failures)
            + "\n",
            flush=True,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
