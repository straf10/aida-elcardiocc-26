"""Backward-compatible entrypoint: same as ``python -m evaluation.evaluator compare``."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .evaluator import DEFAULT_EVAL_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare F1 across methods (delegates to evaluation.evaluator compare)."
    )
    parser.add_argument("--config", default=DEFAULT_EVAL_CONFIG)
    args, rest = parser.parse_known_args()
    root = Path(__file__).resolve().parents[2]
    cfg_path = Path(args.config)
    cfg = str(cfg_path.resolve() if cfg_path.is_absolute() else (root / cfg_path).resolve())
    src = str(root / "src")
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [sys.executable, "-m", "evaluation.evaluator", "compare", "--config", cfg, *rest]
    raise SystemExit(subprocess.call(cmd, cwd=str(root), env=env))


if __name__ == "__main__":
    main()
