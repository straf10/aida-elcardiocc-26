#!/usr/bin/env python3
"""Ενημερώνει το τοπικό main από το origin (ρίζα repo)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> None:
    root = repo_root()
    print("→ checkout main")
    subprocess.run(["git", "checkout", "main"], cwd=root, check=True)
    print("→ pull origin main")
    subprocess.run(["git", "pull", "origin", "main"], cwd=root, check=True)
    print("Έτοιμο: το main είναι ενημερωμένο.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
