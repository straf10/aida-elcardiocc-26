#!/usr/bin/env python3
"""Συγχρονίζει το main και δημιουργεί νέο branch για task."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> None:
    if len(sys.argv) < 2:
        print("Χρήση: python 2-git-start-task.py <όνομα-branch>")
        print("Παράδειγμα: python 2-git-start-task.py feature/alice/mlc-baseline")
        sys.exit(1)
    branch = sys.argv[1]
    root = repo_root()
    print("→ checkout main")
    subprocess.run(["git", "checkout", "main"], cwd=root, check=True)
    print("→ pull origin main")
    subprocess.run(["git", "pull", "origin", "main"], cwd=root, check=True)
    print(f"→ νέο branch: {branch}")
    subprocess.run(["git", "checkout", "-b", branch], cwd=root, check=True)
    print(f"Έτοιμο: δουλεύεις στο branch «{branch}».")
    print("Επόμενα: `git add` / `git commit` όσο δουλεύεις · όταν είσαι έτοιμος → βήμα 3 (finish).")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
