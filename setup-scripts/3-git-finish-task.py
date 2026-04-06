#!/usr/bin/env python3
"""Push του τρέχοντος branch (όχι main) στο origin με -u αν χρειάζεται."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def current_branch(root: Path) -> str:
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return r.stdout.strip()


def is_dirty(root: Path) -> bool:
    r = subprocess.run(
        ["git", "diff-index", "--quiet", "HEAD", "--"],
        cwd=root,
    )
    return r.returncode != 0


def main() -> None:
    root = repo_root()
    cur = current_branch(root)
    if cur in ("main", "master"):
        print(f"Είσαι στο «{cur}». Πήγαινε πρώτα στο branch του task σου.")
        sys.exit(1)
    if is_dirty(root):
        print("Υπάρχουν μη-committed αλλαγές. Κάνε commit ή stash και ξανά.")
        sys.exit(1)
    print(f"→ push origin {cur}")
    subprocess.run(["git", "push", "-u", "origin", cur], cwd=root, check=True)
    print(
        f"Έτοιμο: στάλθηκε το «{cur}». Άνοιξε Pull request προς main στο GitHub."
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
