#!/usr/bin/env python3
"""Συγχρονίζει το main και δημιουργεί νέο branch για task."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def normalize_segment(part: str) -> str:
    """Κενά → παύλες· μόνο γράμματα, αριθμοί, . _ - (κανόνες Git για ref)."""
    s = part.strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-zA-Z0-9._-]", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def normalize_branch_name(branch: str) -> str:
    parts = [p for p in branch.split("/") if p.strip()]
    out: list[str] = []
    for p in parts:
        n = normalize_segment(p)
        if not n:
            print("Μη έγκυρο όνομα branch (κενό τμήμα μετά το καθάρισμα).")
            sys.exit(1)
        out.append(n)
    return "/".join(out)


def read_branch() -> str:
    if len(sys.argv) >= 2:
        raw = sys.argv[1].strip()
        branch = normalize_branch_name(raw)
        if branch != raw:
            print(f"  (κανονικοποιήθηκε: {branch})")
        return branch
    print("")
    print("Νέο branch: feature/<GitHub username>/<όνομα-task> (τα κενά γίνονται παύλες)")
    username = input("  GitHub username: ").strip()
    task = input("  Όνομα task (branch): ").strip()
    if not username or not task:
        print("Χρειάζονται και τα δύο.")
        sys.exit(1)
    if "/" in username:
        print("Το username δεν πρέπει να περιέχει «/».")
        sys.exit(1)
    u = normalize_segment(username)
    t = normalize_segment(task)
    if not u or not t:
        print("Μη έγκυρο username ή όνομα task μετά το καθάρισμα.")
        sys.exit(1)
    branch = f"feature/{u}/{t}"
    if branch != f"feature/{username}/{task}":
        print(f"  (κανονικοποιήθηκε όνομα branch: {branch})")
    return branch


def main() -> None:
    branch = read_branch()
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
