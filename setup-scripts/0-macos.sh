#!/usr/bin/env bash
# Δημιουργεί .venv στη ρίζα του repo (macOS / Linux με bash και Python 3).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "→ git config core.hooksPath .githooks"
  git config core.hooksPath .githooks
else
  echo "Προειδοποίηση: όχι git repo εδώ — παράλειψη hooks." >&2
fi

VENV="$ROOT/.venv"

if [ -d "$VENV" ]; then
  echo "Το .venv υπάρχει ήδη."
else
  echo "→ python -m venv .venv"
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv "$VENV"
  elif command -v python >/dev/null 2>&1; then
    python -m venv "$VENV"
  else
    echo "Δεν βρέθηκε python3/python. Εγκατάστησε Python 3." >&2
    exit 1
  fi
  echo "→ pip install --upgrade pip (μέσα στο venv)"
  "$VENV/bin/python" -m pip install --upgrade pip
  echo "Έτοιμο: δημιουργήθηκε το .venv."
fi

echo ""
echo "Ενεργοποίηση στο τρέχον τερματικό:"
echo "  source $VENV/bin/activate"
if [ -f "$VENV/bin/activate.fish" ]; then
  echo "  (fish) source $VENV/bin/activate.fish"
fi
echo "Απενεργοποίηση: deactivate"
