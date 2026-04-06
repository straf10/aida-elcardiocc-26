#!/usr/bin/env bash
# Συγχρονίζει το main και δημιουργεί νέο branch για task.
# Χρήση: ./scripts/git-start-task.sh feature/onoma-task
set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Χρήση: $0 <όνομα-branch>"
  echo "Παράδειγμα: $0 feature/mlc-baseline"
  exit 1
fi

BRANCH="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

echo "→ checkout main"
git checkout main

echo "→ pull origin main"
git pull origin main

echo "→ νέο branch: $BRANCH"
git checkout -b "$BRANCH"

echo "Έτοιμο: δουλεύεις στο branch «$BRANCH»."
