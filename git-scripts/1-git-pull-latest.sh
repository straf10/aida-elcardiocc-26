#!/usr/bin/env bash
# Ενημερώνει το τοπικό main από το origin (από ρίζα repo).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

echo "→ checkout main"
git checkout main

echo "→ pull origin main"
git pull origin main

echo "Έτοιμο: το main είναι ενημερωμένο."
