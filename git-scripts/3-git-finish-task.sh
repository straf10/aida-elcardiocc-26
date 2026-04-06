#!/usr/bin/env bash
# Push του τρέχοντος branch (όχι main) στο origin με -u αν χρειάζεται.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

CURRENT="$(git rev-parse --abbrev-ref HEAD)"
if [ "$CURRENT" = "main" ] || [ "$CURRENT" = "master" ]; then
  echo "Είσαι στο «$CURRENT». Πήγαινε πρώτα στο branch του task σου."
  exit 1
fi

if ! git diff-index --quiet HEAD -- 2>/dev/null; then
  echo "Υπάρχουν μη-committed αλλαγές. Κάνε commit ή stash και ξανά."
  exit 1
fi

echo "→ push origin $CURRENT"
git push -u origin "$CURRENT"

echo "Έτοιμο: στάλθηκε το «$CURRENT». Άνοιξε Pull request προς main στο GitHub."
