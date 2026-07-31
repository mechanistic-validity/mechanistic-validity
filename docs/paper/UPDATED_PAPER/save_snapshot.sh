#!/usr/bin/env bash
set -euo pipefail

# Save a timestamped snapshot of the main paper tex and pdf into snapshots/
cd "$(dirname "$0")" || exit 1
mkdir -p snapshots
ts=$(date +%Y%m%dT%H%M%S)

src_tex="mechval_framework_v2_PERPLEXITY.tex"
src_pdf="mechval_framework_v2_PERPLEXITY.pdf"

if [ -f "$src_tex" ]; then
  cp "$src_tex" "snapshots/${src_tex%.tex}_${ts}.tex"
  echo "Saved snapshots/${src_tex%.tex}_${ts}.tex"
else
  echo "Warning: $src_tex not found in $(pwd)"
fi

if [ -f "$src_pdf" ]; then
  cp "$src_pdf" "snapshots/${src_pdf%.pdf}_${ts}.pdf"
  echo "Saved snapshots/${src_pdf%.pdf}_${ts}.pdf"
else
  echo "Note: $src_pdf not found; skipping PDF snapshot"
fi

echo "Snapshot complete: $ts"