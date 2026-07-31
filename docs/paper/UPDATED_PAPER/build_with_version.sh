#!/usr/bin/env bash
set -euo pipefail
# Build the PERPLEXITY paper and copy output to a timestamped PDF.
cd "$(dirname "$0")"
echo "Building mechval_framework_v2_PERPLEXITY.tex..."
latexmk -pdf -g mechval_framework_v2_PERPLEXITY.tex
ts=$(date +'%Y%m%d_%H%M%S')
out=mechval_framework_v2_PERPLEXITY_v${ts}.pdf
cp mechval_framework_v2_PERPLEXITY.pdf "$out"
echo "Created $out"
