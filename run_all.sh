#!/usr/bin/env bash
# Reproduce every reported result from the raw public files.
# Usage: bash run_all.sh [work_dir]      (default: ./work)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:-$HERE/work}"
mkdir -p "$WORK/data" "$WORK/outputs"

cd "$WORK/data"
[ -d work-at-olist-data ] || git clone --depth 1 https://github.com/olist/work-at-olist-data
[ -d LeIA ] || git clone --depth 1 https://github.com/rafjaa/LeIA
[ -f stopwords.zip ] || curl -sSL -o stopwords.zip \
  https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip
(cd work-at-olist-data/datasets && sha256sum -c "$HERE/olist_sha256.txt")

DATA="$WORK/data/work-at-olist-data/datasets"
OUT="$WORK/outputs"
cd "$HERE/src"
python3 01_load_audit.py "$DATA" "$OUT"
python3 02_sentiment.py "$OUT" "$WORK/data/LeIA" "$WORK/data/stopwords.zip"
python3 03_patterns.py "$OUT"
python3 04_drivers.py "$OUT"
python3 05_topics.py "$OUT"
python3 06_repurchase.py "$OUT"
python3 07_themes_and_validation.py "$OUT"
echo "Done. Results are in $OUT"
