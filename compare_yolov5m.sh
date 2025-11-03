#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/work/llama_young/esod_sizak"
ESOD_DIR="$ROOT/esod"
BASE_WEIGHTS="$ROOT/yolov5m.pt"
ESOD_WEIGHTS="$ROOT/esod_yolov5m.pt"

if [ ! -f "$BASE_WEIGHTS" ]; then
  echo "Baseline weights not found at $BASE_WEIGHTS" >&2
  exit 1
fi
if [ ! -f "$ESOD_WEIGHTS" ]; then
  echo "ESOD weights not found at $ESOD_WEIGHTS" >&2
  exit 1
fi

OUT_BASE="$ESOD_DIR/runs/test/visdrone_full_yolov5m_base"
OUT_ESOD="$ESOD_DIR/runs/test/visdrone_full_yolov5m_esod"

source "$ROOT/.venv/bin/activate"
cd "$ESOD_DIR"
mkdir -p "$OUT_BASE" "$OUT_ESOD"

python test.py \
  --weights "$BASE_WEIGHTS" \
  --data data/visdrone.yaml \
  --img-size 1536 \
  --batch-size 8 \
  --device 0 \
  --task val \
  --save-txt \
  --project runs/test \
  --name visdrone_full_yolov5m_base \
  --exist-ok | tee "$OUT_BASE/metrics.log"

python test.py \
  --weights "$ESOD_WEIGHTS" \
  --data data/visdrone.yaml \
  --img-size 1536 \
  --batch-size 8 \
  --device 0 \
  --task val \
  --save-txt \
  --project runs/test \
  --name visdrone_full_yolov5m_esod \
  --exist-ok | tee "$OUT_ESOD/metrics.log"

echo "Comparison complete. Logs saved to:\n  $OUT_BASE/metrics.log\n  $OUT_ESOD/metrics.log"
