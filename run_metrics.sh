#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/work/llama_young/esod_sizak"
ESOD_DIR="$ROOT/esod"
OUT_DIR="$ESOD_DIR/runs/test/visdrone_val10_metrics"

source "$ROOT/.venv/bin/activate"
cd "$ESOD_DIR"
mkdir -p "$OUT_DIR"

python test.py \
  --weights ../esod_yolov8m.pt \
  --data data/visdrone_val10.yaml \
  --img-size 1536 \
  --batch-size 1 \
  --device 0 \
  --task val \
  --save-txt \
  --project runs/test \
  --name visdrone_val10_metrics \
  --exist-ok | tee "$OUT_DIR/metrics.log"

echo "Metrics and log saved to $OUT_DIR"
