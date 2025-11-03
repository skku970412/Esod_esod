g#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/work/llama_young/esod_sizak"
ESOD_DIR="$ROOT/esod"
OUT_DIR="$ESOD_DIR/runs/test/visdrone_full_yolov5m"

source "$ROOT/.venv/bin/activate"
cd "$ESOD_DIR"
mkdir -p "$OUT_DIR"

python test.py \
  --weights weights/yolov5m.pt \
  --data data/visdrone.yaml \
  --img-size 1536 \
  --batch-size 8 \
  --device 0 \
  --task val \
  --save-txt \
  --project runs/test \
  --name visdrone_full_yolov5m \
  --exist-ok | tee "$OUT_DIR/metrics.log"

echo "Full-validation metrics and log for YOLOv5m saved to $OUT_DIR"
