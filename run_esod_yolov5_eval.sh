#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/work/llama_young/esod_sizak"
ESOD_DIR="$ROOT/esod"
WEIGHTS="$ROOT/esod_yolov5m.pt"
OUT_DIR="$ESOD_DIR/runs/test/visdrone_full_esod_yolov5m"

if [ ! -f "$WEIGHTS" ]; then
  echo "esod_yolov5m.pt not found at $WEIGHTS" >&2
  exit 1
fi

source "$ROOT/.venv/bin/activate"
cd "$ESOD_DIR"
mkdir -p "$OUT_DIR"

python test.py \
  --weights "$WEIGHTS" \
  --data data/visdrone.yaml \
  --img-size 1536 \
  --batch-size 8 \
  --device 0 \
  --task val \
  --save-txt \
  --project runs/test \
  --name visdrone_full_esod_yolov5m \
  --exist-ok | tee "$OUT_DIR/metrics.log"

echo "ESOD YOLOv5m metrics saved to $OUT_DIR"
