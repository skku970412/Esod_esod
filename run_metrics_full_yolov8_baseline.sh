#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/work/llama_young/esod_sizak"
ESOD_DIR="$ROOT/esod"
OUT_DIR="$ESOD_DIR/runs/test/visdrone_full_yolov8_baseline"
WEIGHTS="$ESOD_DIR/weights/yolov8m.pt"
if [ ! -f "$WEIGHTS" ]; then
  echo "yolov8m.pt not found at $WEIGHTS" >&2
  echo "Download with: curl -L -o $WEIGHTS https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt" >&2
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
  --name visdrone_full_yolov8_baseline \
  --exist-ok | tee "$OUT_DIR/metrics.log"

echo "Baseline metrics saved to $OUT_DIR"
