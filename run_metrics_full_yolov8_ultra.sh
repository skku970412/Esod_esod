#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/work/llama_young/esod_sizak"
ESOD_DIR="$ROOT/esod"
OUT_DIR="$ESOD_DIR/runs/ultra/visdrone_full_yolov8_baseline"
WEIGHTS="$ESOD_DIR/weights/yolov8m.pt"

VAL_TXT="$ESOD_DIR/VisDrone/split/val.txt"
VAL_ABS="$ESOD_DIR/VisDrone/split/val_abs.txt"

if [ ! -f "$VAL_ABS" ]; then
  python - <<'PY'
from pathlib import Path
root = Path('/home/work/llama_young/esod_sizak/esod')
src = root/'VisDrone/split/val.txt'
dst = root/'VisDrone/split/val_abs.txt'
with src.open() as f, dst.open('w') as g:
    for line in f:
        path = line.strip()
        if path:
            g.write(str(root/Path(path)) + '\n')
print('wrote', dst)
PY
fi

DATA_YAML="$ESOD_DIR/data/visdrone_ultra.yaml"
cat > "$DATA_YAML" <<YAML
train: $VAL_ABS
val: $VAL_ABS
test: $VAL_ABS
nc: 10
names: ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
YAML

if [ ! -f "$WEIGHTS" ]; then
  echo "yolov8m.pt not found at $WEIGHTS" >&2
  exit 1
fi

source "$ROOT/.venv/bin/activate"
mkdir -p "$OUT_DIR"

COMMAND="yolo detect val model=$WEIGHTS data=$DATA_YAML imgsz=1536 batch=8 device=0 project=$ESOD_DIR/runs/ultra name=visdrone_full_yolov8_baseline exist_ok=True save=False"

echo "Running: $COMMAND"
$COMMAND | tee "$OUT_DIR/metrics.log"

echo "Baseline metrics saved under $OUT_DIR and Ultralytics runs directory"
