#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/work/llama_young/esod_sizak"
ESOD_DIR="$ROOT/esod"
FRESH_DIR="/home/work/VisDrone_val_fresh"
OUT_DIR="$ESOD_DIR/runs/test/visdrone_full_esodv8m_sam"
YAML="$ESOD_DIR/data/visdrone_valfresh.yaml"

if [ ! -d "$FRESH_DIR/VisDrone2019-DET-val" ]; then
  echo "SAM-prepared VisDrone val data not found at $FRESH_DIR" >&2
  echo "Run ./prepare_val_with_sam.sh first." >&2
  exit 1
fi

cat > "$YAML" <<'YAML'
train: /home/work/VisDrone_val_fresh/split/train.txt
val: /home/work/VisDrone_val_fresh/split/val.txt
test: /home/work/VisDrone_val_fresh/split/val.txt
nc: 80
names: ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
        'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
        'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
        'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
        'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
        'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
        'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard',
        'cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase',
        'scissors','teddy bear','hair drier','toothbrush']
YAML

source "$ROOT/.venv/bin/activate"
cd "$ESOD_DIR"
mkdir -p "$OUT_DIR"

python test.py \
  --weights ../esod_yolov8m.pt \
  --data "$YAML" \
  --img-size 1536 \
  --batch-size 8 \
  --device 0 \
  --task val \
  --save-txt \
  --project runs/test \
  --name visdrone_full_esodv8m_sam \
  --exist-ok | tee "$OUT_DIR/metrics.log"

echo "Full-validation metrics with SAM-enhanced GT saved to $OUT_DIR"
