#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/work/llama_young/esod_sizak"
ESOD_DIR="$ROOT/esod"
OUT_ROOT="$ESOD_DIR/runs/benchmark"
mkdir -p "$OUT_ROOT"

source "$ROOT/.venv/bin/activate"
cd "$ESOD_DIR"

# helper
run_eval() {
  local weights=$1
  local data_yaml=$2
  local imgsz=$3
  local batch=$4
  local name=$5
  local extra=$6

  local out_dir="$OUT_ROOT/$name"
  mkdir -p "$out_dir"

  python test.py \
    --weights "$weights" \
    --data "$data_yaml" \
    --img-size "$imgsz" \
    --batch-size "$batch" \
    --device 0 \
    --task val \
    --save-txt \
    --project "$OUT_ROOT" \
    --name "$name" \
    --exist-ok $extra | tee "$out_dir/metrics.log"
}

# 1) ESOD YOLOv5m @1536
run_eval "$ROOT/esod_yolov5m.pt" data/visdrone.yaml 1536 8 "esod_yolov5m_1536" ""

# 2) ESOD YOLOv8m @1536
run_eval "$ROOT/esod_yolov8m.pt" data/visdrone.yaml 1536 8 "esod_yolov8m_1536" ""

# 3) ESOD YOLOv5m @1920 (1.25x)
run_eval "$ROOT/esod_yolov5m.pt" data/visdrone.yaml 1920 4 "esod_yolov5m_1920" ""

# 4) ESOD YOLOv8m @1920 (1.25x)
run_eval "$ROOT/esod_yolov8m.pt" data/visdrone.yaml 1920 4 "esod_yolov8m_1920" ""

echo "All benchmarks complete. Logs saved under $OUT_ROOT"
