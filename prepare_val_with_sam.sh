#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/work/llama_young/esod_sizak"
VAL_ZIP="/home/work/llama_young/VisDrone2019-DET-val.zip"
FRESH_DIR="/home/work/VisDrone_val_fresh"
ESOD_DIR="$ROOT/esod"

if [ ! -f "$VAL_ZIP" ]; then
  echo "VisDrone2019-DET-val.zip not found at $VAL_ZIP" >&2
  exit 1
fi

echo "[1/3] Preparing clean VisDrone val directory..."
rm -rf "$FRESH_DIR"
mkdir -p "$FRESH_DIR"
unzip -q "$VAL_ZIP" -d "$FRESH_DIR"

echo "[2/3] Activating virtual environment..."
source "$ROOT/.venv/bin/activate"

echo "[3/3] Running SAM-based preprocessing (this may take several minutes)..."
cd "$ESOD_DIR"
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
  python scripts/data_prepare.py --dataset "$FRESH_DIR"

echo "Done. Processed data stored in $FRESH_DIR/VisDrone2019-DET-val"
