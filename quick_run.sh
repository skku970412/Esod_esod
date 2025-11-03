#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/work/llama_young/esod_sizak"
ESOD_DIR="$ROOT/esod"

# 1) venv setup
echo "[1/4] Setting up virtual environment..."
cd "$ROOT"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv --without-pip
  ./.venv/bin/python get-pip.py
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r esod/requirements.txt
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# 2) Link VisDrone dataset
echo "[2/4] Linking VisDrone dataset..."
cd "$ESOD_DIR"
ln -sfn /home/work/llama_young/VisDrone VisDrone

# Optional SAM-based preprocessing (comment out to skip)
echo "[optional] Running data preparation with SAM..."
python scripts/data_prepare.py --dataset VisDrone || echo "Skip data preparation"

# 3) Prepare 10-image subset
echo "[3/4] Preparing 10-image list..."
head -n 10 VisDrone/split/val.txt > VisDrone/split/val_10.txt

# 4) Run detection
echo "[4/4] Running detection on 10 images..."
python detect.py \
  --weights ../esod_yolov8m.pt \
  --source VisDrone/split/val_10.txt \
  --img-size 1536 \
  --device 0 \
  --project runs/detect \
  --name visdrone_val10 \
  --exist-ok

echo "Done. Results in runs/detect/visdrone_val10/"
