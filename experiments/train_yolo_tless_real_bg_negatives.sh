#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO_ROOT="$PROJECT_ROOT/datasets/derived/yolo/tless_pbr_full_plus_real_bg_negatives"
DATA_YAML="$YOLO_ROOT/data.yaml"
RUN_DIR="$PROJECT_ROOT/experiments/runs/yolo"

if [[ ! -f "$DATA_YAML" ]]; then
  python "$PROJECT_ROOT/experiments/create_real_background_negatives.py"
  python "$PROJECT_ROOT/experiments/create_yolo_with_background_negatives.py"
fi

yolo detect train \
  data="$DATA_YAML" \
  model=yolov8n.pt \
  imgsz=640 \
  epochs=20 \
  batch=16 \
  hsv_h=0.05 \
  hsv_s=0.9 \
  hsv_v=0.7 \
  degrees=15 \
  translate=0.2 \
  scale=0.8 \
  shear=3 \
  perspective=0.001 \
  fliplr=0.5 \
  mosaic=1.0 \
  mixup=0.15 \
  erasing=0.6 \
  project="$RUN_DIR" \
  name=tless_pbr_full_dr_plus_real_bg_negatives

