#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO_ROOT="$PROJECT_ROOT/datasets/derived/yolo/tless_pbr_full_to_real"
DATA_YAML="$YOLO_ROOT/data.yaml"
RUN_DIR="$PROJECT_ROOT/experiments/runs/yolo"

if [[ ! -f "$DATA_YAML" ]]; then
  python "$PROJECT_ROOT/experiments/coco_to_yolo_detection.py" \
    --out-root "$YOLO_ROOT" \
    --split train:"$PROJECT_ROOT/datasets/derived/coco/tless_train_pbr_detection.json" \
    --split val:"$PROJECT_ROOT/datasets/derived/coco/tless_test_primesense_detection.json"
fi

yolo detect train \
  data="$DATA_YAML" \
  model=yolov8s.pt \
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
  name=tless_pbr_full_to_real_yolov8s_domain_randomization

