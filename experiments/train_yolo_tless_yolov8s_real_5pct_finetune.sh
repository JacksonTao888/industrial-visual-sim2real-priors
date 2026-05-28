#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_YAML="$PROJECT_ROOT/datasets/derived/yolo/tless_real_5pct_ft/data.yaml"
RUN_DIR="$PROJECT_ROOT/experiments/runs/yolo"
PRETRAINED="$RUN_DIR/tless_pbr_full_to_real_yolov8s_domain_randomization/weights/best.pt"

if [[ ! -f "$PRETRAINED" ]]; then
  echo "Missing pretrained YOLOv8s domain-randomization checkpoint: $PRETRAINED" >&2
  exit 1
fi

if [[ ! -f "$DATA_YAML" ]]; then
  python "$PROJECT_ROOT/experiments/coco_to_yolo_detection.py" \
    --out-root "$PROJECT_ROOT/datasets/derived/yolo/tless_real_5pct_ft" \
    --split train:"$PROJECT_ROOT/datasets/derived/coco/tless_real_train_5pct_50_detection.json" \
    --split val:"$PROJECT_ROOT/datasets/derived/coco/tless_real_val_heldout_900_detection.json"
fi

yolo detect train \
  data="$DATA_YAML" \
  model="$PRETRAINED" \
  imgsz=640 \
  epochs=30 \
  batch=16 \
  lr0=0.001 \
  lrf=0.01 \
  mosaic=0.0 \
  mixup=0.0 \
  degrees=5 \
  translate=0.1 \
  scale=0.3 \
  hsv_h=0.015 \
  hsv_s=0.4 \
  hsv_v=0.3 \
  patience=15 \
  project="$RUN_DIR" \
  name=tless_yolov8s_dr_pretrained_real_5pct_finetune
