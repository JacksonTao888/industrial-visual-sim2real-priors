#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_YAML="$PROJECT_ROOT/datasets/derived/yolo/tless_pbr_5k_to_real/data.yaml"
RUN_DIR="$PROJECT_ROOT/experiments/runs/yolo"

yolo detect train \
  data="$DATA_YAML" \
  model=yolov8n.pt \
  imgsz=640 \
  epochs=20 \
  batch=16 \
  project="$RUN_DIR" \
  name=tless_pbr_5k_to_real_yolov8n
