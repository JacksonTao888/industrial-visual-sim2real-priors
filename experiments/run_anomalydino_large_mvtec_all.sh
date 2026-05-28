#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

conda run --no-capture-output -n sim2real python experiments/run_anomalydino_anomalib.py \
  --dataset mvtec_ad \
  --categories all \
  --train-batch-size 8 \
  --eval-batch-size 8 \
  --num-workers 4 \
  --auto-masking \
  --encoder-name dinov2_vit_large_14 \
  --out-root experiments/runs/no_cad/anomalydino_large
