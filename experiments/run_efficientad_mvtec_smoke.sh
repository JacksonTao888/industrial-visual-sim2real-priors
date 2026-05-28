#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

conda run --no-capture-output -n sim2real python experiments/run_efficientad_anomalib.py \
  --dataset mvtec_ad \
  --categories toothbrush \
  --train-batch-size 1 \
  --eval-batch-size 32 \
  --num-workers 8 \
  --model-size small \
  --max-epochs 1
