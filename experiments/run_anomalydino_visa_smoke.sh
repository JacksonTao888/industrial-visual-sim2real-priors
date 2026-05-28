#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

conda run --no-capture-output -n sim2real python experiments/run_anomalydino_anomalib.py \
  --dataset visa \
  --categories macaroni2 capsules \
  --train-batch-size 16 \
  --eval-batch-size 16 \
  --num-workers 4 \
  --auto-masking
