#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

conda run --no-capture-output -n sim2real python experiments/run_patchcore_anomalib.py \
  --dataset mvtec_ad \
  --categories toothbrush \
  --train-batch-size 16 \
  --eval-batch-size 16 \
  --num-workers 4 \
  --coreset-sampling-ratio 0.01
