#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

conda run --no-capture-output -n sim2real python experiments/run_winclip_anomalib.py \
  --dataset mvtec_ad \
  --categories toothbrush \
  --eval-batch-size 8 \
  --num-workers 4 \
  --k-shot 0
