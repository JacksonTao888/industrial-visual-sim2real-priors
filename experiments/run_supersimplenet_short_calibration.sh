#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

OUT_ROOT="experiments/runs/no_cad/supersimplenet_short_calibration"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

conda run --no-capture-output -n sim2real python experiments/run_supersimplenet_anomalib.py \
  --dataset mvtec_ad \
  --categories toothbrush \
  --out-root "${OUT_ROOT}" \
  --train-batch-size 32 \
  --eval-batch-size 32 \
  --num-workers 0 \
  --max-epochs 999 \
  --max-steps 100 \
  2>&1 | tee "${LOG_DIR}/mvtec_toothbrush_100steps.log"

python experiments/summarize_supersimplenet_results.py \
  --dataset mvtec_ad \
  --result-dir "${OUT_ROOT}/mvtec_ad" \
  --csv "${OUT_ROOT}/mvtec_ad/supersimplenet_results.csv" \
  --summary "${OUT_ROOT}/mvtec_ad/supersimplenet_results.summary.json"

conda run --no-capture-output -n sim2real python experiments/run_supersimplenet_anomalib.py \
  --dataset visa \
  --categories macaroni2 capsules \
  --out-root "${OUT_ROOT}" \
  --train-batch-size 32 \
  --eval-batch-size 32 \
  --num-workers 0 \
  --max-epochs 999 \
  --max-steps 100 \
  2>&1 | tee "${LOG_DIR}/visa_macaroni2_capsules_100steps.log"

python experiments/summarize_supersimplenet_results.py \
  --dataset visa \
  --result-dir "${OUT_ROOT}/visa" \
  --csv "${OUT_ROOT}/visa/supersimplenet_results.csv" \
  --summary "${OUT_ROOT}/visa/supersimplenet_results.summary.json"
