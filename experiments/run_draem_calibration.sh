#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

OUT_ROOT="experiments/runs/no_cad/draem_calibration"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

conda run --no-capture-output -n sim2real python experiments/run_draem_anomalib.py \
  --dataset mvtec_ad \
  --categories toothbrush \
  --out-root "${OUT_ROOT}" \
  --train-batch-size 8 \
  --eval-batch-size 16 \
  --num-workers 4 \
  --max-epochs 100 \
  2>&1 | tee "${LOG_DIR}/mvtec_toothbrush_100epochs.log"

python experiments/summarize_draem_results.py \
  --dataset mvtec_ad \
  --result-dir "${OUT_ROOT}/mvtec_ad" \
  --csv "${OUT_ROOT}/mvtec_ad/draem_results.csv" \
  --summary "${OUT_ROOT}/mvtec_ad/draem_results.summary.json"

conda run --no-capture-output -n sim2real python experiments/run_draem_anomalib.py \
  --dataset visa \
  --categories macaroni2 capsules \
  --out-root "${OUT_ROOT}" \
  --train-batch-size 8 \
  --eval-batch-size 16 \
  --num-workers 4 \
  --max-epochs 999 \
  --max-steps 1000 \
  2>&1 | tee "${LOG_DIR}/visa_macaroni2_capsules_1000steps.log"

python experiments/summarize_draem_results.py \
  --dataset visa \
  --result-dir "${OUT_ROOT}/visa" \
  --csv "${OUT_ROOT}/visa/draem_results.csv" \
  --summary "${OUT_ROOT}/visa/draem_results.summary.json"
