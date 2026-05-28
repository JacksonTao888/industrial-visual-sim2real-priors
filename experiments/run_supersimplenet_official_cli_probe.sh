#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

OUT_ROOT="experiments/runs/no_cad/supersimplenet_official_cli/probe"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

conda run --no-capture-output -n sim2real anomalib train \
  --model anomalib.models.Supersimplenet \
  --data experiments.run_draem_anomalib.MVTecADStringSplit \
  --data.root datasets/mvtec_ad \
  --data.category toothbrush \
  --data.train_batch_size 32 \
  --data.eval_batch_size 32 \
  --data.num_workers 4 \
  --trainer.max_epochs 300 \
  --default_root_dir "${OUT_ROOT}/mvtec_ad/toothbrush" \
  2>&1 | tee "${LOG_DIR}/mvtec_toothbrush_300epochs.log"

conda run --no-capture-output -n sim2real anomalib train \
  --model anomalib.models.Supersimplenet \
  --data experiments.run_draem_anomalib.VisaStringSplit \
  --data.root datasets/visa \
  --data.category macaroni2 \
  --data.train_batch_size 32 \
  --data.eval_batch_size 32 \
  --data.num_workers 4 \
  --trainer.max_epochs 300 \
  --default_root_dir "${OUT_ROOT}/visa/macaroni2" \
  2>&1 | tee "${LOG_DIR}/visa_macaroni2_300epochs.log"

conda run --no-capture-output -n sim2real anomalib train \
  --model anomalib.models.Supersimplenet \
  --data experiments.run_draem_anomalib.VisaStringSplit \
  --data.root datasets/visa \
  --data.category capsules \
  --data.train_batch_size 32 \
  --data.eval_batch_size 32 \
  --data.num_workers 4 \
  --trainer.max_epochs 300 \
  --default_root_dir "${OUT_ROOT}/visa/capsules" \
  2>&1 | tee "${LOG_DIR}/visa_capsules_300epochs.log"
