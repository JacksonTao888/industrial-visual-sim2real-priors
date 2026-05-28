#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BUDGETS=("0.05:budget_005" "0.10:budget_010" "0.25:budget_025")
MVTEC_CATEGORIES=(toothbrush capsule cable)
VISA_CATEGORIES=(macaroni2 capsules pcb1)

for item in "${BUDGETS[@]}"; do
  ratio="${item%%:*}"
  tag="${item##*:}"

  python experiments/run_patchcore_anomalib.py \
    --dataset mvtec_ad \
    --categories "${MVTEC_CATEGORIES[@]}" \
    --train-budget-ratio "$ratio" \
    --out-root "experiments/runs/no_cad/budget_ablation/patchcore/${tag}" \
    --train-batch-size 32 \
    --eval-batch-size 32 \
    --num-workers 8

  python experiments/run_patchcore_anomalib.py \
    --dataset visa \
    --categories "${VISA_CATEGORIES[@]}" \
    --train-budget-ratio "$ratio" \
    --out-root "experiments/runs/no_cad/budget_ablation/patchcore/${tag}" \
    --train-batch-size 32 \
    --eval-batch-size 32 \
    --num-workers 8
done
