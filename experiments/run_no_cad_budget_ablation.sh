#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

conda run --no-capture-output -n sim2real bash experiments/run_patchcore_budget_ablation.sh
conda run --no-capture-output -n sim2real bash experiments/run_anomalydino_budget_ablation.sh
conda run --no-capture-output -n sim2real python experiments/summarize_no_cad_budget_ablation.py
