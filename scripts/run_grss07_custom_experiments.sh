#!/usr/bin/env bash
# Run the GRSS07 custom per-class training-count configs sequentially.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$(conda run -n gjc which python)}"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="$ROOT/experiment_logs"
mkdir -p "$LOG_DIR"

for model in dfinet frekfuse mghofnet msfmamba softformer vbenet; do
    config="configs/${model}_grss07_custom.yaml"
    logfile="$LOG_DIR/${model}_grss07_custom_gpu${GPU_ID}.log"
    echo "[$(date '+%H:%M:%S')] START ${model}_grss07_custom on GPU${GPU_ID}"
    CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
        "$PYTHON" "scripts/train_eval_${model}.py" --config "$config" >"$logfile" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE  ${model}_grss07_custom"
done
