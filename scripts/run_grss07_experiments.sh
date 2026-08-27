#!/usr/bin/env bash
# Run only the GRSS-DFC-2007 few-shot grid in the gjc environment.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export DATASETS_FILTER="grss07"
export GPU_COUNT="${GPU_COUNT:-1}"
export PROCS_PER_GPU="${PROCS_PER_GPU:-2}"
exec bash scripts/run_all_experiments.sh
