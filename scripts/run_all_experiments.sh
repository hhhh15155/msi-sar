#!/usr/bin/env bash
# Run the complete YRD, YRD2509, and YRD2509NEW few-shot grid.
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$(conda run -n gjc which python)}"
GPU_COUNT="${GPU_COUNT:-2}"
PROCS_PER_GPU="${PROCS_PER_GPU:-2}"
MAX_PROCS=$((GPU_COUNT * PROCS_PER_GPU))
LOG_DIR="$ROOT/experiment_logs"
mkdir -p "$LOG_DIR"

QUEUE=()
add_exp() { QUEUE+=("$1|$2|$3"); }

for model in dfinet frekfuse mghofnet softformer; do
    for shot in 5 10 20 50 100 150 200; do
        for dataset in yrd yrd2509 yrd2509new; do
            add_exp \
                "scripts/train_eval_${model}.py" \
                "configs/${model}_${dataset}_fs${shot}.yaml" \
                "${model}_${dataset}_fs${shot}"
        done
    done
done

TOTAL=${#QUEUE[@]}
echo "=============================================="
echo "  Experiments : ${TOTAL}"
echo "  Concurrency : ${MAX_PROCS} (${GPU_COUNT} GPUs x ${PROCS_PER_GPU} tasks)"
echo "  Logs        : ${LOG_DIR}"
echo "=============================================="

is_done() {
    local config="$1"
    if [[ "$config" =~ configs/(.+)_(yrd2509new|yrd2509|yrd)_fs([0-9]+)\.yaml ]]; then
        local model="${BASH_REMATCH[1]}"
        local dataset="${BASH_REMATCH[2]}"
        local shot="${BASH_REMATCH[3]}"
        [[ -f "$ROOT/runs_fewshot/fs${shot}/${model}/${dataset}/run_001/metrics.json" ]] && return 0
    fi
    return 1
}

run_one() {
    local gpu_id="$1"
    local script="$2"
    local config="$3"
    local name="$4"
    local logfile="$LOG_DIR/${name}_gpu${gpu_id}.log"
    echo "[$(date '+%H:%M:%S')] START ${name} on GPU${gpu_id}"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${gpu_id}" \
        "$PYTHON" "$script" --config "$config" >"$logfile" 2>&1
}

DONE=0
SKIPPED=0
FAILED=0
CURRENT=0
declare -a GPU_LOAD=()
declare -A PID_GPU
declare -A PID_NAME
for ((i=0; i<GPU_COUNT; i++)); do GPU_LOAD[i]=0; done

cleanup() {
    echo
    echo "Interrupted. Stopping ${#PID_GPU[@]} tasks..."
    for pid in "${!PID_GPU[@]}"; do kill "$pid" 2>/dev/null || true; done
    echo "Done: ${DONE}, Skipped: ${SKIPPED}, Failed: ${FAILED}"
    exit 130
}
trap cleanup SIGINT SIGTERM

while true; do
    for pid in "${!PID_GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            if wait "$pid" 2>/dev/null; then
                rc=0
            else
                rc=$?
            fi
            gpu="${PID_GPU[$pid]}"
            name="${PID_NAME[$pid]}"
            GPU_LOAD[$gpu]=$((GPU_LOAD[$gpu] - 1))
            if [[ $rc -eq 0 ]]; then
                DONE=$((DONE + 1))
                echo "[$(date '+%H:%M:%S')] DONE  ${name}"
            else
                FAILED=$((FAILED + 1))
                echo "[$(date '+%H:%M:%S')] FAIL  ${name} (rc=${rc})"
            fi
            unset "PID_GPU[$pid]"
            unset "PID_NAME[$pid]"
        fi
    done

    active=${#PID_GPU[@]}
    while [[ $active -lt $MAX_PROCS && $CURRENT -lt $TOTAL ]]; do
        IFS='|' read -r script config name <<<"${QUEUE[$CURRENT]}"
        CURRENT=$((CURRENT + 1))
        if is_done "$config"; then
            echo "[$(date '+%H:%M:%S')] SKIP ${name} (already done)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        best_gpu=0
        for ((i=1; i<GPU_COUNT; i++)); do
            [[ ${GPU_LOAD[$i]} -lt ${GPU_LOAD[$best_gpu]} ]] && best_gpu=$i
        done
        GPU_LOAD[$best_gpu]=$((GPU_LOAD[$best_gpu] + 1))
        run_one "$best_gpu" "$script" "$config" "$name" &
        pid=$!
        PID_GPU[$pid]="$best_gpu"
        PID_NAME[$pid]="$name"
        active=$((active + 1))
    done

    [[ ${#PID_GPU[@]} -eq 0 && $CURRENT -ge $TOTAL ]] && break
    sleep 3
done

echo
echo "=============================================="
echo "  COMPLETE"
echo "  Total:   ${TOTAL}"
echo "  Done:    ${DONE}"
echo "  Skipped: ${SKIPPED}"
echo "  Failed:  ${FAILED}"
echo "=============================================="
[[ $FAILED -gt 0 ]] && exit 1
exit 0
