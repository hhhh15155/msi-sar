#!/usr/bin/env bash
# Run the default YRD/GRSS07 grids or an explicit DATASETS_FILTER subset.
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$(conda run -n gjc which python)}"
GPU_COUNT="${GPU_COUNT:-2}"
PROCS_PER_GPU="${PROCS_PER_GPU:-2}"
MAX_PROCS=$((GPU_COUNT * PROCS_PER_GPU))
LOG_DIR="$ROOT/experiment_logs"
mkdir -p "$LOG_DIR"

GRID_ARGS=(--root "$ROOT")
if [[ -n "${DATASETS_FILTER:-}" ]]; then
    read -r -a FILTERED_DATASETS <<<"$DATASETS_FILTER"
    GRID_ARGS+=(--datasets "${FILTERED_DATASETS[@]}")
fi

if ! GRID_OUTPUT="$("$PYTHON" scripts/experiment_grid.py "${GRID_ARGS[@]}")"; then
    echo "Failed to build experiment grid." >&2
    exit 2
fi
mapfile -t QUEUE <<<"$GRID_OUTPUT"

TOTAL=${#QUEUE[@]}
echo "=============================================="
echo "  Experiments : ${TOTAL}"
echo "  Concurrency : ${MAX_PROCS} (${GPU_COUNT} GPUs x ${PROCS_PER_GPU} tasks)"
echo "  Logs        : ${LOG_DIR}"
echo "=============================================="

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    SKIPPED=0
    PENDING=0
    for entry in "${QUEUE[@]}"; do
        IFS='|' read -r _script _config name already_done <<<"$entry"
        if [[ "$already_done" == "1" ]]; then
            echo "SKIP      ${name} (already done)"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "WOULD RUN ${name}"
            PENDING=$((PENDING + 1))
        fi
    done
    echo "Dry run: ${TOTAL} total, ${SKIPPED} complete, ${PENDING} pending"
    exit 0
fi

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
        IFS='|' read -r script config name already_done <<<"${QUEUE[$CURRENT]}"
        CURRENT=$((CURRENT + 1))
        if [[ "$already_done" == "1" ]]; then
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
