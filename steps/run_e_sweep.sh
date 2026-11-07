#!/usr/bin/env bash
# Observability ladder sweep: E rungs x seeds 51-60, ST-GCN only.
#
# Usage:  bash steps/run_e_sweep.sh [config ...]        (default: E1)
#
# Only the ST-GCN moves on an E rung -- the graph is the treatment and the MLP/GRU
# baselines are the comparison -- so --models stgcn reruns nothing the matching
# C seed already has on disk. Seeds match C's so every rung is paired fold for fold.
# Seeds run in order, so seeds 51-53 are readable while the rest continue.
set -u
cd "$(dirname "$0")/.."          # repo root, wherever it is checked out
PY=".venv/bin/python -u"    # -u so a run's log streams instead of flushing at exit
OUT=outputs/logs
mkdir -p "$OUT"
# Own lock, separate from run_sweep.sh's: two sweeps would fight for the GPU.
exec 200> "$OUT/.e_sweep.lock"
flock -n 200 || { echo "an E sweep is already running -- refusing to start"; exit 3; }
# Launch with:
#   setsid nohup bash steps/run_e_sweep.sh >> outputs/logs/e_driver.log 2>&1 &

CONFIGS=("${@:-E1}")

echo "=== E SWEEP configs=${CONFIGS[*]} $(date +%F' '%H:%M:%S) ==="
for cfg in "${CONFIGS[@]}"; do
  for seed in 51 52 53 54 55 56 57 58 59 60; do
    label="${cfg}_seed${seed}"
    log="$OUT/${label}.log"
    if [ -s "$log" ] && grep -q "Model: majority" "$log"; then
      echo "SKIP ${label} (already complete)"
      continue
    fi
    echo "RUN  ${label} $(date +%H:%M:%S)"
    $PY steps/evaluate.py --config "$cfg" --seed "$seed" --models stgcn \
        > "$log" 2> "$OUT/${label}.err"
    if [ $? -ne 0 ]; then
      echo "FAIL ${label} -- see $OUT/${label}.err"
    fi
  done
done
echo "E SWEEP DONE configs=${CONFIGS[*]} $(date +%H:%M:%S)"
