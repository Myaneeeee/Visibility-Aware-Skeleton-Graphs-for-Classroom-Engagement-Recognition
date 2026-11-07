#!/usr/bin/env bash
# Multi-seed ablation sweep: configs A/B/C x seeds 51-60.
# One log per run; the per-fold numbers are parsed back out of these logs.
#
# Usage:  bash steps/run_sweep.sh [loss]
#   loss = ce (default) -- the pre-registered ladder, PREREGISTRATION.md Sec. 5b.3.
#          weighted     -- off-ladder imbalance arm, PREREGISTRATION.md Sec. 5b.8. Labels <cfg>+w_seed<n>.
#          focal        -- same arm with the gamma=2 focal term. Labels <cfg>+f_seed<n>.
# The suffix keeps a weighted run from ever overwriting the unweighted one it is
# meant to be compared against, and keeps 'ce' log/run names byte-identical to
# what produced the reported results.
set -u
cd "$(dirname "$0")/.."          # repo root, wherever it is checked out
PY=".venv/bin/python -u"    # -u so a run's log streams instead of flushing at exit
OUT=outputs/logs
mkdir -p "$OUT"
# One sweep at a time. A second launch would fight the first for the GPU and
# overwrite its per-run logs, and setsid makes that easy to do by accident.
exec 200> "$OUT/.sweep.lock"
flock -n 200 || { echo "a sweep is already running -- refusing to start"; exit 3; }
# setsid, so the sweep survives the terminal that launched it. Two overnight runs
# were lost to the shell going away mid-run before this was made explicit here.
# Launch with:  setsid nohup bash steps/run_sweep.sh >> outputs/logs/driver.log 2>&1 &

LOSS="${1:-ce}"
case "$LOSS" in
  ce)       SUFFIX="" ;;
  weighted) SUFFIX="+w" ;;
  focal)    SUFFIX="+f" ;;
  *) echo "unknown loss '$LOSS'; expected ce, weighted or focal" >&2; exit 2 ;;
esac

echo "=== SWEEP loss=$LOSS suffix='${SUFFIX:-none}' $(date +%F' '%H:%M:%S) ==="
for cfg in A B C; do
  for seed in 51 52 53 54 55 56 57 58 59 60; do
    label="${cfg}${SUFFIX}_seed${seed}"
    log="$OUT/${label}.log"
    if [ -s "$log" ] && grep -q "Model: majority" "$log"; then
      echo "SKIP ${label} (already complete)"
      continue
    fi
    echo "RUN  ${label} $(date +%H:%M:%S)"
    $PY steps/evaluate.py --config "$cfg" --seed "$seed" --loss "$LOSS" \
        > "$log" 2> "$OUT/${label}.err"
    if [ $? -ne 0 ]; then
      echo "FAIL ${label} -- see $OUT/${label}.err"
    fi
  done
done
echo "SWEEP DONE loss=$LOSS $(date +%H:%M:%S)"
