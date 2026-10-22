#!/usr/bin/env bash
# DeepSeek-V4.1-Flash calibration statistics: one run_ds41.py rank per GPU, then reduce_stats.py.
# usage: run_collect_ds41.sh CALIB OUT "GPU_LIST" [extra run_ds41.py args]
set -euo pipefail
CALIB=$1; OUT=$2; GPUS=($3); shift 3
TOOLS=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/logs"
W=${#GPUS[@]}
pids=()
for r in $(seq 0 $((W-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$r]} python3 "$TOOLS/run_ds41.py" collect --rank "$r" --world "$W" \
    --data "$CALIB" --out "$OUT" "$@" > "$OUT/logs/rank-$r.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" = 0 ]; then python3 "$TOOLS/reduce_stats.py" --stats "$OUT"; else echo "a rank failed" >&2; exit 1; fi
