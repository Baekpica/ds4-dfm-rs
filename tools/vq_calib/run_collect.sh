#!/usr/bin/env bash
# Launch the calibration collector: one rank per GPU, then reduce.
# usage: run_collect.sh CALIB HELDOUT OUT "GPU_LIST" [extra collect_stats.py args]
set -euo pipefail
CALIB=$1; HELDOUT=$2; OUT=$3; GPUS=($4); shift 4
TOOLS=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/logs"
W=${#GPUS[@]}
pids=()
for r in $(seq 0 $((W-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$r]} python3 "$TOOLS/collect_stats.py" --rank "$r" --world "$W" \
    --calib "$CALIB" --heldout "$HELDOUT" --out "$OUT" "$@" > "$OUT/logs/rank-$r.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" = 0 ]; then python3 "$TOOLS/reduce_stats.py" --stats "$OUT"; else echo "a rank failed" >&2; exit 1; fi
