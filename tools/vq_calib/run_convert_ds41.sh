#!/usr/bin/env bash
# DeepSeek-V4.1-Flash iVQ experts: convert_layer_ds41.py over 40 layers on the given GPUs.
# Layers are dealt longest-first (K=8192 layers 0-13 cost ~1.7x) to the least-loaded GPU.
# usage: run_convert_ds41.sh STATS OUT "GPU_LIST" [extra convert_layer_ds41.py args]
set -euo pipefail
STATS=$1; OUT=$2; GPUS=($3); shift 3
TOOLS=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/logs"
mapfile -t PLAN < <(python3 - "${#GPUS[@]}" <<'PY'
import sys
n = int(sys.argv[1])
jobs = sorted(range(40), key=lambda L: (-(1.7 if L < 14 else 1.0), L))
load, plan = [0.0] * n, [[] for _ in range(n)]
for L in jobs:
    g = min(range(n), key=lambda i: (load[i], i))
    load[g] += 1.7 if L < 14 else 1.0
    plan[g].append(L)
for p in plan:
    print(','.join(map(str, p)))
PY
)
pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} python3 "$TOOLS/convert_layer_ds41.py" --layers "${PLAN[$i]}" --stats "$STATS" \
    --out "$OUT" "$@" > "$OUT/logs/gpu-${GPUS[$i]}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
ls "$OUT"/blob-*.bin | wc -l
exit $fail
