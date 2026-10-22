#!/usr/bin/env bash
# Full 60-layer expert conversions for several weighting variants, one after another on all GPUs.
# usage: [WAIT_PID=pid] [GPUS=0,..] [VARIANTS=dir] [RECIPE=file] [STATS=dir] run_variants.sh name:mode:gain:books[:col_power] ...
set -uo pipefail
TOOLS=$(cd "$(dirname "$0")" && pwd)
V=${VARIANTS:-/workspace/axk2/variants}
RECIPE=${RECIPE:-/workspace/axk2/recipe/recipe.json}
STATS=${STATS:-/workspace/axk2/stats}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
run() {
  local name=$1 mode=$2 gain=$3 books=$4 power=${5:-1.0} rc
  mkdir -p "$V/$name/parts" "$V/$name/work"
  rm -f "$V/$name/FAILED"
  echo "{\"variant\": \"$name\", \"col_power\": $power, \"start\": \"$(date -u +%FT%TZ)\"}"
  python3 "$TOOLS/run_recipe.py" --recipe "$RECIPE" --stats "$STATS" \
    --work "$V/$name/work" --output "$V/$name/parts" --mode "$mode" --gain-refit "$gain" --books "$books" \
    --col-power "$power" --gpus "$GPUS" > "$V/$name/run.log" 2>&1
  rc=$?
  # eval_variants.sh stops waiting for a variant whose conversion failed
  [ "$rc" = 0 ] || echo "$rc" > "$V/$name/FAILED"
  echo "{\"variant\": \"$name\", \"exit\": $rc, \"end\": \"$(date -u +%FT%TZ)\", \"parts\": $(ls "$V/$name/parts/"*.gguf 2>/dev/null | wc -l)}"
  return "$rc"
}
if [ -n "${WAIT_PID:-}" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 5; done; fi
fail=0
for spec in "$@"; do run $(echo "$spec" | tr ':' ' ') || fail=1; done
exit $fail
