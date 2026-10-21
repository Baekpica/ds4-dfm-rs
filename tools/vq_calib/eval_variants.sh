#!/usr/bin/env bash
# Wait for each variant's 60 expert parts, then run eval_e2e (experts = parts, non-experts = teacher).
# usage: [VARIANTS=dir] [SOURCE=dir] [STATS=dir] [MANIFEST=file] eval_variants.sh name ...
set -uo pipefail
TOOLS=$(cd "$(dirname "$0")" && pwd)
V=${VARIANTS:-/workspace/axk2/variants}
SOURCE=${SOURCE:-/models/A.X-K2}
STATS=${STATS:-/workspace/axk2/stats}
MANIFEST=${MANIFEST:-/workspace/axk2/calib/frozen/heldout-81681f74_manifest.json}
i=0
for name in "$@"; do
  until [ "$(ls $V/$name/parts/layer-*-experts.json 2>/dev/null | wc -l)" -ge 60 ]; do sleep 10; done
  gpu=$(( i % 8 )); i=$((i+1))
  CUDA_VISIBLE_DEVICES=$gpu python3 $TOOLS/eval_e2e.py --source $SOURCE --stats $STATS --parts $V/$name/parts \
    --experts parts --nonexpert teacher --manifest $MANIFEST --out $V/$name/eval-experts \
    > $V/$name/eval-experts.log 2>&1 &
  echo "{\"eval\": \"$name\", \"gpu\": $gpu, \"start\": \"$(date -u +%FT%TZ)\"}"
done
wait
echo '{"evals": "done"}'
