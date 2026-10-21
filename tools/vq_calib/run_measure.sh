#!/bin/bash
# Held-out measurement of weighted-Lloyd variants, one layer per GPU in parallel.
# usage: run_measure.sh "1 4 8 15 30 45 55 60" "0 1 2 3 4 5 6 7" [OUT [STATS]]
set -u
LAYERS=($1); GPUS=($2)
TOOLS=$(cd "$(dirname "$0")" && pwd)
OUT=${3:-/workspace/axk2/work/measure}
STATS=${4:-/workspace/axk2/stats}
mkdir -p "$OUT/logs"
V="none:none,none:ls,none:none:32:1:0.02:256:split2,none:none:32:1:0.02:256:split3"
V+=",imatrix:ls:32:0.75:0.02:256:shared,imatrix:ls:32:0.75:0.02:256:split2,imatrix:ls:32:0.75:0.02:256:split3"
V+=",imatrix:ls:32:1:0.02:256:split3,imatrix:ls:32:0.5:0.02:256:split3"
V+=",imatrix_freq:ls:32:1:0.02:256:split3,imatrix_freq:ls:32:0.75:0.02:256:split3,imatrix_freq:ls:32:0.75:0.02:256:split2"
V+=",output:ls:32:1:0.02:256:split3,output:ls:32:0.75:0.02:256:split3"
for i in "${!LAYERS[@]}"; do
  L=${LAYERS[$i]}; G=${GPUS[$((i % ${#GPUS[@]}))]}
  CUDA_VISIBLE_DEVICES=$G nohup python3 $TOOLS/measure_layer.py --layer $L --variants "$V" --iterations 32 --stats "$STATS" \
     --out $OUT > $OUT/logs/layer-$(printf %02d $L)-gpu$G.log 2>&1 &
done
wait
