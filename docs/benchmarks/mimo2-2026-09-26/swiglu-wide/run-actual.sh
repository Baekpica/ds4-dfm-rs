#!/usr/bin/env bash
set -euo pipefail
cd /home/sunghoon/orca/workspaces/ds4-dfm-rs/feat-perf-mimo-26
campaign_dir="$PWD/scratch/mimo-prefill-20260926"
export DS4_CUDA_WEIGHT_IPC_MANIFEST="$campaign_dir/owner10.ipc"
model_path=$(python3 -c 'import json; print(json.load(open("scratch/mimo-prefill-20260926/workload.json"))["files"]["model"]["path"])')
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu,power.draw,temperature.gpu --format=csv -l 1 > "$campaign_dir/swiglu6-clocks.csv" &
clock_pid=$!
trap 'kill "$clock_pid" 2>/dev/null || true' EXIT
for sample in 0 1 2; do
    case "$sample" in
        0) arms="original off on" ;;
        1) arms="on original off" ;;
        2) arms="off on original" ;;
    esac
    for arm in $arms; do
        export DS4_MIMO2_SWIGLU_WIDE=0
        binary="$campaign_dir/bench-swiglu6-candidate"
        if [ "$arm" = original ]; then binary="$campaign_dir/bench-gateup3-candidate"; fi
        if [ "$arm" = on ]; then unset DS4_MIMO2_SWIGLU_WIDE; fi
        for phase in warm sample; do
            prefix="$campaign_dir/swiglu6-$sample-$arm-$phase"
            python3 tools/host_memory_guard.py --max-gib 24 --high-gib 21 --reserve-gib 12 --timeout 900 --log "$prefix.memory.jsonl" -- "$binary" --cuda -m "$model_path" --prompt-file speed-bench/promessi_sposi.txt --ctx-start 8192 --ctx-max 8192 --gen-tokens 128 --dump-frontier-logits-dir "$prefix-proof" > "$prefix.csv" 2> "$prefix.log"
            cat "$prefix.csv"
        done
    done
done
