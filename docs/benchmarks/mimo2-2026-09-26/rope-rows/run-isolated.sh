#!/usr/bin/env bash
set -euo pipefail
cd /home/sunghoon/orca/workspaces/ds4-dfm-rs/feat-perf-mimo-26
probe_dir="$PWD/scratch/mimo-prefill-20260926/ncu-rope"
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu,power.draw,temperature.gpu --format=csv -l 1 > "$probe_dir/clocks.csv" &
clock_pid=$!
trap 'kill "$clock_pid" 2>/dev/null || true' EXIT
for sample in 0 1 2; do
    case "$sample" in
        0) arms="original off on" ;;
        1) arms="on original off" ;;
        2) arms="off on original" ;;
    esac
    for heads in 4 8; do
        for arm in $arms; do
            export DS4_MIMO2_ROPE_ROWS=0
            binary="$probe_dir/probe-candidate"
            if [ "$arm" = original ]; then binary="$probe_dir/probe-original"; fi
            if [ "$arm" = on ]; then unset DS4_MIMO2_ROPE_ROWS; fi
            unset PROBE_DUMP
            if [ "$sample" = 0 ]; then export PROBE_DUMP="$probe_dir/$heads-$arm"; fi
            prefix="$probe_dir/$heads-$sample-$arm"
            python3 tools/host_memory_guard.py --max-gib 8 --high-gib 6 --reserve-gib 12 --timeout 180 --log "$prefix.memory.jsonl" -- "$binary" "$heads" 100 16 4096 > "$prefix.log" 2>&1
            rg 'kernel_ms=' "$prefix.log"
        done
    done
done
for heads in 4 8; do
    for arm in off on; do
        for tensor in q k v; do cmp "$probe_dir/$heads-original.$tensor.f32" "$probe_dir/$heads-$arm.$tensor.f32"; done
    done
done
