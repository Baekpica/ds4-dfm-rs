#!/usr/bin/env bash
set -euo pipefail
cd /home/sunghoon/orca/workspaces/ds4-dfm-rs/feat-perf-mimo-26
probe_dir="$PWD/scratch/mimo-prefill-20260926/ncu-swiglu"
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu,power.draw,temperature.gpu --format=csv -l 1 > "$probe_dir/clocks.csv" &
clock_pid=$!
trap 'kill "$clock_pid" 2>/dev/null || true' EXIT
for sample in 0 1 2; do
    case "$sample" in
        0) arms="original 128 256 512" ;;
        1) arms="512 256 original 128" ;;
        2) arms="128 512 256 original" ;;
    esac
    for arm in $arms; do
        binary="$probe_dir/probe-blocks"
        args=(100 "$arm")
        if [ "$arm" = original ]; then binary="$probe_dir/probe-baseline"; args=(100); fi
        unset PROBE_DUMP
        if [ "$sample" = 0 ]; then export PROBE_DUMP="$probe_dir/$arm.bin"; fi
        prefix="$probe_dir/$sample-$arm"
        python3 tools/host_memory_guard.py --max-gib 8 --high-gib 6 --reserve-gib 12 --timeout 180 --log "$prefix.memory.jsonl" -- "$binary" "${args[@]}" > "$prefix.log" 2>&1
        rg 'ms=' "$prefix.log"
    done
done
for arm in 128 256 512; do cmp "$probe_dir/original.bin" "$probe_dir/$arm.bin"; done
