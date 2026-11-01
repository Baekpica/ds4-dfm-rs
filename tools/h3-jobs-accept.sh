#!/bin/bash
# Does the resident route render the same clip as the one-shot route?
#
# One request, three renders at one seed:
#
#   1. the same request twice in one process (`h3-rs --jobs`): the second must
#      hit the prompt encoder and the prepared DiT, and must print the first
#      one's frames hash;
#   2. the one-shot route, `h3-rs --hash`, which loads the weights per clip.
#
# PASS needs all of it: job 2's text and DiT loads reported as hits, job 2's
# frames hash equal to job 1's, and both equal to the one-shot run's.
#
# Artifacts come from the environment, never from a path here:
#   H3_ARTIFACTS    the directory holding the checkpoints (required)
#   H3_ROOT         the repository checkout (default: the current directory)
#   H3_VAE_GEMM, H3_VAE_ATTENTION
#                   the decoder arms, when the operator states them; every run
#                   inherits them, so the comparison stays at one arm
#
# Usage: tools/h3-jobs-accept.sh [-p PROMPT] [--seed N] [--steps N]
#          [--width N --height N --frames N]
set -u

ROOT=${H3_ROOT:-$PWD}
A=${H3_ARTIFACTS:?H3_ARTIFACTS must name the checkpoint directory}
MODEL=$A/MiniMax-H3-FastH3-4step
NVFP4=$A/Comfy-MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors

PROMPT="A red fox trots through fresh snow at dawn, side profile, one tail, eye level."
SEED=42
STEPS=4
WIDTH=672
HEIGHT=384
FRAMES=124

while [ $# -gt 0 ]; do
  case "$1" in
    -p) PROMPT=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --steps) STEPS=$2; shift 2 ;;
    --width) WIDTH=$2; shift 2 ;;
    --height) HEIGHT=$2; shift 2 ;;
    --frames) FRAMES=$2; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for artifact in "$MODEL" "$NVFP4"; do
  [ -e "$artifact" ] || { echo "missing artifact: $artifact" >&2; exit 3; }
done

cd "$ROOT" || exit 1
mkdir -p outputs scratch

# The release's draft, as the clip tool sets it.
export H3_DIT_SOL=0.9

# One request per line: prompt, seed, steps, width, height, frames, output.
printf '%s\t%s\t%s\t%s\t%s\t%s\toutputs/resident-a.mp4\n' \
  "$PROMPT" "$SEED" "$STEPS" "$WIDTH" "$HEIGHT" "$FRAMES" > scratch/jobs.txt
printf '%s\t%s\t%s\t%s\t%s\t%s\toutputs/resident-b.mp4\n' \
  "$PROMPT" "$SEED" "$STEPS" "$WIDTH" "$HEIGHT" "$FRAMES" >> scratch/jobs.txt

guard() {
  local log=$1
  shift
  python3 tools/host_memory_guard.py --max-gib 120 --high-gib 120 \
    --reserve-gib 2 --trip-gib 2 --psi-full-limit 0 --timeout 3600 \
    --log "scratch/$log.memory.jsonl" -- "$@"
}

# The seconds since `$1`, so both lifetimes carry a comparable number.
since() {
  awk -v a="$1" -v b="$(date +%s.%N)" 'BEGIN {printf "%.3f", b-a}'
}

echo "== resident: two requests, one process"
started=$(date +%s.%N)
guard resident "$ROOT/h3-rs" -d "$MODEL" --text-nvfp4 "$NVFP4" \
  --jobs scratch/jobs.txt 2>&1 | tee scratch/resident.out
[ "${PIPESTATUS[0]}" = "0" ] || exit 1
resident_seconds=$(since "$started")
echo "WALL stage=resident seconds=$resident_seconds"

echo "== one-shot: the weights loaded per clip"
started=$(date +%s.%N)
guard accept-oneshot "$ROOT/h3-rs" -d "$MODEL" --text-nvfp4 "$NVFP4" \
  -p "$PROMPT" --width "$WIDTH" --height "$HEIGHT" --frames "$FRAMES" \
  --steps "$STEPS" --seed "$SEED" --hash -o outputs/accept-oneshot.mp4 \
  2>&1 | tee scratch/accept-oneshot.out
[ "${PIPESTATUS[0]}" = "0" ] || exit 1
oneshot_seconds=$(since "$started")
echo "WALL stage=one-shot seconds=$oneshot_seconds"
echo "WALL resident-two-clips=$resident_seconds one-shot=$oneshot_seconds"

field() {
  # field <line> <name>: the `name=value` word's value.
  grep -o "$2=[^ ]*" <<<"$1" | head -1 | cut -d= -f2
}

line1=$(grep '^JOB n=1 ' scratch/resident.out)
line2=$(grep '^JOB n=2 ' scratch/resident.out)
oneshot=$(grep -o 'HASH [^ ]* audio=[0-9a-f]* frames=[0-9a-f]*' scratch/accept-oneshot.out |
  grep -o 'frames=[0-9a-f]*' | cut -d= -f2)

first=$(field "$line1" frames_fnv)
second=$(field "$line2" frames_fnv)
status=0

echo "frames job1=$first job2=$second one-shot=$oneshot"

for marker in ' text=[0-9.]*s \(hit\)' ' load=[0-9.]*s \(hit\)'; do
  grep -Eq "$marker" <<<"$line2" || {
    echo "FAIL job 2 did not report $marker"
    status=1
  }
done

[ -n "$first" ] && [ "$first" = "$second" ] || {
  echo "FAIL job 1 and job 2 differ"
  status=1
}
[ -n "$first" ] && [ "$first" = "$oneshot" ] || {
  echo "FAIL the resident route and the one-shot route differ"
  status=1
}

if [ "$status" = "0" ]; then
  echo "RESULT PASS resident product route"
else
  echo "RESULT FAIL"
fi
exit "$status"
