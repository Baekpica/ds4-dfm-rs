#!/bin/bash
# Prompt -> clip, one process, pure MiniMax-H3.
#
# h3-rs encodes the prompt, the DiT denoises the latent at the canvas you name,
# the H3 video VAE decodes it and the H3 audio VAE decodes the soundtrack; the
# mux writes the MP4. There is no upscaler, no adapter, no second model and no
# Python in the path.
#
# Usage:
#   tools/make-h3-clip.sh -p "prompt" [-o out.mp4] [-d model_dir]
#     [--width N --height N --frames N --steps N --seed N]
#
#   The defaults are the delivery canvas: 1344x768, 121 frames, the FastH3
#   distilled 4 steps. A 672x384 draft renders in well under two minutes;
#   the delivery canvas pays roughly six times the denoise and a long tiled
#   decode.
set -u

ROOT=${H3_ROOT:-$HOME/ds4-sol-h3}
A=${H3_ARTIFACTS:-/mnt/h3-artifacts}
MODEL=$A/MiniMax-H3-FastH3-4step
NVFP4=$A/Comfy-MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
PROMPT=""
OUT=$ROOT/outputs/clip.mp4
WIDTH=1344
HEIGHT=768
FRAMES=121
STEPS=4
SEED=42

while [ $# -gt 0 ]; do
  case "$1" in
    -p) PROMPT=$2; shift 2 ;;
    -o) OUT=$2; shift 2 ;;
    -d) MODEL=$2; shift 2 ;;
    --width) WIDTH=$2; shift 2 ;;
    --height) HEIGHT=$2; shift 2 ;;
    --frames) FRAMES=$2; shift 2 ;;
    --steps) STEPS=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$PROMPT" ]; then
  echo "missing -p prompt" >&2
  exit 2
fi

cd "$ROOT" || exit 1
mkdir -p "$(dirname "$OUT")" scratch

export H3_DIT_SOL=0.9
# The decoder's arms: the campaign's, unless the operator states its own. `bf16i`
# is the faster one and the decode line in the log names whichever ran.
: "${H3_VAE_ATTENTION:=tensor}"
: "${H3_VAE_GEMM:=bf16}"
export H3_VAE_ATTENTION H3_VAE_GEMM

guard() {
  local log=$1
  shift
  python3 tools/host_memory_guard.py --max-gib 120 --high-gib 120 \
    --reserve-gib 2 --trip-gib 2 --psi-full-limit 0 --timeout 3600 \
    --log "scratch/$log.memory.jsonl" -- "$@"
}

started=$(date +%s.%N)

guard clip "$ROOT/h3-rs" -d "$MODEL" --text-nvfp4 "$NVFP4" \
  -p "$PROMPT" --width "$WIDTH" --height "$HEIGHT" --frames "$FRAMES" \
  --steps "$STEPS" --seed "$SEED" -o "$OUT" || exit 1

ended=$(date +%s.%N)

awk -v a="$started" -v b="$ended" 'BEGIN {printf "WALL total=%.3f\n", b-a}'
ffprobe -v error -show_entries format=duration,size \
  -show_entries stream=codec_name,width,height,sample_rate,channels \
  -of default=noprint_wrappers=1 "$OUT"
ffmpeg -v error -i "$OUT" -f null - && echo DECODE_OK
sha256sum "$OUT"
echo "CLIP $OUT"
