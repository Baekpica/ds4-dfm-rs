#!/usr/bin/env bash
# h3-prompt.sh — one prompt in, one clip out, pure MiniMax-H3.
#
# A single process does the whole job: h3-rs encodes the prompt, the DiT
# denoises the latent, the H3 video VAE decodes it at the canvas you asked
# for, the H3 audio VAE decodes the soundtrack, and ffmpeg muxes the MP4.
# There is no second model, no upscaler, no adapter and no Python.
#
# usage: ./h3-prompt.sh [options] "your prompt"
#        ./h3-prompt.sh [options] -p prompt.txt
#
#   -p, --prompt-file F   read the prompt from a file instead of the command line
#   -i, --image PATH      a Ref2VA reference image, repeatable, in order; the
#                         prompt addresses them as <Picture 1>, <Picture 2>, ...
#       --name SLUG       name this run explicitly
#       --seed N          default 42
#       --steps N         default 4, the FastH3 distilled step count
#       --width N         default 1344, the canvas the old stage 2 delivered
#       --height N        default 768
#       --frames N        default 121, requested frames (aligned to 5 + 17k)
#       --crf N           x264 quality, default 19
#       --preset NAME     x264 preset, default veryfast; slow and veryslow keep
#                         far more fine texture for a longer encode
#       --vae-workers N   decode the video through N device contexts
#       --vae-tile N      the decoder's spatial tile in pixels, 256-512
#       --vae-gemm M      fp32 | bf16 | bf16i
#       --vae-attention A scalar | tensor
#       --out DIR         work directory (default outputs/prompt-STAMP-NAME)
#       --force           run even if another render holds the GPU
#   -h, --help            this text
#
# A 672x384 draft renders in well under two minutes; 1344x768 pays roughly
# six times the denoise and a long tiled decode, which is the price of
# drawing the delivery canvas natively instead of upscaling into it.
#
# env: H3_DIR (default ~/ds4-sol-h3, the checkout holding ./h3-rs),
#      ARTIFACTS (default /mnt/h3-artifacts), H3MODEL, NVFP4,
#      H3_VAE_ATTENTION, H3_VAE_GEMM
set -u -o pipefail

usage() { awk 'NR > 1 && /^set / { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"; }
die() { printf 'h3-prompt: %s\n' "$1" >&2; exit 1; }

H3_DIR=${H3_DIR:-$HOME/ds4-sol-h3}
ARTIFACTS=${ARTIFACTS:-/mnt/h3-artifacts}
H3MODEL=${H3MODEL:-$ARTIFACTS/MiniMax-H3-FastH3-4step}
NVFP4=${NVFP4:-$ARTIFACTS/Comfy-MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors}

PROMPT=""
PROMPTFILE=""
NAME=""
SEED=42
STEPS=4
WIDTH=1344
HEIGHT=768
FRAMES=121
CRF=19
PRESET=veryfast
VAE_WORKERS=""
VAE_TILE=""
VAE_GEMM=""
VAE_ATTENTION=""
OUTDIR=""
FORCE=0
IMAGES=()

while [ $# -gt 0 ]; do
  case "$1" in
    -p|--prompt-file) PROMPTFILE=$2; shift 2 ;;
    -i|--image)       IMAGES+=("$2"); shift 2 ;;
    --name)           NAME=$2; shift 2 ;;
    --seed)           SEED=$2; shift 2 ;;
    --steps)          STEPS=$2; shift 2 ;;
    --width)          WIDTH=$2; shift 2 ;;
    --height)         HEIGHT=$2; shift 2 ;;
    --frames)         FRAMES=$2; shift 2 ;;
    --crf)            CRF=$2; shift 2 ;;
    --preset)         PRESET=$2; shift 2 ;;
    --vae-workers)    VAE_WORKERS=$2; shift 2 ;;
    --vae-tile)       VAE_TILE=$2; shift 2 ;;
    --vae-gemm)       VAE_GEMM=$2; shift 2 ;;
    --vae-attention)  VAE_ATTENTION=$2; shift 2 ;;
    --out)            OUTDIR=$2; shift 2 ;;
    --force)          FORCE=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    -*)               usage; die "unknown option: $1" ;;
    *)                PROMPT=$1; shift ;;
  esac
done

[ -n "$PROMPTFILE" ] && {
  [ -f "$PROMPTFILE" ] || die "no such prompt file: $PROMPTFILE"
  PROMPT=$(cat "$PROMPTFILE")
}
[ -n "$PROMPT" ] || { usage; die "no prompt"; }

for img in ${IMAGES[@]+"${IMAGES[@]}"}; do
  [ -f "$img" ] || die "no such reference image: $img"
done

[ -d "$H3_DIR" ] || die "no checkout at $H3_DIR (set H3_DIR)"
[ -x "$H3_DIR/h3-rs" ] || die "no h3-rs binary in $H3_DIR (build it there)"
[ -d "$H3MODEL" ] || die "no model directory: $H3MODEL"
[ -f "$NVFP4" ] || die "no NVFP4 prompt encoder: $NVFP4"

if [ "$FORCE" -eq 0 ]; then
  busy=$(pgrep -a -f 'h3-rs' | grep -v "h3-prompt" || true)
  if [ -n "$busy" ]; then
    printf '%s\n' "$busy" >&2
    die "a render is already on the GPU; wait for it, or pass --force"
  fi
fi

STAMP=$(date +%Y%m%d-%H%M%S)
SLUG=${NAME:-clip}
WORK=${OUTDIR:-$H3_DIR/outputs/prompt-$STAMP-$SLUG}
mkdir -p "$WORK" || die "cannot create $WORK"

LOG=$WORK/run.log
printf '%s\n' "$PROMPT" > "$WORK/prompt.txt"

{
  printf 'canvas     %sx%s, %s frames, %s steps, seed %s\n' "$WIDTH" "$HEIGHT" "$FRAMES" "$STEPS" "$SEED"
  printf 'prompt     %s\n' "$(printf '%s' "$PROMPT" | cut -c1-68)$( [ "${#PROMPT}" -gt 68 ] && printf ' [...]' )"
  printf 'prompt.txt %s words, full text beside the clip\n' "$(printf '%s' "$PROMPT" | wc -w)"
  printf 'encode     crf %s, preset %s\n' "$CRF" "$PRESET"
  if [ "${#IMAGES[@]}" -gt 0 ]; then
    n=0
    for img in "${IMAGES[@]}"; do
      n=$((n + 1))
      printf 'image %-2s   %s (%s)\n' "$n" "$img" "$(sha256sum "$img" | cut -c1-16)"
    done
  fi
  printf 'work       %s\n\n' "$WORK"
} | tee -a "$LOG"

cd "$H3_DIR" || die "cannot enter $H3_DIR"

ARGS=(-d "$H3MODEL" --text-nvfp4 "$NVFP4" -p "$PROMPT"
      --width "$WIDTH" --height "$HEIGHT" --frames "$FRAMES"
      --steps "$STEPS" --seed "$SEED"
      --crf "$CRF" --preset "$PRESET"
      -o "$WORK/clip.mp4")
[ -n "$VAE_WORKERS" ]   && ARGS+=(--vae-workers "$VAE_WORKERS")
[ -n "$VAE_TILE" ]      && ARGS+=(--vae-tile "$VAE_TILE")
[ -n "$VAE_GEMM" ]      && ARGS+=(--vae-gemm "$VAE_GEMM")
[ -n "$VAE_ATTENTION" ] && ARGS+=(--vae-attention "$VAE_ATTENTION")
for img in ${IMAGES[@]+"${IMAGES[@]}"}; do
  ARGS+=(--ref-image "$img")
done

STARTED=$(date +%s)
./h3-rs "${ARGS[@]}" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
ELAPSED=$(( $(date +%s) - STARTED ))

if [ "$status" -ne 0 ]; then
  die "h3-rs exited $status after ${ELAPSED}s; log: $LOG"
fi

[ -f "$WORK/clip.mp4" ] || die "no clip was written; log: $LOG"

printf '\nclip   %s\n' "$WORK/clip.mp4"
printf 'size   %s bytes\n' "$(stat -c %s "$WORK/clip.mp4")"
printf 'wall   %ss\n' "$ELAPSED"
