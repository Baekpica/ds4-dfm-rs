#!/usr/bin/env bash
# h3_perf_sweep — one stamped row per render: identity, shape, phase seconds.
#
# Every measurement keeps the things that make two runs comparable later: the
# binary hash, the git revision, the driver and clock, the checkpoint path and
# size, the page-cache level, and the phase lines the engine printed. A row is
# the unit of evidence; a wall clock alone is not.
#
#   tests/h3_perf_sweep.sh --engine port --out perf/h3.csv \
#     --width 512 --height 512 --frames 56 --steps 8 --anchor ~/h3-api/out/video_2.first.jpg \
#     --prompt "a man rides a crocodile in a lake"
#
#   tests/h3_perf_sweep.sh --engine reference --out perf/h3.csv --same ...
#
# Options:
#   --engine port|reference   h3-rs (this repo) or the reference h3 binary
#   --binary PATH             override the resolved binary
#   --model-dir DIR           checkpoint root (default /mnt/h3-artifacts/MiniMax-H3)
#   --out CSV                 append results here (default tests/h3/perf/perf.csv)
#   --width W --height H      canvas (default 512 512)
#   --frames F                frames (default 56)
#   --steps N                 denoise evaluations (default 8)
#   --layers N                DiT blocks (default: binary default, 50)
#   --seed N                  seed (default 7)
#   --anchor FILE             first-frame anchor
#   --prompt TEXT             prompt (default a fixed perf prompt)
#   --repeats N               how many runs (default 1)
#   --label TEXT              free-form tag in the row
#   --keep-dir DIR            where each run's MP4 and log land
#                             (default: tests/h3/perf/runs)
#   --profile-phases          reference only: pass --profile and H3_PROFILE=1
#                             to emit phase marks. That path synchronizes the
#                             stream, so it is not a timing run (see below).
#   --dry-run                 print the command, run nothing
#
# The reference prints progress with \r and its phase marks need H3_PROFILE=1;
# this script sets it and parses whatever "... seconds=" lines appear, so a
# missing phase is an empty cell, never a zero.
set -uo pipefail

# Everything this script writes stays under the repository it lives in.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

ENGINE=port
BINARY=""
MODEL_DIR=${H3_MODEL_DIR:-/mnt/h3-artifacts/MiniMax-H3}
OUT=""
WIDTH=512
HEIGHT=512
FRAMES=56
STEPS=8
LAYERS=""
SEED=7
ANCHOR=""
PROMPT="a man rides a crocodile in a lake, wind and water"
REPEATS=1
LABEL=""
KEEP_DIR=""
PROFILE_PHASES=0
DRY_RUN=0

usage() { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; }
fail() { echo "h3_perf_sweep: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --engine) ENGINE=$2; shift 2 ;;
    --binary) BINARY=$2; shift 2 ;;
    --model-dir) MODEL_DIR=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --width) WIDTH=$2; shift 2 ;;
    --height) HEIGHT=$2; shift 2 ;;
    --frames) FRAMES=$2; shift 2 ;;
    --steps) STEPS=$2; shift 2 ;;
    --layers) LAYERS=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --anchor) ANCHOR=$2; shift 2 ;;
    --prompt) PROMPT=$2; shift 2 ;;
    --repeats) REPEATS=$2; shift 2 ;;
    --label) LABEL=$2; shift 2 ;;
    --keep-dir) KEEP_DIR=$2; shift 2 ;;
    --profile-phases) PROFILE_PHASES=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) fail "unknown option $1 (try --help)" ;;
  esac
done

[ -n "$OUT" ] || OUT="$REPO_ROOT/tests/h3/perf/perf.csv"

# Runs are evidence, so they stay inside the repository beside their CSV
# rather than in a temporary directory: the MP4 and the engine log of every
# run remain readable, and the next run can be compared with the same files.
[ -n "$KEEP_DIR" ] || KEEP_DIR="$REPO_ROOT/tests/h3/perf/runs"
mkdir -p "$KEEP_DIR" || fail "cannot create $KEEP_DIR"

# The two engines live in different trees; nothing else distinguishes them.
case "$ENGINE" in
  port)
    [ -n "$BINARY" ] || BINARY="$REPO_ROOT/h3-rs"
    EXTRA=()
    P_TEXT='^TEXT '; P_LOAD='^LOAD '; P_DENOISE='^DENOISE '
    P_AUDIO='^AUDIO '; P_VAE='^FRAMES '; P_MUX='^VIDEO '
    ;;
  reference)
    [ -n "$BINARY" ] || BINARY=${H3_REFERENCE:-$HOME/h3-oracle/h3}
    # H3_PROFILE synchronizes the stream at every mark
    # (h3_gpu_cuda.cu:658-669), so a phase-emitting run is a different timing
    # run. Phase rows are opt-in and flagged in the CSV; timing rows are not.
    EXTRA=()
    [ "$PROFILE_PHASES" = 1 ] && EXTRA=(--profile)
    P_TEXT='Qwen text encoder:.*[0-9]s$'
    P_LOAD=': load [0-9.]*s$'
    P_DENOISE='denoise [0-9.]*s$'
    P_AUDIO='audio VAE[^:]*:.*[0-9]s$'
    P_VAE='video VAE[^:]*:.*[0-9]s$'
    P_MUX='^h3: wrote'
    ;;
  *) fail "--engine must be port or reference" ;;
esac

[ -x "$BINARY" ] || fail "no executable at $BINARY"
[ -d "$MODEL_DIR" ] || fail "no checkpoint at $MODEL_DIR"

ARGS=(-d "$MODEL_DIR" -p "$PROMPT" --width "$WIDTH" --height "$HEIGHT"
      --frames "$FRAMES" --steps "$STEPS" --seed "$SEED")
[ -n "$LAYERS" ] && ARGS+=(--layers "$LAYERS")
[ -n "$ANCHOR" ] && ARGS+=(--first-frame "$ANCHOR")
ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})

if [ "$DRY_RUN" = 1 ]; then
  printf '%s %s -o %s\n' "$BINARY" "${ARGS[*]}" "$OUT"
  exit 0
fi

# Identity of the thing that is about to be measured.
BIN_SHA=$(sha256sum "$BINARY" | cut -d' ' -f1)
GIT_REV=$(git -C "$(dirname "$BINARY")" rev-parse --short HEAD 2>/dev/null || echo none)
GIT_DIRTY=$(git -C "$(dirname "$BINARY")" status --porcelain 2>/dev/null | wc -l)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
CLOCK_BEFORE=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader 2>/dev/null | head -1)
CKPT_BYTES=$(du -sb "$MODEL_DIR" 2>/dev/null | cut -f1)
CACHE_BEFORE=$(awk '/^Cached:/ {print $2}' /proc/meminfo)
PROMPT_SHA=$(printf '%s' "$PROMPT" | sha256sum | cut -d' ' -f1)

if [ ! -f "$OUT" ]; then
  mkdir -p "$(dirname "$OUT")"
  echo "ts_utc,engine,label,profile,binary_sha256,git_rev,dirty_files,driver,clock_before,clock_after,width,height,frames,steps,layers,seed,text_s,load_s,denoise_s,per_step_s,audio_s,vae_s,mux_s,wall_s,peak_rss_kb,output_bytes,ckpt_bytes,cache_before_kb,cache_after_kb,prompt_sha256,anchor" >"$OUT" ||
    fail "cannot write the header to $OUT"
fi

# The port prints "seconds=X"; the reference's profile marks print
# "<label>: <phase> <delta>s" (h3_gpu_cuda.cu:658-669). A phase the engine did
# not mark stays empty, never zero.
phase() { # phase <log> <pattern>
  local value
  value=$(grep -aE "$2" "$1" 2>/dev/null | tail -1 |
    sed -n 's/.*seconds=\([0-9.]*\).*/\1/p')
  if [ -z "$value" ]; then
    value=$(grep -aE "$2" "$1" 2>/dev/null | tail -1 |
      sed -n 's/.*[[:space:]]\([0-9][0-9]*\(\.[0-9]*\)\?\)s$/\1/p')
  fi
  printf '%s' "$value"
}

# RFC 4180 quoting, applied only where it is needed: a label or a path can
# carry a comma, and an unquoted comma would shift every later column.
csv_field() {
  local value=$1
  case "$value" in
    *,*|*\"*|*$'\n'*)
      value=${value//\"/\"\"}
      printf '"%s"' "$value"
      ;;
    *) printf '%s' "$value" ;;
  esac
}

run_once() {
  local log wall start end rss out_file tag base
  tag=$(printf '%s' "${LABEL:-run}" | tr -c 'A-Za-z0-9_.-' '-')
  base="$KEEP_DIR/$(date -u +%Y%m%dT%H%M%SZ)-$ENGINE-${WIDTH}x${HEIGHT}-f${FRAMES}-s${STEPS}-$tag"  log="$base.log"
  out_file="$base.mp4"
  start=$(date +%s.%N)

  # H3_PROFILE makes the reference emit its phase marks; the port prints them
  # unconditionally. Both write plain lines to stdout.
  if [ "$ENGINE" = reference ] && [ "$PROFILE_PHASES" = 1 ]; then
    H3_PROFILE=1 "$BINARY" "${ARGS[@]}" -o "$out_file" >"$log" 2>&1
  else
    "$BINARY" "${ARGS[@]}" -o "$out_file" >"$log" 2>&1
  fi
  local status=$?
  end=$(date +%s.%N)
  wall=$(awk -v a="$start" -v b="$end" 'BEGIN {printf "%.3f", b-a}')

  if [ "$status" -ne 0 ]; then
    echo "h3_perf_sweep: run failed (status $status); log kept at $log" >&2
    tail -5 "$log" >&2
    rm -f "$out_file"
    return 1
  fi

  local text_s load_s denoise_s per_step audio_s vae_s mux_s bytes rss_peak
  text_s=$(phase "$log" "$P_TEXT")
  load_s=$(phase "$log" "$P_LOAD")
  denoise_s=$(phase "$log" "$P_DENOISE")
  audio_s=$(phase "$log" "$P_AUDIO")
  vae_s=$(phase "$log" "$P_VAE")
  mux_s=$(phase "$log" "$P_MUX")
  per_step=""
  if [ -n "$denoise_s" ] && [ "$STEPS" -gt 0 ]; then
    per_step=$(awk -v d="$denoise_s" -v s="$STEPS" 'BEGIN {printf "%.3f", d/s}')
  fi
  bytes=$(stat -c %s "$out_file" 2>/dev/null)
  [ -n "$bytes" ] || bytes=""
  # Not collected: the child has already been reaped, so its high-water RSS is
  # gone. The column stays for schema stability and is documented as empty.
  rss_peak=""

  # Free text can contain commas (a label, a path), so those two fields are
  # quoted per RFC 4180; every other field is a number, a hash or a name.
  row=$(printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ENGINE" "$tag" "$PROFILE_PHASES" \
    "$BIN_SHA" "$GIT_REV" \
    "$GIT_DIRTY" "$DRIVER" "$CLOCK_BEFORE" \
    "$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader 2>/dev/null | head -1)" \
    "$WIDTH" "$HEIGHT" "$FRAMES" "$STEPS" "${LAYERS:-default}" "$SEED" \
    "$text_s" "$load_s" "$denoise_s" "$per_step" "$audio_s" "$vae_s" "$mux_s" \
    "$wall" "$rss_peak" "$bytes" "$CKPT_BYTES" "$CACHE_BEFORE" \
    "$(awk '/^Cached:/ {print $2}' /proc/meminfo)" "$PROMPT_SHA" \
    "$(csv_field "${ANCHOR:-none}")")
  printf '%s\n' "$row" >>"$OUT" || fail "cannot append a row to $OUT"

  # The MP4 and its log stay under $KEEP_DIR so a later run can be compared
  # with the same artifact, not just the same numbers.
  echo "h3_perf_sweep: kept $out_file" >&2
}

runs=0
while [ "$runs" -lt "$REPEATS" ]; do
  run_once || exit 1
  runs=$((runs + 1))
done

echo "h3_perf_sweep: $runs row(s) in $OUT"
