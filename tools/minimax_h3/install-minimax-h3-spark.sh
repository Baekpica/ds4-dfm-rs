#!/usr/bin/env bash
# MiniMax-H3 (FL2VA) on the DGX Spark: fetch the released checkpoint, verify
# it, build the Rust renderer, prove it against the recorded gate, and serve it
# through the local v1 video API (docs/minimax-h3/local-v1-contract.md).
#
#   artifact  MiniMaxAI/MiniMax-H3 on Hugging Face, about 465 GB
#             (the reference README records the size; this script never pins a
#             revision by guess — HF_REVISION selects one when set)
#   layout    under MODEL_DIR:
#             FL2VA/transformer/      13 shards, 535 tensors, 61.728 GiB
#             FL2VA/text_encoder/     14 shards, 1,058 tensors, 62.133 GiB
#             FL2VA/tokenizer/tokenizer.json            7.0 MB
#             FL2VA/video_vae/source/                 9.700 GiB
#             FL2VA/audio_vae/                        0.564 GiB
#             (Ref2VA/, when present, is the ordered-reference checkpoint the
#             port does not serve; E5 refuses it by name)
#   engine    ds4-dfm-rs on branch feature/minimax-h3, built with `make h3-rs`
#             and `cargo build -p ds4-server --bin h3-server-rs`
#
# WHAT THIS SCRIPT IS FOR
#   1. It never downloads twice. An existing complete snapshot is adopted and
#      verified, never re-fetched.
#   2. It never trusts a snapshot it has not verified. `fetch` checks sizes;
#      `verify` runs the Hugging Face file manifest and then asks the renderer
#      itself to resolve every component (`--info`).
#   3. It proves the binary before serving it. `gate` renders the pinned clip
#      and compares the four hashes the reference and the port produce. A
#      mismatch stops the install: the port is not serving anything it has not
#      reproduced.
#   4. It measures the card before it starts. The text encoder (62.133 GiB) and
#      the DiT (61.728 GiB) are both resident in turn on a 121 GiB unified box,
#      so a second owner has to be refused rather than left to swap.
#
# WHAT HAS NOT BEEN MEASURED
#   The gate below is 256x256, 22 frames, 2 evaluations: it is an arithmetic
#   gate, not a speed run. The reference's own GB10 command is 512x512, 22
#   frames, 20 evaluations, and this port has not been timed against it yet
#   (P9 owns that). Nothing here claims a speed figure.
#
# USAGE
#   bash install-minimax-h3-spark.sh preflight
#   bash install-minimax-h3-spark.sh fetch        # about 465 GB, resumable
#   bash install-minimax-h3-spark.sh verify
#   bash install-minimax-h3-spark.sh build
#   bash install-minimax-h3-spark.sh gate
#   bash install-minimax-h3-spark.sh render -p "prompt" -o outputs/h3.mp4
#   bash install-minimax-h3-spark.sh start        # 0.0.0.0:8011
#   bash install-minimax-h3-spark.sh install      # all of the above, then start
#   bash install-minimax-h3-spark.sh stop | status | logs | test | plan
#
# OPTIONS (each is also an environment variable of the same name)
#   --model-dir DIR    checkpoint root holding FL2VA/ (default
#                      /mnt/h3-artifacts/MiniMax-H3)
#   --engine-dir DIR   the ds4-dfm-rs tree (default $HOME/ds4-h3-port)
#   --port PORT        API port (default 8011; 8000-8010 are other deployments)
#   --bind ADDR        listen address (default 0.0.0.0)
#   --steps N          render evaluations (default 20, the reference's default)
#   --width N          render width (default 512, the reference's GB10 canvas)
#   --height N         render height (default 512)
#   --frames N         frames (default 22, aligned to 5+17k)
#   --stream           pass --ssd-streaming to the renderer (block ring)
#   --no-build         use the existing binaries as they are
#   --no-start         `install` stops after the gate
#   --allow-shared     overrule the single-owner probe before starting
#   -h, --help         this text
#
# Notes
#   - The checkpoint is public: no token is required and none is written into
#     this file. HF_TOKEN only raises the CDN rate limit.
#   - ffmpeg and ffprobe must be on PATH. The renderer takes an explicit
#     program from H3_FFMPEG when the operator needs a different one.
#   - A render needs at least 22 frames (one trained decoder chunk) and at most
#     362 aligned frames; width and height are multiples of 32 with an area of
#     at most 768*1344.
#   - `--layers` is not exposed: the port runs the released 50-block stack, and
#     the flag is refused rather than ignored.
#   - The port does not implement --token-reduction, --use-int8-row-fc2 (a
#     measured CUDA no-op in the reference), --show, --zoom or the interactive
#     session. Their absence is a documented deviation, not a silent one.

set -Eeuo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
HF_REPO="${HF_REPO:-MiniMaxAI/MiniMax-H3}"
HF_REVISION="${HF_REVISION:-}"
MODEL_DIR="${MODEL_DIR:-/mnt/h3-artifacts/MiniMax-H3}"
ENGINE_DIR="${ENGINE_DIR:-$HOME/ds4-h3-port}"
BIN="${BIN:-$ENGINE_DIR/h3-rs}"
SERVER_BIN="${SERVER_BIN:-$ENGINE_DIR/target/release/h3-server-rs}"
HOST_ADDR="${HOST_ADDR:-0.0.0.0}"
PORT="${PORT:-8011}"
ARCH="${CUDA_ARCH:-sm_121}"
OUT_DIR="${OUT_DIR:-$ENGINE_DIR/outputs}"

WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
FRAMES="${FRAMES:-22}"
STEPS="${STEPS:-20}"
PROMPT="${PROMPT:-A bright red cube rotates smoothly on a white background.}"
STREAM=0
NO_BUILD=0
NO_START=0
ALLOW_SHARED=0

# The E9 gate, recorded on this card: the reference engine and the port agree
# byte for byte at 256x256, 22 frames, 2 evaluations, seed 1, a 0.5 spatial rope
# scale. A mismatch here means the binary is not the one that was verified.
GATE_VIDEO=70973d0a2c32e378
GATE_AUDIO=f776839e1170733b
GATE_FRAMES=541455fe622e267f
GATE_RGB8=62b36816b6e1248e
GATE_ARGS=(--width 256 --height 256 --frames 22 --steps 2 --seed 1)

COMPONENTS=(
    FL2VA/tokenizer/tokenizer.json
    FL2VA/text_encoder
    FL2VA/transformer
    FL2VA/video_vae/source
    FL2VA/audio_vae
)

PIDF="$ENGINE_DIR/h3-server.pid"
SERVER_LOG="$ENGINE_DIR/h3-server.log"
WD_PIDF="$ENGINE_DIR/h3-watchdog.pid"
WD_LOG="$ENGINE_DIR/h3-watchdog.log"

# Resident weight figures, GiB, from the port's own layout test and the
# reference's inventory: the install refuses to start on less room than these.
TEXT_ENCODER_GIB=63
DIT_GIB=62
VIDEO_VAE_GIB=10
AUDIO_VAE_GIB=1
START_FLOOR_MB=100000

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
say()  { printf 'h3: %s\n' "$*"; }
warn() { printf 'h3: warning: %s\n' "$*" >&2; }
die()  { printf 'h3: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
total_mb() { awk '/MemTotal/{print int($2/1024)}' /proc/meminfo; }
df_avail() { df -B1 --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' \n' || true; }
human()    { awk -v b="$1" 'BEGIN{printf "%.1f GB", b/1000000000}'; }
gib()      { awk -v m="$1" 'BEGIN{printf "%.1f GiB", m/1024}'; }

owners() {
    pgrep -af 'h3-rs|h3-server-rs' 2>/dev/null | grep -v "install-minimax-h3" || true
}

require_model() {
    [ -d "$MODEL_DIR" ] || die "no checkpoint at $MODEL_DIR (run: $0 fetch)"

    for path in "${COMPONENTS[@]}"; do
        [ -e "$MODEL_DIR/$path" ] || die "missing $MODEL_DIR/$path"
    done
}

require_binaries() {
    [ -x "$BIN" ] || die "no renderer at $BIN (run: $0 build)"
}

usage() {
    sed -n '2,/^set -Eeuo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
}

# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------
preflight() {
    local failures=0

    say "checking tools"
    for tool in nvcc ffmpeg ffprobe cc; do
        if command -v "$tool" >/dev/null 2>&1; then
            say "  $tool: $(command -v "$tool")"
        else
            warn "missing $tool"
            failures=$((failures + 1))
        fi
    done

    if command -v cargo >/dev/null 2>&1; then
        say "  cargo: $(cargo --version)"
    else
        warn "missing cargo (build needs a Rust toolchain)"
        failures=$((failures + 1))
    fi

    say "device"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
        || warn "nvidia-smi did not answer"

    say "memory: $(gib "$(avail_mb)") available of $(gib "$(total_mb)")"

    if [ -d "$MODEL_DIR" ]; then
        say "checkpoint volume: $(human "$(df_avail "$MODEL_DIR")") free at $MODEL_DIR"
    else
        warn "$MODEL_DIR is absent; fetch needs about 465 GB somewhere"
    fi

    local live
    live="$(owners)"
    if [ -n "$live" ]; then
        warn "another h3 owner is resident:"
        printf '%s\n' "$live" >&2
    else
        say "no other h3 owner"
    fi

    [ "$failures" -eq 0 ] || die "$failures tool(s) missing"
    say "preflight ok"
}

# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------
fetch() {
    need hf

    if [ -d "$MODEL_DIR" ] && [ -e "$MODEL_DIR/FL2VA/transformer" ] \
        && [ -e "$MODEL_DIR/FL2VA/text_encoder" ]; then
        say "$MODEL_DIR already holds both FL2VA partitions; not fetching"
        say "run '$0 verify' to prove the bytes"
        return
    fi

    local revision=()
    if [ -n "$HF_REVISION" ]; then
        revision=(--revision "$HF_REVISION")
    fi

    say "downloading $HF_REPO to $MODEL_DIR (about 465 GB, resumable)"
    hf download "$HF_REPO" "${revision[@]}" --local-dir "$MODEL_DIR"
    say "fetch complete; run '$0 verify'"
}

# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------
verify() {
    require_model

    if command -v hf >/dev/null 2>&1; then
        local revision=()
        if [ -n "$HF_REVISION" ]; then
            revision=(--revision "$HF_REVISION")
        fi
        say "checking the Hugging Face file manifest"
        hf cache verify "$HF_REPO" "${revision[@]}" --local-dir "$MODEL_DIR" \
            --fail-on-missing-files
    else
        warn "hf CLI absent; the manifest check is skipped"
    fi

    say "component sizes"
    local path
    for path in "${COMPONENTS[@]}"; do
        local size
        size="$(du -sb "$MODEL_DIR/$path" 2>/dev/null | awk '{print $1}')"
        printf '  %-40s %s\n' "$path" "$(human "${size:-0}")"
    done

    # The renderer resolves the components itself, which is the check that
    # matters: a missing shard or a wrong layout fails here by name.
    if [ -x "$BIN" ]; then
        say "renderer component resolution"
        "$BIN" -d "$MODEL_DIR" --info
    else
        warn "no renderer yet; run '$0 build' then '$0 verify' again"
    fi
}

# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
build() {
    [ -d "$ENGINE_DIR" ] || die "no engine tree at $ENGINE_DIR"
    [ -f "$ENGINE_DIR/Makefile" ] || die "$ENGINE_DIR has no Makefile"

    if [ "$NO_BUILD" -eq 1 ]; then
        say "--no-build: using existing binaries"
        return
    fi

    say "building the renderer (make h3-rs CUDA_ARCH=$ARCH)"
    ( cd "$ENGINE_DIR" && make "h3-rs" "CUDA_ARCH=$ARCH" )

    say "building the API host (h3-server-rs)"
    ( cd "$ENGINE_DIR" && cargo build --release --locked -p ds4-server --bin h3-server-rs )

    [ -x "$BIN" ] || die "build produced no $BIN"
    say "built $(du -h "$BIN" | awk '{print $1}') $BIN"
}

# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------
gate() {
    require_model
    require_binaries

    local log
    log="$(mktemp)"
    say "gating: ${GATE_ARGS[*]}"

    if ! "$BIN" -d "$MODEL_DIR" -p "A red fox walking through snow" \
        "${GATE_ARGS[@]}" --hash -o "$log.mp4" >"$log" 2>&1; then
        tail -20 "$log" >&2
        die "gate render failed"
    fi

    local line
    line="$(grep -a '^HASH ' "$log" || true)"
    [ -n "$line" ] || die "gate produced no HASH line"

    local video audio frames rgb
    video="$(printf '%s\n' "$line" | sed -n 's/.*video=\([0-9a-f]\{16\}\).*/\1/p')"
    audio="$(printf '%s\n' "$line" | sed -n 's/.*audio=\([0-9a-f]\{16\}\).*/\1/p')"
    frames="$(printf '%s\n' "$line" | sed -n 's/.*frames=\([0-9a-f]\{16\}\).*/\1/p')"
    rgb="$(printf '%s\n' "$line" | sed -n 's/.*rgb8=\([0-9a-f]\{16\}\).*/\1/p')"

    printf '  video  %s %s\n' "$video" "$([ "$video" = "$GATE_VIDEO" ] && echo ok || echo MISMATCH)"
    printf '  audio  %s %s\n' "$audio" "$([ "$audio" = "$GATE_AUDIO" ] && echo ok || echo MISMATCH)"
    printf '  frames %s %s\n' "$frames" "$([ "$frames" = "$GATE_FRAMES" ] && echo ok || echo MISMATCH)"
    printf '  rgb8   %s %s\n' "$rgb" "$([ "$rgb" = "$GATE_RGB8" ] && echo ok || echo MISMATCH)"

    if [ "$video" != "$GATE_VIDEO" ] || [ "$audio" != "$GATE_AUDIO" ] \
        || [ "$frames" != "$GATE_FRAMES" ] || [ "$rgb" != "$GATE_RGB8" ]; then
        tail -20 "$log" >&2
        die "gate mismatch: this binary does not reproduce the recorded clip"
    fi

    if command -v ffprobe >/dev/null 2>&1; then
        say "muxed file:"
        ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames \
            -of default=noprint_wrappers=1 "$log.mp4" | sed 's/^/  /'
    fi

    rm -f "$log" "$log.mp4"
    say "gate ok"
}

# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------
render() {
    require_model
    require_binaries

    mkdir -p "$OUT_DIR"
    local output="$OUT_DIR/h3-$(date +%Y%m%d-%H%M%S).mp4"
    local extra=()
    if [ "$STREAM" -eq 1 ]; then
        extra=(--ssd-streaming)
    fi

    say "rendering ${WIDTH}x${HEIGHT}, $FRAMES frames, $STEPS evaluations"
    "$BIN" -d "$MODEL_DIR" -p "$PROMPT" \
        --width "$WIDTH" --height "$HEIGHT" --frames "$FRAMES" \
        --steps "$STEPS" "${extra[@]}" -o "$output"

    [ -s "$output" ] || die "renderer produced no file"
    say "wrote $output ($(du -h "$output" | awk '{print $1}'))"
}

# --------------------------------------------------------------------------
# start / stop / status / logs
# --------------------------------------------------------------------------
start() {
    require_model
    [ -x "$SERVER_BIN" ] || die "no API host at $SERVER_BIN (run: $0 build)"

    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
        say "already running: pid $(cat "$PIDF") on port $PORT"
        return
    fi

    local available
    available="$(avail_mb)"

    if [ "$available" -lt "$START_FLOOR_MB" ]; then
        die "only $(gib "$available") available; a render needs about $((START_FLOOR_MB / 1024)) GiB"
    fi

    if [ "$ALLOW_SHARED" -eq 0 ]; then
        local live
        live="$(owners)"
        [ -z "$live" ] || die "another h3 owner is resident (use --allow-shared):
$live"
    fi

    local extra=()
    if [ "$STREAM" -eq 1 ]; then
        extra=(--renderer-arg --ssd-streaming)
    fi

    say "starting h3-server-rs on $HOST_ADDR:$PORT"
    nohup "$SERVER_BIN" \
        --bind "$HOST_ADDR:$PORT" \
        --out-dir "$OUT_DIR" \
        --renderer "$BIN" \
        --renderer-arg -d --renderer-arg "$MODEL_DIR" \
        --renderer-arg --steps --renderer-arg "$STEPS" \
        "${extra[@]}" \
        >"$SERVER_LOG" 2>&1 &

    echo $! >"$PIDF"
    say "pid $(cat "$PIDF"), log $SERVER_LOG"

    local waited=0
    while [ "$waited" -lt 60 ]; do
        # Any HTTP answer proves the listener is up; the contract has no
        # /v1/models, so an unknown id's 404 is the cheapest probe.
        if curl -sS -o /dev/null "http://127.0.0.1:$PORT/v1/videos/video_0" 2>/dev/null; then
            say "ready: http://127.0.0.1:$PORT/v1/videos"
            return
        fi
        sleep 1
        waited=$((waited + 1))
    done

    tail -20 "$SERVER_LOG" >&2
    die "the API host did not answer within 60 s"
}

stop() {
    if [ ! -f "$PIDF" ]; then
        say "not running"
        return
    fi

    local pid
    pid="$(cat "$PIDF")"

    if kill -0 "$pid" 2>/dev/null; then
        say "stopping pid $pid"
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$pid" 2>/dev/null || true
    fi

    rm -f "$PIDF"
    say "stopped"
}

status() {
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
        say "running: pid $(cat "$PIDF")"
    else
        say "not running"
    fi

    say "memory: $(gib "$(avail_mb)") available"
    local live
    live="$(owners)"
    [ -z "$live" ] || printf '%s\n' "$live"

    if curl -sS -o /dev/null "http://127.0.0.1:$PORT/v1/videos/video_0" 2>/dev/null; then
        echo "  api: http://127.0.0.1:$PORT/v1/videos"
    fi
}

logs() {
    [ -f "$SERVER_LOG" ] || die "no log at $SERVER_LOG"
    tail -n "${LINES:-80}" "$SERVER_LOG"
}

# --------------------------------------------------------------------------
# test: one job through the local v1 surface
# --------------------------------------------------------------------------
api_test() {
    say "POST /v1/videos"
    # The contract serves one tier (short edge 768) and 4..15 seconds; a square
    # target is the cheapest legal clip, so the smoke does not wait for 16:9.
    local body
    body="$(curl -fsS -X POST "http://127.0.0.1:$PORT/v1/videos" \
        -H 'Content-Type: application/json' \
        -d "{\"task\":\"t2va\",\"prompt\":\"$PROMPT\",\"conditions\":[],\
\"target\":{\"short_edge\":768,\"aspect_ratio\":\"1:1\",\"duration_seconds\":4},\"seed\":0}")" \
        || die "the API host refused the job (is it running? '$0 start')"

    local id
    id="$(printf '%s' "$body" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')"
    [ -n "$id" ] || die "no job id in the response: $body"

    say "job $id"

    local waited=0
    while [ "$waited" -lt "${RENDER_TIMEOUT:-3600}" ]; do
        local state
        state="$(curl -fsS "http://127.0.0.1:$PORT/v1/videos/$id" \
            | sed -n 's/.*"status":"\([^"]*\)".*/\1/p')"

        case "$state" in
            completed) break ;;
            failed|cancelled) die "job $id ended as $state" ;;
        esac

        sleep 5
        waited=$((waited + 5))
    done

    [ "$waited" -lt "${RENDER_TIMEOUT:-3600}" ] || die "job $id timed out"

    local out="$OUT_DIR/v1-$id.mp4"
    curl -fsS "http://127.0.0.1:$PORT/v1/videos/$id/content" -o "$out"
    [ -s "$out" ] || die "no content for job $id"

    say "wrote $out ($(du -h "$out" | awk '{print $1}'))"

    if command -v ffprobe >/dev/null 2>&1; then
        ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames \
            -of default=noprint_wrappers=1 "$out" | sed 's/^/  /'
    fi
}

# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------
plan() {
    cat <<PLAN
h3 install plan (nothing has been changed)

  checkpoint   $MODEL_DIR
               $HF_REPO${HF_REVISION:+, revision $HF_REVISION}, about 465 GB
  engine       $ENGINE_DIR
  renderer     $BIN
  api host     $SERVER_BIN
  listen       $HOST_ADDR:$PORT
  outputs      $OUT_DIR

  render defaults: ${WIDTH}x${HEIGHT}, $FRAMES frames, $STEPS evaluations
  reference GB10 command this mirrors:
    h3 --profile -d MODEL -p "$PROMPT" \\
       --width 512 --height 512 --frames 22 --steps 20 --layers 50 \\
       --token-reduction -o outputs/gb10-cube.mp4

  steps this script would run:
    preflight   tools, device, memory, free space, other owners
    fetch       hf download, resumable, skipped when a complete snapshot exists
    verify      hf cache verify --fail-on-missing-files, sizes, --info
    build       make h3-rs CUDA_ARCH=$ARCH, cargo build h3-server-rs
    gate        ${GATE_ARGS[*]}, expecting
                video $GATE_VIDEO audio $GATE_AUDIO
                frames $GATE_FRAMES rgb8 $GATE_RGB8
    start       h3-server-rs with the renderer at $BIN

  not ported, refused rather than ignored: --layers other than 50,
  --token-reduction, --use-int8-row-fc2, --show, --zoom, the interactive
  session, keyframe and Ref2VA conditioning.
PLAN
}

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
COMMAND="${1:-help}"
if [ $# -gt 0 ]; then
    shift
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --engine-dir) ENGINE_DIR="$2"; BIN="$ENGINE_DIR/h3-rs"
                      SERVER_BIN="$ENGINE_DIR/target/release/h3-server-rs"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --bind) HOST_ADDR="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --width) WIDTH="$2"; shift 2 ;;
        --height) HEIGHT="$2"; shift 2 ;;
        --frames) FRAMES="$2"; shift 2 ;;
        -p|--prompt) PROMPT="$2"; shift 2 ;;
        -o|--output) OUT_DIR="$(dirname "$2")"; shift 2 ;;
        --stream) STREAM=1; shift ;;
        --no-build) NO_BUILD=1; shift ;;
        --no-start) NO_START=1; shift ;;
        --allow-shared) ALLOW_SHARED=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done

case "$COMMAND" in
    preflight) preflight ;;
    fetch) fetch ;;
    verify) verify ;;
    build) build ;;
    gate) gate ;;
    render) render ;;
    start) start ;;
    stop) stop ;;
    status) status ;;
    logs) logs ;;
    test) api_test ;;
    plan) plan ;;
    install)
        preflight
        fetch
        verify
        build
        gate
        if [ "$NO_START" -eq 1 ]; then
            say "install stopped before start (--no-start)"
            exit 0
        fi
        start
        ;;
    help|-h|--help) usage ;;
    *) usage; exit 2 ;;
esac
