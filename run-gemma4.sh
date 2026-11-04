#!/bin/bash
# Run the Gemma-4 26B A4B VQ release as an OpenAI-compatible server on the
# RTX 4070 SUPER, with the five full attention layers in E4M3 KV
# (DS4_GEMMA4_FULL_KV=fp8). The sliding layers keep their bf16 rings, so the
# FP8 option buys context on the layers that grow with it.
#
# usage:
#   ./run-gemma4.sh serve start      start the server in the background
#   ./run-gemma4.sh serve stop       stop the server this script started
#   ./run-gemma4.sh serve status     pid, port, /v1/models, VRAM
#   ./run-gemma4.sh serve logs [n]   tail the capture (default 30 lines)
#   ./run-gemma4.sh plan             print the serving quote, boot nothing
#   ./run-gemma4.sh chat [prompt]    one chat request through the running server
#   ./run-gemma4.sh bench [tokens]   ds4-bench decode with the same env
#   ./run-gemma4.sh help
#
# The weights live on the GPU by default: DS4_CUDA_COPY_MODEL=1 copies the
# 9.37 GiB trunk in about 3 s and the card reads 10.16 GiB of its 12 GiB at
# ctx 16384 with the FP8 full-layer KV and no drafter. The earlier ctx-1057
# graph failure was the drafter's verify workspace, not the trunk: without
# --mtp the copy fits, and the serial lane serves. Set
# DS4_GEMMA4_RESIDENT=0 to fall back to the host-mapped path (2.7 tok/s
# against the device copy's 21 tok/s on the same prompt).
#
# MTP is deliberately absent: DS4_GEMMA4_FULL_KV=fp8 refuses the drafter by
# design ("the drafter reads the full layer's bf16 KV", ds4.c:74110-74117),
# and the owner's 4070 MTP line waits for the CPU-expert phase.
#
# Env overrides:
#   DS4_GEMMA4_MODEL      release directory (default /data/vq-artifacts/gguf-family-it-r8)
#   DS4_GEMMA4_SERVER_BIN server binary (default <root>/ds4-server)
#   DS4_GEMMA4_PORT       port (default 8099)
#   DS4_GEMMA4_CTX        context (default 16384)
#   DS4_GEMMA4_MEM_FLOOR  memory floor in GiB (default 1)
#   DS4_GEMMA4_KV         fp8 (default) or bf16
#   DS4_GEMMA4_RESIDENT   1 (default, weights on the GPU) or 0 (host-mapped)
#   DS4_GEMMA4_FAST_ATTN  1 (default) or 0; DS4_GEMMA4_ATTN_FAST
#   DS4_GEMMA4_LOG        server capture (default misc/scratch/gemma4-serve.log)
#
# The FP8 full-layer KV was verified on a short smoke against bf16; its
# long-context gate is still open. If a long answer degrades, rerun with
# DS4_GEMMA4_KV=bf16 and compare on the same prompt.

set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
BIN="${DS4_GEMMA4_SERVER_BIN:-$ROOT/ds4-server}"
MODEL="${DS4_GEMMA4_MODEL:-/data/vq-artifacts/gguf-family-it-r8}"
PORT="${DS4_GEMMA4_PORT:-8099}"
CTX="${DS4_GEMMA4_CTX:-16384}"
FLOOR="${DS4_GEMMA4_MEM_FLOOR:-1}"
KV="${DS4_GEMMA4_KV:-fp8}"
RESIDENT="${DS4_GEMMA4_RESIDENT:-1}"
FAST="${DS4_GEMMA4_FAST_ATTN:-1}"
SCRATCH="$ROOT/misc/scratch"
mkdir -p "$SCRATCH"
LOG="${DS4_GEMMA4_LOG:-$SCRATCH/gemma4-serve.log}"
PIDFILE="$SCRATCH/gemma4-serve.pid"
PROMPT_FILE="$SCRATCH/gemma4-bench-prompt.txt"
# The engine's default lock (/tmp/ds4.lock) is shared with every other ds4
# model on this host, and the unrelated embeddinggemma server already holds
# it. The Gemma gates use their own lock, so this script does the same; two
# models can then coexist, and a second Gemma server still refuses.
LOCK="${DS4_GEMMA4_LOCK:-/tmp/ds4-gemma4-gate.lock}"

die() { echo "ERROR: $*" >&2; exit 1; }

kv_env() {
  [ "$KV" = "fp8" ] && printf 'DS4_GEMMA4_FULL_KV=fp8' || printf 'DS4_GEMMA4_FULL_KV=bf16'
}

fast_env() {
  [ "$FAST" = "1" ] && printf 'DS4_GEMMA4_ATTN_FAST=1' || printf 'DS4_GEMMA4_ATTN_FAST=0'
}

check_env() {
  [ -x "$BIN" ] || die "server binary not found at $BIN (build it: make ds4-server CUDA_ARCH=sm_89)"
  [ -d "$MODEL" ] || die "release directory not found: $MODEL"
  command -v curl >/dev/null || die "curl is required"
  command -v jq >/dev/null || die "jq is required"
}

# The plan the server will actually boot, printed as JSON by --check-config.
plan_json() {
  env DS4_LOCK_FILE="$LOCK" "$(kv_env)" "$(fast_env)" "$BIN" -m "$MODEL" --cuda -c "$CTX" \
      --mem-floor-gb "$FLOOR" --max-seqs 1 --check-config 2>/dev/null
}

plan_mode() {
  check_env
  local json
  json="$(plan_json)" || die "the plan did not print (server error)"
  printf '%s' "$json" | jq '{quote, issues, effective}'
}

# Refuse before boot when the quote does not fit; the numbers are the message.
preflight() {
  local json code
  json="$(plan_json)" || die "the plan did not print (server error)"
  code="$(printf '%s' "$json" | jq -r '[.issues[].code] | join(",")')"
  if [ -n "$code" ]; then
    printf '%s' "$json" | jq '{quote, issues}'
    die "the serving quote refuses this configuration ($code); lower DS4_GEMMA4_CTX or free host RAM"
  fi
  printf '%s' "$json" | jq -c '{quote: (.quote|{shared_weights,per_bank,floor,available,total}), issues: [.issues[].code]}'
}

pid_of() {
  [ -f "$PIDFILE" ] || return 1
  local pid
  pid=$(cat "$PIDFILE" 2>/dev/null) || return 1
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "ds4-server" || return 1
  printf '%s' "$pid"
}

serve_start() {
  local pid waited=0
  if pid=$(pid_of); then
    echo "server: already running (pid $pid); ./run-gemma4.sh serve stop first"
    return 0
  fi
  check_env
  if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    die "port $PORT already answers /v1/models; stop that server or set DS4_GEMMA4_PORT"
  fi
  echo "model:   $MODEL"
  if [ "$RESIDENT" = "1" ]; then
    echo "backend: cuda (weights on the GPU), ctx $CTX, floor ${FLOOR}G, max-seqs 1"
  else
    echo "backend: cuda (host-mapped weights), ctx $CTX, floor ${FLOOR}G, max-seqs 1"
  fi
  echo "kv:      $KV full layers; fast attention $FAST"
  preflight
  rm -f "$LOG"
  local -a envs=( "DS4_LOCK_FILE=$LOCK" "$(kv_env)" "$(fast_env)" )
  [ "$RESIDENT" = "1" ] && envs+=( "DS4_CUDA_COPY_MODEL=1" )
  setsid env "${envs[@]}" \
      "$BIN" -m "$MODEL" --cuda --host 127.0.0.1 --port "$PORT" \
      -c "$CTX" --mem-floor-gb "$FLOOR" --max-seqs 1 \
      > "$LOG" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$PIDFILE"
  while [ "$waited" -lt 600 ]; do
    grep -q "listening on" "$LOG" 2>/dev/null && break
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "server exited before listening; last lines:"
      tail -12 "$LOG"
      rm -f "$PIDFILE"
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  if ! grep -q "listening on" "$LOG"; then
    echo "server did not listen within ${waited}s; last lines:"
    tail -12 "$LOG"
    kill "$pid" 2>/dev/null
    rm -f "$PIDFILE"
    return 1
  fi
  echo "server:  up (pid $pid)"
  grep -o 'listening on.*' "$LOG" | tail -1 | sed 's/^/         /'
  echo "base_url: http://127.0.0.1:$PORT/v1"
  echo "model id: $(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].id // "unknown"')"
  echo "log:     $LOG"
  echo "note:    one ds4 model at a time; this server holds the slot until stopped"
  echo
  echo "open-grok config block (already added as [model.gemma4-vq-4070]):"
  echo "  [model.gemma4-vq-4070]"
  echo "  model = \"$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].id // "gguf-family-it-r8"')\""
  echo "  base_url = \"http://127.0.0.1:$PORT/v1\""
  echo "  context_window = $CTX"
  echo "  api_key = \"local\""
  echo "  supports_images = false"
}

serve_stop() {
  local pid waited=0
  if ! pid=$(pid_of); then
    echo "server: not running"
    rm -f "$PIDFILE"
    return 0
  fi
  kill "$pid" 2>/dev/null
  while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 30 ]; do
    sleep 1
    waited=$((waited + 1))
  done
  rm -f "$PIDFILE"
  if kill -0 "$pid" 2>/dev/null; then
    echo "server: pid $pid did not exit within 30s" >&2
    return 1
  fi
  echo "server: stopped (pid $pid)"
}

serve_status() {
  local pid models
  if pid=$(pid_of); then
    echo "server: running (pid $pid)"
    grep -o 'listening on.*' "$LOG" 2>/dev/null | tail -1 | sed 's/^/         /'
    echo "base_url: http://127.0.0.1:$PORT/v1"
    models=$(curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" |
             jq -r '.data[0] | "\(.id) ctx \(.context_length)"' 2>/dev/null)
    echo "serving: ${models:-no answer on /v1/models}"
    command -v nvidia-smi >/dev/null &&
      echo "vram:    $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
  else
    echo "server: not running"
  fi
}

serve_logs() {
  [ -f "$LOG" ] || die "no serve log yet at $LOG"
  tail -n "$1" "$LOG"
}

chat_mode() {
  local prompt="${1:-Hello! Answer in one short sentence.}" model body t0 t1 wall
  model=$(curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].id // empty')
  [ -n "$model" ] || die "no server answers on port $PORT; run ./run-gemma4.sh serve start"
  t0=$(date +%s.%N)
  body=$(jq -n --arg m "$model" --arg p "$prompt" \
      '{model:$m,messages:[{role:"user",content:$p}],max_tokens:128,temperature:0}' |
      curl -s -m 1800 -H 'Content-Type: application/json' \
        -d @- "http://127.0.0.1:$PORT/v1/chat/completions")
  t1=$(date +%s.%N)
  wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
  printf '%s' "$body" | jq -r '
    "finish: \(.choices[0].finish_reason)",
    "answer: \((.choices[0].message.content // "") | gsub("\\s+$"; ""))",
    "usage:  \(.usage)"'
  printf '%s' "$body" | jq -r '.usage.completion_tokens // 0' | {
    read -r n
    awk -v w="$wall" -v n="$n" 'BEGIN{if(n>0) printf "rate:   %.2f tok/s (%s tokens in %ss)\n", n/w, n, w; else printf "rate:   n/a (%ss)\n", w}'
  }
}

bench_mode() {
  local tokens="${1:-32}"
  check_env
  [ -f "$PROMPT_FILE" ] || printf '%s\n' \
    'The quick brown fox jumps over the lazy dog. A dense vector quantized model stores its experts in a compact codebook, and the decoder rebuilds each weight row from an index and a residual.' > "$PROMPT_FILE"
  echo "model:   $MODEL"
  if [ "$RESIDENT" = "1" ]; then
    echo "backend: cuda (weights on the GPU), ctx $CTX, gen $tokens, kv $KV"
  else
    echo "backend: cuda (host-mapped weights), ctx $CTX, gen $tokens, kv $KV"
  fi
  local -a envs=( "DS4_LOCK_FILE=$LOCK" "$(kv_env)" "$(fast_env)" )
  [ "$RESIDENT" = "1" ] && envs+=( "DS4_CUDA_COPY_MODEL=1" )
  env "${envs[@]}" "$ROOT/ds4-bench" -m "$MODEL" --cuda \
      --prompt-file "$PROMPT_FILE" --ctx-start "$CTX" --ctx-max "$CTX" \
      --gen-tokens "$tokens"
}

usage() {
  sed -n '5,18p' "$0" | sed 's/^# \{0,1\}//'
}

case "${1:-serve}" in
  serve)
    shift
    case "${1:-start}" in
      start|"") serve_start ;;
      stop)     serve_stop ;;
      status)   serve_status ;;
      logs)     shift; serve_logs "${1:-30}" ;;
      *)        die "serve wants start, stop, status or logs (got: $1)" ;;
    esac ;;
  plan)  plan_mode ;;
  chat)  shift; chat_mode "$*" ;;
  bench) shift; bench_mode "${1:-32}" ;;
  help|-h|--help) usage ;;
  *) die "unknown mode: $1 (see ./run-gemma4.sh help)" ;;
esac
