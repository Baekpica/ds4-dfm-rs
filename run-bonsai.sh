#!/bin/bash
# Run Prism Bonsai 2 27B (Ternary-Bonsai-2-27B-PQ2_0) with the ds4 engine in
# this tree (ds4-dfm-rs, the qwen35 family).
#
# Usage:  ./run-bonsai.sh                    generate from the default prompt (CUDA)
#         ./run-bonsai.sh "a prompt"         generate from your prompt (CUDA)
#         ./run-bonsai.sh cuda ["prompt"]    the CUDA graph explicitly
#         ./run-bonsai.sh cpu ["prompt"]     the built-in CPU reference (the oracle)
#         ./run-bonsai.sh compare ["prompt"] both backends, diffed token for token
#         ./run-bonsai.sh ids                the explicit five-id parity gate (make bonsai-cuda-parity)
#         ./run-bonsai.sh bench [tokens]     decode rate with /usr/bin/time (default 16)
#         ./run-bonsai.sh status             what is built, which artifact, what can run
#         ./run-bonsai.sh session [prompt]   the session path on both backends, diffed
#         ./run-bonsai.sh server [prompt]    one chat request through the Rust server
#         ./run-bonsai.sh serve [start|stop|status|logs]  keep a server up for a client
#         ./run-bonsai.sh help
#
# This model is the "qwen35" family: a dense 64-layer trunk (48 gated
# delta-net layers and 16 gated-attention layers), every matmul weight PQ2_0
# (ternary, 2.125 bits per weight) and stored Hadamard-folded, so the engine
# rotates the activation instead of the weight.
#
# WHAT THIS TREE SUPPORTS
#   One binary, ./ds4-c, serves both backends.  Built with
#   "make ds4-c CUDA_ARCH=sm_89" it runs the CUDA graph (--cuda) and the CPU
#   reference (--cpu); built with "make cpu" it is a CPU-only binary that
#   cannot do --cuda.  This script never rebuilds anything.
#   The only entry is the diagnostic generator --first-token-test: greedy, one
#   token per graph call, with -p "<prompt>".  DS4_QWEN35_STEPS=<n> sets the
#   greedy step count (default 16) and DS4_QWEN35_TOKENS=<comma ids> replaces
#   the prompt with raw token ids, which is how the parity gate is run.
#   "compare" and "ids" diff the token ids the CUDA graph and the CPU
#   reference print for the same input; "ids" is exactly what
#   "make bonsai-cuda-parity" runs.
#
#   Every CUDA run on this box needs DS4_CUDA_COPY_MODEL=1.  The 6.71 GiB map
#   cannot be pinned here (RLIMIT_MEMLOCK is 8192 KiB, hard limit included), so
#   without it the backend falls back to lazy per-range materialisation and
#   dies part-way through the trunk with "Bonsai matmul failed for
#   blk.<n>.<tensor>".  The variable makes the backend copy the model to the
#   device (about 0.7 s, logged as "CUDA copying 6.71 GiB model to device
#   memory").  This script sets it for every --cuda run.
#
# WHAT THIS TREE DOES NOT DO YET (refused by name, never silently skipped)
#   - Batching, MTP and DSpark drafting, SSD/disk KV, tensor parallelism,
#     distributed ranks and session snapshots: the session path and the server
#     refuse each by name (ds4-server reports "qwen35 session snapshots are
#     unsupported" and will not start with --kv-disk-dir).  No CUDA session on
#     this box without DS4_CUDA_COPY_MODEL=1, which this script sets.
#
# THE SERVER PATH
#   ./run-bonsai.sh server drives the Rust ds4-server (./ds4-server, the
#   default host; the C oracle ds4-server-c cannot serve this family): it
#   starts the server, waits for the listener, sends one chat request through
#   the model's own ChatML template, prints the answer and stops the server.
#   DS4_BONSAI_CTX (default 45056) and DS4_BONSAI_MEM_FLOOR (default 1, GiB)
#   must fit the card: the quote counts the real 6.71 GiB of weights, the
#   16-attention-layer KV and this host's free VRAM, so the default 4 GiB floor
#   refuses at every usable context on the 12 GiB RTX 4070 SUPER, and 49152
#   leaves too little margin (measured: it opens pre-open and is refused on a
#   real start).
#
#   ./run-bonsai.sh serve starts the same server in its own session and leaves
#   it running for an OpenAI-compatible client (open-grok, a script, curl), then
#   `serve stop` ends it, `serve status` reports pid, port and what /v1/models
#   answers, `serve logs [n]` tails the capture.  The pid file and log live in
#   misc/scratch; stop only ever kills the pid that file records.  This family
#   holds the single ds4 model slot while it serves, so no other ds4 model can
#   run until it is stopped.
#
# THE SESSION PATH
#   ./run-bonsai.sh session drives the real ds4_session API (create, sync, eval)
#   through the CUDA graph and, separately, through the CPU reference trunk:
#   prefix reuse, rewind by replay, invalidate and the context bound are all
#   wired for both backends, and the diagnostic (DS4_QWEN35_SESSION=1) prints
#   the same token stream.  Prefill is chunked: DS4_QWEN35_PREFILL_CHUNK (512
#   by default, 1024 max) bounds the rows one CUDA forward carries, and a chunk
#   the device cannot hold is halved until the graph opens.
#
# Env overrides: DS4_BONSAI_MODEL (model path), DS4_BONSAI_BIN (binary),
# DS4_BONSAI_SERVER_BIN (Rust server), DS4_BONSAI_BACKEND (cuda|cpu),
# DS4_BONSAI_STEPS (greedy steps), DS4_BONSAI_CTX / DS4_BONSAI_MEM_FLOOR /
# DS4_BONSAI_SERVER_PORT / DS4_BONSAI_SERVER_TOKENS (server and serve modes),
# DS4_BONSAI_WAIT (seconds to wait for a free device slot, default 300),
# DS4_BONSAI_LOG (capture path, default misc/scratch/bonsai-run.log).

set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
BIN="${DS4_BONSAI_BIN:-$ROOT/ds4-c}"
SERVER_BIN="${DS4_BONSAI_SERVER_BIN:-$ROOT/ds4-server}"
SERVER_PORT="${DS4_BONSAI_SERVER_PORT:-8899}"
# The serving context and the memory floor that fits it on this host, measured
# against this card's free device memory; see the sizing note in the header and
# docs/BONSAI.md.
SERVER_CTX="${DS4_BONSAI_CTX:-45056}"
MEM_FLOOR="${DS4_BONSAI_MEM_FLOOR:-1}"
# A served answer needs ~35 tokens: the reasoning block alone runs about 33
# before any content appears.
SERVER_TOKENS="${DS4_BONSAI_SERVER_TOKENS:-64}"
MODEL="${DS4_BONSAI_MODEL:-/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf}"
BACKEND="${DS4_BONSAI_BACKEND:-cuda}"
STEPS="${DS4_BONSAI_STEPS:-16}"
DEFAULT_PROMPT="The capital of France is"
# The explicit parity prompt, the same ids "make bonsai-cuda-parity" uses.
PARITY_TOKENS="760,6511,314,9338,369"
PARITY_STEPS=8
SCRATCH="$ROOT/misc/scratch"
mkdir -p "$SCRATCH"
LOG="${DS4_BONSAI_LOG:-$SCRATCH/bonsai-run.log}"
# The long-running server's pid file and log (`serve`).
SERVE_PID="$SCRATCH/bonsai-serve.pid"
SERVE_LOG="$SCRATCH/bonsai-serve.log"
WAIT="${DS4_BONSAI_WAIT:-300}"
# The engine takes a single global flock (/tmp/ds4.lock), so a second model
# process refuses to start.  This script waits for the slot instead.
LOCK=/tmp/ds4.lock

die() { echo "ERROR: $*" >&2; exit 1; }

# True when this ds4-c was linked against the CUDA runtime.  "make cpu" builds
# a CPU-only binary with no libcudart, which cannot do --cuda.
cuda_capable() {
  ldd "$BIN" 2>/dev/null | grep -q 'libcudart'
}

check_env() {
  [ -x "$BIN" ] || die "ds4-c not found or not executable at $BIN (build it: make ds4-c CUDA_ARCH=sm_89)"
  [ -f "$MODEL" ] || die "model not found: $MODEL"
  if [ "$BACKEND" = "cuda" ]; then
    cuda_capable || die "this ds4-c is the CPU-only build (make cpu); it cannot do --cuda. Rebuild with 'make ds4-c CUDA_ARCH=sm_89', or run with DS4_BONSAI_BACKEND=cpu / --cpu."
    command -v nvidia-smi >/dev/null || die "nvidia-smi not found; the CUDA graph needs it (use --cpu)"
  fi
}

# True while another ds4 family process holds the device slot.
slot_busy() {
  pgrep -x ds4-c >/dev/null 2>&1 && return 0
  pgrep -x ds4 >/dev/null 2>&1 && return 0
  pgrep -x ds4-server >/dev/null 2>&1 && return 0
  fuser "$LOCK" >/dev/null 2>&1 && return 0
  return 1
}

# Wait for the single-process slot.  Only one ds4/ds4-c may run at a time.
wait_slot() {
  local waited=0
  while slot_busy; do
    if [ "$waited" -ge "$WAIT" ]; then
      echo "note: another ds4 process held the slot for ${WAIT}s; trying anyway" >&2
      return 0
    fi
    if [ "$waited" -eq 0 ]; then
      echo "waiting for a free device slot (another ds4 process is running)..." >&2
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

# Generated text from a log: a token line carries the id and the raw text.
# A token whose text is a newline leaves an empty remainder on its own line, so
# an empty remainder is printed as a newline rather than dropped.
continuation() {
  awk '/^token /{sub(/^token [0-9]+: [0-9]+ /,""); if (length($0)==0) printf "\n"; else printf "%s", $0}' "$1"
}

# One decoding run: run_model <backend> <steps> <prompt> <log> [token override] [session]
# Sets DS4_CUDA_COPY_MODEL for CUDA, waits for the slot, retries if the engine
# refuses to start, and writes the wall time and peak RSS to <log>.time.
run_model() {
  local backend="$1" steps="$2" prompt="$3" log="$4" tokens="${5:-}" session="${6:-}"
  local tfile="$log.time" rc=0 deadline=$((SECONDS + WAIT))
  local -a envs

  while :; do
    wait_slot
    envs=( "DS4_QWEN35_STEPS=$steps" )
    [ "$backend" = "cuda" ] && envs+=( "DS4_CUDA_COPY_MODEL=1" )
    [ -n "$tokens" ] && envs+=( "DS4_QWEN35_TOKENS=$tokens" )
    [ -n "$session" ] && envs+=( "DS4_QWEN35_SESSION=1" )
    /usr/bin/time -f '__wall %e\n__rss %M' env "${envs[@]}" \
        "$BIN" -m "$MODEL" --"$backend" --first-token-test -p "$prompt" \
        > "$log" 2> "$tfile"
    rc=$?
    grep -q 'refusing to start' "$tfile" 2>/dev/null || break
    [ "$SECONDS" -lt "$deadline" ] || break
    sleep 5
  done

  if [ "$rc" -ne 0 ]; then
    echo "run failed (exit $rc); last lines of $tfile:"
    tail -5 "$tfile"
    return 1
  fi
  return 0
}

# Wall seconds and peak RSS from the /usr/bin/time capture.
wall_of() { awk '/^__wall /{print $2}' "$1" | tail -1; }
rss_of()  { awk '/^__rss /{print $2}' "$1" | tail -1; }

report_run() {
  local log="$1" steps="$2" wall rss
  wall="$(wall_of "$log.time")"
  rss="$(rss_of "$log.time")"
  echo "wall:         ${wall}s for the prompt plus $steps greedy tokens (includes the model load)"
  printf 'decode rate:  %s tokens/s (steps / wall; steady state is higher)\n' \
    "$(awk -v w="$wall" -v n="$steps" 'BEGIN{if (w>0) printf "%.2f", n/w; else print "n/a"}')"
  [ -n "$rss" ] && printf 'peak rss:     %s MiB\n' "$(awk -v k="$rss" 'BEGIN{printf "%.0f", k/1024}')"
  echo "continuation:"
  printf '  %s\n' "$(continuation "$log")"
}

run_mode() {
  check_env
  echo "model:   $MODEL"
  echo "prompt:  $PROMPT"
  echo "steps:   $STEPS"
  echo "backend: $BACKEND"
  echo "note:    one token per forward; there is no batched prefill yet"
  echo
  run_model "$BACKEND" "$STEPS" "$PROMPT" "$LOG" || return 1
  echo "backend:      $BACKEND"
  report_run "$LOG" "$STEPS"
  echo "log:          $LOG"
}

# Both backends on the same prompt, diffed token id for token id.  This is the
# correctness claim this tree can make: the CUDA graph reproduces the CPU
# reference, which is the oracle the kernels were measured against.
compare_mode() {
  check_env
  echo "model:   $MODEL"
  echo "prompt:  $PROMPT"
  echo "steps:   $STEPS, generated by both backends"
  echo
  echo "--- CUDA graph ---"
  run_model cuda "$STEPS" "$PROMPT" "$LOG" || return 1
  echo "backend:      cuda"
  report_run "$LOG" "$STEPS"
  grep -E '^token ' "$LOG" > $SCRATCH/bonsai-cuda.tokens
  echo
  echo "--- CPU reference (the oracle) ---"
  run_model cpu "$STEPS" "$PROMPT" "$LOG" || return 1
  echo "backend:      cpu"
  report_run "$LOG" "$STEPS"
  grep -E '^token ' "$LOG" > $SCRATCH/bonsai-cpu.tokens
  echo
  echo "--- token-for-token diff ---"
  if diff -q $SCRATCH/bonsai-cpu.tokens $SCRATCH/bonsai-cuda.tokens >/dev/null; then
    echo "IDENTICAL: all $STEPS generated token ids agree, so the CUDA graph"
    echo "           reproduces the CPU reference on this prompt"
    echo "           (per-backend token lines kept at misc/scratch/bonsai-{cpu,cuda}.tokens)"
  else
    echo "DIFFERENT - first differences (cpu vs cuda):"
    diff $SCRATCH/bonsai-cpu.tokens $SCRATCH/bonsai-cuda.tokens | head -20
    return 1
  fi
}

# The explicit five-id parity gate, byte for byte what "make bonsai-cuda-parity"
# runs: the same ids through both backends at the same step count.
ids_mode() {
  check_env
  echo "model:   $MODEL"
  echo "tokens:  $PARITY_TOKENS (explicit ids; the prompt is ignored)"
  echo "steps:   $PARITY_STEPS"
  echo
  echo "--- CUDA graph ---"
  run_model cuda "$PARITY_STEPS" x "$LOG" "$PARITY_TOKENS" || return 1
  echo "backend:      cuda"
  report_run "$LOG" "$PARITY_STEPS"
  grep -E '^token ' "$LOG" > $SCRATCH/bonsai-cuda.tokens
  echo
  echo "--- CPU reference (the oracle) ---"
  run_model cpu "$PARITY_STEPS" x "$LOG" "$PARITY_TOKENS" || return 1
  echo "backend:      cpu"
  report_run "$LOG" "$PARITY_STEPS"
  grep -E '^token ' "$LOG" > $SCRATCH/bonsai-cpu.tokens
  echo
  echo "--- token-for-token diff ---"
  if diff -q $SCRATCH/bonsai-cpu.tokens $SCRATCH/bonsai-cuda.tokens >/dev/null; then
    echo "IDENTICAL: both backends print the same $PARITY_STEPS ids for the same"
    echo "           explicit prompt; this is the reproducible parity gate"
  else
    echo "DIFFERENT - first differences (cpu vs cuda):"
    diff $SCRATCH/bonsai-cpu.tokens $SCRATCH/bonsai-cuda.tokens | head -20
    return 1
  fi
}

bench_mode() {
  check_env
  local n="$1"
  echo "model:   $MODEL"
  echo "tokens:  $n greedy, one forward each, after the prompt"
  echo "timed:   /usr/bin/time"
  echo
  run_model "$BACKEND" "$n" "$DEFAULT_PROMPT" "$LOG" || return 1
  echo "backend:      $BACKEND"
  report_run "$LOG" "$n"
  echo "log:          $LOG"
}

status_mode() {
  echo "binary:  $BIN"
  if [ -x "$BIN" ]; then
    echo "         present, built $(stat -c '%y' "$BIN" | cut -d. -f1)"
    if cuda_capable; then
      echo "         CUDA-linked: runs --cuda and --cpu (both backends)"
    else
      echo "         CPU-only build (make cpu): --cpu only, --cuda is refused"
    fi
  else
    echo "         MISSING or not executable (build it: make ds4-c CUDA_ARCH=sm_89)"
  fi
  echo "model:   $MODEL"
  if [ -f "$MODEL" ]; then
    echo "         present, $(stat -c '%s' "$MODEL" | awk '{printf "%.2f GiB", $1/1073741824}')"
  else
    echo "         MISSING"
  fi
  if command -v nvidia-smi >/dev/null; then
    echo "gpu:     $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
    local busy
    busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | head -3)
    if [ -n "$busy" ]; then
      echo "         in use by another process:"
      echo "$busy" | sed 's/^/           /'
    else
      echo "         no compute process"
    fi
  else
    echo "gpu:     nvidia-smi not found (the CUDA graph needs it)"
  fi
  if slot_busy; then
    echo "slot:    busy (another ds4/ds4-c holds $LOCK); runs will wait"
  else
    echo "slot:    free"
  fi
  echo "entry:   --first-token-test (greedy diagnostic), the session path, and"
  echo "         the Rust server (./run-bonsai.sh server)"
  echo "         DS4_QWEN35_STEPS=<n>, DS4_QWEN35_TOKENS=<comma ids>,"
  echo "         DS4_QWEN35_SESSION=1 (drive the same run through a session),"
  echo "         DS4_QWEN35_PREFILL_CHUNK=<n> (rows per CUDA prefill forward)"
  if [ -x "$SERVER_BIN" ]; then
    echo "server:  $SERVER_BIN"
    echo "         present; serves this family (id from the GGUF stem, aliases"
    echo "         prism-bonsai-2-27b*) with DS4_BONSAI_CTX=$SERVER_CTX and"
    echo "         DS4_BONSAI_MEM_FLOOR=${MEM_FLOOR}G on port $SERVER_PORT"
  else
    echo "server:  $SERVER_BIN MISSING (build it: make ds4-server CUDA_ARCH=sm_89)"
  fi
  echo "supported: generate, cuda, cpu, compare, ids, session, server, serve, bench, status, help"
  echo "refused:   batching, MTP/DSpark, SSD/disk KV, session snapshots, distributed"
  echo "           ranks (each refused by name; --kv-disk-dir stops the server for"
  echo "           this family rather than staying silently unused)"
  echo "runbooks:  make bonsai-cuda-check, make bonsai-cuda-parity,"
  echo "           make test-qwen35-cuda, make test-qwen35-session,"
  echo "           make test-qwen35-session-multichunk, make test-qwen35-rows,"
  echo "           make pq2-0-test, make bonsai-fold-selftest, make bonsai-ref-check"
}

# Both backends through the real session path (create, sync, eval), each diffed
# against the CPU reference.  The reference is the oracle: the CPU session runs
# the same trunk the reference does, and the CUDA session must print the same
# ids.  Both CPU runs are about 3 s per forward, so this mode takes minutes.
session_mode() {
  check_env
  echo "model:   $MODEL"
  echo "prompt:  $PROMPT"
  echo "steps:   $STEPS through the session path, against the CPU reference"
  echo
  echo "--- CPU reference (the oracle) ---"
  run_model cpu "$STEPS" "$PROMPT" "$LOG" || return 1
  echo "backend:      cpu"
  report_run "$LOG" "$STEPS"
  grep -E '^token ' "$LOG" > $SCRATCH/bonsai-ref.tokens
  echo
  echo "--- CPU session (create, sync, eval) ---"
  run_model cpu "$STEPS" "$PROMPT" "$LOG" "" session || return 1
  echo "backend:      cpu (session)"
  report_run "$LOG" "$STEPS"
  grep -E '^token ' "$LOG" > $SCRATCH/bonsai-cpu-session.tokens
  if ! diff -q $SCRATCH/bonsai-ref.tokens $SCRATCH/bonsai-cpu-session.tokens >/dev/null; then
    echo "DIFFERENT - the CPU session and the CPU reference disagree:"
    diff $SCRATCH/bonsai-ref.tokens $SCRATCH/bonsai-cpu-session.tokens | head -20
    return 1
  fi
  echo "IDENTICAL: the CPU session reproduces the CPU reference"
  echo
  echo "--- CUDA session (create, sync, eval) ---"
  run_model cuda "$STEPS" "$PROMPT" "$LOG" "" session || return 1
  echo "backend:      cuda (session)"
  report_run "$LOG" "$STEPS"
  grep -E '^token ' "$LOG" > $SCRATCH/bonsai-cuda-session.tokens
  echo
  echo "--- token-for-token diff ---"
  if diff -q $SCRATCH/bonsai-ref.tokens $SCRATCH/bonsai-cuda-session.tokens >/dev/null; then
    echo "IDENTICAL: all $STEPS generated token ids agree, so the CUDA session"
    echo "           reproduces the CPU reference, which the CPU session does too"
    echo "           (token lines kept at misc/scratch/bonsai-{ref,cpu-session,cuda-session}.tokens)"
  else
    echo "DIFFERENT - first differences (reference vs cuda session):"
    diff $SCRATCH/bonsai-ref.tokens $SCRATCH/bonsai-cuda-session.tokens | head -20
    return 1
  fi
}

# --- long-running server for a client ---------------------------------------
# `serve` keeps one ds4-server up (open-grok, a script, curl) instead of the
# smoke test `server` runs, which stops the server after one request.  The pid
# file is the only handle stop uses, so a foreign ds4-server is never killed.

serve_pid() {
  [ -f "$SERVE_PID" ] || return 1
  local pid
  pid=$(cat "$SERVE_PID" 2>/dev/null) || return 1
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  # The recorded pid must still be a ds4-server, not a reused number.
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "ds4-server" || return 1
  printf '%s' "$pid"
}

serve_start() {
  local pid waited=0 model
  if pid=$(serve_pid); then
    echo "server:  already running (pid $pid); ./run-bonsai.sh serve stop first"
    return 0
  fi
  [ -x "$SERVER_BIN" ] || die "server binary not found at $SERVER_BIN (build it: make ds4-server CUDA_ARCH=sm_89)"
  [ -f "$MODEL" ] || die "model not found: $MODEL"
  if slot_busy; then
    echo "another ds4 process holds the single model slot; stop it first:" >&2
    pgrep -a -x ds4-server >&2
    pgrep -a -x ds4 >&2
    pgrep -a -x ds4-c >&2
    return 1
  fi
  echo "model:   $MODEL"
  echo "backend: $BACKEND, ctx $SERVER_CTX, memory floor ${MEM_FLOOR}G, port $SERVER_PORT"
  rm -f "$SERVE_LOG"
  local -a envs=()
  [ "$BACKEND" = "cuda" ] && envs+=( "DS4_CUDA_COPY_MODEL=1" )
  setsid env "${envs[@]}" "$SERVER_BIN" -m "$MODEL" --backend "$BACKEND" \
      -c "$SERVER_CTX" --mem-floor-gb "$MEM_FLOOR" --host 127.0.0.1 \
      --port "$SERVER_PORT" > "$SERVE_LOG" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$SERVE_PID"
  while [ "$waited" -lt 300 ]; do
    grep -q "listening on" "$SERVE_LOG" 2>/dev/null && break
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "server exited before listening; last lines:"
      tail -6 "$SERVE_LOG"
      rm -f "$SERVE_PID"
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  if ! grep -q "listening on" "$SERVE_LOG"; then
    echo "server did not start listening within ${waited}s; last lines:"
    tail -6 "$SERVE_LOG"
    kill "$pid" 2>/dev/null
    rm -f "$SERVE_PID"
    return 1
  fi
  echo "server:  up (pid $pid)"
  grep -o 'listening on.*' "$SERVE_LOG" | tail -1 | sed 's/^/         /'
  model=$(curl -s -m 10 "http://127.0.0.1:$SERVER_PORT/v1/models" |
          python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
  echo "base_url: http://127.0.0.1:$SERVER_PORT/v1"
  echo "id:      ${model:-unknown} (aliases: prism-bonsai-2-27b*)"
  echo "log:     $SERVE_LOG"
  echo "note:    one ds4 model at a time; this server holds the slot until stopped"
}

serve_stop() {
  local pid waited=0
  if ! pid=$(serve_pid); then
    echo "server:  not running"
    rm -f "$SERVE_PID"
    return 0
  fi
  kill "$pid" 2>/dev/null
  while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 30 ]; do
    sleep 1
    waited=$((waited + 1))
  done
  rm -f "$SERVE_PID"
  if kill -0 "$pid" 2>/dev/null; then
    echo "server:  pid $pid did not exit within 30s" >&2
    return 1
  fi
  echo "server:  stopped (pid $pid)"
}

serve_status() {
  local pid models
  if pid=$(serve_pid); then
    echo "server:  running (pid $pid)"
    grep -o 'listening on.*' "$SERVE_LOG" 2>/dev/null | tail -1 | sed 's/^/         /'
    echo "base_url: http://127.0.0.1:$SERVER_PORT/v1"
    models=$(curl -s -m 5 "http://127.0.0.1:$SERVER_PORT/v1/models" |
             python3 -c 'import json,sys; m=json.load(sys.stdin)["data"][0]; print(m["id"], "ctx", m["context_length"])' 2>/dev/null)
    if [ -n "$models" ]; then
      echo "serving: $models"
    else
      echo "serving: no answer on /v1/models"
    fi
    command -v nvidia-smi >/dev/null &&
      echo "vram:    $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
  else
    echo "server:  not running"
  fi
}

serve_logs() {
  [ -f "$SERVE_LOG" ] || die "no serve log yet at $SERVE_LOG"
  tail -n "$1" "$SERVE_LOG"
}

# One real request through the OpenAI-compatible Rust server: start it, wait
# for the listener, ask through the model's own ChatML template, print the
# answer and stop the server.  The advertised id is read back from
# /v1/models rather than assumed, so a renamed artifact cannot drift here.
server_mode() {
  local slog="$LOG.server" pid waited=0 model body t0 t1 wall
  [ -x "$SERVER_BIN" ] || die "server binary not found at $SERVER_BIN (build it: make ds4-server CUDA_ARCH=sm_89)"
  [ -f "$MODEL" ] || die "model not found: $MODEL"
  if [ "$BACKEND" = "cuda" ] && ! command -v nvidia-smi >/dev/null; then
    die "nvidia-smi not found; the CUDA server needs it (set DS4_BONSAI_BACKEND=cpu)"
  fi
  echo "model:   $MODEL"
  echo "prompt:  $PROMPT"
  echo "ctx:     $SERVER_CTX, memory floor ${MEM_FLOOR}G, port $SERVER_PORT"
  echo "backend: $BACKEND"
  echo "binary:  $SERVER_BIN"
  echo
  wait_slot
  rm -f "$slog"
  local -a envs=()
  [ "$BACKEND" = "cuda" ] && envs+=( "DS4_CUDA_COPY_MODEL=1" )
  env "${envs[@]}" "$SERVER_BIN" -m "$MODEL" --backend "$BACKEND" -c "$SERVER_CTX" \
      --mem-floor-gb "$MEM_FLOOR" --host 127.0.0.1 --port "$SERVER_PORT" \
      > "$slog" 2>&1 &
  pid=$!
  while [ "$waited" -lt 300 ]; do
    grep -q "listening on" "$slog" 2>/dev/null && break
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "server exited before listening; last lines:"
      tail -6 "$slog"
      wait "$pid" 2>/dev/null
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  if ! grep -q "listening on" "$slog"; then
    echo "server did not start listening within ${waited}s; last lines:"
    tail -6 "$slog"
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    return 1
  fi
  echo "server:  up (pid $pid)"
  grep -o 'listening on.*' "$slog" | tail -1 | sed 's/^/         /'
  model=$(curl -s -m 10 "http://127.0.0.1:$SERVER_PORT/v1/models" |
          python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
  [ -n "$model" ] || { echo "could not read the advertised id from /v1/models"; kill "$pid"; wait "$pid" 2>/dev/null; return 1; }
  echo "id:      $model"
  echo
  t0=$(date +%s.%N)
  body=$(curl -s -m 900 "http://127.0.0.1:$SERVER_PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$SERVER_TOKENS,\"temperature\":0}")
  t1=$(date +%s.%N)
  wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
  printf 'wall:    %ss for the request\n' "$wall"
  printf '%s' "$body" | python3 -c '
import json,sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except Exception:
    print("raw response:", raw[:400]); raise SystemExit(0)
c = d["choices"][0]["message"]
print("finish:  ", d["choices"][0].get("finish_reason"))
print("answer:  ", (c.get("content") or "").strip())
reasoning = (c.get("reasoning_content") or "").strip()
if reasoning: print("reasoning:", reasoning[:240])
print("usage:   ", d.get("usage"))
'
  kill "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  echo
  echo "server:  stopped (log at $slog)"
}

usage() {
  sed -n '5,15p' "$0" | sed 's/^# \{0,1\}//'
}

# --- argument parsing -------------------------------------------------------
PROMPT="$DEFAULT_PROMPT"
case "${1:-}" in
  generate)       shift; [ $# -gt 0 ] && PROMPT="$*"; run_mode ;;
  cuda)           shift; BACKEND=cuda; [ $# -gt 0 ] && PROMPT="$*"; run_mode ;;
  cpu)            shift; BACKEND=cpu; [ $# -gt 0 ] && PROMPT="$*"; run_mode ;;
  compare)        shift; [ $# -gt 0 ] && PROMPT="$*"; compare_mode ;;
  ids)            ids_mode ;;
  bench)          shift; bench_mode "${1:-16}" ;;
  status)         status_mode ;;
  session)        shift; [ $# -gt 0 ] && PROMPT="$*"; session_mode ;;
  server)         shift; [ $# -gt 0 ] && PROMPT="$*"; server_mode ;;
  serve)
    shift
    case "${1:-start}" in
      start|"") serve_start ;;
      stop)     serve_stop ;;
      status)   serve_status ;;
      logs)     shift; serve_logs "${1:-20}" ;;
      *)        die "serve wants start, stop, status or logs (got: $1)" ;;
    esac ;;
  help|-h|--help) usage ;;
  "")             run_mode ;;
  -*)             die "unknown option: $1 (see ./run-bonsai.sh help)" ;;
  *)              PROMPT="$*"; run_mode ;;
esac
