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
# MTP is off unless DS4_GEMMA4_MTP names the drafter GGUF. It then requires
# bf16 full-layer KV (the drafter reads the full layer's bf16 KV,
# ds4.c:74110-74117) and the script refuses the fp8 combination. On the
# Spark the drafter and the full 262144 context fit; on the 12 GiB 4070
# they do not.
#
# Env overrides:
#   DS4_GEMMA4_MODEL      release directory (default /data/vq-artifacts/gguf-family-it-r8,
#                         else $HOME/models/gguf-family-it-r8)
#   DS4_GEMMA4_SERVER_BIN server binary (default <root>/ds4-server)
#   DS4_GEMMA4_HOST       listen address (default 127.0.0.1; 0.0.0.0 for the LAN)
#   DS4_GEMMA4_PORT       port (default 8099)
#   DS4_GEMMA4_CTX        context (default 65536, the largest that fits the
#                         12 GiB card resident with the FP8 full-layer KV; the
#                         Spark takes 262144)
#   DS4_GEMMA4_MEM_FLOOR  memory floor in GiB (default 0)
#   DS4_GEMMA4_SESSION_GRAPH_HEADROOM_MB
#                         extra free VRAM the session-graph admission check
#                         demands beyond its own estimate (default 0; the
#                         engine's own default is 1024, which refuses 32768)
#   DS4_GEMMA4_KV         fp8 (default) or bf16; fp8 refuses MTP by design.
#                         fp8 is what fits ctx 32768 on this card, but its
#                         full-layer attention is scalar: at 32,768 tokens it
#                         prefills 9.6x and decodes 61x slower than bf16
#                         (docs/gemma4-vq.md, "The full-layer KV lane").
#   DS4_GEMMA4_MTP        drafter GGUF path (default empty = no MTP)
#   DS4_GEMMA4_MTP_DRAFT  drafts per round when MTP is on (default 6)
#   DS4_GEMMA4_MTP_MODE   off|auto|on (default on; auto leaves Gemma off)
#   DS4_GEMMA4_RESIDENT   1 (default, weights on the GPU) or 0 (host-mapped)
#   DS4_GEMMA4_SSD        1 runs the discrete SSD stream (--ssd-streaming,
#                         ds4_gemma4_map.inc); 0 (default) keeps the
#                         resident/host-mapped path. With 1 the weights stay
#                         host-mapped (RESIDENT is ignored) and the routed
#                         experts stream from the artifact.
#   DS4_GEMMA4_SSD_CACHE  expert cache for the stream (default 4GB, the 4v
#                         measured 4070 point; auto or a count also work)
#   DS4_GEMMA4_MOE_CPU_LAYERS 0..999 (default 0, clamped to the 30 routed
#                         layers, so 30, 99 and 999 all mean "every layer"):
#                         layers below N keep their
#                         routed experts host-cold and run their top-8 on the
#                         CPU, layers at or above N keep theirs on the device.
#                         This is llama.cpp's -ncmoe shape; 0 is today's
#                         all-GPU placement and 30 (or 99, or 999, the knob
#                         clamps like llama.cpp's) equals DS4_GEMMA4_CPU_MOE=1.
#                         With N > 0 the trunk is not copied: the mandatory
#                         resident spans are promoted by the map install.
#   DS4_GEMMA4_MOE_CACHE_MIB  llama.cpp's --moe-cache-mib: N MiB of VRAM slots
#                         over host-cold experts, filled from the mmap as the
#                         router asks. 0 (default) keeps the lane off. The
#                         experts stay in the mmap; a decode miss takes
#                         DS4_GEMMA4_MOE_MISS (evict, or cpu to compute it on
#                         the host). Pair it with MOE_MISS=cpu for the
#                         "hot on GPU, cold on CPU" mix. The quote charges
#                         the pool plus the graph's workspace, and boot
#                         refuses a session graph that would leave less than
#                         384 MiB free at peak (measured: 5,632 MiB at
#                         chunk 4096 runs, 6,144 MiB does not).
#   DS4_GEMMA4_MOE_STAGING_MIB  pinned host ring the slot fills are copied
#                         through, in MiB: 0 (default) means 256, otherwise
#                         64..2048. Each record is copied from the mmap into
#                         a ring slice and submitted to the upload stream, so
#                         the six copies of one expert queue back to back
#                         instead of each blocking until the device is idle.
#   DS4_GEMMA4_PREFILL    prefill rows per forward (default 512, the native
#                         bind's min(ctx, 512); 1 makes an 8K prompt minutes)
#   DS4_GEMMA4_FAST_ATTN  1 (default) or 0; the eight measured fast options
#                         (the attention round's four and Phase 3's four)
#   DS4_GEMMA4_MMA        1 (default) adds the tensor-core prefill attention
#                         (DS4_GEMMA4_ATTN_MMA=1); 0 keeps the tile kernel.
#                         A measured parity path, not bit identity: the
#                         pinned gate's 8 greedy tokens match and the
#                         frontier KL is 8.6e-5, but the docs prompt
#                         diverges on a near-tie and the logits relL2 is
#                         1.7x over the design gate.
#   DS4_GEMMA4_SPLIT      1 (default) adds the split-K decode attention
#                         (DS4_GEMMA4_ATTN_SPLIT=1); 0 keeps the single-CTA
#                         fast kernel. The per-key arithmetic is the fast
#                         kernel's, the softmax combine order differs; the
#                         synthetic gate is 6.6e-7 and 23 of 24 greedy
#                         tokens match, so the launcher enables it. The
#                         full layers (D = 512, group = 8) run the split on
#                         tensor cores (DS4_GEMMA4_ATTN_SPLIT_MMA=1, the
#                         launcher default; 0 keeps the scalar per-key
#                         kernel): +2.5% decode at ctx 256, +3.0% at 1,024
#                         and +9.5% at 13,447, with the greedy tokens and
#                         the frontier logits identical at 13,447.
#   DS4_GEMMA4_PERSIST    1 (default) adds the persistent layer-book MoE
#                         decode (DS4_GEMMA4_MOE_PERSIST=1); 0 keeps the
#                         per-expert loop. One launch per pass, the layer
#                         book in shared memory, the route sort on the
#                         device: +23.7 percent decode at ctx 256, +20.1 at
#                         1,024 and +19.8 at 13,447. Near-exact, not bit
#                         identical: 26 of 30 layers match the per-expert
#                         loop exactly and the other four differ by at most
#                         9.8e-4 (about one bf16 ULP).
#   DS4_GEMMA4_LOG        server capture (default misc/scratch/gemma4-serve.log)
#
# The FP8 full-layer KV was verified on a short smoke against bf16; its
# long-context gate is still open. If a long answer degrades, rerun with
# DS4_GEMMA4_KV=bf16 and compare on the same prompt.
#
# The FP8 full-layer KV keeps the whole context on the card, so this script
# wants no SSD for it: RESIDENT=1 copies the 9.37 GiB trunk to the device and
# the cache lives in VRAM. Measured on 2026-11-05 (12 GiB card, ctx 65536):
# 10.12 GiB resident at boot, 11.45 GiB with the session graph, 833 MiB free;
# a 61,932-token request served in 55.7 s. The sliding layers' bf16 rings are
# fixed at their 1,024 window, so the context cost is the five full layers'
# E4M3 rows only (2,056 bytes per token per layer, ~337 MB from 32K to 64K).
# ctx 98304 is refused by the serving quote; 65536 is the ceiling.
#
# Speed: the E4M3 rows now take the tensor-core prefill and split-K decode
# (DS4_GEMMA4_MMA and DS4_GEMMA4_SPLIT below), which on the GB10 at 32,768
# tokens is 5.9x the old scalar prefill and 38.6x its decode, and lands at
# 93 percent of the bf16 lane's prefill and above its decode (docs/gemma4-vq.md,
# "The E4M3 tensor-core port"). bf16 still costs more memory for the same
# context, so FP8 stays the lane for this card.

set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
BIN="${DS4_GEMMA4_SERVER_BIN:-$ROOT/ds4-server}"
if [ -n "${DS4_GEMMA4_MODEL:-}" ]; then
  MODEL="$DS4_GEMMA4_MODEL"
elif [ -d "$HOME/models/gguf-family-it-r8" ]; then
  # The streamed experts live here: the fast disk is the SSD lane's point.
  MODEL="$HOME/models/gguf-family-it-r8"
elif [ -d /data/vq-artifacts/gguf-family-it-r8 ]; then
  MODEL=/data/vq-artifacts/gguf-family-it-r8
else
  MODEL="$HOME/models/gguf-family-it-r8"
fi
HOST="${DS4_GEMMA4_HOST:-127.0.0.1}"
PORT="${DS4_GEMMA4_PORT:-8099}"
CTX="${DS4_GEMMA4_CTX:-65536}"
FLOOR="${DS4_GEMMA4_MEM_FLOOR:-0}"
# The Gemma model card's standardized sampling is temperature 1.0, top_p 0.95
# and top_k 64. Temperature and top_p travel in the request (the open-grok
# model block sets them); top_k is not an OpenAI field, so the server carries
# it as its default. 0 disables it.
TOP_K="${DS4_GEMMA4_TOP_K:-64}"
# The session graph's admission check asks for a fixed headroom on top of its
# own estimate; at 1 GiB (the engine default) ctx 32768 is refused on this
# card even though the graph then sits in 1.17 GiB of free VRAM. The measured
# boots (2026-11-05): ctx 32768 uses 11.11 GiB of the 12 with the session
# graph, ctx 49152 11.28, ctx 65536 11.45 (833 MiB free); ctx 98304 is
# refused by the serving quote before it boots.
GRAPH_HEADROOM="${DS4_GEMMA4_SESSION_GRAPH_HEADROOM_MB:-0}"
KV="${DS4_GEMMA4_KV:-fp8}"
MTP="${DS4_GEMMA4_MTP:-}"
DRAFT="${DS4_GEMMA4_MTP_DRAFT:-6}"
MTP_MODE="${DS4_GEMMA4_MTP_MODE:-on}"
RESIDENT="${DS4_GEMMA4_RESIDENT:-1}"
SSD="${DS4_GEMMA4_SSD:-0}"
CPU_MOE="${DS4_GEMMA4_CPU_MOE:-0}"
CPU_MOE_LAYERS="${DS4_GEMMA4_MOE_CPU_LAYERS:-0}"
MOE_CACHE_MIB="${DS4_GEMMA4_MOE_CACHE_MIB:-0}"
MOE_STAGING_MIB="${DS4_GEMMA4_MOE_STAGING_MIB:-0}"
SSD_CACHE="${DS4_GEMMA4_SSD_CACHE:-4GB}"
FAST="${DS4_GEMMA4_FAST_ATTN:-1}"
MMA="${DS4_GEMMA4_MMA:-1}"
SPLIT="${DS4_GEMMA4_SPLIT:-1}"
PERSIST="${DS4_GEMMA4_PERSIST:-1}"
# Disk KV parking: empty keeps the disk lane off (the default, and what the
# owner's endpoint runs). Pointing it at a directory lets the server park a
# session's whole cache, keyed by the rendered text prefix, and restore it on
# a later request instead of re-prefilling (crates/ds4-kv, generate.rs). The
# cold maximum must cover the card's 65,536-token ceiling or a full-context
# session is never parked.
KV_DIR="${DS4_GEMMA4_KV_DIR:-}"
KV_SPACE="${DS4_GEMMA4_KV_SPACE:-32G}"
KV_COLD_MAX="${DS4_GEMMA4_KV_COLD_MAX:-65536}"
# The three knobs above are this script's, not the engine's. The engine warns
# about any DS4_GEMMA4_* it does not read, so they are removed from the child
# environment rather than left to trip that check.
LAUNCHER_ENVS=( -u DS4_GEMMA4_KV_DIR -u DS4_GEMMA4_KV_SPACE
                -u DS4_GEMMA4_KV_COLD_MAX )
# The GB10's unified 128 GB takes a 2,048-row chunk; the 12 GiB 4070 stays
# at 512, the native bind's own default. A wider chunk needs the MoE
# workspace to follow it (ds4_gemma4_graph.inc); 8,192 still fails in the
# router GEMM, and 2,048 is the measured Spark point.
if [ "$(uname -m)" = "aarch64" ]; then PREFILL_DEFAULT=2048; else PREFILL_DEFAULT=512; fi
PREFILL="${DS4_GEMMA4_PREFILL:-$PREFILL_DEFAULT}"
GROK_NAME="${DS4_GEMMA4_GROK_NAME:-gemma4-vq-4070}"
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

# The eight measured fast options are one arm: the attention round's four
# (decode attention, prefill attention, the direct VQ prefill GEMM and the
# batched dense-down Q8) and Phase 3's four (fused RMSNorm+RoPE, parallel
# router top-k, the family's vocabulary head and the family's router dot).
# All eight are bit-identical to the oracle path on their gates
# (docs/gemma4-attention-round-2026-11-03.md, docs/gemma4-phase3-2026-11-03.md);
# FAST=0 restores the oracle path for an A/B.
fast_envs() {
  if [ "$FAST" = "1" ]; then
    FAST_ENVS=( "DS4_GEMMA4_ATTN_FAST=1" "DS4_GEMMA4_ATTN_PREFILL=1"
                "DS4_GEMMA4_VQ_PREFILL=1" "DS4_GEMMA4_Q8_BATCH=1"
                "DS4_GEMMA4_FUSE_ROPE=1" "DS4_GEMMA4_ROUTER_PAR=1"
                "DS4_GEMMA4_HEAD_GEMV=1" "DS4_GEMMA4_ROUTER_DOT=1" )
  else
    FAST_ENVS=( "DS4_GEMMA4_ATTN_FAST=0" "DS4_GEMMA4_ATTN_PREFILL=0"
                "DS4_GEMMA4_VQ_PREFILL=0" "DS4_GEMMA4_Q8_BATCH=0"
                "DS4_GEMMA4_FUSE_ROPE=0" "DS4_GEMMA4_ROUTER_PAR=0"
                "DS4_GEMMA4_HEAD_GEMV=0" "DS4_GEMMA4_ROUTER_DOT=0" )
  fi
  if [ "$MMA" = "1" ]; then FAST_ENVS+=( "DS4_GEMMA4_ATTN_MMA=1" ); fi
  if [ "$SPLIT" = "1" ]; then
    FAST_ENVS+=( "DS4_GEMMA4_ATTN_SPLIT=1"
                 "DS4_GEMMA4_ATTN_SPLIT_MMA=${DS4_GEMMA4_ATTN_SPLIT_MMA:-1}" )
  fi
  if [ "$PERSIST" = "1" ]; then FAST_ENVS+=( "DS4_GEMMA4_MOE_PERSIST=1" ); fi
}

check_env() {
  [ -x "$BIN" ] || die "server binary not found at $BIN (build it: make ds4-server CUDA_ARCH=sm_89)"
  [ -d "$MODEL" ] || die "release directory not found: $MODEL"
  command -v curl >/dev/null || die "curl is required"
  command -v jq >/dev/null || die "jq is required"
  if [ -n "$MTP" ]; then
    [ -f "$MTP" ] || die "drafter not found: $MTP"
    [ "$KV" = "fp8" ] && die "MTP needs bf16 KV (DS4_GEMMA4_KV=bf16): the drafter reads the full layer's bf16 KV"
  fi
}

# The plan the server will actually boot, printed as JSON by --check-config.
plan_json() {
  fast_envs
  local -a extra=() envs=( "DS4_LOCK_FILE=$LOCK" "$(kv_env)"
      "DS4_GEMMA4_PREFILL_CHUNK=$PREFILL" "${FAST_ENVS[@]}"
      "DS4_SESSION_GRAPH_HEADROOM_MB=$GRAPH_HEADROOM"
      "DS4_GEMMA4_MOE_CPU_LAYERS=$CPU_MOE_LAYERS"
      "DS4_GEMMA4_MOE_CACHE_MIB=$MOE_CACHE_MIB"
      "DS4_GEMMA4_MOE_STAGING_MIB=$MOE_STAGING_MIB" )
  if [ "$SSD" = "1" ]; then
    extra+=( --ssd-streaming --ssd-streaming-cache-experts "$SSD_CACHE" )
  fi
  if [ -n "$KV_DIR" ]; then
    extra+=( --kv-disk-dir "$KV_DIR" --kv-disk-space "$KV_SPACE"
             --kv-cache-cold-max-tokens "$KV_COLD_MAX" )
  fi
  # Gemma-4 MTP runs only on the serial lane (mtp_lane); the auto reuse
  # picks the partial/bank lane here, so the drafter pins exact reuse and
  # DS4_SERVER_CONTINUOUS=0 makes the driver serial (serving.rs:581-587).
  if [ -n "$MTP" ]; then
    extra+=( --mtp "$MTP" --mtp-draft "$DRAFT" --mtp-mode "$MTP_MODE" --prefix-reuse exact )
    envs+=( "DS4_SERVER_CONTINUOUS=0" )
  fi
  env "${LAUNCHER_ENVS[@]}" "${envs[@]}" "$BIN" -m "$MODEL" --cuda -c "$CTX" \
      --mem-floor-gb "$FLOOR" --max-seqs 1 --top-k "$TOP_K" "${extra[@]}" \
      --check-config 2>/dev/null
}

plan_mode() {
  check_env
  local json
  json="$(plan_json)" || die "the plan did not print (server error)"
  printf '%s' "$json" | jq '{quote, issues, effective}'
}

# Refuse before boot when the quote does not fit; the numbers are the message.
#
# The quote is the plan-time estimate and prices what the engine will
# allocate: the device-resident spans of a host-cold lane, the expert slot
# pool, the KV rows and the session graph's workspace. It is taken before
# CUDA exists, so it cannot see the process's own runtime; the exact verdict
# is the engine's at boot, after the pool and the spans are installed, with
# the family's 384 MiB peak floor (ds4_gemma4_plan.h G4_SESSION_FIT_MARGIN).
# An arm can therefore pass here and still be refused at boot, with the
# workspace bytes named -- that is the intended two-tier behaviour.
preflight() {
  local json code
  json="$(plan_json)" || die "the plan did not print (server error)"
  code="$(printf '%s' "$json" | jq -r '[.issues[] | select(.level == "error") | .code] | join(",")')"
  if [ -n "$code" ]; then
    printf '%s' "$json" | jq '{quote, issues}'
    die "the serving quote refuses this configuration ($code); lower DS4_GEMMA4_CTX, DS4_GEMMA4_PREFILL or DS4_GEMMA4_MOE_CACHE_MIB"
  fi
  printf '%s' "$json" | jq -c '{quote: (.quote|{shared_weights,expert_cache,per_bank,scratch,floor,available,total}), issues: [.issues[].code]}'
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
  if [ "$SSD" = "1" ]; then
    echo "backend: cuda (weights streamed from SSD, cache $SSD_CACHE), ctx $CTX, floor ${FLOOR}G, max-seqs 1, host $HOST:$PORT"
  elif [ "$RESIDENT" = "1" ]; then
    echo "backend: cuda (weights on the GPU), ctx $CTX, floor ${FLOOR}G, max-seqs 1, host $HOST:$PORT"
  else
    echo "backend: cuda (host-mapped weights), ctx $CTX, floor ${FLOOR}G, max-seqs 1, host $HOST:$PORT"
  fi
  echo "kv:      $KV full layers; fast attention $FAST; mtp ${MTP:-off}"
  if [ "$CPU_MOE" = "1" ]; then
    echo "moe:     all 30 layers on the host (CPU_MOE=1)"
  elif [ "$CPU_MOE_LAYERS" != "0" ]; then
    echo "moe:     layers 0..$((CPU_MOE_LAYERS - 1)) on the host, the rest on the device"
  elif [ "$MOE_CACHE_MIB" != "0" ]; then
    echo "moe:     host-cold experts with a ${MOE_CACHE_MIB} MiB GPU slot cache"
    echo "upload:  pinned staging ring $(( MOE_STAGING_MIB == 0 ? 256 : MOE_STAGING_MIB )) MiB, queued copies"
  fi
  preflight
  rm -f "$LOG"
  fast_envs
  local -a envs=( "DS4_LOCK_FILE=$LOCK" "$(kv_env)"
      "DS4_GEMMA4_PREFILL_CHUNK=$PREFILL" "${FAST_ENVS[@]}"
      "DS4_SESSION_GRAPH_HEADROOM_MB=$GRAPH_HEADROOM" )
  [ "$RESIDENT" = "1" ] && [ "$SSD" != "1" ] && [ "$CPU_MOE" != "1" ] && \
    [ "$CPU_MOE_LAYERS" = "0" ] && [ "$MOE_CACHE_MIB" = "0" ] && \
    envs+=( "DS4_CUDA_COPY_MODEL=1" )
  # Per-layer host placement: the engine reads this itself, so it is passed
  # explicitly and echoed with the rest of the placement.
  envs+=( "DS4_GEMMA4_MOE_CPU_LAYERS=$CPU_MOE_LAYERS" )
  envs+=( "DS4_GEMMA4_MOE_CACHE_MIB=$MOE_CACHE_MIB" )
  envs+=( "DS4_GEMMA4_MOE_STAGING_MIB=$MOE_STAGING_MIB" )
  local -a extra=()
  # SSD streaming (ds4_gemma4_map.inc, the discrete arm): resident spans are
  # promoted to the device and the routed experts stream from the artifact
  # with the given cache. The weights stay host-mapped, so RESIDENT=1 would
  # copy the whole model and defeat the lane; it is ignored here.
  if [ "$SSD" = "1" ]; then
    extra+=( --ssd-streaming --ssd-streaming-cache-experts "$SSD_CACHE" )
  fi
  if [ -n "$KV_DIR" ]; then
    extra+=( --kv-disk-dir "$KV_DIR" --kv-disk-space "$KV_SPACE"
             --kv-cache-cold-max-tokens "$KV_COLD_MAX" )
    echo "kv disk: $KV_DIR (space $KV_SPACE, cold max $KV_COLD_MAX tokens)"
  fi
  # Gemma-4 MTP runs only on the serial lane (mtp_lane); the auto reuse
  # picks the partial/bank lane here, so the drafter pins exact reuse and
  # DS4_SERVER_CONTINUOUS=0 makes the driver serial (serving.rs:581-587).
  if [ -n "$MTP" ]; then
    extra+=( --mtp "$MTP" --mtp-draft "$DRAFT" --mtp-mode "$MTP_MODE" --prefix-reuse exact )
    envs+=( "DS4_SERVER_CONTINUOUS=0" )
  fi
  setsid env "${LAUNCHER_ENVS[@]}" "${envs[@]}" \
      "$BIN" -m "$MODEL" --cuda --host "$HOST" --port "$PORT" \
      -c "$CTX" --mem-floor-gb "$FLOOR" --max-seqs 1 --top-k "$TOP_K" "${extra[@]}" \
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
  local base_host="$HOST"
  [ "$HOST" = "0.0.0.0" ] && base_host=$(hostname -I 2>/dev/null | awk '{print $1}')
  [ -n "$base_host" ] || base_host=127.0.0.1
  echo "server:  up (pid $pid)"
  grep -o 'listening on.*' "$LOG" | tail -1 | sed 's/^/         /'
  echo "base_url: http://$base_host:$PORT/v1"
  echo "model id: $(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].id // "unknown"')"
  echo "log:     $LOG"
  echo "note:    one ds4 model at a time; this server holds the slot until stopped"
  echo
  echo "open-grok config block ([model.$GROK_NAME]):"
  echo "  [model.\"$GROK_NAME\"]"
  echo "  model = \"$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].id // "gguf-family-it-r8"')\""
  echo "  base_url = \"http://$base_host:$PORT/v1\""
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
  fast_envs
  local -a envs=( "DS4_LOCK_FILE=$LOCK" "$(kv_env)"
      "DS4_GEMMA4_PREFILL_CHUNK=$PREFILL" "${FAST_ENVS[@]}"
      "DS4_SESSION_GRAPH_HEADROOM_MB=$GRAPH_HEADROOM" )
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
