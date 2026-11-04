#!/bin/sh
# gemma4_spark_sweep.sh -- the Gemma-4 26B A4B VQ frontier sweep in the
# speed-bench CSV standard.
#
# The measured surface is the deployed route, the same line run-gemma4.sh
# serve start runs: one process, one greedy chat request per frontier, the
# contexts grown from the pinned speed-bench/promessi_sposi.txt corpus, the
# CSV columns plot_speed.py reads, clocks sampled across every request.
# Rates are the server's own timings (crates/ds4-server/src/stream.rs), never
# a wall-clock computation, so the numbers are the ones a client sees.
#
# Protocol: temperature 0, 128 generated tokens per frontier, contexts
# 2048..65536 step 2048 by default. Single-turn chat renders do not extend
# the previous prompt's checkpoint, so these rows measure full prefill;
# prefill_cached_tokens records any actual reuse.
#
# usage:
#   boot mode (fresh process; the box must be free):
#     gemma4_spark_sweep.sh <server> <release_dir> [port]
#   attach mode (measures a server already listening; never kills anything):
#     G4_SWEEP_URL=http://127.0.0.1:8099 gemma4_spark_sweep.sh
#
# env:
#   G4_SWEEP_CTX         frontier list (default "2048 4096 ... 65536")
#   G4_SWEEP_GEN         greedy tokens per frontier (default 128)
#   G4_SWEEP_CSV         output CSV (default <here>/gb10_spark_gemma4_<bin8>.csv)
#   G4_SWEEP_OUT         log/meta dir (default /tmp/gemma4_sweep_<bin8>)
#   G4_SWEEP_URL         attach-mode base URL (skips boot)
#   G4_SWEEP_PORT_ARGS   extra boot args (boot mode only; recorded)
#   G4_SWEEP_CORPUS      corpus override (default speed-bench/promessi_sposi.txt)
#   G4_SWEEP_NO_MODEL_SHA  1 = record part sizes only (the release is ~10 GB)
#   G4_SWEEP_NO_SVG      1 = skip the plot_speed.py pass
#   G4_SWEEP_TITLE       plot title (default "GB10 Spark Gemma-4 26B A4B VQ <bin8> t/s")
#
# The record lands beside the CSV as <csv>.meta.txt (artifact, build, backend,
# workload, clocks -- the README's "record when adding results").
set -u

HERE=$(cd "$(dirname "$0")" && pwd)

CTXS="${G4_SWEEP_CTX:-$(seq 2048 2048 65536 | tr '\n' ' ')}"
GEN="${G4_SWEEP_GEN:-128}"
CORPUS="${G4_SWEEP_CORPUS:-$HERE/promessi_sposi.txt}"
PORT_EXTRA="${G4_SWEEP_PORT_ARGS:-}"
URL="${G4_SWEEP_URL:-}"

# The server line the launcher runs on the Spark (run-gemma4.sh, aarch64):
# the fast set, the persistent MoE, the 2048-row prefill chunk, the weights
# copied to the device. The split span is the shipped default.
FAST_ENVS="DS4_GEMMA4_ATTN_FAST=1 DS4_GEMMA4_ATTN_PREFILL=1 \
DS4_GEMMA4_VQ_PREFILL=1 DS4_GEMMA4_Q8_BATCH=1 DS4_GEMMA4_FUSE_ROPE=1 \
DS4_GEMMA4_ROUTER_PAR=1 DS4_GEMMA4_HEAD_GEMV=1 DS4_GEMMA4_ROUTER_DOT=1 \
DS4_GEMMA4_ATTN_MMA=1 DS4_GEMMA4_ATTN_SPLIT=1 DS4_GEMMA4_ATTN_SPLIT_MMA=1 \
DS4_GEMMA4_MOE_PERSIST=1"
PREFILL="${G4_SWEEP_PREFILL:-2048}"

CAL_BYTES=8192          # ~2.4K tokens at the measured ratio; offset 0
CAL_OFF=0
MSR_OFF=262144          # measured prompt base: never overlaps the calibration
CURL_TIMEOUT=3600
BOOT_TRIES=900          # x BOOT_SLEEP = 30 min
BOOT_SLEEP=2
CLK_SLEEP=1

fail() { echo "gemma4 sweep: FAIL ($1)" >&2; exit 1; }
note() { echo "gemma4 sweep: $1"; }

for tool in jq curl sha256sum awk; do
    command -v "$tool" >/dev/null 2>&1 || fail "$tool not found"
done
[ -f "$CORPUS" ] || fail "corpus not found: $CORPUS"

# ------------------------------------------------------------------- target

MODE=attach
SRV_PID=""
if [ -z "$URL" ]; then
    MODE=boot
    SRV="${1:-}"; MODEL="${2:-}"; PORT="${3:-8099}"
    [ -n "$SRV" ] && [ -n "$MODEL" ] || \
        fail "usage: gemma4_spark_sweep.sh <server> <release_dir> [port] (or set G4_SWEEP_URL)"
    [ -x "$SRV" ] || fail "server not executable: $SRV"
    [ -d "$MODEL" ] || fail "release dir not found: $MODEL"
    URL="http://127.0.0.1:$PORT"
    BIN_SHA=$(sha256sum "$SRV" | awk '{print $1}')
else
    PID=$(pgrep -x ds4-server | head -1)
    [ -n "$PID" ] || fail "attach mode found no ds4-server process (pgrep -x)"
    SRV=$(readlink -f "/proc/$PID/exe") || fail "cannot resolve /proc/$PID/exe"
    BIN_SHA=$(sha256sum "$SRV" | awk '{print $1}')
    CMDLINE=$(tr '\0' ' ' < "/proc/$PID/cmdline")
fi
case "$URL" in
    */v1/chat/completions) : ;;
    *) URL="${URL%/}/v1/chat/completions" ;;
esac
BIN8=$(printf '%s' "$BIN_SHA" | cut -c1-8)

CSV="${G4_SWEEP_CSV:-$HERE/gb10_spark_gemma4_${BIN8}.csv}"
OUT="${G4_SWEEP_OUT:-/tmp/gemma4_sweep_${BIN8}}"
mkdir -p "$OUT" || fail "cannot create $OUT"

# ---------------------------------------------------------------- primitives

now() { date +%s.%N; }

is_int() { case "$1" in ''|*[!0-9]*) return 1 ;; *) return 0 ;; esac; }

corpus_slice() { # <offset-bytes> <bytes> <out>
    tail -c "+$(( $1 + 1 ))" "$CORPUS" | head -c "$2" > "$3" || true
}

median_of() { # <file-of-numbers>
    sort -n "$1" 2>/dev/null | awk 'NF { a[++n] = $1 }
        END { if (n == 0) { print ""; exit }
              print (n % 2) ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2 }'
}

start_clocks() { # <out-file>
    : > "$1"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    ( while :; do
          nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits 2>/dev/null >> "$1"
          sleep "$CLK_SLEEP"
      done ) &
    CLK_PID=$!
}

stop_clocks() {
    [ -n "${CLK_PID:-}" ] || return 0
    kill "$CLK_PID" 2>/dev/null
    wait "$CLK_PID" 2>/dev/null
    CLK_PID=""
}

cleanup() {
    [ -n "${CLK_PID:-}" ] && kill "$CLK_PID" 2>/dev/null
    [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
}
trap cleanup EXIT

# ------------------------------------------------------------------ boot/run

B_S=""
CTX_MAX=$(printf '%s\n' $CTXS | sort -n | tail -1)
boot_server() {
    pgrep -x ds4-server >/dev/null 2>&1 && \
        fail "a ds4-server is already running; boot mode needs the box free (use G4_SWEEP_URL to attach)"
    SLOG="$OUT/server.log"
    rm -f "$SLOG"
    B_T0=$(now)
    # shellcheck disable=SC2086
    # env, not assignment prefixes: $FAST_ENVS is expanded, and the shell
    # only recognizes NAME=value prefixes lexically.
    env DS4_LOCK_FILE="$OUT/lock" DS4_CUDA_COPY_MODEL=1 \
        DS4_GEMMA4_PREFILL_CHUNK="$PREFILL" $FAST_ENVS \
        "$SRV" -m "$MODEL" --cuda --host 127.0.0.1 --port "$PORT" \
        -c "$CTX_MAX" --mem-floor-gb 1 --max-seqs 1 $PORT_EXTRA \
        >> "$SLOG" 2>&1 &
    SRV_PID=$!
    i=0
    while [ "$i" -lt "$BOOT_TRIES" ]; do
        grep -q "listening on" "$SLOG" 2>/dev/null && break
        kill -0 "$SRV_PID" 2>/dev/null || fail "server died during boot (see $SLOG)"
        sleep "$BOOT_SLEEP"
        i=$(( i + 1 ))
    done
    grep -q "listening on" "$SLOG" 2>/dev/null || fail "server did not start listening (see $SLOG)"
    B_S=$(awk -v a="$B_T0" -v b="$(now)" 'BEGIN { printf "%.1f", b - a }')
    note "booted in ${B_S}s (log $SLOG)"
}

# One request; leaves R_CODE R_WALL R_CLK.  A transport stall is a failure;
# an HTTP error is the caller's to record and skip.
run_request() { # <tag> <req-file> <resp-file>
    clkf="$OUT/$1.clk"
    start_clocks "$clkf"
    t0=$(now)
    R_CODE=$(curl -sS -m "$CURL_TIMEOUT" -o "$3" -w '%{http_code}' \
        "$URL" -H 'content-type: application/json' -d @"$2") \
      || { stop_clocks; fail "$1 transport failure"; }
    t1=$(now)
    stop_clocks
    cat "$clkf" >> "$OUT/clocks.all" 2>/dev/null || true
    R_WALL=$(awk -v a="$t0" -v b="$(now)" 'BEGIN { printf "%.3f", b - a }')
    R_CLK=$(median_of "$clkf")
}

# -------------------------------------------------------------------- driver

{
    echo "gemma4 speed sweep (speed-bench standard)"
    echo "started $(date -Iseconds)"
    echo "mode $MODE"
    if [ "$MODE" = "attach" ]; then
        echo "server /proc/$PID/exe -> $SRV sha256 $BIN_SHA"
        echo "cmdline $CMDLINE"
    else
        echo "server $SRV sha256 $BIN_SHA"
    fi
    echo "ctx $CTXS; gen $GEN; prefill chunk $PREFILL; port extra args: ${PORT_EXTRA:-<none>}"
    echo "corpus $CORPUS sha256 $(sha256sum "$CORPUS" | awk '{print $1}')"
    if [ "$MODE" = "boot" ]; then
        echo "model $MODEL"
        for part in "$MODEL"/*.gguf; do
            [ -f "$part" ] || continue
            if [ "${G4_SWEEP_NO_MODEL_SHA:-0}" = "1" ]; then
                echo "part $(basename "$part") bytes $(stat -c %s "$part") sha256 skipped"
            else
                echo "part $(basename "$part") bytes $(stat -c %s "$part") sha256 $(sha256sum "$part" | awk '{print $1}')"
            fi
        done
    fi
} | tee "$OUT/meta.txt"

[ "$MODE" = "boot" ] && boot_server

# Calibration: the bytes-per-token ratio of the corpus is measured, not
# assumed.  Offset 0 keeps the probe off the measured slice, so the probe's
# checkpoint can never warm a frontier's prompt.
corpus_slice "$CAL_OFF" "$CAL_BYTES" "$OUT/cal.txt"
jq -n --rawfile p "$OUT/cal.txt" --argjson mt 1 \
    '{messages:[{role:"user",content:$p}],max_tokens:$mt,temperature:0}' \
    > "$OUT/cal.json"
run_request cal "$OUT/cal.json" "$OUT/cal.resp.json"
[ "$R_CODE" = "200" ] || fail "calibration HTTP $R_CODE"
cal_tok=$(jq -r '.timings.prefill_tokens // empty' "$OUT/cal.resp.json")
is_int "$cal_tok" && [ "$cal_tok" -gt 0 ] || fail "calibration reported no token count"
ratio=$(awk -v b="$CAL_BYTES" -v t="$cal_tok" 'BEGIN { printf "%.4f", b / t }')
note "calibration: $cal_tok tokens over $CAL_BYTES bytes, ratio $ratio bytes/token"

corpus_bytes=$(wc -c < "$CORPUS")
need=$(awk -v t="$CTX_MAX" -v r="$ratio" 'BEGIN { printf "%d", t * r }')
avail=$(( corpus_bytes - MSR_OFF ))
[ "$need" -le "$avail" ] || \
    fail "ctx $CTX_MAX needs $need bytes past offset $MSR_OFF but the corpus has $avail"

: > "$CSV"
echo "ctx_tokens,prefill_tokens,prefill_cached_tokens,prefill_tps,gen_tokens,gen_tps,first_token_sec,tok_per_step,wall_s,clock_mhz" >> "$CSV"

rows=0
for ctx in $CTXS; do
    bytes=$(awk -v t="$ctx" -v r="$ratio" 'BEGIN { printf "%d", t * r }')
    corpus_slice "$MSR_OFF" "$bytes" "$OUT/prompt_$ctx.txt"
    jq -n --rawfile p "$OUT/prompt_$ctx.txt" --argjson mt "$GEN" \
        '{messages:[{role:"user",content:$p}],max_tokens:$mt,temperature:0}' \
        > "$OUT/req_$ctx.json"
    run_request "ctx_$ctx" "$OUT/req_$ctx.json" "$OUT/resp_$ctx.json"

    if [ "$R_CODE" != "200" ]; then
        note "ctx=$ctx: HTTP $R_CODE, row skipped"
        continue
    fi

    pftok=$(jq -r '.timings.prefill_tokens // empty' "$OUT/resp_$ctx.json")
    pfc=$(jq -r '.timings.prefill_cached_tokens // empty' "$OUT/resp_$ctx.json")
    pftps=$(jq -r '.timings.prefill_tok_s // empty' "$OUT/resp_$ctx.json")
    dctok=$(jq -r '.usage.completion_tokens // empty' "$OUT/resp_$ctx.json")
    dctps=$(jq -r '.timings.decode_tok_s // empty' "$OUT/resp_$ctx.json")
    ttft=$(jq -r '.timings.ttft_ms // empty' "$OUT/resp_$ctx.json")
    tps=$(jq -r '.timings.tok_per_step // empty' "$OUT/resp_$ctx.json")
    if [ -z "$pftok" ] || [ -z "$pftps" ] || [ -z "$dctps" ]; then
        note "ctx=$ctx: no timings in the response, row skipped (see $OUT/resp_$ctx.json)"
        continue
    fi

    fts=$(awk -v ms="$ttft" 'BEGIN { printf "%.4f", ms / 1000 }')
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$ctx" "$pftok" "$pfc" "$pftps" "$dctok" "$dctps" "$fts" "$tps" "$R_WALL" "${R_CLK:-}" \
        >> "$CSV"
    rows=$(( rows + 1 ))
    note "ctx=$ctx: prefill $pftok tok (cached $pfc) @ $pftps t/s, decode $dctok tok @ $dctps t/s, ttft $ttft ms, wall $R_WALL s, clock ${R_CLK:-?} MHz"
done

# ------------------------------------------------------------------- record

all_clk=$(median_of "$OUT/clocks.all")
{
    echo "csv $CSV ($rows rows)"
    echo "clock median over all samples ${all_clk:-?} MHz"
    echo "boot_s ${B_S:-n/a}"
    echo "reuse: route default (on unless the boot line passed --no-kv-reuse);"
    echo "       prefill_cached_tokens per row shows what the checkpoint restored"
    echo "       (0 = full prefill -- single-turn renders do not extend the"
    echo "       previous prompt, so this sweep reads full-context prefill_tokens)"
    echo "finished $(date -Iseconds)"
} | tee -a "$OUT/meta.txt"
cp "$OUT/meta.txt" "$CSV.meta.txt"

SVG=""
TITLE="${G4_SWEEP_TITLE:-GB10 Spark Gemma-4 26B A4B VQ ${BIN8} t/s}"
if [ "${G4_SWEEP_NO_SVG:-0}" != "1" ] && command -v python3 >/dev/null 2>&1; then
    if python3 "$HERE/plot_speed.py" "$CSV" --title "$TITLE" >/dev/null 2>&1; then
        SVG="${CSV%.csv}_ts.svg"
    fi
fi

[ "$rows" -gt 0 ] || fail "no frontier produced a row (see $OUT)"
note "wrote $CSV ($rows rows), ${SVG:-no svg}, meta $CSV.meta.txt"
