#!/bin/sh
# ds41 performance sweep (campaign step 0, handoff section 3 item 4).
#
# Measures one side at a time on the canonical deployment line so the Spark
# window is spent measuring, not discovering the harness.
#
# WHY THIS INSTRUMENT, and not ds4-perf/ds4-bench (checked first, as rule 0.2
# requires):
#   - ds4-bench-perf's context sweep cannot touch this family.  The CLI's V4.1
#     route takes raw token ids and refuses a prompt outright -- "the V4.1
#     family runs on --gen-ids <file> (raw token ids; its chat rendering is not
#     ported yet)" (crates/ds4-cli/src/lib.rs:680-692) -- so --prompt-file with
#     --ctx-start/--ctx-max/--step-incr has no ds41 path behind it, and the
#     V4.1 route has no --ctx at all (docs/serving-capabilities.md:69).
#   - ds4-perf serving grades serving contracts (declared reuse_kind, family
#     match, resolved plan) and measures HTTP latency; it states that it does
#     not measure token-level decode throughput (docs/ds4-perf.md, the serving
#     section), and the V4.1 path has no reuse to grade.
#   So the deployed server's own timings block is the only source that reports
#   prefill, decode, ttft and tokens/step on this route
#   (crates/ds4-server/src/stream.rs:363-380).  Using the server's own numbers
#   beats an external wall clock, and this harness never computes a rate it
#   could have read.
#
# One boot per (side, context) step, because the main GGUF is ~113 GB: the two
# sides cannot be resident together, and the recorded protocol asks for a fresh
# process per run with the clocks logged beside every number
# (docs/qwen38-benchmark-reference.md:56-84 -- independent frontiers, 128 greedy
# outputs, interleaved runs, medians).
#
# --v41-prof IS A PREFILL-ONLY INSTRUMENT (learned the hard way, 2026-10-12):
# it disables the decode graph path by design (v41_graph_allowed /
# v41_graph_batch_ready require !g_ds4_v41_prof, ds4_ds41_graph.inc:145-157)
# and adds a ds4_gpu_flush_commands() after every layer
# (ds4_ds41_forward.inc:1005), so a decode measured under it runs eager and
# per-layer flushed -- the port read 39 t/s against the engine's graphed 43.9
# until a no-prof cell showed 42.2.  Prefill is unaffected on both sides
# (n >= 64 flushes per layer either way), so prefill cells may keep the flag;
# NEVER take decode numbers from a --v41-prof run.
#
# PROTOCOL DIFFERENCE, stated so no one reads these as qwen numbers: the qwen
# card sweep is 32 frontiers per cell; the ds41 servers expose no frontier
# surface, so one prompt per context is the closest analog.  The shape is kept
# (greedy, temperature 0, fixed contexts, fresh process, clocks recorded,
# interleaved repeats), the frontier count is not.
#
# Per step, inside one boot:
#   1. turn 1: one request whose prompt is sized to the target context.
#   2. turn 2: the SAME conversation continued -- turn 1's messages verbatim,
#      plus the assistant's answer and a short new user turn.  The prefix is
#      byte-identical, so a side that reuses KV reports
#      prefill_cached_tokens ~= the turn-1 context while a side that does not
#      re-prefills the whole thing.  That difference, not the per-token rate,
#      is what an agent session feels.  The PORT's V4.1 route has neither
#      prefix reuse nor disk KV (docs/serving-capabilities.md:69); the engine
#      oracle gained prefix reuse 2026-10-12 (a8e878a, core_v41_kvreuse.c) and
#      reports it in its usage block, while disk KV stays unimplemented on
#      both (/data/YoungAi/src/server/server_generate_v41.c:8).
#
# The corpus is Italian prose, so the characters-per-token ratio is measured,
# not assumed: one calibration probe runs in the first boot of the sweep and is
# reused for every side and step (it is a property of the text and the
# tokenizer, not of the server).  The probe reads its token count from the
# port's response timings (crates/ds4-server/src/stream.rs:363-380) or from the
# engine's own prefill progress total (server_checkpoint.c:69); the engine's
# usage block also carries prompt_tokens and, since the reuse, cached_tokens
# (server_http.c:343-345).
#
# Recorded per request: ttft, prefill tokens and t/s, decode t/s, tokens per
# step (the acceptance proxy = decode_tokens/decode_steps), the engram prep
# split out of the port's host-gaps line, and the graphics clock sampled across
# the request.  Every record carries the binary sha256, the gguf sha256 and the
# clock median.
#
# usage: ds41_perf_sweep.sh <engine_server> <port_server> <gguf> <engram_dir> <port> <sidecar> [ctx steps] [gen tokens] [mtp draft]
#   ctx steps   comma list of target contexts (default 2048,8192,32768,65536)
#   gen tokens  greedy tokens per request (default 128)
#   mtp draft   draft width for the port; empty = the route's own default
#               (default empty, deliberately: the recorded ds41 numbers were
#               taken on the route default, and the qwen protocol's draft 2 is
#               another family's shape)
#
# env:
#   DS41_PERF_SIDES    sides to sweep, comma list from eng,port (default both)
#   DS41_PERF_RUNS     interleaved repeats, fresh boot each; medians reported
#                      (default 1; the sweep costs RUNS x sides x steps boots)
#   DS41_PERF_OUT      output directory (default /tmp/ds41_perf_sweep)
#   DS41_PERF_CORPUS   prompt corpus (default speed-bench/promessi_sposi.txt,
#                      the sha256-pinned corpus the qwen campaign used)
#   DS41_PERF_CAL_BYTES  calibration probe size in bytes (default 8192, about
#                      2.4K tokens at the measured ratio)
#   DS41_PERF_PORT_ARGS  extra arguments appended to the port boot line, e.g.
#                      "--v41-prof" for per-layer ms and the full dspark lines
#                      (crates/ds4-server/src/bin/ds4-server-rs.rs:255).  The
#                      value is recorded in meta.txt so a profile run is
#                      distinguishable from a clean one.
#   DS41_PERF_NO_CLOCKS  set to 1 to skip the clocks.gr sampling
#   DS41_PERF_NO_GGUF_SHA set to 1 to skip hashing the 113 GB gguf
set -u

ENG="$1" SRV="$2" MODEL="$3" ENGRAM="$4" PORT="$5"
ZC="${6:-}"
CTXS="${7:-2048,8192,32768,65536}"
GEN="${8:-128}"
DRAFT="${9:-}"

SIDES="${DS41_PERF_SIDES:-eng,port}"
RUNS="${DS41_PERF_RUNS:-1}"
CAL_BYTES="${DS41_PERF_CAL_BYTES:-8192}"
ROOT="$(dirname "$0")/.."
CORPUS="${DS41_PERF_CORPUS:-$ROOT/speed-bench/promessi_sposi.txt}"
OUT="${DS41_PERF_OUT:-/tmp/ds41_perf_sweep}"

# Corpus offsets keep the calibration probe and the measured prompt on
# different text, so the probe can never warm the measured prompt's prefix.
CAL_OFF=0
MSR_OFF=262144
FOLLOWUP="Reply with one short sentence naming the chapter."

# Bounds. A 64K prefill runs for minutes at this family's rate and the boot of
# a 113 GB artifact takes minutes more; both are given room, and a stall longer
# than either is a failure rather than a wait.
BOOT_TRIES=900          # x BOOT_SLEEP = 30 min
BOOT_SLEEP=2
CURL_TIMEOUT=3600
CLK_SLEEP=1

LOG="$OUT/log"
RES="$OUT/results.jsonl"
RATIO_FILE="$OUT/ratio.txt"

fail() { echo "ds41 perf sweep: FAIL ($1)" >&2; exit 1; }
note() { echo "ds41 perf sweep: $1"; }

[ $# -ge 5 ] || fail "usage: ds41_perf_sweep.sh <engine_server> <port_server> <gguf> <engram_dir> <port> [sidecar] [ctx steps] [gen tokens] [mtp draft]"
for tool in jq curl sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || fail "$tool not found"
done
[ -x "$ENG" ] || fail "engine server not executable: $ENG"
[ -x "$SRV" ] || fail "port server not executable: $SRV"
[ -f "$MODEL" ] || fail "gguf not found: $MODEL"
[ -d "$ENGRAM" ] || fail "engram dir not found: $ENGRAM"
[ -f "$CORPUS" ] || fail "corpus not found: $CORPUS"
[ -n "$ZC" ] || note "no sidecar given: booting without --zchain/--mem-budget-mb"

mkdir -p "$LOG" || fail "cannot create $OUT"
: > "$RES"

DFLAG=""
[ -n "$DRAFT" ] && DFLAG="--mtp-draft $DRAFT"
ZFLAG=""
[ -n "$ZC" ] && ZFLAG="--zchain $ZC --mem-budget-mb 110000"
PORT_EXTRA="${DS41_PERF_PORT_ARGS:-}"

# ---------------------------------------------------------------- primitives

now() { date +%s.%N; }

log_size() { if [ -f "$1" ]; then wc -c < "$1" | tr -d ' '; else echo 0; fi; }

is_int() { case "$1" in ''|*[!0-9]*) return 1 ;; *) return 0 ;; esac; }

# The per-request slice of a server log: post-boot lines only, so a turn-2
# parse can never read turn 1's numbers.
slice() { # <log> <off-before> <off-after> <out>
    if [ "$3" -le "$2" ]; then : > "$4"; return 0; fi
    tail -c "+$(( $2 + 1 ))" "$1" 2>/dev/null | head -c "$(( $3 - $2 ))" > "$4"
}

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
    if [ "${DS41_PERF_NO_CLOCKS:-0}" = "1" ] || ! command -v nvidia-smi >/dev/null 2>&1; then
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

# ---------------------------------------------------------------- server boot

SRV_PID=""
SLOG=""

cleanup() {
    [ -n "${CLK_PID:-}" ] && kill "$CLK_PID" 2>/dev/null
    [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
}
trap cleanup EXIT

boot() { # <side> <tag>
    SIDE="$1"
    SLOG="$LOG/$2.srv.log"
    rm -f "$SLOG"
    : > "$SLOG"
    B_T0=$(now)
    case "$SIDE" in
        eng) "$ENG" --cuda -m "$MODEL" --engram-dir "$ENGRAM" --port "$PORT" \
                     $ZFLAG --host 127.0.0.1 >> "$SLOG" 2>&1 & ;;
        port) "$SRV" --cuda -m "$MODEL" --port "$PORT" --engram-dir "$ENGRAM" \
                     $DFLAG $ZFLAG $PORT_EXTRA --host 127.0.0.1 >> "$SLOG" 2>&1 & ;;
        *) fail "unknown side $SIDE" ;;
    esac
    SRV_PID=$!
    i=0
    while [ "$i" -lt "$BOOT_TRIES" ]; do
        grep -q "listening on" "$SLOG" 2>/dev/null && break
        kill -0 "$SRV_PID" 2>/dev/null || fail "$SIDE died during boot (see $SLOG)"
        sleep "$BOOT_SLEEP"
        i=$(( i + 1 ))
    done
    grep -q "listening on" "$SLOG" 2>/dev/null || fail "$SIDE did not start listening (see $SLOG)"
    B_S=$(awk -v a="$B_T0" -v b="$(now)" 'BEGIN { printf "%.1f", b - a }')
}

shutdown() {
    [ -n "$SRV_PID" ] || return 0
    kill "$SRV_PID" 2>/dev/null
    wait "$SRV_PID" 2>/dev/null
    SRV_PID=""
}

# ---------------------------------------------------------------- one request

# Runs one request, times it, slices its server-log region and samples the
# graphics clock across it.  Leaves R_CODE R_WALL R_CLK R_SLICE.
run_request() { # <side> <tag> <req-file> <resp-file>
    R_SLICE="$LOG/$1_$2.req.log"
    off0=$(log_size "$SLOG")
    clkf="$LOG/$1_$2.clk"
    start_clocks "$clkf"
    t0=$(now)
    R_CODE=$(curl -sS -m "$CURL_TIMEOUT" -o "$4" -w '%{http_code}' \
        "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'content-type: application/json' -d @"$3") \
      || { stop_clocks; fail "$1 $2 transport failure"; }
    t1=$(now)
    stop_clocks
    R_WALL=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b - a }')
    R_CLK=$(median_of "$clkf")
    slice "$SLOG" "$off0" "$(log_size "$SLOG")" "$R_SLICE"
    [ "$R_CODE" = "200" ] || fail "$1 $2 returned HTTP $R_CODE (body $4, log $R_SLICE)"
}

# --------------------------------------------------------------- field reads

# Prompt token count of a response.  The port reports it in the timings block;
# the engine reports it as the prefill progress total (server_checkpoint.c:69).
tokens_of() { # <side> <resp> <req-log>
    if [ "$1" = "port" ]; then
        jq -r '.timings.prefill_tokens // empty' "$2" 2>/dev/null
        return 0
    fi
    v=$(sed -n 's/.*chunk [0-9]*\/\([0-9]*\) ([0-9.]*%) chunk=.*/\1/p' "$3" | tail -1)
    [ -n "$v" ] || v=$(jq -r '.usage.prompt_tokens // empty' "$2" 2>/dev/null)
    printf '%s' "$v"
}

# turn-1 log-derived numbers.  host gaps/round is the native per-decode-round
# split (ds4_ds41_draft.inc:1012-1013) and prints only when spec rounds ran.
prep_ms() { sed -n 's/.*engram prep \([0-9.]*\).*/\1/p' "$1" | tail -1; }
host_gaps_line() { sed -n 's/.*\(\[v41\] host gaps\/round:.*\)/\1/p' "$1" | tail -1; }
eng_prefill_tps() { sed -n 's/.*chunk [0-9]*\/[0-9]* ([0-9.]*%) chunk=[0-9.]* t\/s avg=\([0-9.]*\) t\/s.*/\1/p' "$1" | tail -1; }
eng_decode_tps() { sed -n 's/.*decoding chunk=[0-9.]* t\/s avg=\([0-9.]*\) t\/s.*/\1/p' "$1" | tail -1; }
# The engine's cached count comes from the response usage block.  Its log's
# ctx span is built with cached=0 at request begin
# (server_generate_v41.c:362), so it can never show a reuse hit (measured
# 2026-10-12: span 0 against usage cached_tokens 2149).
eng_cached() { # <resp-json>
    jq -r '.usage.prompt_tokens_details.cached_tokens // empty' "$1" 2>/dev/null
}

# One record as a single JSON object.  Empty readings become null, so a missing
# number is visible as null in the summary instead of being silently zero.
rec() { # <side> <ctx> <run> <ttft> <pf-tok> <pf-cached> <pf-tps> <dc-tps> <tok/step> <prep> <wall> <clk> <t2-pf-tok> <t2-pf-cached> <t2-pf-tps> <t2-dc-tps> <t2-ttft> <t2-tok/step> <t2-wall> <boot-s> <gaps-line>
    jq -c -n \
        --arg side "$1" --argjson ctx "$2" --argjson run "$3" \
        --arg ttft "$4" --arg pftok "$5" --arg pfc "$6" --arg pftps "$7" \
        --arg dctps "$8" --arg tps "$9" --arg prep "${10}" --arg wall "${11}" \
        --arg clk "${12}" --arg t2tok "${13}" --arg t2c "${14}" \
        --arg t2pftps "${15}" --arg t2dctps "${16}" --arg t2ttft "${17}" \
        --arg t2tps "${18}" --arg t2wall "${19}" --arg boot "${20}" \
        --arg gaps "${21}" \
        'def num: if . == "" then null else (tonumber? // null) end;
         def txt: if . == "" then null else . end;
         {side:$side, ctx_target:$ctx, run:$run, boot_s:($boot|num),
          turn1:{ttft_ms:($ttft|num), prefill_tokens:($pftok|num),
                 prefill_cached_tokens:($pfc|num), prefill_tok_s:($pftps|num),
                 decode_tok_s:($dctps|num), tok_per_step:($tps|num),
                 engram_prep_ms:($prep|num), wall_s:($wall|num),
                 clock_median_mhz:($clk|num), host_gaps:($gaps|txt)},
          turn2:{prefill_tokens:($t2tok|num), prefill_cached_tokens:($t2c|num),
                 prefill_tok_s:($t2pftps|num), decode_tok_s:($t2dctps|num),
                 ttft_ms:($t2ttft|num), tok_per_step:($t2tps|num),
                 wall_s:($t2wall|num)}}' >> "$RES"
}

# ---------------------------------------------------------------- one step

# The ratio is measured once and reused: it describes the corpus and the
# tokenizer, both shared by the two sides.
ensure_ratio() { # <side>
    if [ -s "$RATIO_FILE" ]; then
        return 0
    fi
    corpus_slice "$CAL_OFF" "$CAL_BYTES" "$LOG/cal.txt"
    jq -n --rawfile p "$LOG/cal.txt" --argjson mt 1 \
        '{messages:[{role:"user",content:$p}],max_tokens:$mt,temperature:0,reasoning_effort:"none"}' \
        > "$LOG/cal.json"
    run_request "$1" "cal" "$LOG/cal.json" "$LOG/cal.resp.json"
    cal_tok=$(tokens_of "$1" "$LOG/cal.resp.json" "$R_SLICE")
    is_int "$cal_tok" || fail "$1 reported no calibration token count (see $R_SLICE)"
    [ "$cal_tok" -gt 0 ] || fail "$1 calibration returned 0 prompt tokens"
    ratio=$(awk -v b="$CAL_BYTES" -v t="$cal_tok" 'BEGIN { printf "%.4f", b / t }')
    printf '%s\n' "$ratio" > "$RATIO_FILE"
    note "calibration on $1: $cal_tok tokens over $CAL_BYTES bytes, ratio $ratio bytes/token"
}

run_step() { # <side> <ctx-target> <run>
    RUN_SIDE="$1" RUN_CTX="$2" RUN_IDX="$3"
    note "$RUN_SIDE ctx=$RUN_CTX run=$RUN_IDX: booting"
    boot "$RUN_SIDE" "${RUN_SIDE}_${RUN_CTX}_r${RUN_IDX}"
    ensure_ratio "$RUN_SIDE"
    ratio=$(cat "$RATIO_FILE")

    # turn 1: the measured prompt, on a different slice than the calibration.
    bytes=$(awk -v t="$RUN_CTX" -v r="$ratio" 'BEGIN { printf "%d", t * r }')
    corpus_slice "$MSR_OFF" "$bytes" "$LOG/$RUN_SIDE.t1.txt"
    jq -n --rawfile p "$LOG/$RUN_SIDE.t1.txt" --argjson mt "$GEN" \
        '{messages:[{role:"user",content:$p}],max_tokens:$mt,temperature:0,reasoning_effort:"none"}' \
        > "$LOG/$RUN_SIDE.t1.json"
    run_request "$RUN_SIDE" "t1_$RUN_CTX" "$LOG/$RUN_SIDE.t1.json" "$LOG/$RUN_SIDE.t1.resp.json"
    T1_LOG="$R_SLICE"
    T1_WALL="$R_WALL"
    T1_CLK="$R_CLK"
    t1_pftok=$(tokens_of "$RUN_SIDE" "$LOG/$RUN_SIDE.t1.resp.json" "$T1_LOG")
    t1_gaps=$(host_gaps_line "$T1_LOG")
    t1_prep=$(prep_ms "$T1_LOG")

    # turn 2: turn 1's conversation continued verbatim.
    jq -r '.choices[0].message.content // ""' "$LOG/$RUN_SIDE.t1.resp.json" > "$LOG/$RUN_SIDE.t1.answer.txt" 2>/dev/null
    jq -n --rawfile p "$LOG/$RUN_SIDE.t1.txt" --rawfile a "$LOG/$RUN_SIDE.t1.answer.txt" \
          --arg q "$FOLLOWUP" --argjson mt "$GEN" \
        '{messages:[{role:"user",content:$p},{role:"assistant",content:$a},{role:"user",content:$q}],
          max_tokens:$mt,temperature:0,reasoning_effort:"none"}' \
        > "$LOG/$RUN_SIDE.t2.json"
    run_request "$RUN_SIDE" "t2_$RUN_CTX" "$LOG/$RUN_SIDE.t2.json" "$LOG/$RUN_SIDE.t2.resp.json"
    t2_pftok=$(tokens_of "$RUN_SIDE" "$LOG/$RUN_SIDE.t2.resp.json" "$R_SLICE")

    if [ "$RUN_SIDE" = "port" ]; then
        t1_ttft=$(jq -r '.timings.ttft_ms // empty' "$LOG/$RUN_SIDE.t1.resp.json")
        t1_pfc=$(jq -r '.timings.prefill_cached_tokens // empty' "$LOG/$RUN_SIDE.t1.resp.json")
        t1_pftps=$(jq -r '.timings.prefill_tok_s // empty' "$LOG/$RUN_SIDE.t1.resp.json")
        t1_dctps=$(jq -r '.timings.decode_tok_s // empty' "$LOG/$RUN_SIDE.t1.resp.json")
        t1_tps=$(jq -r '.timings.tok_per_step // empty' "$LOG/$RUN_SIDE.t1.resp.json")
        t2_ttft=$(jq -r '.timings.ttft_ms // empty' "$LOG/$RUN_SIDE.t2.resp.json")
        t2_pfc=$(jq -r '.timings.prefill_cached_tokens // empty' "$LOG/$RUN_SIDE.t2.resp.json")
        t2_pftps=$(jq -r '.timings.prefill_tok_s // empty' "$LOG/$RUN_SIDE.t2.resp.json")
        t2_dctps=$(jq -r '.timings.decode_tok_s // empty' "$LOG/$RUN_SIDE.t2.resp.json")
        t2_tps=$(jq -r '.timings.tok_per_step // empty' "$LOG/$RUN_SIDE.t2.resp.json")
    else
        t1_ttft=""
        t1_pfc=$(eng_cached "$LOG/$RUN_SIDE.t1.resp.json")
        t1_pftps=$(eng_prefill_tps "$T1_LOG")
        t1_dctps=$(eng_decode_tps "$T1_LOG")
        t1_tps=""
        t2_ttft=""
        t2_pfc=$(eng_cached "$LOG/$RUN_SIDE.t2.resp.json")
        t2_pftps=$(eng_prefill_tps "$LOG/${RUN_SIDE}_t2_${RUN_CTX}.req.log")
        t2_dctps=$(eng_decode_tps "$LOG/${RUN_SIDE}_t2_${RUN_CTX}.req.log")
        t2_tps=""
    fi

    rec "$RUN_SIDE" "$RUN_CTX" "$RUN_IDX" "$t1_ttft" "$t1_pftok" "$t1_pfc" \
        "$t1_pftps" "$t1_dctps" "$t1_tps" "$t1_prep" "$T1_WALL" "$T1_CLK" \
        "$t2_pftok" "$t2_pfc" "$t2_pftps" "$t2_dctps" "$t2_ttft" "$t2_tps" \
        "" "$B_S" "$t1_gaps"

    note "$RUN_SIDE ctx=$RUN_CTX: prefill $t1_pftok tok @ ${t1_pftps:-?} t/s, decode ${t1_dctps:-?} t/s, ttft ${t1_ttft:-?} ms; turn 2 prefill $t2_pftok tok (${t2_pfc:-?} cached)"
    shutdown
}

# ------------------------------------------------------------------- driver

{
    echo "ds41 perf sweep"
    echo "started $(date -Iseconds)"
    echo "sides $SIDES; ctx $CTXS; gen $GEN; runs $RUNS; draft ${DRAFT:-<route default>}"
    echo "port $PORT; sidecar ${ZC:-<none>}"
    echo "port extra args: ${PORT_EXTRA:-<none>}"
    echo "corpus $CORPUS"
    echo "corpus sha256 $(sha256sum "$CORPUS" | awk '{print $1}')"
    echo "engine binary $ENG sha256 $(sha256sum "$ENG" | awk '{print $1}')"
    echo "port binary   $SRV sha256 $(sha256sum "$SRV" | awk '{print $1}')"
    if [ "${DS41_PERF_NO_GGUF_SHA:-0}" = "1" ]; then
        echo "gguf $MODEL size $(stat -c %s "$MODEL") sha256 skipped"
    else
        echo "gguf $MODEL sha256 $(sha256sum "$MODEL" | awk '{print $1}')"
    fi
} | tee "$OUT/meta.txt"

SIDES_LIST=$(printf '%s' "$SIDES" | tr ',' ' ')
CTXS_LIST=$(printf '%s' "$CTXS" | tr ',' ' ')

for r in $(seq 1 "$RUNS"); do
    for side in $SIDES_LIST; do
        for ctx in $CTXS_LIST; do
            run_step "$side" "$ctx" "$r"
        done
    done
done

# ---------------------------------------------------------------- summary

jq -s . "$RES" > "$OUT/summary.json" || fail "summary.json build failed"
cp "$RES" "$OUT/summary.jsonl"

# Medians over the interleaved runs, per side and target context.  A metric a
# side cannot report (the engine has no ttft or tokens/step on this wire) stays
# "na" and prints as "-", never as a zero that would read as a measurement.
# Columns of medians.tsv: side, ctx, prefill t/s, decode t/s, ttft ms,
# tokens/step, turn-2 prefill t/s, turn-2 cached tokens, clock MHz, runs.
jq -r '[.side, (.ctx_target|tostring),
        (.turn1.prefill_tok_s // "na" | tostring),
        (.turn1.decode_tok_s // "na" | tostring),
        (.turn1.ttft_ms // "na" | tostring),
        (.turn1.tok_per_step // "na" | tostring),
        (.turn2.prefill_tok_s // "na" | tostring),
        (.turn2.prefill_cached_tokens // "na" | tostring),
        (.turn1.clock_median_mhz // "na" | tostring)] | @tsv' "$RES" \
  | awk -F'\t' '
      { k = $1 "\t" $2; n[k]++
        for (c = 1; c <= 7; c++) { vals[k, c] = vals[k, c] " " $(c + 2) } }
      END {
          for (key in n) {
              split(key, kk, "\t")
              printf "%s\t%s", kk[1], kk[2]
              for (c = 1; c <= 7; c++) {
                  cnt = split(vals[key, c], a, " "); num = 0
                  for (i = 1; i <= cnt; i++) if (a[i] != "na") { num++; col[num] = a[i] + 0 }
                  if (num == 0) { printf "\t-"; continue }
                  for (i = 1; i <= num; i++) for (j = i + 1; j <= num; j++) if (col[j] < col[i]) { t = col[i]; col[i] = col[j]; col[j] = t }
                  m = (num % 2) ? col[(num + 1) / 2] : (col[num / 2] + col[num / 2 + 1]) / 2
                  printf "\t%.3f", m
              }
              printf "\t%d\n", n[key]
          }
      }' | sort > "$OUT/medians.tsv"

{
    echo "ds41 perf sweep summary -- medians over $RUNS interleaved run(s)"
    echo
    printf '%-5s %-7s %11s %11s %9s %12s %14s %12s %9s %5s\n' \
        side ctx prefill_tps decode_tps ttft_ms tok_per_step t2_prefill_tps t2_cached clk_mhz runs
    awk 'BEGIN { FS = OFS = "\t" }
         { printf "%-5s %-7s %11s %11s %9s %12s %14s %12s %9s %5s\n",
                  $1, $2, $3, $4, $5, $6, $7, $8, $9, $10 }' "$OUT/medians.tsv"
    echo
    echo "dash = the side does not report that metric on this wire"
    echo "per-request records: $RES (run-level, one JSON object per boot)"
    echo "hashes and clocks:   $OUT/meta.txt"
} > "$OUT/summary.txt"

cat "$OUT/summary.txt"
