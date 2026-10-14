#!/bin/sh
# ds41_vision_sweep.sh -- the ds41 vision path in the speed-bench CSV standard.
#
# WHY A SEPARATE SWEEP (checked first, rule 0.2): ds41_spark_sweep.sh measures
# text frontiers against one server; the vision path has its own entry and its
# own cost centre.  Images enter the port through the server's OpenAI image_url
# parts (crates/ds4-server/src/generate.rs expand_v41_images), the tower reports
# its own elapsed time in the server's log (ds4_ds41_vision.inc:152 "[v41]
# vision tower: N image(s) X.XXs", the sidecar weights read per block and freed
# after), and the per-image plan prints at :237 ("[v41] image N: WxH patch ->
# WxH token (span N)").  The request rates stay the server's own timings
# (crates/ds4-server/src/stream.rs:383-409: ttft_ms, prefill_tokens,
# prefill_cached_tokens, prefill_tok_s, decode_tok_s) -- never a wall-clock
# computation.  Text rows come from the pinned corpus at prompt lengths that
# bracket the image rows, so an image row's ttft and prefill rate read against
# text alone.
#
# Protocol (speed-bench README): one fresh server process per run; one greedy
# request per row (temperature 0, reasoning off, 128 generated tokens); the
# corpus bytes-per-token ratio is measured by a probe, never assumed; clocks
# sampled across every request; every sample retained.  Single-turn renders do
# not extend the previous prompt's checkpoint, so prefill_cached_tokens reads 0
# and each row is a full prefill.  Image rows send the repo's fixtures
# (tests/fixtures/vision/) with one fixed prompt.
#
# usage:
#   boot mode (fresh process; the box must be free):
#     ds41_vision_sweep.sh <server> <gguf> <engram_dir> <zchain_dir> <vision_gguf> [port]
#   attach mode (measures a server already listening; never kills anything):
#     DS41_VISION_URL=http://127.0.0.1:8000 ds41_vision_sweep.sh
#
# env: DS41_VISION_CSV, DS41_VISION_OUT, DS41_VISION_FIXTURES (fixture dir,
#      default tests/fixtures/vision), DS41_VISION_NO_GGUF_SHA=1,
#      DS41_VISION_NO_SVG=1, CLK_SLEEP
set -u

HERE=$(cd "$(dirname "$0")" && pwd) || exit 1
ROOT=$(cd "$HERE/.." && pwd) || exit 1
CORPUS="$HERE/promessi_sposi.txt"
FIX_DIR="${DS41_VISION_FIXTURES:-$ROOT/tests/fixtures/vision}"
FIXTURES="review-market-panel.png bug.jpeg homer.png"
URL="${DS41_VISION_URL:-}"
GEN=128
PROMPT="What is in this image? List all the text and numbers you can see."
CTXS="500 950"          # text rows: target prompt lengths, bracketing the image spans
CAL_BYTES=8192          # calibration probe at offset 0
CAL_OFF=0
MSR_OFF=262144          # measured slices never overlap the calibration
CURL_TIMEOUT=3600
BOOT_TRIES=900          # x BOOT_SLEEP = 30 min (a 113 GB artifact loads slow)
BOOT_SLEEP=2
CLK_SLEEP=1

fail() { echo "ds41 vision sweep: FAIL ($1)" >&2; exit 1; }
note() { echo "ds41 vision sweep: $1"; }

for tool in jq curl sha256sum awk base64; do
    command -v "$tool" >/dev/null 2>&1 || fail "$tool not found"
done
[ -f "$CORPUS" ] || fail "corpus not found: $CORPUS"
[ -d "$FIX_DIR" ] || fail "fixture dir not found: $FIX_DIR"

# ------------------------------------------------------------------- target

MODE=attach
SRV_PID=""
if [ -z "$URL" ]; then
    MODE=boot
    SRV="${1:-}"; MODEL="${2:-}"; ENGRAM="${3:-}"; ZC="${4:-}"; VIS="${5:-}"; PORT="${6:-8099}"
    [ -n "$SRV" ] && [ -n "$MODEL" ] && [ -n "$ENGRAM" ] && [ -n "$ZC" ] && [ -n "$VIS" ] || \
        fail "usage: ds41_vision_sweep.sh <server> <gguf> <engram_dir> <zchain_dir> <vision_gguf> [port]"
    [ -x "$SRV" ] || fail "server not executable: $SRV"
    [ -f "$MODEL" ] || fail "gguf not found: $MODEL"
    [ -d "$ENGRAM" ] || fail "engram dir not found: $ENGRAM"
    [ -d "$ZC" ] || fail "zchain dir not found: $ZC"
    [ -f "$VIS" ] || fail "vision sidecar not found: $VIS"
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

CSV="${DS41_VISION_CSV:-$HERE/gb10_spark_ds41vision_${BIN8}.csv}"
OUT="${DS41_VISION_OUT:-/tmp/ds41_vision_${BIN8}}"
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
    command -v nvidia-smi >/dev/null 2>&1 || return 0
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

boot_server() {
    pgrep -x ds4-server >/dev/null 2>&1 && \
        fail "a ds4-server is already running; boot mode needs the box free (use DS41_VISION_URL to attach)"
    SLOG="$OUT/server.log"
    rm -f "$SLOG"
    B_T0=$(now)
    # Bind all interfaces (install.sh's own default) so a LAN client can reach
    # the booted server; the URL stays 127.0.0.1 (0.0.0.0 is a bind address,
    # not a connect destination).
    "$SRV" --cuda -m "$MODEL" --zchain "$ZC" --mem-budget-mb 110000 \
           --engram-dir "$ENGRAM" --vision-gguf "$VIS" \
           --host 0.0.0.0 --port "$PORT" >> "$SLOG" 2>&1 &
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

# One request; leaves R_CODE R_WALL R_CLK.  A transport stall is a failure; an
# HTTP error is the caller's to record and skip.  In boot mode the server's new
# log lines for this request land in $OUT/<tag>.newlines.
run_request() { # <tag> <req-file> <resp-file>
    L0=0
    [ "$MODE" = "boot" ] && L0=$(wc -l < "$SLOG" 2>/dev/null || echo 0)
    clkf="$OUT/$1.clk"
    start_clocks "$clkf"
    t0=$(now)
    R_CODE=$(curl -sS -m "$CURL_TIMEOUT" -o "$3" -w '%{http_code}' \
        "$URL" -H 'content-type: application/json' -d @"$2") \
      || { stop_clocks; fail "$1 transport failure"; }
    t1=$(now)
    stop_clocks
    cat "$clkf" >> "$OUT/clocks.all" 2>/dev/null || true
    R_WALL=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b - a }')
    R_CLK=$(median_of "$clkf")
    if [ "$MODE" = "boot" ]; then
        tail -n "+$(( L0 + 1 ))" "$SLOG" 2>/dev/null > "$OUT/$1.newlines" || : > "$OUT/$1.newlines"
    else
        : > "$OUT/$1.newlines"
    fi
}

# One CSV row.  The ten standard columns first (ctx_tokens, prefill_tokens,
# prefill_cached_tokens, prefill_tps, gen_tokens, gen_tps, first_token_sec,
# tok_per_step, wall_s, clock_mhz), then the vision extras (kind, image, patch,
# span, tower_s).  A row without the server's own rates is a failure, never an
# empty cell.
row() { # <tag> <ctx-tokens> <kind> <image> <patch> <span> <tower-s> <resp-file>
    tag=$1; ctx=$2; kind=$3; image=$4; patch=$5; span=$6; tower=$7; resp=$8
    pf=$(jq -r '.timings.prefill_tokens // empty' "$resp" 2>/dev/null)
    pfc=$(jq -r '.timings.prefill_cached_tokens // 0' "$resp" 2>/dev/null)
    ptps=$(jq -r '.timings.prefill_tok_s // empty' "$resp" 2>/dev/null)
    gen=$(jq -r '.usage.completion_tokens // empty' "$resp" 2>/dev/null)
    gtps=$(jq -r '.timings.decode_tok_s // empty' "$resp" 2>/dev/null)
    ttft_ms=$(jq -r '.timings.ttft_ms // empty' "$resp" 2>/dev/null)
    tps=$(jq -r '.timings.tok_per_step // empty' "$resp" 2>/dev/null)
    is_int "$pf" || fail "$tag reported no prefill_tokens"
    [ -n "$ptps" ] && [ -n "$gtps" ] && [ -n "$ttft_ms" ] || fail "$tag reported no rates/ttft"
    fts=$(awk -v ms="$ttft_ms" 'BEGIN { printf "%.4f", ms / 1000 }')
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$ctx" "$pf" "$pfc" "$ptps" "$gen" "$gtps" "$fts" "${tps:--}" \
        "$R_WALL" "${R_CLK:-}" "$kind" "$image" "$patch" "$span" "$tower" >> "$CSV"
    note "$tag: prefill $pf tok @ $ptps t/s, decode $gen tok @ $gtps t/s, ttft ${fts}s, tower ${tower}s, wall $R_WALL s, clock ${R_CLK:-?} MHz"
}

# --------------------------------------------------------------- image rows

image_row() { # <name> <file>
    name=$1; file=$2
    case "$file" in
        *.jpg|*.jpeg) mime=image/jpeg ;;
        *) mime=image/png ;;
    esac
    # The data URL goes through a file: bug.jpeg's base64 (2.4 MB) exceeds the
    # kernel's 128 KB per-argument limit, so the old inline --arg form died
    # with E2BIG ("jq: Argument list too long") and posted an empty body
    # (HTTP 400 invalid JSON) on the Spark run of 2026-10-14.
    { printf 'data:%s;base64,' "$mime"
      base64 -w0 "$file" 2>/dev/null || base64 "$file" | tr -d '\n'
    } > "$OUT/$name.url"
    jq -n --arg p "$PROMPT" --rawfile u "$OUT/$name.url" --argjson mt "$GEN" \
        '{messages:[{role:"user",content:[
            {type:"text",text:$p},
            {type:"image_url",image_url:{url:$u}}]}],
          max_tokens:$mt,temperature:0,reasoning_effort:"none"}' > "$OUT/$name.json"
    run_request "$name" "$OUT/$name.json" "$OUT/$name.resp.json"
    [ "$R_CODE" = "200" ] || fail "$name HTTP $R_CODE ($(head -c 200 "$OUT/$name.resp.json"))"
    tower=$(grep -o 'vision tower: [0-9]* image(s) [0-9.]*s' "$OUT/$name.newlines" 2>/dev/null \
        | tail -1 | awk '{ sub(/s$/, "", $NF); print $NF }')
    plan=$(grep -o 'image [0-9]*: [0-9]*x[0-9]* patch -> [0-9]*x[0-9]* token (span [0-9]*)' \
        "$OUT/$name.newlines" 2>/dev/null | tail -1)
    patch=$(printf '%s' "$plan" | sed -n 's/.*: \([0-9]*x[0-9]*\) patch.*/\1/p')
    span=$(printf '%s' "$plan" | sed -n 's/.*span \([0-9]*\).*/\1/p')
    ctx=$(jq -r '.timings.prefill_tokens // empty' "$OUT/$name.resp.json" 2>/dev/null)
    row "$name" "$ctx" image "$(basename "$file")" "${patch:--}" "${span:--}" "${tower:--}" "$OUT/$name.resp.json"
}

# ---------------------------------------------------------------- text rows

text_row() { # <name> <bytes>
    name=$1; bytes=$2
    corpus_slice "$MSR_OFF" "$bytes" "$OUT/$name.txt"
    jq -n --rawfile p "$OUT/$name.txt" --argjson mt "$GEN" \
        '{messages:[{role:"user",content:$p}],max_tokens:$mt,temperature:0,reasoning_effort:"none"}' \
        > "$OUT/$name.json"
    run_request "$name" "$OUT/$name.json" "$OUT/$name.resp.json"
    [ "$R_CODE" = "200" ] || fail "$name HTTP $R_CODE"
    ctx=$(jq -r '.timings.prefill_tokens // empty' "$OUT/$name.resp.json" 2>/dev/null)
    row "$name" "$ctx" text "-" "-" "-" "-" "$OUT/$name.resp.json"
}

# -------------------------------------------------------------------- driver

{
    echo "ds41 vision sweep (speed-bench standard)"
    echo "started $(date -Iseconds)"
    echo "mode $MODE"
    if [ "$MODE" = "attach" ]; then
        echo "server /proc/$PID/exe -> $SRV sha256 $BIN_SHA"
        echo "cmdline $CMDLINE"
    else
        echo "server $SRV sha256 $BIN_SHA"
        echo "vision sidecar $VIS sha256 $(sha256sum "$VIS" | awk '{print $1}') size $(stat -c %s "$VIS")"
    fi
    echo "prompt \"$PROMPT\"; gen $GEN; text targets $CTXS"
    echo "fixtures $FIX_DIR"
    for f in $FIXTURES; do
        [ -f "$FIX_DIR/$f" ] && echo "fixture $f sha256 $(sha256sum "$FIX_DIR/$f" | awk '{print $1}')"
    done
    echo "corpus $CORPUS sha256 $(sha256sum "$CORPUS" | awk '{print $1}')"
    if [ "$MODE" = "boot" ]; then
        if [ "${DS41_VISION_NO_GGUF_SHA:-0}" = "1" ]; then
            echo "gguf $MODEL size $(stat -c %s "$MODEL") sha256 skipped"
        else
            echo "gguf $MODEL sha256 $(sha256sum "$MODEL" | awk '{print $1}')"
        fi
    fi
} | tee "$OUT/meta.txt"

[ "$MODE" = "boot" ] && boot_server

# The corpus bytes-per-token ratio is measured, not assumed.  Offset 0 keeps
# the probe off the measured slices.
corpus_slice "$CAL_OFF" "$CAL_BYTES" "$OUT/cal.txt"
jq -n --rawfile p "$OUT/cal.txt" --argjson mt 1 \
    '{messages:[{role:"user",content:$p}],max_tokens:$mt,temperature:0,reasoning_effort:"none"}' \
    > "$OUT/cal.json"
run_request cal "$OUT/cal.json" "$OUT/cal.resp.json"
[ "$R_CODE" = "200" ] || fail "calibration HTTP $R_CODE"
cal_tok=$(jq -r '.timings.prefill_tokens // empty' "$OUT/cal.resp.json")
is_int "$cal_tok" && [ "$cal_tok" -gt 0 ] || fail "calibration reported no token count"
ratio=$(awk -v b="$CAL_BYTES" -v t="$cal_tok" 'BEGIN { printf "%.4f", b / t }')
note "calibration: $cal_tok tokens over $CAL_BYTES bytes, ratio $ratio bytes/token"

: > "$CSV"
echo "ctx_tokens,prefill_tokens,prefill_cached_tokens,prefill_tps,gen_tokens,gen_tps,first_token_sec,tok_per_step,wall_s,clock_mhz,kind,image,patch,span,tower_s" >> "$CSV"

rows=0
for f in $FIXTURES; do
    [ -f "$FIX_DIR/$f" ] || { note "fixture missing, skipped: $FIX_DIR/$f"; continue; }
    image_row "$(basename "${f%.*}")" "$FIX_DIR/$f"
    rows=$(( rows + 1 ))
done
for t in $CTXS; do
    bytes=$(awk -v t="$t" -v r="$ratio" 'BEGIN { printf "%d", t * r }')
    text_row "text_${t}" "$bytes"
    rows=$(( rows + 1 ))
done

# ------------------------------------------------------------------- record

all_clk=$(median_of "$OUT/clocks.all")
{
    echo "csv $CSV ($rows rows)"
    echo "clock median over all samples ${all_clk:-?} MHz"
    echo "boot_s ${B_S:-n/a}"
    echo "reuse: prefill_cached_tokens per row shows what the checkpoint restored"
    echo "       (0 = full prefill; single-turn renders do not extend the previous"
    echo "       prompt).  Image rows: ctx_tokens is the actual rendered prompt."
    echo "finished $(date -Iseconds)"
} | tee -a "$OUT/meta.txt"
cp "$OUT/meta.txt" "$CSV.meta.txt"

SVG=""
if [ "${DS41_VISION_NO_SVG:-0}" != "1" ] && command -v python3 >/dev/null 2>&1; then
    if python3 "$HERE/plot_speed.py" "$CSV" --title "GB10 Spark ds41 vision ${BIN8} t/s" >/dev/null 2>&1; then
        SVG="${CSV%.csv}_ts.svg"
    fi
fi

[ "$rows" -gt 0 ] || fail "no row was produced (see $OUT)"
note "wrote $CSV ($rows rows), ${SVG:-no svg}, meta $CSV.meta.txt"
