#!/bin/sh
# ds41_kv_reuse_gate.sh -- the V4.1 KV prefix reuse's exact gate.
#
# The port of the engine's own gate (gguf-tools/scripts/kv_reuse_gate.sh,
# /data/YoungAi @ 7fb26cc; that file is the protocol's source of truth).
# What it decides: whether the checkpoint holds everything decode mutates or
# later reads (core_v41_kvreuse.c:7-13).  Tokens alone are not the gate: the
# engine's own mutation test (a deliberately un-restored compressor remainder
# row) passed 200 tokens; the row-level comparison catches it.
#
#   A1 = prompt P, temperature 0, 1 token -> t1 (the first token's text)
#   A  = prompt P, N tokens              -> ids g0 g1 ... plus one whole
#        logits row per emitted token (A's decode rounds mutate the window
#        ring, the compressor remainder rows and the group counts)
#   B  = prompt P+t1, N-1 tokens         -> must restore A's prompt-end
#        checkpoint, prefill exactly 1 token, and reproduce A's steps 1..
#   C  = the same request as B           -> must reuse all of it (suffix 0)
#        and match B's ids and rows bit for bit
#
# Why byte-exact rows: A's step k+1 and B's step k are the same n=1 forward,
# so a complete checkpoint makes the rows equal.  The port reads the rows
# from DS41_GEN_LOGITS (raw f32 rows, one per single-token decision;
# ds4_ds41_draft.inc:634-640) where the engine records a weighted sum
# (core_v41_sample.c:55-66): byte equality is strictly stronger, so this gate
# is not weaker than the engine's.  The requests carry frequency_penalty
# 1e-30: any non-zero penalty takes the host penalty route (sampling and
# speculation off), and 1e-30 moves no fp32 logit -- the engine's own trick.
#
# Note for the raw wire: a Completion request renders its prompt untemplated
# (render_prompt_ex, generate.rs:1279) and tokenizes it as text
# (`tokenize_text`), so P and P+t1 tokenize the engine's way.  If the
# tokenizer merges across the boundary, B reuses nothing and this gate says
# so (precondition not established) -- change the prompt byte count then.
# One server process holds A/B/C: the checkpoint is process state.
#
# usage: ds41_kv_reuse_gate.sh <server> <gguf> <engram_dir> [port] [sidecar]
#                             [prompt-file] [prompt-bytes] [tokens] [extra...]
#   defaults: port 8099, no sidecar, speed-bench/promessi_sposi.txt,
#   12000 bytes, 48 tokens.  Extra arguments are appended to the server boot
#   (e.g. --no-graph, to attribute a graph-vs-direct FP question).
set -u

SRV="$1" MODEL="$2" ENGRAM="$3"
PORT="${4:-8099}"
ZC="${5:-}"
PFILE="${6:-speed-bench/promessi_sposi.txt}"
NCH="${7:-12000}"
N="${8:-48}"
shift 8 2>/dev/null || true
EXTRA="${*:-}"

ROOT="$(dirname "$0")/.."
OUT="${DS41_GATE_OUT:-/tmp/ds41_kv_reuse_gate}"
LOG="$OUT/log"
mkdir -p "$LOG" || { echo "gate: cannot create $OUT" >&2; exit 1; }
SLOG="$LOG/srv.log"
ROWS="$OUT/rows.bin"
TMP="$OUT/tmp"
mkdir -p "$TMP"
: > "$SLOG"

for tool in jq curl dd cmp iconv; do
    command -v "$tool" >/dev/null 2>&1 || { echo "gate: $tool not found" >&2; exit 1; }
done
[ -x "$SRV" ] || { echo "gate: server not executable: $SRV" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "gate: gguf not found: $MODEL" >&2; exit 1; }
[ -d "$ENGRAM" ] || { echo "gate: engram dir not found: $ENGRAM" >&2; exit 1; }
[ -f "$PFILE" ] || { echo "gate: prompt file not found: $PFILE" >&2; exit 1; }

fail() { echo "ds41 KV reuse gate: FAIL ($1)" >&2; exit 1; }
note() { echo "ds41 KV reuse gate: $1"; }

SRV_PID=""
cleanup() { [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null; }
trap cleanup EXIT

ZFLAG=""
[ -n "$ZC" ] && ZFLAG="--zchain $ZC --mem-budget-mb 110000"

# The prompt is read once and used for every request (the same P, the engine's
# protocol); iconv drops a truncated multibyte tail from the byte cut.
iconv -c -f utf-8 -t utf-8 < "$PFILE" | head -c "$NCH" > "$LOG/p.txt" || fail "prompt slice failed"
P=$(cat "$LOG/p.txt")

request() { # <prompt-file> <max_tokens> <resp-out> ; prints the HTTP code
    jq -n --rawfile p "$1" --argjson mt "$2" \
        '{prompt:$p, max_tokens:$mt, temperature:0, frequency_penalty:1e-30}' > "$LOG/req.json"
    curl -sS -m 900 --noproxy '*' -o "$3" -w '%{http_code}' \
        "http://127.0.0.1:$PORT/v1/completions" \
        -H 'content-type: application/json' -d @"$LOG/req.json"
}

# The emitted ids: the native's own trace line (ds4_ds41_draft.inc:831, one
# absolute position + id per token, in emission order).
ids_of() { sed -n 's/^\[emit\] [0-9][0-9]* \(-\{0,1\}[0-9][0-9]*\)$/\1/p' "$1"; }
prefill_line() { sed -n 's/.*\(\[v41\] prefill [0-9]* token .*\)$/\1/p' "$1" | tail -1; }
reuse_line() { sed -n 's/.*\(\[v41\] KV reuse [0-9]*\/[0-9]* token (checkpoint [0-9.]* MB), prefill suffix [0-9]*\)$/\1/p' "$1" | tail -1; }
# reuse X Y (tokens restored / prompt tokens) from a reuse line
reuse_x() { printf '%s' "$1" | sed -n 's/\[v41\] KV reuse \([0-9]*\)\/[0-9]* token.*/\1/p'; }
reuse_y() { printf '%s' "$1" | sed -n 's/\[v41\] KV reuse [0-9]*\/\([0-9]*\) token.*/\1/p'; }
reuse_suffix() { printf '%s' "$1" | sed -n 's/.*prefill suffix \([0-9]*\)$/\1/p'; }

# One request inside the running server, with its log slice and row dump.
slice_from() { # <byte-offset> <out>
    tail -c "+$(( $1 + 1 ))" "$SLOG" > "$2"
}
off_now() { wc -c < "$SLOG" | tr -d ' '; }

note "booting $SRV (port $PORT, DS41_GEN_LOGITS $ROWS)"
DS41_GEN_LOGITS="$ROWS" "$SRV" --cuda -m "$MODEL" --port "$PORT" --engram-dir "$ENGRAM" \
    $ZFLAG --emit-trace $EXTRA --host 127.0.0.1 >> "$SLOG" 2>&1 &
SRV_PID=$!
i=0
while [ "$i" -lt 900 ]; do
    grep -q "listening on" "$SLOG" 2>/dev/null && break
    kill -0 "$SRV_PID" 2>/dev/null || fail "server died during boot (see $SLOG)"
    sleep 2
    i=$((i + 1))
done
grep -q "listening on" "$SLOG" 2>/dev/null || fail "server did not start listening (see $SLOG)"
note "up"

req_a1() {
    off=$(off_now)
    code=$(request "$LOG/p.txt" 1 "$LOG/a1.json") || fail "A1 transport"
    [ "$code" = "200" ] || fail "A1 HTTP $code"
    slice_from "$off" "$LOG/a1.log"
    t1=$(jq -j '.choices[0].text // ""' "$LOG/a1.json"; printf x)
    t1=${t1%x}
    printf '%s' "$P$t1" > "$LOG/pt1.txt"
}

req_a() {
    off=$(off_now)
    code=$(request "$LOG/p.txt" "$N" "$LOG/a.json") || fail "A transport"
    [ "$code" = "200" ] || fail "A HTTP $code"
    slice_from "$off" "$LOG/a.log"
    cp "$ROWS" "$LOG/a.rows" 2>/dev/null || fail "A wrote no logits dump"
    ids_of "$LOG/a.log" > "$LOG/a.ids"
    reuse_line "$LOG/a.log" > "$LOG/a.reuse"
    prefill_line "$LOG/a.log" > "$LOG/a.prefill"
}

req_b() {
    off=$(off_now)
    code=$(request "$LOG/pt1.txt" "$((N - 1))" "$LOG/b.json") || fail "B transport"
    [ "$code" = "200" ] || fail "B HTTP $code"
    slice_from "$off" "$LOG/b.log"
    cp "$ROWS" "$LOG/b.rows" 2>/dev/null || fail "B wrote no logits dump"
    ids_of "$LOG/b.log" > "$LOG/b.ids"
    reuse_line "$LOG/b.log" > "$LOG/b.reuse"
    prefill_line "$LOG/b.log" > "$LOG/b.prefill"
}

req_c() {
    off=$(off_now)
    code=$(request "$LOG/pt1.txt" "$((N - 1))" "$LOG/c.json") || fail "C transport"
    [ "$code" = "200" ] || fail "C HTTP $code"
    slice_from "$off" "$LOG/c.log"
    cp "$ROWS" "$LOG/c.rows" 2>/dev/null || fail "C wrote no logits dump"
    ids_of "$LOG/c.log" > "$LOG/c.ids"
    reuse_line "$LOG/c.log" > "$LOG/c.reuse"
    prefill_line "$LOG/c.log" > "$LOG/c.prefill"
}

# The row size is derived, never assumed: one row per emitted token plus the
# initial prefill pick's row; a run that ends on EOS dumps one fewer (its own
# row is the last).  row_size() accepts exactly those two readings.
row_size() { # <rows-file> <ids-file> <V>
    n=$(wc -l < "$2" | tr -d ' ')
    sz=$(wc -c < "$1" | tr -d ' ')
    [ "$n" -gt 0 ] || return 1
    per=$((4 * $3))
    [ $((sz % per)) -eq 0 ] || return 1
    r=$((sz / per))
    [ "$r" = "$n" ] || [ "$r" = "$((n + 1))" ]
}

row_cmp() { # <fileA> <idxA> <fileB> <idxB> <rowbytes>
    dd if="$1" bs="$5" skip="$2" count=1 2>/dev/null > "$TMP/ra"
    dd if="$3" bs="$5" skip="$4" count=1 2>/dev/null > "$TMP/rb"
    cmp -s "$TMP/ra" "$TMP/rb"
}

# --------------------------------------------------------------- the protocol
req_a1
req_a
req_b
req_c

n_a=$(wc -l < "$LOG/a.ids" | tr -d ' ')
n_b=$(wc -l < "$LOG/b.ids" | tr -d ' ')
n_c=$(wc -l < "$LOG/c.ids" | tr -d ' ')
[ "$n_a" -ge 8 ] && [ "$n_b" -ge 8 ] || fail "too few tokens emitted (A $n_a, B $n_b); raise the token count"

v_a=0
sz_a=$(wc -c < "$LOG/a.rows" | tr -d ' ')
for cand in $((sz_a / (4 * (n_a + 1)))) $((sz_a / (4 * n_a))); do
    [ "$cand" -gt 1000 ] || continue
    if row_size "$LOG/a.rows" "$LOG/a.ids" "$cand"; then
        v_a=$cand
        break
    fi
done
[ "$v_a" -gt 0 ] || fail "A's dump does not line up with its emitted count -- speculation likely ran (the penalty arm must be engaged: frequency_penalty non-zero; do not boot with --dspark)"
row_size "$LOG/b.rows" "$LOG/b.ids" "$v_a" || fail "B's dump does not line up with its emitted count (V=$v_a)"
row_size "$LOG/c.rows" "$LOG/c.ids" "$v_a" || fail "C's dump does not line up with its emitted count (V=$v_a)"
rowbytes=$((4 * v_a))

rb=$(cat "$LOG/b.reuse")
rc=$(cat "$LOG/c.reuse")
[ -n "$rb" ] || fail "B reused nothing (no KV reuse line); the tokenizer likely merged across the boundary -- change the prompt byte count"
bx=$(reuse_x "$rb"); by=$(reuse_y "$rb"); bs=$(reuse_suffix "$rb")
[ "$bs" = "1" ] && [ "$by" = "$((bx + 1))" ] || fail "precondition not established: B restored $bx/$by with suffix $bs, not the +1-token extension of A"

# B's ids must be A's shifted by one.
k=0
while [ "$k" -lt "$n_b" ]; do
    a_id=$(sed -n "$((k + 2))p" "$LOG/a.ids")
    b_id=$(sed -n "$((k + 1))p" "$LOG/b.ids")
    [ "$a_id" = "$b_id" ] || {
        echo "ds41 KV reuse gate: FAIL B token $k diverges: B=$b_id A[$(($k + 1))]=$a_id"
        echo "  A: $(jq -r '.choices[0].text // ""' "$LOG/a.json" | head -c 200)"
        echo "  B: $(jq -r '.choices[0].text // ""' "$LOG/b.json" | head -c 200)"
        exit 1
    }
    row_cmp "$LOG/b.rows" "$k" "$LOG/a.rows" "$((k + 1))" "$rowbytes" || {
        echo "ds41 KV reuse gate: FAIL B row $k differs from A row $(($k + 1)) (token still identical: $b_id)"
        echo "  the checkpoint missed something decode mutates or reads (core_v41_kvreuse.c:7-13)"
        exit 1
    }
    k=$((k + 1))
done
echo "ds41 KV reuse gate: PASS (B) -- $n_b tokens and $n_b whole rows match A shifted by one ($rb)"

# C: the same prompt again -- the checkpoint is B's own, so it must restore
# whole (suffix 0), and both ids and rows must equal B's bit for bit.
cx=$(reuse_x "$rc"); cy=$(reuse_y "$rc"); cs=$(reuse_suffix "$rc")
[ "$cs" = "0" ] && [ "$cx" = "$cy" ] || fail "C did not restore the whole prompt ($rc)"
[ "$n_c" = "$n_b" ] || fail "C emitted $n_c tokens, B emitted $n_b"
k=0
while [ "$k" -lt "$n_c" ]; do
    b_id=$(sed -n "$((k + 1))p" "$LOG/b.ids")
    c_id=$(sed -n "$((k + 1))p" "$LOG/c.ids")
    [ "$b_id" = "$c_id" ] || fail "C token $k diverges from B: C=$c_id B=$b_id"
    row_cmp "$LOG/c.rows" "$k" "$LOG/b.rows" "$k" "$rowbytes" || fail "C row $k differs from B row $k (token still identical: $c_id)"
    k=$((k + 1))
done
echo "ds41 KV reuse gate: PASS (C) -- identical prompt restored whole ($rc), $n_c tokens and rows bit-equal to B"

echo
echo "prefill lines (the measured cell):"
echo "  A: $(cat "$LOG/a.prefill")"
echo "  B: $(cat "$LOG/b.prefill")"
echo "  C: $(cat "$LOG/c.prefill")"
echo "artifacts: $OUT (log/, a.rows, b.rows, c.rows)"
