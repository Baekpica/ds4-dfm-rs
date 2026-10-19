#!/bin/sh
# ds41_kv_post_gate.sh -- the post-decode checkpoint's exact gate.
#
# The port saves a second checkpoint at the end of decode (K_post): the prompt
# followed by the tokens the model emitted, so a continuation that replays
# those bytes reuses the assistant block (ds4_ds41_kvreuse.inc, the named
# deviation from core_v41_api.c:192-195).  This gate decides whether that
# restored state is the state a cold run would have built: same tokens, same
# logits rows, bit for bit.
#
#   warm boot (--prefix-reuse exact), one process:
#     A  prompt P, N tokens, temperature 0     -> ids g0..gN-1 and text tA
#                                                 (leaves K_pre = P, K_post = P+g)
#     D  prompt P+tA, N tokens                 -> must report a reuse past |P|
#                                                 tokens: the assistant block
#   cold boot (--prefix-reuse off), one process:
#     C  prompt P+tA, N tokens                 -> the reference
#
#   PASS when D's emitted ids equal C's and every one of D's logits rows equals
#   C's, byte for byte.
#
#   The state's committed frontier is one behind the emission count -- the last
#   emitted token launches no forward, so its row is not in the buffers -- and
#   the checkpoint claims exactly the frontier.  D therefore reuses |P| + N - 1
#   tokens and prefills the one token the engine never computed, plus nothing
#   else; the assertion is that the reuse passes the prompt (the block), not
#   that it covers the whole emission.
#
# Rows come from DS41_GEN_LOGITS (raw f32 rows, one per single-token decision,
# ds4_ds41_draft.inc).  frequency_penalty 1e-30 keeps the engine on the host
# penalty route, so speculation and sampling are off and the decisions are
# comparable; the vocabulary width is derived from A's dump, never assumed.
#
# usage: ds41_kv_post_gate.sh <server> <gguf> <engram_dir> [port] [sidecar]
#                             [prompt-file] [prompt-bytes] [tokens]
#                             [warm-extra...] -- [cold-extra...]
#   defaults: port 8099, no sidecar, speed-bench/promessi_sposi.txt,
#   12000 bytes, 48 tokens.  Server flags before "--" go to the warm boot only,
#   those after it to the cold boot only.
set -u

SRV="$1" MODEL="$2" ENGRAM="$3"
PORT="${4:-8099}"
ZC="${5:-}"
PFILE="${6:-speed-bench/promessi_sposi.txt}"
NCH="${7:-12000}"
N="${8:-48}"
WARM_EXTRA=""
COLD_EXTRA=""
if [ "$#" -gt 8 ]; then
    shift 8
    seen_sep=0
    for a in "$@"; do
        if [ "$a" = "--" ]; then seen_sep=1; continue; fi
        if [ "$seen_sep" -eq 0 ]; then WARM_EXTRA="$WARM_EXTRA $a"; else COLD_EXTRA="$COLD_EXTRA $a"; fi
    done
fi

OUT="${DS41_POST_GATE_OUT:-/tmp/ds41_kv_post_gate}"
LOG="$OUT/log"
TMP="$OUT/tmp"
mkdir -p "$LOG" "$TMP" || { echo "gate: cannot create $OUT" >&2; exit 1; }

for tool in jq curl cmp iconv dd awk; do
    command -v "$tool" >/dev/null 2>&1 || { echo "gate: $tool not found" >&2; exit 1; }
done
[ -x "$SRV" ] || { echo "gate: server not executable: $SRV" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "gate: gguf not found: $MODEL" >&2; exit 1; }
[ -d "$ENGRAM" ] || { echo "gate: engram dir not found: $ENGRAM" >&2; exit 1; }
[ -f "$PFILE" ] || { echo "gate: prompt file not found: $PFILE" >&2; exit 1; }

fail() { echo "ds41 KV post gate: FAIL ($1)" >&2; exit 1; }
note() { echo "ds41 KV post gate: $1"; }

SRV_PID=""
cleanup() { [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null; }
trap cleanup EXIT

ZFLAG=""
[ -n "$ZC" ] && ZFLAG="--zchain $ZC --mem-budget-mb 110000"

# The prompt is read once and used for every request (the same P, the engine's
# protocol); iconv drops a truncated multibyte tail from the byte cut.
iconv -c -f utf-8 -t utf-8 < "$PFILE" | head -c "$NCH" > "$LOG/p.txt" || fail "prompt slice failed"

boot() { # <reuse-mode> <rows-path> <server-log> <extra-flags>
    mode="$1" rows="$2" slog="$3" extra="$4"
    : > "$slog"
    DS41_GEN_LOGITS="$rows" "$SRV" --cuda -m "$MODEL" --port "$PORT" --engram-dir "$ENGRAM" \
        $ZFLAG --emit-trace --prefix-reuse "$mode" $extra --host 127.0.0.1 >> "$slog" 2>&1 &
    SRV_PID=$!
    i=0
    while [ "$i" -lt 900 ]; do
        grep -q "listening on" "$slog" 2>/dev/null && return 0
        kill -0 "$SRV_PID" 2>/dev/null || return 1
        sleep 2
        i=$((i + 1))
    done
    return 1
}

stop_server() {
    [ -n "$SRV_PID" ] || return 0
    kill "$SRV_PID" 2>/dev/null
    wait "$SRV_PID" 2>/dev/null
    SRV_PID=""
    sleep 3   # let the port go before the next boot
}

request() { # <prompt-file> <max_tokens> <resp-out> ; prints the HTTP code
    jq -n --rawfile p "$1" --argjson mt "$2" \
        '{prompt:$p, max_tokens:$mt, temperature:0, frequency_penalty:1e-30}' > "$LOG/req.json"
    curl -sS -m 900 --noproxy '*' -o "$3" -w '%{http_code}' \
        "http://127.0.0.1:$PORT/v1/completions" \
        -H 'content-type: application/json' -d @"$LOG/req.json"
}

# The emitted ids: the native's own trace line, one absolute position + id per
# token, in emission order (ds4_ds41_draft.inc).
ids_of() { sed -n 's/^\[emit\] [0-9][0-9]* \(-\{0,1\}[0-9][0-9]*\)$/\1/p' "$1"; }
reuse_line() { sed -n 's/.*\(\[v41\] KV reuse [0-9]*\/[0-9]* token (checkpoint [0-9.]* MB), prefill suffix [0-9]*\)$/\1/p' "$1" | tail -1; }
reuse_x() { printf '%s' "$1" | sed -n 's/\[v41\] KV reuse \([0-9]*\)\/[0-9]* token.*/\1/p'; }
prefill_line() { sed -n 's/.*\(\[v41\] prefill [0-9]* token .*\)$/\1/p' "$1" | tail -1; }
prefill_n() { printf '%s' "$1" | sed -n 's/\[v41\] prefill \([0-9]*\) token.*/\1/p'; }
rows_in() { # <rows-file> <row-bytes> ; prints the row count, or fails
    sz=$(wc -c < "$1" | tr -d ' ')
    [ "$2" -gt 0 ] || return 1
    [ $((sz % $2)) -eq 0 ] || return 1
    printf '%s' "$((sz / $2))"
}
row_cmp() { # <fileA> <idxA> <fileB> <idxB> <rowbytes>
    dd if="$1" bs="$5" skip="$2" count=1 2>/dev/null > "$TMP/ra"
    dd if="$3" bs="$5" skip="$4" count=1 2>/dev/null > "$TMP/rb"
    cmp -s "$TMP/ra" "$TMP/rb"
}

##############################################################################
# Warm boot: A builds both checkpoints, D must reuse the assistant block.
##############################################################################
note "warm boot (prefix-reuse exact, port $PORT)"
boot exact "$OUT/warm.rows" "$LOG/warm.log" "$WARM_EXTRA" || fail "warm server did not start"

code=$(request "$LOG/p.txt" "$N" "$LOG/a.json") || fail "A transport"
[ "$code" = "200" ] || fail "A HTTP $code"
cp "$OUT/warm.rows" "$LOG/a.rows" 2>/dev/null || fail "A wrote no logits dump"
ids_of "$LOG/warm.log" > "$LOG/a.ids"
prefill_line "$LOG/warm.log" > "$LOG/a.prefill"
n_a=$(wc -l < "$LOG/a.ids" | tr -d ' ')
[ "$n_a" -ge 8 ] || fail "A emitted $n_a tokens; raise the token count"
ptok=$(prefill_n "$(cat "$LOG/a.prefill")")
[ -n "$ptok" ] || fail "A reported no prefill count"
sz_a=$(wc -c < "$LOG/a.rows" | tr -d ' ')
v=0
if [ $((sz_a % (4 * n_a))) -eq 0 ]; then v=$((sz_a / (4 * n_a))); fi
[ "$v" -gt 1000 ] || fail "A's dump does not line up with its emitted count -- speculation likely ran (the penalty arm must be engaged)"
rowbytes=$((4 * v))
note "A: prompt $ptok token, emitted $n_a token, vocabulary $v"

# The client's replay: the model's own bytes appended to the prompt.
cat "$LOG/p.txt" > "$LOG/pfull.txt"
jq -j '.choices[0].text // ""' "$LOG/a.json" >> "$LOG/pfull.txt"

code=$(request "$LOG/pfull.txt" "$N" "$LOG/d.json") || fail "D transport"
[ "$code" = "200" ] || fail "D HTTP $code"
cp "$OUT/warm.rows" "$LOG/d.rows" 2>/dev/null || fail "D wrote no logits dump"
ids_of "$LOG/warm.log" > "$LOG/d.ids.all"
tail -n "$N" "$LOG/d.ids.all" > "$LOG/d.ids"
reuse_line "$LOG/warm.log" > "$LOG/d.reuse"
prefill_line "$LOG/warm.log" > "$LOG/d.prefill"
dx=$(reuse_x "$(cat "$LOG/d.reuse")")
note "D: $(cat "$LOG/d.reuse")"
note "D: $(cat "$LOG/d.prefill")"
n_warm=$(rows_in "$LOG/d.rows" "$rowbytes") || fail "D's dump does not line up with the vocabulary width"
# The dump is opened per run, so it holds this request's decisions only.
[ "$n_warm" -eq "$N" ] || fail "D's dump holds $n_warm rows, not the request's $N"

##############################################################################
# Cold boot: the same request with no checkpoint at all.
##############################################################################
stop_server
note "cold boot (prefix-reuse off, port $PORT)"
boot off "$OUT/cold.rows" "$LOG/cold.log" "$COLD_EXTRA" || fail "cold server did not start"

code=$(request "$LOG/pfull.txt" "$N" "$LOG/c.json") || fail "C transport"
[ "$code" = "200" ] || fail "C HTTP $code"
cp "$OUT/cold.rows" "$LOG/c.rows" 2>/dev/null || fail "C wrote no logits dump"
ids_of "$LOG/cold.log" > "$LOG/c.ids.all"
tail -n "$N" "$LOG/c.ids.all" > "$LOG/c.ids"
prefill_line "$LOG/cold.log" > "$LOG/c.prefill"
note "C: $(cat "$LOG/c.prefill")"
n_cold=$(rows_in "$LOG/c.rows" "$rowbytes") || fail "C's dump does not line up with the vocabulary width"
[ "$n_cold" -ge "$N" ] || fail "C's dump holds $n_cold rows, fewer than N"
stop_server

##############################################################################
# Verdict
##############################################################################
rows=PASS
i=0
while [ "$i" -lt "$N" ]; do
    row_cmp "$LOG/d.rows" "$i" "$LOG/c.rows" "$i" "$rowbytes" || { rows=FAIL; break; }
    i=$((i + 1))
done
ids=FAIL
cmp -s "$LOG/d.ids" "$LOG/c.ids" && ids=PASS
ext=FAIL
[ -n "$dx" ] && [ "$dx" -gt "$ptok" ] && ext=PASS

printf 'row parity (D vs C, %s rows x %s bytes): %s\n' "$N" "$rowbytes" "$rows"
printf 'emitted ids (D vs C):                %s\n' "$ids"
printf 'the extension fired (reuse %s > prompt %s): %s\n' "${dx:-none}" "$ptok" "$ext"

if [ "$rows" = PASS ] && [ "$ids" = PASS ] && [ "$ext" = PASS ]; then
    note "PASS (the restored post-decode state is the cold state, bit for bit)"
    exit 0
fi
[ "$ext" = FAIL ] && printf 'note: the tokenizer likely merged across the prompt/emission boundary; change the prompt byte count (%s)\n' "$NCH" >&2
exit 1
