#!/bin/sh
# ds41 P2 long-prompt gate (unit P2): the 835-token fixture
# (tests/ds41-long-prompt.txt) is the big-block case -- the whole prompt
# reaches the feed as one block (min(DS4_V41_CHUNK, n_prompt),
# ds41_forward.h:173), so each engram layer's round is 835 rows, over the
# ring depth (512): the port's fetch worker carries the reads the way the
# engine's fetch thread does (core_v41_engram.c:119-150).  The 354-token
# chat-gate request stays under the depth (main-thread rounds).
#
# Four boots, one request each (greedy, reasoning off, no tools), all on the
# canonical deployment line:
#   1 the source-built engine (--trace): the oracle ids;
#   2 the port, P2 protocol (default): ids + the prefill and host-gaps
#     readings;
#   3 the port, DS41_FEED_SYNC=1: the block-synchronous control, same
#     readings;
#   4 the port, DS41_FEED_WAIT_WITHHOLD=1: every per-layer wait must fail so
#     the route refuses (HTTP 500, "engram wait withheld") -- the proof the
#     wait is load-bearing, not decorative.
#
# PASS: engine/port(P2)/port(sync) prompt and generated ids and messages
# equal; the P2 host-gaps line shows the engram prep out of the critical
# path (its per-round engram prep at least 2x smaller than the sync
# protocol's); P2's prefill is not worse than sync's (10% noise slack);
# the withhold boot refuses.  The readings print for the plan.
#
# usage: ds41_long_gate.sh <engine_server> <port_server> <gguf> <engram_dir> <port> [zchain_dir]
set -u

ENG="$1" SRV="$2" MODEL="$3" ENGRAM="$4" PORT="$5"
ZC="${6:-}"
ZFLAG=""
[ -n "$ZC" ] && ZFLAG="--zchain $ZC --mem-budget-mb 110000"
LOG=/tmp/ds41_long_gate
TXT="$(dirname "$0")/ds41-long-prompt.txt"

fail() { echo "ds41 long gate: FAIL ($1)"; exit 1; }

[ -f "$TXT" ] || fail "long prompt fixture not found at $TXT"

EPID=""
SPID=""
cleanup() {
    if [ -n "$EPID" ]; then
        kill "$EPID" 2>/dev/null; wait "$EPID" 2>/dev/null
        EPID=""
    fi
    if [ -n "$SPID" ]; then
        kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
        SPID=""
    fi
}
trap cleanup EXIT

mkdir -p "$LOG"
jq -n --rawfile lp "$TXT" \
    '{messages:[{role:"user",content:$lp}],max_tokens:48,temperature:0,reasoning_effort:"none"}' \
    > "$LOG/req.json"
grep -q '"content"' "$LOG/req.json" || fail "request build failed (jq --rawfile)"

boot_wait() { # <log> <pid> <name>
    i=0
    while [ $i -lt 400 ]; do
        grep -q "listening on" "$1" && return 0
        kill -0 "$2" 2>/dev/null || fail "$3 died during boot (see $1)"
        sleep 2; i=$((i + 1))
    done
    fail "$3 did not start listening"
}

send() { # <name> <resp-file>
    curl -sS -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'content-type: application/json' -d @"$LOG/req.json" > "$2" \
      || fail "$1 request failed"
    grep -q '"finish_reason"' "$2" || fail "$1 response has no finish_reason"
}

# ---- 1. the engine (oracle) ----
rm -f "$LOG/eng.log" "$LOG/eng.trace"
"$ENG" --cuda -m "$MODEL" --engram-dir "$ENGRAM" --port "$PORT" --trace "$LOG/eng.trace" $ZFLAG --host 127.0.0.1 > "$LOG/eng.log" 2>&1 &
EPID=$!
boot_wait "$LOG/eng.log" "$EPID" "engine server"
send engine "$LOG/eng.resp.json"
sleep 2
kill "$EPID" 2>/dev/null; wait "$EPID" 2>/dev/null
EPID=""

# ---- 2. the port, P2 protocol ----
# The feed (and its protocol lines) opens at the first request, not at boot.
rm -f "$LOG/p2.srv.log"
"$SRV" --cuda -m "$MODEL" --port "$PORT" --engram-dir "$ENGRAM" --emit-trace $ZFLAG --host 127.0.0.1 > "$LOG/p2.srv.log" 2>&1 &
SPID=$!
boot_wait "$LOG/p2.srv.log" "$SPID" "port server (P2)"
grep -q "V4.1 route:" "$LOG/p2.srv.log" || fail "no V4.1 route line at boot (P2)"
send P2 "$LOG/p2.resp.json"
grep -q "engram rows: io_uring (" "$LOG/p2.srv.log" || fail "P2 boot did not open the io_uring ring"
grep -q "synchronous protocol" "$LOG/p2.srv.log" && fail "P2 boot fell back to the synchronous protocol"
grep -q "engram fetch thread refused to start" "$LOG/p2.srv.log" && fail "P2 fetch worker did not start"
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
SPID=""

# ---- 3. the port, DS41_FEED_SYNC=1 (the block-synchronous control) ----
rm -f "$LOG/sync.srv.log"
DS41_FEED_SYNC=1 "$SRV" --cuda -m "$MODEL" --port "$PORT" --engram-dir "$ENGRAM" --emit-trace $ZFLAG --host 127.0.0.1 > "$LOG/sync.srv.log" 2>&1 &
SPID=$!
boot_wait "$LOG/sync.srv.log" "$SPID" "port server (sync)"
send sync "$LOG/sync.resp.json"
grep -q "synchronous protocol (DS41_FEED_SYNC)" "$LOG/sync.srv.log" || fail "sync boot did not select the synchronous protocol"
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
SPID=""

# ---- 4. the port, DS41_FEED_WAIT_WITHHOLD=1 (the refusal control) ----
rm -f "$LOG/withhold.srv.log"
DS41_FEED_WAIT_WITHHOLD=1 "$SRV" --cuda -m "$MODEL" --port "$PORT" --engram-dir "$ENGRAM" --emit-trace $ZFLAG --host 127.0.0.1 > "$LOG/withhold.srv.log" 2>&1 &
SPID=$!
boot_wait "$LOG/withhold.srv.log" "$SPID" "port server (withhold)"
code=$(curl -sS -m 900 -o "$LOG/withhold.resp.json" -w '%{http_code}' \
    "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' -d @"$LOG/req.json") \
  || fail "withhold request failed at the transport level"
[ "$code" = "500" ] || fail "withhold control returned HTTP $code (expected 500)"
grep -q "withhold true" "$LOG/withhold.srv.log" || fail "withhold boot did not arm the hook"
# The Rust error text is collapsed by the bridge ("V4.1 generate failed"); the
# native's own line is the refusal evidence: the layer's rows never arrived and
# the forward refused instead of consuming stale rows.
grep -q "feed wait failed" "$LOG/withhold.srv.log" \
  || fail "withhold control did not refuse the engram layer (was the wait bypassed?)"
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
SPID=""

# ---- per-side id extraction ----
# Engine: one "===== request" block; "prompt:"/"generated:" inside
# (server_trace.c:222-230).  Port: [ptok] before the request's
# "[v41] prefill" line, [emit] after (core_v41_api.c:171-173 / :246).
awk '
    /^===== request / { r++ }
    /^prompt:/    { sub(/^prompt: */, "");    P[r] = $0 }
    /^generated:/ { sub(/^generated: */, ""); G[r] = $0 }
    END { for (i = 1; i <= r; i++) print i "|" P[i] "|" G[i] }
' "$LOG/eng.trace" > "$LOG/eng.ids"
for side in p2 sync; do
    awk '
        BEGIN { r = 0 }
        /\[v41\] prefill/ { pf = 1; next }
        /\[ptok\]/ { if (pf) { r++; pf = 0 } ; split($0, a, " "); P[r] = P[r] " " a[3]; next }
        /\[emit\]/ { split($0, a, " "); G[r] = G[r] " " a[3]; next }
        END { for (i = 0; i <= r; i++) { sub(/^ /, "", P[i]); sub(/^ /, "", G[i]); print i+1 "|" P[i] "|" G[i] } }
    ' "$LOG/$side.srv.log" > "$LOG/$side.ids"
done

# ---- parity: prompt ids, generated ids, messages, finish ----
for side in p2 sync; do
    awk -F'|' 'NR==1{print $2}' "$LOG/eng.ids"  > "$LOG/eng.prompt.ids"
    awk -F'|' 'NR==1{print $3}' "$LOG/eng.ids"  > "$LOG/eng.gen.ids"
    awk -F'|' 'NR==1{print $2}' "$LOG/$side.ids" > "$LOG/$side.prompt.ids"
    awk -F'|' 'NR==1{print $3}' "$LOG/$side.ids" > "$LOG/$side.gen.ids"
    diff "$LOG/eng.prompt.ids" "$LOG/$side.prompt.ids" > "$LOG/$side.prompt.diff" 2>&1 \
      || fail "$side prompt ids differ from the engine's (see $LOG/$side.prompt.diff)"
    diff "$LOG/eng.gen.ids" "$LOG/$side.gen.ids" > "$LOG/$side.gen.diff" 2>&1 \
      || fail "$side generated ids differ from the engine's (see $LOG/$side.gen.diff)"
    jq -S '.choices[0].message' "$LOG/eng.resp.json" > "$LOG/eng.$side.msg.json"
    jq -S '.choices[0].message' "$LOG/$side.resp.json" > "$LOG/$side.msg.json"
    diff "$LOG/eng.$side.msg.json" "$LOG/$side.msg.json" > "$LOG/$side.msg.diff" 2>&1 \
      || fail "$side response message differs from the engine's (see $LOG/$side.msg.diff)"
    efin=$(sed -n 's/.*"finish_reason":"\([^"]*\)".*/\1/p' "$LOG/eng.resp.json")
    sfin=$(sed -n 's/.*"finish_reason":"\([^"]*\)".*/\1/p' "$LOG/$side.resp.json")
    [ "$efin" = "$sfin" ] || fail "$side finish_reason differs (engine=$efin $side=$sfin)"
done
[ "$(jq '.choices[0].message.tool_calls | length' "$LOG/eng.resp.json")" = "0" ] \
  || fail "the engine emitted a tool call on a no-tools request (unexpected)"

# ---- the readings ----
pref_s() { grep -o '\[v41\] prefill [0-9]* token [0-9.]*s' "$1" | tail -1 | sed 's/.* token \([0-9.]*\)s/\1/'; }
prep_ms() { grep -o 'engram prep [0-9.]*' "$1" | tail -1 | awk '{print $3}'; }
p2s=$(pref_s "$LOG/p2.srv.log");  p2prep=$(prep_ms "$LOG/p2.srv.log")
sys=$(pref_s "$LOG/sync.srv.log"); sprep=$(prep_ms "$LOG/sync.srv.log")
[ -n "$p2s" ] && [ -n "$sys" ] || fail "a boot printed no [v41] prefill line"
[ -n "$p2prep" ] && [ -n "$sprep" ] || fail "a boot printed no host-gaps line"

awk -v p="$p2prep" -v s="$sprep" 'BEGIN { exit !(p * 2 <= s) }' \
  || fail "the P2 engram prep is not out of the critical path (P2 ${p2prep} ms vs sync ${sprep} ms)"
awk -v p="$p2s" -v s="$sys" 'BEGIN { exit !(p <= s * 1.10) }' \
  || fail "P2 prefill worse than the sync protocol by >10% (P2 ${p2s}s vs sync ${sys}s)"

ptok=$(jq '.timings.prefill_tokens' "$LOG/p2.resp.json")
echo "ds41 long gate: PASS"
echo "  prompt: ${ptok} tokens (the 835-token fixture + the head)"
echo "  prefill: P2 ${p2s}s ($(jq '.timings.prefill_tok_s' "$LOG/p2.resp.json") t/s) vs sync ${sys}s ($(jq '.timings.prefill_tok_s' "$LOG/sync.resp.json") t/s)"
echo "  engram prep/round: P2 ${p2prep} ms vs sync ${sprep} ms"
echo "  decode: P2 $(jq '.timings.decode_tok_s' "$LOG/p2.resp.json") t/s vs sync $(jq '.timings.decode_tok_s' "$LOG/sync.resp.json") t/s"
echo "  ids: prompt + generated equal to the engine's on both protocols"
echo "  withhold control: HTTP 500, the native refused the engram layer as required"
