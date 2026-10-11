#!/bin/sh
# ds41 V4.1 tool-call sampling gate (P3 close): under default sampling (temp 1)
# the tool-call SYNTAX positions must be greedy on both sides -- the port's
# hook (SemAccum::syntax_greedy, the engine's c6e2706 port) must answer every
# position exactly the way the engine's (server_generate_v41.c
# v41_syntax_greedy) does, or the sampled streams diverge.  N tool-call
# requests, one seed each (deterministic coins, comparable across sides); the
# bar: every generated id list byte-equal, every finish_reason equal, every
# parsed message equal, and every response a tool_calls call -- the engine's
# own post-fix rate (c6e2706: "默认采样 40/40 返回 tool_calls, 修前 18/20").
#
# Temperature is OMITTED on purpose: the default sampling (the engine's
# request_init default temp 1.0, gated by the sampling gate's case B).
#
# usage: ds41_toolcall_gate.sh <engine_server> <port_server> <gguf> <engram_dir> <port> [zchain_dir]
set -u

ENG="$1" SRV="$2" MODEL="$3" ENGRAM="$4" PORT="$5"
ZC="${6:-}"
ZFLAG=""
[ -n "$ZC" ] && ZFLAG="--zchain $ZC --mem-budget-mb 110000"
LOG=/tmp/ds41_toolcall_gate
N=40

fail() { echo "ds41 toolcall gate: FAIL ($1)"; exit 1; }

rm -rf "$LOG"; mkdir -p "$LOG"

boot_wait() { # <log> <pid> <name>
    i=0
    while [ $i -lt 400 ]; do
        grep -q "listening on" "$1" && return 0
        kill -0 "$2" 2>/dev/null || fail "$3 died during boot (see $1)"
        sleep 2; i=$((i + 1))
    done
    fail "$3 did not start listening"
}

fire() { # <side> <url> <log> <pid>
    i=1
    while [ $i -le "$N" ]; do
        case $((i % 5)) in
            1) CITY=Paris ;;
            2) CITY=Stockholm ;;
            3) CITY=Tokyo ;;
            4) CITY=Berlin ;;
            0) CITY=Lisbon ;;
        esac
        cat > "$LOG/req.$i.json" <<JSON
{"messages":[{"role":"system","content":"You are a terse assistant."},{"role":"user","content":"What is the weather in $CITY? Use the tool."}],"max_tokens":96,"seed":$((1000 + i)),"reasoning_effort":"none","tools":[{"type":"function","function":{"name":"get_weather","description":"Get the weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}
JSON
        curl -sS -m 600 "$2/v1/chat/completions" -H 'content-type: application/json' \
            -d @"$LOG/req.$i.json" > "$LOG/$1.resp.$i.json" || fail "$1 request $i failed"
        grep -q '"finish_reason"' "$LOG/$1.resp.$i.json" || fail "$1 response $i has no finish_reason"
        kill -0 "$4" 2>/dev/null || fail "$1 died at request $i (see $3)"
        i=$((i + 1))
    done
}

# ---- engine side: boot with --trace (the oracle) ----
rm -f "$LOG.eng.log" "$LOG.eng.trace"
"$ENG" -m "$MODEL" --engram-dir "$ENGRAM" --port "$PORT" --trace "$LOG.eng.trace" $ZFLAG > "$LOG.eng.log" 2>&1 &
EPID=$!
trap 'kill $EPID 2>/dev/null; wait $EPID 2>/dev/null' EXIT
boot_wait "$LOG.eng.log" "$EPID" "engine server"
fire eng "http://127.0.0.1:$PORT" "$LOG.eng.log" "$EPID"
kill "$EPID" 2>/dev/null; wait "$EPID" 2>/dev/null
trap - EXIT

# ---- port side: the same requests through ds4-server-rs --emit-trace ----
rm -f "$LOG.srv.log"
"$SRV" -m "$MODEL" --port "$PORT" --engram-dir "$ENGRAM" --emit-trace $ZFLAG > "$LOG.srv.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
boot_wait "$LOG.srv.log" "$SPID" "port server"
grep -q "V4.1 serving route" "$LOG.srv.log" || fail "no V4.1 route line at boot"
fire srv "http://127.0.0.1:$PORT" "$LOG.srv.log" "$SPID"
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
trap - EXIT

# ---- per-request extraction ----
# Engine trace: each request block ("===== request <id>") carries one
# "generated:" id line (server_trace.c:222-230).  Port log: the "[v41]
# prefill" line closes a request's [ptok] block and opens its [emit] block
# (ds4_ds41_draft.inc:773/789/834).
awk -v d="$LOG" '/^===== request/ { n++ } /^generated:/ { line=$0; sub(/^generated: */, "", line); print line > (d "/eng.gen." n) }' "$LOG.eng.trace"
awk -v d="$LOG" '/\[v41\] prefill/ { n++ } /^\[emit\] / { print $3 > (d "/srv.gen." n) }' "$LOG.srv.log"

# ---- per-request parity: ids, finish reason, parsed message ----
i=1; engok=0; srvok=0
while [ $i -le "$N" ]; do
    [ -s "$LOG/eng.gen.$i" ] || fail "no engine generated ids for request $i (trace extraction)"
    [ -s "$LOG/srv.gen.$i" ] || fail "no port generated ids for request $i (log extraction)"
    diff "$LOG/eng.gen.$i" "$LOG/srv.gen.$i" > "$LOG/gen.$i.diff" 2>&1 \
      || fail "request $i: generated ids differ from the engine's (see $LOG/gen.$i.diff)"

    ef=$(sed -n 's/.*"finish_reason":"\([^"]*\)".*/\1/p' "$LOG/eng.resp.$i.json")
    sf=$(sed -n 's/.*"finish_reason":"\([^"]*\)".*/\1/p' "$LOG/srv.resp.$i.json")
    [ "$ef" = "$sf" ] || fail "request $i: finish_reason differs (engine=$ef port=$sf)"
    [ "$ef" = "length" ] && fail "request $i hit the token cap (raise max_tokens)"

    ec=$(jq '.choices[0].message.tool_calls | length // 0' "$LOG/eng.resp.$i.json")
    sc=$(jq '.choices[0].message.tool_calls | length // 0' "$LOG/srv.resp.$i.json")
    [ "$ec" -ge 1 ] && engok=$((engok + 1))
    [ "$sc" -ge 1 ] && srvok=$((srvok + 1))

    jq -S '.choices[0].message | del(.tool_calls[]?.id)' "$LOG/eng.resp.$i.json" > "$LOG/eng.msg.$i.json"
    jq -S '.choices[0].message | del(.tool_calls[]?.id)' "$LOG/srv.resp.$i.json" > "$LOG/srv.msg.$i.json"
    diff "$LOG/eng.msg.$i.json" "$LOG/srv.msg.$i.json" > "$LOG/msg.$i.diff" 2>&1 \
      || fail "request $i: parsed message differs from the engine's (see $LOG/msg.$i.diff)"
    i=$((i + 1))
done

[ "$srvok" -eq "$N" ] || fail "the port returned tool_calls on $srvok/$N requests (the engine: $engok/$N)"
[ "$engok" -eq "$N" ] || fail "the engine returned tool_calls on $engok/$N requests (the c6e2706 bar is $N/$N)"

echo "ds41 toolcall gate: PASS ($N/$N requests at default sampling: generated ids, finish_reason and parsed message equal to the engine's; tool_calls $srvok/$N port vs $engok/$N engine)"
