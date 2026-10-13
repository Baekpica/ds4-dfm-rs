#!/bin/sh
# ds41 V4.1 wire gate (unit H): the Anthropic (/v1/messages) and Responses
# (/v1/responses) surfaces on the v41 route.  The engine --trace dumps the
# rendered prompt and both id lists per request (server_trace.c:222-230); the
# port --emit-trace prints [ptok] (core_v41_api.c:171-173) and [emit] (:246).
# Both surfaces share the chat renderer (server_httpd.c:246-252), so the ids
# must be byte-identical at greedy; this drives the two parse paths end to end.
#
# Three greedy requests (reasoning off) through one boot per side:
#   1  POST /v1/messages   messages + max_tokens + reasoning_effort
#   2  POST /v1/responses  input + max_output_tokens + reasoning.effort
#   3  POST /v1/chat/completions  THE install.sh smoke request verbatim
#      (model "deepseek-chat" = the alias that disables thinking,
#      server_msgs.c:300-301; temperature 0, max_tokens 48, the finance
#      question -- install.sh's own smoke)
#   4  POST /v1/messages   the chat gate's weather+tool request, Anthropic shape
#   5  POST /v1/responses  the same, in the Responses shape
#      The tools legs: the engine's one-style parser rejects the artifact's
#      mixed DSML block (serves it as text); the port parses it into the
#      surface's own tool item (the named deviation, plan 6.23).  The ids
#      must still be byte-identical, and the port's call must be traceable
#      in the engine's text block.
#
# usage: ds41_wire_gate.sh <engine_server> <port_server> <gguf> <engram_dir> <port> [zchain_dir]
#
# With [zchain_dir] both sides boot with the deployment's own line:
# --cuda -m <gguf> --zchain <dir> --engram-dir <dir> --mem-budget-mb 110000
# --host 127.0.0.1 --port <port> (install.sh's start, verbatim modulo the
# trace instruments).  The dir must have no spaces.
set -u

ENG="$1" SRV="$2" MODEL="$3" ENGRAM="$4" PORT="$5"
ZC="${6:-}"
ZFLAG=""
# The canonical install.sh line: --zchain <sidecar> --mem-budget-mb 110000
# (its BUDGET_MB).  The budget is inert for the ids; carrying it exercises
# both sides' flag acceptance and the L1 load gate at boot.
[ -n "$ZC" ] && ZFLAG="--zchain $ZC --mem-budget-mb 110000"
LOG=/tmp/ds41_wire_gate

fail() { echo "ds41 wire gate: FAIL ($1)"; exit 1; }

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
cat > "$LOG/req.anthropic.json" <<'JSON'
{"max_tokens":48,"temperature":0,"reasoning_effort":"none","messages":[{"role":"user","content":"Name three colors."}]}
JSON
cat > "$LOG/req.responses.json" <<'JSON'
{"input":"Name three colors.","max_output_tokens":48,"temperature":0,"reasoning":{"effort":"none"}}
JSON
cat > "$LOG/req.smoke.json" <<'JSON'
{"model":"deepseek-chat","temperature":0,"max_tokens":48,"messages":[{"role":"user","content":"In one sentence, what is a price-to-earnings ratio?"}]}
JSON
cat > "$LOG/req.anthropic.tools.json" <<'JSON'
{"max_tokens":96,"temperature":0,"reasoning_effort":"none","messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],"tools":[{"name":"get_weather","description":"Get the weather for a city","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]}
JSON
cat > "$LOG/req.responses.tools.json" <<'JSON'
{"input":"What is the weather in Paris? Use the tool.","max_output_tokens":96,"temperature":0,"reasoning":{"effort":"none"},"tools":[{"type":"function","name":"get_weather","description":"Get the weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]}
JSON

boot_wait() { # <log> <pid> <name>
    i=0
    while [ $i -lt 400 ]; do
        grep -q "listening on" "$1" && return 0
        kill -0 "$2" 2>/dev/null || fail "$3 died during boot (see $1)"
        sleep 2; i=$((i + 1))
    done
    fail "$3 did not start listening"
}

# ---- engine side (the oracle) ----
# rm first: the redirect truncates only once the child runs, so a grep racing
# that truncation would read the PREVIOUS run's log and pass boot_wait
# instantly.
rm -f "$LOG/eng.log" "$LOG/eng.trace"
"$ENG" --cuda -m "$MODEL" --engram-dir "$ENGRAM" --port "$PORT" --trace "$LOG/eng.trace" $ZFLAG --host 127.0.0.1 > "$LOG/eng.log" 2>&1 &
EPID=$!
boot_wait "$LOG/eng.log" "$EPID" "engine server"
curl -sS -m 900 "http://127.0.0.1:$PORT/v1/messages" \
    -H 'content-type: application/json' -d @"$LOG/req.anthropic.json" > "$LOG/eng.anthropic.json" \
  || fail "engine anthropic request failed"
grep -q '"stop_reason"' "$LOG/eng.anthropic.json" || fail "engine anthropic response has no stop_reason"
curl -sS -m 900 "http://127.0.0.1:$PORT/v1/responses" \
    -H 'content-type: application/json' -d @"$LOG/req.responses.json" > "$LOG/eng.responses.json" \
  || fail "engine responses request failed"
grep -q '"status"' "$LOG/eng.responses.json" || fail "engine responses response has no status"
curl -sS -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' -d @"$LOG/req.smoke.json" > "$LOG/eng.smoke.json" \
  || fail "engine smoke request failed"
grep -q '"choices"' "$LOG/eng.smoke.json" || fail "engine smoke response has no choices"
curl -sS -m 900 "http://127.0.0.1:$PORT/v1/messages" \
    -H 'content-type: application/json' -d @"$LOG/req.anthropic.tools.json" > "$LOG/eng.anthropic.tools.json" \
  || fail "engine anthropic tools request failed"
grep -q '"stop_reason"' "$LOG/eng.anthropic.tools.json" || fail "engine anthropic tools response has no stop_reason"
curl -sS -m 900 "http://127.0.0.1:$PORT/v1/responses" \
    -H 'content-type: application/json' -d @"$LOG/req.responses.tools.json" > "$LOG/eng.responses.tools.json" \
  || fail "engine responses tools request failed"
grep -q '"status"' "$LOG/eng.responses.tools.json" || fail "engine responses tools response has no status"
kill "$EPID" 2>/dev/null; wait "$EPID" 2>/dev/null
EPID=""

# ---- port side: the same requests ----
rm -f "$LOG/srv.log"
"$SRV" --cuda -m "$MODEL" --port "$PORT" --engram-dir "$ENGRAM" --emit-trace $ZFLAG --host 127.0.0.1 > "$LOG/srv.log" 2>&1 &
SPID=$!
boot_wait "$LOG/srv.log" "$SPID" "port server"
grep -q "V4.1 route:" "$LOG/srv.log" || fail "no V4.1 route line at boot"
curl -sS -m 900 "http://127.0.0.1:$PORT/v1/messages" \
    -H 'content-type: application/json' -d @"$LOG/req.anthropic.json" > "$LOG/srv.anthropic.json" \
  || fail "port anthropic request failed"
grep -q '"stop_reason"' "$LOG/srv.anthropic.json" || fail "port anthropic response has no stop_reason"
curl -sS -m 900 "http://127.0.0.1:$PORT/v1/responses" \
    -H 'content-type: application/json' -d @"$LOG/req.responses.json" > "$LOG/srv.responses.json" \
  || fail "port responses request failed"
grep -q '"status"' "$LOG/srv.responses.json" || fail "port responses response has no status"
curl -sS -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' -d @"$LOG/req.smoke.json" > "$LOG/srv.smoke.json" \
  || fail "port smoke request failed"
grep -q '"choices"' "$LOG/srv.smoke.json" || fail "port smoke response has no choices"
curl -sS -m 900 "http://127.0.0.1:$PORT/v1/messages" \
    -H 'content-type: application/json' -d @"$LOG/req.anthropic.tools.json" > "$LOG/srv.anthropic.tools.json" \
  || fail "port anthropic tools request failed"
grep -q '"stop_reason"' "$LOG/srv.anthropic.tools.json" || fail "port anthropic tools response has no stop_reason"
curl -sS -m 900 "http://127.0.0.1:$PORT/v1/responses" \
    -H 'content-type: application/json' -d @"$LOG/req.responses.tools.json" > "$LOG/srv.responses.tools.json" \
  || fail "port responses tools request failed"
grep -q '"status"' "$LOG/srv.responses.tools.json" || fail "port responses tools response has no status"
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
SPID=""

# ---- per-request id extraction ----
# Engine: one "===== request <id>" .. block per request; "prompt:"/"generated:"
# lines inside (server_trace.c:222-230).  Port: [ptok] blocks precede their
# request's "[v41] prefill" line, [emit] lines follow it; a new [ptok] block
# after a prefill line starts the next request.
awk '
    /^===== request / { r++ }
    /^prompt:/    { sub(/^prompt: */, "");    P[r] = $0 }
    /^generated:/ { sub(/^generated: */, ""); G[r] = $0 }
    END { for (i = 1; i <= r; i++) print i "|" P[i] "|" G[i] }
' "$LOG/eng.trace" > "$LOG/eng.ids"
awk '
    BEGIN { r = 0 }
    /\[v41\] prefill/ { pf = 1; next }
    /\[ptok\]/ { if (pf) { r++; pf = 0 } ; split($0, a, " "); P[r] = P[r] " " a[3]; next }
    /\[emit\]/ { split($0, a, " "); G[r] = G[r] " " a[3]; next }
    END { for (i = 0; i <= r; i++) { sub(/^ /, "", P[i]); sub(/^ /, "", G[i]); print i+1 "|" P[i] "|" G[i] } }
' "$LOG/srv.log" > "$LOG/srv.ids"

# ---- per-surface parity ----
check() { # <name> <request index>
    name="$1"; n="$2"
    e=$(sed -n "${n}p" "$LOG/eng.ids")
    s=$(sed -n "${n}p" "$LOG/srv.ids")
    [ -n "$e" ] || fail "$name: engine ids missing"
    [ -n "$s" ] || fail "$name: port ids missing"
    ep=$(printf '%s' "$e" | cut -d'|' -f2)
    eg=$(printf '%s' "$e" | cut -d'|' -f3)
    sp=$(printf '%s' "$s" | cut -d'|' -f2)
    sg=$(printf '%s' "$s" | cut -d'|' -f3)
    [ "$ep" = "$sp" ] || { printf 'engine: %s\nport:   %s\n' "$ep" "$sp" > "$LOG/$name.prompt.diff"; fail "$name: prompt ids differ (see $LOG/$name.prompt.diff)"; }
    [ "$eg" = "$sg" ] || { printf 'engine: %s\nport:   %s\n' "$eg" "$sg" > "$LOG/$name.gen.diff"; fail "$name: generated ids differ (see $LOG/$name.gen.diff)"; }
    echo "  $name: ids identical ($(printf '%s' "$eg" | wc -w) generated)"
}

check anthropic 1
check responses 2
check smoke 3
check anthropic-tools 4
check responses-tools 5

# ---- finish parity ----
es=$(sed -n 's/.*"stop_reason":"\([^"]*\)".*/\1/p' "$LOG/eng.anthropic.json")
ss=$(sed -n 's/.*"stop_reason":"\([^"]*\)".*/\1/p' "$LOG/srv.anthropic.json")
[ "$es" = "$ss" ] || fail "anthropic stop_reason differs (engine=$es port=$ss)"
[ "$es" = "max_tokens" ] && fail "anthropic response hit the token cap"
er=$(sed -n 's/.*"status":"\([^"]*\)".*/\1/p' "$LOG/eng.responses.json")
sr=$(sed -n 's/.*"status":"\([^"]*\)".*/\1/p' "$LOG/srv.responses.json")
[ "$er" = "$sr" ] || fail "responses status differs (engine=$er port=$sr)"
[ "$er" = "incomplete" ] && fail "responses response incomplete (token cap?)"
efin=$(sed -n 's/.*"finish_reason":"\([^"]*\)".*/\1/p' "$LOG/eng.smoke.json")
sfin=$(sed -n 's/.*"finish_reason":"\([^"]*\)".*/\1/p' "$LOG/srv.smoke.json")
[ "$efin" = "$sfin" ] || fail "smoke finish_reason differs (engine=$efin port=$sfin)"
[ "$efin" = "length" ] && fail "smoke response hit the token cap"

# ---- the tools legs (plan 6.23 item 8): the port's tool surface ----
# At greedy the artifact emits the DSML block in the mixed style; the
# engine's one-style parser rejects it (text, no tool item) and the port
# parses it into the surface's own tool item (the named deviation).  A
# pure-style emission parses on both sides and the surfaces must then
# match modulo the random ids.  Either way the ids above are identical.
et=$(jq '[.content[]? | select(.type=="tool_use")] | length' "$LOG/eng.anthropic.tools.json")
st=$(jq '[.content[]? | select(.type=="tool_use")] | length' "$LOG/srv.anthropic.tools.json")
[ "$st" -ge 1 ] || fail "anthropic tools: the port served no tool_use block"
if [ "$et" = "0" ]; then
    ssr=$(sed -n 's/.*"stop_reason":"\([^"]*\)".*/\1/p' "$LOG/srv.anthropic.tools.json")
    [ "$ssr" = "tool_use" ] || fail "anthropic tools: port stop_reason is $ssr (expected tool_use)"
    ec=$(jq -r '[.content[]? | select(.type=="text") | .text] | join("")' "$LOG/eng.anthropic.tools.json")
    jq -r '.content[] | select(.type=="tool_use") | .name' "$LOG/srv.anthropic.tools.json" > "$LOG/anthropic.tools.names"
    while IFS= read -r name; do
        case "$ec" in *"name=\"$name\""*) ;; *) fail "anthropic tools: call name '$name' is not in the engine's text block" ;; esac
    done < "$LOG/anthropic.tools.names"
    jq -r '.content[] | select(.type=="tool_use") | .input | .[]' "$LOG/srv.anthropic.tools.json" > "$LOG/anthropic.tools.vals"
    while IFS= read -r v; do
        case "$ec" in *"$v"*) ;; *) fail "anthropic tools: value '$v' is not in the engine's text block" ;; esac
    done < "$LOG/anthropic.tools.vals"
    echo "  anthropic-tools: ids identical; the mixed-style call parsed into tool_use (plan 6.23)"
else
    jq -S '[.content[] | select(.type=="tool_use") | {name, input}]' "$LOG/eng.anthropic.tools.json" > "$LOG/eng.anthropic.tools.msg"
    jq -S '[.content[] | select(.type=="tool_use") | {name, input}]' "$LOG/srv.anthropic.tools.json" > "$LOG/srv.anthropic.tools.msg"
    diff "$LOG/eng.anthropic.tools.msg" "$LOG/srv.anthropic.tools.msg" > "$LOG/anthropic.tools.msg.diff" 2>&1 \
      || fail "anthropic tools: the port's tool_use blocks differ from the engine's (see $LOG/anthropic.tools.msg.diff)"
    echo "  anthropic-tools: ids identical; both sides parsed the call"
fi

er=$(jq '[.output[]? | select(.type=="function_call")] | length' "$LOG/eng.responses.tools.json")
sr=$(jq '[.output[]? | select(.type=="function_call")] | length' "$LOG/srv.responses.tools.json")
[ "$sr" -ge 1 ] || fail "responses tools: the port served no function_call item"
if [ "$er" = "0" ]; then
    ec=$(jq -r '[.output[]? | select(.type=="message") | .content[]? | select(.type=="output_text") | .text] | join("")' "$LOG/eng.responses.tools.json")
    jq -r '.output[] | select(.type=="function_call") | .name' "$LOG/srv.responses.tools.json" > "$LOG/responses.tools.names"
    while IFS= read -r name; do
        case "$ec" in *"name=\"$name\""*) ;; *) fail "responses tools: call name '$name' is not in the engine's text block" ;; esac
    done < "$LOG/responses.tools.names"
    jq -r '.output[] | select(.type=="function_call") | .arguments | fromjson | .[]' "$LOG/srv.responses.tools.json" > "$LOG/responses.tools.vals"
    while IFS= read -r v; do
        case "$ec" in *"$v"*) ;; *) fail "responses tools: value '$v' is not in the engine's text block" ;; esac
    done < "$LOG/responses.tools.vals"
    echo "  responses-tools: ids identical; the mixed-style call parsed into function_call (plan 6.23)"
else
    jq -S '[.output[] | select(.type=="function_call") | {name, arguments}]' "$LOG/eng.responses.tools.json" > "$LOG/eng.responses.tools.msg"
    jq -S '[.output[] | select(.type=="function_call") | {name, arguments}]' "$LOG/srv.responses.tools.json" > "$LOG/srv.responses.tools.msg"
    diff "$LOG/eng.responses.tools.msg" "$LOG/srv.responses.tools.msg" > "$LOG/responses.tools.msg.diff" 2>&1 \
      || fail "responses tools: the port's function_call items differ from the engine's (see $LOG/responses.tools.msg.diff)"
    echo "  responses-tools: ids identical; both sides parsed the call"
fi

echo "ds41 wire gate: PASS (anthropic + responses + the install.sh smoke + the two tools legs: prompt ids, generated ids equal the engine's; tool items parsed or traced to the engine's text)"
