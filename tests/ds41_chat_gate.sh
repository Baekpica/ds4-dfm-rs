#!/bin/sh
# ds41 V4.1 chat gate (unit G): the port's served chat must equal the ENGINE's
# on the same request — rendered prompt ids, generated ids and the response
# shape.  The engine server --trace dumps the rendered prompt text and both id
# lists per request (server_trace.c:222-230); the port server --emit-trace
# prints the same ids ([ptok] core_v41_api.c:171-173, [emit] :246/:387).
#
# One chat+tools request (greedy, reasoning off): it covers the V4.1 head, the
# tool-schema block and the DSML tool path in one shot.
#
# usage: ds41_chat_gate.sh <engine_server> <port_server> <gguf> <engram_dir> <port> [zchain_dir]
#
# With [zchain_dir] both sides boot with the deployment's own line
# (--zchain <dir> --mem-budget-mb 110000, install.sh's start); the dir must
# have no spaces.
set -u

ENG="$1" SRV="$2" MODEL="$3" ENGRAM="$4" PORT="$5"
ZC="${6:-}"
ZFLAG=""
[ -n "$ZC" ] && ZFLAG="--zchain $ZC --mem-budget-mb 110000"
LOG=/tmp/ds41_chat_gate
REQ=/tmp/ds41_chat_gate.req.json

fail() { echo "ds41 chat gate: FAIL ($1)"; exit 1; }

cat > "$REQ" <<'JSON'
{"messages":[{"role":"system","content":"You are a terse assistant."},{"role":"user","content":"What is the weather in Paris? Use the tool."}],"max_tokens":96,"temperature":0,"reasoning_effort":"none","tools":[{"type":"function","function":{"name":"get_weather","description":"Get the weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}
JSON

# ---- engine side: boot with --trace (the oracle) ----
# rm first: the redirect truncates only once the child runs, so a grep racing
# that truncation would read the PREVIOUS run's log and pass boot_wait
# instantly (hit 2026-10-10 18:49 in the sampling gate).
rm -f "$LOG.eng.log" "$LOG.eng.trace"
"$ENG" -m "$MODEL" --engram-dir "$ENGRAM" --port "$PORT" --trace "$LOG.eng.trace" $ZFLAG > "$LOG.eng.log" 2>&1 &
EPID=$!
trap 'kill $EPID 2>/dev/null; wait $EPID 2>/dev/null' EXIT

i=0
while [ $i -lt 400 ]; do
    grep -q "listening on" "$LOG.eng.log" && break
    kill -0 "$EPID" 2>/dev/null || fail "engine server died during boot (see $LOG.eng.log)"
    sleep 2; i=$((i + 1))
done
grep -q "listening on" "$LOG.eng.log" || fail "engine server did not start listening"

curl -sS -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' -d @"$REQ" > "$LOG.eng.resp.json" \
  || fail "engine chat request failed"
grep -q '"finish_reason"' "$LOG.eng.resp.json" || fail "engine response has no finish_reason"
grep -q '"content":"' "$LOG.eng.resp.json" || fail "engine response has no content field"

kill "$EPID" 2>/dev/null; wait "$EPID" 2>/dev/null
trap - EXIT

# The trace: the prompt id line is one line starting with "prompt:"; the
# generated ids are the next line.  Extract both (single lines, no trailing
# newline, so the two sides compare byte for byte).
grep -q '^prompt:' "$LOG.eng.trace" || fail "engine trace has no prompt id line"
sed -n 's/^prompt: *//p' "$LOG.eng.trace" | tr '\n' ' ' | sed 's/ $//' > "$LOG.eng.prompt.ids"
sed -n 's/^generated: *//p' "$LOG.eng.trace" | tr '\n' ' ' | sed 's/ $//' > "$LOG.eng.gen.ids"
grep -q '^--- rendered prompt ---' "$LOG.eng.trace" || fail "engine trace has no rendered prompt"

# ---- port side: the same request through ds4-server-rs --emit-trace ----
rm -f "$LOG.srv.log"
"$SRV" -m "$MODEL" --port "$PORT" --engram-dir "$ENGRAM" --emit-trace $ZFLAG > "$LOG.srv.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT

i=0
while [ $i -lt 400 ]; do
    grep -q "listening on" "$LOG.srv.log" && break
    kill -0 "$SPID" 2>/dev/null || fail "port server died during boot (see $LOG.srv.log)"
    sleep 2; i=$((i + 1))
done
grep -q "listening on" "$LOG.srv.log" || fail "port server did not start listening"
grep -q "V4.1 serving route" "$LOG.srv.log" || fail "no V4.1 route line at boot"

curl -sS -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' -d @"$REQ" > "$LOG.srv.resp.json" \
  || fail "port chat request failed"
grep -q '"finish_reason"' "$LOG.srv.resp.json" || fail "port response has no finish_reason"

kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
trap - EXIT

# The port log: [ptok] <pos> <id> lines carry the rendered prompt ids, [emit]
# lines the generated ones; the prefill line separates the two blocks.
grep -q '\[ptok\]' "$LOG.srv.log" || fail "port log has no [ptok] lines (--emit-trace wired?)"
grep -o '\[ptok\] [0-9]* [0-9]*' "$LOG.srv.log" | awk '{print $3}' | tr '\n' ' ' | sed 's/ $//' > "$LOG.srv.prompt.ids"
grep -o '\[emit\] [0-9]* [0-9]*' "$LOG.srv.log" | awk '{print $3}' | tr '\n' ' ' | sed 's/ $//' > "$LOG.srv.gen.ids"

# ---- parity: prompt ids, generated ids, and the response message ----
diff "$LOG.eng.prompt.ids" "$LOG.srv.prompt.ids" > "$LOG.prompt.diff" 2>&1 \
  || fail "the port's prompt ids differ from the engine's (see $LOG.prompt.diff)"
diff "$LOG.eng.gen.ids" "$LOG.srv.gen.ids" > "$LOG.gen.diff" 2>&1 \
  || fail "the port's generated ids differ from the engine's (see $LOG.gen.diff)"

# ---- the response message and the finish reason ----
# When both sides parse the same way the messages must be byte-equal (modulo
# the random tool-call id, server_msgs.c random_tool_id).  On this artifact
# at greedy the model emits the call in a mixed tag style: the engine's
# parser picks one style from the open tag (server_dsml_parse.c:338) and the
# mix matches none, so the engine returns the call as assistant text with
# finish=stop, while the port parses the mix into tool_calls (the named
# deviation, plan 6.23, requested for open-grok 2026-10-10).  The ids above
# already prove the token streams equal; the deviation branch checks the
# parse: the port's content is the engine's text before the block, and every
# parsed call's name and argument values occur inside the engine's text.
efin=$(sed -n 's/.*"finish_reason":"\([^"]*\)".*/\1/p' "$LOG.eng.resp.json")
sfin=$(sed -n 's/.*"finish_reason":"\([^"]*\)".*/\1/p' "$LOG.srv.resp.json")
en=$(jq '.choices[0].message.tool_calls | length' "$LOG.eng.resp.json")
sn=$(jq '.choices[0].message.tool_calls | length' "$LOG.srv.resp.json")
if [ "$sn" -ge 1 ] && [ "$en" = "0" ]; then
    [ "$efin" = "stop" ] || fail "engine finish_reason is $efin (expected stop: the mixed-style call as text)"
    [ "$sfin" = "tool_calls" ] || fail "port finish_reason is $sfin (expected tool_calls: the tolerant parse, plan 6.23)"
    ec=$(jq -r '.choices[0].message.content' "$LOG.eng.resp.json")
    sc=$(jq -r '.choices[0].message.content' "$LOG.srv.resp.json")
    case "$ec" in "$sc"*) ;; *) fail "the port's content is not the engine's text before the DSML block" ;; esac
    i=0
    while [ "$i" -lt "$sn" ]; do
        name=$(jq -r ".choices[0].message.tool_calls[$i].function.name" "$LOG.srv.resp.json")
        case "$ec" in *"name=\"$name\""*) ;; *) fail "call $i name '$name' is not in the engine's text block" ;; esac
        jq -r ".choices[0].message.tool_calls[$i].function.arguments | fromjson | .[]" \
            "$LOG.srv.resp.json" > "$LOG.call.$i.vals"
        while IFS= read -r v; do
            case "$ec" in *"$v"*) ;; *) fail "call $i value '$v' is not in the engine's text block" ;; esac
        done < "$LOG.call.$i.vals"
        i=$((i + 1))
    done
else
    jq -S '.choices[0].message | del(.tool_calls[]?.id)' "$LOG.eng.resp.json" > "$LOG.eng.msg.json"
    jq -S '.choices[0].message | del(.tool_calls[]?.id)' "$LOG.srv.resp.json" > "$LOG.srv.msg.json"
    diff "$LOG.eng.msg.json" "$LOG.srv.msg.json" > "$LOG.msg.diff" 2>&1 \
      || fail "the port's response message differs from the engine's (see $LOG.msg.diff)"
    [ "$efin" = "$sfin" ] || fail "finish_reason differs (engine=$efin port=$sfin)"
fi
if [ "$efin" = "length" ]; then
    fail "the response hit the token cap (raise max_tokens in REQ)"
fi

echo "ds41 chat gate: PASS (prompt ids and generated ids equal the engine's on the same chat+tools request; message equal, or the mixed-style call parsed into tool_calls per plan 6.23)"
