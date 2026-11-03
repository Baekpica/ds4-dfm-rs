#!/bin/sh
# Gemma-4 VQ OpenAI serving gate (P9). Boots ds4-server on the R13+8 release
# and proves the four parts the handoff names:
#
#   models      GET /v1/models answers and lists the family id
#   cuda        the backend line says CUDA and the build is sm_89
#   completion  POST /v1/chat/completions returns a nonempty completion with
#               a finish_reason
#   determinism the same request twice returns the same content, finish
#               reason and completion-token count
#
# The two responses are compared field by field, not as whole documents: the
# response id and created stamp differ by design.
#
# usage: gemma4_serving_http_gate.sh <server> <release-dir> <port> [prompt]
set -u

SRV="$1" MODEL="$2" PORT="$3" PROMPT="${4:-Hello}"
LOG=/tmp/gemma4_http_gate
LOCK=/tmp/ds4-gemma4-gate.lock

fail() { echo "gemma4 http gate: FAIL ($1)"; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "$1 is not on PATH"; }
need curl
need jq

rm -f "$LOG.server.log" "$LOG.models.json" "$LOG.chat1.json" "$LOG.chat2.json"
DS4_LOCK_FILE="$LOCK" "$SRV" -m "$MODEL" --cuda --host 127.0.0.1 --port "$PORT" \
    --mem-floor-gb 2 > "$LOG.server.log" 2>&1 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null' EXIT

i=0
while [ $i -lt 300 ]; do
    grep -q "listening on" "$LOG.server.log" && break
    kill -0 "$SRV_PID" 2>/dev/null || fail "server died during boot (see $LOG.server.log)"
    sleep 2
    i=$((i + 1))
done
grep -q "listening on" "$LOG.server.log" || fail "server did not start listening"
grep -q "CUDA backend initialized" "$LOG.server.log" || fail "no CUDA backend line at boot"
grep -q "sm_89" "$LOG.server.log" || fail "the backend did not report sm_89"

curl -sS -m 30 "http://127.0.0.1:$PORT/v1/models" > "$LOG.models.json" \
    || fail "GET /v1/models failed"
jq -e '.object == "list" and (.data | length) > 0' "$LOG.models.json" >/dev/null \
    || fail "GET /v1/models: no list object"
ID=$(jq -r '.data[0].id' "$LOG.models.json")
[ -n "$ID" ] && [ "$ID" != "null" ] || fail "GET /v1/models: no model id"

request() {
    printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":16,"temperature":0}' \
        "$ID" "$PROMPT"
}
curl -sS -m 300 -H 'Content-Type: application/json' -d "$(request)" \
    "http://127.0.0.1:$PORT/v1/chat/completions" > "$LOG.chat1.json" \
    || fail "chat completion 1 failed"
curl -sS -m 300 -H 'Content-Type: application/json' -d "$(request)" \
    "http://127.0.0.1:$PORT/v1/chat/completions" > "$LOG.chat2.json" \
    || fail "chat completion 2 failed"

for f in "$LOG.chat1.json" "$LOG.chat2.json"; do
    jq -e '.choices[0].message.content != null and .choices[0].finish_reason != null' \
        "$f" >/dev/null || fail "$(basename "$f"): no choices, content or finish_reason"
done

field() { jq -r "$2" "$1"; }
TEXT1=$(field "$LOG.chat1.json" '.choices[0].message.content')
TEXT2=$(field "$LOG.chat2.json" '.choices[0].message.content')
FIN1=$(field "$LOG.chat1.json" '.choices[0].finish_reason')
FIN2=$(field "$LOG.chat2.json" '.choices[0].finish_reason')
N1=$(field "$LOG.chat1.json" '.usage.completion_tokens')
N2=$(field "$LOG.chat2.json" '.usage.completion_tokens')
[ -n "$TEXT1" ] || fail "chat completion 1 returned empty content"
[ "$TEXT1" = "$TEXT2" ] || fail "two identical requests returned different text"
[ "$FIN1" = "$FIN2" ] || fail "two identical requests returned different finish reasons"
[ "$N1" = "$N2" ] || fail "two identical requests returned different token counts"

echo "gemma4 http gate: PASS"
echo "  model:        $ID"
echo "  backend:      $(grep -m1 'CUDA backend initialized' "$LOG.server.log" | sed 's/^ds4: //')"
echo "  finish:       $FIN1 after $N1 tokens"
echo "  text:         $TEXT1"
