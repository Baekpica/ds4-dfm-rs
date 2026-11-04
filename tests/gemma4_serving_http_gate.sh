#!/bin/sh
# Gemma-4 VQ OpenAI serving gate (P9), reworked to the repository gate
# standard: boot ds4-server on the R13+8 release, poll the listener with a
# dead-process check, drive the OpenAI surface end to end, and prove:
#
#   boot         the log carries "listening on" and the CUDA backend line
#                "ds4: CUDA backend initialized on <name> (sm_<major><minor>)"
#                (ds4_cuda.cu:4496-4498), and the stamped capability equals the
#                expected one: sm_89 by default (the RTX 4070 SUPER), sm_121 on
#                the Spark GB10 (argument 4 overrides it)
#   models       GET /v1/models is HTTP 200, an object "list" with a data
#                array and a nonempty model id
#   completion   POST /v1/chat/completions (non-stream, temperature 0) is HTTP
#                200 with nonempty message content, a nonempty finish_reason,
#                and usage.completion_tokens > 0
#   determinism  the same request twice returns the same content, finish_reason
#                and completion-token count, compared field by field (the
#                response id and created stamp differ by design)
#   stream       POST /v1/chat/completions with "stream":true answers SSE: data
#                lines, at least one delta with nonempty content, a
#                finish_reason chunk, no error event, and data: [DONE] as the
#                LAST data line
#   refusal      the negative control: max_tokens -1 is a parse error, so the
#                server must answer HTTP 400 with error.type
#                "invalid_request_error" and error.message
#                "max_tokens must be >= 0". A 200 here means the server never
#                parsed the body and the gate could pass blind; it fails by
#                name instead.
#   mtp          optional (argument 6): boot a second server with
#                --mtp <file> --mtp-draft 6, the ceiling (G4_MTP_DRAFTS,
#                ds4_gemma4_plan.h:71; values above six are refused,
#                crates/ds4-core/src/lib.rs:341-363), and require its content,
#                finish_reason and completion-token count to equal the no-MTP
#                run. That is the P8 losslessness claim over the server route.
#                Skipped, and said so in the PASS line, when no path is given.
#
# One server at a time: the engine's single-instance lock (DS4_LOCK_FILE)
# admits one model-loading process, so the first server is stopped and reaped
# before the second boots. The MTP arm takes port+1 so a recycled socket
# cannot race the bind.
#
# usage: gemma4_serving_http_gate.sh <server> <release-dir> <port> \
#            [expected-sm] [prompt] [mtp-gguf]
#
#   <release-dir> is host-local: /data/vq-artifacts/gguf-family-it-r8 on the
#   4070 SUPER, ~/models/gguf-family-it-r8 on the Spark.
#   [expected-sm] defaults to sm_89; pass sm_121 for the Spark build.
#   [prompt] is inserted with jq --arg, so quotes and backslashes in it are
#   escaped instead of breaking the request body.
#
# Boot flags come from the source, not from habit:
#   -c 2048 --mem-floor-gb 2   the serving line this family is documented with
#        (gemma4-port-handoff.md:149-151: -m <release> --cuda -c 2048 --host
#        127.0.0.1 --port <p> --mem-floor-gb 2, the gate's own lock file).
#        The 2 GiB floor is required on a 12 GiB card: 6.71 GiB of weights
#        plus the default 4 GiB floor leaves no room for the context quote
#        (docs/gemma4-vq.md "Limits and open work").
#   reasoning_effort "none"    closes Gemma's thought channel before the model
#        turn (render.rs:3413-3467; pinned by the test
#        chat_without_thinking_closes_the_channel, render.rs:3690-3700), so
#        content lands inside max_tokens instead of after the model's
#        deliberation. Without the field this family thinks by default:
#        parse.rs:3182 sets thinking_enabled = !engine_defaults and only the
#        V4.1 branch sets engine_defaults
#        (crates/ds4-server/src/bin/ds4-server-rs.rs:681-692).
#
# The status and body of every assertion. In this tree `make ds4-server`
# builds the Rust server (Makefile:722-731) and `ds4-server-c` the C engine
# server; both were read, and the route, the parser and the envelopes agree,
# so the gate's expectations hold for either binary the operator passes:
#   /v1/models        {"object":"list","data":[{...,"id":...}]}, 200
#                     (ds4_server.c:21820-21838; the Rust port: models.rs:
#                     157-174, serve.rs:1171-1184)
#   non-stream chat   {"object":"chat.completion","choices":[{"message":
#                     {"content":...},"finish_reason":...}],"usage":{...}},
#                     200 (ds4_server.c:10845-10874; stream.rs:2488-2549)
#   stream            "data: {...chat.completion.chunk...}" records with
#                     delta.content, a finish_reason chunk, then
#                     "data: [DONE]" (ds4_server.c:8804-8806; stream.rs:588-623
#                     and 655-658)
#   refusal           parse error -> 400 with the OpenAI envelope
#                     (ds4_server.c:5382-5387 sets "max_tokens must be >= 0",
#                     5523-5527 keeps it at the shared bad: exit, 23260-23264
#                     sends it, 13599-13617 and 8596-8612 build the envelope;
#                     the Rust port mirrors it: parse.rs:684-698 and 669-674,
#                     serve.rs:1275-1283, error.rs:20-28 and 44-50)
set -u

fail() { echo "gemma4 http gate: FAIL ($1)"; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "$1 is not on PATH"; }

# Before any positional is read: with set -u an absent $1 aborts the script
# with an unbound-variable error instead of the usage line.
[ "$#" -ge 3 ] || fail "usage: gemma4_serving_http_gate.sh <server> <release-dir> <port> [expected-sm] [prompt] [mtp-gguf]"

SRV="$1" MODEL="$2" PORT="$3" EXPECT_SM="${4:-sm_89}" PROMPT="${5:-Hello}" MTP="${6:-}"
LOG=/tmp/gemma4_http_gate
LOCK=/tmp/ds4-gemma4-gate.lock
MTP_PORT=$((PORT + 1))
CHAT_MAX=32   # a short answer; a "length" stop is still a valid completion

need curl
need jq

# rm first: the redirect truncates only once the child runs, so a grep racing
# that truncation would read the PREVIOUS run's log and pass boot_wait
# instantly (the same trap ds41_chat_gate.sh records).
rm -f "$LOG.server.log" "$LOG.models.json" "$LOG.chat1.json" "$LOG.chat2.json" \
      "$LOG.stream.sse" "$LOG.stream.data.jsonl" "$LOG.neg.json" \
      "$LOG.mtp.log" "$LOG.mtp.chat.json"

start_server() {   # $1 log  $2 port  $3 mtp path ("" = the no-MTP arm)
    rm -f "$1"
    if [ -n "$3" ]; then
        DS4_LOCK_FILE="$LOCK" "$SRV" -m "$MODEL" --cuda --host 127.0.0.1 --port "$2" \
            -c 2048 --mem-floor-gb 2 --mtp "$3" --mtp-draft 6 > "$1" 2>&1 &
    else
        DS4_LOCK_FILE="$LOCK" "$SRV" -m "$MODEL" --cuda --host 127.0.0.1 --port "$2" \
            -c 2048 --mem-floor-gb 2 > "$1" 2>&1 &
    fi
    SRV_PID=$!
    trap 'kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null' EXIT
    i=0
    while [ "$i" -lt 300 ]; do
        grep -q "listening on" "$1" && break
        kill -0 "$SRV_PID" 2>/dev/null || fail "server died during boot (see $1)"
        sleep 2
        i=$((i + 1))
    done
    grep -q "listening on" "$1" || fail "server did not start listening (see $1)"
}

stop_server() {
    kill "$SRV_PID" 2>/dev/null
    wait "$SRV_PID" 2>/dev/null
    trap - EXIT
}

chat_request() {   # $1 base url  $2 out  $3 body
    code=$(curl -sS -m 600 -o "$2" -w '%{http_code}' \
        -H 'Content-Type: application/json' -d "$3" "$1/v1/chat/completions") \
        || fail "POST $1/v1/chat/completions failed"
    [ "$code" = "200" ] || fail "POST $1/v1/chat/completions returned HTTP $code (see $2)"
}

check_completion() {   # $1 file  $2 label; sets TEXT, FIN, NTOK
    jq -e '.object == "chat.completion"' "$1" >/dev/null 2>&1 \
        || fail "$2: not a chat.completion object (see $1)"
    jq -e '(.choices | type) == "array" and (.choices | length) > 0' "$1" >/dev/null 2>&1 \
        || fail "$2: no choices array"
    jq -e '(.choices[0].message.content | type) == "string" and (.choices[0].message.content | length) > 0' \
        "$1" >/dev/null 2>&1 || fail "$2: empty or missing content (see $1)"
    jq -e '(.choices[0].finish_reason | type) == "string" and (.choices[0].finish_reason | length) > 0' \
        "$1" >/dev/null 2>&1 || fail "$2: missing or empty finish_reason (see $1)"
    jq -e '(.usage.completion_tokens | type) == "number" and .usage.completion_tokens > 0' \
        "$1" >/dev/null 2>&1 || fail "$2: usage.completion_tokens is not > 0 (see $1)"
    TEXT=$(jq -r '.choices[0].message.content' "$1")
    FIN=$(jq -r '.choices[0].finish_reason' "$1")
    NTOK=$(jq -r '.usage.completion_tokens' "$1")
}

# ---- arm 1: no MTP ----
start_server "$LOG.server.log" "$PORT" ""

grep -q "CUDA backend initialized" "$LOG.server.log" \
    || fail "no CUDA backend line at boot (see $LOG.server.log)"
DEVICE=$(sed -n 's/^ds4: CUDA backend initialized on \(.*\) (sm_[0-9][0-9]*)$/\1/p' \
    "$LOG.server.log" | head -1)
SM=$(sed -n 's/^ds4: CUDA backend initialized on .* (sm_\([0-9][0-9]*\))$/\1/p' \
    "$LOG.server.log" | head -1)
[ -n "$SM" ] || fail "the CUDA backend line carries no (sm_<n>) stamp (see $LOG.server.log)"
EXPECT_N=${EXPECT_SM#sm_}
[ "$SM" = "$EXPECT_N" ] || fail "the backend reports sm_$SM, expected $EXPECT_SM"

# ---- GET /v1/models ----
code=$(curl -sS -m 30 -o "$LOG.models.json" -w '%{http_code}' \
    "http://127.0.0.1:$PORT/v1/models") || fail "GET /v1/models failed"
[ "$code" = "200" ] || fail "GET /v1/models returned HTTP $code"
jq -e '.object == "list"' "$LOG.models.json" >/dev/null 2>&1 \
    || fail "GET /v1/models: no list object (see $LOG.models.json)"
jq -e '(.data | type) == "array" and (.data | length) > 0' "$LOG.models.json" >/dev/null 2>&1 \
    || fail "GET /v1/models: no data array"
ID=$(jq -r '.data[0].id // ""' "$LOG.models.json")
[ -n "$ID" ] && [ "$ID" != "null" ] || fail "GET /v1/models: no model id"

BODY=$(jq -c -n --arg m "$ID" --arg p "$PROMPT" --argjson n "$CHAT_MAX" \
    '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0,reasoning_effort:"none"}') \
    || fail "could not build the request body"

# ---- non-stream completion, twice, temperature 0 ----
chat_request "http://127.0.0.1:$PORT" "$LOG.chat1.json" "$BODY"
check_completion "$LOG.chat1.json" "completion 1"
TEXT1=$TEXT FIN1=$FIN N1=$NTOK

chat_request "http://127.0.0.1:$PORT" "$LOG.chat2.json" "$BODY"
check_completion "$LOG.chat2.json" "completion 2"

# ---- determinism: field by field, never as whole documents ----
[ "$TEXT1" = "$TEXT" ] || fail "two identical requests returned different content"
[ "$FIN1" = "$FIN" ] || fail "two identical requests returned different finish reasons"
[ "$N1" = "$NTOK" ] || fail "two identical requests returned different completion-token counts"

# ---- streaming ----
STREAM_BODY=$(printf '%s' "$BODY" | jq -c '. + {stream:true}') \
    || fail "could not build the stream body"
code=$(curl -sS -N -m 600 -o "$LOG.stream.sse" -w '%{http_code}' \
    -H 'Content-Type: application/json' -d "$STREAM_BODY" \
    "http://127.0.0.1:$PORT/v1/chat/completions") || fail "POST /v1/chat/completions (stream) failed"
[ "$code" = "200" ] || fail "stream: HTTP $code (see $LOG.stream.sse)"
grep -q '^event: error' "$LOG.stream.sse" && fail "stream: the server sent an error event (see $LOG.stream.sse)"
grep -q '^data: ' "$LOG.stream.sse" || fail "stream: no SSE data lines (see $LOG.stream.sse)"
grep -q '^data: \[DONE\]' "$LOG.stream.sse" || fail "stream: no [DONE] terminal"
awk 'substr($0, 1, 6) == "data: " { sub(/^data: /, ""); if ($0 != "[DONE]") print }' \
    "$LOG.stream.sse" > "$LOG.stream.data.jsonl"
[ -s "$LOG.stream.data.jsonl" ] || fail "stream: no data records besides [DONE]"
jq -s -e '[.[] | .choices[]? | (.delta?.content? // "") | select(length > 0)] | length > 0' \
    "$LOG.stream.data.jsonl" >/dev/null 2>&1 \
    || fail "stream: no delta with nonempty content (see $LOG.stream.data.jsonl)"
jq -s -e '[.[] | .choices[]? | .finish_reason? | select(. != null)] | length > 0' \
    "$LOG.stream.data.jsonl" >/dev/null 2>&1 \
    || fail "stream: no finish_reason chunk (see $LOG.stream.data.jsonl)"
LAST=$(awk '/^data: / { last = $0 } END { print last }' "$LOG.stream.sse")
[ "$LAST" = "data: [DONE]" ] || fail "stream: [DONE] is not the last data line (last: $LAST)"

# ---- negative control: the server must reject a body it cannot parse ----
NEG=$(jq -c -n --arg m "$ID" --arg p "$PROMPT" \
    '{model:$m,messages:[{role:"user",content:$p}],max_tokens:-1,temperature:0}') \
    || fail "could not build the negative-control body"
code=$(curl -sS -m 30 -o "$LOG.neg.json" -w '%{http_code}' \
    -H 'Content-Type: application/json' -d "$NEG" \
    "http://127.0.0.1:$PORT/v1/chat/completions") || fail "the negative control could not be sent"
[ "$code" != "200" ] || fail "negative control (max_tokens -1) was ACCEPTED: the gate would pass blind"
[ "$code" = "400" ] || fail "negative control: HTTP $code, expected 400 (see $LOG.neg.json)"
jq -e 'has("choices") | not' "$LOG.neg.json" >/dev/null 2>&1 \
    || fail "negative control: the rejection body carries choices (see $LOG.neg.json)"
jq -e '.error.type == "invalid_request_error"' "$LOG.neg.json" >/dev/null 2>&1 \
    || fail "negative control: error.type is not invalid_request_error (see $LOG.neg.json)"
jq -e '.error.message == "max_tokens must be >= 0"' "$LOG.neg.json" >/dev/null 2>&1 \
    || fail "negative control: error.message is not 'max_tokens must be >= 0' (see $LOG.neg.json)"

# ---- arm 2 (optional): the drafter must not change what is served ----
MTP_SUMMARY="skipped (no mtp gguf given)"
if [ -n "$MTP" ]; then
    stop_server
    start_server "$LOG.mtp.log" "$MTP_PORT" "$MTP"
    grep -q "CUDA backend initialized" "$LOG.mtp.log" \
        || fail "no CUDA backend line at the MTP boot (see $LOG.mtp.log)"
    chat_request "http://127.0.0.1:$MTP_PORT" "$LOG.mtp.chat.json" "$BODY"
    check_completion "$LOG.mtp.chat.json" "mtp completion"
    [ "$TEXT" = "$TEXT1" ] || fail "the MTP run's content differs from the no-MTP run"
    [ "$FIN" = "$FIN1" ] || fail "the MTP run's finish_reason differs from the no-MTP run"
    [ "$NTOK" = "$N1" ] || fail "the MTP run's completion-token count differs from the no-MTP run"
    stop_server
    MTP_SUMMARY="--mtp --mtp-draft 6: content, finish_reason and $N1 completion tokens equal the no-MTP run"
fi

stop_server

echo "gemma4 http gate: PASS (boot: listening + CUDA backend on $DEVICE sm_$SM; \
/v1/models: list object with id $ID; \
completion: non-stream 200, finish=$FIN1 after $N1 tokens; \
determinism: content, finish_reason and token count equal on the repeat; \
stream: SSE data lines, a content delta, a finish_reason chunk and [DONE] last; \
refusal: max_tokens -1 rejected 400 invalid_request_error 'max_tokens must be >= 0'; \
mtp: $MTP_SUMMARY)"
echo "  model:        $ID"
echo "  backend:      $(grep -m1 'CUDA backend initialized' "$LOG.server.log" | sed 's/^ds4: //')"
echo "  finish:       $FIN1 after $N1 tokens"
echo "  text:         $TEXT1"
