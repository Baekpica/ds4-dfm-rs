#!/bin/sh
# ds41 V4.1 server-route gate (unit F, P5.5): boot ds4-server on the
# artifact, drive the push route over HTTP, and prove the served ids equal
# the CLI's on the same prompt text — the HTTP plumbing must not change what
# the engine produces.  Plus the boot refusals (fast, before the model load).
#
#   cli-ref     the CLI's [emit] ids for the prompt text, captured FIRST:
#               the single-instance guard admits one model-loading process
#               at a time, so the CLI cannot run while the server holds the
#               lock (and two concurrent 105 GiB loads are forbidden anyway).
#               --dump-tokens prints the prompt ids the CLI then generates
#               from, so both sides run the identical prompt tokens.
#   boot        /v1/models answers and the log carries the V4.1 route line
#   completion  POST /v1/completions (non-stream): 200, nonempty text, a
#               finish_reason, and the log's [emit] ids equal the CLI's
#               (the requests are serialized, so the log's [emit] block at
#               this point is exactly this request's)
#   stream      POST /v1/completions (stream): SSE data deltas + [DONE]
#   refusals    --ctx on a V4.1 model and --zchain are refused by name
#
# usage: ds41_serving_http_gate.sh <server> <cli> <gguf> <engram_dir> <port> <prompt> <n>
set -u

SRV="$1" CLI="$2" MODEL="$3" ENGRAM="$4" PORT="$5" PROMPT="$6" N="${7:-16}"
LOG=/tmp/ds41_http_gate

fail() { echo "ds41 http gate: FAIL ($1)"; exit 1; }

# ---- CLI reference (before the server: one model-loading process at a time) ----
"$CLI" -m "$MODEL" --dump-tokens -p "$PROMPT" > "$LOG.tokens.txt" 2>&1 \
  || fail "CLI --dump-tokens failed (see $LOG.tokens.txt)"
IDS=$(head -1 "$LOG.tokens.txt" | tr -d '[],')
[ -n "$IDS" ] || fail "CLI --dump-tokens printed no id line"
"$CLI" -m "$MODEL" --engram-dir "$ENGRAM" --gen-ids /dev/stdin -n "$N" --temp 0 --no-dspark --emit-trace <<EOF > "$LOG.cli.log" 2>&1
$IDS
EOF
[ $? -eq 0 ] || fail "CLI generate failed (see $LOG.cli.log)"
grep -o '\[emit\] [0-9]* [0-9]*' "$LOG.cli.log" | sed 's/\[emit\] //' > "$LOG.cli.emit"
[ -s "$LOG.cli.emit" ] || fail "CLI run printed no [emit] lines"

# ---- boot (the model load takes ~2.5 min; poll the listener line) ----
"$SRV" -m "$MODEL" --port "$PORT" --engram-dir "$ENGRAM" --emit-trace --no-dspark > "$LOG.server.log" 2>&1 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null' EXIT

i=0
while [ $i -lt 300 ]; do
    grep -q "listening on" "$LOG.server.log" && break
    kill -0 "$SRV_PID" 2>/dev/null || fail "server died during boot (see $LOG.server.log)"
    sleep 2; i=$((i + 1))
done
grep -q "listening on" "$LOG.server.log" || fail "server did not start listening"
grep -q "V4.1 serving route" "$LOG.server.log" || fail "no V4.1 route line at boot (ctx from metadata?)"

# ---- /v1/models ----
curl -sS -m 30 "http://127.0.0.1:$PORT/v1/models" > "$LOG.models.json" 2>&1 \
  || fail "GET /v1/models failed"
grep -q '"object":"list"' "$LOG.models.json" || fail "GET /v1/models: no list object"

# ---- non-streaming completion: the parity run (one worker serializes the
# requests, so every [emit] line in the log at this point belongs to it) ----
curl -sS -m 600 "http://127.0.0.1:$PORT/v1/completions" \
    -H 'content-type: application/json' \
    -d "{\"prompt\":\"$PROMPT\",\"max_tokens\":$N,\"temperature\":0}" > "$LOG.completion.json" 2>&1 \
  || fail "POST /v1/completions failed"
grep -q '"finish_reason"' "$LOG.completion.json" || fail "completion: no finish_reason (see $LOG.completion.json)"
grep -q '"text":""' "$LOG.completion.json" && fail "completion: empty text (see $LOG.completion.json)"

# ---- parity: the server's [emit] ids == the CLI's on the same prompt text ----
grep -o '\[emit\] [0-9]* [0-9]*' "$LOG.server.log" | sed 's/\[emit\] //' > "$LOG.srv.emit"
[ -s "$LOG.srv.emit" ] || fail "server run printed no [emit] lines (--emit-trace wired?)"
diff "$LOG.cli.emit" "$LOG.srv.emit" > "$LOG.emit.diff" 2>&1 \
  || fail "the server's [emit] ids differ from the CLI's (see $LOG.emit.diff)"

# ---- streaming completion ----
curl -sS -m 600 -N "http://127.0.0.1:$PORT/v1/completions" \
    -H 'content-type: application/json' \
    -d "{\"prompt\":\"$PROMPT\",\"max_tokens\":$N,\"temperature\":0,\"stream\":true}" > "$LOG.stream.sse" 2>&1 \
  || fail "POST /v1/completions (stream) failed"
grep -q '^data: ' "$LOG.stream.sse" || fail "stream: no SSE data lines (see $LOG.stream.sse)"
grep -q '\[DONE\]' "$LOG.stream.sse" || fail "stream: no [DONE] terminal"

kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null
trap - EXIT

# ---- boot refusals (before any model load) ----
"$SRV" -m "$MODEL" --ctx 4096 --port "$PORT" > "$LOG.ctx.log" 2>&1
rc=$?
[ "$rc" -ne 0 ] || fail "negative control (--ctx on V4.1) PASSED (blind gate)"
grep -q -- "--ctx is refused" "$LOG.ctx.log" || fail "--ctx: not refused by name"

"$SRV" -m "$MODEL" --zchain /tmp > "$LOG.zchain.log" 2>&1
rc=$?
[ "$rc" -ne 0 ] || fail "negative control (--zchain) PASSED (blind gate)"
grep -q -- "--zchain is not supported" "$LOG.zchain.log" || fail "--zchain: not refused by name"

echo "ds41 http gate: PASS (CLI reference, boot + /v1/models, non-stream and stream completions, the served ids equal the CLI's on the same prompt, --ctx and --zchain refused by name)"
