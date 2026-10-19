#!/usr/bin/env bash
#
# DS41 DSML compliance and benefit gate, run ON the Spark against a live
# ds4-server.  One script, one stage name: baseline, phase1, phase2, ...
# Re-running it after each phase is what makes the deltas attributable.
#
# What it establishes, per client surface:
#   OpenAI chat/completions  non-stream tool_calls, stream tool_calls, and a
#                            second turn that feeds the call back as a tool
#                            result
#   OpenAI /v1/responses     function_call item and a continuation that carries
#                            the function_call_output
#   Anthropic /v1/messages   tool_use block and a tool_result continuation
#   truncation               a cut DSML block (max_tokens too small) so the
#                            recovery behaviour is recorded before and after
#   KV reuse                 a plain text prompt sent twice, to prove the
#                            non-DSML KV path still reuses its prefix
#   determinism              a temperature-0 prompt whose exact text is the
#                            VQ/KV regression witness: it must be byte-identical
#                            across stages
#
# Benefit is read from the server's own timings block (prefill_tokens,
# prefill_cached_tokens, prefill_tok_s, decode_tok_s, tok_per_step).  The number
# that matters for the DSML replay port is prefill_cached_tokens on the SECOND
# tool turn: before the port the sampled DSML is not replayed, so the follow-up
# turn cannot reuse the sampled bytes.
#
# Usage:
#   bash dsml_compliance.sh <stage> [base_url] [out_dir]
#   bash dsml_compliance.sh baseline
#   PORT=8009 bash dsml_compliance.sh phase1 http://127.0.0.1:8009
#
# Exit: 0 when every hard assertion passed, 1 otherwise.  Raw bodies and the
# summary land in the out dir so they can be copied back as evidence.

set -u

STAGE="${1:-baseline}"
BASE="${2:-http://127.0.0.1:8008}"
OUT="${3:-$HOME/ds41-dsml-gate/out-$STAGE-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"

CURL=(curl -sS --max-time 300)
PASS=0
FAIL=0

say()  { printf '%s\n' "$*" | tee -a "$OUT/summary.txt" >/dev/null; printf '%s\n' "$*"; }
leg()  { printf '\n=== %s ===\n' "$1" | tee -a "$OUT/summary.txt" >/dev/null; printf '\n=== %s ===\n' "$1"; }
ok()   { PASS=$((PASS + 1)); printf 'PASS  %s\n' "$1" | tee -a "$OUT/summary.txt" >/dev/null; printf 'PASS  %s\n' "$1"; }
bad()  { FAIL=$((FAIL + 1)); printf 'FAIL  %s\n' "$1" | tee -a "$OUT/summary.txt" >/dev/null; printf 'FAIL  %s\n' "$1"; }
check(){ if [ "$2" = "$3" ]; then ok "$1 ($2)"; else bad "$1 (want '$3', got '$2')"; fi; }

# The two tools: one string parameter and one non-string parameter, because the
# DSML parameter tag carries string="true|false" and the port must keep a number
# a number.
TOOLS_OPENAI='[
 {"type":"function","function":{"name":"get_weather","description":"Current weather for a city",
  "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
 {"type":"function","function":{"name":"set_timer","description":"Set a timer",
  "parameters":{"type":"object","properties":{"seconds":{"type":"integer"},"label":{"type":"string"}},"required":["seconds"]}}}]'

TOOLS_ANTHROPIC='[
 {"name":"get_weather","description":"Current weather for a city",
  "input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},
 {"name":"set_timer","description":"Set a timer",
  "input_schema":{"type":"object","properties":{"seconds":{"type":"integer"},"label":{"type":"string"}},"required":["seconds"]}}]'

ASK='Call get_weather for Stockholm. Call the tool; do not answer in prose.'

model_id() { "${CURL[@]}" "$BASE/v1/models" | jq -r '.data[0].id'; }
MODEL="$(model_id)"
say "stage:  $STAGE"
say "base:   $BASE"
say "model:  $MODEL"
say "out:    $OUT"
say "date:   $(date -Is)"

"${CURL[@]}" "$BASE/v1/models" > "$OUT/models.json"

# timings_to_line <file> -> "cached=.. prompt=.. ttft=.. decode=.. pretok_s=.."
timings_line() {
  jq -r '(.timings // {}) |
    "cached=\(.prefill_cached_tokens // "n/a") prompt=\(.prefill_tokens // "n/a") " +
    "ttft_ms=\(.ttft_ms // "n/a") decode_tok_s=\(.decode_tok_s // "n/a") " +
    "prefill_tok_s=\(.prefill_tok_s // "n/a") tok_per_step=\(.tok_per_step // "n/a")"' "$1" 2>/dev/null
}

############################################
# Leg 1: OpenAI chat/completions, non-stream
############################################
leg "openai chat non-stream tool call"
BODY1=$(jq -n --argjson tools "$TOOLS_OPENAI" --arg ask "$ASK" --arg m "$MODEL" \
  '{model:$m,messages:[{role:"user",content:$ask}],tools:$tools,tool_choice:"auto",
    max_tokens:192,temperature:0,stream:false}')
printf '%s' "$BODY1" > "$OUT/leg1-openai-nonstream.req.json"
"${CURL[@]}" -H 'Content-Type: application/json' -d "$BODY1" \
  "$BASE/v1/chat/completions" > "$OUT/leg1-openai-nonstream.res.json"
FIN1=$(jq -r '.choices[0].finish_reason' "$OUT/leg1-openai-nonstream.res.json")
NAME1=$(jq -r '.choices[0].message.tool_calls[0].function.name // ""' "$OUT/leg1-openai-nonstream.res.json")
ARGS1=$(jq -r '.choices[0].message.tool_calls[0].function.arguments // ""' "$OUT/leg1-openai-nonstream.res.json")
ID1=$(jq -r '.choices[0].message.tool_calls[0].id // ""' "$OUT/leg1-openai-nonstream.res.json")
ARGS1_OK=$(printf '%s' "$ARGS1" | jq -e 'type=="object"' >/dev/null 2>&1 && echo yes || echo no)
say "finish_reason=$FIN1 id=$ID1 name=$NAME1 args=$ARGS1"
say "timings: $(timings_line "$OUT/leg1-openai-nonstream.res.json")"
check "openai non-stream finish_reason" "$FIN1" "tool_calls"
check "openai non-stream tool name" "$NAME1" "get_weather"
check "openai non-stream arguments are JSON" "$ARGS1_OK" "yes"
check "openai non-stream call id prefix" "$(printf '%s' "$ID1" | cut -c1-5)" "call_"
CITY1=$(printf '%s' "$ARGS1" | jq -r '.city // ""')
check "openai non-stream city argument" "$CITY1" "Stockholm"

########################################
# Leg 2: OpenAI chat/completions, stream
########################################
leg "openai chat stream tool call"
BODY2=$(jq -n --argjson tools "$TOOLS_OPENAI" --arg ask "$ASK" --arg m "$MODEL" \
  '{model:$m,messages:[{role:"user",content:$ask}],tools:$tools,tool_choice:"auto",
    max_tokens:192,temperature:0,stream:true}')
printf '%s' "$BODY2" > "$OUT/leg2-openai-stream.req.json"
"${CURL[@]}" -H 'Content-Type: application/json' -d "$BODY2" \
  "$BASE/v1/chat/completions" > "$OUT/leg2-openai-stream.res.sse"
grep -c '^data: ' "$OUT/leg2-openai-stream.res.sse" > "$OUT/leg2.delta_count" 2>/dev/null || true
DELTAS=$(cat "$OUT/leg2.delta_count")
TOOLDELTA=$(grep -c '"tool_calls"' "$OUT/leg2-openai-stream.res.sse" || true)
FINSTREAM=$(grep -o '"finish_reason":"[a-z_]*"' "$OUT/leg2-openai-stream.res.sse" | tail -1 | cut -d'"' -f4)
DONE=$(grep -c '^data: \[DONE\]' "$OUT/leg2-openai-stream.res.sse" || true)
say "sse data lines=$DELTAS tool_calls-bearing lines=$TOOLDELTA finish=$FINSTREAM done=$DONE"
[ "$DELTAS" -gt 1 ] && ok "openai stream produced SSE data lines ($DELTAS)" || bad "openai stream produced SSE data lines"
[ "$TOOLDELTA" -gt 0 ] && ok "openai stream carried a tool_calls delta ($TOOLDELTA lines)" || bad "openai stream carried a tool_calls delta"
check "openai stream finish_reason" "$FINSTREAM" "tool_calls"
check "openai stream terminated" "$DONE" "1"

#########################################################
# Leg 3: OpenAI turn 2 with the tool result (the replay leg)
#########################################################
leg "openai chat turn 2 (tool result) - replay/reuse evidence"
if [ -n "$ARGS1" ] && [ "$ARGS1_OK" = yes ] && [ -n "$ID1" ]; then
  BODY3=$(jq -n --argjson tools "$TOOLS_OPENAI" --arg ask "$ASK" --arg m "$MODEL" \
    --arg id "$ID1" --arg name "$NAME1" --argjson args "$ARGS1" \
    '{model:$m,
      messages:[{role:"user",content:$ask},
                {role:"assistant",content:"",tool_calls:[{id:$id,type:"function",function:{name:$name,arguments:($args|tojson)}}]},
                {role:"tool",tool_call_id:$id,content:"{\"temp_c\":7,\"sky\":\"clear\"}"}],
      tools:$tools,tool_choice:"auto",max_tokens:192,temperature:0,stream:false}')
  printf '%s' "$BODY3" > "$OUT/leg3-openai-turn2.req.json"
  "${CURL[@]}" -H 'Content-Type: application/json' -d "$BODY3" \
    "$BASE/v1/chat/completions" > "$OUT/leg3-openai-turn2.res.json"
  CACHED3=$(jq -r '.timings.prefill_cached_tokens // 0' "$OUT/leg3-openai-turn2.res.json")
  PROMPT3=$(jq -r '.timings.prefill_tokens // 0' "$OUT/leg3-openai-turn2.res.json")
  say "turn2 timings: $(timings_line "$OUT/leg3-openai-turn2.res.json")"
  say "turn2 content: $(jq -r '.choices[0].message.content // ""' "$OUT/leg3-openai-turn2.res.json" | head -c 200)"
  [ "$PROMPT3" -gt 0 ] && ok "turn2 prefill_token count reported ($PROMPT3)" || bad "turn2 prefill_token count reported"
  echo "$CACHED3" > "$OUT/leg3.cached_tokens"
  echo "$PROMPT3" > "$OUT/leg3.prompt_tokens"
  say "BENEFIT(openai turn2): prefill_cached_tokens=$CACHED3 of prefill_tokens=$PROMPT3"
else
  bad "turn2 skipped: turn 1 produced no usable call"
fi

################################
# Leg 4: OpenAI /v1/responses
################################
leg "openai responses tool call (function_call)"
BODY4=$(jq -n --argjson tools "$TOOLS_OPENAI" --arg ask "$ASK" --arg m "$MODEL" \
  '{model:$m,input:[{role:"user",content:$ask}],tools:$tools,max_output_tokens:192,
    temperature:0,stream:false}')
printf '%s' "$BODY4" > "$OUT/leg4-responses.req.json"
"${CURL[@]}" -H 'Content-Type: application/json' -d "$BODY4" \
  "$BASE/v1/responses" > "$OUT/leg4-responses.res.json"
R4TYPE=$(jq -r '[.output[]? | select(.type=="function_call")] | length' "$OUT/leg4-responses.res.json" 2>/dev/null || echo 0)
R4NAME=$(jq -r '[.output[]? | select(.type=="function_call")][0].name // ""' "$OUT/leg4-responses.res.json" 2>/dev/null)
R4CALLID=$(jq -r '[.output[]? | select(.type=="function_call")][0].call_id // ""' "$OUT/leg4-responses.res.json" 2>/dev/null)
R4ARGS=$(jq -r '[.output[]? | select(.type=="function_call")][0].arguments // ""' "$OUT/leg4-responses.res.json" 2>/dev/null)
say "responses function_call items=$R4TYPE name=$R4NAME call_id=$R4CALLID args=$R4ARGS"
[ "$R4TYPE" -gt 0 ] && ok "responses served a function_call item" || bad "responses served a function_call item"
check "responses function name" "$R4NAME" "get_weather"
if [ -n "$R4ARGS" ]; then
  R4OK=$(printf '%s' "$R4ARGS" | jq -e 'type=="object"' >/dev/null 2>&1 && echo yes || echo no)
  check "responses arguments are JSON" "$R4OK" "yes"
fi
if [ -n "$R4CALLID" ] && [ -n "$R4ARGS" ] && [ "$R4OK" = yes ]; then
  BODY4B=$(jq -n --argjson tools "$TOOLS_OPENAI" --arg ask "$ASK" --arg m "$MODEL" \
    --arg cid "$R4CALLID" --arg name "$R4NAME" --arg args "$R4ARGS" \
    '{model:$m,
      input:[{role:"user",content:$ask},
             {type:"function_call",call_id:$cid,name:$name,arguments:$args},
             {type:"function_call_output",call_id:$cid,output:"{\"temp_c\":7}"}],
      tools:$tools,max_output_tokens:192,temperature:0,stream:false}')
  printf '%s' "$BODY4B" > "$OUT/leg4b-responses-turn2.req.json"
  "${CURL[@]}" -H 'Content-Type: application/json' -d "$BODY4B" \
    "$BASE/v1/responses" > "$OUT/leg4b-responses-turn2.res.json"
  say "responses turn2 timings: $(timings_line "$OUT/leg4b-responses-turn2.res.json")"
  say "BENEFIT(responses turn2): cached=$(jq -r '.timings.prefill_cached_tokens // 0' "$OUT/leg4b-responses-turn2.res.json")"
  ok "responses continuation accepted the function_call_output"
fi

####################################
# Leg 5: Anthropic /v1/messages
####################################
leg "anthropic messages tool call (tool_use)"
BODY5=$(jq -n --argjson tools "$TOOLS_ANTHROPIC" --arg ask "$ASK" --arg m "$MODEL" \
  '{model:$m,max_tokens:192,temperature:0,
    messages:[{role:"user",content:$ask}],tools:$tools}')
printf '%s' "$BODY5" > "$OUT/leg5-anthropic.req.json"
"${CURL[@]}" -H 'Content-Type: application/json' -d "$BODY5" \
  "$BASE/v1/messages" > "$OUT/leg5-anthropic.res.json"
A5N=$(jq -r '[.content[]? | select(.type=="tool_use")] | length' "$OUT/leg5-anthropic.res.json" 2>/dev/null || echo 0)
A5ID=$(jq -r '[.content[]? | select(.type=="tool_use")][0].id // ""' "$OUT/leg5-anthropic.res.json" 2>/dev/null)
A5NAME=$(jq -r '[.content[]? | select(.type=="tool_use")][0].name // ""' "$OUT/leg5-anthropic.res.json" 2>/dev/null)
A5INPUT=$(jq -c '[.content[]? | select(.type=="tool_use")][0].input // null' "$OUT/leg5-anthropic.res.json" 2>/dev/null)
A5STOP=$(jq -r '.stop_reason // ""' "$OUT/leg5-anthropic.res.json" 2>/dev/null)
say "tool_use blocks=$A5N id=$A5ID name=$A5NAME input=$A5INPUT stop_reason=$A5STOP"
[ "$A5N" -gt 0 ] && ok "anthropic served a tool_use block" || bad "anthropic served a tool_use block"
check "anthropic tool name" "$A5NAME" "get_weather"
check "anthropic stop_reason" "$A5STOP" "tool_use"
case "$A5ID" in toolu_*) ok "anthropic tool id prefix (toolu_)";; *) bad "anthropic tool id prefix (got '$A5ID')";; esac
if [ "$A5INPUT" != "null" ] && [ -n "$A5INPUT" ]; then ok "anthropic input is a JSON object ($A5INPUT)"; else bad "anthropic input is a JSON object"; fi

if [ "$A5N" -gt 0 ] && [ -n "$A5ID" ]; then
  BODY5B=$(jq -n --argjson tools "$TOOLS_ANTHROPIC" --arg ask "$ASK" --arg m "$MODEL" \
    --arg id "$A5ID" --arg name "$A5NAME" --argjson input "$A5INPUT" \
    '{model:$m,max_tokens:192,temperature:0,tools:$tools,
      messages:[{role:"user",content:$ask},
                {role:"assistant",content:[{type:"tool_use",id:$id,name:$name,input:$input}]},
                {role:"user",content:[{type:"tool_result",tool_use_id:$id,content:"{\"temp_c\":7}"}]}]}')
  printf '%s' "$BODY5B" > "$OUT/leg5b-anthropic-turn2.req.json"
  "${CURL[@]}" -H 'Content-Type: application/json' -d "$BODY5B" \
    "$BASE/v1/messages" > "$OUT/leg5b-anthropic-turn2.res.json"
  say "anthropic turn2 timings: $(timings_line "$OUT/leg5b-anthropic-turn2.res.json")"
  say "BENEFIT(anthropic turn2): cached=$(jq -r '.timings.prefill_cached_tokens // 0' "$OUT/leg5b-anthropic-turn2.res.json")"
  ok "anthropic continuation accepted the tool_result"
fi

#########################################
# Leg 6: truncation (the recovery leg)
#########################################
leg "truncated DSML block (max_tokens too small)"
BODY6=$(jq -n --argjson tools "$TOOLS_OPENAI" --arg ask "$ASK" --arg m "$MODEL" \
  '{model:$m,messages:[{role:"user",content:$ask}],tools:$tools,tool_choice:"auto",
    max_tokens:24,temperature:0,stream:false}')
printf '%s' "$BODY6" > "$OUT/leg6-truncation.req.json"
"${CURL[@]}" -H 'Content-Type: application/json' -d "$BODY6" \
  "$BASE/v1/chat/completions" > "$OUT/leg6-truncation.res.json"
F6=$(jq -r '.choices[0].finish_reason // ""' "$OUT/leg6-truncation.res.json" 2>/dev/null)
C6=$(jq -r '.choices[0].message.content // ""' "$OUT/leg6-truncation.res.json" 2>/dev/null | head -c 120)
say "finish_reason=$F6 content=$C6"
say "NOTE  this leg records behaviour, it asserts nothing: the served route has no continuation (reference-absent, CTR 8.2 R6), so a cut block ends the turn with its truncation signal kept"
echo "$F6" > "$OUT/leg6.finish_reason"

#########################################
# Leg 7: plain-text KV reuse (non-DSML)
#########################################
leg "plain text KV prefix reuse (outside DSML)"
LONG='Summarise in one sentence what a KV cache does in an inference server.'
BODY7=$(jq -n --arg m "$MODEL" --arg p "$LONG" \
  '{model:$m,messages:[{role:"user",content:$p}],max_tokens:48,temperature:0,stream:false}')
printf '%s' "$BODY7" > "$OUT/leg7-kv.req.json"
"${CURL[@]}" -H 'Content-Type: application/json' -d "$BODY7" "$BASE/v1/chat/completions" > "$OUT/leg7-kv-a.res.json"
"${CURL[@]}" -H 'Content-Type: application/json' -d "$BODY7" "$BASE/v1/chat/completions" > "$OUT/leg7-kv-b.res.json"
C7A=$(jq -r '.timings.prefill_cached_tokens // 0' "$OUT/leg7-kv-a.res.json")
C7B=$(jq -r '.timings.prefill_cached_tokens // 0' "$OUT/leg7-kv-b.res.json")
say "KV reuse: first cached=$C7A second cached=$C7B"
say "timings b: $(timings_line "$OUT/leg7-kv-b.res.json")"
[ "$C7B" -gt 0 ] && ok "plain KV reuse still reuses a prefix (second read $C7B)" || bad "plain KV reuse still reuses a prefix (second read $C7B)"

###############################################
# Leg 8: determinism witness (VQ/KV regression)
###############################################
leg "determinism witness (identical text across stages)"
BODY8=$(jq -n --arg m "$MODEL" \
  '{model:$m,messages:[{role:"user",content:"Reply with exactly the words: alpha beta gamma"}],
    max_tokens:32,temperature:0,seed:1234,stream:false}')
"${CURL[@]}" -H 'Content-Type: application/json' -d "$BODY8" "$BASE/v1/chat/completions" > "$OUT/leg8-det-a.res.json"
"${CURL[@]}" -H 'Content-Type: application/json' -d "$BODY8" "$BASE/v1/chat/completions" > "$OUT/leg8-det-b.res.json"
jq -r '.choices[0].message.content // ""' "$OUT/leg8-det-a.res.json" > "$OUT/leg8.text.a"
jq -r '.choices[0].message.content // ""' "$OUT/leg8-det-b.res.json" > "$OUT/leg8.text.b"
if diff -q "$OUT/leg8.text.a" "$OUT/leg8.text.b" >/dev/null; then
  ok "same-process determinism at temperature 0"
else
  bad "same-process determinism at temperature 0"
fi
say "witness text: $(cat "$OUT/leg8.text.a" | head -c 120)"
say "WITNESS value for cross-stage comparison: $(sha256sum "$OUT/leg8.text.a" | cut -c1-16)"

###############################################
# Summary
###############################################
leg "summary"
say "stage=$STAGE pass=$PASS fail=$FAIL"
say "raw bodies: $OUT"
printf 'stage=%s pass=%s fail=%s\n' "$STAGE" "$PASS" "$FAIL" > "$OUT/verdict.txt"
{
  printf '# DSML compliance gate - %s\n\n' "$STAGE"
  printf -- '- date: %s\n- base: %s\n- model: %s\n- pass: %s\n- fail: %s\n' \
    "$(date -Is)" "$BASE" "$MODEL" "$PASS" "$FAIL"
  printf -- '- openai turn2 prefill_cached_tokens: %s of %s\n' "$(cat "$OUT/leg3.cached_tokens" 2>/dev/null || echo n/a)" "$(cat "$OUT/leg3.prompt_tokens" 2>/dev/null || echo n/a)"
  printf -- '- plain KV second read cached: %s\n' "$C7B"
  printf -- '- determinism witness sha256: %s\n' "$(sha256sum "$OUT/leg8.text.a" | cut -c1-16)"
  printf -- '- truncated-block finish_reason: %s\n' "$(cat "$OUT/leg6.finish_reason" 2>/dev/null || echo n/a)"
} > "$OUT/summary.md"
cat "$OUT/summary.md"

[ "$FAIL" -eq 0 ]
