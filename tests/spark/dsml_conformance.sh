#!/usr/bin/env bash
#
# DS41 DSML conformance suite (normative).  Probes are grouped so that each
# measurement is attributable:
#
#   A-xx  served-shape probes: one request each, asserts the client-visible
#         shape (tool_calls, tool_use, function_call, typing, multi-call).
#   B-xx  replay pairs: turn 1 immediately followed by turn 2 on the same
#         server slot, which is the only arrangement in which a cached-token
#         count measures THIS turn's replay.  A leg separated from its turn 1
#         by another request reads cached=0 for reasons that have nothing to do
#         with replay.
#   C-xx  recovery probes.
#   D-xx  regression probes outside DSML (plain KV reuse, determinism witness).
#   E-xx  input validation: the oracle's guard over textual fields.
#
# Every probe writes a row to results.tsv:
#   id <TAB> title <TAB> expected <TAB> observed <TAB> verdict <TAB> artifact
#
# Usage: bash dsml_conformance.sh <base_url> <out_dir>

set -u

BASE="${1:-http://192.168.1.91:8008}"
OUT="${2:-$PWD/ds41-dsml-conformance-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
TSV="$OUT/results.tsv"
: > "$TSV"

CURL=(curl -sS --max-time 300)
row() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" >> "$TSV"; printf '%-6s %-5s %s\n' "$1" "$5" "$2"; }
hdr() { printf '\n--- %s ---\n' "$1"; }
tokens() { jq -r '.timings.prefill_tokens // 0' "$1" 2>/dev/null; }
cached() { jq -r '.timings.prefill_cached_tokens // 0' "$1" 2>/dev/null; }

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

ASK_W='Call get_weather for Stockholm. Call the tool; do not answer in prose.'
ASK_T='Call set_timer with seconds 90 and label tea. Call the tool; do not answer in prose.'
ASK_BOTH='Call get_weather for Stockholm and set_timer with seconds 90. Call both tools.'

MODEL=$("${CURL[@]}" "$BASE/v1/models" | jq -r '.data[0].id')
RUNID="$(date +%s)-$RANDOM"
printf 'base=%s\nmodel=%s\ndate=%s\nrun=%s\n' "$BASE" "$MODEL" "$(date -Is)" "$RUNID" | tee "$OUT/header.txt"
ask() { printf '%s [ref %s %s]' "$2" "$1" "$RUNID"; }

chat() { printf '%s' "$2" > "$OUT/$1.req.json"; "${CURL[@]}" -H 'Content-Type: application/json' -d "$2" "$BASE/v1/chat/completions" > "$OUT/$1.res.json"; }
anth() { printf '%s' "$2" > "$OUT/$1.req.json"; "${CURL[@]}" -H 'Content-Type: application/json' -d "$2" "$BASE/v1/messages" > "$OUT/$1.res.json"; }
resp() { printf '%s' "$2" > "$OUT/$1.req.json"; "${CURL[@]}" -H 'Content-Type: application/json' -d "$2" "$BASE/v1/responses" > "$OUT/$1.res.json"; }

##############################################################################
# A. Served shape
##############################################################################

hdr "A-01 openai chat non-stream tool call"
A1="$(ask a01 "$ASK_W")"
B=$(jq -n --argjson t "$TOOLS_OPENAI" --arg a "$A1" --arg m "$MODEL" \
  '{model:$m,messages:[{role:"user",content:$a}],tools:$t,tool_choice:"auto",max_tokens:192,temperature:0,stream:false}')
chat a01-openai "$B"
FIN=$(jq -r '.choices[0].finish_reason // ""' "$OUT/a01-openai.res.json")
NAME=$(jq -r '.choices[0].message.tool_calls[0].function.name // ""' "$OUT/a01-openai.res.json")
ID=$(jq -r '.choices[0].message.tool_calls[0].id // ""' "$OUT/a01-openai.res.json")
ARGS=$(jq -r '.choices[0].message.tool_calls[0].function.arguments // ""' "$OUT/a01-openai.res.json")
OBS="finish=$FIN name=$NAME id=${ID:0:5}... args=$ARGS"
if [ "$FIN" = tool_calls ] && [ "$NAME" = get_weather ] && printf '%s' "$ARGS" | jq -e '.city=="Stockholm"' >/dev/null 2>&1; then V=PASS; else V=FAIL; fi
row A-01 "OpenAI chat served one DSML tool call as tool_calls" "finish=tool_calls, get_weather {city:Stockholm}" "$OBS" "$V" "$OUT/a01-openai.res.json"

hdr "A-02 non-string DSML parameter typing"
A2="$(ask a02 "$ASK_T")"
B=$(jq -n --argjson t "$TOOLS_OPENAI" --arg a "$A2" --arg m "$MODEL" \
  '{model:$m,messages:[{role:"user",content:$a}],tools:$t,tool_choice:"auto",max_tokens:192,temperature:0,stream:false}')
chat a02-typing "$B"
TARGS=$(jq -r '.choices[0].message.tool_calls[0].function.arguments // ""' "$OUT/a02-typing.res.json")
SEC=$(printf '%s' "$TARGS" | jq -r '.seconds|type' 2>/dev/null || echo unparsed)
OBS="arguments=$TARGS seconds-type=$SEC"
[ "$SEC" = number ] && V=PASS || V=FAIL
row A-02 "A numeric DSML parameter stays a JSON number" "seconds is a number" "$OBS" "$V" "$OUT/a02-typing.res.json"

hdr "A-03 multiple invokes in one block"
A3="$(ask a03 "$ASK_BOTH")"
B=$(jq -n --argjson t "$TOOLS_OPENAI" --arg a "$A3" --arg m "$MODEL" \
  '{model:$m,messages:[{role:"user",content:$a}],tools:$t,tool_choice:"auto",max_tokens:256,temperature:0,stream:false}')
chat a03-multicall "$B"
NC=$(jq -r '[.choices[0].message.tool_calls[]?]|length' "$OUT/a03-multicall.res.json")
OBS="calls=$NC names=$(jq -rc '[.choices[0].message.tool_calls[]?.function.name]' "$OUT/a03-multicall.res.json")"
[ "$NC" -ge 2 ] && V=PASS || V=FAIL
row A-03 "Two invokes in one DSML block become two calls" "2 or more tool_calls" "$OBS" "$V" "$OUT/a03-multicall.res.json"

hdr "A-04 openai chat stream tool call"
A4="$(ask a04 "$ASK_W")"
B=$(jq -n --argjson t "$TOOLS_OPENAI" --arg a "$A4" --arg m "$MODEL" \
  '{model:$m,messages:[{role:"user",content:$a}],tools:$t,tool_choice:"auto",max_tokens:192,temperature:0,stream:true}')
printf '%s' "$B" > "$OUT/a04.req.json"
"${CURL[@]}" -H 'Content-Type: application/json' -d "$B" "$BASE/v1/chat/completions" > "$OUT/a04-openai-stream.sse"
TD=$(grep -c '"tool_calls"' "$OUT/a04-openai-stream.sse" || true)
FS=$(grep -o '"finish_reason":"[a-z_]*"' "$OUT/a04-openai-stream.sse" | tail -1 | cut -d'"' -f4)
OBS="tool_calls lines=$TD finish=$FS"
if [ "$TD" -gt 0 ] && [ "$FS" = tool_calls ]; then V=PASS; else V=FAIL; fi
row A-04 "The streamed turn carries the tool call" "tool_calls delta, finish_reason=tool_calls" "$OBS" "$V" "$OUT/a04-openai-stream.sse"

hdr "A-05 anthropic tool_use"
A5="$(ask a05 "$ASK_W")"
B=$(jq -n --argjson t "$TOOLS_ANTHROPIC" --arg a "$A5" --arg m "$MODEL" \
  '{model:$m,max_tokens:192,temperature:0,messages:[{role:"user",content:$a}],tools:$t}')
anth a05-anthropic "$B"
AID=$(jq -r '[.content[]?|select(.type=="tool_use")][0].id // ""' "$OUT/a05-anthropic.res.json")
ANAME=$(jq -r '[.content[]?|select(.type=="tool_use")][0].name // ""' "$OUT/a05-anthropic.res.json")
AINPUT=$(jq -c '[.content[]?|select(.type=="tool_use")][0].input // null' "$OUT/a05-anthropic.res.json")
ASTOP=$(jq -r '.stop_reason // ""' "$OUT/a05-anthropic.res.json")
case "$AID" in toolu_*) IDOK=1;; *) IDOK=0;; esac
OBS="id=${AID:0:6}... name=$ANAME input=$AINPUT stop=$ASTOP"
if [ "$ANAME" = get_weather ] && [ "$ASTOP" = tool_use ] && [ "$IDOK" = 1 ]; then V=PASS; else V=FAIL; fi
row A-05 "Anthropic served a tool_use block" "toolu_ id, name, input object, stop_reason=tool_use" "$OBS" "$V" "$OUT/a05-anthropic.res.json"

hdr "A-06 responses function_call"
A6="$(ask a06 "$ASK_W")"
B=$(jq -n --argjson t "$TOOLS_OPENAI" --arg a "$A6" --arg m "$MODEL" \
  '{model:$m,input:[{role:"user",content:$a}],tools:$t,max_output_tokens:192,temperature:0,stream:false}')
resp a06-responses "$B"
RN=$(jq -r '[.output[]?|select(.type=="function_call")]|length' "$OUT/a06-responses.res.json" 2>/dev/null || echo 0)
RNAME=$(jq -r '[.output[]?|select(.type=="function_call")][0].name // ""' "$OUT/a06-responses.res.json")
RCID=$(jq -r '[.output[]?|select(.type=="function_call")][0].call_id // ""' "$OUT/a06-responses.res.json")
RARGS=$(jq -r '[.output[]?|select(.type=="function_call")][0].arguments // ""' "$OUT/a06-responses.res.json")
OBS="items=$RN name=$RNAME call_id=${RCID:0:5}... args=$RARGS"
if [ "$RN" -ge 1 ] && [ "$RNAME" = get_weather ]; then V=PASS; else V=FAIL; fi
row A-06 "Responses served a function_call item" "one function_call item with name and arguments" "$OBS" "$V" "$OUT/a06-responses.res.json"

##############################################################################
# B. Replay pairs: turn 1 then turn 2, back to back on the same slot
##############################################################################

hdr "B-01 chat replay pair (exact echo)"
B1="$(ask b01 "$ASK_W")"
B=$(jq -n --argjson t "$TOOLS_OPENAI" --arg a "$B1" --arg m "$MODEL" \
  '{model:$m,messages:[{role:"user",content:$a}],tools:$t,tool_choice:"auto",max_tokens:192,temperature:0,stream:false}')
chat b01-chat-t1 "$B"
T1FIN=$(jq -r '.choices[0].finish_reason // ""' "$OUT/b01-chat-t1.res.json")
T1P=$(tokens "$OUT/b01-chat-t1.res.json")
B1ID=$(jq -r '.choices[0].message.tool_calls[0].id // ""' "$OUT/b01-chat-t1.res.json")
B1NAME=$(jq -r '.choices[0].message.tool_calls[0].function.name // ""' "$OUT/b01-chat-t1.res.json")
B1ARGS=$(jq -r '.choices[0].message.tool_calls[0].function.arguments // ""' "$OUT/b01-chat-t1.res.json")
if [ -n "$B1ID" ] && [ -n "$B1ARGS" ]; then
  B=$(jq -n --argjson t "$TOOLS_OPENAI" --arg a "$B1" --arg m "$MODEL" \
    --arg id "$B1ID" --arg name "$B1NAME" --argjson args "$B1ARGS" \
    '{model:$m,messages:[{role:"user",content:$a},
       {role:"assistant",content:"",tool_calls:[{id:$id,type:"function",function:{name:$name,arguments:($args|tojson)}}]},
       {role:"tool",tool_call_id:$id,content:"{\"temp_c\":7}"}],
      tools:$t,tool_choice:"auto",max_tokens:192,temperature:0,stream:false}')
  chat b01-chat-t2 "$B"
  C=$(cached "$OUT/b01-chat-t2.res.json"); P=$(tokens "$OUT/b01-chat-t2.res.json"); G=$((C - T1P))
  OBS="turn-1 prompt=$T1P finish=$T1FIN; turn-2 prompt=$P cached=$C; replay gain=$G tokens"
  [ "$G" -gt 0 ] && V=PASS || V=FAIL
  row B-01 "Chat turn 2 reuses the sampled assistant DSML block" "cached exceeds turn-1 prompt (the block replayed)" "$OBS" "$V" "$OUT/b01-chat-t2.res.json"
else
  row B-01 "Chat turn 2 reuses the sampled assistant DSML block" "cached exceeds turn-1 prompt" "turn 1 produced no call" FAIL "$OUT/b01-chat-t1.res.json"
fi

hdr "B-02 chat replay pair (echo spelling differs)"
B2="$(ask b02 "$ASK_W")"
B=$(jq -n --argjson t "$TOOLS_OPENAI" --arg a "$B2" --arg m "$MODEL" \
  '{model:$m,messages:[{role:"user",content:$a}],tools:$t,tool_choice:"auto",max_tokens:192,temperature:0,stream:false}')
chat b02-chat-t1 "$B"
B2P=$(tokens "$OUT/b02-chat-t1.res.json")
B2ID=$(jq -r '.choices[0].message.tool_calls[0].id // ""' "$OUT/b02-chat-t1.res.json")
B2NAME=$(jq -r '.choices[0].message.tool_calls[0].function.name // ""' "$OUT/b02-chat-t1.res.json")
B2ARGS=$(jq -r '.choices[0].message.tool_calls[0].function.arguments // ""' "$OUT/b02-chat-t1.res.json")
if [ -n "$B2ID" ] && [ -n "$B2ARGS" ]; then
  # Same call, different spelling: keys reordered and a space after each colon.
  # Exact replay must reuse the sampled bytes regardless; canonical re-rendering
  # cannot, because the client's spelling is not what the model emitted.
  VARIANT=$(printf '%s' "$B2ARGS" | jq -c 'to_entries|reverse|from_entries' | sed 's/:/: /g')
  B=$(jq -n --argjson t "$TOOLS_OPENAI" --arg a "$B2" --arg m "$MODEL" \
    --arg id "$B2ID" --arg name "$B2NAME" --arg args "$VARIANT" \
    '{model:$m,messages:[{role:"user",content:$a},
       {role:"assistant",content:"",tool_calls:[{id:$id,type:"function",function:{name:$name,arguments:$args}}]},
       {role:"tool",tool_call_id:$id,content:"{\"temp_c\":7}"}],
      tools:$t,tool_choice:"auto",max_tokens:192,temperature:0,stream:false}')
  chat b02-chat-t2 "$B"
  C=$(cached "$OUT/b02-chat-t2.res.json"); P=$(tokens "$OUT/b02-chat-t2.res.json"); G=$((C - B2P))
  OBS="echo=$VARIANT; turn-1 prompt=$B2P; turn-2 prompt=$P cached=$C; replay gain=$G tokens"
  [ "$G" -gt 0 ] && V=PASS || V=FAIL
  row B-02 "Replay survives a differently spelled client echo" "cached exceeds turn-1 prompt with a reordered echo" "$OBS" "$V" "$OUT/b02-chat-t2.res.json"
else
  row B-02 "Replay survives a differently spelled client echo" "cached exceeds turn-1 prompt" "turn 1 produced no call" FAIL "$OUT/b02-chat-t1.res.json"
fi

hdr "B-03 anthropic replay pair"
B3="$(ask b03 "$ASK_W")"
B=$(jq -n --argjson t "$TOOLS_ANTHROPIC" --arg a "$B3" --arg m "$MODEL" \
  '{model:$m,max_tokens:192,temperature:0,messages:[{role:"user",content:$a}],tools:$t}')
anth b03-anthropic-t1 "$B"
B3P=$(tokens "$OUT/b03-anthropic-t1.res.json")
B3ID=$(jq -r '[.content[]?|select(.type=="tool_use")][0].id // ""' "$OUT/b03-anthropic-t1.res.json")
B3N=$(jq -r '[.content[]?|select(.type=="tool_use")][0].name // ""' "$OUT/b03-anthropic-t1.res.json")
B3IN=$(jq -c '[.content[]?|select(.type=="tool_use")][0].input // null' "$OUT/b03-anthropic-t1.res.json")
if [ -n "$B3ID" ] && [ "$B3IN" != "null" ]; then
  B=$(jq -n --argjson t "$TOOLS_ANTHROPIC" --arg a "$B3" --arg m "$MODEL" \
    --arg id "$B3ID" --arg name "$B3N" --argjson input "$B3IN" \
    '{model:$m,max_tokens:192,temperature:0,tools:$t,
      messages:[{role:"user",content:$a},
                {role:"assistant",content:[{type:"tool_use",id:$id,name:$name,input:$input}]},
                {role:"user",content:[{type:"tool_result",tool_use_id:$id,content:"{\"temp_c\":7}"}]}]}')
  anth b03-anthropic-t2 "$B"
  C=$(cached "$OUT/b03-anthropic-t2.res.json"); P=$(tokens "$OUT/b03-anthropic-t2.res.json"); G=$((C - B3P))
  OBS="turn-1 prompt=$B3P; turn-2 prompt=$P cached=$C; replay gain=$G tokens"
  [ "$G" -gt 0 ] && V=PASS || V=FAIL
  row B-03 "Anthropic turn 2 reuses the sampled tool_use block" "cached exceeds turn-1 prompt" "$OBS" "$V" "$OUT/b03-anthropic-t2.res.json"
else
  row B-03 "Anthropic turn 2 reuses the sampled tool_use block" "cached exceeds turn-1 prompt" "turn 1 produced no call" FAIL "$OUT/b03-anthropic-t1.res.json"
fi

hdr "B-04 responses replay pair"
B4="$(ask b04 "$ASK_W")"
B=$(jq -n --argjson t "$TOOLS_OPENAI" --arg a "$B4" --arg m "$MODEL" \
  '{model:$m,input:[{role:"user",content:$a}],tools:$t,max_output_tokens:192,temperature:0,stream:false}')
resp b04-responses-t1 "$B"
B4P=$(tokens "$OUT/b04-responses-t1.res.json")
B4CID=$(jq -r '[.output[]?|select(.type=="function_call")][0].call_id // ""' "$OUT/b04-responses-t1.res.json")
B4N=$(jq -r '[.output[]?|select(.type=="function_call")][0].name // ""' "$OUT/b04-responses-t1.res.json")
B4A=$(jq -r '[.output[]?|select(.type=="function_call")][0].arguments // ""' "$OUT/b04-responses-t1.res.json")
if [ -n "$B4CID" ] && [ -n "$B4A" ]; then
  B=$(jq -n --argjson t "$TOOLS_OPENAI" --arg a "$B4" --arg m "$MODEL" \
    --arg cid "$B4CID" --arg name "$B4N" --arg args "$B4A" \
    '{model:$m,input:[{role:"user",content:$a},
       {type:"function_call",call_id:$cid,name:$name,arguments:$args},
       {type:"function_call_output",call_id:$cid,output:"{\"temp_c\":7}"}],
      tools:$t,max_output_tokens:192,temperature:0,stream:false}')
  resp b04-responses-t2 "$B"
  C=$(cached "$OUT/b04-responses-t2.res.json"); P=$(tokens "$OUT/b04-responses-t2.res.json"); G=$((C - B4P))
  OBS="turn-1 prompt=$B4P; turn-2 prompt=$P cached=$C; replay gain=$G tokens"
  [ "$G" -gt 0 ] && V=PASS || V=FAIL
  row B-04 "Responses continuation reuses the sampled function_call" "cached exceeds turn-1 prompt" "$OBS" "$V" "$OUT/b04-responses-t2.res.json"
else
  row B-04 "Responses continuation reuses the sampled function_call" "cached exceeds turn-1 prompt" "turn 1 produced no call" FAIL "$OUT/b04-responses-t1.res.json"
fi

##############################################################################
# C. Recovery
##############################################################################

hdr "C-01 truncated DSML block"
C1="$(ask c01 "$ASK_W")"
B=$(jq -n --argjson t "$TOOLS_OPENAI" --arg a "$C1" --arg m "$MODEL" \
  '{model:$m,messages:[{role:"user",content:$a}],tools:$t,tool_choice:"auto",max_tokens:24,temperature:0,stream:false}')
chat c01-truncated "$B"
FINC=$(jq -r '.choices[0].finish_reason // ""' "$OUT/c01-truncated.res.json")
CONT=$(jq -r '.choices[0].message.content // ""' "$OUT/c01-truncated.res.json")
NCALLC=$(jq -r '[.choices[0].message.tool_calls[]?]|length' "$OUT/c01-truncated.res.json")
RAW=$(printf '%s' "$CONT" | grep -c 'DSML' || true)
OBS="finish=$FINC calls=$NCALLC raw-DSML lines in content=$RAW"
# The V4 pull route recovers a cut block into a call, or continues the turn with
# a model-visible tool error.  The served v41 route has no session and its own
# reference reports the failure instead (server_generate_v41.c:10, :258-263), so
# that criterion describes a route this artifact never enters and TP-11 is
# struck as inapplicable (CTR 8.2 R6, operator decision 2026-10-19).  The row
# asserts what the route owes the client in its place: the cut is reported in
# the client's own vocabulary, and half a block is never fabricated into a call.
# It fails if the truncation signal regresses to stop (NC-07) or if a partial
# block reaches the client as an executable call.
if [ "$FINC" = length ] && [ "$NCALLC" -eq 0 ]; then V=PASS; else V=FAIL; fi
row C-01 "A cut DSML block reports length and fabricates no call" "finish=length, 0 tool_calls (reference-absent boundary, CTR 8.2 R6)" "$OBS" "$V" "$OUT/c01-truncated.res.json"

# C-02/C-03: the cut point is not controllable, so the probe sweeps output
# budgets and scores every finish_reason observed.  The client of record
# (/data/open-grok, chat_completions.rs) deserializes finish_reason into a
# CLOSED enum, so any other value is a hard stream failure, not a soft one.
LEGAL_RE='^(stop|length|tool_calls|content_filter|function_call)$'
BUDGETS="8 16 24 32 48"

hdr "C-02 cut block, non-stream: finish_reason stays legal"
ILLEGAL=""
SEEN=""
for K in $BUDGETS; do
  A="$(ask c02k$K "$ASK_BOTH")"
  B=$(jq -n --argjson t "$TOOLS_OPENAI" --arg a "$A" --arg m "$MODEL" \
    '{model:$m,messages:[{role:"user",content:$a}],tools:$t,tool_choice:"auto",max_tokens:('"$K"'),temperature:0,stream:false}')
  chat "c02-cut$K" "$B"
  F=$(jq -r '.choices[0].finish_reason // ""' "$OUT/c02-cut$K.res.json")
  SEEN="$SEEN $K:$F"
  [ -n "$F" ] && GOT_FINISH=1
  if [ -n "$F" ] && ! printf '%s' "$F" | grep -Eq "$LEGAL_RE"; then ILLEGAL="$ILLEGAL $K:$F"; fi
done
# A body that never carries a finish_reason is not evidence of legality.
if [ "${GOT_FINISH:-0}" -eq 0 ]; then ILLEGAL="$ILLEGAL no-finish-reason-seen"; fi
[ -z "$ILLEGAL" ] && V=PASS || V=FAIL
row C-02 "A block cut at any budget keeps finish_reason inside the client's enum" "every finish_reason in {stop,length,tool_calls,content_filter,function_call}" "budgets:$SEEN" "$V" "$OUT/c02-cut48.res.json"

hdr "C-03 cut block, stream: no illegal finish_reason in the SSE"
ILLEGAL=""
SEEN=""
for K in $BUDGETS; do
  A="$(ask c03k$K "$ASK_BOTH")"
  B=$(jq -n --argjson t "$TOOLS_OPENAI" --arg a "$A" --arg m "$MODEL" \
    '{model:$m,messages:[{role:"user",content:$a}],tools:$t,tool_choice:"auto",max_tokens:('"$K"'),temperature:0,stream:true,stream_options:{include_usage:true}}')
  printf '%s' "$B" > "$OUT/c03-cut$K.req.json"
  "${CURL[@]}" -H 'Content-Type: application/json' -d "$B" "$BASE/v1/chat/completions" > "$OUT/c03-cut$K.sse"
  # Every finish_reason the stream carries, in order (the nulls of the delta
  # chunks are expected and skipped).
  F=$(grep -o '"finish_reason":"[^"]*"' "$OUT/c03-cut$K.sse" | sed 's/.*:"//; s/"$//' | sort -u | tr '\n' ',')
  SEEN="$SEEN $K:${F:-none}"
  [ -n "$F" ] && GOT_FINISH=1
  for one in $(printf '%s' "$F" | tr ',' ' '); do
    if ! printf '%s' "$one" | grep -Eq "$LEGAL_RE"; then ILLEGAL="$ILLEGAL $K:$one"; fi
  done
done
# A stream that never carries a finish_reason is not evidence of legality.
if [ "${GOT_FINISH:-0}" -eq 0 ]; then ILLEGAL="$ILLEGAL no-finish-reason-seen"; fi
[ -z "$ILLEGAL" ] && V=PASS || V=FAIL
row C-03 "A cut block never streams an illegal finish_reason" "every streamed finish_reason inside the client's enum" "budgets:$SEEN" "$V" "$OUT/c03-cut48.sse"

##############################################################################
# D. Regression outside DSML
##############################################################################

hdr "D-01 plain text KV prefix reuse"
P='Summarise in one sentence what a KV cache does in an inference server.'
B=$(jq -n --arg m "$MODEL" --arg p "$P" '{model:$m,messages:[{role:"user",content:$p}],max_tokens:48,temperature:0,stream:false}')
chat d01-kv-a "$B"; chat d01-kv-b "$B"
D=$(cached "$OUT/d01-kv-b.res.json")
OBS="first cached=$(cached "$OUT/d01-kv-a.res.json") second cached=$D"
[ "$D" -gt 0 ] && V=PASS || V=FAIL
row D-01 "Plain text KV prefix reuse still works" "second read reuses a prefix" "$OBS" "$V" "$OUT/d01-kv-b.res.json"

hdr "D-02 determinism witness"
B=$(jq -n --arg m "$MODEL" '{model:$m,messages:[{role:"user",content:"Reply with exactly the words: alpha beta gamma"}],max_tokens:32,temperature:0,seed:1234,stream:false}')
chat d02-det-a "$B"; chat d02-det-b "$B"
jq -r '.choices[0].message.content' "$OUT/d02-det-a.res.json" > "$OUT/d02.text.a"
jq -r '.choices[0].message.content' "$OUT/d02-det-b.res.json" > "$OUT/d02.text.b"
if diff -q "$OUT/d02.text.a" "$OUT/d02.text.b" >/dev/null; then V=PASS; else V=FAIL; fi
WIT=$(sha256sum "$OUT/d02.text.a" | cut -c1-16)
OBS="text=[$(head -c 40 "$OUT/d02.text.a")] sha16=$WIT"
row D-02 "Temperature-0 text is reproducible (cross-stage witness)" "two calls identical, witness recorded" "$OBS" "$V" "$OUT/d02.text.a"

##############################################################################
# E. Input validation: the oracle's guard over textual fields
##############################################################################

# A user-supplied image placeholder in a textual field is not a way to place an
# image.  The oracle refuses the literal in `content` and in `reasoning_content`
# (encoding.py:355-367, reached from encode_messages:826) and the port mirrors
# that refusal in the oracle's own sentences (CTR 8.9, R7).  Without it the
# literal tokenizes to the placeholder id and, with no image attached, no count
# ever sees it (generate.rs:3107), so the model gets a lone image token.
PH='<｜deepseek_image｜>'

hdr "E-01 a user-supplied image placeholder in content is refused"
B=$(jq -n --arg m "$MODEL" --arg ph "$PH" \
  '{model:$m,messages:[{role:"user",content:("Describe this: "+$ph)}],max_tokens:8}')
printf '%s' "$B" > "$OUT/e01-content.req.json"
CODE=$("${CURL[@]}" -o "$OUT/e01-content.res.json" -w '%{http_code}' \
  -H 'Content-Type: application/json' -d "$B" "$BASE/v1/chat/completions")
MSG=$(jq -r '.error.message // ""' "$OUT/e01-content.res.json")
OBS="http=$CODE msg=[$(printf '%s' "$MSG" | head -c 44)]"
if [ "$CODE" = 400 ] && printf '%s' "$MSG" | grep -q 'image special token'; then V=PASS; else V=FAIL; fi
row E-01 "A user-supplied image placeholder in content is refused" "400 naming the image special token (encoding.py:359)" "$OBS" "$V" "$OUT/e01-content.res.json"

hdr "E-02 the same literal in reasoning_content is refused"
B=$(jq -n --arg m "$MODEL" --arg ph "$PH" \
  '{model:$m,messages:[{role:"user",content:"Say hi"},{role:"assistant",reasoning_content:("a prior thought "+$ph),content:"hi"},{role:"user",content:"Again, one word"}],max_tokens:8}')
printf '%s' "$B" > "$OUT/e02-reasoning.req.json"
CODE=$("${CURL[@]}" -o "$OUT/e02-reasoning.res.json" -w '%{http_code}' \
  -H 'Content-Type: application/json' -d "$B" "$BASE/v1/chat/completions")
MSG=$(jq -r '.error.message // ""' "$OUT/e02-reasoning.res.json")
OBS="http=$CODE msg=[$(printf '%s' "$MSG" | head -c 44)]"
if [ "$CODE" = 400 ] && printf '%s' "$MSG" | grep -q 'reasoning_content'; then V=PASS; else V=FAIL; fi
row E-02 "A user-supplied image placeholder in reasoning_content is refused" "400 naming reasoning_content (encoding.py:365)" "$OBS" "$V" "$OUT/e02-reasoning.res.json"

hdr "E-03 a plain text turn is still admitted"
B=$(jq -n --arg m "$MODEL" \
  '{model:$m,messages:[{role:"user",content:"Reply with the single word: ok"}],max_tokens:8}')
printf '%s' "$B" > "$OUT/e03-control.req.json"
CODE=$("${CURL[@]}" -o "$OUT/e03-control.res.json" -w '%{http_code}' \
  -H 'Content-Type: application/json' -d "$B" "$BASE/v1/chat/completions")
CONT=$(jq -r '.choices[0].message.content // ""' "$OUT/e03-control.res.json")
OBS="http=$CODE content=[$(printf '%s' "$CONT" | head -c 24)]"
[ "$CODE" = 200 ] && V=PASS || V=FAIL
row E-03 "A plain text turn is still admitted" "200: the refusal is not blanket" "$OBS" "$V" "$OUT/e03-control.res.json"

##############################################################################
# Summary
##############################################################################
hdr "summary"
PASSN=$(awk -F'\t' '$5=="PASS"' "$TSV" | wc -l)
FAILN=$(awk -F'\t' '$5=="FAIL"' "$TSV" | wc -l)
printf 'PASS=%s FAIL=%s\n' "$PASSN" "$FAILN" | tee -a "$OUT/header.txt"
printf '\nFailing requirements:\n' | tee -a "$OUT/header.txt"
awk -F'\t' '$5=="FAIL"{printf "  %s  %s\n  observed: %s\n", $1, $2, $4}' "$TSV" | tee -a "$OUT/header.txt"
printf '%s\tpass=%s fail=%s\n' verdict "$PASSN" "$FAILN" >> "$TSV"
echo
echo "results: $TSV"
