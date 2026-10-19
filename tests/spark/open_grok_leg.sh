#!/usr/bin/env bash
#
# Client-of-record leg (docs/ds41-dsml-conformance.md clause 6.4, TP-18).
#
# The curl suites establish the wire shape.  They cannot establish that the one
# client that actually consumes this server survives a tool turn, which is what
# this line exists for.  So this leg drives /data/open-grok itself, headless,
# against the server the operator's config names:
#
#   ~/.opengrok/config.toml  [model."deepseek-v4.1-1017-flash-ds4"]
#                            base_url = "http://192.168.1.91:8008/v1"
#                            api_backend = "chat_completions"
#
# PRECONDITION: the model id below must point at the server under test.  The
# gate procedure (clause 6.1) starts the branch binary on port 8008 with the
# operator's serve stopped, so that entry then names the IUT.  Nothing here
# edits config.
#
# Legs:
#   G1  one tool call in a turn: the tool runs, its result returns, the turn
#       completes (the client accepts the server's finish_reason).
#   G2  two tool calls in one turn (the client sets parallel_tool_calls):
#       both results return.
#   G3  the malformed/truncated turn of TP-17 is scored by the conformance
#       suite's C-02/C-03 probes, because the client cannot itself choose a
#       truncation point; this leg re-runs C-02's request shape so the evidence
#       sits with the client run.
#
# Usage: bash open_grok_leg.sh [model] [out_dir]

set -u

MODEL="${1:-deepseek-v4.1-1017-flash-ds4}"
OUT="${2:-$PWD/ds41-open-grok-leg-$(date +%Y%m%d-%H%M%S)}"
GROK="${GROK_BIN:-/data/open-grok/bin/open-grok}"
SCRATCH="$(mktemp -d /tmp/ds41-client-leg-XXXXXX)"

mkdir -p "$OUT"
# The legs run from a scratch directory (cd below), so resolve OUT to an
# absolute path first: a relative OUT would stop resolving after the cd.
OUT="$(cd "$OUT" && pwd)"
SUM="$OUT/summary.txt"
PASS=0
FAIL=0

ok()  { PASS=$((PASS + 1)); printf 'PASS  %s\n' "$1" | tee -a "$SUM"; }
bad() { FAIL=$((FAIL + 1)); printf 'FAIL  %s\n' "$1" | tee -a "$SUM"; }
say() { printf '%s\n' "$*" | tee -a "$SUM" >/dev/null; }

cd "$SCRATCH" || exit 1
printf 'model=%s\nclient=%s\nscratch=%s\ndate=%s\n' \
  "$MODEL" "$GROK" "$SCRATCH" "$(date -Is)" | tee "$OUT/header.txt"

if [ ! -x "$GROK" ]; then
  bad "client binary present ($GROK)"
  printf 'verdict pass=%s fail=%s\n' "$PASS" "$FAIL" | tee -a "$SUM"
  exit 1
fi

# G1: one tool call.  The prompt names a file operation the agent cannot answer
# from memory, so it must call a tool and read the result back.
say "=== G1 one tool call ==="
printf 'dsml-leg-marker\n' > "$SCRATCH/witness.txt"
G1_PROMPT="Use your bash tool to run: cat witness.txt — then reply with exactly the file contents and nothing else."
timeout 300 "$GROK" -m "$MODEL" --always-approve --output-format json --max-turns 4 \
  -p "$G1_PROMPT" > "$OUT/g1.json" 2> "$OUT/g1.stderr"
G1_RC=$?
G1_HIT=$(grep -c 'dsml-leg-marker' "$OUT/g1.json" 2>/dev/null || true)
say "g1 rc=$G1_RC marker-hits=$G1_HIT"
if [ "$G1_RC" -eq 0 ] && [ "${G1_HIT:-0}" -ge 1 ]; then
  ok "G1 one tool call completes and its result reaches the model"
else
  bad "G1 one tool call (rc=$G1_RC, marker-hits=${G1_HIT:-0})"
fi

# G2: two tool calls in one turn, which is where the client sends
# parallel_tool_calls=true and sends the results back in completion order.
say "=== G2 two tool calls in one turn ==="
printf 'alpha\n' > "$SCRATCH/w1.txt"
printf 'beta\n'  > "$SCRATCH/w2.txt"
G2_PROMPT="In a single turn, use your bash tool to cat both w1.txt and w2.txt, then reply with the two file contents and nothing else."
timeout 300 "$GROK" -m "$MODEL" --always-approve --output-format json --max-turns 4 \
  -p "$G2_PROMPT" > "$OUT/g2.json" 2> "$OUT/g2.stderr"
G2_RC=$?
G2_A=$(grep -c 'alpha' "$OUT/g2.json" 2>/dev/null || true)
G2_B=$(grep -c 'beta' "$OUT/g2.json" 2>/dev/null || true)
say "g2 rc=$G2_RC alpha-hits=$G2_A beta-hits=$G2_B"
if [ "$G2_RC" -eq 0 ] && [ "${G2_A:-0}" -ge 1 ] && [ "${G2_B:-0}" -ge 1 ]; then
  ok "G2 two tool calls in one turn both return"
else
  bad "G2 two tool calls (rc=$G2_RC, alpha=${G2_A:-0}, beta=${G2_B:-0})"
fi

# G3: the malformed-turn shape, re-run beside the client evidence.
say "=== G3 malformed turn (wire legality) ==="
BASE="${BASE_URL:-http://192.168.1.91:8008}"
TOOLS='[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'
B=$(jq -n --argjson t "$TOOLS" --arg m "$MODEL" \
  '{model:$m,messages:[{role:"user",content:"Call get_weather for Stockholm and set a timer for 90 seconds. Call both tools."}],tools:$t,tool_choice:"auto",max_tokens:16,temperature:0,stream:true,stream_options:{include_usage:true}}')
printf '%s' "$B" > "$OUT/g3.req.json"
curl -sS --max-time 300 -H 'Content-Type: application/json' -d "$B" "$BASE/v1/chat/completions" > "$OUT/g3.sse"
G3_F=$(grep -o '"finish_reason":"[^"]*"' "$OUT/g3.sse" | sed 's/.*:"//; s/"$//' | sort -u | tr '\n' ',')
say "g3 finish_reasons=${G3_F:-none}"
if printf '%s' "$G3_F" | grep -Eq '^(stop|length|tool_calls|content_filter|function_call),$'; then
  ok "G3 malformed turn stays inside the client's finish_reason enum"
else
  bad "G3 malformed turn (finish_reasons=${G3_F:-none})"
fi

printf 'verdict pass=%s fail=%s\n' "$PASS" "$FAIL" | tee -a "$SUM"
echo "results: $OUT"
[ "$FAIL" -eq 0 ] || exit 1
