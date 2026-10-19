#!/usr/bin/env bash
#
# DS41 DSML full gate - every fix of this branch in one run, fix by fix.
#
# This is the single entry point the operator runs to test the whole DSML
# carriage for the DeepSeek V4.1 (ds41) family and to obtain the evidence that
# each landed fix works.  It does not invent a protocol: it drives the line's
# own instruments and then maps every fix to the probe that measures it.
#
#   tests/spark/dsml_conformance.sh   normative served suite (A/B/C/D/E probes)
#   tests/spark/dsml_compliance.sh    compliance and benefit gate
#   tests/spark/open_grok_leg.sh      the client of record (opt out: --skip-client)
#   cargo test / make test-route-parity   host legs (oracle fixtures, parity)
#
# Why the client leg runs by default: the curl suites establish only the SHAPE
# the server returns.  They cannot establish that the one real OpenAI client that
# consumes this server survives a tool turn, which is the whole point of an
# OpenAI-compatible carriage.  open-grok speaks the OpenAI chat_completions wire
# with tools and parallel_tool_calls, so its leg is the OpenAI-compatibility
# evidence: G1 one tool call, G2 two tool calls in one turn, G3 the malformed
# turn.  It starts its own headless client process; it does not touch any tmux
# session the operator has open.
#
# Output: one evidence directory holding every raw body, every sub-suite's own
# results, and fix-index.tsv - the table that answers "did fix N work?".
#
# PRECONDITION the script cannot check for you: the base URL must be served by
# the binary built from the commit under test.  The served suite reads only the
# wire, so a server built from any other revision produces a full set of
# green rows about that other revision.  The header records the local revision
# and both /v1/models and /v1/stats so a mismatch is at least visible in the
# evidence.  Gate procedure: stop the operator's serve, start the branch binary
# on port 8008, run this, then restart the operator's serve.
#
# Usage:
#   bash tests/spark/ds41_dsml_full_gate.sh
#   bash tests/spark/ds41_dsml_full_gate.sh --base http://192.168.1.91:8008
#   bash tests/spark/ds41_dsml_full_gate.sh --host-only
#   bash tests/spark/ds41_dsml_full_gate.sh --skip-client
#   bash tests/spark/ds41_dsml_full_gate.sh --out /path/to/evidence-dir
#
# Exit: 0 when every HARD fix row passed, 1 when any failed, 2 when a stage
# could not run at all.

set -u
set -o pipefail

BASE="http://192.168.1.91:8008"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT=""
HOST_ONLY=0
SKIP_HOST=0
SKIP_PARITY=0
SKIP_CLIENT=0

while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --host-only) HOST_ONLY=1; shift;;
    --skip-host) SKIP_HOST=1; shift;;
    --skip-parity) SKIP_PARITY=1; shift;;
    --skip-client) SKIP_CLIENT=1; shift;;
    -h|--help) sed -n '2,44p' "${BASH_SOURCE[0]}"; exit 0;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2;;
  esac
done

[ -n "$OUT" ] || OUT="$PWD/ds41-dsml-full-gate-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
CONF_OUT="$OUT/conformance"
COMP_OUT="$OUT/compliance"
CLIENT_OUT="$OUT/client"
HOST_LOG="$OUT/host-tests.log"
FIX_TSV="$OUT/fix-index.tsv"

CURL=(curl -sS --max-time 300)
STAGE_FAIL=0

say()  { printf '%s\n' "$*"; }
hdr()  { printf '\n============================================================\n%s\n============================================================\n' "$1"; }

##############################################################################
# 0. Header and preconditions
##############################################################################

hdr "0. environment"
REV="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git -C "$ROOT" branch --show-current 2>/dev/null || echo unknown)"
# Only tracked modifications invalidate the revision: an untracked new file,
# this harness included, is not part of what the commit describes.
TRACKED_DIRTY="$(git -C "$ROOT" status --porcelain 2>/dev/null | grep -vc '^??')"
UNTRACKED="$(git -C "$ROOT" status --porcelain 2>/dev/null | grep -c '^??')"
DIRTY_LIST="$(git -C "$ROOT" status --porcelain 2>/dev/null | grep -v '^??' | awk '{print $2}' | tr '\n' ' ')"

{
  printf 'date:        %s\n' "$(date -Is)"
  printf 'base:        %s\n' "$BASE"
  printf 'repo:        %s\n' "$ROOT"
  printf 'revision:    %s\n' "$REV"
  printf 'branch:      %s\n' "$BRANCH"
  printf 'modified:    %s\n' "$TRACKED_DIRTY"
  printf 'untracked:   %s\n' "$UNTRACKED"
  printf 'harness:     %s\n' "${BASH_SOURCE[0]}"
} | tee "$OUT/header.txt"

if [ "$TRACKED_DIRTY" -ne 0 ]; then
  say "WARNING: tracked files are modified ($DIRTY_LIST); the revision above does not describe what is served"
fi

MODELS_JSON="$("${CURL[@]}" "$BASE/v1/models" 2>/dev/null || true)"
if [ -z "$MODELS_JSON" ]; then
  say "FATAL: $BASE/v1/models is unreachable; the served legs cannot run"
  [ "$HOST_ONLY" -eq 1 ] || exit 2
  MODEL="unreachable"
else
  printf '%s' "$MODELS_JSON" > "$OUT/models.json"
  MODEL="$(printf '%s' "$MODELS_JSON" | jq -r '.data[0].id // ""')"
  say "model: $MODEL"
  "${CURL[@]}" "$BASE/v1/stats" > "$OUT/stats-before.json" 2>/dev/null || true
fi
printf '%s\n' "$MODEL" > "$OUT/model.txt"

##############################################################################
# 1. Host legs: the fixes that are decided off the wire
#
# Each fix below is pinned by an oracle fixture or a parity tape, so the host
# leg is where it is decided.  One cargo invocation per target keeps the verdict
# attributable; a filtered lib run covers the trace module without dragging in
# the whole lib suite.
##############################################################################

HOST_TSV="$OUT/host.tsv"
: > "$HOST_TSV"

host_target() {                       # host_target <label> <cargo args...>
  local label="$1"; shift
  local log="$OUT/host-$label.log"
  if ( cd "$ROOT" && cargo test -q "$@" ) > "$log" 2>&1; then
    local n
    n="$(grep -oE '[0-9]+ passed' "$log" | awk '{p+=$1} END{print p+0}')"
    printf '%s\tPASS\t%s\n' "$label" "$n" >> "$HOST_TSV"
    printf '  %-18s PASS  %s tests\n' "$label" "$n"
  else
    local n
    n="$(grep -oE '[0-9]+ passed' "$log" | awk '{p+=$1} END{print p+0}')"
    printf '%s\tFAIL\t%s\n' "$label" "$n" >> "$HOST_TSV"
    printf '  %-18s FAIL  (%s tests passed) see %s\n' "$label" "$n" "$log"
    STAGE_FAIL=1
  fi
}

host_result() {                       # host_result <label> -> PASS|FAIL|MISSING
  awk -F'\t' -v l="$1" '$1==l{print $2}' "$HOST_TSV" | tail -1
}

if [ "$SKIP_HOST" -eq 0 ]; then
  hdr "1. host legs (oracle fixtures, DSML parse/render, repair, parity)"
  host_target ref_encoding -p ds4-server --test ref_encoding
  host_target ref_decode   -p ds4-server --test ref_decode
  host_target dsml         -p ds4-server --test dsml
  host_target render       -p ds4-server --test render
  host_target retry        -p ds4-server --test retry
  host_target parse        -p ds4-server --test parse
  host_target tools        -p ds4-server --test tools
  host_target stream       -p ds4-server --test stream
  host_target tool_stream  -p ds4-server --test tool_stream
  host_target stray_image  -p ds4-server --test stray_image_token
  host_target trace        -p ds4-server --lib trace::

  # The C repair oracle, driven the way CI drives it.  Known host-dependent
  # flake: cache_identity::tests::bounded_sidecar_and_ple compares a stat-only
  # identity across two same-size writes and fails when tmpfs gives both writes
  # the same coarse timestamp (measured: 241/300 back-to-back rewrites collide
  # on this host).  It is unrelated to the carriage, so a run whose only failure
  # is that test is recorded as FLAKY rather than FAIL.
  if [ "$SKIP_PARITY" -eq 0 ]; then
    PARITY_LOG="$OUT/parity.log"
    if ( cd "$ROOT" && make test-route-parity ) > "$PARITY_LOG" 2>&1; then
      printf 'parity\tPASS\t0\n' >> "$HOST_TSV"
      say "  parity             PASS  (C oracles, retry_c_oracle included)"
    else
      # Count failed TESTS, not the word FAILED: the log carries both the
      # per-test line and the summary line, so a line count overstates it.
      PARITY_FAILED="$(grep -oE '[0-9]+ failed' "$PARITY_LOG" | awk '{s+=$1} END{print s+0}')"
      if [ "$PARITY_FAILED" -le 1 ] \
         && grep -q 'cache_identity::tests::bounded_sidecar_and_ple' "$PARITY_LOG"; then
        printf 'parity\tFLAKY\t0\n' >> "$HOST_TSV"
        say "  parity             FLAKY (only cache_identity flaked, known host-dependent) see $PARITY_LOG"
      else
        printf 'parity\tFAIL\t0\n' >> "$HOST_TSV"
        say "  parity             FAIL ($PARITY_FAILED tests failed) see $PARITY_LOG"
        STAGE_FAIL=1
      fi
    fi
  else
    printf 'parity\tSKIPPED\t0\n' >> "$HOST_TSV"
  fi
fi

##############################################################################
# 2. Served legs
##############################################################################

if [ "$HOST_ONLY" -eq 0 ]; then
  hdr "2a. normative conformance suite (A/B/C/D/E probes)"
  bash "$ROOT/tests/spark/dsml_conformance.sh" "$BASE" "$CONF_OUT" || true

  hdr "2b. compliance and benefit gate"
  bash "$ROOT/tests/spark/dsml_compliance.sh" full "$BASE" "$COMP_OUT" || true

  if [ "$SKIP_CLIENT" -eq 0 ]; then
    hdr "2c. client of record: open-grok over the OpenAI wire (G1/G2/G3)"
    # The leg takes the CLIENT's model alias as its first argument and reads the
    # server from BASE_URL, not from an argument; passing the base URL here would
    # silently run it against the operator's configured default instead.
    BASE_URL="$BASE" bash "$ROOT/tests/spark/open_grok_leg.sh" \
      "${GROK_MODEL:-deepseek-v4.1-1017-flash-ds4}" "$CLIENT_OUT" || true
  fi

  "${CURL[@]}" "$BASE/v1/stats" > "$OUT/stats-after.json" 2>/dev/null || true
fi

conf_verdict() {                      # conf_verdict <probe id> -> PASS|FAIL|MISSING
  local tsv="$CONF_OUT/results.tsv"
  [ -f "$tsv" ] || { printf 'MISSING'; return; }
  local v
  v="$(awk -F'\t' -v id="$1" '$1==id{print $5}' "$tsv" | tail -1)"
  [ -n "$v" ] && printf '%s' "$v" || printf 'MISSING'
}

comp_verdict() {                      # comp_verdict <substring> -> PASS|FAIL|MISSING
  local sum="$COMP_OUT/summary.txt"
  [ -f "$sum" ] || { printf 'MISSING'; return; }
  if grep -qF "PASS  $1" "$sum"; then printf 'PASS'; return; fi
  if grep -qF "FAIL  $1" "$sum"; then printf 'FAIL'; return; fi
  printf 'MISSING'
}

client_verdict() {                    # client_verdict <substring> -> PASS|FAIL|MISSING
  local sum="$CLIENT_OUT/summary.txt"
  [ -f "$sum" ] || { printf 'MISSING'; return; }
  if grep -qF "PASS  $1" "$sum"; then printf 'PASS'; return; fi
  if grep -qF "FAIL  $1" "$sum"; then printf 'FAIL'; return; fi
  printf 'MISSING'
}

##############################################################################
# 3. Fix index - every fix in the commit, mapped to the probe that decides it
#
# class HARD     the fix must hold; a failure is a regression against this line
# class RESERVED the criterion is unreachable without the KV/session work the
#                operator excluded from this line (CTR clause 1.3, clause 9);
#                a failure here is the documented reservation, not a regression
##############################################################################

# A row is only as good as its evidence: FAIL wins over everything, an absent
# probe makes the row MISSING (unverified, never silently green), a reserved
# criterion is reported as such, and a probe the operator deliberately skipped
# is neutral rather than missing.
worst() {                             # worst <verdict...> -> FAIL|MISSING|RESERVED|PASS
  local out=PASS
  for v in "$@"; do
    case "$v" in
      FAIL) out=FAIL;;
      MISSING) [ "$out" = PASS ] && out=MISSING;;
      RESERVED) [ "$out" = PASS ] && out=RESERVED;;
      SKIPPED|FLAKY|"") ;;
    esac
  done
  printf '%s' "$out"
}

ROWS=0
ROWS_FAIL=0
ROWS_RESERVED=0
ROWS_MISSING=0

fix_row() {                           # fix_row <id> <title> <req> <class> <verdicts...>
  local id="$1" title="$2" req="$3" class="$4"; shift 4
  local v
  v="$(worst "$@")"
  # The probes report PASS or FAIL, never RESERVED, so a reserved criterion will
  # always arrive here as FAIL.  Recording it as a regression is what the CTR
  # warns against (clause 3, clause 9): the gain rows cannot pass without the KV
  # work this line excludes, so their failure is the reservation itself.
  if [ "$class" = RESERVED ] && [ "$v" = FAIL ]; then
    v=RESERVED
  fi
  local short="$*"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$title" "$req" "$class" "$v" "$short" >> "$FIX_TSV"
  printf '  %-9s %-6s %-9s %s\n' "$id" "$v" "$class" "$title"
  printf '            evidence: %s\n' "$short"
  ROWS=$((ROWS + 1))
  case "$v" in
    FAIL) ROWS_FAIL=$((ROWS_FAIL + 1));;
    RESERVED) ROWS_RESERVED=$((ROWS_RESERVED + 1));;
    MISSING) ROWS_MISSING=$((ROWS_MISSING + 1));;
  esac
}

if [ "$HOST_ONLY" -eq 0 ] || [ -s "$HOST_TSV" ]; then
  hdr "3. fix index - did every fix in this commit work?"
  : > "$FIX_TSV"
  printf 'id\ttitle\trequirement\tclass\tverdict\tevidence\n' >> "$FIX_TSV"

  fix_row FIX-01 "Effort header: the requested budget is rendered (low and numeric)" \
    "O-01 / enc-think-low, enc-think-budget" HARD "$(host_result ref_encoding)"
  fix_row FIX-02 "A mid-conversation system message keeps the oracle's position" \
    "O-02 / enc-midsystem-chat" HARD "$(host_result ref_encoding)"
  fix_row FIX-03 "Tool results are ordered by the assistant's call order" \
    "O-03 / enc-toolresults-order" HARD "$(host_result ref_encoding)"
  fix_row FIX-04 "DSML decode agrees with the oracle on every case" \
    "ref_decode cases" HARD "$(host_result ref_decode)"
  fix_row FIX-05 "Repair contract: only a missing wrapper is supplied" \
    "G8 / repair parity" HARD "$(host_result retry)" "$(host_result parity)"
  fix_row FIX-06 "Typed DSML parameters keep their JSON types" \
    "R-02 / A-02" HARD "$(conf_verdict A-02)" "$(comp_verdict 'openai non-stream arguments are JSON')"
  fix_row FIX-07 "OpenAI chat serves one DSML call as tool_calls" \
    "R-01 / A-01" HARD "$(conf_verdict A-01)" "$(comp_verdict 'openai non-stream finish_reason')" "$(comp_verdict 'openai non-stream tool name')"
  fix_row FIX-08 "The streamed turn carries the call and terminates" \
    "R-04 / A-04" HARD "$(conf_verdict A-04)" "$(comp_verdict 'openai stream carried a tool_calls delta')" "$(comp_verdict 'openai stream terminated')"
  fix_row FIX-09 "Two invokes in one block become two calls" \
    "R-03 / A-03" HARD "$(conf_verdict A-03)"
  fix_row FIX-10 "Anthropic serves a tool_use block with a toolu_ id" \
    "R-05 / A-05" HARD "$(conf_verdict A-05)" "$(comp_verdict 'anthropic tool id prefix (toolu_)')" "$(comp_verdict 'anthropic stop_reason')"
  fix_row FIX-11 "Responses serves a function_call item" \
    "R-06 / A-06" HARD "$(conf_verdict A-06)" "$(comp_verdict 'responses served a function_call item')"
  fix_row FIX-12 "A cut block reports length and fabricates no call" \
    "NC-02 / R-17 / C-01" HARD "$(conf_verdict C-01)"
  fix_row FIX-13 "Every finish_reason stays inside the client's closed enum" \
    "NC-07 / R-17 / C-02, C-03" HARD "$(conf_verdict C-02)" "$(conf_verdict C-03)"
  fix_row FIX-14 "A user-supplied image placeholder in text is refused, controls admitted" \
    "NC-09 / R-19 / E-01..E-03" HARD "$(conf_verdict E-01)" "$(conf_verdict E-02)" "$(conf_verdict E-03)" "$(host_result stray_image)"
  fix_row FIX-15 "Replay and trace diagnostics exist and work" \
    "NC-03 / R-12" HARD "$(host_result trace)"
  fix_row FIX-16 "Plain-text KV reuse still reuses a prefix" \
    "R-13 / D-01" HARD "$(conf_verdict D-01)" "$(comp_verdict 'plain KV reuse still reuses a prefix')"
  fix_row FIX-17 "Temperature-0 text is reproducible (witness)" \
    "R-14 / D-02" HARD "$(conf_verdict D-02)" "$(comp_verdict 'same-process determinism at temperature 0')"
  fix_row FIX-18 "Exact DSML replay reuses the sampled block on every surface" \
    "R-07..R-10 (reserved: needs the excluded KV work)" RESERVED \
    "$(conf_verdict B-01)" "$(conf_verdict B-02)" "$(conf_verdict B-03)" "$(conf_verdict B-04)"
  fix_row FIX-19 "A real OpenAI client completes a tool loop over the wire" \
    "R-18 / TP-18 / G1, G2, G3" HARD \
    "$(client_verdict 'G1 one tool call completes')" \
    "$(client_verdict 'G2 two tool calls in one turn both return')" \
    "$(client_verdict 'G3 malformed turn stays inside the client')"
fi

##############################################################################
# 4. Report
##############################################################################

hdr "4. report"

if [ -f "$CONF_OUT/results.tsv" ]; then
  CP="$(awk -F'\t' '$5=="PASS"' "$CONF_OUT/results.tsv" | wc -l)"
  CF="$(awk -F'\t' '$5=="FAIL"' "$CONF_OUT/results.tsv" | wc -l)"
  say "conformance rows: PASS=$CP FAIL=$CF"
fi
if [ -f "$COMP_OUT/summary.txt" ]; then
  say "compliance: $(grep -E '^stage=full' "$COMP_OUT/summary.txt" | tail -1)"
fi
say "fix index: rows=$ROWS failures=$ROWS_FAIL reserved=$ROWS_RESERVED unverified=$ROWS_MISSING"

# A green verdict means every row was measured; unverified rows are reported as
# INCOMPLETE rather than folded into a pass.
VERDICT=PASS
if [ "$ROWS_FAIL" -gt 0 ]; then VERDICT=FAIL; fi
if [ "$ROWS_MISSING" -gt 0 ] && [ "$VERDICT" = PASS ]; then VERDICT=INCOMPLETE; fi
if [ "$STAGE_FAIL" -ne 0 ] && [ "$VERDICT" = PASS ]; then VERDICT=HOST-STAGE-FAIL; fi

{
  printf '# DS41 DSML full gate\n\n'
  printf -- '- date: %s\n' "$(date -Is)"
  printf -- '- base: %s\n' "$BASE"
  printf -- '- model: %s\n' "$(cat "$OUT/model.txt" 2>/dev/null)"
  printf -- '- revision: %s (branch %s, modified %s, untracked %s)\n' "$REV" "$BRANCH" "$TRACKED_DIRTY" "$UNTRACKED"
  printf -- '- fix rows: %s, failures: %s, reserved: %s, unverified: %s\n' "$ROWS" "$ROWS_FAIL" "$ROWS_RESERVED" "$ROWS_MISSING"
  printf -- '- verdict: %s\n' "$VERDICT"
  printf -- '- precondition: the server must be the binary built from this revision; the wire cannot prove it, so the served rows attest the served build and no more\n\n'
  printf 'Fix index:\n\n'
  if [ -f "$FIX_TSV" ]; then
    awk -F'\t' 'NR>1{printf "- %s %s: %s (%s) - evidence: %s\n", $1, $5, $2, $4, $6}' "$FIX_TSV"
  fi
  printf '\nReserved rows are the criteria the operator put outside this line (CTR 1.3); they need the KV/session work and are not regressions.\n'
  printf 'Unverified rows are probes that produced no verdict; INCOMPLETE means the run does not establish those fixes either way.\n'
} > "$OUT/summary.md"
cat "$OUT/summary.md"

printf 'verdict=%s rows=%s failures=%s reserved=%s unverified=%s\n' \
  "$VERDICT" "$ROWS" "$ROWS_FAIL" "$ROWS_RESERVED" "$ROWS_MISSING" > "$OUT/verdict.txt"
printf '\nevidence: %s\n' "$OUT"

[ "$ROWS_FAIL" -eq 0 ] || exit 1
[ "$STAGE_FAIL" -eq 0 ] || exit 1
exit 0
