#!/bin/sh
# ds41 greedy-generate gate runner (unit E): the port's generate entry against
# the engine's own captures, in both modes, plus the required failing negative
# controls — a gate that only ever passes cannot tell correct from blind.
#
#   non-spec positive    DS41_NO_DSPARK=1 vs the --no-dspark engine golden:
#                        PASS, the row reference clean on every comparable
#                        position, the feed history equal to the golden, AND
#                        the n=1 decode graph walked with reuse (steps >
#                        captures — a silent direct fallback would make
#                        "graph == direct" vacuously green, the engine's
#                        2026-09-18 conviction);
#   spec positive        DS41_VERIFY_K=<k> vs the --dspark-verify <k> engine
#                        golden: PASS, the same history check, the [dspark]
#                        round lines identical (draft ids, main ids, acc, conf
#                        sigmoids) — with k pinned because the scheduler's k
#                        follows measured wall-clock costs — and the verify-
#                        batch graph walked with reuse;
#   DS41_NO_GRAPH=1      the same non-spec run with the graph disabled: PASS
#                        and no [graph] walked line (graph == direct == golden);
#   DS41_EMIT_OFFSET=3   the feed's declared position drifts (the E0 shape):
#                        the port must refuse the block ("engram feed pos0");
#   DS41_ROW_SHIFT=1     right declaration, wrong hash target: the row
#                        reference must MISMATCH while the port stays quiet
#                        (the two defenses are independent).
#
# usage: ds41_generate_gate.sh <harness> <gguf> <ids> <englog> <n> <engram_dir> <rows_ref> [<englog_spec> [<verify_k>]]
set -u

H="$1" MODEL="$2" IDS="$3" ENGLOG="$4" N="$5" ENGRAM="$6" REF="$7"
ENGLOG_SPEC="${8:-}"
VK="${9:-5}"
LOG=/tmp/ds41_gen_gate

fail() { echo "ds41 generate gate: FAIL ($1)"; exit 1; }

[ -n "$REF" ] || fail "ROWS_REF is required (the emitted positions must be gated)"
ARGS="--engram-dir $ENGRAM --rows-ref $REF"

# ---- non-spec positive ----
DS41_NO_DSPARK=1 "$H" "$MODEL" "$IDS" "$ENGLOG" "$N" $ARGS > "$LOG.positive.log" 2>&1 \
  || fail "non-spec run exited nonzero (see $LOG.positive.log)"
grep -q "DS41 generate gate: PASS" "$LOG.positive.log" || fail "non-spec run did not PASS"
[ "$(grep -c ", 0 mismatches" "$LOG.positive.log")" -ge 2 ] || fail "non-spec run: the row reference is not clean on both engram layers"
grep -q "harness history vs golden: 0 mismatches" "$LOG.positive.log" || fail "non-spec run: the feed history drifted from the golden"

# ---- the n=1 decode graph must have walked AND been reused ----
grep -q "decode step captured" "$LOG.positive.log" || fail "non-spec run: no n=1 graph captured (silent direct fallback?)"
NS_STEPS=$(sed -n 's/.*walked \([0-9][0-9]*\) pure-decode steps.*/\1/p' "$LOG.positive.log" | head -1)
NS_CAPS=$(sed -n 's/.*pure-decode steps, \([0-9][0-9]*\) captures.*/\1/p' "$LOG.positive.log" | head -1)
[ -n "$NS_STEPS" ] && [ -n "$NS_CAPS" ] || fail "non-spec run: no [graph] walked line (the n=1 graph never ran)"
[ "$NS_STEPS" -gt "$NS_CAPS" ] || fail "non-spec run: no graph reuse (steps=$NS_STEPS captures=$NS_CAPS)"

# ---- DS41_NO_GRAPH=1 control: direct == golden, and the graph stayed off ----
DS41_NO_DSPARK=1 DS41_NO_GRAPH=1 "$H" "$MODEL" "$IDS" "$ENGLOG" "$N" $ARGS > "$LOG.nograph.log" 2>&1 \
  || fail "no-graph control exited nonzero (see $LOG.nograph.log)"
grep -q "DS41 generate gate: PASS" "$LOG.nograph.log" || fail "no-graph control did not PASS (direct path != golden?)"
grep -q "pure-decode steps" "$LOG.nograph.log" && fail "DS41_NO_GRAPH=1 still walked the graph"

# ---- spec positive + the [dspark] trace diff ----
if [ -n "$ENGLOG_SPEC" ]; then
  DS41_VERIFY_K="$VK" "$H" "$MODEL" "$IDS" "$ENGLOG_SPEC" "$N" $ARGS > "$LOG.spec.log" 2>&1 \
    || fail "spec run exited nonzero (see $LOG.spec.log)"
  grep -q "DS41 generate gate: PASS" "$LOG.spec.log" || fail "spec run did not PASS"
  grep -q "harness history vs golden: 0 mismatches" "$LOG.spec.log" || fail "spec run: the feed history drifted from the golden"
  grep -o '\[dspark\] .*' "$ENGLOG_SPEC" | sed 's/[[:space:]]*$//' > "$LOG.eng.dspark"
  grep -o '\[dspark\] .*' "$LOG.spec.log" | sed 's/[[:space:]]*$//' > "$LOG.port.dspark"
  [ -s "$LOG.eng.dspark" ] || fail "the spec golden has no [dspark] lines (captured with --emit-trace?)"
  [ -s "$LOG.port.dspark" ] || fail "the port printed no [dspark] lines (g_ds4_v41_prof set?)"
  diff "$LOG.eng.dspark" "$LOG.port.dspark" > "$LOG.dspark.diff" 2>&1 \
    || fail "the spec [dspark] trace differs (see $LOG.dspark.diff)"
  # the engine's own rule: temperature-0 spec and non-spec emit the same ids
  grep -o '\[emit\] [0-9]* [0-9]*' "$ENGLOG_SPEC" | sed 's/\[emit\] //' > "$LOG.eng.spec.emit"
  grep -o '\[emit\] [0-9]* [0-9]*' "$ENGLOG" | sed 's/\[emit\] //' > "$LOG.eng.nospec.emit"
  diff "$LOG.eng.spec.emit" "$LOG.eng.nospec.emit" > /dev/null \
    || fail "engine goldens inconsistent: spec ids != non-spec ids (the engine's own byte-identity rule)"
  # ---- the verify-batch graph must have walked AND been reused ----
  SP_ROUNDS=$(sed -n 's/.*verify batches walked \([0-9][0-9]*\) rounds.*/\1/p' "$LOG.spec.log" | head -1)
  [ -n "$SP_ROUNDS" ] || fail "spec run: no verify-batch graph walked line (silent direct fallback?)"
  SP_CAPS=$(grep -c "row graph captured: position bucket" "$LOG.spec.log")
  [ "$SP_CAPS" -ge 1 ] || fail "spec run: no verify-batch graph captured"
  [ "$SP_ROUNDS" -gt "$SP_CAPS" ] || fail "spec run: no batch graph reuse (rounds=$SP_ROUNDS captures=$SP_CAPS)"
fi

# ---- negative controls (non-spec; the machinery is mode-independent) ----
DS41_NO_DSPARK=1 DS41_EMIT_OFFSET=3 "$H" "$MODEL" "$IDS" "$ENGLOG" "$N" $ARGS > "$LOG.emit_offset.log" 2>&1
rc=$?
[ "$rc" -ne 0 ] || fail "negative control DS41_EMIT_OFFSET=3 PASSED (blind gate)"
grep -q "engram feed pos0" "$LOG.emit_offset.log" || fail "DS41_EMIT_OFFSET=3: the port did not refuse the drifted feed"

DS41_NO_DSPARK=1 DS41_ROW_SHIFT=1 "$H" "$MODEL" "$IDS" "$ENGLOG" "$N" $ARGS > "$LOG.row_shift.log" 2>&1
rc=$?
[ "$rc" -ne 0 ] || fail "negative control DS41_ROW_SHIFT=1 PASSED (blind row reference)"
grep -q "MISMATCH" "$LOG.row_shift.log" || fail "DS41_ROW_SHIFT=1: the row reference did not fire"
grep -q "engram feed pos0" "$LOG.row_shift.log" && fail "DS41_ROW_SHIFT=1: the port refused too — the controls are not independent"

echo "ds41 generate gate: positive PASS (non-spec + spec, row reference and history clean, n=1 and batch graphs walked with reuse, no-graph control golden), negative controls FAIL as required"
