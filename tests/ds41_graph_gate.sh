#!/bin/sh
# ds41 decode-graph long-run gate (unit E3, phase 2): a run long enough to
# cross the 1024-position bucket and to open the draft graphs (pos0 >= the
# 128 window), which the 56-token phase-1 gate cannot reach.
#
# The engine's own rule is the bar: temperature-0 graph output byte-identical
# to direct dispatch, spec output byte-identical to pure decode.  On top:
#  - the n=1 graph must CAPTURE TWICE (buckets [9,1023] then [1024,...]) and
#    replay on both sides of the crossing (steps > captures);
#  - the verify-batch graph likewise (its rows straddle the boundary earlier);
#  - the draft graph must be captured and REUSED (graph rounds > captures);
#  - a scheduler run (k unpinned) must still emit the golden ids -- k only
#    affects speed, never the output.
#
# Engine goldens (capture once on the Spark; N=1100 from the 8-token prompt
# reaches position ~1108 > 1024):
#   ~/youngai/model/bin/ds4 --cuda -m <gguf> --engram-dir ~/youngai/deepseek-engram \
#     --gen-ids <ids> -n 1100 --temp 0 --no-dspark --emit-trace > /tmp/long_nospec.log 2>&1
#   ~/youngai/model/bin/ds4 --cuda -m <gguf> --engram-dir ~/youngai/deepseek-engram \
#     --gen-ids <ids> -n 1100 --temp 0 --dspark-verify 5 --emit-trace > /tmp/long_spec.log 2>&1
#
# usage: ds41_graph_gate.sh <harness> <gguf> <ids> <englog_nospec> <englog_spec> <n> <engram_dir> [<verify_k>]
set -u

H="$1" MODEL="$2" IDS="$3" ENGLOG="$4" ENGLOG_SPEC="$5" N="$6" ENGRAM="$7"
VK="${8:-5}"
LOG=/tmp/ds41_graph_gate

fail() { echo "ds41 graph gate: FAIL ($1)"; exit 1; }

[ -n "$ENGLOG" ] && [ -n "$ENGLOG_SPEC" ] || fail "both engine goldens are required"
ARGS="--engram-dir $ENGRAM"

# ---- non-spec: the n=1 graph, bucket crossing, reuse ----
DS41_NO_DSPARK=1 "$H" "$MODEL" "$IDS" "$ENGLOG" "$N" $ARGS > "$LOG.nospec.log" 2>&1 \
  || fail "non-spec run exited nonzero (see $LOG.nospec.log)"
grep -q "DS41 generate gate: PASS" "$LOG.nospec.log" || fail "non-spec run did not PASS (ids vs the engine golden)"
NS_STEPS=$(sed -n 's/.*walked \([0-9][0-9]*\) pure-decode steps.*/\1/p' "$LOG.nospec.log" | head -1)
NS_CAPS=$(sed -n 's/.*pure-decode steps, \([0-9][0-9]*\) captures.*/\1/p' "$LOG.nospec.log" | head -1)
[ -n "$NS_STEPS" ] && [ -n "$NS_CAPS" ] || fail "non-spec run: no [graph] walked line"
[ "$NS_CAPS" -ge 2 ] || fail "non-spec run: the 1024 bucket was never crossed (captures=$NS_CAPS)"
# The far-side capture's lower bound is the first position the graph was
# re-captured at (n=1 crosses exactly at 1024; a batch re-captures at the
# first row whose LAST row leaves the old cap -- the engine's condition
# pos0+n-1 > cap, mirrored).
NS_LO_MAX=$(grep -o "1-row graph captured: position bucket \[[0-9]*" "$LOG.nospec.log" | sed 's/.*\[//' | sort -n | tail -1)
[ -n "$NS_LO_MAX" ] && [ "$NS_LO_MAX" -ge 1024 ] || fail "non-spec run: no re-capture on the far side of 1024 (max bucket lo=$NS_LO_MAX)"
[ "$NS_STEPS" -gt "$NS_CAPS" ] || fail "non-spec run: no graph reuse (steps=$NS_STEPS captures=$NS_CAPS)"

# ---- spec pinned k: batch graph crossing + the draft graph ----
DS41_VERIFY_K="$VK" "$H" "$MODEL" "$IDS" "$ENGLOG_SPEC" "$N" $ARGS > "$LOG.spec.log" 2>&1 \
  || fail "spec run exited nonzero (see $LOG.spec.log)"
grep -q "DS41 generate gate: PASS" "$LOG.spec.log" || fail "spec run did not PASS (ids vs the engine golden)"
grep -o '\[dspark\] .*' "$ENGLOG_SPEC" | sed 's/[[:space:]]*$//' > "$LOG.eng.dspark"
grep -o '\[dspark\] .*' "$LOG.spec.log" | sed 's/[[:space:]]*$//' > "$LOG.port.dspark"
[ -s "$LOG.eng.dspark" ] || fail "the spec golden has no [dspark] lines"
diff "$LOG.eng.dspark" "$LOG.port.dspark" > "$LOG.dspark.diff" 2>&1 \
  || fail "the spec [dspark] trace differs (see $LOG.dspark.diff)"
SP_ROUNDS=$(sed -n 's/.*verify batches walked \([0-9][0-9]*\) rounds.*/\1/p' "$LOG.spec.log" | head -1)
[ -n "$SP_ROUNDS" ] || fail "spec run: no verify-batch graph walked line"
SP_CAPS=$(grep -c "row graph captured: position bucket" "$LOG.spec.log")
[ "$SP_CAPS" -ge 2 ] || fail "spec run: the batch graph never crossed 1024 (captures=$SP_CAPS)"
SP_LO_MAX=$(grep -o "row graph captured: position bucket \[[0-9]*" "$LOG.spec.log" | sed 's/.*\[//' | sort -n | tail -1)
[ -n "$SP_LO_MAX" ] && [ "$SP_LO_MAX" -ge 1024 ] || fail "spec run: no batch re-capture on the far side of 1024 (max bucket lo=$SP_LO_MAX)"
[ "$SP_ROUNDS" -gt "$SP_CAPS" ] || fail "spec run: no batch graph reuse (rounds=$SP_ROUNDS captures=$SP_CAPS)"
grep -q "draft graph (top-up .* rows) captured" "$LOG.spec.log" || fail "spec run: no draft graph was captured (pos0 >= 128 never reached?)"
D_GRAND=$(sed -n 's/.*cudaGraphLaunch(draft, \([0-9][0-9]*\) graph rounds).*/\1/p' "$LOG.spec.log" | head -1)
D_CAPS=$(sed -n 's/.*draft graphs \([0-9][0-9]*\) for .*/\1/p' "$LOG.spec.log" | head -1)
[ -n "$D_GRAND" ] && [ -n "$D_CAPS" ] || fail "spec run: no draft-graph host-gap line"
[ "$D_CAPS" -ge 1 ] || fail "spec run: no draft graph captured"
[ "$D_GRAND" -gt "$D_CAPS" ] || fail "spec run: no draft graph reuse (rounds=$D_GRAND captures=$D_CAPS)"

# ---- scheduler (k unpinned): the output is k-independent ----
"$H" "$MODEL" "$IDS" "$ENGLOG" "$N" $ARGS > "$LOG.sched.log" 2>&1 \
  || fail "scheduler run exited nonzero (see $LOG.sched.log)"
grep -q "DS41 generate gate: PASS" "$LOG.sched.log" \
  || fail "scheduler run did not PASS (k must not change the temperature-0 output)"
grep -q "verify batches walked" "$LOG.sched.log" || fail "scheduler run: no batch graph walked line"

echo "ds41 graph gate: PASS (bucket crossing re-captured and reused on both graphs, draft graphs captured and reused, scheduler output golden)"
