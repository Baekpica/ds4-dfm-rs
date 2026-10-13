#!/bin/sh
# ds41 Rust serving gate (unit F, P5.4): the Rust host's V4.1 one-shot route
# (crates/ds4-cli --gen-ids) against the engine's own captures, with the
# [emit] position+id instrument as the comparison, plus the required failing
# negative controls — a gate that only ever passes cannot tell correct from
# blind.
#
#   default              drafter armed (the GGUF carries the draft parameters)
#                        and the decode graph on: the [emit] id sequence
#                        equals the engine golden 1:1, every position is the
#                        engine's own convention (np + i), the verify-batch
#                        graph walked with reuse (rounds > captures — a silent
#                        direct fallback would make "graph == direct"
#                        vacuously green, the engine's 2026-09-18
#                        conviction), and the DSpark summary printed;
#   --no-graph           the same ids with the graph disabled and no [graph]
#                        walked line (graph == direct == golden);
#   --no-dspark          the same ids with the drafter off, the n=1 decode
#                        graph walked with reuse, and no DSpark summary;
#   --dspark-verify <k>  ids equal AND the [dspark] round lines identical to
#                        the engine's spec golden (k pinned because the
#                        scheduler's k follows measured wall-clock costs),
#                        the verify-batch graph walked with reuse, and the
#                        engine's own byte-identity rule on the port's own
#                        runs (spec ids == non-spec ids);
#   negatives            a text prompt without --gen-ids and --temp > 0 must
#                        both be refused by name (exit nonzero + message).
#
# usage: ds41_serving_gate.sh <ds4-binary> <gguf> <ids> <englog> <n> <engram_dir> [<englog_spec> [<verify_k>]]
set -u

BIN="$1" MODEL="$2" IDS="$3" ENGLOG="$4" N="$5" ENGRAM="$6"
ENGLOG_SPEC="${7:-}"
VK="${8:-5}"
LOG=/tmp/ds41_serving_gate

fail() { echo "ds41 serving gate: FAIL ($1)"; exit 1; }

# ---- the engine golden: [emit] position + id lines from its own run ----
grep -o '\[emit\] [0-9]* [0-9]*' "$ENGLOG" > "$LOG.golden.raw" || true
[ -s "$LOG.golden.raw" ] || fail "no [emit] lines in $ENGLOG (was the engine run with --emit-trace?)"
sed 's/\[emit\] //' "$LOG.golden.raw" > "$LOG.golden.emit"
NP_FILE=$(wc -w < "$IDS" | tr -d ' ')
NP_GOLD=$(head -1 "$LOG.golden.emit" | cut -d' ' -f1)
[ "$NP_FILE" = "$NP_GOLD" ] || fail "the golden's first position ($NP_GOLD) != the prompt length ($NP_FILE)"
# the engine's own position convention (tests/test_ds41_generate.cu:562):
# the i-th emitted token sits at np + i
awk -v np="$NP_FILE" '$1 != np + NR - 1 { print "position " $1 " at line " NR; bad = 1 } END { exit bad }' \
    "$LOG.golden.emit" > "$LOG.golden.poscheck" \
  || fail "the engine golden's positions are not sequential from the prompt length: $(head -1 "$LOG.golden.poscheck")"

run() { # run <tag> <extra flags...>
    tag="$1"; shift
    "$BIN" -m "$MODEL" --engram-dir "$ENGRAM" --gen-ids "$IDS" -n "$N" --temp 0 --emit-trace "$@" \
        > "$LOG.$tag.log" 2>&1 || fail "$tag run exited nonzero (see $LOG.$tag.log)"
}

check_ids() { # check_ids <tag>
    tag="$1"
    grep -o '\[emit\] [0-9]* [0-9]*' "$LOG.$tag.log" > "$LOG.$tag.raw" || true
    [ -s "$LOG.$tag.raw" ] || fail "$tag run printed no [emit] lines (--emit-trace wired to the native?)"
    sed 's/\[emit\] //' "$LOG.$tag.raw" > "$LOG.$tag.emit"
    awk -v np="$NP_FILE" '$1 != np + NR - 1 { print "position " $1 " at line " NR; bad = 1 } END { exit bad }' \
        "$LOG.$tag.emit" > "$LOG.$tag.poscheck" \
      || fail "$tag run: positions are not sequential from $NP_FILE (rollback drift?): $(head -1 "$LOG.$tag.poscheck")"
    diff "$LOG.golden.emit" "$LOG.$tag.emit" > "$LOG.$tag.emit.diff" 2>&1 \
      || fail "$tag run: the [emit] id sequence differs from the golden (see $LOG.$tag.emit.diff)"
}

# ---- default: drafter armed, decode graph on ----
run default
check_ids default
grep -q "DSpark:" "$LOG.default.log" || fail "default run: no DSpark summary (drafter not armed?)"
DEF_ROUNDS=$(sed -n 's/.*verify batches walked \([0-9][0-9]*\) rounds.*/\1/p' "$LOG.default.log" | head -1)
[ -n "$DEF_ROUNDS" ] || fail "default run: no verify-batch graph walked line (silent direct fallback?)"
DEF_CAPS=$(grep -c "row graph captured: position bucket" "$LOG.default.log")
[ "$DEF_CAPS" -ge 1 ] || fail "default run: no verify-batch graph captured"
[ "$DEF_ROUNDS" -gt "$DEF_CAPS" ] || fail "default run: no batch graph reuse (rounds=$DEF_ROUNDS captures=$DEF_CAPS)"

# ---- --no-graph control: direct == golden, and the graph stayed off ----
run nograph --no-graph
check_ids nograph
grep -q "pure-decode steps\|verify batches walked\|row graph captured" "$LOG.nograph.log" \
  && fail "--no-graph still walked the graph"

# ---- --no-dspark control: drafter off, the n=1 graph walked and reused ----
run nodspark --no-dspark
check_ids nodspark
grep -q "DSpark:" "$LOG.nodspark.log" && fail "--no-dspark still printed the DSpark summary"
grep -q '\[dspark\]' "$LOG.nodspark.log" && fail "--no-dspark still printed [dspark] rounds"
NS_STEPS=$(sed -n 's/.*walked \([0-9][0-9]*\) pure-decode steps.*/\1/p' "$LOG.nodspark.log" | head -1)
NS_CAPS=$(sed -n 's/.*pure-decode steps, \([0-9][0-9]*\) captures.*/\1/p' "$LOG.nodspark.log" | head -1)
[ -n "$NS_STEPS" ] && [ -n "$NS_CAPS" ] || fail "--no-dspark run: no [graph] walked line (the n=1 graph never ran)"
[ "$NS_STEPS" -gt "$NS_CAPS" ] || fail "--no-dspark run: no graph reuse (steps=$NS_STEPS captures=$NS_CAPS)"

# ---- spec positive: pinned k, the [dspark] trace diff ----
if [ -n "$ENGLOG_SPEC" ]; then
    run spec --dspark-verify "$VK"
    check_ids spec
    # the engine's own rule: temperature-0 spec and non-spec emit the same ids
    diff "$LOG.nodspark.emit" "$LOG.spec.emit" > /dev/null \
      || fail "the port's own runs disagree: spec ids != non-spec ids (the engine's byte-identity rule)"
    grep -o '\[dspark\] .*' "$ENGLOG_SPEC" | sed 's/[[:space:]]*$//' > "$LOG.eng.dspark"
    grep -o '\[dspark\] .*' "$LOG.spec.log" | sed 's/[[:space:]]*$//' > "$LOG.port.dspark"
    [ -s "$LOG.eng.dspark" ] || fail "the spec golden has no [dspark] lines (captured with --emit-trace?)"
    [ -s "$LOG.port.dspark" ] || fail "the port printed no [dspark] lines (--emit-trace wired?)"
    diff "$LOG.eng.dspark" "$LOG.port.dspark" > "$LOG.dspark.diff" 2>&1 \
      || fail "the spec [dspark] trace differs (see $LOG.dspark.diff)"
    SP_ROUNDS=$(sed -n 's/.*verify batches walked \([0-9][0-9]*\) rounds.*/\1/p' "$LOG.spec.log" | head -1)
    [ -n "$SP_ROUNDS" ] || fail "spec run: no verify-batch graph walked line (silent direct fallback?)"
    SP_CAPS=$(grep -c "row graph captured: position bucket" "$LOG.spec.log")
    [ "$SP_CAPS" -ge 1 ] || fail "spec run: no verify-batch graph captured"
    [ "$SP_ROUNDS" -gt "$SP_CAPS" ] || fail "spec run: no batch graph reuse (rounds=$SP_ROUNDS captures=$SP_CAPS)"
fi

# ---- negative controls: the two named refusals ----
"$BIN" -m "$MODEL" -p "hello" -n 4 --temp 0 > "$LOG.noids.log" 2>&1
rc=$?
[ "$rc" -ne 0 ] || fail "negative control (text prompt without --gen-ids) PASSED (blind gate)"
grep -q "runs on --gen-ids" "$LOG.noids.log" || fail "text prompt without --gen-ids: not refused by name"

"$BIN" -m "$MODEL" -p "hello" --engram-dir "$ENGRAM" --gen-ids "$IDS" -n 4 --temp 0.5 > "$LOG.temp.log" 2>&1
rc=$?
[ "$rc" -ne 0 ] || fail "negative control (--temp 0.5) PASSED (blind gate)"
grep -q "V4.1 sampling is not ported" "$LOG.temp.log" || fail "--temp 0.5: not refused by name"

echo "ds41 serving gate: positive PASS (default + no-graph + no-dspark + spec, [emit] ids and positions equal to the engine golden, [dspark] trace identical, n=1 and batch graphs walked with reuse), negative controls FAIL as required"
