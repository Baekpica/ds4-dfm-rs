#!/bin/sh
# ds41_kv_disk_turn_gate.sh -- the disk turn checkpoint across a restart.
#
# What it decides: whether the checkpoint this port takes when the run ends (T:
# the prompt followed by the ids the state committed, ds4_ds41_kvreuse.inc,
# v41_kvreuse_save_turn) survives a process boundary through the KVC disk store
# (crates/ds4-kv; v41_disk.rs) and a FRESH process restores it as the frontier
# the run left -- the restart-resume the disk half exists for.  The in-process
# half is gated by ds41_kv_post_gate.sh, the disk half's prompt class by
# ds41_kv_disk_gate.sh; this gate is the two together: stop, start, continue.
#
#   A  = one process, prompt P, N+1 tokens -> ids g0..gN and one row per token.
#        Its run end writes T under the key P+g0..g(N-1) (the ids the state
#        committed: the last sampled token launched no forward), so the store
#        holds two entries: P (the prompt class, saved at the prompt end) and T
#        (the turn class).  A's line reports T's token count.
#   B  = FRESH process, prompt P+tA (the model's own bytes), N tokens -> must
#        load T's payload from disk, adopt it, and land the reuse on the turn
#        class: "KV reuse np+N/np+N+1 token (turn checkpoint ..), prefill suffix
#        1".  A cold prefill of the same request pays np+N+1.
#   C  = FRESH process, --no-kv-reuse, the same request -> the reference.
#   D  = FRESH process, the prompt P alone -> the prompt class must still serve
#        it (no "turn" tag, suffix 0): the turn entry must not shadow K at P's
#        own length (v41_kvreuse_pick excludes T at equal length).
#
# The verdict is structural for B (the disk load line, the turn tag, the reuse
# the turn frontier gives, suffix 1) and the FP class for its rows: B's first
# row is the forward the client's last token launches, a position A never
# computed, so no row is bit-comparable with A and the comparison is against the
# cold run C -- the class core_v41_kvreuse.c:16-19 names for its own reuse (~1
# max, ~0.2 RMS; a span missed or left stale lands in a different continuation,
# 10 to 30 apart).  The identical-row count and the leading ids the arms share
# are reported, not asserted.
#
# usage: ds41_kv_disk_turn_gate.sh <server> <gguf> <engram_dir> [port] [sidecar]
#                                   [prompt-file] [prompt-bytes] [tokens]
#                                   [extra...]
#   defaults: port 8099, no sidecar, speed-bench/promessi_sposi.txt, 12000
#   bytes, 48 tokens.  Extra arguments are appended to every boot.
set -u

SRV="$1" MODEL="$2" ENGRAM="$3"
PORT="${4:-8099}"
ZC="${5:-}"
PFILE="${6:-speed-bench/promessi_sposi.txt}"
NCH="${7:-12000}"
N="${8:-48}"
EXTRA=""
# shift only when in range: dash exits 2 on `shift 8` with fewer than 8
# positional parameters, and that killed a sibling gate silently at launch.
if [ "$#" -gt 8 ]; then
    shift 8
    EXTRA="$*"
fi

OUT="${DS41_TURN_GATE_OUT:-/tmp/ds41_kv_disk_turn_gate}"
# The first row's delta bound: the reuse class measures about 1.1 max and 0.2
# RMS (ds41_kv_post_gate.sh, same route); a missed or stale span reads 10 to 30.
MAX_DELTA="${DS41_TURN_GATE_MAX_DELTA:-2.0}"
LOG="$OUT/log"
KVDIR="$OUT/kv"
mkdir -p "$LOG" || { echo "gate: cannot create $OUT" >&2; exit 1; }
rm -rf "$KVDIR"
mkdir -p "$KVDIR"
TMP="$OUT/tmp"
mkdir -p "$TMP"

for tool in jq curl dd cmp iconv awk; do
    command -v "$tool" >/dev/null 2>&1 || { echo "gate: $tool not found" >&2; exit 1; }
done
[ -x "$SRV" ] || { echo "gate: server not executable: $SRV" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "gate: gguf not found: $MODEL" >&2; exit 1; }
[ -d "$ENGRAM" ] || { echo "gate: engram dir not found: $ENGRAM" >&2; exit 1; }
[ -f "$PFILE" ] || { echo "gate: prompt file not found: $PFILE" >&2; exit 1; }

fail() { echo "ds41 KV disk turn gate: FAIL ($1)" >&2; exit 1; }
note() { echo "ds41 KV disk turn gate: $1"; }

SRV_PID=""
stop() {
    [ -n "$SRV_PID" ] || return 0
    kill "$SRV_PID" 2>/dev/null
    wait "$SRV_PID" 2>/dev/null
    SRV_PID=""
    sleep 3   # let the port go before the next boot
}
trap stop EXIT

ZFLAG=""
[ -n "$ZC" ] && ZFLAG="--zchain $ZC --mem-budget-mb 110000"

# The prompt is read once and used for every request; iconv drops a truncated
# multibyte tail from the byte cut.
iconv -c -f utf-8 -t utf-8 < "$PFILE" | head -c "$NCH" > "$LOG/p.txt" || fail "prompt slice failed"

boot() { # <name> <rows-file> [extra...]
    name="$1"; rows="$2"; shift 2
    SLOG="$LOG/$name.srv.log"
    : > "$SLOG"
    DS41_GEN_LOGITS="$rows" "$SRV" --cuda -m "$MODEL" --port "$PORT" --engram-dir "$ENGRAM" \
        $ZFLAG --emit-trace --kv-disk-dir "$KVDIR" $EXTRA "$@" --host 127.0.0.1 >> "$SLOG" 2>&1 &
    SRV_PID=$!
    i=0
    while [ "$i" -lt 900 ]; do
        grep -q "listening on" "$SLOG" 2>/dev/null && return 0
        kill -0 "$SRV_PID" 2>/dev/null || fail "$name: server died during boot (see $SLOG)"
        sleep 2
        i=$((i + 1))
    done
    fail "$name: server did not start listening (see $SLOG)"
}

request() { # <prompt-file> <max_tokens> <resp-out> ; prints the HTTP code
    jq -n --rawfile p "$1" --argjson mt "$2" \
        '{prompt:$p, max_tokens:$mt, temperature:0, frequency_penalty:1e-30}' > "$LOG/req.json"
    curl -sS -m 900 --noproxy '*' -o "$3" -w '%{http_code}' \
        "http://127.0.0.1:$PORT/v1/completions" \
        -H 'content-type: application/json' -d @"$LOG/req.json"
}

ids_of() { sed -n 's/^\[emit\] [0-9][0-9]* \(-\{0,1\}[0-9][0-9]*\)$/\1/p' "$1"; }
prefill_line() { sed -n 's/.*\(\[v41\] prefill [0-9]* token .*\)$/\1/p' "$1" | tail -1; }
prefill_n() { printf '%s' "$1" | sed -n 's/\[v41\] prefill \([0-9]*\) token.*/\1/p'; }
reuse_line() { sed -n 's/.*\(\[v41\] KV reuse [0-9]*\/[0-9]* token (\(turn \)\{0,1\}checkpoint [0-9.]* MB), prefill suffix [0-9]*\)$/\1/p' "$1" | tail -1; }
disk_line() { sed -n 's/.*\(\[v41\] KV disk .*checkpoint loaded .*\)$/\1/p' "$1" | tail -1; }
turn_line() { sed -n 's/.*\(\[v41\] KV turn checkpoint [0-9]* token ([0-9.]* MB)\)$/\1/p' "$1" | tail -1; }
turn_n() { printf '%s' "$1" | sed -n 's/\[v41\] KV turn checkpoint \([0-9]*\) token.*/\1/p'; }
reuse_x() { printf '%s' "$1" | sed -n 's/\[v41\] KV reuse \([0-9]*\)\/[0-9]* token.*/\1/p'; }
reuse_y() { printf '%s' "$1" | sed -n 's/\[v41\] KV reuse [0-9]*\/\([0-9]*\) token.*/\1/p'; }
reuse_suffix() { printf '%s' "$1" | sed -n 's/.*prefill suffix \([0-9]*\)$/\1/p'; }
reuse_turn() { printf '%s' "$1" | grep -c '(turn checkpoint' | tr -d ' '; }
kv_files() { find "$1" -maxdepth 1 -name '*.kv' | wc -l | tr -d ' '; }

rows_in() { # <rows-file> <row-bytes> ; prints the row count, or fails
    sz=$(wc -c < "$1" | tr -d ' ')
    [ "$2" -gt 0 ] || return 1
    [ $((sz % $2)) -eq 0 ] || return 1
    printf '%s' "$((sz / $2))"
}
row_delta() { # <fileA> <fileB' first row> <rowbytes> ; prints "max rms"
    dd if="$1" bs="$3" count=1 2>/dev/null | od -An -t f4 -v > "$TMP/row_a"
    dd if="$2" bs="$3" count=1 2>/dev/null | od -An -t f4 -v > "$TMP/row_b"
    awk 'BEGIN { mx = 0; s = 0; n = 0 }
         FNR == NR { for (i = 1; i <= NF; i++) { a[++na] = $i } ; next }
         { for (i = 1; i <= NF; i++) { d = $i - a[++nb]; if (d < 0) { d = -d }
                                       if (d > mx) { mx = d } ; s += d * d; n++ } }
         END { printf "%.4f %.5f", mx, (n ? sqrt(s / n) : 0) }' "$TMP/row_a" "$TMP/row_b"
}
row_cmp() { # <fileA> <idxA> <fileB> <idxB> <rowbytes>
    dd if="$1" bs="$5" skip="$2" count=1 2>/dev/null > "$TMP/ra"
    dd if="$3" bs="$5" skip="$4" count=1 2>/dev/null > "$TMP/rb"
    cmp -s "$TMP/ra" "$TMP/rb"
}
lead_ids() { # <fileA> <fileB> ; how many leading ids the arms share
    awk 'FNR == NR { a[FNR] = $0; n = FNR; next }
         { if ($0 != a[FNR]) { print FNR - 1; found = 1; exit } }
         END { if (!found) print n }' "$1" "$2"
}
delta_max() { printf '%s' "$1" | awk '{ print $1 }'; }

##############################################################################
# A: the run that leaves both checkpoints, the turn one written to disk
##############################################################################
note "A: boot, P -> $((N + 1)) token (the run whose end writes T)"
boot a "$OUT/a.rows"
grep -q "KV disk cache" "$LOG/a.srv.log" || fail "A: the store did not open (see $LOG/a.srv.log)"
code=$(request "$LOG/p.txt" "$((N + 1))" "$LOG/a.json") || fail "A transport"
[ "$code" = "200" ] || fail "A HTTP $code"
cp "$OUT/a.rows" "$LOG/a.rows" 2>/dev/null || fail "A wrote no logits dump"
ids_of "$LOG/a.srv.log" > "$LOG/a.ids"
prefill_line "$LOG/a.srv.log" > "$LOG/a.prefill"
n_a=$(wc -l < "$LOG/a.ids" | tr -d ' ')
[ "$n_a" -ge 8 ] || fail "A emitted $n_a tokens; raise the token count"
ptok=$(prefill_n "$(cat "$LOG/a.prefill")")
[ -n "$ptok" ] || fail "A reported no prefill count"
tl=$(turn_line "$LOG/a.srv.log")
[ -n "$tl" ] || fail "A wrote no turn checkpoint line; the run end did not save T"
tn=$(turn_n "$tl")
[ "$tn" = "$((ptok + N))" ] || fail "A's turn checkpoint holds $tn token, not the committed frontier $((ptok + N))"
[ "$(kv_files "$KVDIR")" -ge 2 ] || fail "A left $(kv_files "$KVDIR") entry in $KVDIR, not the prompt and turn pair"
sz_a=$(wc -c < "$LOG/a.rows" | tr -d ' ')
v=0
if [ $((sz_a % (4 * n_a))) -eq 0 ]; then
    v=$((sz_a / (4 * n_a)))
fi
[ "$v" -gt 1000 ] || fail "A's dump does not line up with its emitted count -- speculation likely ran (the penalty arm must be engaged: frequency_penalty non-zero; do not boot with --dspark)"
rowbytes=$((4 * v))
note "A: $(cat "$LOG/a.prefill")"
note "A: $tl, $n_a emitted token, vocabulary $v, $(kv_files "$KVDIR") store entries"
stop

# The client's replay: the model's own bytes appended to the prompt.
cat "$LOG/p.txt" > "$LOG/pfull.txt"
jq -j '.choices[0].text // ""' "$LOG/a.json" >> "$LOG/pfull.txt"

##############################################################################
# B: fresh process -- must load T from disk and resume at the turn end
##############################################################################
note "B: fresh process, P+tA -> must load the turn checkpoint and reuse it"
boot b "$OUT/b.rows"
code=$(request "$LOG/pfull.txt" "$N" "$LOG/b.json") || fail "B transport"
[ "$code" = "200" ] || fail "B HTTP $code"
cp "$OUT/b.rows" "$LOG/b.rows" 2>/dev/null || fail "B wrote no logits dump"
db=$(disk_line "$LOG/b.srv.log")
[ -n "$db" ] || fail "B loaded no disk checkpoint; the turn payload did not survive the restart"
rb=$(reuse_line "$LOG/b.srv.log")
[ -n "$rb" ] || fail "B reused nothing (no KV reuse line); the tokenizer likely merged across the boundary -- change the prompt byte count"
[ "$(reuse_turn "$rb")" -ge 1 ] || fail "B reused the prompt class, not the turn class ($rb); the payload's class or the pick is wrong"
bx=$(reuse_x "$rb"); by=$(reuse_y "$rb"); bs=$(reuse_suffix "$rb")
[ "$bx" = "$((ptok + N))" ] || fail "B reused $bx token; the turn frontier is $((ptok + N))"
[ "$bs" = "1" ] || fail "B prefilled a suffix of $bs token, not the 1 the client's last token launches"
[ "$by" = "$((bx + 1))" ] || fail "B's prompt is $by token against a reuse of $bx; the replay did not retokenize as emitted"
prefill_line "$LOG/b.srv.log" > "$LOG/b.prefill"
pb=$(prefill_n "$(cat "$LOG/b.prefill")")
[ "$pb" = "1" ] || fail "B prefilled $pb token, not 1 (the reuse should have carried the rest)"
ids_of "$LOG/b.srv.log" > "$LOG/b.ids"
n_b=$(wc -l < "$LOG/b.ids" | tr -d ' ')
[ "$n_b" -ge 8 ] || fail "B emitted $n_b tokens; raise the token count"
rows_in "$LOG/b.rows" "$rowbytes" >/dev/null || fail "B's dump does not line up with the vocabulary width"
note "B: $db"
note "B: $rb"
note "B: $(cat "$LOG/b.prefill") against A's $(cat "$LOG/a.prefill")"
stop

##############################################################################
# C: the cold reference for the same request (no checkpoint at all)
##############################################################################
note "C: fresh process, --no-kv-reuse, the same request -> the reference"
boot c "$OUT/c.rows" --no-kv-reuse
code=$(request "$LOG/pfull.txt" "$N" "$LOG/c.json") || fail "C transport"
[ "$code" = "200" ] || fail "C HTTP $code"
cp "$OUT/c.rows" "$LOG/c.rows" 2>/dev/null || fail "C wrote no logits dump"
[ -z "$(disk_line "$LOG/c.srv.log")" ] || fail "C loaded a checkpoint under --no-kv-reuse"
[ -z "$(reuse_line "$LOG/c.srv.log")" ] || fail "C reused tokens under --no-kv-reuse"
prefill_line "$LOG/c.srv.log" > "$LOG/c.prefill"
[ "$(prefill_n "$(cat "$LOG/c.prefill")")" = "$by" ] || fail "C prefilled $(prefill_n "$(cat "$LOG/c.prefill")") token, not the whole $by"
ids_of "$LOG/c.srv.log" > "$LOG/c.ids"
n_c=$(wc -l < "$LOG/c.ids" | tr -d ' ')
[ "$n_c" -ge 8 ] || fail "C emitted $n_c tokens"
rows_in "$LOG/c.rows" "$rowbytes" >/dev/null || fail "C's dump does not line up with the vocabulary width"
note "C: $(cat "$LOG/c.prefill")"
stop

##############################################################################
# D: the prompt class still serves the prompt itself (no shadowing at P)
##############################################################################
note "D: fresh process, the prompt P alone -> the prompt class, suffix 0"
boot d "$OUT/d.rows"
code=$(request "$LOG/p.txt" "$N" "$LOG/d.json") || fail "D transport"
[ "$code" = "200" ] || fail "D HTTP $code"
rd=$(reuse_line "$LOG/d.srv.log")
[ "$(reuse_turn "$rd")" = "0" ] || fail "D landed on the turn class for the prompt alone ($rd)"
[ "$(reuse_x "$rd")" = "$ptok" ] && [ "$(reuse_suffix "$rd")" = "0" ] || fail "D did not restore P whole ($rd)"
note "D: $rd"
stop

##############################################################################
# Verdict
##############################################################################
dr=$(row_delta "$LOG/b.rows" "$LOG/c.rows" "$rowbytes")
mx=$(delta_max "$dr")
rows=0
i=0
while [ "$i" -lt "$N" ]; do
    row_cmp "$LOG/b.rows" "$i" "$LOG/c.rows" "$i" "$rowbytes" && rows=$((rows + 1))
    i=$((i + 1))
done
lead=$(lead_ids "$LOG/b.ids" "$LOG/c.ids")

echo
echo "the measured cell:"
echo "  A: $(cat "$LOG/a.prefill")"
echo "  A: $tl"
echo "  B: $(cat "$LOG/b.prefill")"
echo "  B: $rb"
echo "  C: $(cat "$LOG/c.prefill")"
echo "  D: $rd"
echo "B against the cold run: row 0 max |d| $dr, $rows of $N rows byte-identical, $lead leading ids shared"

awk -v mx="$mx" -v bound="$MAX_DELTA" 'BEGIN { exit (mx <= bound) ? 0 : 1 }' || \
    fail "B's first row sits $mx from the cold run's, past the reuse class bound $bound"
echo "ds41 KV disk turn gate: PASS -- the restart resumed at the turn end ($rb), row 0 within the reuse class"
echo "artifacts: $OUT (log/, kv/)"
