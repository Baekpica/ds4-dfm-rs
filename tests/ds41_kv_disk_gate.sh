#!/bin/sh
# ds41_kv_disk_gate.sh -- the V4.1 disk half's (item 4) exact gate.
#
# What it decides: whether a checkpoint that crossed a process boundary
# through the KVC disk store (crates/ds4-kv; the native payload in
# ds4_ds41_kvreuse.inc) restores exactly what the in-memory checkpoint
# restores.  The in-process half is already gated (ds41_kv_reuse_gate.sh);
# this gate adds the file: export -> import -> restore must be
# byte-equivalent, and the negatives must stay cold, never crash.
#
#   A1 = prompt P, 1 token -> t1; A's process exports P's checkpoint
#   A  = prompt P, N tokens -> ids + one whole logits row per emitted token
#        (the in-process reference; A restores A1's own checkpoint)
#   B  = FRESH process, prompt P+t1, N-1 tokens -> must load A's checkpoint
#        from disk, restore |P|/|P+t1| with suffix 1, and reproduce A's
#        steps 1.. (ids and rows bit for bit); B exports its own checkpoint
#   C  = FRESH process, the same request as B -> must load B's checkpoint,
#        restore whole (suffix 0) and match B bit for bit
#   D  = FRESH process, an unrelated prompt -> cold: no load line, no reuse
#   E  = FRESH process on a truncated copy of the store -> cold, HTTP 200,
#        tokens still emitted (a short file is a miss, never a half load)
#   F  = FRESH process with --no-kv-reuse on the warm store -> cold and no
#        new file: the kill switch disables both halves
#   G  = FRESH process with a 1 MiB budget -> 200, no crash, no new file
#        (the export refuses by budget; the store's own contract)
#
# The row comparison is the U-B gate's, for the same reason: a complete
# checkpoint makes A's step k+1 and B's step k the same n=1 forward, and
# DS41_GEN_LOGITS dumps the raw f32 row per single-token decision
# (ds4_ds41_draft.inc:632-640), byte-exact.  frequency_penalty 1e-30 keeps
# the host penalty route (speculation off) and moves no fp32 logit.
#
# The store's default min_tokens is 512; this gate boots with
# --kv-cache-min-tokens 8 so short prompts store at all (the flag's own
# meaning is the store's, ds4_kvstore_default_options).
#
# usage: ds41_kv_disk_gate.sh <server> <gguf> <engram_dir> [port] [sidecar]
#                             [prompt-file] [prompt-bytes] [tokens] [extra...]
#   defaults: port 8099, no sidecar, speed-bench/promessi_sposi.txt,
#   12000 bytes, 48 tokens.  Extra arguments are appended to every boot.
set -u

SRV="$1" MODEL="$2" ENGRAM="$3"
PORT="${4:-8099}"
ZC="${5:-}"
PFILE="${6:-speed-bench/promessi_sposi.txt}"
NCH="${7:-12000}"
N="${8:-48}"
EXTRA=""
# shift only when in range: dash exits 2 on `shift 8` with fewer than 8
# positional parameters, and that killed this gate silently at launch.
if [ "$#" -gt 8 ]; then
    shift 8
    EXTRA="$*"
fi
MIN=8

ROOT="$(dirname "$0")/.."
OUT="${DS41_GATE_OUT:-/tmp/ds41_kv_disk_gate}"
LOG="$OUT/log"
KVDIR="$OUT/kv"
KVDIR_COPY="$OUT/kv-trunc"
KVDIR_SMALL="$OUT/kv-small"
mkdir -p "$LOG" || { echo "gate: cannot create $OUT" >&2; exit 1; }
rm -rf "$KVDIR" "$KVDIR_COPY" "$KVDIR_SMALL"
mkdir -p "$KVDIR" "$KVDIR_COPY" "$KVDIR_SMALL"
TMP="$OUT/tmp"
mkdir -p "$TMP"

for tool in jq curl dd cmp iconv; do
    command -v "$tool" >/dev/null 2>&1 || { echo "gate: $tool not found" >&2; exit 1; }
done
[ -x "$SRV" ] || { echo "gate: server not executable: $SRV" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "gate: gguf not found: $MODEL" >&2; exit 1; }
[ -d "$ENGRAM" ] || { echo "gate: engram dir not found: $ENGRAM" >&2; exit 1; }
[ -f "$PFILE" ] || { echo "gate: prompt file not found: $PFILE" >&2; exit 1; }

fail() { echo "ds41 KV disk gate: FAIL ($1)" >&2; exit 1; }
note() { echo "ds41 KV disk gate: $1"; }

SRV_PID=""
SLOG=""
stop() {
    [ -n "$SRV_PID" ] || return 0
    kill "$SRV_PID" 2>/dev/null
    wait "$SRV_PID" 2>/dev/null
    SRV_PID=""
}
trap stop EXIT

ZFLAG=""
[ -n "$ZC" ] && ZFLAG="--zchain $ZC --mem-budget-mb 110000"

# The prompt is read once and used for every request; iconv drops a truncated
# multibyte tail from the byte cut.  D uses a disjoint byte range.
iconv -c -f utf-8 -t utf-8 < "$PFILE" | head -c "$NCH" > "$LOG/p.txt" || fail "prompt slice failed"
iconv -c -f utf-8 -t utf-8 < "$PFILE" | tail -c "+$((NCH + 4000))" | head -c "$NCH" > "$LOG/pd.txt" || fail "negative prompt slice failed"

request() { # <prompt-file> <max_tokens> <resp-out> ; prints the HTTP code
    jq -n --rawfile p "$1" --argjson mt "$2" \
        '{prompt:$p, max_tokens:$mt, temperature:0, frequency_penalty:1e-30}' > "$LOG/req.json"
    curl -sS -m 900 --noproxy '*' -o "$3" -w '%{http_code}' \
        "http://127.0.0.1:$PORT/v1/completions" \
        -H 'content-type: application/json' -d @"$LOG/req.json"
}

boot() { # <name> <kvdir> <rows-file> [extra...]
    name="$1"; kvdir="$2"; rows="$3"; shift 3
    SLOG="$LOG/$name.srv.log"
    : > "$SLOG"
    DS41_GEN_LOGITS="$rows" "$SRV" --cuda -m "$MODEL" --port "$PORT" --engram-dir "$ENGRAM" \
        $ZFLAG --emit-trace --kv-disk-dir "$kvdir" --kv-cache-min-tokens "$MIN" \
        $EXTRA "$@" --host 127.0.0.1 >> "$SLOG" 2>&1 &
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

ids_of() { sed -n 's/^\[emit\] [0-9][0-9]* \(-\{0,1\}[0-9][0-9]*\)$/\1/p' "$1"; }
prefill_line() { sed -n 's/.*\(\[v41\] prefill [0-9]* token .*\)$/\1/p' "$1" | tail -1; }
reuse_line() { sed -n 's/.*\(\[v41\] KV reuse [0-9]*\/[0-9]* token (checkpoint [0-9.]* MB), prefill suffix [0-9]*\)$/\1/p' "$1" | tail -1; }
disk_line() { sed -n 's/.*\(\[v41\] KV disk checkpoint loaded .*\)$/\1/p' "$1" | tail -1; }
reuse_x() { printf '%s' "$1" | sed -n 's/\[v41\] KV reuse \([0-9]*\)\/[0-9]* token.*/\1/p'; }
reuse_y() { printf '%s' "$1" | sed -n 's/\[v41\] KV reuse [0-9]*\/\([0-9]*\) token.*/\1/p'; }
reuse_suffix() { printf '%s' "$1" | sed -n 's/.*prefill suffix \([0-9]*\)$/\1/p'; }
slice_from() { tail -c "+$(( $1 + 1 ))" "$SLOG" > "$2"; }
off_now() { wc -c < "$SLOG" | tr -d ' '; }
kv_files() { find "$1" -maxdepth 1 -name '*.kv' | wc -l | tr -d ' '; }

row_size() { # <rows-file> <ids-file> <V>
    n=$(wc -l < "$2" | tr -d ' ')
    sz=$(wc -c < "$1" | tr -d ' ')
    [ "$n" -gt 0 ] || return 1
    per=$((4 * $3))
    [ $((sz % per)) -eq 0 ] || return 1
    r=$((sz / per))
    [ "$r" = "$n" ]
}

row_cmp() { # <fileA> <idxA> <fileB> <idxB> <rowbytes>
    dd if="$1" bs="$5" skip="$2" count=1 2>/dev/null > "$TMP/ra"
    dd if="$3" bs="$5" skip="$4" count=1 2>/dev/null > "$TMP/rb"
    cmp -s "$TMP/ra" "$TMP/rb"
}

one_request() { # <slice-name> <prompt-file> <max-tokens>
    off=$(off_now)
    code=$(request "$2" "$3" "$LOG/$1.json") || fail "$1 transport"
    [ "$code" = "200" ] || fail "$1 HTTP $code"
    slice_from "$off" "$LOG/$1.log"
    ids_of "$LOG/$1.log" > "$LOG/$1.ids"
}

# ---------------------------------------------------------- A: the reference
note "A: boot, P -> 1 token, then P -> $N (the in-process reference)"
boot a "$KVDIR" "$LOG/a.rows"
grep -q "KV disk cache" "$LOG/a.srv.log" || fail "A: the store did not open (see $LOG/a.srv.log)"
one_request a1 "$LOG/p.txt" 1
t1=$(jq -j '.choices[0].text // ""' "$LOG/a1.json"; printf x)
t1=${t1%x}
printf '%s' "$(cat "$LOG/p.txt")$t1" > "$LOG/pt1.txt"
one_request a "$LOG/p.txt" "$N"
cp "$LOG/a.rows" "$LOG/a.rows.snap"
ids_of "$LOG/a.log" > "$LOG/a.ids"
n_a=$(wc -l < "$LOG/a.ids" | tr -d ' ')
[ "$n_a" -ge 8 ] || fail "A emitted $n_a tokens; raise the token count"
[ "$(kv_files "$KVDIR")" -ge 1 ] || fail "A exported no checkpoint to $KVDIR"
stop

# ------------------------------------------------- B: the disk restore (fresh)
note "B: fresh process, P+t1 -> must load A's checkpoint from disk"
boot b "$KVDIR" "$LOG/b.rows"
one_request b "$LOG/pt1.txt" "$((N - 1))"
db=$(disk_line "$LOG/b.log")
[ -n "$db" ] || fail "B loaded no disk checkpoint ($db); the export/import round trip is broken"
rb=$(reuse_line "$LOG/b.log")
[ -n "$rb" ] || fail "B reused nothing (no KV reuse line); the tokenizer likely merged across the boundary -- change the prompt byte count"
bx=$(reuse_x "$rb"); by=$(reuse_y "$rb"); bs=$(reuse_suffix "$rb")
[ "$bs" = "1" ] && [ "$by" = "$((bx + 1))" ] || fail "precondition not established: B restored $bx/$by with suffix $bs, not the +1-token extension of A"
ids_of "$LOG/b.log" > "$LOG/b.ids"
n_b=$(wc -l < "$LOG/b.ids" | tr -d ' ')
[ "$n_b" -ge 8 ] || fail "B emitted $n_b tokens; raise the token count"

v_a=0
sz_a=$(wc -c < "$LOG/a.rows.snap" | tr -d ' ')
# Pure penalty decode writes one full row per emitted token. Trying n+1
# first can infer a false vocabulary width when both divisions are exact.
if [ $((sz_a % (4 * n_a))) -eq 0 ]; then
    v_a=$((sz_a / (4 * n_a)))
fi
[ "$v_a" -gt 1000 ] || v_a=0
[ "$v_a" -gt 0 ] || fail "A's dump does not line up with its emitted count -- speculation likely ran (the penalty arm must be engaged: frequency_penalty non-zero; do not boot with --dspark)"
row_size "$LOG/b.rows" "$LOG/b.ids" "$v_a" || fail "B's dump does not line up with its emitted count (V=$v_a)"
rowbytes=$((4 * v_a))

k=0
while [ "$k" -lt "$n_b" ]; do
    a_id=$(sed -n "$((k + 2))p" "$LOG/a.ids")
    b_id=$(sed -n "$((k + 1))p" "$LOG/b.ids")
    [ "$a_id" = "$b_id" ] || fail "B token $k diverges: B=$b_id A[$(($k + 1))]=$a_id"
    row_cmp "$LOG/b.rows" "$k" "$LOG/a.rows.snap" "$((k + 1))" "$rowbytes" || {
        echo "ds41 KV disk gate: FAIL B row $k differs from A row $(($k + 1)) (token still identical: $b_id)"
        echo "  the exported payload is not the in-memory checkpoint ($db)"
        exit 1
    }
    k=$((k + 1))
done
echo "ds41 KV disk gate: PASS (B) -- disk load, $rb, $n_b tokens and $n_b whole rows match A shifted by one"
stop

# ------------------------------------------- C: the second fresh process, warm
note "C: fresh process, the same request -> must load B's checkpoint whole"
boot c "$KVDIR" "$LOG/c.rows"
one_request c "$LOG/pt1.txt" "$((N - 1))"
dc=$(disk_line "$LOG/c.log")
[ -n "$dc" ] || fail "C loaded no disk checkpoint"
rc=$(reuse_line "$LOG/c.log")
[ -n "$rc" ] || fail "C reused nothing"
cx=$(reuse_x "$rc"); cy=$(reuse_y "$rc"); cs=$(reuse_suffix "$rc")
[ "$cs" = "0" ] && [ "$cx" = "$cy" ] || fail "C did not restore the whole prompt ($rc)"
ids_of "$LOG/c.log" > "$LOG/c.ids"
n_c=$(wc -l < "$LOG/c.ids" | tr -d ' ')
[ "$n_c" = "$n_b" ] || fail "C emitted $n_c tokens, B emitted $n_b"
row_size "$LOG/c.rows" "$LOG/c.ids" "$v_a" || fail "C's dump does not line up with its emitted count (V=$v_a)"
k=0
while [ "$k" -lt "$n_c" ]; do
    b_id=$(sed -n "$((k + 1))p" "$LOG/b.ids")
    c_id=$(sed -n "$((k + 1))p" "$LOG/c.ids")
    [ "$b_id" = "$c_id" ] || fail "C token $k diverges from B: C=$c_id B=$b_id"
    row_cmp "$LOG/c.rows" "$k" "$LOG/b.rows" "$k" "$rowbytes" || fail "C row $k differs from B row $k (token still identical: $c_id)"
    k=$((k + 1))
done
echo "ds41 KV disk gate: PASS (C) -- identical prompt restored whole ($rc), $n_c tokens and rows bit-equal to B"
stop

# ------------------------------------------------------------- D: unrelated
note "D: fresh process, an unrelated prompt -> cold"
boot d "$KVDIR" "$LOG/d.rows"
one_request d "$LOG/pd.txt" "$N"
[ -z "$(disk_line "$LOG/d.log")" ] || fail "D loaded a checkpoint for an unrelated prompt"
[ -z "$(reuse_line "$LOG/d.log")" ] || fail "D reused tokens for an unrelated prompt"
[ "$(wc -l < "$LOG/d.ids" | tr -d ' ')" -ge 1 ] || fail "D emitted nothing"
stop

# ------------------------------------------------------- E: the truncated file
note "E: fresh process on a truncated copy -> cold, no crash"
for f in "$KVDIR"/*.kv; do
    [ -e "$f" ] || continue
    sz=$(wc -c < "$f" | tr -d ' ')
    head -c "$((sz / 2))" "$f" > "$KVDIR_COPY/$(basename "$f")"
done
[ "$(kv_files "$KVDIR_COPY")" -ge 1 ] || fail "E: the truncated copy is empty"
boot e "$KVDIR_COPY" "$LOG/e.rows"
one_request e "$LOG/pt1.txt" "$((N - 1))"
[ -z "$(disk_line "$LOG/e.log")" ] || fail "E loaded a truncated file"
[ -z "$(reuse_line "$LOG/e.log")" ] || fail "E reused tokens from a truncated file"
[ "$(wc -l < "$LOG/e.ids" | tr -d ' ')" -ge 1 ] || fail "E emitted nothing"
stop

# ------------------------------------------------------------ F: the kill switch
note "F: fresh process, --no-kv-reuse on the warm store -> cold, no new file"
before=$(kv_files "$KVDIR")
boot f "$KVDIR" "$LOG/f.rows" --no-kv-reuse
one_request f "$LOG/pt1.txt" "$((N - 1))"
[ -z "$(disk_line "$LOG/f.log")" ] || fail "F loaded a checkpoint under --no-kv-reuse"
[ -z "$(reuse_line "$LOG/f.log")" ] || fail "F reused tokens under --no-kv-reuse"
[ "$(kv_files "$KVDIR")" = "$before" ] || fail "F wrote a checkpoint under --no-kv-reuse"
stop

# --------------------------------------------------------------- G: the budget
note "G: fresh process, 1 MiB budget -> 200, no new file"
boot g "$KVDIR_SMALL" "$LOG/g.rows" --kv-disk-space-mb 1
one_request g "$LOG/pt1.txt" "$((N - 1))"
[ "$(kv_files "$KVDIR_SMALL")" = "0" ] || fail "G wrote a checkpoint past the 1 MiB budget"
[ "$(wc -l < "$LOG/g.ids" | tr -d ' ')" -ge 1 ] || fail "G emitted nothing"
stop

echo
echo "prefill lines (the measured cell):"
echo "  A: $(prefill_line "$LOG/a.log")"
echo "  B: $(prefill_line "$LOG/b.log")"
echo "  C: $(prefill_line "$LOG/c.log")"
echo "disk loads:"
echo "  B: $db"
echo "  C: $dc"
echo "ds41 KV disk gate: PASS -- all sides (A/B/C and the four negatives)"
echo "artifacts: $OUT (log/, kv/, kv-trunc/, kv-small/)"
