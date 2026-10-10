#!/bin/sh
# ds41 V4.1 sampling gate (unit H): the port's served ids must equal the
# ENGINE's on the same requests under sampling, with the seed pinned.
#
# What makes ids byte-identical under sampling: same logits (gated since P4),
# same device kernel (cuda_v41_sample.inc.cu vs cuda/ds41_sample.cuh), the same
# coins (splitmix64 from seed/pos/i/salt) and the same rejection decisions;
# under speculation also the same draft draws (stream 1) and q rows.  Cases:
#   A  temp 0.7 + seed 12345                 (--no-dspark)  pure decode sampling
#   B  seed 12345 only (engine temp 1.0)     (--no-dspark)  the omitted default
#   C  temp 0.7 + frequency_penalty 0.5      (--no-dspark)  the host penalty route
#   D  temp 0.7 + seed 12345                 (default)      sampling + speculation
# Two boots per side: A/B/C share the --no-dspark boot, D the default one.
#
# usage: ds41_sampling_gate.sh <engine_server> <port_server> <gguf> <engram_dir> <port> [zchain_dir]
#
# With [zchain_dir] both sides boot with --zchain <dir>: the engine's own
# deployment spelling (install.sh passes --zchain "$MODEL/$SIDECAR"; the
# finance sidecar …-grrb-vqfin41_vqhalf_a_n8192-engine is the default
# domain).  The dir must have no spaces.
set -u

ENG="$1" SRV="$2" MODEL="$3" ENGRAM="$4" PORT="$5"
ZC="${6:-}"
ZFLAG=""
# The canonical install.sh line: --zchain <sidecar> --mem-budget-mb 110000
# (its BUDGET_MB).  The budget is inert for the ids; carrying it exercises
# both sides' flag acceptance and the L1 load gate at boot.
[ -n "$ZC" ] && ZFLAG="--zchain $ZC --mem-budget-mb 110000"
LOG=/tmp/ds41_sampling_gate

fail() { echo "ds41 sampling gate: FAIL ($1)"; exit 1; }

mkdir -p "$LOG"
cat > "$LOG/req.A.json" <<'JSON'
{"messages":[{"role":"user","content":"Name three colors."}],"max_tokens":48,"temperature":0.7,"seed":12345,"reasoning_effort":"none"}
JSON
cat > "$LOG/req.B.json" <<'JSON'
{"messages":[{"role":"user","content":"Name three colors."}],"max_tokens":48,"seed":12345,"reasoning_effort":"none"}
JSON
cat > "$LOG/req.C.json" <<'JSON'
{"messages":[{"role":"user","content":"Name three colors."}],"max_tokens":48,"temperature":0.7,"frequency_penalty":0.5,"seed":12345,"reasoning_effort":"none"}
JSON
cp "$LOG/req.A.json" "$LOG/req.D.json"

boot_wait() { # <log> <pid> <name>
    i=0
    while [ $i -lt 400 ]; do
        grep -q "listening on" "$1" && return 0
        kill -0 "$2" 2>/dev/null || fail "$3 died during boot (see $1)"
        sleep 2; i=$((i + 1))
    done
    fail "$3 did not start listening"
}

run_pair() { # <name> <extra flags...>
    name="$1"; shift
    eflags="$*"
    # ---- engine side (the oracle) ----
    # rm first: the redirect truncates only once the child runs, so a grep
    # racing that truncation would read the PREVIOUS run's log and pass
    # boot_wait instantly (hit 2026-10-10 18:49: port "request A failed"
    # while the server was still loading).
    rm -f "$LOG/$name.eng.log" "$LOG/$name.eng.trace"
    "$ENG" -m "$MODEL" --engram-dir "$ENGRAM" --port "$PORT" --trace "$LOG/$name.eng.trace" $ZFLAG $eflags > "$LOG/$name.eng.log" 2>&1 &
    EPID=$!
    boot_wait "$LOG/$name.eng.log" "$EPID" "engine server ($name)"
    for c in $CASES; do
        curl -sS -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H 'content-type: application/json' -d @"$LOG/req.$c.json" > "$LOG/$name.eng.$c.json" \
          || fail "engine request $c failed"
        grep -q '"finish_reason"' "$LOG/$name.eng.$c.json" || fail "engine response $c has no finish_reason"
    done
    kill "$EPID" 2>/dev/null; wait "$EPID" 2>/dev/null

    # ---- port side: the same requests ----
    rm -f "$LOG/$name.srv.log"
    "$SRV" -m "$MODEL" --port "$PORT" --engram-dir "$ENGRAM" --emit-trace $ZFLAG $eflags > "$LOG/$name.srv.log" 2>&1 &
    SPID=$!
    boot_wait "$LOG/$name.srv.log" "$SPID" "port server ($name)"
    grep -q "V4.1 serving route" "$LOG/$name.srv.log" || fail "no V4.1 route line at boot ($name)"
    for c in $CASES; do
        curl -sS -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H 'content-type: application/json' -d @"$LOG/req.$c.json" > "$LOG/$name.srv.$c.json" \
          || fail "port request $c failed"
        grep -q '"finish_reason"' "$LOG/$name.srv.$c.json" || fail "port response $c has no finish_reason"
    done
    kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null

    # ---- per-request id extraction ----
    # Engine: one "===== request <id>" .. "===== end request" block per
    # request; inside it the "prompt:" / "generated:" id lines
    # (server_trace.c:222-230).
    awk '
        /^===== request / { r++ }
        /^prompt:/    { sub(/^prompt: */, "");    P[r] = $0 }
        /^generated:/ { sub(/^generated: */, ""); G[r] = $0 }
        END { for (i = 1; i <= r; i++) print i "|" P[i] "|" G[i] }
    ' "$LOG/$name.eng.trace" > "$LOG/$name.eng.ids"
    # Port: [ptok] blocks precede their request's "[v41] prefill" line, [emit]
    # lines follow it; a new [ptok] block after a prefill line starts the next
    # request (core_v41_api.c:171-173 / :246).  r starts at 0: uninitialized,
    # awk would file request 1 under P[""] and the loop prints P[0] empty.
    awk '
        BEGIN { r = 0 }
        /\[v41\] prefill/ { pf = 1; next }
        /\[ptok\]/ { if (pf) { r++; pf = 0 } ; split($0, a, " "); P[r] = P[r] " " a[3]; next }
        /\[emit\]/ { split($0, a, " "); G[r] = G[r] " " a[3]; next }
        END { for (i = 0; i <= r; i++) { sub(/^ /, "", P[i]); sub(/^ /, "", G[i]); print i+1 "|" P[i] "|" G[i] } }
    ' "$LOG/$name.srv.log" > "$LOG/$name.srv.ids"

    # ---- per-case parity ----
    idx=0
    for c in $CASES; do
        idx=$((idx + 1))
        n=$idx
        e=$(sed -n "${n}p" "$LOG/$name.eng.ids")
        s=$(sed -n "${n}p" "$LOG/$name.srv.ids")
        [ -n "$e" ] || fail "$name: engine ids for case $c missing"
        [ -n "$s" ] || fail "$name: port ids for case $c missing"
        ep=$(printf '%s' "$e" | cut -d'|' -f2)
        eg=$(printf '%s' "$e" | cut -d'|' -f3)
        sp=$(printf '%s' "$s" | cut -d'|' -f2)
        sg=$(printf '%s' "$s" | cut -d'|' -f3)
        [ "$ep" = "$sp" ] || { printf 'engine: %s\nport:   %s\n' "$ep" "$sp" > "$LOG/$name.$c.prompt.diff"; fail "$name case $c: prompt ids differ (see $LOG/$name.$c.prompt.diff)"; }
        [ "$eg" = "$sg" ] || { printf 'engine: %s\nport:   %s\n' "$eg" "$sg" > "$LOG/$name.$c.gen.diff"; fail "$name case $c: generated ids differ (see $LOG/$name.$c.gen.diff)"; }
        efin=$(sed -n 's/.*"finish_reason":"\([^"]*\)".*/\1/p' "$LOG/$name.eng.$c.json")
        sfin=$(sed -n 's/.*"finish_reason":"\([^"]*\)".*/\1/p' "$LOG/$name.srv.$c.json")
        [ "$efin" = "$sfin" ] || fail "$name case $c: finish_reason differs (engine=$efin port=$sfin)"
        [ "$efin" = "length" ] && fail "$name case $c: hit the token cap"
        echo "  $name case $c: ids identical ($(printf '%s' "$eg" | wc -w) generated), finish $efin"
    done
}

CASES="A B C"
run_pair nodspark --no-dspark
CASES="D"
run_pair dspark

echo "ds41 sampling gate: PASS (engine vs port ids byte-identical: pure-decode sampling, the omitted-temperature default, the penalty route, and sampling under speculation)"
