#!/bin/sh
# ds41 V4.1 wire gate (unit H): the Anthropic (/v1/messages) and Responses
# (/v1/responses) surfaces on the v41 route.  The engine --trace dumps the
# rendered prompt and both id lists per request (server_trace.c:222-230); the
# port --emit-trace prints [ptok] (core_v41_api.c:171-173) and [emit] (:246).
# Both surfaces share the chat renderer (server_httpd.c:246-252), so the ids
# must be byte-identical at greedy; this drives the two parse paths end to end.
#
# Two greedy requests (reasoning off) through one boot per side:
#   1  POST /v1/messages   messages + max_tokens + reasoning_effort
#   2  POST /v1/responses  input + max_output_tokens + reasoning.effort
#
# usage: ds41_wire_gate.sh <engine_server> <port_server> <gguf> <engram_dir> <port> [zchain_dir]
#
# With [zchain_dir] both sides boot with --zchain <dir>, the engine's own
# deployment spelling (install.sh).  The dir must have no spaces.
set -u

ENG="$1" SRV="$2" MODEL="$3" ENGRAM="$4" PORT="$5"
ZC="${6:-}"
ZFLAG=""
# The canonical install.sh line: --zchain <sidecar> --mem-budget-mb 110000
# (its BUDGET_MB).  The budget is inert for the ids; carrying it exercises
# both sides' flag acceptance and the L1 load gate at boot.
[ -n "$ZC" ] && ZFLAG="--zchain $ZC --mem-budget-mb 110000"
LOG=/tmp/ds41_wire_gate

fail() { echo "ds41 wire gate: FAIL ($1)"; exit 1; }

mkdir -p "$LOG"
cat > "$LOG/req.anthropic.json" <<'JSON'
{"max_tokens":48,"temperature":0,"reasoning_effort":"none","messages":[{"role":"user","content":"Name three colors."}]}
JSON
cat > "$LOG/req.responses.json" <<'JSON'
{"input":"Name three colors.","max_output_tokens":48,"temperature":0,"reasoning":{"effort":"none"}}
JSON

boot_wait() { # <log> <pid> <name>
    i=0
    while [ $i -lt 400 ]; do
        grep -q "listening on" "$1" && return 0
        kill -0 "$2" 2>/dev/null || fail "$3 died during boot (see $1)"
        sleep 2; i=$((i + 1))
    done
    fail "$3 did not start listening"
}

# ---- engine side (the oracle) ----
# rm first: the redirect truncates only once the child runs, so a grep racing
# that truncation would read the PREVIOUS run's log and pass boot_wait
# instantly.
rm -f "$LOG/eng.log" "$LOG/eng.trace"
"$ENG" -m "$MODEL" --engram-dir "$ENGRAM" --port "$PORT" --trace "$LOG/eng.trace" $ZFLAG > "$LOG/eng.log" 2>&1 &
EPID=$!
boot_wait "$LOG/eng.log" "$EPID" "engine server"
curl -sS -m 900 "http://127.0.0.1:$PORT/v1/messages" \
    -H 'content-type: application/json' -d @"$LOG/req.anthropic.json" > "$LOG/eng.anthropic.json" \
  || fail "engine anthropic request failed"
grep -q '"stop_reason"' "$LOG/eng.anthropic.json" || fail "engine anthropic response has no stop_reason"
curl -sS -m 900 "http://127.0.0.1:$PORT/v1/responses" \
    -H 'content-type: application/json' -d @"$LOG/req.responses.json" > "$LOG/eng.responses.json" \
  || fail "engine responses request failed"
grep -q '"status"' "$LOG/eng.responses.json" || fail "engine responses response has no status"
kill "$EPID" 2>/dev/null; wait "$EPID" 2>/dev/null

# ---- port side: the same requests ----
rm -f "$LOG/srv.log"
"$SRV" -m "$MODEL" --port "$PORT" --engram-dir "$ENGRAM" --emit-trace $ZFLAG > "$LOG/srv.log" 2>&1 &
SPID=$!
boot_wait "$LOG/srv.log" "$SPID" "port server"
grep -q "V4.1 serving route" "$LOG/srv.log" || fail "no V4.1 route line at boot"
curl -sS -m 900 "http://127.0.0.1:$PORT/v1/messages" \
    -H 'content-type: application/json' -d @"$LOG/req.anthropic.json" > "$LOG/srv.anthropic.json" \
  || fail "port anthropic request failed"
grep -q '"stop_reason"' "$LOG/srv.anthropic.json" || fail "port anthropic response has no stop_reason"
curl -sS -m 900 "http://127.0.0.1:$PORT/v1/responses" \
    -H 'content-type: application/json' -d @"$LOG/req.responses.json" > "$LOG/srv.responses.json" \
  || fail "port responses request failed"
grep -q '"status"' "$LOG/srv.responses.json" || fail "port responses response has no status"
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null

# ---- per-request id extraction ----
# Engine: one "===== request <id>" .. block per request; "prompt:"/"generated:"
# lines inside (server_trace.c:222-230).  Port: [ptok] blocks precede their
# request's "[v41] prefill" line, [emit] lines follow it; a new [ptok] block
# after a prefill line starts the next request.
awk '
    /^===== request / { r++ }
    /^prompt:/    { sub(/^prompt: */, "");    P[r] = $0 }
    /^generated:/ { sub(/^generated: */, ""); G[r] = $0 }
    END { for (i = 1; i <= r; i++) print i "|" P[i] "|" G[i] }
' "$LOG/eng.trace" > "$LOG/eng.ids"
awk '
    BEGIN { r = 0 }
    /\[v41\] prefill/ { pf = 1; next }
    /\[ptok\]/ { if (pf) { r++; pf = 0 } ; split($0, a, " "); P[r] = P[r] " " a[3]; next }
    /\[emit\]/ { split($0, a, " "); G[r] = G[r] " " a[3]; next }
    END { for (i = 0; i <= r; i++) { sub(/^ /, "", P[i]); sub(/^ /, "", G[i]); print i+1 "|" P[i] "|" G[i] } }
' "$LOG/srv.log" > "$LOG/srv.ids"

# ---- per-surface parity ----
check() { # <name> <request index>
    name="$1"; n="$2"
    e=$(sed -n "${n}p" "$LOG/eng.ids")
    s=$(sed -n "${n}p" "$LOG/srv.ids")
    [ -n "$e" ] || fail "$name: engine ids missing"
    [ -n "$s" ] || fail "$name: port ids missing"
    ep=$(printf '%s' "$e" | cut -d'|' -f2)
    eg=$(printf '%s' "$e" | cut -d'|' -f3)
    sp=$(printf '%s' "$s" | cut -d'|' -f2)
    sg=$(printf '%s' "$s" | cut -d'|' -f3)
    [ "$ep" = "$sp" ] || { printf 'engine: %s\nport:   %s\n' "$ep" "$sp" > "$LOG/$name.prompt.diff"; fail "$name: prompt ids differ (see $LOG/$name.prompt.diff)"; }
    [ "$eg" = "$sg" ] || { printf 'engine: %s\nport:   %s\n' "$eg" "$sg" > "$LOG/$name.gen.diff"; fail "$name: generated ids differ (see $LOG/$name.gen.diff)"; }
    echo "  $name: ids identical ($(printf '%s' "$eg" | wc -w) generated)"
}

check anthropic 1
check responses 2

# ---- finish parity ----
es=$(sed -n 's/.*"stop_reason":"\([^"]*\)".*/\1/p' "$LOG/eng.anthropic.json")
ss=$(sed -n 's/.*"stop_reason":"\([^"]*\)".*/\1/p' "$LOG/srv.anthropic.json")
[ "$es" = "$ss" ] || fail "anthropic stop_reason differs (engine=$es port=$ss)"
[ "$es" = "max_tokens" ] && fail "anthropic response hit the token cap"
er=$(sed -n 's/.*"status":"\([^"]*\)".*/\1/p' "$LOG/eng.responses.json")
sr=$(sed -n 's/.*"status":"\([^"]*\)".*/\1/p' "$LOG/srv.responses.json")
[ "$er" = "$sr" ] || fail "responses status differs (engine=$er port=$sr)"
[ "$er" = "incomplete" ] && fail "responses response incomplete (token cap?)"

echo "ds41 wire gate: PASS (anthropic + responses: prompt ids, generated ids and finish reasons equal the engine's)"
