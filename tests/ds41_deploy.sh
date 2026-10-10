#!/bin/sh
# ds41 deploy launcher - install.sh's start() contract, ported to ./ds4-server.
#
# install.sh (the engine's own deployment recipe,
# huggingface.co/wenzhouwu/YoungAi-DeepSeek-V4.1-Flash) starts the engine as
#   ds4-server --cuda -m <gguf> --zchain <sidecar> --engram-dir <dir>
#              [--posttrain <dir>] --mem-budget-mb 110000 --host H --port P
# under two shell guardrails (a START_FLOOR_MB check before loading, and a
# KILL_MB watchdog whose two consecutive 5 s samples below the floor stop the
# server) and finishes with a smoke request.  This script is that same
# sequence for the port's server, with install.sh's own constants
# (BUDGET_MB 110000, START_FLOOR_MB 100000, RUN_FLOOR_MB 10000, KILL_MB 2500)
# and the same smoke (model "deepseek-chat", temp 0, the finance question).
# The server-side functions it drives are the port's own: the --zchain/--
# posttrain mount (zchain.rs + the P4-7 stores), --engram-dir, the
# --mem-budget-mb L1 gate (ds4_l1_budget_gate, core_util.c:236-256).
#
# usage: ds41_deploy.sh start <gguf> <sidecar_dir> <engram_dir>
#                        [--posttrain <dir>] [--port N] [--host H]
#        ds41_deploy.sh stop | status
#
# Env: DS41_SERVER (default ./ds4-server), DS41_DEPLOY_DIR (default /tmp/ds41-deploy).
set -u

SRV=${DS41_SERVER:-./ds4-server}
DIR=${DS41_DEPLOY_DIR:-/tmp/ds41-deploy}
LOG=$DIR/server.log
PIDF=$DIR/server.pid
WDPIDF=$DIR/watchdog.pid
INFOF=$DIR/server.info

BUDGET_MB=110000        # install.sh BUDGET_MB
START_FLOOR_MB=100000   # install.sh START_FLOOR_MB
RUN_FLOOR_MB=10000      # install.sh RUN_FLOOR_MB
KILL_MB=2500            # install.sh KILL_MB

avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
alive() { [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

stop() {
    [ -s "$WDPIDF" ] && kill "$(cat "$WDPIDF")" 2>/dev/null
    if alive; then
        kill "$(cat "$PIDF")"; i=0
        while [ $i -lt 20 ] && alive; do sleep 1; i=$((i + 1)); done
        alive && kill -9 "$(cat "$PIDF")"
        echo "ds41_deploy: stopped"
    else
        echo "ds41_deploy: not running"
    fi
    rm -f "$PIDF" "$WDPIDF" "$INFOF"
}

status() {
    if alive; then
        echo "ds41_deploy: running (pid $(cat "$PIDF")): $(cat "$INFOF" 2>/dev/null)"
    else
        echo "ds41_deploy: not running"
    fi
    echo "ds41_deploy: memory available $(avail_mb) MB"
}

case "${1:-}" in
stop) stop; exit 0;;
status) status; exit 0;;
start) shift;;
*) echo "usage: ds41_deploy.sh start <gguf> <sidecar_dir> <engram_dir> [--posttrain <dir>] [--port N] [--host H] | stop | status"; exit 2;;
esac

MODEL=${1:-}; SIDECAR=${2:-}; ENGRAM=${3:-}
[ -n "$MODEL" ] && [ -n "$SIDECAR" ] && [ -n "$ENGRAM" ] || { echo "ds41_deploy: need <gguf> <sidecar_dir> <engram_dir>"; exit 2; }
shift 3
PT=""; PORT=8000; HOST=127.0.0.1
while [ $# -gt 0 ]; do
    case "$1" in
    --posttrain) PT=${2:?--posttrain needs a value}; shift 2;;
    --port) PORT=${2:?--port needs a value}; shift 2;;
    --host) HOST=${2:?--host needs a value}; shift 2;;
    *) echo "ds41_deploy: unknown argument $1"; exit 2;;
    esac
done

alive && { echo "ds41_deploy: already running (pid $(cat "$PIDF"), $(cat "$INFOF" 2>/dev/null)); stop it first"; exit 1; }
[ -x "$SRV" ] || { echo "ds41_deploy: $SRV is not executable"; exit 1; }
a=$(avail_mb)
[ "$a" -ge "$START_FLOOR_MB" ] || { echo "ds41_deploy: only $a MB available, need $START_FLOOR_MB (install.sh START_FLOOR_MB)"; exit 1; }
mkdir -p "$DIR"
if [ -n "$PT" ]; then
    nohup "$SRV" --cuda -m "$MODEL" --zchain "$SIDECAR" --engram-dir "$ENGRAM" --posttrain "$PT" \
        --mem-budget-mb "$BUDGET_MB" --host "$HOST" --port "$PORT" >"$LOG" 2>&1 </dev/null &
else
    nohup "$SRV" --cuda -m "$MODEL" --zchain "$SIDECAR" --engram-dir "$ENGRAM" \
        --mem-budget-mb "$BUDGET_MB" --host "$HOST" --port "$PORT" >"$LOG" 2>&1 </dev/null &
fi
echo $! >"$PIDF"
echo "sidecar $SIDECAR on $HOST:$PORT" >"$INFOF"
# install.sh's watchdog(), inline in the child: two consecutive 5 s samples
# below KILL_MB stop the server before the host stalls.
nohup sh -c '
    pid="$1"; floor="$2"; log="$3"; bad=0
    while kill -0 "$pid" 2>/dev/null; do
        a=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)
        if [ "$a" -lt "$floor" ]; then bad=$((bad + 1)); else bad=0; fi
        if [ "$bad" -ge 2 ]; then
            echo "$(date) MemAvailable ${a} MB < $floor MB twice, stopping the server" >>"$log"
            kill "$pid"; exit 0
        fi
        sleep 5
    done
' sh "$(cat "$PIDF")" "$KILL_MB" "$DIR/watchdog.log" >/dev/null 2>&1 </dev/null &
echo $! >"$WDPIDF"
echo "ds41_deploy: starting on $HOST:$PORT (loading takes ~2 minutes; log: $LOG)"
i=0
while [ $i -lt 180 ]; do
    curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
    alive || { tail -8 "$LOG"; echo "ds41_deploy: the server exited while loading (log: $LOG)"; exit 1; }
    sleep 5; i=$((i + 1))
done
curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null || { stop; echo "ds41_deploy: the server did not come up in 15 minutes (log: $LOG)"; exit 1; }
a=$(avail_mb)
[ "$a" -ge "$RUN_FLOOR_MB" ] || { stop; echo "ds41_deploy: only $a MB left after loading (need $RUN_FLOOR_MB)"; exit 1; }
echo "ds41_deploy: loaded; $a MB of memory left. smoke test:"
out=$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-chat","temperature":0,"max_tokens":48,"messages":[{"role":"user","content":"In one sentence, what is a price-to-earnings ratio?"}]}')
printf '%s' "$out" | grep -q '"choices"' || { echo "$out" | head -c 400; stop; echo "ds41_deploy: the smoke request failed"; exit 1; }
printf '%s\n' "$out" | sed -n 's/.*"content":"\([^"]*\)".*/  -> \1/p' | head -1
echo "ds41_deploy: ready: http://$HOST:$PORT/v1 (OpenAI: /v1/chat/completions /v1/completions /v1/responses | Anthropic: /v1/messages)"
echo "ds41_deploy: stop with: $0 stop"
