#!/usr/bin/env bash
# install.sh - deploy and start the ds4-dfm-rs V4.1 server (the port) on a
# DGX Spark: the port's own binary, built from THIS tree (`make ds4-server`),
# over the model artifact files (the GGUF, the n-gram shards, the domain
# sidecars) that live in --dir.
#
# THIS DEPLOYMENT RUNS ONLY ON THE NVIDIA DGX SPARK (owner rule 2026-10-10:
# "we will only run this in the nvidia-spark box").  The preflight below
# enforces it: aarch64 Linux, a GB10 GPU, the CUDA 13 runtime, 128 GB of
# memory - anything else is refused before a byte of the model is read.
#
# The reference engine (the YoungAi tree) is a porting reference only: this
# script never loads or starts its server - the process it starts is
# $REPO/ds4-server, built here.  (The test gates do boot the reference as
# their oracle, but only the SOURCE-BUILT one, per the owner's rule; see
# docs/deepseek41-port-plan.md 6.4.)
#
#   bash install.sh                  check the machine, check the artifact
#                                    files, build the port, start it on
#                                    0.0.0.0:8000 and run the same smoke
#                                    test the reference install.sh runs
#   bash install.sh --domain code    the same, serving with the coding sidecar
#                                    (domains: finance, code, law, medicine,
#                                    science, none; finance is the default)
#   bash install.sh --no-start       everything except starting the server
#   bash install.sh start | stop | status
#
# Options:
#   --dir DIR          where the artifact files live (default: $HOME/youngai:
#                      model/ + deepseek-engram/ - the download layout)
#   --domain NAME      which domain sidecar the server loads; the reference's
#                      own mapping (its install.sh sidecar_of, lines 41-48)
#   --posttrain        also load the experimental post-training file
#                      (finance only, the reference's own constraint)
#   --host ADDR        listen address (default 0.0.0.0)
#   --port N           listen port (default 8000)
#   --no-build         do not run `make ds4-server` even if the binary is stale
#   --no-start         install/check only
#
# The start contract is the reference install.sh's own, same constants: the
# START_FLOOR_MB 100000 check before loading, the KILL_MB 2500 watchdog (two
# consecutive 5 s samples below it stop the server), the /v1/models wait, the
# RUN_FLOOR_MB 10000 check after loading, and the smoke request.  The server
# line is the reference's own spelling, driven by the port's functions:
# --cuda -m <gguf> --zchain <sidecar> --engram-dir <dir> [--posttrain <dir>]
# --mem-budget-mb 110000 --host H --port P.
set -uo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
GGUF="DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf"
SHARDS=(model-00047-of-00048.safetensors model-00048-of-00048.safetensors)
SHARD_BYTES=(101535150936 101537926640)
POSTTRAIN="posttrain-experimental-20260924"
POSTTRAIN_DOMAIN="finance"
# Domain -> sidecar directory, verbatim from the engine's install.sh:41-48.
sidecar_of() {
    case "$1" in
        finance) echo "DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-vqfin41_vqhalf_a_n8192-engine";;
        code)    echo "DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-code_fit_n15360-engine";;
        law)     echo "DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-law_fit_n15360-engine";;
        medicine) echo "DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-med_fit_n15360-engine";;
        science) echo "DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-sci_fit_n15360-engine";;
        none)    echo "";;
        *)       return 1;;
    esac
}
DOMAINS="finance | code | law | medicine | science | none"
BUDGET_MB=110000        # engine install.sh BUDGET_MB
START_FLOOR_MB=100000   # engine install.sh START_FLOOR_MB
RUN_FLOOR_MB=10000      # engine install.sh RUN_FLOOR_MB
KILL_MB=2500            # engine install.sh KILL_MB

CMD="install"; DIR="$HOME/youngai"; HOST="0.0.0.0"; PORT=8000; DOMAIN="finance"
WITH_PT=0; START=1; BUILD=1
while [ $# -gt 0 ]; do
    case "$1" in
        install|start|stop|status) CMD="$1"; shift;;
        --dir) DIR="$2"; shift 2;;
        --domain) DOMAIN="$2"; shift 2;;
        --posttrain) WITH_PT=1; shift;;
        --host) HOST="$2"; shift 2;;
        --port) PORT="$2"; shift 2;;
        --no-build) BUILD=0; shift;;
        --no-start) START=0; shift;;
        -h|--help) awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"; exit 0;;
        *) echo "unknown argument: $1 (see --help)"; exit 2;;
    esac
done
MODEL="$DIR/model"; ENGRAM="$DIR/deepseek-engram"
RUN="$DIR/ds4-rs-run"; LOG="$RUN/server.log"; PIDF="$RUN/server.pid"
WDPIDF="$RUN/watchdog.pid"; INFOF="$RUN/server.info"

say() { echo "[ds4-rs $(date +%H:%M:%S)] $*"; }
die() { echo "[ds4-rs] ERROR: $*" >&2; exit 1; }
SIDECAR=$(sidecar_of "$DOMAIN") || die "unknown --domain '$DOMAIN' (choose $DOMAINS)"
[ "$WITH_PT" = 0 ] || [ "$DOMAIN" = "$POSTTRAIN_DOMAIN" ] \
    || die "--posttrain was solved on the $POSTTRAIN_DOMAIN sidecar and cannot be stacked on --domain $DOMAIN"
avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
fsize() { stat -L -c %s "$1" 2>/dev/null || echo 0; }
alive() { [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

preflight() {
    [ "$(uname -s)/$(uname -m)" = "Linux/aarch64" ] || die "this deployment ships aarch64 Linux binaries for DGX Spark; this machine is $(uname -s)/$(uname -m)"
    command -v nvidia-smi >/dev/null || die "nvidia-smi not found: no NVIDIA driver"
    local gpu; gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    case "$gpu" in *GB10*) ;; *) die "GPU is '$gpu'; the binary is compiled for GB10 (sm_121) only";; esac
    local ldc; ldc=$(command -v ldconfig || echo /sbin/ldconfig)
    for lib in libcudart.so.13 libcublas.so.13 libcublasLt.so.13; do
        "$ldc" -p | grep -q "$lib" || die "$lib not found: the CUDA 13 runtime is required (DGX OS ships it)"
    done
    local total; total=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo)
    [ "$total" -ge 115000 ] || die "only $total MB of memory; the model needs a 128 GB DGX Spark"
    say "machine ok: $gpu, CUDA 13 runtime, $total MB memory"
}

engram_ok() {   # both official shards in $ENGRAM, with the right sizes
    [ "$(fsize "$ENGRAM/${SHARDS[0]}")" = "${SHARD_BYTES[0]}" ] && [ "$(fsize "$ENGRAM/${SHARDS[1]}")" = "${SHARD_BYTES[1]}" ]
}

files_ok() {
    [ "$(fsize "$MODEL/$GGUF")" = "113556639424" ] \
        || die "the artifact is not in $MODEL: the deployment needs $GGUF + the n-gram shards + a domain sidecar there (or point --dir at where they are)"
    engram_ok || die "the n-gram tables are not in $ENGRAM (model-00047/48-of-00048.safetensors, the official sizes)"
    [ -z "$SIDECAR" ] || [ -s "$MODEL/$SIDECAR/manifest.txt" ] \
        || die "the $DOMAIN sidecar is not in $MODEL (need $SIDECAR)"
    [ "$WITH_PT" = 0 ] || [ -s "$MODEL/$POSTTRAIN/manifest.txt" ] \
        || die "the post-train directory is not in $MODEL (need $POSTTRAIN)"
    say "artifact files ok in $DIR (gguf + n-gram shards + ${DOMAIN} sidecar)"
}

build() {
    [ "$BUILD" = 1 ] || { [ -x "$REPO/ds4-server" ] || die "$REPO/ds4-server is missing (--no-build given)"; return 0; }
    say "building the port: make ds4-server"
    ( cd "$REPO" && make ds4-server ) || die "make ds4-server failed"
    [ -x "$REPO/ds4-server" ] || die "make ds4-server produced no binary"
}

stop() {
    [ -s "$WDPIDF" ] && kill "$(cat "$WDPIDF")" 2>/dev/null
    if alive; then
        kill "$(cat "$PIDF")"; local i
        for i in $(seq 1 20); do alive || break; sleep 1; done
        alive && kill -9 "$(cat "$PIDF")"
        say "stopped"
    else
        say "not running"
    fi
    rm -f "$PIDF" "$WDPIDF" "$INFOF"
}

status() {
    if alive; then say "running (pid $(cat "$PIDF")): $(cat "$INFOF" 2>/dev/null)"; else say "not running"; fi
    say "memory available: $(avail_mb) MB"
}

start() {
    alive && { say "already running (pid $(cat "$PIDF"), $(cat "$INFOF" 2>/dev/null)); stop it first to switch"; return; }
    files_ok
    local a; a=$(avail_mb)
    [ "$a" -ge "$START_FLOOR_MB" ] || die "only $a MB of memory available, need $START_FLOOR_MB: close other programs first"
    mkdir -p "$RUN"
    local zc=() pt=()
    [ -n "$SIDECAR" ] && zc=(--zchain "$MODEL/$SIDECAR")
    [ "$WITH_PT" = 1 ] && pt=(--posttrain "$MODEL/$POSTTRAIN")
    say "starting ./ds4-server on $HOST:$PORT, domain $DOMAIN (loading takes ~2 minutes; log: $LOG)"
    nohup "$REPO/ds4-server" --cuda -m "$MODEL/$GGUF" "${zc[@]}" --engram-dir "$ENGRAM" "${pt[@]}" \
        --mem-budget-mb "$BUDGET_MB" --host "$HOST" --port "$PORT" >"$LOG" 2>&1 </dev/null &
    echo $! >"$PIDF"; echo "domain $DOMAIN on $HOST:$PORT" >"$INFOF"
    # The engine install.sh's watchdog, inline in the child: two consecutive
    # 5 s samples below KILL_MB stop the server before the host stalls.
    nohup bash -c '
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
    ' ds4-rs-watchdog "$(cat "$PIDF")" "$KILL_MB" "$RUN/watchdog.log" >/dev/null 2>&1 </dev/null &
    echo $! >"$WDPIDF"
    local i
    for i in $(seq 1 180); do
        curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
        alive || { tail -8 "$LOG"; die "the server exited while loading (full log: $LOG)"; }
        sleep 5
    done
    curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null || { stop; die "the server did not come up in 15 minutes (log: $LOG)"; }
    a=$(avail_mb)
    [ "$a" -ge "$RUN_FLOOR_MB" ] || { stop; die "only $a MB left after loading (need $RUN_FLOOR_MB); the machine would stall"; }
    say "loaded; $a MB of memory left. smoke test:"
    local out q="In one sentence, what is a price-to-earnings ratio?"
    case "$DOMAIN" in          # one question from the served domain, the engine install.sh's own
        code)     q="In one sentence, what does git rebase do?";;
        law)      q="In one sentence, what is a statute of limitations?";;
        medicine) q="In one sentence, what is hypertension?";;
        science)  q="In one sentence, what is entropy in thermodynamics?";;
    esac
    out=$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"deepseek-chat\",\"temperature\":0,\"max_tokens\":48,\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}]}")
    printf '%s' "$out" | grep -q '"choices"' || { echo "$out" | head -c 400; stop; die "the smoke request failed"; }
    printf '%s\n' "$out" | sed -n 's/.*"content":"\([^"]*\)".*/  -> \1/p' | head -1
    say "ready: http://$HOST:$PORT/v1, domain $DOMAIN  (OpenAI: /v1/chat/completions /v1/completions /v1/responses | Anthropic: /v1/messages)"
    say "stop with: bash install.sh stop --dir $DIR"
}

case "$CMD" in
    install) preflight; files_ok; build
             say "installed in $REPO (binary: $REPO/ds4-server)"
             [ "$START" = 1 ] && start;;
    start)   preflight; files_ok; build; start;;
    stop)    stop;;
    status)  status;;
esac
