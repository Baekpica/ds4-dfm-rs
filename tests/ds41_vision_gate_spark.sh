#!/usr/bin/env bash
# ds41 vision-tower acceptance gate on the DGX Spark.  Port of the engine's
# gguf-tools/scripts/vision_gate_spark.sh (engine 846c929c5), translated to
# English.  Each required check's raw output lands in $OUT:
#
#   1. golden gate: ./ds4 --image <img> --vision-golden <golden> runs the
#      engine's four comparisons (the port's reader is
#      crates/ds4-core/src/v41_golden.rs: plan account / pixels / patches /
#      tower rows).  The port requires all three image/golden pairs.
#   2. text invariance: the same prompt at temp 0 greedy 64 tokens, with and
#      without --vision-gguf, must be byte-identical (the sidecar must not
#      touch the text path).  The engine's step runs -p directly; the port's
#      ds41 CLI runs on --gen-ids (its chat rendering is not ported), so the
#      prompt is tokenized once with --dump-tokens (vocab only, no GPU) and
#      both runs share the ids file.
#   3. see an image: the port's image entry is the server (the CLI one-shot
#      image request is not ported), so this step starts ./ds4-server with
#      the sidecar on a spare port, posts one OpenAI image request per
#      fixture, prints each answer as-is (judged by a human: always show the
#      raw output), and stops the server again.
#   4. no leftover processes: every run is a one-shot ./ds4 (or the one
#      server started and stopped here); the production model process is
#      stopped by the caller before the window (this script refuses to run
#      while one is up).
#   5. the 400 without the sidecar (port addition for the record): the server
#      started without --vision-gguf answers a ds41 image request with HTTP
#      400 and the message that names the sidecar.
#   6. the admission line (port addition for the record): a prefill whose 2x
#      peak exceeds MemAvailable is refused with the serving admission's
#      line (v41_run.rs:1875-1889, the engine's server_sched_v41.c:58), not
#      attempted.
#
# usage: ds41_vision_gate_spark.sh [main gguf] [vision gguf] [zchain dir]
# Memory: every model run loads the ~110 GB model once (~100 s); the script
# requires MemAvailable >= 100 GB (a running production server fails this
# check, as in the engine's script).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

MODEL_DIR=${MODEL_DIR:-$HOME/youngai/model}
MAIN=${1:-$MODEL_DIR/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf}
VIS=${2:-$MODEL_DIR/DeepSeek-V4.1-Flash.vision.gguf}
ZCH=${3:-$MODEL_DIR/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-code_fit_n15360-engine}
ENGRAM_DIR=${ENGRAM_DIR:-$HOME/youngai/deepseek-engram}
BIN=${BIN:-$ROOT/ds4}
SERVER=${SERVER:-$ROOT/ds4-server}
IMG_DIR=${IMG_DIR:-$ROOT/tests/fixtures/vision}
GOLD_DIR=${GOLD_DIR:-$IMG_DIR}
SEE_PORT=${SEE_PORT:-8099}
BUD=110000
OUT=${OUT:-$ROOT/docs/benchmarks/ds41-vision-2026-10-13/gate_$(date +%m%d_%H%M)}
mkdir -p "$OUT"

die() { echo "GATE FAIL: $*" | tee -a "$OUT/gate.log"; exit 1; }
log() { echo "$*" | tee -a "$OUT/gate.log"; }

wait_server() {  # $1 pid, $2 log: wait for /v1/models, die if it exits or never comes up
    local i
    for i in $(seq 1 180); do
        kill -0 "$1" 2>/dev/null || die "the server exited while loading (see $2)"
        curl -sf -m 3 "http://127.0.0.1:$SEE_PORT/v1/models" >/dev/null && return 0
        sleep 5
    done
    die "the server did not come up (see $2)"
}

stop_server() {
    if [ -n "${SEE_PID:-}" ]; then
        kill "$SEE_PID" 2>/dev/null || true
        wait "$SEE_PID" 2>/dev/null || true
        SEE_PID=
    fi
}

[ -f "$MAIN" ] || die "main GGUF missing: $MAIN"
[ -f "$VIS" ] || die "vision sidecar missing: $VIS"
[ -d "$ZCH" ] || die "zchain dir missing: $ZCH"
[ -x "$BIN" ] || die "port binary missing: $BIN (run 'make ds4')"
[ -x "$SERVER" ] || die "port server missing: $SERVER (run 'make ds4-server')"

# Validate every fixture before any model load; a partial run cannot pass.
FIXTURES=(panel:review-market-panel.png:golden_panel.bin bug:bug.jpeg:golden_bug.bin homer:homer.png:golden_homer.bin)
for g in "${FIXTURES[@]}"; do
    n=${g%%:*}; rest=${g#*:}; img=${rest%%:*}; gold=${rest#*:}
    [ -f "$IMG_DIR/$img" ] || die "$n image missing: $IMG_DIR/$img"
    [ -f "$GOLD_DIR/$gold" ] || die "$n golden missing: $GOLD_DIR/$gold"
done

# The engine's preflight: a live production model subprocess refuses the run
# (the caller stops it: the port's ds41_install.sh stop), and the box must have
# room for the one model copy this gate loads.
if pgrep -f 'ds4-serve[r]' >/dev/null; then
    die "a production ds4-server is running; stop it first (bash ds41_install.sh stop --dir <dir>)"
fi
AV=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
[ "$AV" -ge 100000 ] || die "MemAvailable ${AV} MB < 100000; not enough room for the model"

# The engine's script runs without --engram-dir (its machine resolved the
# GGUF's table paths); the port passes the install layout's engram dir when
# it exists, so the n-gram tables resolve on any Spark.
ZC=()
[ -d "$ENGRAM_DIR" ] && ZC=(--engram-dir "$ENGRAM_DIR")

log "[gate] main $MAIN  sidecar $VIS  zchain $ZCH  out $OUT"
COMMON=(--cuda -m "$MAIN" --zchain "$ZCH" --mem-budget-mb "$BUD" --no-dspark "${ZC[@]}")

# ---- 1. golden gate ----
# name:image:golden triples: the engine's own panel fixture, the engine tree's
# jpeg, and the owner-provided homer.png.  The engine's script used img.jpeg
# (the engine author's own image); bug.jpeg is this repo's jpeg fixture.
for g in "${FIXTURES[@]}"; do
    n=${g%%:*}; rest=${g#*:}; img=${rest%%:*}; gold=${rest#*:}
    log "== 1. golden gate $n"
    "$BIN" "${COMMON[@]}" --vision-gguf "$VIS" --image "$IMG_DIR/$img" --vision-golden "$GOLD_DIR/$gold" \
        > "$OUT/golden_$n.txt" 2>&1 \
        || die "golden $n failed (exit $?; see $OUT/golden_$n.txt)"
    log "step 1 $n: exit 0"
    grep -E '\[v41-golden\]|vision-golden' "$OUT/golden_$n.txt" | tee -a "$OUT/gate.log"
done

# ---- 2. text invariance ----
P="Explain in three sentences what marginal cost is."
log "== 2. text invariance (temp 0, 64 tokens)"
IDS="$OUT/text_ids.txt"
"$BIN" --dump-tokens -m "$MAIN" -p "$P" > "$OUT/text_ids.dump" 2>&1 \
    || die "dump-tokens failed (see $OUT/text_ids.dump)"
head -1 "$OUT/text_ids.dump" | tr -d '[]' | tr ',' ' ' > "$IDS"
[ -s "$IDS" ] || die "no token ids in $OUT/text_ids.dump"
"$BIN" "${COMMON[@]}" --gen-ids "$IDS" -n 64 --temp 0 > "$OUT/text_base.txt" 2> "$OUT/text_base.err" \
    || die "base text generation failed (see $OUT/text_base.err)"
"$BIN" "${COMMON[@]}" --vision-gguf "$VIS" --gen-ids "$IDS" -n 64 --temp 0 > "$OUT/text_vis.txt" 2> "$OUT/text_vis.err" \
    || die "sidecar text generation failed (see $OUT/text_vis.err)"
if cmp -s "$OUT/text_base.txt" "$OUT/text_vis.txt"; then
    log "text byte-identical: yes"
else
    log "text byte-identical: NO (the sidecar touched the text path)"
    diff "$OUT/text_base.txt" "$OUT/text_vis.txt" > "$OUT/text.diff" || true
    head -5 "$OUT/text.diff" | tee -a "$OUT/gate.log"
    die "text invariance failed (see $OUT/text.diff)"
fi
log "--- text as-is: $(head -c 300 "$OUT/text_vis.txt")"

# ---- 3. see an image (the port's image entry: the server) ----
log "== 3. see an image (server on 127.0.0.1:$SEE_PORT)"
SEE_PID=
trap stop_server EXIT
"$SERVER" --cuda -m "$MAIN" --zchain "$ZCH" "${ZC[@]}" --vision-gguf "$VIS" \
    --mem-budget-mb "$BUD" --host 127.0.0.1 --port "$SEE_PORT" > "$OUT/see_server.log" 2>&1 &
SEE_PID=$!
wait_server "$SEE_PID" "$OUT/see_server.log"

for g in "${FIXTURES[@]}"; do
    rest=${g#*:}; img=${rest%%:*}
    log "== 3. see $img"
    python3 - "$SEE_PORT" "$IMG_DIR/$img" "$OUT/see_${img%.*}.txt" 2>&1 <<'PY' | tee -a "$OUT/gate.log" || die "image request failed: $img"
import base64, json, sys, urllib.request
port, img, out = sys.argv[1], sys.argv[2], sys.argv[3]
base = f"http://127.0.0.1:{port}"
mid = json.load(urllib.request.urlopen(base + "/v1/models", timeout=30))["data"][0]["id"]
mime = "image/jpeg" if img.lower().endswith((".jpg", ".jpeg")) else "image/png"
b64 = base64.b64encode(open(img, "rb").read()).decode()
body = json.dumps({
    "model": mid, "temperature": 0, "max_tokens": 200,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What is in this image? List all the text and numbers you can see."},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]}],
}).encode()
req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
resp = json.load(urllib.request.urlopen(req, timeout=300))
answer = resp["choices"][0]["message"]["content"]
if not isinstance(answer, str) or not answer.strip():
    raise SystemExit("image request returned no answer")
usage = resp.get("usage", {})
open(out, "w").write(answer + "\n")
print(f"[see] {img}: prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}")
PY
    log "--- answer as-is:"
    cat "$OUT/see_${img%.*}.txt" | tee -a "$OUT/gate.log"
    echo | tee -a "$OUT/gate.log"
done
stop_server

# ---- 5. 400 without the sidecar (port addition for the record) ----
log "== 5. 400 without the sidecar (server on 127.0.0.1:$SEE_PORT)"
"$SERVER" --cuda -m "$MAIN" --zchain "$ZCH" "${ZC[@]}" \
    --mem-budget-mb "$BUD" --host 127.0.0.1 --port "$SEE_PORT" > "$OUT/no_sidecar_server.log" 2>&1 &
SEE_PID=$!
wait_server "$SEE_PID" "$OUT/no_sidecar_server.log"
python3 - "$SEE_PORT" "$IMG_DIR/review-market-panel.png" 2>&1 <<'PY' | tee -a "$OUT/gate.log" || die "missing-sidecar rejection failed"
import base64, json, sys, urllib.error, urllib.request
from http import HTTPStatus
port, img = sys.argv[1], sys.argv[2]
base = f"http://127.0.0.1:{port}"
mid = json.load(urllib.request.urlopen(base + "/v1/models", timeout=30))["data"][0]["id"]
b64 = base64.b64encode(open(img, "rb").read()).decode()
body = json.dumps({
    "model": mid, "temperature": 0, "max_tokens": 16,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
    ]}],
}).encode()
req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
try:
    resp = urllib.request.urlopen(req, timeout=120)
    print(f"step 5: HTTP {resp.status} (expected 400)")
    print(resp.read().decode()[:300])
    raise SystemExit("image request succeeded without the vision sidecar")
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"step 5: HTTP {e.code}")
    print(body[:300])
    if e.code != HTTPStatus.BAD_REQUEST or "--vision-gguf" not in body:
        raise SystemExit("expected HTTP 400 naming --vision-gguf")
PY
stop_server

# ---- 6. the admission line (port addition for the record) ----
# A prefill whose 2x peak exceeds MemAvailable is refused with the serving
# admission's line (v41_run.rs:1875-1889, the engine's server_sched_v41.c:58),
# not attempted.  200k ids make the peak far exceed what is left once the
# model is resident.
log "== 6. admission line (200k-token prefill)"
python3 -c "print(' '.join(['100'] * 200000))" > "$OUT/admission_ids.txt"
if "$BIN" "${COMMON[@]}" --gen-ids "$OUT/admission_ids.txt" -n 1 --temp 0 > "$OUT/admission.txt" 2>&1; then
    die "admission prefill unexpectedly succeeded (see $OUT/admission.txt)"
else
    log "step 6: exit $?"
fi
grep -F 'prefill needs' "$OUT/admission.txt" | tee -a "$OUT/gate.log" \
    || die "expected prefill admission refusal (see $OUT/admission.txt)"

log "[gate] done, artifacts in $OUT"
