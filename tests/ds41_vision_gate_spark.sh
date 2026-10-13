#!/usr/bin/env bash
# ds41 vision-tower acceptance gate on the DGX Spark.  Port of the engine's
# gguf-tools/scripts/vision_gate_spark.sh (engine 846c929c5), translated to
# English.  Four steps, each step's raw output lands in $OUT:
#
#   1. golden gate: ./ds4 --image <img> --vision-golden <golden> runs the
#      engine's four comparisons (the port's reader is
#      crates/ds4-core/src/v41_golden.rs: plan account / pixels / patches /
#      tower rows).  A fixture without a golden is skipped, as in the engine.
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
#
# usage: ds41_vision_gate_spark.sh [main gguf] [vision gguf] [zchain dir]
# Memory: every model run loads the ~110 GB model once (~100 s); the script
# requires MemAvailable >= 100 GB (a running production server fails this
# check, as in the engine's script).
set -uo pipefail

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

[ -f "$MAIN" ] || die "main GGUF missing: $MAIN"
[ -f "$VIS" ] || die "vision sidecar missing: $VIS"
[ -d "$ZCH" ] || die "zchain dir missing: $ZCH"
[ -x "$BIN" ] || die "port binary missing: $BIN (run 'make ds4')"
[ -x "$SERVER" ] || die "port server missing: $SERVER (run 'make ds4-server')"

# The engine's preflight: a live production model subprocess refuses the run
# (the caller stops it: the port's install.sh stop), and the box must have
# room for the one model copy this gate loads.
if pgrep -f 'ds4-serve[r]' >/dev/null; then
    die "a production ds4-server is running; stop it first (bash install.sh stop --dir <dir>)"
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
for g in panel:review-market-panel.png:golden_panel.bin bug:bug.jpeg:golden_bug.bin homer:homer.png:golden_homer.bin; do
    n=${g%%:*}; rest=${g#*:}; img=${rest%%:*}; gold=${rest#*:}
    [ -f "$IMG_DIR/$img" ] || { log "skip $n: no image $IMG_DIR/$img"; continue; }
    [ -f "$GOLD_DIR/$gold" ] || { log "skip $n: no golden $GOLD_DIR/$gold"; continue; }
    log "== 1. golden gate $n"
    "$BIN" "${COMMON[@]}" --vision-gguf "$VIS" --image "$IMG_DIR/$img" --vision-golden "$GOLD_DIR/$gold" \
        > "$OUT/golden_$n.txt" 2>&1
    rc=$?
    log "step 1 $n: exit $rc"
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
"$BIN" "${COMMON[@]}" --gen-ids "$IDS" -n 64 --temp 0 > "$OUT/text_base.txt" 2> "$OUT/text_base.err"
"$BIN" "${COMMON[@]}" --vision-gguf "$VIS" --gen-ids "$IDS" -n 64 --temp 0 > "$OUT/text_vis.txt" 2> "$OUT/text_vis.err"
if cmp -s "$OUT/text_base.txt" "$OUT/text_vis.txt"; then
    log "text byte-identical: yes"
else
    log "text byte-identical: NO (the sidecar touched the text path)"
    diff "$OUT/text_base.txt" "$OUT/text_vis.txt" | head -5 | tee -a "$OUT/gate.log"
fi
log "--- text as-is: $(head -c 300 "$OUT/text_vis.txt")"

# ---- 3. see an image (the port's image entry: the server) ----
log "== 3. see an image (server on 127.0.0.1:$SEE_PORT)"
"$SERVER" --cuda -m "$MAIN" --zchain "$ZCH" "${ZC[@]}" --vision-gguf "$VIS" \
    --mem-budget-mb "$BUD" --host 127.0.0.1 --port "$SEE_PORT" > "$OUT/see_server.log" 2>&1 &
SEE_PID=$!
trap 'kill "$SEE_PID" 2>/dev/null' EXIT
for i in $(seq 1 180); do
    curl -sf -m 3 "http://127.0.0.1:$SEE_PORT/v1/models" >/dev/null && break
    kill -0 "$SEE_PID" 2>/dev/null || die "the server exited while loading (see $OUT/see_server.log)"
    sleep 5
done
curl -sf -m 3 "http://127.0.0.1:$SEE_PORT/v1/models" >/dev/null || die "the server did not come up (see $OUT/see_server.log)"

for img in review-market-panel.png bug.jpeg homer.png; do
    [ -f "$IMG_DIR/$img" ] || continue
    log "== 3. see $img"
    python3 - "$SEE_PORT" "$IMG_DIR/$img" "$OUT/see_${img%.*}.txt" <<'PY'
import base64, json, sys, urllib.request
port, img, out = sys.argv[1], sys.argv[2], sys.argv[3]
base = f"http://127.0.0.1:{port}"
mid = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]
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
usage = resp.get("usage", {})
open(out, "w").write(answer + "\n")
print(f"[see] {img}: prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}")
PY
    log "--- answer as-is:"
    cat "$OUT/see_${img%.*}.txt" | tee -a "$OUT/gate.log"
    echo | tee -a "$OUT/gate.log"
done
kill "$SEE_PID" 2>/dev/null
wait "$SEE_PID" 2>/dev/null
trap - EXIT

log "[gate] done, artifacts in $OUT"
