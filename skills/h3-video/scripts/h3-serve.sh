#!/usr/bin/env bash
# Start the local MiniMax-H3 video server (h3-server-rs) with the in-process
# engine. Every flag is one of the binary's own (crates/ds4-server/src/bin/
# h3-server-rs.rs:43-80); this script only supplies defaults.
#
#   H3_BIN        binary to run        (default: h3-server-rs on PATH)
#   H3_MODEL_DIR  checkpoint root      (default: /mnt/h3-artifacts/MiniMax-H3)
#   H3_STEPS      denoise evaluations  (default: 20; use 2 for a smoke test)
#   H3_BIND       listen address       (default: 127.0.0.1:8090)
#   H3_OUT_DIR    MP4 directory        (default: h3-videos)
#   H3_TIMEOUT    per-render seconds   (default: 3600)
#
# Extra arguments are passed through, e.g. --max-jobs 2.
set -euo pipefail

BIN=${H3_BIN:-h3-server-rs}
MODEL_DIR=${H3_MODEL_DIR:-/mnt/h3-artifacts/MiniMax-H3}
STEPS=${H3_STEPS:-20}
BIND=${H3_BIND:-127.0.0.1:8090}
OUT_DIR=${H3_OUT_DIR:-h3-videos}
TIMEOUT=${H3_TIMEOUT:-3600}

if ! command -v "$BIN" >/dev/null 2>&1; then
  echo "h3-serve: $BIN is not on PATH; set H3_BIN or build it with make h3-server-rs" >&2
  exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "h3-serve: no checkpoint at $MODEL_DIR; set H3_MODEL_DIR" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "h3-serve: ffmpeg is not on PATH; the engine muxes through it" >&2
  exit 1
fi

echo "h3-serve: $BIN on http://$BIND, model $MODEL_DIR, $STEPS steps, out $OUT_DIR" >&2
exec "$BIN" \
  --bind "$BIND" \
  --out-dir "$OUT_DIR" \
  --model-dir "$MODEL_DIR" \
  --steps "$STEPS" \
  --timeout-secs "$TIMEOUT" \
  "$@"
