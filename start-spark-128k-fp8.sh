#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="${1:-${MODEL_PATH:-/home/leandro/models/spark-x2.6-4b-vq-gguf}}"
CTX="${CTX:-131072}"
PORT="${PORT:-8099}"
HOST="${HOST:-0.0.0.0}"

# Full FP8 all the way for Spark-X2.6-4B VQ:
#   - DS4_SPARK2_5_WEIGHTS=fp8 : Restores all 180 VQ matrices, all 36 g_proj
#     matrices, all 73 RMSNorm vectors, and the tied token_embd/output LM head
#     into a single 3.89 GiB GPU-resident FP8 E4M3 + block-128 scale image.
#   - DS4_SPARK2_5_FULL_KV=fp8 : Keeps all 36 layers (both the 9 full-context
#     layers and the 27 sliding-window layers) in FP8 E4M3 K|V|scales format.
#   - DS4_CUDA_FP8_KV=1        : Global CUDA FP8 KV cache flag.
export DS4_SPARK2_5_WEIGHTS=fp8
export DS4_SPARK2_5_FULL_KV=fp8
export DS4_CUDA_FP8_KV=1
export DS4_LOCK_FILE="${DS4_LOCK_FILE:-/tmp/ds4-spark-${PORT}.lock}"

exec ./ds4-server \
    --cuda \
    -m "$MODEL_PATH" \
    -c "$CTX" \
    --host "$HOST" \
    --port "$PORT" \
    --mem-floor-gb 1 \
    --max-seqs 1 \
    --model-id spark-x2.6-4b-vq-2s \
    "$@"
