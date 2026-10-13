#!/usr/bin/env bash
# Capture the DeepSeek V4.1 golden set from the shipped engine binaries.
#
# P0 of docs/deepseek41-port-plan.md: the C engine's own behaviour on a fixed
# prompt set at temperature 0 is the port's correctness oracle, and it has to
# be frozen before any forward work starts. Runs on the DGX Spark, where the
# artifact lives.
#
# Per prompt: tokenize with --dump-tokens (vocab-only, no GPU), then
# --score-ids over those exact ids, which writes the per-position logits
# (S x V f32) and prints the segment PPL. Ids are the exact instrument: the
# engine's own comment in cli_diag.c says re-tokenized text does not round
# trip (79 vs 75 tokens measured 2026-09-20).
#
#   bash tests/capture_ds41_golden.sh              # with the engram tables
#   NO_ENGRAM=1 bash tests/capture_ds41_golden.sh  # fixture variant, no engram
#   NO_ZCHAIN=1 bash tests/capture_ds41_golden.sh  # bare variant (no --zchain) -> $OUT/bare
#
# Output: $OUT/{<name>.ids,<name>.logits.bin,<name>.log,MANIFEST}
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-$HOME/youngai/model}
GGUF=${GGUF:-$MODEL_DIR/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf}
SIDECAR=${SIDECAR:-$MODEL_DIR/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-vqfin41_vqhalf_a_n8192-engine}
BIN=${BIN:-$MODEL_DIR/bin/ds4}
OUT=${OUT:-$MODEL_DIR/golden}
# The two official n-gram shards, installed by install.sh. The deployed path
# uses them (the README's server and CLI examples both pass --engram-dir);
# NO_ENGRAM=1 captures the fixture variant instead.
ENGRAM_DIR=${ENGRAM_DIR:-$HOME/youngai/deepseek-engram}
ENGRAM_ARGS=()
if [ "${NO_ENGRAM:-0}" = "1" ]; then
  ENGRAM_ARGS=(--v41-no-engram)
  OUT="$OUT/no-engram"
elif [ -d "$ENGRAM_DIR" ]; then
  ENGRAM_ARGS=(--engram-dir "$ENGRAM_DIR")
else
  echo "engram tables not found at $ENGRAM_DIR; use NO_ENGRAM=1 for the fixture variant" >&2
  exit 1
fi

# Sidecar axis: the default capture ran the engine WITH --zchain (the gr/rb
# repair).  The port's forward does not apply the sidecar yet, so its trace
# gate needs the BARE variant (same ids, same engram, no --zchain):
#   NO_ZCHAIN=1 bash tests/capture_ds41_golden.sh   # -> $OUT/bare
ZCHAIN_ARGS=(--zchain "$SIDECAR")
if [ "${NO_ZCHAIN:-0}" = "1" ]; then
  ZCHAIN_ARGS=()
  OUT="$OUT/bare"
fi

# Fixed prompt set: short English, code, factual, a longer passage, Chinese.
NAMES=(p1 p2 p3 p4 p5)
PROMPTS=(
  "Explain the price-to-earnings ratio."
  "Write a Python function that returns the nth Fibonacci number."
  "What is the capital of France?"
  "The quick brown fox jumps over the lazy dog. The dog sleeps. A fox runs."
  "用一句话解释什么是市盈率。"
)

cd "$MODEL_DIR"
mkdir -p "$OUT"

for i in "${!NAMES[@]}"; do
  name=${NAMES[$i]}
  prompt=${PROMPTS[$i]}
  echo "=== $name: $prompt"

  # Tokenize (vocab only; no GPU, no model load).
  ids=$("$BIN" -m "$GGUF" -p "$prompt" --dump-tokens 2>/dev/null | head -1 | tr -d '[],')
  if [ -z "$ids" ]; then
    echo "tokenize failed for $name" >&2
    exit 1
  fi
  echo "$ids" > "$OUT/$name.ids"

  # Per-position logits for the exact id sequence.
  "$BIN" --cuda -m "$GGUF" "${ZCHAIN_ARGS[@]}" "${ENGRAM_ARGS[@]}" \
      --score-ids "$OUT/$name.ids" --score-out "$OUT/$name.logits.bin" \
      2>&1 | tee "$OUT/$name.log"
done

# Manifest: every captured file, plus the inputs that pin the run.
{
  echo "# DeepSeek V4.1 golden set, captured by tests/capture_ds41_golden.sh"
  echo "# engine: $BIN"
  echo "# engram: ${ENGRAM_ARGS[*]:-none}"
  echo "# zchain: ${ZCHAIN_ARGS[*]:-none}"
  sha256sum "$BIN" "$GGUF"
  sha256sum "$OUT"/*.ids "$OUT"/*.logits.bin "$OUT"/*.log
} > "$OUT/MANIFEST"

echo "=== golden set in $OUT"
cat "$OUT/MANIFEST"
