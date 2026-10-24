#!/bin/sh
# Build the MiniMax-H3 VQ converter toolchain.
#
# CPU-only: no CUDA, no checkpoint, no GPU needed to build or to encode.
# The sources are the ones the embeddinggemma2 dense bake used, copied
# verbatim; only the paths differ. See README.md for provenance.
#
# The command lines below are the bake's own build and its RECIPE.md section 4,
# with the include roots pointed at this directory.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
YA="$HERE/youngai"
OUT=${OUT:-$HERE/bin}
CC=${CC:-cc}
CFLAGS="-O3 -std=c99 -Wall -D_POSIX_C_SOURCE=200809L"
INC="-I$YA/gguf-tools/quantize -I$YA/src/common -I$HERE/bake"

mkdir -p "$OUT"

echo "== H3 driver: plan / encode / verify / merge / check =="
# shellcheck disable=SC2086
$CC $CFLAGS $INC -o "$OUT/h3_pack_cpu" "$HERE/bake/h3_pack.c" \
    "$YA/gguf-tools/quantize/quants.c" \
    "$YA/gguf-tools/quantize/onebit_quant.c" \
    "$YA/src/common/ds4_quantfmt.c" -lm -lpthread

echo "== the unmodified bake driver, kept for comparison =="
# shellcheck disable=SC2086
$CC $CFLAGS $INC -o "$OUT/eg2_pack_cpu" "$HERE/bake/eg2_pack.c" \
    "$YA/gguf-tools/quantize/quants.c" \
    "$YA/gguf-tools/quantize/onebit_quant.c" \
    "$YA/src/common/ds4_quantfmt.c" -lm -lpthread

echo "== GGUF wrap of the container =="
# shellcheck disable=SC2086
$CC $CFLAGS $INC -o "$OUT/h3_gguf" "$HERE/bake/eg2_gguf.c" \
    "$YA/src/common/ds4_quantfmt.c" -lm

echo "== single-tensor ladder tool =="
# eg2_vq.c already carries the block-quantizer sources it needs, so it links
# without onebit_quant.c; adding that object is a duplicate-symbol error.
# shellcheck disable=SC2086
$CC $CFLAGS $INC -o "$OUT/h3_vq" "$HERE/bake/eg2_vq.c" \
    "$YA/src/common/ds4_quantfmt.c" -lm -lpthread

echo "built into $OUT"
ls -la "$OUT"
