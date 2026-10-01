#!/usr/bin/env bash
# Default checks are model-free; --cuda adds GPU parity, --full adds make test.
# Bonsai artifact gates remain explicit: test-qwen35-rows, bonsai-fold-selftest,
# and bonsai-ref-check need DS4_BONSAI_MODEL.
# Set DS4_FAST=1 to skip the Rust/C catalogue parity build.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

mode=${1:-}
case "$mode" in
  ""|--cuda|--full) ;;
  *) echo "usage: tests/run.sh [--cuda|--full]" >&2; exit 2 ;;
esac

ARCH=${CUDA_ARCH:-native}
fail=0
step() {
  echo "== $*"
  "$@" || { echo "-- FAILED: $*"; fail=1; }
}

step bash tests/qa-gate.sh
step make pq2-0-test
step make test-qwen35-ref
if [[ "${DS4_FAST:-0}" != "1" ]]; then
  step make test-catalog-parity
fi

if [[ "$mode" == "--full" ]]; then
  step make test
fi

if [[ "$mode" == "--cuda" || "$mode" == "--full" ]]; then
  step make test-cuda-tokentile-ldmatrix CUDA_ARCH="$ARCH"
  step make test-solar-fattn CUDA_ARCH="$ARCH"
  step make test-inkling-attention CUDA_ARCH="$ARCH"
  step make test-qwen35-cuda CUDA_ARCH="$ARCH"
fi

if (( fail == 0 )); then
  echo "tests/run.sh: all checks passed"
else
  echo "tests/run.sh: FAILURES above"
fi
exit "$fail"
