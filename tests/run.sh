#!/usr/bin/env bash
# tests/run.sh — the checks a push must pass on this machine.
#
# Everything here is model-free and needs no GPU, so it runs from a cold
# checkout.  The gates that need the Bonsai GGUF, or the card, are run by hand
# and are listed in the handoff:
#
#   DS4_BONSAI_MODEL=/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf make test-qwen35-rows
#   make cpu && make bonsai-fold-selftest
#   make cpu && DS4_QWEN35_STEPS=8 make bonsai-ref-check
#   make test-qwen35-cuda CUDA_ARCH=sm_89
#
# Set DS4_FAST=1 to skip the Rust/C catalogue parity build.
set -u
cd "$(dirname "$0")/.." || exit 1

fail=0
step() {
  echo "== $*"
  "$@" || { echo "-- FAILED: $*"; fail=1; }
}

step make pq2-0-test
if [[ "${DS4_FAST:-0}" != "1" ]]; then
  step make test-catalog-parity
fi

# Rule 19: the QA-tester gate.  It fails until a fresh qa-evidence/qa-report.md
# covers every surface added relative to the base ref and ends with
# "verdict: overall PASS".
echo "== bash tests/qa-gate.sh"
bash tests/qa-gate.sh || { echo "-- FAILED: tests/qa-gate.sh"; fail=1; }

if (( fail == 0 )); then
  echo "tests/run.sh: all checks passed"
else
  echo "tests/run.sh: FAILURES above"
fi
exit $fail
