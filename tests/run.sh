#!/usr/bin/env bash
# tests/run.sh - local unit entry for ds4-dfm-rs.
#
# The rule 19 QA gate runs FIRST, so a commit-for-delivery or a push without
# fresh QA evidence fails here; the CUDA parity suites then still run and the
# exit status accumulates, so one invocation reports everything that is wrong.
#
#   bash tests/run.sh              QA gate + the three kernel parity suites
#   bash tests/run.sh --full       also the model-free suites (make test)
#   QA_MODEL=<slug> bash tests/run.sh
#
# CUDA_ARCH selects the arch for the CUDA builds (default native).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

ARCH=${CUDA_ARCH:-native}
status=0

bash tests/qa-gate.sh || status=1

# Kernel parity suites for the paths this tree exercises without weights.
make test-cuda-tokentile-ldmatrix CUDA_ARCH="$ARCH" || status=1
make test-solar-fattn CUDA_ARCH="$ARCH" || status=1
make test-inkling-attention CUDA_ARCH="$ARCH" || status=1

if [[ "${1:-}" == "--full" ]]; then
  make test || status=1
fi

exit "$status"
