#!/usr/bin/env bash
# ds4-advisor launcher: sweep the shapes a limited-VRAM card can serve at one
# context, rank them, and leave the endpoint up on the configuration this
# machine actually uses.
#
#   ./run-advisor.sh                 # 128K, fp8, 1 GiB kept free, full sweep
#   ./run-advisor.sh --dry-run       # arm list and estimates, runs nothing
#   ./run-advisor.sh --only pool-budget-fp8
#   ./run-advisor.sh --ctx 65536 --free-margin 384
#
# The sweep needs the card to itself: --stop-endpoint stops the launcher's
# server, and the guard restores it afterwards. The environment exported below
# is what that restore starts, so the endpoint that comes back is the 128K
# arm measured on 2026-11-05 (pool 5,856 MiB, chunk 512, 1,160 MiB free),
# not the launcher's default 64K resident arm.
#
# The advisor prices the expert pool from the VRAM free at that moment (the
# compositor and the driver are already gone from it) and checks the measured
# free memory against --free-margin; an arm that eats the margin is reported
# and cannot be ranked. Every arm runs with DS4_GEMMA4_MOE_MISS=evict: the GPU
# is the only expert compute unit, a miss fills the slot before the kernel.
#
# Outputs (JSON and CSV) land in misc/scratch/ with the context and margin in
# the name. Exit status is the advisor's.
set -uo pipefail

cd "$(dirname "$0")"

CTX=${DS4_ADVISOR_CTX:-131072}
MARGIN=${DS4_ADVISOR_MARGIN:-1024}
# auto = the engine quote, with the static model as the fallback. A dry run
# with a server answering on the port cannot get a quote (the card is full),
# and auto still prints useful estimates there.
ESTIMATE=${DS4_ADVISOR_ESTIMATE:-auto}
OUT_DIR=${DS4_ADVISOR_OUT:-misc/scratch}
PORT=${DS4_ADVISOR_PORT:-8099}

# The arm the endpoint is restored on (and the one this machine serves).
export DS4_GEMMA4_CTX="$CTX"
export DS4_GEMMA4_PREFILL="${DS4_GEMMA4_PREFILL:-512}"
export DS4_GEMMA4_MOE_CACHE_MIB="${DS4_GEMMA4_MOE_CACHE_MIB:-5856}"

if [ ! -x ./ds4-advisor ]; then
    echo "run-advisor: ./ds4-advisor is missing; build it with: make ds4-advisor" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
JSON="$OUT_DIR/advisor-ctx$CTX-$MARGIN-$STAMP.json"
CSV="$OUT_DIR/advisor-ctx$CTX-$MARGIN-$STAMP.csv"

echo "run-advisor: ctx $CTX, margin $MARGIN MiB, estimate $ESTIMATE"
echo "run-advisor: json $JSON"
echo "run-advisor: csv  $CSV"

./ds4-advisor \
    --ctx "$CTX" \
    --free-margin "$MARGIN" \
    --pool-budget \
    --estimate "$ESTIMATE" \
    --stop-endpoint \
    --port "$PORT" \
    --json "$JSON" \
    --csv "$CSV" \
    "$@"
status=$?

# The guard restores a server it stopped. Make sure one answers either way,
# so the machine is never left without its endpoint because the sweep failed.
if ! curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "run-advisor: nothing answers on $PORT, starting the endpoint"
    ./run-gemma4.sh serve start || true
fi

exit "$status"
