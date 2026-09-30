# Final host and serving checks

October 1, GB10, MQ87 revision `65b235a4`. The final Rust server relink
includes the bounded qualification policy; native inference is unchanged
from the recorded long gates. [Receipts](final-validation-evidence.json).

Fmt, clippy, serialized workspace tests (1420 passed, 11 ignored),
all-target checks and the native host check pass. Refreshed catalog/server
C oracles and Python runner tests (23 passed) also pass. CUDA/state,
memcheck and fresh-process speed proofs remain in the round reports.

At configured 256K/two banks, chunk 2048, partial reuse and MTP off:

| Final-binary request | Input | Cached | Output | Finish |
| --- | ---: | ---: | ---: | --- |
| Short cold seed | 2048 | 0 | 13 | stop |
| Follow | 2084 | 2061 | 13 | stop |
| Fresh disk restart | 2120 | 2097 | 13 | stop |

All three retrieve the exact phrase. Two overlapping arithmetic requests
also answer correctly; native execution records `served=2 fallback=0`.
Governor and census faults remain zero. The qualified 256K/two-bank worker
is left on port 8002 with a 32-GiB disk store and persistence threshold
1024. Its live memory samples are a snapshot, not a completed soak.

The existing disk identity includes executable metadata. Relinking therefore
invalidates the earlier binary's cache identity. An extra long-cache probe
misses that identity and is cancelled; it is not a passed restoration gate.
The table uses a new store and identical binaries across restart. These
short checks do not repeat the full [256K/512K fixtures](long-context.md).
The user-stopped 1M gate remains incomplete and DSpark acceleration remains
unqualified.
