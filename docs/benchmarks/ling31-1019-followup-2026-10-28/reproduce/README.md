# Follow-up Decode measurement

`mtp_perf.rs` is the exact measured helper, archived as evidence. It uses
`ds4-core` session APIs and the repository's existing Ling token fixture.
It is not part of the serving interface. Its source hash, native build
hashes, commands, controls and individual timings are recorded in
[`../d1.json`](../d1.json).

Build the CUDA objects with the repository's live `/usr/local/cuda`
toolchain first. To reproduce using Cargo's dependency resolution, copy
the helper to an unused example name:

```sh
cp docs/benchmarks/ling31-1019-followup-2026-10-28/reproduce/mtp_perf.rs \
  crates/ds4-cli/examples/ling31_followup_perf.rs
make -f Makefile -f - ling31-followup-perf <<'MAKE'
.PHONY: ling31-followup-perf
ling31-followup-perf: native/bridge/ds4_bridge.o $(CORE_OBJS)
	cargo rustc -p ds4-cli --example ling31_followup_perf --release --features native -- \
		-C metadata=$(DS4_RS_LINK_FINGERPRINT) \
		$(patsubst %,-C link-arg=$(DS4_RS_ROOT)/%,$(DS4_RS_LINK_OBJS)) \
		$(DS4_RS_LIBS)
MAKE
```

Run from the repository root, one process at a time under the host memory
guard (2 GiB reserve/trip), after checking GPU occupancy and clearing
caches following the previous model's exit. The helper arguments are:

```text
target/release/examples/ling31_followup_perf MODEL \
  tests/fixtures/ling31/evaluation-token-ids.json 2 128 OUTPUT.json
```

Alternate `DS4_LING31_KDA_RELOAD=0` / `=1` in off,on,on,off,off,on order.
Keep one binary and all other settings fixed. The output records every
case's prompt IDs, generated IDs, acceptance cycles, full frontier/end
logits and separate prefill/decode seconds. Compare values as F32 bytes,
not just the generated text. The aggregate excludes the separately
reported 8K corpus case. Preserve all raw runs, including mixed signs.

This helper warms an 8K session plus eight speculative calls, then opens
fresh sessions for the measured cases. It is a different timing recipe
from the published incremental `ds4-bench` sweep.

For D2, use `workload_perf.rs` in the copy/build command above. It adds a
genuine ordinary-model open, separate from the MTP-enabled model:

```text
target/release/examples/ling31_followup_perf MODEL \
  tests/fixtures/ling31/evaluation-token-ids.json plain 128 OUTPUT.json
target/release/examples/ling31_followup_perf MODEL \
  tests/fixtures/ling31/evaluation-token-ids.json mtp2 128 OUTPUT.json
```

Run each mode in fresh processes, alternating `DS4_LING31_KDA_CONV4=0`
and `=1` with the same order. Ordinary `eval` on an MTP-enabled model still
maintains its draft state, so combining both modes in one model load is
not a valid ordinary-Decode measurement. [`../d2.json`](../d2.json) records
all twelve receipts, mode-specific samples and full F32 proof hashes.

For D3, use the same `workload_perf.rs` helper in `mtp2` mode with
`DS4_LING31_VERIFY_REGS=0` / `=1`, preserving the same six-process order.
[`../d3.json`](../d3.json) records its receipts and 66 F32 proof hashes.
The ordinary kernels retain their assembly; the final published sweeps
record ordinary and MTP absolute throughput separately.
