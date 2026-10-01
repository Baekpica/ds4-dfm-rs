# IQuest-Q1 GB10 optimization — 2026-10-01

P1 adopted; campaign ongoing: **1/3 prefill, 0/3 dedicated decode rounds**.
Pinned mixed-quant artifact; 2048 prompt tokens, 8192 capacity, chunk 128,
32 EOS-suppressed greedy outputs, MTP off. One VMM owner, fresh ABBAAB workers,
each preceded by a separate warmup; observed clocks 2190–2197 MHz, unchanged policy.

| Phase | Baseline median (range), tok/s | P1 median (range), tok/s | Gain |
|---|---:|---:|---:|
| Prefill | 61.99 (61.95–62.03) | 97.22 (97.22–97.29) | 56.83% |
| Decode | 3.37 (3.37–3.37) | 4.49 (4.49–4.50) | 33.23% incidental |

Attention initially occupied 65.25%/76.21% of prefill/decode kernel time. P1 preserves the reduction order while using warp shuffles. SASS also shows compiler-generated two-key scheduling; the gain is not solely barrier removal. Shared memory stays 512 B/block and tensor allocations are unchanged; registers rise 26→38, with no spills.

All 12 workers retain identical 160K prefill logits and 32 tokens. Ordinary prefill/final logits and 392,827,956/398,955,700-byte native payloads are exact; four restore checks pass, faults unchanged. Reference 13, 32 attention cases including F32 sinks/rings, 1M reduction readbacks and racecheck pass. Six model-free verifier tests pass.

Synthetic resident attention medians improve 1.452→0.884 ms (row 1) and 29.043→11.322 ms (rows 128), three fresh samples each. Cache-flushed full-counter NCU prefill is 29.35→12.86 ms, LSU 85.15→61.23%, occupancy 93.36→96.71%. P1 toy profiling retained an idle owner; baseline NCU had none. These are separate cache regimes and not whole-model speed measurements.

The retained whole-workload profile totals 20.954 s prefill/6.995 s decode kernel time; attention remains 9.486 s (45.27%)/4.771 s (68.20%). Subsequent rounds start from this profile.

[Compact evidence and hashes](2026-10-01-iquest-q1-optimization-gb10.json). Shard hashes are release-manifest-derived with sizes/mtimes checked, not a fresh 88 GB hash pass. Existing [family limits](../iquest-q1.md) remain: no new 512K, long-context, MTP or quality qualification. Full-model state proof here does not cross the SWA ring.
