# IQuest-Q1 GB10 optimization — 2026-10-01

P1/P2 adopted; ongoing: **2/3 prefill, 0/3 dedicated decode rounds**. P3 remains an unqualified candidate, disabled by default.

The primary workload is now **8192 cold-KV prompt tokens**, capacity 16384, chunk 128, 32 EOS-suppressed greedy outputs, MTP off. Six fresh ABBAAB processes open empty sessions without separate warmup workers. The canonical weight owner persists; weights/OS caches are not claimed cold, and native startup prewarming remains unchanged. Clocks span 2184–2197 MHz, with every run's median 2190 MHz and unchanged clock policy.

| 8K phase | P1 median (range), tok/s | P2 median (range), tok/s | Change |
|---|---:|---:|---:|
| Prefill | 46.79 (46.75–46.79) | 78.76 (78.75–78.77) | +68.33% |
| Decode | 2.21 (2.20–2.21) | 2.20 (2.20–2.20) | −0.45%, rounded CSV |

All six workers have identical finite 160K prefill logits and 32 tokens across five independent comparisons. Every receipt confirms all 8192 tokens were prefilled. The 8K prefill/final full logits and **1,007,842,356/1,009,583,284-byte native payloads are exact**; four restore checks pass, forced tokens match greedy, and faults remain unchanged. Prefill exceeds the 4223-row physical SWA ring; the subsequent 32 decode tokens do not cross its next wrap. The repeated input is the original prompt, one newline, then the original again; SHA256 `f8082000683e432d7f2ff6f5342234c8c1b0c3a90adf6688e653a9e6162cc3fa`. The complete repeated input has 9654 tokens; it supplies a throughput workload, not quality evidence. Source: P2 commit `dca2bb06`; binary hash and raw receipt hashes are in the JSON.

The fresh retained 8K profile measures 103.94 s prefill and 14.52 s decode host time. Attention takes 56.292 s (54.40%) and 12.157 s (84.53%) of each phase's aggregate GPU kernel time. This profiled run supplies attribution, separately from the fresh speed A/B.

**Initial 2K evidence follows; its warmup protocol and P1 results are unchanged.**
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

[Compact evidence and hashes](2026-10-01-iquest-q1-optimization-gb10.json). Shard hashes are release-manifest-derived with sizes/mtimes checked, not a fresh 88 GB hash pass. Existing [family limits](../iquest-q1.md) remain: no new 512K, long-context, MTP or quality qualification. The initial 2K state proof does not cross the SWA ring; the separate 8K proof above covers committed state after wrapped prefill.

P2 assigns four independent head warps per CTA at rows128/full or SWA4096. It preserves the product/tree/recurrence/BF16 contract; tails, decode and MTP retain P1. Three fresh samples per arm give prefill **97.05 (97.05–97.29)→128.38 (128.25–128.52) tok/s, +32.28%**; decode medians are both 4.49. All 12 workers retain exact logits/tokens; ordinary prefill/final payloads and four restore checks are exact. The 64 long attention cases plus Reference13 pass three-way full-output parity, including F32 sinks and permuted positions.

Synthetic rows128 attention is 11.293→5.658 ms. Cache-flushed NCU is 12.86→5.84 ms, regs 38→40, static shared 512→0 B/block, no spills; LSU 87.31%, occupancy 88.52%. Linked production SASS matches the standalone instruction sequences; total shared allocation is 1536→1024 B/block, including the separately reported 1 KiB driver allocation. SASS preserves arithmetic and removes CTA barriers; it also changes scheduling and unrolling. No new tensor allocation. `DS4_IQUEST_ATTN_WARP=0` retains P1; parent `DS4_IQUEST_ATTN_SHUFFLE=0` restores the original path. The retained 8K cold-KV profile and A/B results are recorded above.
