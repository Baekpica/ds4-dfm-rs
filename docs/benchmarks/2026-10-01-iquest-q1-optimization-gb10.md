# IQuest-Q1 GB10 optimization — 2026-10-01

P1/P2/P3 and D1/D2 adopted; ongoing: **3/3 prefill, 2/3 dedicated decode rounds**. P3 changes arithmetic; its bounded quality gates are described below.

The primary workload is now **8192 cold-KV prompt tokens**, capacity 16384, chunk 128, 32 EOS-suppressed greedy outputs, MTP off. Six fresh ABBAAB processes open empty sessions without separate warmup workers. The canonical weight owner persists; weights/OS caches are not claimed cold, and native startup prewarming remains unchanged. Clocks span 2184–2197 MHz, with every run's median 2190 MHz and unchanged clock policy.

| 8K phase | P1 median (range), tok/s | P2 median (range), tok/s | Change |
|---|---:|---:|---:|
| Prefill | 46.79 (46.75–46.79) | 78.76 (78.75–78.77) | +68.33% |
| Decode | 2.21 (2.20–2.21) | 2.20 (2.20–2.20) | −0.45%, rounded CSV |

All six workers have identical finite 160K prefill logits and 32 tokens across five independent comparisons. Every receipt confirms all 8192 tokens were prefilled. The 8K prefill/final full logits and **1,007,842,356/1,009,583,284-byte native payloads are exact**; four restore checks pass, forced tokens match greedy, and faults remain unchanged. Prefill exceeds the 4223-row physical SWA ring; the subsequent 32 decode tokens do not cross its next wrap. The repeated input is the original prompt, one newline, then the original again; SHA256 `f8082000683e432d7f2ff6f5342234c8c1b0c3a90adf6688e653a9e6162cc3fa`. The complete repeated input has 9654 tokens; it supplies a throughput workload, not quality evidence. Source: P2 commit `dca2bb06`; binary hash and raw receipt hashes are in the JSON.

The retained P2 8K profile measures 103.94 s prefill and 14.52 s decode host time. Attention takes 56.292 s (54.40%) and 12.157 s (84.53%) of each phase's aggregate GPU kernel time. This profiled run supplies attribution, separately from the fresh speed A/B.

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


## P3: tiled prefill

P2 spends 54.40% of prefill GPU time walking attention keys independently per query. P3 shares Q8 decoding across 64 queries and uses TF32 operand pairs, bounded MMA partials and FP32 running accumulation. Learned F32 sinks and both BF16 boundaries remain. Dispatch is rows128/full or SWA4096; `DS4_IQUEST_ATTN_TILED=0` retains P2.

Six fresh 8K cold-KV ABBAAB workers give **78.82 (78.77–78.83)→132.56 (132.45–132.67) tok/s, +68.18%**. Decode medians remain **2.20→2.20 tok/s**; candidate range 2.20–2.21. Clocks are 2190–2197 MHz. Each arm is deterministic, but cross-arm logits and generated wording differ.

This is a changed-arithmetic optimization, not lossless execution. Full prefill logits have RMS 1.1791, relative L2 0.3393 and maximum absolute difference 4.90625. First/final argmax agree; 31/32 forced steps retain the greedy choice, with “a” versus “just” at step19. First-layer KV is exact; deeper values differ. All values are finite, retained chronology is intact and four byte-exact self-restores pass. The exact-state verifier therefore reports a mismatch, not a pass.

Four bounded answer pairs pass on both paths: arithmetic, Python reasoning, short retrieval and **8676-token retrieval** of record137. These support adoption with the stated arithmetic change; they do not establish corpus-wide quality equivalence. The synthetic throughput prompt is not answer-quality evidence.

An earlier candidate was rejected for biased long-running MMA accumulation. The analytic 8192-key unit-V regression fails that candidate (maximum error 2.9981e-5) and passes the repair (5.9605e-8; tolerance 3.8147e-6). Sixty-four synthetic component cases and racecheck also pass. The raw-output diagnostic instantiation is used only by this regression test.

Resident full8K attention medians improve 23.505→4.615 ms; candidate samples span 3.783–4.751 ms. Separate cache-flushed NCU measures 24.685→5.829 ms. The kernel uses 255 registers, 34848 B static shared plus 1024 B driver shared, without spills. Whole-model allocation grows by **24576 B**, released on model drop.

The default-path 8K profile measures 62.002 s prefill and 14.515 s decode host time. Prefill GPU time is 49.75% MMQ, 20.81% worklist and 22.65% tiled attention. Decode attention remains 84.51%; this is the starting point for the dedicated decode rounds. Profiled latency is separate from speed A/B evidence.

The throughput prompt is reproducible as:

```python
original = (
    "Read this inventory and then answer the last question.\n"
    + "".join(f"Item {i}: blue square in the archive.\n" for i in range(480))
    + "\nWhat is 17 plus 25? Reply with the number and one short sentence.\n"
)
repeated = original + "\n" + original
```

Encode the raw repeated prompt and use its first8192 tokens. The JSON pins source, binary and receipt hashes, including the final default-path full-vector/token match to the explicit-on candidate.


## D1: cached single-row attention

The retained P3 decode spends 84.51% of GPU time in attention. Fresh row1 NCU attributes 71.9% of warp cycles per instruction to long-scoreboard waits. D1 cooperatively stages 128 compressed Q8 keys and values per CTA; one warp then executes the unchanged ascending-key reduction and recurrence. The full/SWA4096 single-row path is enabled by default; `DS4_IQUEST_ATTN_CACHED=0` restores the retained path. Recursive window512 and wider calls retain their previous dispatch.

Six fresh 8K cold-KV workers give Decode **2.20 (2.20–2.21)→4.51 (4.51–4.51) tok/s, +105%**. Prefill is **132.54 (132.49–132.57)→132.47 (132.46–132.63), −0.05%**, within the observed sample overlap. Clocks remain 2190–2197 MHz. All160K prefill logits and32 tokens match exactly across all six workers.

Separate whole-model proof compares all prefill/final logits and **1,007,842,356/1,009,583,284 bytes of native state** exactly. All32 forced tokens match greedy, four self-restores pass, and faults/speculative counters are unchanged. All243 component comparisons pass, including partial tiles, F32 sinks, Q8 extremes, ring wrapping and diagnostic wider rows; racecheck reports zero hazards. These are ordinary-mode gates; final MTP/serving checks remain pending.

The final resident target median improves **3.5266→2.0236 ms**; cache-flushed NCU improves **6.7416→2.6470 ms**. Registers rise38→40 and static shared512→34816 B, without spills. Tracked tensor/allocator counters are equal; no global scratch is added. System-wide available memory varies with page cache and is not an allocation equality claim.

The fresh retained profile measures **61.815 s prefill /7.104 s decode** host time. Decode attention is4.743 s/68.07% of aggregate GPU time, and the serial router is0.784 s/11.26%. This profile selects the next target; it is separate from the speed A/B.


## D2: warp expert selection

The retained D1 serial router consumes 0.784 s, 11.26% of decode GPU time. Its single thread repeatedly scans 256 experts for top8. D2 uses the warp-selection structure found in Qwen/DS4, while preserving IQuest's lower-ID ties, first-unused NaN behavior and serial selected-softmax order. Explicit unused masks retain valid negative-infinity candidates. Only single-row calls change; `DS4_IQUEST_ROUTER_WARP=0` restores the serial path.

Six fresh 8K cold-KV workers give Decode **4.51 (4.51–4.52)→5.07 (5.06–5.07) tok/s, +12.42%**. Prefill is **132.56 (132.43–132.58)→132.68 (132.56–132.83), +0.09%**; this incidental difference earns no prefill-round credit. All six full prefill vectors and 32-token streams are exact.

The separate ordinary-mode proof retains exact prefill/final logits, both complete native payloads, all 32 greedy choices and four self-restores. It also matches the committed D1 state. Tracked allocation counters, faults and speculative counters are unchanged. Twenty-two CPU-ID/CUDA-weight cases pass; 1,800 adversarial rows compare all 14,400 IDs and weight bits exactly against the retained CUDA kernel, including malformed inputs. Racecheck reports zero hazards. Model-free workspace tests pass: 1,473 passed, 0 failed, 12 ignored.

Resident router median improves **282.096→6.506 μs**; cache-flushed NCU improves **292.320→9.600 μs**. The CTA changes from one thread to one warp, registers32→36, with no shared memory, spills or new allocation. The retained whole profile measures **61.785 s prefill /6.331 s decode** host time. Attention now occupies **4.736 s, 76.46%** of decode GPU time; detailed profiling of that retained path precedes the final round.
