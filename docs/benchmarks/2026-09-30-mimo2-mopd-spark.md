# MiMo-V2.6-Flash-MOPD on GB10

Artifact: `MQ-IQ2-XXS-XS-Q8-MM-BF16`, Hugging Face revision
`16965fb66e6ab3290e6cd20ddbfa30adda692764`. Baseline: `7a78fcd8`;
`c42c969b` changes performance controls only. These results qualify the
recorded workloads, not a general quality score or a 64K curve.

## Prefill 1: direct token-compact Q8

The retained 8K/128 profile took 6.858 s for prefill: paired IQ2 Gate/Up
32.47%, IQ2_XS Down 22.05%, all input quantizers 3.26%. Gate/Up quantized
and stored eight identical copies of each token's activation.

The Spark IQ2_XXS path now quantizes once and reads through the sorted token
map. Predicate: M2048/K4096, 256 experts, top-k 8, widths 256–8192, aligned
weights and available D2R. Q8 bytes, dequantization, MMA and ordered FP32
accumulation are unchanged. `DS4_MIMO2_INPUT_Q8_COMPACT=0` restores the
original layout. Refused D2R rebuilds sorted input for generic MMQ.

At 4096 tokens, input payload falls from 144 to 18 MiB without a gather
launch. Registers/shared memory remain 128/37,744 bytes per thread/block.
Cold-cache NCU matrix time rises 24.088 → 24.617 ms; issued instructions
rise 1.07%, local loads fall 12.5%, and L2 read sectors fall 2.67%. Less
quantization outweighs the matrix cost in the complete entry and workload.
A refused fast path can temporarily retain both compact and sorted buffers.

| Fresh-process measurement | OFF | ON | Change |
| --- | ---: | ---: | ---: |
| Complete Gate/Up entry, ms | 24.063 | 23.926 | −0.57% |
| 8K/128 prefill, tok/s | 1192.705 | 1196.760 | +0.34% |
| 8K/128 decode, tok/s | 24.485 | 24.440 | −0.18% |
| 8K/1024 prefill, tok/s | 1191.105 | 1195.540 | +0.37% |
| 8K/1024 decode, tok/s | 24.775 | 24.855 | +0.32% |

Entry: three samples per arm, 16 operator warmups and 100 repetitions,
production geometry with deterministic synthetic finite inputs. Whole
workload: `speed-bench/promessi_sposi.txt`, chunk4096, greedy, MTP/DFlash
inactive, one shared VMM owner, one worker at a time. Separate complete
8K/128 warmup precedes each fresh process. Original/OFF/ON orders rotate
for three rounds; three additional OFF/ON pairs confirm the small gain.
All six prefill pairs improve (+0.19% to +0.60%). The 1024-token check uses
OFF/ON/ON/OFF; it does not reproduce the small short-decode decrease.
Observed busy clocks: 2184–2190 MHz; configured range: 300–2200 MHz.

Proof: all 19 measured full-vocabulary frontiers (152,576 finite logits)
and 128/1024-token streams match exactly. A real first-4K Gate/Up capture
also matches all 512 MiB of outputs byte-for-byte. Native paired tests cover
widths1/9/255/256/511/512/4096/8192, top-k6, disabled D2R, disabled Y
indirection, original schedule, and forced D2R refusal: 26 passing cases.
`tests/mimo2_gateup_compact.cu` checks dispatch and exports parity buffers.
`cargo test -p ds4-perf --locked` passes 99 tests; fmt/clippy pass.

NCU uses captured real weights, Q8 and routing in a bounded operator replay
with Owner absent (`--set full --clock-control none --cache-control all`).
Quantizer diagnostics use synthetic F32 values with the captured routing;
those timings are not whole-model throughput. Clock-limited pre-reboot
runs, a failed full-model replay, a timed-out ownerless capture, and the
slower extra-gather prototype are excluded from adopted performance claims.

Local evidence: `scratch/mimo-mopd-20260930/direct-compact-decision.json`,
`direct-compact-ab-normal/comparison-extended.json`,
`direct-compact-decode-1024/comparison.json`, `direct-compact-native/`, and
`direct-compact-target-profile/`. Baseline benchmark SHA256:
`68322644c870fac8cfff890471ca2a937b920fd9684e4e4103af391434d25640`;
retained candidate benchmark SHA256:
`f6aa97e74fb2cf6226adadb200c7cbfc0fc77a3c5011de47ff8f723eb04329d1`.

## Prefill 2: pipelined Down worklists

After retaining Prefill 1 and rejecting compiler-fence prototypes, a fresh
8K/128 profile took 6.824 s: raw IQ2_XS Down consumed 1.505 s (22.06%).
Of its first-4K scheduled columns, 85.19% used 128-column tiles, which
cannot use the existing two-stage IQ pipeline. Full-counter NCU showed
34.47% issue activity and 1.40 long-scoreboard stalls per selected issue.

The device builder and persistent consumer now accept a private maximum
width. Generic callers retain 128. Only Spark fused, unweighted MiMo Down
(M4096/K2048, 256 experts, eight assignments per forward token, widths
256–8192) selects 64. It reuses the existing pipelined dot primitive;
weights, sorted D4 Q8, output layout and FP32 accumulation are unchanged.
`DS4_MIMO2_DOWN_PIPE64=0`, disabled worklists, disabled pipeline or disabled
narrow tails restore the old path. Decode and generic IQ callers retain it.

This is a measured resource tradeoff. The real first chunk grows from
13,536 to 21,152 work items; capacity grows 195,840 → 293,376 bytes
(+95.25 KiB, before pool rounding). Scheduled columns fall 3.13% as padding
shrinks. NCU executed instructions rise 12.58% and L2 read sectors 26.28%;
L2 read misses fall 0.39%. Registers/shared memory stay 255/80,128 bytes;
the isolated 64-column kernel has no stack spill. Issue activity rises to
42.26% and long-scoreboard stalls fall to 0.84. Cold NCU kernel time falls
15.602 → 15.136 ms; its throughput is excluded from whole-model results.

| Fresh-process measurement | OFF | ON | Change |
| --- | ---: | ---: | ---: |
| Real captured Down kernel, ms | 15.600 | 15.167 | −2.77% |
| Complete production Down entry, ms | 18.645 | 18.255 | −2.10% |
| 8K/128 prefill, tok/s | 1197.440 | 1206.290 | +0.74% |
| 8K/128 decode, tok/s | 24.320 | 24.180 | −0.58% |
| 8K/1024 prefill, tok/s | 1195.080 | 1203.765 | +0.73% |
| 8K/1024 decode, tok/s | 24.630 | 24.630 | 0.00% |

Both isolated targets use the actual captured first-4K weights, inputs and
routing: three fresh OFF/ON pairs, 16 warmups, 100 repetitions. The complete
entry uses the production device sorting, SwiGLU/Q8 quantizer and worklist
builder. Every complete-entry output matches the captured in-model 512 MiB
byte-for-byte. Isolated CUDA allocations replace full-model VMM imports;
only the kernel prototype rebuilds the captured worklist on CPU.

Whole-model conditions match Prefill 1. Three rotated retained-original/
new-OFF/new-ON rounds produce nine samples; all three prefill pairs improve
(+0.56% to +0.89%). The short Decode decrease prompts a fresh 1024-token
OFF/ON/ON/OFF check with a separate 8K/128 warmup before each process.
Long Decode rates vary 24.59–24.67 tok/s across both arms, with equal
medians; the short decrease is not reproduced. Observed busy clocks:
2184–2190 MHz, within the confirmed 300–2200 MHz range.

All 13 measured full-vocabulary frontiers and 128/1024-token streams match
exactly. `tests/mimo2_down_pipe.cu` checks production dispatch, capacity,
tags and exactly-once valid output coverage. Fourteen fresh OFF/ON pairs
cover decode1, widths32/255/256/511/512/4096/8192, concentrated8192,
negative IDs, generic IQ and worklist/pipeline/tail kills; all outputs match.
Negative IDs reserve a sorted prefix in the existing MMID contract. An
initial test assertion incorrectly required that prefix to be empty;
it was corrected without a production change and the failed diagnostic
is excluded. The implementation and corrected coverage test were reviewed.
Perf tests pass99; fmt/clippy and the production CUDA build pass.

Evidence: `scratch/mimo-mopd-20260930/prefill2-down-decision.json`,
`prefill2-down-ab-normal/`, `prefill2-down-decode-1024/`, `prefill2-native/`,
`prefill2-down-entry-ab/`, `prefill2-down-operator-ab/`, and
`prefill2-pipe64-counter-comparison.json`. Retained candidate benchmark SHA256:
`f8d941248fac1400bf9656ad5727d29abbbc4c1dc73c4cdde17f1260455bd204`.
