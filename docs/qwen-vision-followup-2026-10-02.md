# Darwin vision follow-up (2026-10-02)

Starting from [the retained Quad path](qwen-vision-2026-10-02.md), the first
additional round reduces four-image TTFT **5.569 → 5.092 s (-8.57%)** on GB10.
This qualifies Darwin `MQ-Q5-SSD-PLE-BF16` with FP8 PLE, seq1 and fixed still images.

## First additional round: packed shared K/V

Fresh whole-worker profiling finds 54 attention calls totaling 3924.8 ms across
screen and four-image requests: 52.11% of all captured GPU kernel time.
Attention accounts for 651.955/1560.8 ms (41.77%) of screen TTFT and
3272.827/5618.9 ms (58.25%) of four-image TTFT.
An isolated 3072-row, 16-head, 72-dimension FP32 launch still saturates LSU
at 92.78%. It executes six scalar shared loads per warp/key with negligible
bank conflicts. The target is instruction pressure, not a bank-padding problem.

Pack each key's first 64 K and V dimensions into 32 `float4` records and its remaining
eight into `float2` records. The hot loop emits one `LDS128` and one masked
`LDS64`; packing stores emit `STS128`/`STS64`. Four queries, 32-key tiles,
runtime `rsqrtf(head_dim)`, dot reduction, online softmax and explicit FMA
association remain unchanged. Segment-crossing blocks retain the per-row helper.
The default is limited to runtime `sm_121`, 72-value heads and at least 512 total patches.
`DS4_QWEN_VISION_PACK=0` restores scalar Quad; `=1` selects eligible packing.

Matched full-counter NCU captures use the existing LCG42 synthetic input,
one image segment, application replay and `--clock-control none`.
They preserve geometry/layout/arithmetic without model initialization.

| Metric | Scalar Quad | Packed Quad |
|---|---:|---:|
| Kernel duration | 24.091 ms | 20.572 ms (-14.61%) |
| Shared-load instructions | 226,492,416 | 75,497,472 (-66.67%) |
| Shared-load wavefronts | 226,500,456 | 188,834,869 |
| Shared-store instructions | 21,233,664 | 5,898,240 |
| Shared-store wavefronts | 21,968,664 | 22,941,468 |
| Executed warp instructions | 7.020 billion | 5.741 billion |
| Registers/thread | 56 | 56 |
| Achieved occupancy | 64.42% | 64.63% |
| Dynamic shared memory/block | 18,432 bytes | 18,432 bytes |
| Static instructions / encoded bytes | 2304 / 36,864 | 2320 / 37,120 |
| Local/shared spilling requests | 0 / 0 | 0 / 0 |

No workspace or query arithmetic work is added. Global-load requests remain 22,167,552.
The packed body is 256 bytes larger (+0.69%); retaining the scalar body adds
the new packed body's full 37,120 bytes to the combined binary.
Shared-store wavefronts increase 4.43%. MIO-throttle warps per active issue fall
2.326 → 0.488, while short-scoreboard stalls rise 1.444 → 5.709. LSU utilization
remains 92.78 → 93.31%; the stalls redistribute. Both captures observe 2.194 GHz.
[Counters](benchmarks/qwen-vision-followup-2026-10-02/round1-counters.json) and
[code generation](benchmarks/qwen-vision-followup-2026-10-02/round1-codegen.json)
retain these measurements.

## Fresh worker comparison

Twelve workers provide three repeats per arm/suite in 0/1/1/0/0/1 order.
Each uses empty isolated disk KV; every timed prompt is uncached. Fixed order
is small/screen/document/photo/large/multi, with text last in the 32-token suite.
PLE and allocator state may warm within each worker. Settings match the prior
report: context 262144, max-seqs 1, native chunk 8192, partial reuse, MTP2,
2-GiB PLE/16 workers, continuous lane, 32-GiB disk KV and the same VMM owner.
Graph fit/headroom remain 1/1024 MiB; guard max/high/reserve/trip remain 38/36/2/2 GiB.

| One-token case | Patches / prompt | Scalar TTFT | Packed TTFT | Reduction |
|---|---:|---:|---:|---:|
| Small | 256 / 93 | 286.4 ms | 287.7 ms | -0.45% |
| Screen | 3072 / 797 | 1560.2 ms | 1473.7 ms | 5.54% |
| Document | 6144 / 1565 | 4241.2 ms | 3876.9 ms | 8.59% |
| Photo | 1024 / 285 | 550.1 ms | 538.2 ms | 2.16% |
| Large | 8160 / 2069 | 6382.7 ms | 5738.7 ms | 10.09% |
| Four images | 10496 / 2659 | 5569.0 ms | 5091.5 ms | 8.57% |

Small-input dispatch is unchanged. Its TTFT ranges overlap, but client wall
increases 290.3 → 298.6 ms (+8.3 ms); that transport-wall variation is retained.
For 32 four-image outputs, TTFT is 5572.1 → 5097.9 ms and wall 6879.2 → 6418.7 ms
(-6.69%). Decode is 23.9 [23.8,24.0] → 23.7 [23.6,23.9] tok/s; LM prefill is
1487.7 → 1480.3 tok/s. The short text control is 34.3 → 34.1 tok/s with overlap.
Six additional fresh workers reproduce the image gain (5601.1 → 5116.5 ms)
and the unchanged text prompt stops naturally at 106 tokens: decode 35.5 → 35.5
tok/s, wall 3161.0 → 3163.0 ms. [Image comparison](benchmarks/qwen-vision-followup-2026-10-02/round1-latency.csv),
[samples](benchmarks/qwen-vision-followup-2026-10-02/round1-samples.csv),
[longer output control](benchmarks/qwen-vision-followup-2026-10-02/round1-text-latency.csv)
and [its samples](benchmarks/qwen-vision-followup-2026-10-02/round1-text-samples.csv)
preserve all medians and ranges. This establishes an image-processing gain.

All 90 responses match choices, finish reasons, usage, request and fixture bytes
across paths/repeats. Audits cover 108 stats files and 72 ownership snapshots;
fault maxima are zero and every guard exits normally. The 502 clock samples
are 2190–2197 MHz within the preserved 300–2200-MHz range. Minimum MemAvailable
is 20.64 GiB; full memory PSI avg10 reaches 0.54% in the image A/B.

## Correctness and serving

All 13 attention shapes pass scalar/packed/default/repeat exact comparisons,
cross-image isolation and sampled F64 max/relative-RMS bounds 5e-5. Memcheck
reports zero errors. Five Darwin cases run scalar/scalar/packed/packed: all
features, 32 full 248320-value raw logit frontiers, 32 greedy IDs, live payload
and M-RoPE bytes match. Saved state is prompt+31; output 32 is pending.
This plain-decode gate excludes EOS/EOT. [Numeric rows](benchmarks/qwen-vision-followup-2026-10-02/round1-numeric.csv)
record the exact sizes and boundaries.

The [independent functional audit](benchmarks/qwen-vision-followup-2026-10-02/round1-functional.json)
passes 10 requests, 26 HTTP responses, 13 stats snapshots, 93 receipt hashes and 16
embedded images. Screenshot 3, changed-pixel 9, invoice 385.00, Earth and four-image
tool arguments 3/385 pass. Exact tool history returns **3, 385** with 3006/3046 cached
tokens and encoder skip. The 18K continuation reuses 18136/18166 tokens. A fresh
worker restores the frozen request with 534 successful KV reads totaling
1,151,496,300 bytes and returns the same **12** as an empty-cache, encoder-computed
control. Six cases actively use MTP. Faults/sheds remain zero; guards exit
normally, minimum MemAvailable is 20.19 GiB and full PSI avg10 reaches 0.75%.

## Provenance and reproduction

Measured source is `7d5ba917269923776b94e35a62314a99f304f75d` plus diff SHA256
`573673fbff684a56acc79e8d10df9f561f95574a1886190b55b0dd70e43f0366`.
Measured worker SHA256 is `87ec571075f436b5267a5a6f441e33226e9043db8335cc58bde7650d3b3a5f5c`.
The [scalar SASS check](benchmarks/qwen-vision-followup-2026-10-02/round1-scalar-sass.json)
proves its 2304 instructions match the retained baseline. The default build's
13-shape gate passes; its [nonzero SASS comparison](benchmarks/qwen-vision-followup-2026-10-02/round1-default-sass.json)
matches both measured kernels exactly. Default worker SHA256 is
`6b60a3d55b16ce3b3a830cb32b2d630f94c05360649da0452ddbf5d3f4ba824c`.

Artifact revision remains `0caa1f4961fc9d1ef9de400df7f11dd7ac6a6cd1`;
[full artifact hashes](benchmarks/qwen-vision-2026-10-02/artifacts.json) and
manifest SHA256 is `22a1f667f731c776f78fe1eca86ca1dafadde48aa9e35ecf8bf985cae1f24c9b`
are unchanged. [Qualification](benchmarks/qwen-vision-followup-2026-10-02/round1-qualification.json)
retains source/build pins and audit counts. Raw evidence is ignored under
`scratch/qwen-multi-image-r2-20261002/`.

Use the prior report's build, fixture and serving procedure with
`DS4_QWEN_VISION_QUAD=1` and `DS4_QWEN_VISION_PACK=0/1`. For full-model parity,
use [Rust token preparation](../tests/fixtures/qwen-images/README.md#rust-tokenized-full-model-gate)
and `DS4_QWEN_VISION_GATE_CONTROL=DS4_QWEN_VISION_PACK`.
Configured 256K capacity remains unchanged; this round covers seq1, fixed one
to four still images and an 18K reuse fixture. It does not qualify filled 256K
image quality, long Agent throughput, other artifacts/devices, video or concurrency.
