# Darwin vision follow-up (2026-10-02)

Two additional rounds are retained after [the Quad path](qwen-vision-2026-10-02.md):
packed K/V reduces four-image TTFT **5.569 → 5.092 s (-8.57%)**; bias/RoPE fusion
reduces **5.103 → 5.082 s (-0.41%)**, reproduced at -0.16%. Each uses its own
matched baseline; these are separate comparisons. Qualification covers Darwin
`MQ-Q5-SSD-PLE-BF16` with FP8 PLE, GB10, seq1 and fixed still images.

## First additional round: packed shared K/V

Fresh whole-worker profiling finds 54 attention calls totaling 3924.8 ms across
screen and four-image requests: 52.11% of captured GPU kernel time, including
startup and prewarm.
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

## Rejected experiment: eight packed queries

Fresh packed profiling still attributes 54.26% of four-image TTFT to attention.
The eight-query experiment reuses each packed K/V load across eight queries,
with 64 queries per block and unchanged 32-key tiles, FP32 arithmetic and
18,432-byte dynamic shared storage. It was rejected; packed Quad remains.

Matched 3072-row full-counter NCU shows 20.552 → 20.358 ms (-0.94%).
Shared-load instructions halve, and global-load requests fall
22,167,552 → 11,550,720. Registers rise 56 → 94; register-limited residency
falls 4 → 2 blocks and achieved occupancy 64.65% → 32.41%. No spilling occurs.
LSU utilization falls 93.36% → 89.45%. Short-scoreboard stalls fall
5.697 → 0.686, while MIO-throttle rises 0.488 → 1.444.
The static body grows 37,120 → 69,760 bytes (+87.93%); retaining Quad would
add the entire new 69,760-byte body. [Counters](benchmarks/qwen-vision-followup-2026-10-02/rejected-oct-counters.json)
and [code generation](benchmarks/qwen-vision-followup-2026-10-02/rejected-oct-codegen.json)
record the traffic/resource tradeoff.

Twelve fresh workers provide three repeats per arm/suite with the same serving
shape and 0/1/1/0/0/1 order. One-token screen/document medians improve
0.49%/0.31%, but photo regresses 0.46%, large 0.87% and four images 0.34%.
With 32 outputs, large/four-image TTFT regress 0.78%/0.36%; their wall times
regress 0.71%/0.57%. Large TTFT ranges are disjoint in both caps; four-image
cap32 ranges are disjoint, while cap1 overlaps. Four-image decode is
24.1 [24.0,24.1] → 23.8 [23.7,23.8] tok/s. [Latencies](benchmarks/qwen-vision-followup-2026-10-02/rejected-oct-latency.csv)
and [all samples](benchmarks/qwen-vision-followup-2026-10-02/rejected-oct-samples.csv)
retain the smaller variations and text control.

Larger query blocks also expand segment-crossing fallback: helper queries
increase 103 → 199 for 1031 rows/259-row segments and 96 → 192 for
8193/2051. Unit diagnostic timings regress 0.860 → 1.221 ms and
37.667 → 41.081 ms; these are not fresh workload A/B claims.

All 23 attention cases and memcheck pass. Five native cases retain exact
features, 32 raw full-vocabulary logit frontiers, tokens, payload and M-RoPE
across Quad/Quad/OCT/OCT. The [numeric rows](benchmarks/qwen-vision-followup-2026-10-02/rejected-oct-numeric.csv)
retain the prompt+31 state boundary. The HTTP audit passes 78 responses,
90 stats and 48 ownership snapshots with zero fault maxima and normal guard
exits. Clocks remain 2190–2197 MHz; minimum MemAvailable is 20.56 GiB.
[Qualification](benchmarks/qwen-vision-followup-2026-10-02/rejected-oct-qualification.json)
pins source, binaries and artifacts. Independent functional MTP/tool/restore
checks were not run for this rejected path. Correctness passed, but the
whole-workload regressions and register/code costs do not justify adoption.

## Second additional round: bias/RoPE fusion

The fused path is retained after two fresh worker comparisons. Its four-image
one-token TTFT improves **5103.2 → 5082.4 ms (-0.41%)**, then
**5112.4 → 5104.0 ms (-0.16%)**; both campaigns have disjoint timing ranges.

Fresh retained profiling measures 27 separate bias/RoPE pairs: 56.047 ms of
5127.4-ms four-image TTFT (1.093%) and 16.042 ms of 1468.8-ms screen TTFT
(1.092%). The separate FP32 bias pass writes all QKV, then rotation reads/writes
Q/K again and repeats indexing. One pair thread now rounds each Q/K bias sum
before the unchanged `powf`, `sincosf` and rotation expressions, then stores
both biased V halves. `__fadd_rn` preserves the FP32 boundary; six `FADD.FTZ`
instructions retain the build's FTZ behavior. Eligibility remains runtime
`sm_121`, head width 72 and at least 512 patches, independent of attention controls.
`DS4_QWEN_VISION_FUSE_ROPE=0` restores the pair; unset/1 selects fusion.

Matched fresh-process NCU uses the same binary, 3072×16×72 synthetic FP32
QKV/bias, native screenshot merge-2 positions, full application replay and
`--clock-control none`. The tiny registered bias map differs from imported
VMM production weights. Kernel sums exclude startup/upload/host time.

| Metric | Separate bias / RoPE | Fused |
|---|---:|---:|
| Serial kernel time | 0.549824 ms | 0.449792 ms (-18.19%) |
| Global-load requests | 995,328 | 774,144 (-22.22%) |
| Global-store requests | 552,960 | 331,776 (-40.00%) |
| Executed warp instructions | 25,214,976 | 10,229,760 (-59.43%) |
| Registers/thread | 20 / 22 | 32 |
| Achieved occupancy | 79.54% / 86.46% | 81.49% |
| Dynamic shared / compiler-reserved bytes | 0 / 1024 | 0 / 1024 |
| Local/shared spills | 0 / 0 | 0 / 0 |

The new body adds **4352 encoded bytes**, with both fallback bodies retained
(6400 → 10,752 bytes combined). No extra tensor or dynamic shared allocation
is added. Register/occupancy rates remain per kernel. DRAM counters are unavailable;
requests/sectors do not establish DRAM bytes. [Counters](benchmarks/qwen-vision-followup-2026-10-02/round2-counters.json)
and [code generation](benchmarks/qwen-vision-followup-2026-10-02/round2-codegen.json)
record the arithmetic, traffic and resource costs.

Twelve workers provide three repeats per arm/suite in 0/1/1/0/0/1 order,
using the same serving shape and empty isolated disk KV. All timed prompts
are uncached; fixed order can warm PLE/allocator state within a worker.
One-token screen/document TTFT changes -0.68%/-0.17%. Small, whose dispatch
is unchanged, varies +0.21%; photo varies +0.11%. Both ranges overlap. Large improves
5748.3 → 5735.8 ms (-0.22%) with disjoint ranges. For 32 four-image outputs,
TTFT changes 5090.5 → 5080.2 ms (-0.20%) with overlap; wall is
6393.3 → 6385.4 ms (-0.12%), decode 24.0 [23.6,24.1] → 23.9 [23.9,24.1] tok/s.
Large cap32's on-arm TTFT outlier **5814.6 ms** remains in the samples.
Six further workers reproduce the one-token image gain and the text prompt
stops naturally at 106 tokens: decode 35.3 [35.1,35.5] → 35.4 [35.2,35.6] tok/s
with overlap; wall 3179.8 → 3166.1 ms. Short text decode also overlaps.
[Image latencies](benchmarks/qwen-vision-followup-2026-10-02/round2-latency.csv),
[samples](benchmarks/qwen-vision-followup-2026-10-02/round2-samples.csv),
[text control](benchmarks/qwen-vision-followup-2026-10-02/round2-text-latency.csv)
and [its samples](benchmarks/qwen-vision-followup-2026-10-02/round2-text-samples.csv)
retain all medians and ranges. All 90 responses match choices, usage and request/fixture
bytes; 108 stats and 72 ownership snapshots pass, faults remain zero and guards exit
normally. The 496 clock samples are 2190–2197 MHz within the retained 300–2200-MHz
range; minimum MemAvailable is 20.52 GiB and full PSI avg10 remains zero.

All 16 full-QKV fixtures pass off/on/repeat/default byte comparisons, including V,
merge-2 positions, rounding/subnormal edges, head/frequency variants and tails.
Memcheck reports zero errors. Five native cases retain exact features, 32 raw
248320-value logit frontiers, tokens, payload and M-RoPE across separate/separate/
fused/fused. [Numeric rows](benchmarks/qwen-vision-followup-2026-10-02/round2-numeric.csv)
retain the prompt+31 state boundary; output 32 is pending.

The [independent functional audit](benchmarks/qwen-vision-followup-2026-10-02/round2-functional.json)
passes 10 requests, 26 HTTP responses, 13 stats, 93 receipt hashes and 16 embedded
images. Changed pixels return 9; tool-result reuse is 3006/3046 cached tokens.
The 18K continuation and fresh disk restore reuse 18136/18166 tokens. Restore
records 534 successful reads totaling 1,151,496,300 bytes; its same frozen request
and an empty-cache/encoder-computed control both return **12**. Six cases actively
use MTP. Faults/sheds remain zero, guards exit normally, minimum MemAvailable is
20.11 GiB and maximum full PSI avg10 is 0.18%.

Measured source is `228fd11e864374a5cbdf4d0fcab2bcea32c9acca` plus diff SHA256
`ca95263e47f3d247f2f57dfa9f53611c1f9924f3e314960e828235e270b10f22`.
Worker SHA256 is `25012b163bd7990d598418073ae26aa3e733a164374ddd15be63bbf45228934f`.
[Qualification](benchmarks/qwen-vision-followup-2026-10-02/round2-qualification.json)
pins builds, the unchanged artifact manifest, audits and raw receipts.
The default build passes all 16 full-QKV fixtures. Its
[nonzero SASS comparison](benchmarks/qwen-vision-followup-2026-10-02/round2-default-sass.json)
matches assembly and encoding for bias (160 instructions), RoPE (240), fused (272)
and packed attention (2320). Default worker SHA256 is
`485c7773e0ade5ac55961703b81dea875acc38bbdff7d2bb9eedb42c8cb1616a`.
Reproduce with `DS4_QWEN_VISION_FUSE_ROPE=0/1`; the full-model gate uses
`DS4_QWEN_VISION_GATE_CONTROL=DS4_QWEN_VISION_FUSE_ROPE`.
Configured 256K capacity remains unchanged. Evidence covers seq1, fixed one/four
still images and an 18K reuse/restore fixture. Filled 256K image quality,
concurrency, video, other artifacts/devices and long Agent throughput remain outside it.

## Retained profile and remaining work

Fresh single-worker Nsight captures retain the exact request boundaries above.
The 27 QKV pairs become 27 fused calls: four-image QKV time falls
56.047 → 44.923 ms (-19.85%), screen 16.042 → 11.111 ms (-30.74%).
These diagnostic captures locate work; the repeated A/B comparisons establish
the latency gain. [Whole-profile evidence](benchmarks/qwen-vision-followup-2026-10-02/round2-whole-profile.json)
retains request counts, region boundaries, source/build pins and raw receipt hashes.

Attention remains the largest target: 27 calls total 2786.646 ms of the retained
four-image request's 5118.4-ms TTFT (54.44%). No unexpected fallback or new
redundant transformation was identified. The wider-query mapping already failed
end-to-end qualification and increased register/code costs. No further candidate
is adopted in this campaign; another change requires fresh whole-workload
measurement and detailed target profiling. The fixed image/18K fixture boundary
continues to apply; this establishes neither filled-256K quality nor Agent throughput.
