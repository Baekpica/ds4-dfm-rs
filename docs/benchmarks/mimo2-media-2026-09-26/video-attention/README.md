# Video attention: coalesced K (adopted)

The retained baseline contains the image optimizations, corrected LayerNorm,
and accepted audio path (`3dfd0077`). Adopt the coalesced K path for
**512–3072 rows**: fresh paired HTTP runs consistently reduce TTFT and wall
time, preserve full logits and outputs, and retain the larger-row fallback.

## Workload and evidence

- Four-second FineVideo clip, row 29, 8 distinct sampled frames at 2 fps;
  resize 640×352, 4 visual pairs, 880 patches and 220 features per pair.
- MiMo V2.6 Flash RL MQ-IQ2-XXS-XS-Q8 with BF16 projector, context 8192,
  continuous width 0, prefix reuse/MTP off, prefill chunk 4096, output cap 128.
- Profiled HTTP wall **8229.924 ms**, TTFT **3456.9 ms**, decode **26.6 tok/s**.
  The response reached the output cap; this is one profile, not a quality score.
- Export completed before worker termination. Guard/payload exits were 0;
  busy GPU samples were 2190 MHz within the preserved 300–2200 MHz range.

The [fixture](fixture.json) pins input/request hashes, dataset revision and
inherited activity annotation. The [request result](whole-request.json) records
the complete answer/tokens; the annotation is not an exact-answer oracle.
[Launch](whole-launch.json), [trace hashes](receipt.json), and
[guard/clocks](whole-guard-clocks.json) preserve provenance.

## Current bottleneck

| Video kernel | Calls | Total ms | Video kernel time | HTTP TTFT |
|---|---:|---:|---:|---:|
| Full attention | 16 | 322.292 | 41.67% | 9.32% |
| Window attention | 96 | 224.191 | 28.98% | 6.49% |
| Patch projection | 4 | 72.783 | 9.41% | 2.11% |

Full attention is the first detailed target: 880 rows, Q32/KV8/HD64,
interleaved stride 3072 with offsets 0/2048/2560, noncausal full attention
without sinks. Observed launch: grid 28160, block 128, 40 registers/thread,
3784 bytes dynamic shared memory. All 16 full and 96 window calls used the
retained kernels; no scalar attention fallback appeared.

The [whole summary](whole-summary.json) also records the complete video
kernel ranking and copy geometry. Language decode dominates total request
GPU time; the table isolates the video stage requested for this campaign.

## Host and timing limits

Video GPU intervals span 1181.541 ms, including 401.975 ms idle. The first
pair contains a 381.810 ms idle gap overlapping 338.182 ms of
`cuLibraryLoadData`; later pairs each span about 196 ms with about 1 ms idle.
This cold initialization cost is separate from repeated attention execution.

All video device copies total 6.212 ms. The much larger 722.088 ms of
`cudaMemcpy` API duration overlaps queued GPU work and is not additional
transfer cost. Poll/futex/sleep durations also overlap other threads.

There are no request/stage NVTX boundaries. Pair intervals are inferred from
patch launches and distinctive upload/readback sizes. FFmpeg and packing
before GPU execution are not separately timed. Percentages contextualize
cost and do not form an additive wall-time partition.

## Detailed target profile

The isolated unchanged kernel matches the observed 880-row layout and
resources. Full NCU collected **2221 metrics in 40 passes**: **19.893 ms**,
40 registers/thread, no spills, 90.52% occupancy, L1 data throughput 82.60%,
L2 throughput 81.31%, and 94.26% cycles with no eligible warp. Source sampling
had no overflow or dropped bytes.

The [counter summary](ncu-baseline.json) and
[exact source/PC excerpt](ncu-baseline-source.json) identify the QK dot loop:

- Frozen source line 147 has **940950 / 2047650 samples (45.95%)**.
  Its theoretical global sectors are **1585971200 versus 198246400 ideal**,
  an **8×** amplification. Neighboring lanes read different 3072-float rows
  at the same dimension; the K loads show MIO/LG throttling.
- Representative K load PC `0xf664c971c250` has 24780800 theoretical sectors
  versus 3097600 ideal. These source metrics describe access coalescing,
  not off-chip DRAM bytes; L2 hit rate is 99.86%.
- PC `0xf664c971cec0` follows the QK deferred barrier and has 538937 barrier
  samples. Those waits cannot all be attributed to the later scalar max.
  Separate PCs after max and denominator computation have 102100 and 89093
  barrier samples. Barrier overhead alone does not explain this profile.

The selected hypothesis is to load K cooperatively into a shared-memory
tile and read its transposed layout while retaining ascending dot arithmetic.
Shared storage/barriers, bank conflicts, occupancy and numerical parity are
measured below; coalescing is a resource tradeoff.

The owner was stopped for this isolated capture. NCU used `--set full`,
`--clock-control none`, kernel replay, `--cache-control all`, two skipped
launches and one capture. Cache control resets caches for replay; this is
distinct from the ordinary two-warmup smoke (**20.374 ms**). The probe's
7144 ms event time inside NCU includes replay overhead. Synthetic QKV and
standalone compilation also differ from whole-model data/cache interleaving.
[Build](ncu-build.json) and [capture](ncu-profile.json) receipts pin commands,
source, binary and raw report hashes.

## Candidate and measured costs

The candidate stages 32 K rows in a padded shared transpose tile. The
production wrapper enables it only for **512–3072 rows**, within the existing
full-vision layout/mask predicate. `DS4_MIMO2_VISION_COALESCED=0` restores the
retained score-cache kernel; `DS4_MIMO2_VISION_ATTN=0` restores scalar attention.
Windowed vision, audio and larger full-attention shapes retain their paths.

Full NCU at 880 rows gives:

| Metric | Retained | Candidate |
|---|---:|---:|
| Kernel duration | 19.893 ms | 10.360 ms |
| Global load sectors | 1,784,442,880 | 396,718,080 |
| Useful bytes/global load sector | 7.114 | 32 |
| Executed warp instructions | 480,803,840 | 1,438,131,200 |
| Shared memory/CTA | 3784 B | 12232 B |
| Registers/thread | 40 | 34 |
| Achieved occupancy | 90.52% | 52.94% |
| Spill requests | 0 | 0 |

Coalescing reduces sectors by **77.77%**, but adds **8448 bytes shared/CTA**
and **199.11% executed warp instructions**. FFMA, FMUL, FADD and MUFU
predicated thread counts match exactly; integer addressing, staging and
synchronization work increase. No persistent device allocation is added.
The [comparison](ncu-comparison.json) preserves these costs and source counts.

The candidate's leading shared-store PC has 593073 long-scoreboard samples
waiting on its preceding global K load. Its shared wavefront count equals
ideal; aggregate shared-conflict counters are nonzero, but this hotspot does
not demonstrate a dominant bank-conflict bottleneck.

## Shape pilot and qualification limits

Each arm used one fresh process, two warmups and three timed events.
Every full output dump matches byte for byte. These are viability pilots;
they do not establish repeated model-level gains.

| Rows | Retained mean ms | Candidate mean ms | Dispatch decision |
|---:|---:|---:|---|
| 512 | 7.057 | 3.130 | Candidate range |
| 880 | 20.079 | 10.212 | Candidate range |
| 1024 | 27.102 | 13.759 | Candidate range |
| 3072 | 230.009 | 201.946 | Candidate range |
| 6144 | 1226.533 | 2884.020 | Keep retained kernel |
| 8192 | 2633.811 | 6314.378 | Keep retained kernel |

The large-row regressions reject this candidate in those regimes. Extra
staging work and reduced residency are plausible contributors, not a separate
causal profile at those sizes. [Pilot evidence](isolated-pilot.json) preserves
all event samples, output hashes and clean guard exits.

[Independent source review](review.json) found no correctness blocker:
tile bounds and barriers are safe, arithmetic order is preserved, and the
retained full/window/audio/LayerNorm bodies are unchanged. The candidate
body matches the profiled scratch function; the frozen production proofs
below qualify the compiled path. [Build](candidate-build.json) and
[capture](candidate-profile.json) receipts preserve isolated provenance.

## Wrapper tests and frozen production build

The actual-wrapper gate passed **24 checks**, including OFF/default/ON,
511/512/513 and 3071/3072/3073 dispatch boundaries, 879/880/881 tile tails,
non-dyadic and nonfinite inputs, masks/layouts, input preservation and guards.
The retained full-attention suite passed **27 checks** and the window suite
passed **38 checks**. All three build/test/guard/payload exits were zero.
[Receipts](wrapper-tests.json) and [test output](wrapper-tests.txt) record the
exact checks; source, binary and log hashes were independently verified.

The final native build uses the original compiler flags. Paired AV and image
proofs have identical frozen host inputs and native objects except for
`ds4_cuda.o`; the two changed CUDA source inputs match the reviewed candidate.
The original AV receipt predates audio adoption, but its frozen source/object
matches the retained `3dfd0077` baseline. [Build provenance](build-provenance.json)
records this distinction, all four proof binaries, and the frozen production
server/benchmark hashes.

## Full-model correctness

**15 fresh processes** compare original/OFF/ON for two videos and three images.
Every process produces **152576 finite FP32 logits**; all full vectors, argmax,
generated tokens, stop IDs, answers, request bytes, rendered prompts, tokenized
prompts and media geometry match exactly. All guard/payload exits are zero.
[Comparisons and compact references](model-proof.json) and
[launch/artifact receipts](model-receipt.json) preserve the checks.

The videos contain 4/13 timestamped visual pairs and 880/2860 total feature
tokens; these aggregate feature counts are not per-kernel attention rows.
Image feature counts are screen 768, photo 256 and document 1536, covering
the candidate range and larger document fallback.

Both video continuations reach the 128-token cap; all image continuations
reach 64 tokens. This qualifies bounded output parity, not complete-caption
quality. Contact-sheet review agrees with the central hand/conservation/Liz
Rose and flooded/news/Kesennuma content; the first clip's red-thread wording
is uncertain because the red line is an overlay. The optimization preserves
these same outputs.

## Fresh HTTP A/B and decision

Two videos use **18 fresh workers**; the screen regression uses **9**.
Each worker performs one warmup and one measured request. Three rounds rotate
original/OFF/ON order, keeping context 8192, continuous width 0, prefix reuse
and MTP off, native chunk 4096, and the pinned request bytes unchanged.
All **54 warmup/measured HTTP proofs are exact**. Timings below are measured
request medians; warmup and profiler timings do not decide adoption.

| Workload | TTFT original/OFF/ON ms | Wall original/OFF/ON ms | TTFT gain vs original | Decode original/OFF/ON tok/s |
|---|---:|---:|---:|---:|
| Video 4 s | 2142.4 / 2144.5 / 2002.5 | 6906.7 / 6903.6 / 6765.1 | 6.53% | 26.7 / 26.7 / 26.7 |
| Video 13 s | 5592.1 / 5602.4 / 5127.2 | 10450.9 / 10457.7 / 9980.6 | 8.31% | 26.3 / 26.3 / 26.3 |
| Screen | 2338.7 / 2338.2 / 2218.0 | 4707.4 / 4700.4 / 4581.7 | 5.16% | 26.7 / 26.8 / 26.8 |

Every paired ON comparison improves TTFT and wall time against **both**
original and OFF. Wall median gains versus original are **2.05%, 4.50% and
2.67%**. Video decode medians are unchanged; the screen's 0.1 tok/s reporting
variation does not establish a separate decode gain.
[Video samples](http-ab.json), [screen samples](image-http-ab.json), and
[54 response proofs](http-proof.json) retain all pairs and bounded outputs.

All **27 guard/payload exits are zero**, with no intervention. The harnesses
request max/high 24/21 GiB and reserve 12 GiB; admission-adjusted effective
caps are preserved per process in [guard receipts](http-guards.json).
[556 clock samples](http-clocks.json), including 447 busy samples, span
**2184–2197 MHz**, within the unchanged 300–2200 MHz range; memory clocks
report N/A. [Launch receipts](http-launch.json) pin options and artifacts.

**Adopt within 512–3072 rows.** The repeated end-to-end gain justifies the
8448-byte shared-memory increase per CTA and extra staging instructions;
attention arithmetic and persistent device allocations are unchanged.
Reject the measured large-row regime and keep its retained path. These
fixtures show no correctness or decode regression; qualification does not
extend to unmeasured model families, layouts, or complete video answers.

After qualification, the task owner was terminated cleanly. No GPU process
remained; host available memory recovered to 116 GiB. The user-managed clock
range was preserved. [Cleanup receipt](cleanup.json).
