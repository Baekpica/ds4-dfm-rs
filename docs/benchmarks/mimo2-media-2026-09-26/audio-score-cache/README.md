# Audio round 1: causal-attention score cache

**Adopted on the corrected LayerNorm baseline.** Three fresh isolated pairs
reduce median kernel time 34.175408→1.907222 ms (94.42%). All six corrected
model proofs and 36 HTTP warm/sample responses match exactly. Fresh paired
HTTP runs reduce median TTFT 31.43%/36.02% for the two audio fixtures, with
unchanged decode medians. Initial model checks exposed an existing LayerNorm
race, repaired and committed separately before this qualification.

The historical baseline is image R2 commit `33a1a2b4`. Qualification now
uses commit `8fcc1574`, which adds the independently verified
[LayerNorm repair](../layernorm-race/README.md), retaining both image and
all three accepted text changes. The attention candidate remains separate.

## Corrected model proof and build provenance

Six fresh Rust proof processes cover `eval-audio-3` and `eval-audio-0`
across corrected original, candidate OFF and ON. Every comparison matches
all 152576 finite logits, generated tokens, stop ID, answer, input bytes
and media geometry: maximum absolute difference and RMS are both zero.
The complete outputs contain 30 and 42 tokens, respectively. All six
guards exit zero. [Results](model-proof-corrected.json),
[verified input/output hashes](model-proof-hashes.json),
[launch receipt](model-proof-launch.json).

Audio-3 matches the normalized reference transcription. Audio-0 preserves
the corrected baseline's `I too agreed` wording against reference
`I to agree`; candidate parity does not establish perfect transcription
quality. The recorded normalization removes punctuation and whitespace
and is not a WER score. These proof timings are not HTTP speed A/B.

Frozen original and candidate inputs/objects remain stable during linking.
Only `ds4_cuda.o` differs; other native objects match. Original was linked
before the repair commit, but its frozen header/wrapper hashes match
`8fcc1574`. Standard native flags are retained, without split compilation.
The candidate proof SHA256 is
`72e8d3573e22de4d179deb04a7e5d3d922f4efff684cd7f76460a46b4a428ee8`.
[Build provenance](build-provenance.json),
[original freeze receipt](build-corrected-original.json),
[candidate freeze receipt](build-corrected-candidate.json),
[source/test review](review-corrected.json).

## Corrected HTTP A/B and adoption

Three rounds rotate original/OFF/ON, ON/original/OFF, OFF/ON/original.
Each of the 18 fresh server processes handles one warm request followed by
one measured request with identical pinned bytes. Context is 8192, native
chunk 4096, continuous width 0, prefix reuse/MTP off, temperature 0, and
output cap 128. The table shows measured-request medians.

| Fixture / metric | Original | OFF | ON |
|---|---:|---:|---:|
| audio-3 wall ms | 2327.506 | 2324.819 | 1932.846 |
| audio-3 TTFT ms | 1245.9 | 1246.6 | 854.3 |
| audio-3 decode tok/s | 27.1 | 27.1 | 27.1 |
| audio-0 wall ms | 3072.358 | 3071.417 | 2515.308 |
| audio-0 TTFT ms | 1543.7 | 1542.7 | 987.6 |
| audio-0 decode tok/s | 27.0 | 27.0 | 27.0 |

ON improves wall time and TTFT in every paired sample against both original
and OFF for both fixtures. Against original, wall-time medians improve
16.96%/18.13%; TTFT improves 31.43%/36.02%. Server-reported prefill throughput
rises 72.5→107.4 and 72.1→114.8 tok/s. Decode medians are unchanged; the
26.9–27.1 tok/s samples show no systematic decode penalty.
[All samples and paired comparisons](http-ab-corrected.json),
[frozen binaries, request hashes and launch settings](http-launch-corrected.json).

All 36 warm/sample responses match tokens, content, finish reason and prompt/
completion counts within each fixture. All finish with `stop`; none is
truncated. The separate corrected model proof covers full logits and media
geometry. Audio-0's reference wording limitation above remains unchanged.
[36 response proofs](http-proof-corrected.json).

All 18 guards and payloads exit zero without intervention. The requested
24/21 GiB caps and 12 GiB reserve were preserved; receipts record effective
admission limits. Across 204 telemetry samples, observed SM clocks are
2190–2197 MHz within the user-managed 300–2200 MHz range. GB10 reports memory
clock as unavailable. Only the shared owner remains after cleanup.
[Guard exits](http-guards-corrected.json), [clock/CPU telemetry](http-clocks-corrected.json).

Adoption follows exact corrected model/HTTP parity and consistent useful
end-to-end gains. The added shared-memory and arithmetic tradeoffs are
quantified below; no persistent tensor allocation is added. Qualification
is limited to the recorded audio fixtures and serving shape.

## Corrected baseline: fresh whole measurement

The same pinned audio request completes in 3030.740 ms, TTFT 1954.8 ms,
with 87 prompt tokens, 30 generated tokens and decode 27.1 tok/s.
The repaired server contains no audio attention candidate.

| Attention regime | Calls | GPU time | HTTP wall share |
|---|---:|---:|---:|
| Codec full causal: selected again | 12 | 409.846 ms | 13.52% |
| Codec causal window 128 | 12 | 115.370 ms | 3.81% |
| Local group 4 | 6 | 2.389 ms | 0.08% |

The inferred audio interval is 1039.477 ms, language prefill 545.660 ms,
and decode including initial graph capture 1218.990 ms. There are no NVTX
stage events; the same source/copy boundary caveats below apply. Export
completed normally, the guard exited zero, and busy clocks were 2190 MHz.
[Corrected trace](nsys-corrected.json), [request result](request-corrected.json),
[frozen server and launch](workload-corrected.json).

The scalar attention body and original isolated probe source are unchanged
by the LayerNorm repair. Geometry remains grid 35 × block 256, 40 registers,
zero shared memory for each codec call. The earlier isolated full NCU
target never executes LayerNorm, so its diagnosis remains applicable.
This source/shape check does not substitute for corrected model and HTTP
proofs. [Source hashes and confirmation](corrected-baseline.json).

## Historical pre-fix whole measurement

The pinned `eval-audio-3` fixture contains 11.125 seconds of speech. A fresh
Rust-server request returned its complete transcription after case
normalization: 87 prompt tokens, 30 generated tokens, wall 3035.149 ms,
TTFT 1952.1 ms and decode 27.0 tok/s. GB10, MiMo V2.6 Flash RL
MQ-IQ2-XXS-XS-Q8 plus BF16 projector, context 8192, native chunk 4096,
continuous width 0 and prefix reuse/MTP off; output cap 128.
[Fixture provenance](fixture.json), [launch and artifact receipt](workload.json).

| Attention regime | Calls | GPU time | HTTP wall share |
|---|---:|---:|---:|
| Codec full causal: selected | 12 | 410.349 ms | 13.52% |
| Codec causal window 128 | 12 | 114.920 ms | 3.79% |
| Local group 4 | 6 | 2.368 ms | 0.08% |

The measured codec shape is 557 rows, Q=KV16, head dimension 64, distinct
FP32 Q/K/V buffers, strides 1024, offsets 0, no sinks and causal full
attention. Source/GGUF metadata and 48 RoPE uploads of 142592 bytes agree
on the row count. The audio becomes 1113 mel frames, 557 codec rows,
279 RVQ rows, 280 padded rows and 70 language features.

GPU intervals inferred from source order and copies are 1040.754 ms for
audio upload through feature readback, 545.990 ms for language prefill
through the first logits kernel, and 1212.904 ms for decode including its
first graph capture. There are no NVTX stage ranges. These are not host
stage timings or the server's TTFT/decode boundaries. The cold request also
includes lazy module loading and allocations; warm paired HTTP runs must
separate retained gain from those first-use costs. [Trace summary](nsys-before.json).

The original profiling helper reported failure after export because an
inactive Nsight help message contained the substring `collecting data`.
Independent validation confirmed successful stop/export, a populated SQLite
trace, clean guard exit and export completion before worker termination.
The helper now checks exact active state. This was a status-parser false
positive, not a failed kernel capture; no GPU rerun was needed.
[Preserved raw failure](nsys-helper-failure.json),
[post-hoc validation receipt](nsys-validation.json).

## NCU diagnosis and change

Full NCU uses a faithful separate-buffer reproducer with synthetic Q/K/V
and warmed cache state; these values differ from actual projected audio
features. Each capture collected 40 passes. The scalar grid has only
35 CTAs for 48 SMs, 16.37% occupancy, 98.36% no-eligible cycles and 80.80%
L1 activity. Highest sampled stalls occur on output FFMA after V/output
loads, followed by repeated global stores. A 99.84% L2 hit rate does not
remove those dependency waits. [Baseline analysis](baseline-analysis.json).

One CTA per query/head caches Q and each unscaled causal-prefix dot once.
It preserves ascending dimension, maximum, denominator and key accumulation
order, including the multiply/subtract expression before `expf`. Output
dimensions accumulate in registers and write once. Only full causal audio
uses this path; local/windowed audio and image kernels retain their paths.

| Full NCU metric | Original | Candidate |
|---|---:|---:|
| Duration | 34.070592 ms | 1.895104 ms |
| Grid × block | 35 × 256 | 8912 × 128 |
| Registers/thread; spills | 40; 0 | 40; 0 |
| Dynamic shared memory/CTA | 0 | 2492 B |
| Achieved occupancy | 16.37% | 88.31% |
| Executed warp instructions | 90,969,956 | 52,554,896 |
| Global load requests | 39,996,928 | 10,243,840 |
| Global store requests | 5,017,472 | 17,824 |
| L2 read sectors | 128,778,311 | 67,572,335 |
| L2 write sectors | 159,701,744 | 71,296 |

Global stores fall 99.64%, L2 reads 47.53% and total warp instructions
42.23%. FFMA and EX2 counts halve; FMUL and reciprocal counts rise because
normalization is repeated per output dimension. Shared memory is the added
resource cost; no persistent tensor allocation is introduced. L2 utilization
rises from 37.45% to 88.01% as the work finishes much faster. Cache-sector
counts are not off-chip DRAM byte measurements. [Exact counters](ncu-comparison.json).

Neither capture reports sampling overflow. NCU did not lock clocks;
measured SM frequencies were 2.196894/2.195248 GHz. The user-managed
300–2200 MHz range is unchanged. [Baseline NCU](ncu-baseline.txt),
[candidate NCU](ncu-candidate.txt). Probe stdout during profiling includes
replay overhead and is not ordinary kernel timing.

## Isolated A/B and correctness

Three fresh process pairs alternate order, each with ten warmups and twenty
measured iterations. The table reports each process's mean.

| Pair | Original ms | Candidate ms |
|---|---:|---:|
| 0 | 34.175408 | 1.905907 |
| 1 | 34.168283 | 1.908411 |
| 2 | 34.193997 | 1.907222 |
| Median | 34.175408 | 1.907222 |

All repeated outputs have the same diagnostic hash. Independent pilot
comparison verifies all 570368 FP32 values, 2281472 bytes, exactly; both
dumps have SHA256
`0c6ea13b7466110e6d21b2da9dc0b5786bcf3546cf73bfd2480d6d5c6643e37e`.
The repeated hashes do not replace that full-buffer comparison.
[All samples and pilot hashes](isolated.json).

The [focused test](../../../../tests/mimo2_audio_attn.cu) passes actual-wrapper
OFF/default/ON exact output checks at rows 1/31/32/33/127/128/129/557/727/
1024/8192. It covers independent Q/K/V, cancellation, mixed magnitudes,
sharp scores, nonfinite cases, row/dimension permutations, guards and input
immutability. Unsupported masks, sinks, head counts, layouts and aliases
use fallback. The small double oracle's largest absolute error is 1.77e-7.
[Test receipt](test-result.json), [output](test-output.txt).

All 11 recorded pilot, isolated, profiler and focused-test guard exits are
clean. Busy isolated clock samples are 2190 MHz. [Guards](guards.json),
[clocks](clocks.json). [Original build](baseline-source.json) and
[candidate build](candidate-source.json) record their exact commands:
device optimization/architecture flags match; baseline additionally includes
production host flags. The corrected production A/B uses the standard build.

Dispatch requires the measured topology, full causal attention and rows
1–8192. Dynamic shared storage is bounded at 33032 B; larger or different
layouts retain fallback. `DS4_MIMO2_AUDIO_ATTN=0` disables this candidate.
Source: [kernel](../../../../cuda/mimo2_media.cuh),
[wrapper](../../../../ds4_mimo2_gpu.cuh).

Corrected model and matched HTTP A/B proofs are complete. Pre-fix full-model
differences remain historical investigation evidence and do not qualify the
candidate. Restart whole-workload profiling on this retained baseline before
selecting another optimization.
Raw reports, source CSVs, binaries and full dumps remain under
`scratch/mimo-media-20260926/ncu-audio-r1/`.
[Current decision state](receipt.json).
