# MiMo media optimization — 2026-09-26

GB10, MiMo V2.6 Flash RL MQ-IQ2-XXS-XS-Q8 with BF16 projector. The three
accepted [text rounds](../mimo2-2026-09-26/README.md) remain retained.
Each media round starts with a fresh whole-request measurement, then profiles
the selected target separately and decides from correctness plus matched A/B.

## Image round 1: accepted

Full-vision attention caches each dot product once and accumulates outputs
in registers. Three rotated original/OFF/ON HTTP pairs on `screen.png` give:

| Metric | Original median | ON median | Change |
|---|---:|---:|---:|
| TTFT | 12817.3 ms | 4439.1 ms | −65.37% |
| Request wall | 15175.239 ms | 6794.645 ms | −55.23% |
| Reported prefill | 62.2 tok/s | 180.6 tok/s | +190.35% |
| Decode | 26.9 tok/s | 26.9 tok/s | unchanged |

Context 8192, continuous width 0, prefix reuse/MTP off, native chunk 4096,
64-token maximum, temperature 0. Each fresh worker has one warm and one
measured request. All 18 HTTP responses match exactly; separate Rust proofs
match all 152,576 logits, 64 tokens and input bytes for screen/photo/document
across original/OFF/ON. All guard exits are clean. GPU clocks remain within
the user-managed 300–2200 MHz range; busy samples are 2190 MHz.

The isolated kernel improves 2325.206→230.050 ms, while dynamic shared
memory rises to 12,552 B/CTA at the measured shape. There is no persistent
tensor allocation. The optional normalized-weight refinement showed no pilot
gain and was not retained. The main score-cache optimization is accepted.
[Full evidence, numerical scope and fallback](attn-score-cache/README.md).

## Image round 2: accepted

A fresh whole profile identified window attention: 24 calls took 2.307 s,
30.59% of request wall time. Bounding valid keys and caching their scores
reduces isolated latency 96.444→8.360 ms. Three rotated original/OFF/ON HTTP
pairs on the retained R1 baseline give:

| Metric | R1 original median | ON median | Change |
|---|---:|---:|---:|
| TTFT | 4450.7 ms | 2338.1 ms | −47.47% |
| Request wall | 6805.385 ms | 4695.218 ms | −31.01% |
| Reported prefill | 180.1 tok/s | 345.9 tok/s | +92.06% |
| Decode | 26.8 tok/s | 26.9 tok/s | no consistent regression |

Kernel and screen/photo/document model proofs are exact. Seventeen of 18
HTTP token/content signatures match: one measured OFF response varies in
wording, while all original and ON responses match. This reviewed control
variation is documented explicitly. All guards/fault checks pass. The
candidate uses 780 B shared memory per CTA and four additional registers,
with no persistent allocation or spills.
[Full evidence and HTTP caveat](window-score-cache/README.md).

## Correctness repair before audio

Audio validation exposed an existing shared-memory reuse race in LayerNorm.
Commit `8fcc1574` adds a consume-before-reuse barrier without changing the
reduction order. Racecheck changes from failure to zero hazards; audio and
vision numerical tests pass. Two fresh repetitions of each audio fixture
now match every logit and token. Earlier image results remain dated evidence;
subsequent rounds use the corrected baseline. [Proof and limits](layernorm-race/README.md).

## Audio round 1: accepted

Full causal codec attention caches scores and accumulates output in registers.
The isolated median improves 34.175→1.907 ms. Corrected original/OFF/ON
model proofs match all 152,576 logits and generated tokens on both fixtures.
Three fresh HTTP pairs per fixture give:

| Speech length | Original TTFT | ON TTFT | Change | Request wall change |
|---|---:|---:|---:|---:|
| 11.125 s | 1245.9 ms | 854.3 ms | −31.43% | −16.96% |
| 14.530 s | 1543.7 ms | 987.6 ms | −36.02% | −18.13% |

All 36 warm/measured HTTP responses match their fixture's reference run.
Decode medians remain 27.1/27.0 tok/s. This establishes parity with the
corrected baseline; the second fixture's existing transcription mismatch
is documented. Shared memory costs 2492 B/CTA at 557 rows; no persistent
tensor allocation is added. [Full evidence](audio-score-cache/README.md).

## Video round 1: accepted

Fresh profiling identifies full vision attention as the largest video input
kernel. Coalesced K reads through a padded shared tile reduce its isolated
880-row latency 20.079→10.212 ms. Three fresh HTTP pairs per video give:

| Video length | Original TTFT | ON TTFT | Change | Request wall change |
|---|---:|---:|---:|---:|
| 4 s | 2142.4 ms | 2002.5 ms | −6.53% | −2.05% |
| 13 s | 5592.1 ms | 5127.2 ms | −8.31% | −4.50% |

Decode medians remain 26.7/26.3 tok/s. Screen-image regression A/B also
improves TTFT 2338.7→2218.0 ms. All 15 video/image model proofs and 54
HTTP responses match their controls. Video outputs reach the fixed 128-token
cap; this verifies the same bounded continuation, not complete caption quality.

Dispatch is limited to 512–3072 rows. Larger-shape pilots regress and retain
the previous full-attention kernel. The candidate adds 8448 B shared memory
per CTA and about 199% more executed warp instructions; floating-point
operation counts are unchanged. No persistent allocation is added. The
measured gains justify this scoped tradeoff. [Full evidence](video-attention/README.md).

The requested media campaign completes with two image rounds, one audio
round and one video round. Dated evidence is limited to the recorded GB10
artifacts, serving settings and fixtures; it does not qualify other serving
shapes or complete audio/video quality scores.
