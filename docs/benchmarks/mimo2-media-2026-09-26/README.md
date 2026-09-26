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

Further rounds start from this retained baseline with a fresh whole profile.
Audio/video still need their own measured rounds; these image results do
not qualify their latency or output quality. Dated evidence is limited to
the recorded artifacts, serving settings and fixtures.
