# GLM VQ KDA Q4_K speed baseline

Use the 136-matrix KDA Q/K/V/O Q4_K artifact for subsequent optimization,
as requested by the user. Two fresh Q4 runs increase median decode from
7.345 to 8.89 tok/s (+21.03%). Median prefill-plus-decode time falls from
9.210 to 8.377 s (-9.04%). Both Q4 phase sums are below both original
controls. A prefill gain is not established.

The subsequent [direct BF16 KV handoff](glm53-vq-kv-bf16-2026-10-16.md)
removes the conversion bottleneck identified below on this same Q4 artifact.

## Artifact and arithmetic

The active local artifact is `GLM-5.3-Flash-VQ-Q4K-KDA-GGUF`, manifest
`6ea4054b04cfb1bd2e26db318e022a5918b49c8002d7d5e1cf17e502a074a044`.
Only 136 nonexpert KDA Q/K/V/O matrices change. Routed/shared experts,
codebooks and all other 39,726 tensor entries retain their original bytes.
The original VQ artifact remains the same-source control.

Selected logical weight reads fall from 9,126,805,504 to 2,566,914,048
bytes/token. These are tensor payload sizes, not DRAM-counter readings.
The objective is execution speed, not the smaller artifact. Decode uses
the existing [cooperative Q4 load and fused dot path](glm53-vq-q4-stream-2026-10-16.md);
wide prefill uses Q4_K/Q8_1 MMQ, including activation quantization.
The accepted VQ register-MMA and asynchronous double buffering remain active.

The [earlier quality assessment](glm53-vq-kda-q4-2026-10-16.md) remains valid:
the 136-matrix diagnostic had mean target NLL +0.00556 nat over 106 positions
and a Korean date/prompt-echo regression. The user explicitly relaxed that
tradeoff and selected Q4_K for speed work. Those tests were not repeated
or relabeled as passes. Structural state, position and layout correctness
remain required.

## Matched execution

All runs use 1,984 raw code input tokens, 32 forced non-EOS greedy tokens,
context 2,048, chunk 1,024, MTP/capture off and one in-process VMM owner.
CUDA/cuBLAS mappings, executable and compiled source hashes match.
Observed busy clocks are 2,184–2,190 MHz under the preserved 300–2,200 MHz
policy. Processes are fresh; OS file-cache state is inherited.

| Artifact | Run | Prefill tok/s | Decode tok/s | Phase sum, s |
|---|---:|---:|---:|---:|
| Original | 1 | 459.60 | 7.32 | 8.688 |
| Original | 2 | 368.17 | 7.37 | 9.731 |
| KDA Q4_K | 1 | 423.44 | 8.81 | 8.318 |
| KDA Q4_K | 2 | 407.51 | 8.97 | 8.436 |

The original controls are the two arena-off runs from the immediately
preceding [allocation experiment](glm53-vq-work-arena-2026-10-16.md), reused
after exact identity/settings checks. The Q4 measurements follow them in
the same session; this is not an interleaved artifact ABBA sequence.

Phase sums are inferred from rounded throughput, excluding model loading
and proof-file output. Process wall times are recorded separately:
199.0/238.3 s for original and 194.6/205.3 s for Q4. Loading/cache variation
prevents treating those differences as a startup speed guarantee.
`first_token_sec` measures the first decode call, not HTTP TTFT.

Median prefill changes only 413.885 to 415.475 tok/s (+0.38%), within the
observed variation. Decode improves in both Q4 runs. MMA work items change
from 33,435 to 33,356 with actual routing; this is an artifact comparison,
not an isolated Q4-kernel claim.

All 154,880 frontier logits are finite. Generated IDs match on this peaked
raw-code fixture; frontier RMS difference is 0.391754. Neither fact changes
the previous quality assessment or establishes broad quality.

## Current bottleneck

One Q4 Nsight Systems run records prefill/decode at 297.67/9.00 tok/s.
Its prefill phase is 6.665 s: 4.400 s GPU busy and 2.265 s idle. The 71
initial `cudaMalloc` calls take 1.740 s; 340 MMQ `cudaMallocAsync` calls
take 0.009 s. API times overlap GPU work and are not additive phase costs.
This explains much of that trace's prefill delay, not every plain-run
variation.

Decode takes 3.557 s. The 4,352 cooperative Q4 calls take 0.388 s.
Expanded attention KV still makes 352 BF16-to-FP32 and 352 FP32-to-BF16
conversions: together 1.145 s, **32.19% of decode**. Routed VQ takes 0.322 s.
The next candidate is a direct BF16 handoff between KV projection and
attention. The already-completed conversion profile applies to these
unchanged shapes and kernels; it need not be repeated.

Peak traced DEVICE allocation is 76.508 GiB; minimum system available
memory is 39.175 GiB. The allocation ledger ends empty. GB10 uses unified
memory; these values must not be added as RAM plus VRAM. Guards retain
104/100 GiB max/high, a 12 GiB reserve/trip floor, no scope swap and a
1,800 s deadline; PSI is telemetry.

## Provenance and limits

The reused executable is
`21a6ae086b5194796481572de4dfb471863ddbaa77f53befa5561172cdabb148`.
It contains the subsequently removed, default-off work-arena candidate;
all runs explicitly set that flag to zero. Receipts verify the executable
against its compiled source snapshot and separately record current checkout
files. This is not a clean-HEAD build claim. No native rebuild, conversion
or unchanged arithmetic test was needed to select the existing Q4 artifact.

[Evidence](evidence/glm53-vq-kda-q4-speed-2026-10-16.json) contains both
controls, both Q4 runs, library/build/artifact identities and the current
profile. Scope remains 2K eager text. MTP, capture, 8K, images, live tools,
multi-turn reuse, banks, snapshots and disk KV remain unqualified.
