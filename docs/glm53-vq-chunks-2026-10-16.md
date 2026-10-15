# GLM VQ prefill chunk assessment

No chunk policy is changed. The runtime already defaults to 2,048;
previous performance qualification explicitly used 512. A 1,024-token
candidate reduces GPU work, but its first-use whole-inference gain does
not repeat. Keep these findings separate from a default-selection change.

## Measured target

The retained original-artifact trace spends 2.055 s, or 40.52% of prefill,
in routed VQ. Attention/KDA takes 26.25% and projections 19.81%.
The accepted [register-MMA path](glm53-vq-mma-2026-10-15.md) already uses
YoungAi `7fb26cc`'s `cuda_vq_reg_mma.inc.cu:29-208` staging/layout ideas:
cooperative loads, persistent E4M3 codebook reuse and two `cp.async` buffers.
Larger expert groups might amortize its packed-weight restoration further.

At chunk 512, actual routing produces 45,795 work items for 666,624
token/expert pairs: 45.49% of maximum 32-token capacity. This measures
weight-reuse capacity, not wasted tensor arithmetic. The kernel already
executes only `ceil(tokens/8)` occupied subtiles
(`cuda/glm_vq_mma.cuh:157-212`).

An isolated 280-item replay copies eight actual layer-three gate matrices
35 times to distinct addresses: 478,248,960 weight bytes. It repeats saved
layer-zero activations and uses uniform widths, so it does not reproduce
the full model's routing or input distribution.

| Tokens per expert item | Gate time, ms |
|---|---:|
| 8 | 3.408 |
| 16 | 3.714 |
| 32 | 4.368 |

These timings use 16 warmups and 64 repeats. Doubling useful rows from
16 to 32 adds 17.6% time. A full NCU capture of width 16 records 128
registers, no spills, 31.95% occupancy and 18.51% tensor-pipe activity.
This supports the reuse hypothesis; it is not an end-to-end speed proof.
Unchanged numerical fixtures and the existing library were reused.

## Fresh-process screen

All five runs use the original VQ artifact, the same `6b5a00e2` native
build and system CUDA/cuBLAS mappings. Input is 1,984 raw code tokens,
output 32 forced non-EOS greedy transitions, context 2,048, MTP/capture
off, one in-process VMM owner. Observed busy clocks are 2,184–2,190 MHz
under the unchanged 300–2,200 MHz policy. Processes are fresh; the OS
file cache is inherited, without a controlled cold/warm-expert protocol.

| Run order | Chunk | Prefill tok/s | Decode tok/s | Inference phase sum, s |
|---:|---:|---:|---:|---:|
| 1 | 512 | 401.57 | 7.34 | 9.300 |
| 2 | 2,048 | 331.54 | 7.35 | 10.338 |
| 3 | 1,024 | 456.16 | 7.35 | 8.703 |
| 4 | 1,024 | 356.69 | 7.38 | 9.898 |
| 5 | 512 | 396.90 | 7.33 | 9.364 |

Phase sums are inferred from rounded throughput, excluding model loading
and proof-file output. Process wall times, 207.6–290.4 s, are recorded
separately. `first_token_sec` is first decode-call time, not HTTP TTFT.

The two-run median phase sums are 9.332 s for 512 and 9.301 s for 1,024.
The approximately 0.34% difference is smaller than the observed variation,
and one candidate run improves while the other regresses. This is a
repeatability failure, not a fixed minimum-gain rule. The single 2,048
observation does not establish its general ranking.

At 1,024, work items fall to 33,435 and maximum 32-token capacity use
rises to 62.31%; at 2,048 they are 27,406 and 76.01%. All five runs have
finite frontier logits and identical 32 generated IDs. The frontier is
sharply peaked; these facts do not establish general quality. The frozen
four-case long-prefix quality pair remains unrun because no new setting
is adopted. Short-prompt and primitive proofs remain reusable.

## Why the GPU saving did not establish a useful gain

One 1,024-token Nsight Systems trace uses the same instrumentation,
native source hashes, binary, artifact and libraries as the retained 512
trace. Its checkout has only the intervening documentation commit.

| Prefill observation | Chunk 512 | Chunk 1,024 |
|---|---:|---:|
| NVTX wall, s | 5.072 | 5.979 |
| GPU busy, s | 4.836 | 4.138 |
| Routed VQ, s | 2.055 | 1.544 |
| Attention/KDA core, s | 1.331 | 1.289 |
| Projections, s | 1.005 | 0.843 |
| GPU idle, s | 0.236 | 1.841 |
| 71 prefill `cudaMalloc` calls, s | 0.035 | 1.643 |
| First kernel after prefill starts, s | 0.044 | 1.647 |

VQ GPU time falls 24.86%, but the extra allocation delay exceeds the
0.697 s total GPU saving. Attention is not the regression source in this
trace. CUDA API durations overlap GPU work and must not be added to
the phase costs. GPU idle is time without recorded kernels or transfers,
an upper bound on host gaps. The trace explains its own slowdown; it does
not prove the cause of each uninstrumented slow run or the allocator/OS
trigger. A previous same-size allocation sequence also showed this delay
in the [FP8 investigation](glm53-vq-fp8-2026-10-16.md).

Peak traced DEVICE allocations rise from 81.722 to 82.607 GiB; minimum
system available memory is 33.433 GiB for the candidate. These are not
CPU RAM plus VRAM: GB10 uses unified memory. Mapped files and untraced
driver bookkeeping are excluded from that allocation peak. Both traces
end with an empty allocation ledger. Guards retain 104/100 GiB max/high,
12 GiB reserve/trip, no scope swap and a 1,800 s deadline; PSI is telemetry.

The next investigation is initial graph/workspace allocation cost.
Moving that cost outside a benchmark timer would not improve request
latency. Decode remains projection-dominated, approximately 60% of wall
time. No kernel, precision, artifact or runtime default changed in this
round. [Evidence](evidence/glm53-vq-chunks-2026-10-16.json) includes receipts,
source/build identities, counters and raw-file hashes.

These observations cover bounded 2K eager execution. They do not qualify
8K, graphs, MTP, vision, live tools, multi-turn reuse, banks, snapshots or
disk KV.
