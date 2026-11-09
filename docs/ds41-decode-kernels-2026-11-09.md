# DeepSeek V4.1 decode kernels — the step budget, the expert ceiling, and what is left

2026-11-09. Artifact `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf`
(113,556,639,424 B), server binary `139aef25…` (the pre-stager build), GB10
at 2,398-2,444 MHz under the 300-2,200 MHz policy (no clock service on this
host). Traces: `/tmp/l1k_ng_20261109-082442` (plain decode, eager),
`/tmp/l16k2_20261109-084910` (width 2, eager, MTP on). Route dump:
`/tmp/routedump_20261109-091314/routes.bin` (1,118 routed-MoE calls from the
hooked scratch build `c44a0341`, 1,040 of them main-MoE).

This record closes the decode kernel unit. Two things it establishes: what a
decode step actually costs, and that the two cheap fixes for the last gap are
measured neutral.

## 1. The step budget, on a clean window

The earlier width-2 budget (64.27 ms/step) was cut over a window whose first
90 ms still carried the prefill tail: every `v41_q4k_to_bf16_kernel` and
cuBLAS GEMM launch in it stops at 216.140 s while the decode still runs to
217.0 s, so the 1.13 ms of q4k conversion plus 0.95 ms of cuBLAS per step
were prefill, not decode. The window below starts well after that boundary
(216.400-217.050 s) and holds 11 steps, counted by the `output.weight`
launch (grid 16,160, one per step) and by the routed-MoE cadence.

| item | ms/step | notes |
| --- | ---: | --- |
| q4_K projections (skeleton + shared experts + head ×2) | 27.66 | 4.75 GB at 172 GB/s |
| routed experts | 17.80 | 3.21 GB at 180.6 GB/s |
| fp8 (draft towers + Engram wkv) | 4.24 | |
| hyper-connections (mix + fused + post) | 2.88 | 272 launches/step |
| attention mma + sparse merge | 2.54 | |
| bf16 GEMV (router path) | 2.29 | |
| rms 0.58, router 0.40, rope 0.35, swiglu 0.27, vq_order 0.26, act_quant 0.20, topk 0.08, rest ~0.5 | 2.64 | 413 elementwise launches/step |
| **total kernel time** | **60.39** | 20,033 launches, 1,821/step |

The untraced arm is 58 ms/step (34.5 tok/s at 2.00 tokens per step), so the
trace costs about the 2 ms it did in the width-1 cut. At width 1 the plain
token is 33.1 ms untraced; its traced kernel sum is 38.6-39.4 ms per token,
which is the same trace overhead and not a second budget.

Two consequences worth keeping:

- Owner option (b) from the handoff, the "draft/verify GEMV path" worth
  "1.13 ms of q4k_to_bf16 plus 0.95 of cuBLAS", does not exist. Every q4_K
  entry already takes the direct GEMV for `n <= 8`
  (`cuda/ds41_q4k.cuh:342`, `:366`), and no GEMM runs during decode.
- The head is read twice per step (draft logits and verify logits, 355.1 MiB
  each) for 3.3 ms of the 27.66.

## 2. The routed-expert ceiling, reproduced

`misc/scratch/ds41_vq_ceiling.cu` replays the dumped routes through the
served launchers over a 1.6 GB synthetic v3 blob (256 experts, so the
working set is 70x L2). Five reps, one warm pass discarded:

| arm | 12-bit | 13-bit |
| --- | ---: | ---: |
| real (served kernel) | 180.6 GB/s | 170.2 GB/s |
| pattern probe (same addresses, arithmetic off) | 232.3 | 207.7 |
| contiguous float4 over the same bytes | 253.9 | 254.3 |

The in-model 180 GB/s reproduces the harness to 0.3%, so the harness is the
model. The gather pattern is not the defect (232 of 254 = 91%); the kernel
runs 22% under its own read pattern.

## 3. Attribution, and two fixes measured neutral

Four arms walk the same spans with one component of the round loop removed
(`attr` mode in the same file, one pass per arm, host clock around the pass):

| arm | GB/s | what it removes |
| --- | ---: | --- |
| full | 124.5 | nothing (replica of the served loop) |
| nodot | 129.8 | codeword gather, 8 cvt, 8 FMA |
| nocw | 140.5 | codeword gather + 8 cvt |
| noidx | 221.6 | 2 shfl + funnelshift |
| stage | 126.3 | the 2 shfl, replaced by a per-warp shared index block |

Two readings, both negative:

- `noidx` is not usable: with a constant index the compiler drops the
  bitstream read, so its rate is computed over bytes it no longer moves. Its
  apparent 78% gain is that.
- The valid replacement for it, `stage` (the same 384 B index block staged
  per warp in shared, read with two LDS.32 per round, bit-identical
  indices and accumulation order), buys 126.3 against 124.5 — +1.4%, inside
  the noise. The shuffle chain is therefore not what costs the 22%.
- The codeword gather plus its 8 conversions is worth ~13% of the served
  rate at most (full to nocw), and the dot another ~4%. Together they do not
  close the gap to 232.

What would settle the residual is a stall census on the real kernel, and
that needs counters this host refuses to a non-root user:
`ncu --clock-control none …` returns `ERR_NVGPUCTRPERM` (verified
2025.3.1). Everything else has been measured.

## 4. What is left, in measured size

| lever | ceiling | cost |
| --- | ---: | --- |
| routed expert rate 180.6 → 232.3 | 4.0 ms/step, 6.6% | needs a stall census first; two cheap fixes measured neutral |
| q4_K rate 172 → its per-launch floors (185-190) | 1.9 ms/step, 3.1% | memory-system contention with the expert stream |
| head read twice per step | 1.7 ms/step, 2.9% | needs the draft to see verify logits, which it cannot |
| **decode kernel work, total** | **~10%** | |

## 5. The one large lever, and why it is not ours to take

Bytes per token are 5.76 GB at width 1, and the skeleton is 4.75 GB of the
width-2 step: attention 2.85 GiB, shared experts 0.86, head 0.36 read twice.
It runs Q4_K at 172 GB/s. The expert blob runs 8-dim VQ at 1.58-1.59
bits/weight at 180.6 GB/s, i.e. the same rate for 2.8x fewer bytes.

Applying VQ to the skeleton and the head would take that 4.75 GB to ~1.7 GB
and the step from 60.4 ms to about 42 ms, a 30-40% decode gain, at a quality
cost that has to be gated on our own positions.

It is not a kernel change. `/mnt/models/ds41-1017/RECIPE.md` pins the
skeleton to q4_K ("q4_K for every FP8 projection outside the routed experts
and for the BF16 embedding and output head"), and the quantizer's supported
skeleton roles are Q4_K, Q8_0, BF16 and F32 only
(`VQ_tools/docs/recipe-v41-identical.md:37`). A VQ skeleton is a new artifact
recipe, not this port's call: it produces a file the reference engine does
not load, so it also ends byte-comparability with `/data/YoungAi`.

## 6. Conclusion

The decode step is within 10% of what this artifact and this kernel
structure can reach on GB10. The routed-expert arithmetic that K2.1 ranked
first is real (22% of that stream's rate) but resists the two contained
fixes tried, and the rest of the kernel surface is at its floor. Further
decode work should be opened only with counters enabled, and the large
remaining gain is an artifact decision.
