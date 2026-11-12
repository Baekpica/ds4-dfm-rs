# DeepSeek V4.1 Flash hVQ: the prefill dense chain, November 12

The follow-up to the [first three prefill rounds](deepseek41-optimization-2026-11-12.md),
branch `feat/perf-ds41-dense` from `main` 5a9edb95 (PR #148 merge). Same
artifact, host, protocol and gates; the baseline is that record's final
build.

## Result

Full prompt prefill, tok/s, the fastest of three fresh server processes per
build (`main`: the [2K–64K record](benchmarks/ds41-hvq-spark-64k-2026-11-12/README.md)'s
final runs at the same frontiers):

| Frontier (input tokens) | `main` 5a9edb95 | This branch | Gain |
|---|---:|---:|---:|
| 2K (2,149) | 1,120.9 | 1,293.1 | +15.4% |
| 8K (8,449) | 1,410.7 | 1,624.0 | +15.1% |
| 16K (16,546) | 1,497.4 | 1,715.2 | +14.5% |
| 32K (32,749) | 1,449.5 | 1,680.9 | +16.0% |
| 64K (66,608) | 1,500.9 | 1,643.2 | (all three runs stalled) |

Kernel sum per 2,048-row block of a 64K prefill: 1,301 → 1,135 ms (−12.8%).
Decode kernels are unchanged. The [2K–64K record on `main` 58738c04](benchmarks/ds41-hvq-spark-64k-2026-11-12-dense/README.md)
has three fresh 32-frontier sweeps after both PRs and the graph.

## Conditions

As in the [first rounds](deepseek41-optimization-2026-11-12.md#conditions): one
DGX Spark (GB10, driver 615.71.10, CUDA 13.4.92, `sm_121a`), clocks
300–2,200 MHz (busy 2,184–2,190 MHz), a fresh guarded `ds4-server` per run,
`speed-bench/ds41_spark_sweep.sh` after its calibration prompt, full prefill,
128 greedy tokens, automatic MTP. Each round's A/B interleaves three fresh
pairs: round 4 on one binary with its switch, round 5 on two binaries. The
result table takes `main` from the record's final runs, three hours earlier.

Engram reads stalled more often in this session: 2–6 of a run's 67 prefill
blocks waited 0.8–2.0 s on a round, in both arms, and some requests lost up
to 3 s outside the per-layer timers. A stall only adds time, so a cell below
is the fastest of three runs; the receipts keep every sample.

## Where the time went

`nsys`, 65,536-token CLI prefill, per 2,048-row block of layers 0–19 (kernel
sum 1,301 ms on `main`). The dense chain was 475 ms of it:

| Piece | ms per block | Note |
|---|---:|---|
| cuBLAS BF16 GEMMs (cutlass `Kernel2`) | 173.5 | q_b, wo_b, shared gate and up, engram wkv at 52–58% of the 108 TFLOPS peak |
| `nvjet` GEMMs | 99.8 | wo_a, kv, q_a, shared down, router gate; mHC's TF32 GEMM (32.3) |
| Q4_K → BF16 weight copies | 42.9 | every call dequantizes the whole matrix: 324 MB of BF16 per layer from 91 MB of Q4_K |
| F32 → BF16 inputs | 74.2 | `xn` converted five times per layer, `o` (268 MB) once |
| BF16 rounds of F32 outputs | 84.7 | q_b's 268 MB output alone 44.3 |

The q4_K projections are 663 GFLOP per layer (13.3 TFLOP per block): 123 ms
at the F16 `mma` peak.

## Rounds

### 4. q4_K projections on FP16 tensor cores

The GLM VQ family's prefill kernel (`glmvq_q4p`, `cuda/glm_vq_q4p.cuh`) reads
raw GGUF Q4_K rows: producer warps dequantize each weight once into MMA
fragments, consumers run an F16 GEMM, and a per-token prep turns F32 rows
into its F16 operand. A model-free probe at ds41's shapes and 2,048 tokens
(per call; cuBLAS from the trace):

| Projection | q4p | cuBLAS GEMM alone |
|---|---:|---:|
| q_b (1280 → 32768) | 2.47 ms, 70 TFLOPS | 2.89 ms |
| wo_b (8192 → 5120) | 2.32 ms, 74 TFLOPS | 3.07 ms |
| shared gate, up (5120 → 2304) | 0.63 ms, 77 TFLOPS | 0.77 ms |
| shared down (2304 → 5120) | 0.68 ms | 0.66 ms |
| q_a (5120 → 1280) | 0.39 ms | 0.36 ms |
| kv (5120 → 512) | 0.19 ms | 0.14 ms |

`cuda/ds41_q4k.cuh` routes every rounded q4_K projection of a prefill block
(n > 8) to it: one prep, one GEMM whose epilogue writes the BF16-rounded F32
value the round pass wrote. No q4_K tensor of the artifact leaves the
kernel's folded F16 scale range (a CPU scan of all 329; the engine checks
each once per process). The head (unrounded) and grouped wo_a stay on
cuBLAS in this round. `DS4_DS41_Q4P=0` restores cuBLAS.

The prefill MoE tail went to one pass in the same build: it copied the
routed sum into `y`, added the shared expert and rounded; `y = bf16(routed +
so)` is the same F32 add (logits bit-identical, 24.5 → 10.6 ms per block).

Kernel sum per block: 1,301 → 1,165 ms. `glmvq_q4p` and its preps take 182
ms where the GEMMs, weight copies, input conversions and rounds of the same
projections took 318.

| Frontier | `DS4_DS41_Q4P=0` | q4p | Gain |
|---|---:|---:|---:|
| 2K | 1,128.3 | 1,267.8 | +12.4% |
| 8K | 1,422.2 | 1,590.6 | +11.8% |
| 16K | 1,511.7 | 1,687.2 | +11.6% |
| 32K | 1,477.6 | 1,642.5 | +11.2% |
| 64K | 1,324.6 (stalled in all three) | 1,698.0 | |

### 5. Shared operands and the output pair

After round 4 (`nsys`, same prefill, per block), wo_a still ran the cuBLAS
route: the F32 → BF16 conversion of the attention output `o` (268 MB, 33.0
ms), the weight copy (8.7), eight `nvjet` GEMMs (37.3) and the round (10.6).
Four q4p preps of `xn` per layer (kv, q_a, gate, up) wrote the same operand
twice.

- `ds4_gpu_v41_matmul_q4k_pair_tensor`: kv and q_a, and the shared gate and
  up, from one prepared `xn` (kv's norm, rope and quantization now follow
  both GEMMs; independent buffers).
- `ds4_gpu_v41_out_proj_q4k_tensor`: `o` is prepared as eight group planes
  (`v41_q4p_prep_seg`), each group's q4p GEMM writes its BF16 output to a
  plane, and wo_b's prep reads the planes back as rows; `low` is never
  written. Values equal wo_a and wo_b on q4p one by one
  (`make test-ds41-q4p` compares the bits).

Kernel sum per block: 1,165 → 1,135 ms. wo_a and wo_b take 131 ms per
block (q4p GEMMs 89, preps 42) where they took about 145 (wo_a's cuBLAS
route 90, wo_b's prep and GEMM 55); `glmvq_q4p_prep` falls from 34.2 to
13.6 ms.

| Frontier | Round 4 | Round 5 | Gain |
|---|---:|---:|---:|
| 2K | 1,260.7 (1,254.1–1,260.7) | 1,293.1 (1,291.9–1,293.1) | +2.6% |
| 8K | 1,587.1 (1,580.8–1,587.1) | 1,624.0 (1,617.6–1,624.0) | +2.3% |

Fastest of three, the range in brackets; 16K–64K stalled in two or three
runs of an arm and are left out of the comparison (receipts keep them).

## Numerics

The weights round to F16 twice, fp16(fp16(q S) − M) with S and M the
sub-block scale and min (times 2^6), against one BF16 rounding before: half
the error against the exact Q4_K value (`make test-ds41-q4p`: 1.7e-3 against
2.4e-3 relative RMS, both including the BF16 output rounding). BF16-valued
inputs are exact in F16's normal range. A named deviation from the engine,
which runs bf16(w) × bf16(x) on cuBLAS; the engine's own record calls BF16
prefill weights a precision loss against the GGUF.

Teacher-forced logits of the 26 held-out cases (58,777 positions), against
the original model and paired with `main`; the floor is the 1,024- vs
2,048-token chunk switch on this host:

| Arm | KL to original, all / generated | Same top-1 as original | NLL, all / generated |
|---|---:|---:|---:|
| `main` | 0.2759 / 0.2464 | 84.38% | 1.3789 / 1.3605 |
| Round 4 (tail fold alone, `DS4_DS41_Q4P=0`) | 0.2759 / 0.2464 | 84.38% | 1.3789 / 1.3605 |
| Round 4 | 0.2759 / 0.2468 | 84.31% | 1.3780 / 1.3599 |
| Rounds 4–5 | 0.2781 / 0.2463 | 84.33% | 1.3802 / 1.3591 |

| Paired with `main` | KL | p99 | Same top-1 | Logit rel-RMS | ΔNLL (95% interval) | ΔKL to original |
|---|---:|---:|---:|---:|---:|---:|
| Chunk switch (floor) | 0.0154 | 0.275 | 97.65% | 0.065 | -0.0009 (-0.0034 to +0.0013) | +0.0004 (-0.0019 to +0.0040) |
| Tail fold alone | 0 | 0 | 100% | 0 | 0 | 0 |
| Round 4 | 0.0206 | 0.322 | 96.82% | 0.085 | -0.0009 (-0.0073 to +0.0040) | 0.0000 (-0.0025 to +0.0027) |
| Rounds 4–5 | 0.0208 | 0.346 | 96.83% | 0.084 | +0.0012 (-0.0042 to +0.0066) | +0.0022 (-0.0008 to +0.0055) |

Both arms move logits 1.3 times as far as the chunk switch, as the FP8
expert round did. Round 5 adds wo_a's F16 weights; its distance to the
original moves +0.0022 over all positions and -0.0001 over generated ones,
both intervals across zero.

## Decode

No decode kernel changed. Greedy continuations change with the prefill
arithmetic, and decode with them: over the five A/B frontiers 32.0 tok/s
without q4p and 32.8 with it; round 5 32.9 and 32.8 tok/s with 1.02 and
1.03 accepted drafts per round, and the same round time (59.3 and 60.1 ms).

## Next levers

`nsys` on this branch, same 65,536-token prefill, per block (1,135 ms):

| Kernel class | ms per block | Share |
|---|---:|---:|
| Routed experts (FP8 kernels, reduce, quantization) | 334.9 | 29.5% |
| Sparse attention | 270.4 | 23.8% |
| q4_K projections (`glmvq_q4p` 189.1, preps 55.4) | 244.5 | 21.5% |
| mHC (post 62.5, fused 44.0, mix GEMM 33.0, row rsqrt 31.5) | 171.0 | 15.1% |
| Engram wkv and gate | 31.6 | 2.8% |
| Index scores and top-k | 31.3 | 2.8% |

- mHC's mix reads the 168 MB hyper state twice per half-layer (the row rsqrt
  and the TF32 GEMM, both at DRAM speed): one pass would take about 32 ms
  off a block.
- The attention output's prep re-reads the 268 MB F32 `o` (1.7 ms per
  layer): the attention epilogue could write q4p's F16 operand planes
  after the inverse rope.
- Sparse attention runs at about 13 TFLOPS, 13.5 ms per layer.
