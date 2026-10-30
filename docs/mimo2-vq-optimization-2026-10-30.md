# MiMo-V2.6-Flash VQ optimization, 2026-10-30

GB10 (sm_121, CUDA 13.4), busy SM clock 2,184–2,190 MHz under the
300–2,200 MHz policy, UNCENSORED release (`f1cd8523`). Sweeps: ds4-bench,
`speed-bench/promessi_sposi.txt`, 2,048-token frontiers 2K–64K, 128 greedy
tokens each, fresh process, memory guard (2 GiB reserve; minimum available
24.5 GiB, PSI 0). [Family page](mimo2-vq.md).

## Baseline

The first working build (`eb07ac7d`) averaged **597.9 / 20.72 tok/s**
(prefill / decode) over the sweep: 1,301.8 / 24.73 at 2K, 289.7 / 17.28 at
64K. Prefill fell with context: each added key cost 84 µs per 2,048-row
chunk, nine full layers' attention at 8.9 TFLOPS (8% of the F16 peak).
Decode lost 0.27 µs per key per token: the FP8 rows streamed at 43 GB/s.

## R1: full-layer prefill attention (`mimo2_ga8::prefill`)

The old kernel widened FP8 rows with scalar byte loads, synchronously, once
per 64 (row, head) pairs, and ran Q and P as two f16 components (twice the
MMAs). The new one: eight warps hold two query heads of one KV head (128
pairs per tile), 64-key tiles, the next tile's codes loaded into registers
(LDG.128) during the current tile's MMAs, packed E4M3→f16 conversion, ldmatrix
fragments, single f16 Q and P, exp2 softmax. Model-free harness
(`make test-mimo2-ga8`, 2,048 rows):

| Frontier | Old | New |
|---:|---:|---:|
| 4,096 | 32.6 ms (13.2 TFLOPS) | 5.8 ms (74.0) |
| 32,768 | 318.5 ms (8.9) | 40.8 ms (69.6) |
| 63,488 | 616.6 ms (8.8) | 79.8 ms (67.8) |

FP64 error over the stored codes stays within 4× the old kernel's (0.004 vs
0.002 absolute on unit-variance values). Sweep: **1,129.7 / 20.72**
(+89%); 64K frontier 289.7 → 936.6.

Rejected: the same kernel for the f16 SWA rows. At 4,096 rows their call is
bound by F32 Q and output traffic (2.50 → 2.47 ms), and one f16 Q raised the
FP64 error from 5e-6 to 3e-3 against exact f16 keys.

## R2: warp-specialized prefill experts

A 4,096-row cold chunk (2,544 ms, nsys) spent 47.6% in routed experts,
24.4% in Q8_0 MMQ projections and ~20% in bandwidth-bound elementwise passes;
attention was 5.8% at 4K. MiMo's experts now run MiniMax's
`cuda/minimax_mma8w.cuh` kernels (64-pair items, producer warps, codewords as
m16n8k32 B fragments) with MiMo's gate-form SwiGLU (`mw_swiglu_quant` gained
a clamp template). Frontier logits and greedy tokens equal the
`glmvq_mma8_kernel` path byte for byte at 4K and 8K. 4,096-row chunks:
1,472.5 → 1,586.4 tok/s at 4K (+7.7%), 1,435.2 → 1,548.5 at 8K (+7.9%).

## Remaining prefill costs (4,096-row chunk, after R2)

| Kernel | ms | Share | Against |
|---|---:|---:|---|
| `mw_kernel` (47 × gate, up, down) | 1,210.6 | 47.6% | MMA floor ~607 ms (R13+8 two MMAs) |
| `mul_mat_q` Q8_0 qkv / o_proj / dense | 619.7 | 24.4% | 59 TOPS of the 215 INT8 peak |
| SWA attention (39 layers) | 103.0 | 4.0% | F32 Q/output traffic |
| `mimo2_split_rope` | 93.2 | 3.7% | 229 GB/s |
| expert reduce, SwiGLU quant | 131.6 | 5.2% | bandwidth |
| MMQ sanitize + Q8_1 quantize | 117.2 | 4.6% | bandwidth |
| full-layer attention (4K) | 45.0 | 1.8% | 69 TFLOPS |

## R3: full-layer decode and verify attention (`mimo2_ga8::decode`)

The split walk read FP8 rows at 41 GB/s: per key, a warp-wide shuffle
reduction and an online-softmax update per query head, scalar decodes into
shared tiles, and one launch per verify row. The new kernel treats the 16
query heads of a KV head as the m16 rows of f16 MMAs, so one row of codes
serves all of them. The k order is chosen so a lane's fragment is contiguous
in memory: K covers dims 48t..48t+47 of one key, the output dims
16g..16g+15 of four keys' values (byte-permuted into the B fragment). The
four KV heads of a key slice share a block, which stages whole 32-row steps
with contiguous cp.async into a two-stage ring; K scales multiply the
scores, V codes widen to f16(code × scale). Verify rows run in the same
launch and meet in L2. A slice spans at least 128 keys of the live position
(read on the device under graph replay), and the slices fill one wave.

| Rows, frontier | Split walk | New | |
|---|---:|---:|---:|
| 1, 4,100 | 0.123 ms | 0.045 ms | 118 GB/s |
| 1, 65,536 | 2.098 ms | 0.468 ms | 182 GB/s |
| 1, 262,144 | 8.325 ms | 1.783 ms | 191 GB/s |
| 4, 65,536 | 8.461 ms | 0.449 ms | 189 GB/s per row set |

Steps on the way: one warp per (slice, KV head) with per-lane loads
(126 GB/s) and the same with a register or per-lane cp.async prefetch
(124–137 GB/s) were latency- and pattern-bound; a read-only probe of that
pattern capped at 150 GB/s against 211–222 for whole rows, which the final
staging reads. FP64 error stays within 4× the walk's on every case.

## Sweep after R1–R3

| 2K–64K sweep, one fresh process | Prefill | Decode |
|---|---:|---:|
| Baseline `eb07ac7d` | 597.9 | 20.72 |
| R1 | 1,129.7 | 20.72 |
| R1–R3 | 1,227.4 | 24.14 |

Per frontier (R1–R3): 1,486.1 / 25.03 at 2K, 1,216.4 / 24.15 at 32K,
998.3 / 23.21 at 64K. Decode lost 0.27 µs per key and token before R3,
0.05 after. Three fresh processes per variant
([publication sweep](benchmarks/mimo2-vq-spark-64k-2026-10-30/README.md))
recorded 1,226.5 / 23.97 without drafting and 1,228.4 / 25.16 with two
MTP drafts.

## Numerical gates

R1 and R3 change rounding (single f16 Q and P, exp2, 64-key tiles, K scale
after the dot); R2 is bit-identical. Both gates use `tests/mimo2_vq_quality`.

**Assistant turns against the teacher** (24,785 targets of the handoff's 14
chat rows, UNCENSORED teacher, FP8 K/V): KL 0.0897 → 0.0891, top-1 91.81 →
91.75%, NLL 0.7491 → 0.7463 (teacher 0.7421); the gate allows KL ≤ 0.109 and
top-1 ≥ 89.7%.

**32K frontier** (`--long`: 32,768-token prompt of `promessi_sposi.txt`,
then 1,024 teacher-forced positions, paired against F16 K/V rows):

| Configuration against F16 K/V | ΔNLL ± s.e. | mean \|ΔNLL\| | top-1 agreement |
|---|---:|---:|---:|
| R1–R3 (default) | +0.0023 ± 0.0042 | 0.072 | 95.9% |
| `DS4_MIMO2VQ_GA8=0` | −0.0039 ± 0.0044 | 0.076 | 95.8% |

The accepted `DS4_MIMO2VQ_GAIN_DOT=0` switch moves the default by 0.066 /
96.6% (ΔNLL +0.0026 ± 0.0044). Prefill chunks of 2,048 and 4,096 rows give
identical logits (every kernel is row-independent), so chunking is no noise
reference here.

## Drafting

Verify widths reuse the decode kernels: experts on the rows' union, R3's
attention for every verify row in one launch. Two changes for drafting:
the verify rows (and DFlash's proposals) pass through the 0.62 GiB head
in one matmul rather than one per row, and DFlash gathers its 1,024-slot
tap window with at most two device copies instead of 1,024 host
round trips and one upload. `ds4-bench`, 256 forced tokens, one process
each:

| Decode at the 2K / 32K frontier | tok/s | Tokens per cycle |
|---|---:|---:|
| Plain | 25.07 / 24.15 | 1 |
| Embedded MTP, draft 2 | 30.14 / 26.32 | 2.29 / 2.02 |
| Embedded MTP, draft 3 | 28.79 / 24.66 | 2.51 / 2.17 |
| DFlash, draft 7 | 26.61 / 17.43 | 3.12 / 2.05 |

The DFlash fixes took it from 3.19 to 26.61 tok/s at 2K (same draft 7). A
draft-2 cycle costs 76 ms at both frontiers (1.9 plain steps), a DFlash
cycle 117 ms: eight verify rows read the union of their experts, and each
trial still re-projects its 1,024 context taps. Emitted
tokens follow the verify forward's own argmax: the first token after each
prompt matches plain decode, later ones diverge at FP near-ties (attention
summation order differs between one and three rows).

## Next levers

- **Short-context prefill.** Routed experts take 48% of a 4,096-row chunk at
  about half their MMA floor (two MMAs on R13+8 layers); Q8_0 projections run
  MMQ at 59 of 215 INT8 TOPS (24%); bandwidth-bound passes (RoPE split,
  expert reduce, SwiGLU quantization, MMQ scrub and Q8_1 quantization) take
  about 20%. Reading Q and K/V straight from the qkv rows with RoPE applied in
  the attention and store kernels would remove the 3.7% split pass.
- **Long-context prefill.** Full-layer attention runs at 63-69% of the f16
  peak; at 64K it is still a third of the chunk. E4M3 Q and P on FP8 MMAs
  would double the peak and need their own long-context gate.
- **Decode.** Weight streams run at 220-245 GB/s (the token's 8.4 GB floor is
  about 36.5 ms of the 39.9 ms step); the expert down kernel reaches 182 GB/s.
- **Drafting.** The embedded MTP blocks draft blind: their K/V holds only
  earlier draft rows at a cursor from 0, never the prompt (since #57), and a
  restored snapshot (each sweep frontier) empties it. Filling it during
  prefill and from verify rows at absolute positions is the MTP lever. A
  DFlash trial still re-projects its 1,024 context taps and verifies eight
  rows (117 ms); incremental context K/V rows and a margin gate (Beam's
  DFlash2) are the next steps. SWA verify rows use the walk kernel rather
  than the one-row tile.
