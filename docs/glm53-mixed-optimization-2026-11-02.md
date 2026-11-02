# GLM-5.3 Flash Uncensored mixed quant: prefill and decode, November 2

Optimization rounds on the
[mixed IQ2_XXS/IQ2_XS/Q2_K artifact](https://huggingface.co/Baekpica/GLM-5.3-Flash-Uncensored-Mixed-Quant-GGUF)
(the [Uncensored record](glm53-uncensored.md)), branch `feat/perf-glm-mxq`
from `main` 8c2068b5. The compact (non-VQ) GLM graph reused few of the GLM
VQ path's kernels; most rounds move a VQ kernel onto this artifact's
formats (F16 latent cache and pools, F32 residuals, Q8_0/Q4_K projections).

## Result

| Workload | main 8c2068b5 | This branch |
|---|---:|---:|
| 2K-64K sweep, prefill mean | 183.2 tok/s | 784.7 tok/s (4.28x) |
| 2K-64K sweep, decode mean | 15.09 tok/s | 18.92 tok/s (+25.4%) |
| Cold 8K prefill | 215.9 tok/s | 805.9 tok/s (3.73x) |
| Decode after it, 64 tokens | 15.17 tok/s | 19.06 tok/s (+25.6%) |

[Sweep record](benchmarks/glm53-mixed-spark-64k-2026-11-02/README.md): three
final runs, one base run, 2,048-token steps, 128 greedy tokens, MTP off.

## Conditions

One DGX Spark (GB10, driver 615.71.10, CUDA 13.4.92, `sm_121a`), clock
service 300-2,200 MHz (enabled, applied at boot 2026-10-27). One raw VMM
weight owner (`ds4_weight_server --backend vmm`, no repacks); every arm is
a fresh `ds4-bench-perf` worker. Prompt `speed-bench/promessi_sposi.txt`,
8,192-token cold prefill in 2,048-row chunks, 64 greedy tokens, MTP off,
context allocation 32,768. Each A/B is two interleaved fresh pairs on one
binary with the round's kill switch, unless noted.

## Bounds

| Phase | Work per token | Floor on GB10 |
|---|---|---|
| Decode, MTP off | 10.52 GB: KDA projections 3.85, routed experts (8 of 288) 2.30, DSA 1.54, shared experts 1.12, head 0.67, dense 0.48, router 0.20, KDA state read/write 0.29 | 21.9-23.8 tok/s at 230-250 GB/s |
| Prefill, 8K | 35.7 GFLOP: routed 16.9, Q8/Q4 projections 14.7, DSA attention 3.0, absorption 0.4, indexer and KDA ~0.5; 90 GB of weights per 2,048-row chunk | ~6,000 tok/s at the INT8 `mma` peak, ~3,000 at F16; weight stream ~5,500 |

The VQ artifact of the same model reaches 1,343 tok/s (8K) on this engine,
the practical prefill reference.

## Prefill rounds

| Round | Bottleneck (share of the 8K prefill) | Change | 8K prefill tok/s |
|---|---|---|---:|
| Baseline | | | 216.1 |
| Flash attention | Selected latent attention 41.7%: one SIMT block per (row, head), each head re-reading the row's 2,051 keys | GLM VQ flash tile as a kernel template; F16 instance on the compact cache | 216.1 -> 362.9 (+68%) |
| Absorbed GEMMs | Q absorption + V projection 32% of what remained: SIMT re-reading Q8_0 K_b/V_b per (row, head) | K_b/V_b widened to F16 per call (KDA chunk scratch), 64-head strided-batched F16 GEMMs around the flash tile | 362.6 -> 532.2 (+47%) |
| Rank-128 SGEMM | KDA f_b/g_b (Q8_0, K=128) 13.8%: K < 256 fell to a warp-per-row SIMT kernel | F32-widened weight + one TF32 SGEMM | 531.9 -> 612.9 (+15%) |
| Index scores | Pool scores 18.3% of a 32K prefill: one block per (pool, row) re-reading the row's query | GLM VQ fused 4-row x 128-pool MMA tile, F16 instance | 612.9 -> 646.8 (+5.5%); 32K 527.4 -> 640.5 (+21.5%) |
| Dead scrub | Non-finite scrub after every routed MMQ (0.5 s); consumers already guard | Skipped for GLM's routed MMQ | 646.3 -> 664.6 (+2.8%), byte-identical |
| Pipelined tiles | Routed MMQ = a + b/rows (6.20 s at 1,024 rows, 4.58 s at 2,048); b matches the 1.95 GB/layer expert stream: 128-column tiles ran the unpipelined K loop | IQ2 worklists take the pipelined 64-column tile | 664.6 -> 675.8 (+1.7%), byte-identical |
| mHC mix | RMS (written out), BF16 widening and a 24-output GEMM: four passes, 1.0 s | GLM VQ one-pass F32 mix kernels (prefill and decode) | 673.2 -> 723.3 (+7.4%); decode +3.3% |
| KDA | Solar F32 chunk kernels + gated head norm 1.64 s | GLM VQ tensor-core local + scan with the gated norm | 723.7 -> 805.5 (+11.3%) |

## Decode rounds

| Round | Bottleneck (ms of a decode token) | Change | Decode tok/s |
|---|---|---|---:|
| Baseline | | | 15.15 (8K) |
| Key-split attention | 8.0 of 64 ms: one SIMT block per head, 725 us per layer | Flash tile with keys split over 48 CTAs + combine; partials in the KDA scratch | 15.58 -> 17.59 (+12.9%, 8K) |
| Q4_K GEMV | KDA q/k 7.1 ms: one row padded to an 8-column MMQ tile (181 GB/s) | MMVQ GEMV for GLM's one-row Q4_K | 17.95 -> 18.39 (+2.5%, 2K) |
| Latency kernels | Router top-8 on one thread (16.7 us x 42), separate 4096-wide RMS after each mHC split (9.6 us x 90), 126 routed scrubs | Warp argmax with the serial scan's tie order; existing fused split + norm; scrub skipped | 17.79 -> 17.95 and 18.39 -> 18.55 (2K), byte-identical |
| L2 prefetch | Q8_0/Q4_K GEMVs (54% of a token) stall on DRAM latency | llama.cpp #26705 backport: prefetch the weight blocks two iterations ahead (Spark only) | 18.59 -> 19.03 (+2.4%, 2K), byte-identical |
| Row-dot projections | Decode K_b/V_b rows walked by one thread each (83 + 77 us per layer) | One warp per Q8_0 row, lanes on consecutive bytes | 19.03 -> 19.31 (+1.5%, 2K) |

## Numerics

Each value change is gated on teacher-forced NLL (`tests/glm53_nll`: 512
positions after an 8,192-token prefix, 256 after 32,768) against the
reference path, with the prefill chunk-size switch (1,024 vs 2,048 rows)
as the noise floor. Single-frontier logits are chaotic here: the chunk
switch alone moves 8K frontier KL by 6.6e-2 and splits greedy streams
after 9 tokens.

| Arm | 8K paired d (se) | 32K paired d (se) |
|---|---:|---:|
| Chunk-size switch (floor) | -0.011 (0.009) | +0.013 (0.013) |
| All prefill rounds + key-split decode | -0.005 (0.009) | +0.004 (0.012) |
| Final (all rounds) | -0.001 (0.009) | -0.015 (0.012) |

On an English story prompt (`tests/long_context_story_prompt.txt`, 512
positions after 8,192 tokens, all switches off as the reference) the final
build moves NLL by +0.061 (se 0.042) and the chunk-size switch by +0.099
(se 0.043); top-1 hits are 477 and 475 of 512 against the reference's 483.

Value changes: F16 query, P and latents in attention; F16 K_b/V_b and
query in the absorbed GEMMs; TF32 operands in the rank-128 projections;
F16 index queries; F32 (not BF16) activations in the mHC mix; the VQ
recurrence's BF16 boundaries (q/k/v after conv and SiLU, beta, output) in
prefill KDA. Model-free fixtures (`make test-glm53-flash`) check each path
against FP64 or the production reference kernels.

`ds4 --serving-check` on this artifact (owner import, essay prompt, ctx
8,192, 2,048-row grid) passes all ten reuse, rewind, continue, disk,
snapshot and bank items; continue matches 16 of 17 top-1 tokens with one
near-tie (margin 0.83) between prefill- and decode-fed rows.

## Rejected

- 4,096-row prefill chunks (Rust and native caps raised locally): 8K
  prefill 722.8 -> 676.7 tok/s (-6.4%) while the workspace grew 2.87 ->
  5.74 GiB; the MoE fit above predicted +8%.

## Next levers

- Routed MMQ is 42.6% of the 8K prefill at ~32 TOPS (INT8 peak 215): an
  IQ2 grid-codeword MMA kernel in the GLM VQ expert style.
- MTP: verification runs width-one forwards; GLM VQ's batched verify
  (per-row KDA journal, indexer stash) needs this graph's state paths.
- Dense MMQ re-quantizes the same input per projection (KDA 6x, DSA 4x):
  ~2% of the 8K prefill.
- mHC expand + next mix in one F32 pass (~2.5%): the VQ kernel rounds the
  residual to BF16.
- Decode GEMVs and expert vec kernels run near the DRAM floor; past the
  latency kernels, decode speed comes from MTP or fewer bytes.

## Sources checked

Upstream `antirez/ds4` (main and 0aaea5a) and about 25 of its forks: no
GLM CUDA speed work after 0aaea5a; its Q absorption already used a
F16-widened K_b with batched cuBLAS, and llama.cpp runs GLM5-next DSA
through MMA flash attention over the sparse mask. FlashInfer's
`sparse_mla_sm120` (FP8, SM12x) is the vLLM backend for this model on
GB10.
