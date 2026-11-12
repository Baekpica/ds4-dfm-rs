# DeepSeek V4.1 Flash hVQ: prefill, November 12

Three prefill rounds on
[DeepSeek-V4.1-Flash-1017 hVQ](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-hVQ-GGUF)
behind `ds4-server`, branch `feat/perf-ds41-flash` from `main` 2f7a19e5. VQ
and iVQ share every tensor shape and kernel with it. The reference engine is
YoungAi at `7fb26cc3`; each round names where it follows and where it leaves
that engine.

## Result

Full prompt prefill, tok/s, medians of three fresh server processes per build:

| Frontier (input tokens) | main 2f7a19e5 | This branch | Gain |
|---|---:|---:|---:|
| 2K (2,149) | 956.2 | 1,122.2 | +17.4% |
| 8K (8,449) | 1,032.4 | 1,414.7 | +37.0% |
| 16K (16,546) | 1,032.7 | 1,496.6 | +44.9% |
| 32K (32,749) | 934.9 | 1,457.8 | +55.9% |
| 64K (66,608) | 861.7 | 1,508.7 | +75.1% |

Over the 32 frontiers of the 2K–64K record the run mean rises from 955.5 to
1,448.3 tok/s (+51.6%). Time to first token at 64K falls from 77.4 s to 44.2 s. Decode kernels are
untouched; [Decode](#decode) covers what a prefill value change does to it.
The [2K–64K record](benchmarks/ds41-hvq-spark-64k-2026-11-12/README.md) has
the 32-frontier sweeps and the graph.

## Conditions

One DGX Spark (GB10, driver 615.71.10, CUDA 13.4.92, `sm_121a`); clock
service 300–2,200 MHz, busy SM clocks 2,184–2,190 MHz. Each run is a fresh
`ds4-server` under the host memory guard (2 GiB reserve and trip) on cores
5–9 and 15–19, `clear_cache` before it, vision sidecar attached,
`--mem-budget-mb 110000 --mem-floor-gb 2`. `speed-bench/ds41_spark_sweep.sh`
sends its 2,512-token calibration prompt and then one full-prefill request
per frontier: temperature 0, 128 tokens, automatic MTP, zero reused tokens.
An A/B is three fresh processes per arm, interleaved, on one binary with the
round's kill switch (round 1: two binaries); cells are medians.

## Where the time went

`nsys`, CLI prefill of 65,536 tokens on `main`: 31 blocks of 2,048 rows
through layers 0–19 and the layer-20 KV, the last block through all 40.

| Kernel class | ms per 2,048-row block | Share |
|---|---:|---:|
| Routed experts, BF16 register-direct (`vqs`) with gather and reduce | 602.5 | 29.9% |
| Index scores, scalar (113 ms for the block ending at 8K, 841 ms at 59K) | 449.8 | 22.3% |
| Dense chain: cuBLAS BF16 GEMMs and F32/BF16/Q4_K conversions | 477.7 | 23.7% |
| Sparse attention | 273.0 | 13.6% |
| mHC | 108.5 | 5.4% |

The same block took the server 0.25 s longer than the CLI. Round 1 is that
difference.

## Rounds

| Round | Bottleneck | Change | 2K | 8K | 16K | 32K | 64K |
|---|---|---|---:|---:|---:|---:|---:|
| `main` | | | 956.2 | 1,032.4 | 1,032.7 | 934.9 | 861.7 |
| 1. Engram prefetch | Every block waits 0.25 s on its first Engram layer | The prefetch buffer grows with the declared block | 945.7 | 1,135.2 | 1,154.7 | 1,044.5 | 972.7 |
| 2. Index scores | Scalar scores grow with the context: 35% of a block at 59K | The engine's s8 tensor-core kernel, on by default for prefill | 961.7 | 1,179.2 | 1,239.1 | 1,204.7 | 1,216.3 |
| 3. Routed experts | BF16 kernel rebuilds weights per fragment: 30.1 ms per layer | E4M3 activations on MiniMax's warp-specialized kernels | 1,122.2 | 1,414.7 | 1,496.6 | 1,457.8 | 1,508.7 |

Round 3's row and its off arm (961.7 / 1,181.0 / 1,241.4 / 1,206.9 /
1,221.6) are one interleaved A/B; rounds 1 and 2 are another.

### 1. The Engram prefetch buffer

The feed reads the next block's first Engram round while the current block
computes. Its buffer took the size of the first block ever declared and
never grew. `ds4-server` keeps the feed across requests, so one short first
declaration decided the whole process: the sweep's calibration prompt
declares its 464-token tail, and every later 2,048-token block ran without a
prefetch and waited on layer 1. A CLI process serves one prompt and never
showed it; the per-layer times of one prompt in both hosts did.

The buffer now grows where the fetch worker cannot be writing it. The same
rows reach the device: outputs are unchanged. `DS41_FEED_NO_PREFETCH` still
turns the prefetch off. Gain: +10.0% to +12.9% from 8K up.

A process whose first multi-block prompt declared a full block never had
the loss. The [hVQ record](deepseek41-hvq-evaluation-2026-11-11.md)'s one
pass was taken on such a server (957.6 / 1,134.6 / 1,156.6 / 1,043.4 / 980.0
tok/s) and agrees with this round's row within 1.3%: `main` in the tables
here is a fresh boot, which starts with the calibration prompt.

Rejected on the way: batching the io_uring waits. The ring already completes
276K reads/s alone, the drive's limit.

### 2. Index scores on tensor cores

A prefill block scores its 2,048 rows against every visible group of each
index source, so the scalar kernel's time grows with the context. The engine has
an s8 MMA kernel for it behind `--idx-mma`, off by default
(`cuda_v41_indexer_mma.inc.cu`). Queries and keys are E2M1 mantissas times a
power-of-two scale per 32 dims, so each block dot is exact in s32 and only
the order of the four block sums differs.

The port takes that kernel for eager prefill blocks by default, a deviation
in the default only. Decode, verify and captured steps keep the scalar
kernel. `DS4_DS41_IDX_MMA=0` restores it. Scores of the block ending at 8K:
113 ms → 8.5 ms; at 59K: 841 ms → 50 ms. Logits are identical in the model
(below).

### 3. Routed experts on FP8 tensor cores

Hardware bound per 2,048-row block and layer: 12,288 pairs reach up to 384
experts, so each expert's three matrices stream once, 2.77 GB on 13-bit
layers (2.56 GB on 12-bit): 11.8 ms at 235 GB/s. The products are 870 GFLOP:
8.1 ms at the BF16 `mma` peak, 4.0 ms at the FP8 peak. The BF16
register-direct kernel took 27.3 ms and 30.1 ms with its gather and reduce:
it builds each BF16 weight fragment from the E4M3 book.

A DQVL v3 payload has the row layout MiniMax's warp-specialized kernels read
(F16 gains, 12-bit indices, a 13th-bit plane), and its codewords are
`mma.m16n8k32.e4m3` operands as stored. Prefill blocks now take those
kernels (`cuda/ds41_vq_prefill_fp8.cuh`): inputs become E4M3 with one amax
scale per row and 128 dims, 12-bit layers pad their book and read a zero
plane. The kernels take 15.3 ms per layer, 16.8 ms with quantization, SwiGLU
and reduce. Over a block's 20 layers the kernels run at 75% of the stream
bound (230 ms for 54.1 GB against 306 ms).

Numerical class: a value change, and a named deviation from the engine,
which multiplies BF16 activations. The official model quantizes every GEMM
input to FP8 with a power-of-two scale per 32 dims. Decode and verify rows
keep their kernels. `DS4_DS41_VQ_FP8=0` restores the BF16 kernels.

The kernels failed at this shape before they ran in the model: the 13th-bit
plane ring was indexed by the unit's stage number, and a unit of 4n + 2
stages (down reads 2,304 columns, 18 stages) rewrote an entry two stages
early: 0.7% of down outputs differed. MiniMax and MiMo run 4n stages. The
fix and the shape are in `make test-minimax-mma8w`.

Short prompts gain too, one pair each: 325.7 → 367.5 tok/s at 151 tokens,
425.4 → 491.0 at 280, 528.8 → 615.9 at 566, 595.6 → 693.7 at 1,120.

## Numerics

Teacher-forced logits of the 26 held-out cases of the
[hVQ evaluation](deepseek41-hvq-evaluation-2026-11-11.md) (58,777 positions,
44,157 generated), same scoring entry, against the original model's logits
and against the build before the rounds. The noise floor is an existing
configuration change on this host: 1,024- instead of 2,048-token prefill
chunks.

| Arm | KL to original, all / generated | Same top-1 as original, all | NLL, all / generated |
|---|---:|---:|---:|
| Before the rounds | 0.2773 / 0.2467 | 84.38% | 1.3831 / 1.3603 |
| 1,024-token chunks (floor) | 0.2778 / 0.2463 | 84.32% | 1.3822 / 1.3601 |
| Rounds 1–2 | 0.2773 / 0.2467 | 84.38% | 1.3831 / 1.3603 |
| Rounds 1–3 | 0.2759 / 0.2464 | 84.38% | 1.3789 / 1.3605 |

| Paired with the build before | KL | Median | p99 | Same top-1 | Logit rel-RMS | ΔNLL (95% interval over cases) | ΔKL to original |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1,024-token chunks (floor) | 0.0154 | 0.00015 | 0.275 | 97.65% | 0.065 | -0.0009 (-0.0034 to +0.0013) | +0.0004 (-0.0019 to +0.0040) |
| Rounds 1–2, and both kill switches | 0 | 0 | 0 | 100% | 0 | 0 | 0 |
| Rounds 1–3 | 0.0199 | 0.00054 | 0.323 | 96.99% | 0.081 | -0.0042 (-0.0097 to +0.0004) | -0.0014 (-0.0035 to +0.0007) |

Rounds 1 and 2 leave every logit bit as it was. Round 3 moves logits 1.3
times as far as the chunk switch does and does not move them away from the
original: both distances to it stay inside their intervals. The model-free
index test (`make test-ds41-indexer`) holds the tensor-core scores to the
scalar kernel's bit for bit at model-like block scales; with scales 2^24
apart 0.2% of scores move by at most 1.4 BF16 steps.

## Decode

No decode kernel changed, and a round costs the same in both arms (59.1 ms
against 59.2 ms; verify 51.3 against 51.4). Round 3 still moves decode rates
per prompt: greedy continuations leave the BF16 arm's within their first
tokens, and the drafter's acceptance follows the text.

| Workload | Experts in prefill | Decode tok/s | Accepted drafts per round | Plain steps per request |
|---|---|---:|---:|---:|
| 5 frontiers, 2K–64K | BF16 | 34.08 | 1.081 | 4.4 |
| | FP8 | 31.98 | 0.985 | 11.0 |
| 32 prompts, 256–8,192 tokens | BF16 | 32.09 | 0.900 | 11.8 |
| | FP8 | 31.94 | 0.883 | 11.4 |
| 32 frontiers, 2K–64K, three runs each | BF16 (`main`) | 32.36 | 0.997 | 8.1 |
| | FP8 | 32.09 | 0.965 | 9.0 |

Five prompts cannot separate a text effect from an arithmetic one. Paired by
prompt, the 32 short prompts differ by -0.15 tok/s (standard error 0.24) and
-0.017 accepted drafts (0.023); the 32 sweep frontiers by -0.27 tok/s (0.37)
and -0.033 accepted drafts (0.040), FP8 ahead on 15 of 32. Neither shows a
decode cost.

## Limits

- One block in about a hundred waits 0.8–2.0 s on an Engram round, at layer
  1 or 14, in every build (55 of 3,360 blocks in the sweep record, whose
  receipt lists each). The reads are
  O_DIRECT, memory pressure stayed at zero and no reader path has such a
  timeout: the cause is below the engine and not isolated. It costs a
  frontier up to 15% in one run; medians hide most of it, the raw CSVs keep it.
- The FP8 arm needs the aligned device copy, as the BF16 register-direct
  kernel does. The model-free MoE fixture is not aligned, so the arm's item
  construction is covered by the in-model score alone.
- The numerical gate is teacher-forced on 26 cases of at most 8,192 tokens.
  Long-context decode after FP8 prefill is covered by the sweeps' decode
  account, not by a paired NLL.
- One host and one text corpus for speed.

## Next levers

`nsys` on this branch, same 65,536-token prefill: a 2,048-row block takes
1,301 ms of kernels, against 1,993 ms before.

| Kernel class | ms per block | Share | Note |
|---|---:|---:|---|
| Dense chain | 475.2 | 36.5% | cuBLAS BF16 GEMMs 173.5 at 55–59 of 108 TFLOPS; F32 → BF16 and Q4_K → BF16 conversions around them 201.8 (555 launches a block); `nvjet` GEMMs 99.8 |
| Routed experts | 336.7 | 25.9% | Kernels at 75% of the stream bound |
| Sparse attention | 273.4 | 21.0% | 13.7 ms per layer, constant across the context |
| mHC | 106.4 | 8.2% | |
| Index scores and top-k | 31.9 | 2.5% | |

- The dense chain is now the largest class and the cheapest to shrink:
  projections that take the F32 row and the Q4_K weight directly remove
  three conversion launches per GEMM.
- Attention next; experts are within a quarter of their DRAM bound, and a
  larger block would halve their stream per token.

## Sources checked

YoungAi `7fb26cc3` (`fable5.md`, `docs/speed.md`): the s8 index kernel passed
its byte-identical gates there and stayed opt-in (106K prefill 133.5 → 96.3
s); a tail-block split to exactly one window was tried and dropped for draft
acceptance (1.87 → 1.41); `speed.md` records the official precision:
FP8 activations, one power-of-two scale per 32 dims, for every GEMM.

[PR #144](https://github.com/Baekpica/ds4-dfm-rs/pull/144) (JLFN, closed
for the contribution checklist) proposed s8 and native MXFP4 index arms
among other changes and measured +23.8% at 64K on the V4.1 artifact.
Round 2 is a separate port of the engine's s8 kernel alone with its parity
test, and measures +25.0%.
