# Beam-1.0 VQ prefill and decode optimization on GB10, October 21

This round starts at merged main `dab8bd75` (the Beam VQ family, PR #105) and
ends at `391b0bf1`. Results apply to the
[Beam-1.0-501B-A23B VQ release](https://huggingface.co/Baekpica/Beam-1.0-501B-A23B-VQ-GGUF)
(`faf8f1b0` parts) on one DGX Spark. The goal was to bring the largest prefill
and decode kernels toward the floors this host can reach; the DFlash2 drafter's
own kernels were not tuned.

## Floors

Measured on this GB10 (driver 615.71.09, CUDA 13.4, `sm_121a`) at the busy
SM clock of 2,190 MHz under the host's 300–2,200 MHz cap; both MMA peaks are
48 SMs × 2,048 (FP8) or 1,024 (F16) FLOP per clock at that clock:

| Quantity | Measured |
|---|---:|
| Streaming read, model-free (LPDDR5X spec 273 GB/s) | 241–246 GB/s |
| `mma.sync` E4M3 m16n8k32, FP32 accumulate | 215.6 TFLOPS |
| `mma.sync` F16 m16n8k16, FP32 accumulate | 107.8 TFLOPS |

A decode kernel's floor is its bytes at 246 GB/s. A 12-bit expert projection's
prefill floor is every touched expert matrix read once: 627 MB, 2.57 ms per
2,048-row chunk (its FP8 MMA work is 2.0–2.2 ms). Attention's is its FLOPs at
the F16 peak.

## Results

The [2K–64K sweep](benchmarks/beam-vq-spark-64k-2026-10-21-tuned/README.md)
(2,048 new tokens and 128 greedy tokens per frontier, three fresh processes per
variant) against the [same sweep at `dab8bd75`](benchmarks/beam-vq-spark-64k-2026-10-21/README.md):

| Workload | `dab8bd75` | `391b0bf1` |
|---|---:|---:|
| Sweep 2K–64K, MTP off: prefill / decode tok/s | 830.4 / 15.86 | **1,241.2 / 25.23** |
| Sweep 2K–32K, MTP off | 881.5 / 18.70 | 1,326.0 / 26.03 |
| Sweep 2K–64K, DFlash2 (draft 7, margin 2) | 821.1 / 15.41 | 1,225.7 / 24.73 |
| Decode at 2K / 64K | 22.57 / 11.26 (−50%) | 26.61 / 23.67 (−11%) |
| Prefill at its peak / 64K | 925.7 (6K) / 730.7 | 1,413.7 (4K) / 1,087.2 (−23%) |
| Cold one-shot 8K prefill (`ds4-bench` 8192/16) | 907.5 | 1,385.2 |
| Five greedy chat prompts: no drafter → DFlash2 (draft 7, margin 2) | 24.0 → 30.9 | 26.23 → 32.03 |

![Beam-1.0 VQ prefill and decode on one DGX Spark](benchmarks/beam-vq-spark-64k-2026-10-21-tuned/throughput.png)

## Rounds

**R1, RoPE rows once per forward (`30c071e7`).** `beam_qk` evaluated pow,
cos and sin (log1p and exp on the full layers) in double for every element;
GB10 runs FP64 at 1/64 of the FP32 rate, and the q/k norms were 22% of an 8K
prefill. A kernel at the start of each forward writes every row's cos, sin and
query scale with the same double math. 8K prefill 907.5 → 1,140.1 tok/s;
frontier logits and 64 greedy tokens byte-identical.

**R2, tiled split-key decode attention (`ac00aea0`).** The full and sliding
layers decoded on kernels that walked keys one at a time per warp, about
25 GB/s on the KV rows (47 ms per token for the full layers at 64K). The new
kernel splits each KV head's keys into about 48 chunks, serves its six query
heads together and scores 32-key tiles with one lane per key. Decode 8K
21.1 → 25.6, 64K 11.3 → 21.0 tok/s; decode NLL −0.68% against the GB10
reference rerun (band 1%). `DS4_BEAM_ATTN_LEGACY=1` restores the old kernels.

**R3, TMA for the FP8 decode rows (`b764c296`).** The SIMT kernel read FP8
rows at about 135 GB/s: per-lane loads fetched every 32-byte sector twice. On
sm_90+ one block per key split serves all eight KV heads and takes each
2,080-byte row with one bulk copy into a three-stage ring; QK and PV run on
f16 MMAs. 64K: 136 → 229 GB/s (model-free), decode 21.0 → 22.3 tok/s.

**R4, one launch for the q/k/v/gate decode GEMVs (`9d269a3b`).** The
1,024-row k/v matrices took the scalar Q4_K kernel at about 128 GB/s, and the
attention kernel's 99 KiB shared-memory carveout left the following GEMVs a
28 KiB L1. One launch walks the four matrices with each row's arithmetic
unchanged and prefers a 50% carveout: 203.7 → 177.0 µs (233 GB/s), decode 8K
25.0 → 26.4, 64K 22.3 → 22.9 tok/s.

**R5, embedding rows read ahead (`726a144e`).** Cold, the host gather of each
chunk's 2,048 BF16 embedding rows faulted the mapped SSD part one page at a
time (GPU idle ~66 ms per chunk). `MADV_WILLNEED` on the next chunk's rows and
on each gather's own rows: cold 8K prefill 1,133 → 1,207.5 tok/s.

**R7, warp-specialized 12-bit experts (`a5ca7007`).** On real routes
`glmvq_mma8_kernel` ran gate, up and down at 40–43% of their 2.57 ms floor.
`cuda/beam_mma8w.cuh` runs them as tokens × weights^T in persistent CTAs
(eight consumer and four producer warps, a three-stage mbarrier ring, the
next unit's index block streamed into L2). A codeword is exactly the
m16n8k32 B fragment, so one LDS.64 feeds the MMAs; activations and mid rows
are stored in the matching ldmatrix k order. Per 2,048-row chunk: gate 6.01 →
3.96 ms, up 6.37 → 4.40, down 5.96 → 4.41 (65/58/58% of the floor). Cold 8K
prefill 1,209.4 → 1,385.2 tok/s (same binary, `DS4_BEAM_MMA_WS=0`); p512 NLL
−0.28% against the reference (pass). Down equals `glmvq_mma8_kernel` bit for
bit on the same mid rows. (R6, `1a904935`, only casts two enum comparisons.)

**R8, warp-specialized FP8 prefill attention (`391b0bf1`).** After R7 the
sweep's prefill fell 30% from its 6K peak to 64K: the nine full layers'
attention ran 2,048 rows over a 63,488-key prefix at 41% of the F16 peak
(74.7 ms per layer, 16.7 instructions per HMMA: the mask chain and output
rescale on every tile, scalar decodes, a barrier between fill and use). The
Solar K-FP8/V-FP4 walk now serves FP8 rows too: producer warps load the next
tile's bytes into registers (raw tiles would not fit beside the decoded double
buffer) and decode them while the Solar consumers walk the current tile. 74.7
→ 50.0 ms (60% of the F16 peak), byte-identical (`tests/test_solar_fattn`
covers FP8 rows now). Cold 32K prefill 1,287.7 → 1,330.3 tok/s.

## Kernels against their floors

**Decode**, one token at an 8K frontier (nsys at `9d269a3b`; the decode path
is unchanged after it): 37.4 ms per token against about 30.4 ms for its bytes
(7.24 GB of weights and 0.24 GB of KV rows at 246 GB/s).

| Kernel | ms per token | Bytes per call | Effective | Of 246 GB/s |
|---|---:|---:|---:|---:|
| q/k/v/gate Q4_K, one launch (52 layers; also the dense gate/up) | 10.6 | 41.3 MB | 220–233 GB/s | 90–95% |
| Routed gate/up, eight experts (49 layers) | 9.1 | 39.3 MB (12-bit) | 211 GB/s | 86% |
| o_proj, dense down and head Q4_K | 7.0 | 17.7 MB (o_proj) | 221 GB/s (o_proj) | 90% |
| Routed down | 5.1 | 19.7 MB | 190 GB/s | 77% |
| F32 router GEMV | 1.1 | 5.24 MB | 229 GB/s | 93% |
| Full-layer FP8 attention (9 layers) | 0.67 | 17.0 MB | 227 GB/s | 92% |
| Sliding attention (43 layers) | 0.74 | 2.1 MB | ~123 GB/s | ~50% (17 µs calls) |

**Prefill**, per 2,048-row chunk (nsys at `a5ca7007`, real-route replay, and
`tests/test_solar_fattn` for attention):

| Kernel | Time | Floor | Of floor |
|---|---:|---:|---:|
| 12-bit gate / up / down (`beam_w`), in the model | 4.47 / 5.00 / 4.94 ms | 2.57 ms weights | 51–57% |
| the same on one median route set | 3.96 / 4.40 / 4.41 ms | 2.57 ms | 58–65% |
| 13-bit gate / up / down (`glmvq_mma8_kernel`) | 7.3 / 7.7 / 7.1 ms | ~2.8 ms weights | ~38% |
| Q4_K q / gate, o_proj, k / v (q4p) | 1.47, 1.57, 0.32 ms | F16 peak | 81%, 76%, 62% |
| Full-layer FP8 attention over a 63,488-key prefix | 50.0 ms | F16 peak | 60% |

The expert kernels reach 81% of the FP8 peak when every unit carries 128
tokens (4,096 rows, synthetic routes); on Beam's skewed routes many units carry
few tokens and wait on their index stream. The attention walk is bound by its
consumers (tensor pipe 61%; producers wait on the empty barrier).

## Investigated, not retained

- **Fused gate+up** (one unit = 128 rows of both, 64-token items, SwiGLU in
  registers, no BF16 gate rows): 9.6–10.8 ms against 8.6 for the separate
  pair. 240 KB blocks per unit overflowed L2 (1.65 GB DRAM for a 1.3 GB
  minimum) and 64-token items re-read hot experts' blocks.
- **Unit orders.** Global tile-major (mixing hot and cold experts) +20%;
  heavy/light alternation +5%; longest-first ±0.
- **Next-unit L2 prefetch pacing.** Front-loaded (first half or quarter of
  the unit) +5–10%; starting a quarter in ±2%; none +45%.
- **Two or four producer warps** for the fused kernel: 11.0 vs 13.6 ms (four
  cost issue slots); the separate kernels need four (two starved consumers
  27% of the time).
- **Down in halves of four m16 tiles, and up's gate tile prefetched into L2:**
  bit-identical, ±1%.
- Earlier in the round: two m16 row tiles per warp and 1×128 activation
  scales in the warp-specialized prototypes, and a cross-task prefetch in the
  GVQ decode kernels, all within noise.

## Next levers

- **4,096-row prefill chunks** (`DS4_BEAM_PREFILL_CHUNK=4096`): +12.3% at 8K
  and +12.1% at 32K one-shot prefill on the R7 build, since each expert's
  index stream then serves twice the tokens. It needs about 1.4 GiB more and
  does not fit the 1M single-bank plan, so it wants a context-dependent default.
  The 2K–64K sweep (2,048 new tokens per frontier) does not see it.
- **Attention softmax over 32- or 64-key steps.** The FP8 walk is
  consumer-bound (tensor pipe 61%, producers wait on EMPTY); fewer online
  softmax updates change the rounding order, so they need a long-context gate.
  At 100% of the F16 peak the 64K frontier's attention would still cost
  0.27 s, a 15% slope from the peak.
- **13-bit layers** (8 of 49) still run `glmvq_mma8_kernel` at about 40% of
  their floor (7.1–7.7 ms per call; the 64 KiB book leaves room for two
  64-token stages).
- **Small expert units.** A unit streams its whole 120 KB index block through
  a three-stage ring regardless of its tokens; with Beam's skew, units under
  ~46 tokens are bound by that ring's latency.
- **q4p k/v projections** (1,024 rows) run at 62% of the F16 peak from wave
  tails; one launch over q/k/v/gate would remove them.

## Lessons

The general ones are in [AGENTS.md](../AGENTS.md) ("GB10 bottleneck
know-how"): replay real routes in MoE reproducers, attribute NCU stalls to
source lines, keep dependent loads off unit boundaries, ptxas reorders
`mma.sync` (compute VQ as tokens × weights^T), runtime indices into struct
arrays go to the stack, L2 holds CTAs × (current + prefetched) blocks,
mbarrier waits need a suspend hint, raw tiles may not fit beside a decoded
double buffer, and every included header belongs on the object's rule.

## Reproduction

- **Real routes.** A temporary hook in `beam_group` (`ds4_beam_gpu.cuh`)
  appended each MoE call's `rows` and selected experts to a file during one
  8K prefill (196 calls); a model-free harness built the engine's items from
  one call, ran the old and new kernels on random books, indices and
  activations, and compared outputs. Neither is committed.
- **Attention.** `make tests/test_solar_fattn && ./tests/test_solar_fattn`
  checks both warp-specialized walks against the pair kernel and prints their
  times, including Beam's 64K frontier.
- **NCU.** Only model-free processes were profiled (owner and model down).
  Per-line attribution: `ncu --page source --print-source sass --csv`, then
  `nvdisasm -g -c` on the harness cubin.
- **Bench.** `ds4-bench --cuda -m Beam-1.0-501B-A23B-VQ-GGUF --prompt-file
  speed-bench/promessi_sposi.txt --ctx-start N --ctx-max N --gen-tokens 16`,
  fresh processes with the page cache dropped, the kill switches above for
  same-binary A/B. Quality: `tests/beam_quality MODEL goldens.json OUT 512 4096
  ref.json` against the GB10 reference rerun.
