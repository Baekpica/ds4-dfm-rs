# dots3-note prefill and decode rounds (2026-09-06)

First optimization campaign on the dots3-note serial lane after the
`dots3note` architecture rename (PR #12).  Same host as the Qwen and K2
campaigns (DGX Spark GB10, driver 610.43.02, CUDA 13.3), artifact
`dots3-note-prev-Mixed-Quant-GGUF` (MQ87, 10 shards; shard 1 rewritten with
`general.architecture=dots3note`, sha256 `c2b8cbf9…`), the VMM weight owner
(`ds4_weight_server --backend vmm --reserve-gb 24 --repack-q8-aligned`, 620
ranges, 571 derived artifacts) resident throughout, one fresh `ds4-bench`
process per run with the serial worker environment
(`DS4_SERVER_CONTINUOUS=0`).  Corpus: the official `modeling_dots3_note.py`
source, one cold 8,192-token prefill (`--ctx-start 8192 --ctx-max 8192
--ctx-alloc 8257`, two 4,096-row chunks) — three prefill-only runs plus one
run with 64 greedy tokens per variant, medians.  Every round is measured on
one binary through its kill switch (K2 protocol), so the cells share the
hour and the owner state.  Base is `main` at `1f7933f`.  Scratch:
`scratch/dots3-opt-20260906/` (`scripts/`, `logs/<cell>/`, `nsys/`).

## What the profile said

Baseline (`b3da596`): 276.4 tok/s prefill, 11.69 tok/s decode.  nsys on the
8K + 64 run (`nsys/baseline-8k-gen64.sqlite`, `scripts/kern.py`):

| prefill (31.2 s busy, 95 % of wall) | share |
|---|---:|
| `dots3_latent_attention_kernel` (warp per (token, head), FP32 latent walk) | 51.3 % |
| `motif3_value_project_q8_0_transposed_kernel` | 11.5 % |
| `dots3_qk_absorb_q8_0_kernel` | 10.3 % |
| routed IQ2_XXS gate/up + Q2_K down (D2R) | 12.3 % |
| `dots3_idx_score_kernel` | 3.9 % |
| dense `mul_mat_q` + quantize + D2R | 6.4 % |

| decode (5.3 s busy over 64 tokens) | share |
|---|---:|
| `dots3_latent_attention_kernel` (16 / 8 blocks per token, 2048 / 513 serial key steps) | 37.2 % |
| aligned Q8_0 dense vec (198 launches per token, ~5.0 GB) | 24.5 % |
| routed experts (aligned IQ2 gate/up/mid + Q2_K vec) | 12.4 % |
| absorb + value projection (raw Q8_0 walks) | 9.2 % |

Per token the always-active weights are 9.98 GB (`attn_output` 2.6 GB,
`attn_kv_b` 0.97, `attn_q_b` 0.94, routed experts 2.4, LM head 0.83), i.e.
41.6 ms at 240 GB/s; the dense and expert tiers already run at that rate,
so the decode work was the attention walk and the two raw-row kernels.  The
prefill work was the three FP32 kernels of the absorbed MLA path, none of
which used the tensor cores.

## Round 1: latent attention on tensor cores

`dots3_fattn_hmma_kernel` (`cuda/mmq/ds4_fattn.cu`,
`ds4_mmq_dots3_prefill_attn_hmma`).  In the absorbed MLA form every head of
a token shares the token's key rows, so one token's 128 (full) or 64 (SWA)
heads form a proper GEMM against that token's key set: S = Q_abs ·
[latent | k_pe]^T over 576 / 1088 dims, O = P · latent over 512 / 1024.
The block owns one token (its DSA top-2048 list is gathered row by row;
SWA layers walk the 513-key ring) and 32 (full, 32-key tiles) or 16 (SWA,
16-key tiles) heads under the 100 KiB shared-memory budget; QK is split
across warps by (m-tile, n-tile, k-slice) with the slices summed through
shared memory, PV across the latent columns; fragments come from
`ldmatrix`, and the next key tile is fetched into registers while the
current one is consumed.  Operands are FP16: the BF16 cache rows convert
exactly, and FP16 keeps three more mantissa bits than BF16 for the rounded
Q and P (the BF16 version failed the resident gate's one-shot/split
consistency at 0.9984 < 0.999; FP16 passes).  Decode widths (rows < 8)
keep the scalar kernel; `DS4_DOTS3_ATTN_NO_HMMA=1` restores it everywhere.

Model-free probe (`tests/test_dots3_cuda`, `DS4_DOTS3_PROFILE_ATTN=1`,
4,096-row chunk at position 4,096): full 402 → 75.7 ms, SWA 86.0 → 31.7 ms;
parity against the scalar kernel rel RMS 2.5e-4.  Rejected on the way:
staging the shared KV row in shared memory inside the scalar kernel
(−13 %: two barriers per key beat the 8× load saving) and a two-chain QK
accumulation in the HMMA kernel (72 → 81 ms; the kernel is bound by
shared-memory traffic, not MMA latency).

## Round 2: value projection on tensor cores

`dots3_value_project_hmma_kernel`.  The owner's transposed Q8_0 artifact
already stores, per head and 32-wide latent block, the 128 value columns
contiguously — a k-major layout that feeds the MMA B operand through
`ldmatrix.trans`.  Block = (head, 64 tokens), the int8 codes are exact in
FP16 so each block's product is accumulated separately and folded in with
its FP32 per-column scale; only the activations are rounded.  Probe: 12.9 ms
(full) / 11.6 ms (SWA) per 4,096-row launch against ~37 ms; rel RMS 2.1e-4
against an FP32 host reference.  `DS4_DOTS3_VALUE_NO_HMMA=1`.

## Round 3: Q/K absorption on tensor cores

`dots3_absorb_hmma_kernel` over the raw Q8_0 `attn_kv_b` rows: the A
operand (16 tokens × nope) lives in registers, the kernel walks the latent
columns one 32-wide Q8_0 block at a time and dequantizes that [nope × 32]
slab to FP16 in a double-buffered shared tile (nine aligned words per
34-byte block, the last two bytes fetched as a half word so no read crosses
the tensor's end).  Probe: 10.4 / 10.9 ms per launch against ~36 ms; rel
RMS 2.9e-4.  `DS4_DOTS3_ABSORB_NO_HMMA=1`.

## Decode 1: split-K latent attention

`dots3_latent_attention_split_kernel` + combine: grid.z splits the key list
into 16 ranges, each warp walks one range with the scalar kernel's
arithmetic and publishes (unnormalized O, running max, running sum) to the
graph's `attn_partial` scratch, the combine kernel merges.  Probe: full
0.87 → 0.13 ms, SWA 0.22 → 0.04 ms per layer; rel RMS ~1e-6 (fp32 reorder).
Rows ≤ 2 only; `DS4_DOTS3_ATTN_NO_SPLIT=1`.

## Same-binary chain (prefill R1–R3, decode D1)

`bins/ds4-bench.r3d1`, 17:09–17:23 KST, 8K prefill median of three / decode
tok/s from the 64-token run:

| cell | prefill tok/s | Δ | decode tok/s | TTFM s |
|---|---:|---:|---:|---:|
| all off (baseline path) | 278.34 | — | 11.66 | 0.1403 |
| R1 attention HMMA | 459.43 | +65.1 % | 11.66 | 0.1393 |
| R1 + R2 value HMMA | 522.51 | +13.7 % | 11.65 | 0.1394 |
| R1–R3 (+ absorb HMMA) | 595.29 | +13.9 % | 11.65 | 0.1386 |
| R1–R3 + D1 split attention | 594.40 | — | 16.59 | 0.1139 |

Cumulative: prefill 278.3 → 595.3 tok/s (+113.9 %), decode 11.66 → 16.59
tok/s (+42.3 %).

## Decode 2: value projection at decode widths

The transposed value projection ran one 128-thread block per (head, token)
whose threads each issued 512 dependent load batches (~115 GB/s on the
8.4 MB per layer).  `dots3_value_project_q8_0_decode_kernel`: four thread
groups of 128 columns split the latent blocks (group g takes b = g, g + 4,
…) and sum through shared memory, four times the warps per layer; per-block
dots keep the transposed kernel's order, only the block sum is
re-associated (fixture rel RMS 1.3e-7).  `DS4_DOTS3_VALUE_NO_DECODE=1`.
Same-binary decode A/B: 16.58 → 16.72 tok/s (+0.8 %).

The absorption keeps its wide kernel.  Three rewrites of its raw-row walk
were measured and rejected: one lane per column (one byte per lane per d,
16.58 → 15.95 tok/s: twice the sector requests), one lane per 32-column
block fetching nine aligned words (16.60 → 16.29: 16 active lanes issuing
4-byte loads at a 34-byte stride), and 16-byte lanes over the contiguous
row with the block scales shuffled from their owning lanes (16.78 → 15.40:
32 shuffles per d).  All three moved fewer bytes than the wide kernel; its
two-byte loads simply issue fewer L1 wavefronts per row than any narrower
lane mapping, and at 8.9–13 MB per layer the kernel is wavefront-, not
DRAM-bound.  The dispatch comment records this.

## Decode 3: launch fusion on the attention side

A dots3 layer issued ~45 launches per decode token; the idle time between
them (host launch issue, ~3 µs each) and the F32 round trips between
elementwise stages were the remaining non-bandwidth cost.  Five fusions,
each byte-identical to the chain it replaces (same arithmetic, same order,
the 256-lane reduction trees included): kv finish (latent RMSNorm + rope-tail
RMSNorm + interleaved rope + BF16 cache store, was four launches with two
F32 round trips); the headwise sigmoid gate applied in the value projection
epilogue (HMMA and decode kernels; the gate logits are computed first);
indexer key finish (LayerNorm + rope + BF16 boundary + FP8 round trip +
cache scatter, was five) and query finish (rope + FP8, was two) with the
head-weight scale folded into the score kernel; FFN block output
`x += routed + shared` in one pass (was an add and a residual add).
`DS4_DOTS3_NO_FUSED=1` restores the separate launches.  Same-binary A/B
(decode 2 kernels on in both cells, the rejected block-per-lane absorb
included): prefill 595.4 → 602.7 tok/s (+1.2 %), decode 16.27 → 16.35 tok/s
(+0.5 %).

## Scoreboard

Final path = rounds 1–3, decode 1, the grouped value projection and the
fusions (`bins/ds4-bench.d4` cell `c-d2v4-off`, 18:00 KST, medians of three
8K + 64-token runs):

| | baseline path | final | Δ |
|---|---:|---:|---:|
| 8,192-token cold prefill | 278.3 tok/s | 604.3 tok/s | +117.1 % |
| greedy decode after it | 11.66 tok/s | 16.78 tok/s | +43.9 % |
| time to first token | 0.140 s | 0.113 s | −19.3 % |

The decode step is now ~60 ms against a ~42 ms floor set by the 9.98 GB
of always-active weights per token; the remaining gap is the ~2,000
launches per token (host issue), the idx top-k pipeline (2.4 ms), and
the raw-row absorption (4 ms).  A dots3 MTP draft (the block is bound,
not executed) is the only lever left with a large upside.

## Gates

- Fixture `tests/test_dots3_cuda` (model-free): attention HMMA vs scalar
  (five cases, rel RMS ≤ 2.8e-4), decode split vs serial (five cases,
  ≤ 1e-6), value / absorb HMMA vs FP32 host references (≤ 3e-4), grouped
  decode value vs the one-group kernel (1.3e-7) and vs the reference, eight
  fused launches byte-identical to their chains.
- Frontier logits at 8,192 tokens on the final binary (`bins/ds4-bench.final`),
  every switch off vs cumulative rounds (`logs/rd-*/`, `logs/final-dump-*/`):

  | rounds on | argmax | top-10 | KL | rel RMS | greedy IDs equal |
  |---|---:|---:|---:|---:|---:|
  | R1 | 284 / 284 | 10/10 | 8.3e-4 | 5.9e-2 | 31 / 64 |
  | R1–R2 | 284 / 284 | 10/10 | 8.4e-4 | 5.7e-2 | 31 / 64 |
  | R1–R3 | 284 / 284 | 9/10 | 2.2e-3 | 5.9e-2 | 31 / 64 |
  | R1–R3 + D1 | same logits as R1–R3 (decode-only change) | | | | 64 / 64 |
  | R1–R3 + D1–D2 | same | | | | 31 / 64 |
  | all (+ D3, bit-identical) | same | | | | 31 / 64 |

  The 8K frontier stays inside the arithmetic-round band of the K2 gate
  (same argmax, top-10 ≥ 8, KL ≤ 0.05, rel RMS ≤ 0.11) for every cumulative
  step.  Token 31 of the 64-token greedy continuation is a near tie: the
  FP16 attention flips it, the split attention's fp32 reorder flips it
  back, the grouped value projection flips it again, and the continuation
  after it differs in each case (the BF16 variant of round 1 happened to
  keep all 64).  This is the near-tie behaviour the playbook documents, not
  a drift: the tokens before it and the frontier statistics are stable
  across all six rounds.
- `tests/test_dots3_resident` on the final binary (VMM owner; CPU FP32
  reference on the 21-token chat prompt, chunk/ring parity on 1,600 tokens,
  DSA 2,600-token determinism, 262,144-context graph): **passed** —
  cos(gpu, ref) 0.997816 (baseline 0.997536), cos(one-shot, split) 0.999253
  (gate 0.999; the BF16 attention had 0.998409), argmax 3925 on all three,
  chunk/ring first token 63594/63594 with batch_cos 0.99910 and cache_cos
  1.0, DSA argmax 151721/151721, 256K graph + cache 2.98 GiB, cleanup
  remainder 0 bytes.  `logs/final-resident/resident.log`.
