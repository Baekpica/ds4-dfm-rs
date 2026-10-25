# MiniMax-M3.1-Flash VQ optimization, 2026-10-25

One GB10 (`sm_121a`, CUDA 13.4, driver KMD 615.71.10, SM clock capped at
300-2,200 MHz by the host, 2,171-2,197 MHz under load). Card protocol:
`ds4-bench --ctx-alloc 524288 --mtp-draft 3 --step-incr 2048 --gen-tokens 128
--prompt-file speed-bench/promessi_sposi.txt`, frontiers 2K-32K and 64K, each a
2,048-row segment on the restored prefix. Targets: prefill >= 1,000 and decode
>= 22 tok/s at every frontier; decode first (user decision).

## Commits

| Commit | Change | Effect |
|---|---|---|
| `843a1bf3` | SSD-tier prefill attention reads a device stage of the layer's blocks | 32K +2.3% |
| `67712efa` | Next attention block loaded during the MMAs | +0.6% |
| `c17a5d5c` | SSD-tier mark event only for streamed resolves | noise |
| `7d1d6591` | Decode q/k/v + indexer, and shared gate/up + router logits, one launch each | -1.6 ms per token |
| `43ea277f` | Prefill up kernel runs as GATE + a SwiGLU pass | +0.8% prefill |
| `05af5fcf` | Three-slot R13 expert ring | ~+1% |
| `aff45fe4`, `dbdf82da`, `11317f6f` | BF16 heads, q and branch rows in prefill | bit-identical byte cuts |
| `088bedc0` | DSpark verify rows share one expert launch per stage | sweep mean decode 20.32 -> 21.24, 64K 18.39 -> 19.18 |
| `1b3b4f38` | R13 index copy pieces rotated across lanes (32-way L1 bank conflict) | R13 gate/up launch 4.54 -> 4.40 ms in-engine; 8K prefill 1,031 -> 1,040 |
| `41bb8b89` | Decode score keys of earlier forwards load before the PDL wait; the unread own block is skipped | score 0.610 -> 0.453 ms per token |
| `39517dc2` | Router top-4 runs in the last router CTA of the shared gate/up launch, logit CTAs first | router 0.261 -> 0, shared + router 5.944 -> 5.448 ms per token; 8K token 47.68 -> 46.73 ms |

Card sweep, decode / prefill tok/s, at the round's start (`11317f6f`),
after the verify change (`088bedc0`) and at the end (`39517dc2`):

| Frontier | `11317f6f` | `088bedc0` | `39517dc2` |
|---:|---|---|---|
| 2K | 22.55 / 912 | 22.03 / 852 | 24.07 / 820 |
| 4K | 22.48 / 1,025 | 22.08 / 1,033 | 24.87 / 1,036 |
| 6K | 24.85 / 1,025 | 27.31 / 1,022 | 27.80 / 1,032 |
| 8K | 18.18 / 1,015 | 18.92 / 1,019 | 19.55 / 1,024 |
| 10K | 23.43 / 1,020 | 29.54 / 1,013 | 30.05 / 1,024 |
| 12K | 17.99 / 1,010 | 18.78 / 1,007 | 19.03 / 1,010 |
| 14K | 21.10 / 1,001 | 21.56 / 999 | 21.90 / 1,007 |
| 16K | 20.98 / 994 | 21.38 / 1,002 | 21.75 / 996 |
| 18K | 20.92 / 994 | 19.89 / 997 | 19.87 / 1,007 |
| 20K | 18.61 / 993 | 19.79 / 1,000 | 20.12 / 998 |
| 22K | 18.38 / 989 | 18.58 / 987 | 18.90 / 994 |
| 24K | 19.18 / 991 | 19.79 / 993 | 20.09 / 1,000 |
| 26K | 20.80 / 983 | 21.95 / 988 | 21.95 / 990 |
| 28K | 19.02 / 982 | 20.39 / 982 | 20.73 / 986 |
| 30K | 17.63 / 982 | 18.04 / 982 | 18.54 / 981 |
| 32K | 19.03 / 970 | 19.73 / 977 | 20.00 / 976 |
| mean 2K-32K decode | 20.32 | 21.24 | 21.83 |
| 64K | 18.39 / 934 | 19.06 / 937 (2 runs) | 19.46 / 941 (2 runs) |

Frontier logits and the 128 generated tokens are identical across the three
builds (18K's tokens diverged at token 55 once in `088bedc0` runs: a changed
gate schedule computed a position at another verify width). The 2K
frontier's prefill varies with the first segment's lazy graph allocation.
One 64K run of `39517dc2` (692 / 14.3, its 63K cold prefill and first token
slow as well) did not reproduce in two alternating runs against
`088bedc0`.

## Where the time goes (per-kernel analysis, HEAD `11317f6f`)

Prefill, one 2,048-row segment at 64K (2,160.8 ms):

| Kernel | Time | Against its floor | Cause |
|---|---|---|---|
| `mw_kernel` (routed experts) | 969 ms (45%) | 2.22x (437 ms; DRAM 103-123 GB/s, tensor pipe ~34%) | consumers load indices and 16-32 book gathers before each stage's first MMA (42% of their samples); R13 index copies conflict 32-way in L1 (rows 1,152 B apart); 29% of items carry <= 16 pairs |
| `glmvq_q4p` | 516 ms (24%) | 84% of the FP16 peak in the k-loop (cuBLAS's best on this host) | compute-bound; k and v launches run 1.33 waves |
| Row kernels (residual, reduce, tail, quantizers, router) | 281 ms (13%) | 94-122% of DRAM floor, except the tail (83%) and the cuBLAS F32 router (39%) | F32 buffers holding BF16 values, separate passes re-reading them, synchronous host grouping (3.8 ms) |
| `mm_attn_mma_kernel` | 219 ms (10%; 154 at 8K) | 35% of the FLOP floor (50% at 8K) | 8-row CTAs walk the union of their rows' blocks in lockstep (~39 blocks at 64K, 3.3 of 8 warps busy per block, decode and two barriers per block) |
| MSA indexer (scorer, top-k, projections) | 144 ms (57 at 8K) | scorer 72.5% of the BF16 tensor pipe | BF16 operand rate (keys are NVFP4 already); grows linearly with context |

Decode, 8K token (plain 47.6 ms; byte floor ~42 ms):

| Kernel | Time | Cause |
|---|---|---|
| Q4_K/Q8_0 GEMVs (241 launches) | 28.7 ms, 92% of the byte floor | ~10 us lost per dependent boundary (serial staging copies, slow CTAs at the end of joined grids); at 4 rows tokens 2-3 load in serialized branches (+13.2 ms per verify) |
| Routed experts | 15.1 ms, 88% | one block in flight per warp (down has no prefetch) |
| Short kernels + idle | 3.4 + 0.9 ms | latency chain proj -> tail -> score -> topk -> readback -> attention; per-layer SSD-tier readback after a restore (0.73 ms) |

DSpark: a trial (draft 14.1 ms + 4-row verify) costs ~109 ms after `088bedc0`
against a 49.5 ms gated plain step, so it pays only above ~2.2 kept tokens.
8K, 12K and 20K-30K keep 1.7-2.0, the gate decodes mostly plain there, and
failed probes cost 7-10% of those frontiers. 22 tok/s there needs plain
decode <= 45.45 ms, near-zero probe waste and trials <= 78-91 ms.

## Measured and rejected

- SSD-tier pinned pool for every 64K block (9,660 -> 32,700 slots, +1.6
  GiB): 64K plain decode 18.6 -> 20.2-20.4, but the 512K 64K run was killed
  by systemd-oomd (user slice memory pressure 58% > 50% for 20 s at ~3.6 GiB
  available) and the first segment paid +1.4 s of pinning (2K prefill 912 ->
  563). Reverted.
- Expert prefill, model-free harness on real 8K routes: a k-outer consumer
  loop with one activation scale per row (no per-stage FFMA or scale copies)
  ran even; loading the next stage's codewords between a stage's MMA halves
  (two B buffers) ran 11% slower on R13 (consumers hold two of three slots,
  so producers fall behind) and spilled on R13+8. The analysis model had
  projected 25-35% for this loop.
- Decode: L2 prefetch with `evict_last` before o_proj, plain
  `prefetch.global.L2`, persistent GEMVs with a grid barrier, 16-warp or
  next-row-carry expert decode, DSpark gate variants (2-row probes, pause
  backoff): slower or even.

## Output gate

Frontier logits are chaotic: the accepted `DS4_MINIMAX_ATTN_SIMT=1` switch
(F32 attention) moves them by rel-RMS 0.15-0.76 and KL up to 14.5 across the
sweep, with greedy tokens diverging within 0-3 tokens. Value changes are
judged with `tests/minimax_quality MODEL ref2.json OUT 512 16384` against the
handoff reference rerun on this host (assistant positions: paired NLL within
2%, top-1 >= 0.95). Baseline at `088bedc0`: top-1 0.9625, NLL 1.4220 against
1.4285.

## Open levers (estimates from the per-kernel analysis, not measured)

| Lever | Class | Estimate |
|---|---|---|
| Plain decode: staging copies in flight (cp.async); slow CTAs first in the q/k/v + indexer grid; expert down prefetch + reduce fold; route fold into the expert prologue; skip the SSD-tier readback while the pool holds every block | bit-identical | -1.5..-3.5 ms per token |
| Drafter context deferred during gated plain steps | drafter only | -1.1 ms per plain step |
| Confidence-gated verify (verify only confident proposals) | gate | failed probe ~76 -> 7-14 ms |
| Verify widths: exact token-count GEMV templates, BF16 inputs | bit-identical | -5..-9 ms per 4-row verify |
| Prefill row kernels: `cur` in BF16, residual writing the indexer's Q8_1 and the experts' E4M3 rows, reduce fused into the residual | bit-identical | ~-119 ms per segment |
| q4p: one launch for q/k/v, gate+up with SwiGLU epilogue, attention writing o_proj's FP16 input | bit-identical | ~-45 ms per segment |
| Attention: row regrouping by shared blocks, warp ping-pong, softmax trims, conflict-free decode stores | bit-identical | -44..-56 ms at 64K |
| NVFP4 x NVFP4 block scorer (`mxf4nvf4`) | value change | -77..-82 ms at 64K |
| Shared expert re-quantized to GVQ R13 | value change (artifact) | -4.4 ms per plain token |
