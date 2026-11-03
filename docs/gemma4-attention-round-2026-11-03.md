# Gemma-4 attention and dense-down round — 2026-11-03 (evening)

Round starts from `7fba3bdc` on `feat/gemma4-family` (the handoff commit on
top of the direct VQ prefill round, `9e23197f`). Artifact
`/data/vq-artifacts/gguf-family-it-r8` (R13+8 dense-VQ family build);
`release.json` SHA256
`0bd8a1abd260909735431d3d4da1925648228e07a0d97bd66fd07573448bcb36`.

One RTX 4070 SUPER, 56 SM, `sm_89`, driver 610.43.03, CUDA 13.3.73,
12282 MiB, max SM clock 3105 MHz, kernel 6.12.113+deb13-amd64.
One model process at a time; the model is device-resident
(`DS4_CUDA_COPY_MODEL=1`, 9.37 GiB copied, 0.05 GiB KV at ctx 256).
No MTP, no drafter, no CPU weight offload.

## Adopted rounds

| Round | Env | Change | Measured |
|---|---|---|---:|
| R1 | — | Direct-prefill scatter made deterministic and bit-equal to the reference loop | two runs byte-identical (was relL2 0.085–0.125 apart) |
| R2 | — | `expf` instead of `exp(double)` in the three attention kernels | attention 139.8 → 70.2 ms |
| R3 | `DS4_GEMMA4_ATTN_PREFILL=1` | Prefill attention with a shared K/V tile, one CTA per (row tile, KV head) | prefill attention 33.6 → 9.5 ms, bit-identical |
| R4 | `DS4_GEMMA4_ATTN_FAST=1` | Decode attention, staging vectorized to `uint4` | decode attention 36.7 → 5.8 ms, bit-identical |
| R5 | `DS4_GEMMA4_Q8_BATCH=1` | Dense down through the batch warp8 Q8_0 kernel | dense down 138.2 → 65.1 ms, tokens identical |
| R6 | `DS4_GEMMA4_Q8_BATCH=1` | The same entry on a token-tiled kernel (4 tokens per weight read) | dense down 65.1 → 51.8 ms, tokens identical |

R2 is not optional: the reference computes the softmax in float32
(`modeling_gemma4.py:839`, `F.softmax(..., dtype=torch.float32)`), and the
double exp the oracle used is two FP64 evaluations per key on a pipe that
runs at 1/64 of FP32 on these cards. R1 is a correctness fix, not a speed
change; it was found by the GB10 run of the previous round (see below).

## Round protocol

Fresh processes, same executable, artifact and prompt; three consecutive
off runs followed by three consecutive on runs at ctx 256 with one 256-token
prefill and one generated token. The prefill rate is the bench's own
`prefill_tps`; the ranges do not overlap. The "on" arm is all four options
together (`VQ_PREFILL`, `ATTN_PREFILL`, `ATTN_FAST`, `Q8_BATCH`); the "off"
arm sets none.

```sh
DS4_CUDA_COPY_MODEL=1 ./ds4-bench -m /data/vq-artifacts/gguf-family-it-r8 \
  --cuda --prompt-file /tmp/g4-bench-256.txt --ctx-start 256 --ctx-max 256 \
  --gen-tokens 1
# Repeat with DS4_GEMMA4_VQ_PREFILL=1 DS4_GEMMA4_ATTN_PREFILL=1 \
#   DS4_GEMMA4_ATTN_FAST=1 DS4_GEMMA4_Q8_BATCH=1 in fresh processes.
```

Prompt SHA256:
`ecde80544a343f191c655ab6591d71db415d68aa9fce6945fe9bc1c9dab44d26`.

## Results

| Run | Off (tok/s) | On (tok/s) | Off gen (tok/s) | On gen (tok/s) |
|---|---:|---:|---:|---:|
| 1 | 395.51 | 551.86 | 13.12 | 25.27 |
| 2 | 398.88 | 543.19 | 14.36 | 24.76 |
| 3 | 372.83 | 534.57 | 12.12 | 24.70 |
| Median | 395.51 | 543.19 | 13.12 | 24.76 |

Prefill +37.3% median (+39.6% mean), decode +88.7% median. The off arm is
itself faster than the previous round's 338.14 tok/s median because R2 also
serves the oracle path; the like-for-like number for this round is the
same-binary A/B above.

The same A/B was repeated after R6 (same binary apart from the dense-down
kernel), on a host that was carrying a load average of about 4.4 from other
work, so the spread is wider:

| Run | Off (tok/s) | On (tok/s) | Off gen (tok/s) | On gen (tok/s) |
|---|---:|---:|---:|---:|
| 1 | 355.10 | 633.84 | 10.77 | 28.31 |
| 2 | 384.19 | 573.99 | 13.36 | 23.84 |
| 3 | 362.94 | 613.20 | 13.39 | 19.27 |
| Median | 362.94 | 613.20 | 13.36 | 23.84 |

Median prefill +69.0%, decode +78.4%, but the off median itself moved 8%
between the two sets with no code change on that arm, which is the size of
the external noise here. R6's own contribution is the kernel measurement
(2.17 -> 1.72 ms per layer, 51.8 ms per chunk).

Kernel attribution, nsys, same binary and protocol (total GPU kernel time
and the four kernels this round touches):

| Kernel | Before | After |
|---|---:|---:|
| all kernels, total | 511.5 ms | 288.0 ms |
| `g4_attn_kernel` prefill (grid 1024) | 3.25 ms/call, 97.5 ms | 1.12 ms/call, 33.6 ms |
| `g4_attn_kernel` decode (grid 4) | 1.41 ms/call, 42.3 ms | 1.22 ms/call, 36.6 ms |
| `g4_attn_tile_kernel<16>` (25 sliding layers) | — | 0.28 ms/call, 7.1 ms |
| `g4_attn_tile_kernel<8>` (5 full layers) | — | 0.50–0.70 ms/call, 2.5 ms |
| `g4_attn_fast_kernel` (decode, 25 + 5) | 0.39 / 0.31 ms/call | 0.18 / 0.24 ms/call, 5.8 ms |
| `matmul_q8_0_preq_kernel` (dense down) | 4.61 ms/call, 138.2 ms | — |
| `matmul_q8_0_preq_batch_warp8_kernel<16>` | — | 2.17 ms/call, 65.1 ms |
| `matmul_q8_0_preq_batch_warp8_tok_kernel<4>` | — | 1.72 ms/call, 51.8 ms |

The intermediate steps are measured too: R2 alone took the attention from
139.8 to 70.2 ms and the prefill from 338.14 to 353.27 tok/s (off arm) and
442.21 to 512.62 (VQ-prefill arm); R3 took the prefill attention from 33.6
to 9.5 ms and R4 the decode attention from 36.7 to 5.8 ms; R5 took the
dense down from 138.2 to 65.1 ms; R6 took it to 51.8 ms (TOK 8 measured
1.88 ms per layer, worse, so 4 is the chosen tile).

## Why the direct-prefill scatter had to be fixed first

The GB10 run of the previous round found that the direct VQ prefill was not
run-to-run reproducible: with `DS4_GEMMA4_VQ_PREFILL=1`, two identical runs
differed on the frontier logits by relL2 1.158e-1, 8.46e-2 and 1.25e-1
(three pairs), while the per-expert path was bit-reproducible (off vs off2
0.0, all 262144 logits identical). Cause: `g4_moe_scatter_kernel` does
`dst[target] = bf16(dst[target] + src*w)` and the direct path ran it once
over all pairs, so the eight experts of one output row were read-modify-
written by eight threads in the same launch.

`g4_moe_scatter_ordered_kernel` replaces it for that path: one thread per
output element, walking the token's eight grouped rows in ascending expert
order (the order the reference's per-expert loop adds them) and rounding
after each add. The accumulation is now the reference's own sequence, so
the result is both deterministic and the value the eighth reference launch
would leave. Two fresh full-stack runs are byte-identical on the frontier
logits, tokens and cycles.

## Numerical class

- R3 and R4 are bit-identical to the oracle on the live model: the full
  262144-logit frontier dump matches byte for byte, as do the tokens and
  cycles. Both keep the oracle's per-lane dot order, warp reduction, float32
  online softmax and accumulators; only the K/V load path changes.
- R5 changes the dense-down sum order (16-thread groups of five 32-value
  blocks instead of the preq kernel's 256-thread tree). Eight greedy tokens
  are identical to every previous record: `[6113, 1698, 236761, 28333, 506,
  877, 6910, 3136]`. The frontier logits move relL2 5.10e-2 (max 3.08), the
  argmax is unchanged and the top-5 is the same five tokens in a different
  order.
- R6 keeps R5's per-token sum structure but the compiler schedules the four
  independent accumulators differently, so the two batch kernels are not
  bit-identical to each other: against R5 the frontier logits move relL2
  3.89e-2 (max 2.31) with the same argmax and four of five top-5 tokens,
  and the eight greedy tokens are identical. That is the same class R5
  itself has against the preq kernel.
- R2 moves the oracle's own arithmetic to the reference's precision class
  (float32 exp). The token array is unchanged.

## Limits

- Measured at ctx 256 only, device-resident, one prompt, one device. The
  tile kernel's ring wrap (`k % ring`) and the drafter's shorter `keys`
  count are exercised by the code path but not by this prompt (positions
  0–255 with ring 1289 do not wrap; the trunk passes keys = 0).
- The A/B is three runs per arm; the spread within each arm reached 60
  tok/s in the last set (the host carried a load average of about 4.4 from
  other work during it), so the end-to-end medians are weak. The kernel
  attribution is the per-change evidence, and the two A/B sets agree on
  direction.
- The dense down is still 1.72 ms/layer after the token tile. Its weights
  are read once per four tokens now; the remaining cost is the per-token
  dot work, not traffic (TOK 8 was worse). A shared-memory weight tile or
  the MMQ tile machinery is the way further, not a wider token tile.
- The GB10 measured R5 (the token tile landed after that sync); the GB10
  numbers for R6 are not measured.

## GB10 (same round, same protocol)

The GB10 run (sm_121a, 9.37 GiB copy, `make -j8 cuda-spark`, clean relink)
measured the round with R5, not R6 (the token tile landed after the sync):

- Two full-stack gen-8 runs byte-identical on the frontier logits, tokens
  and cycles, and the same eight-token array.
- A/B, three fresh processes per arm: off 275.11 / 273.13 / 274.44 tok/s
  (median 274.44), on 421.71 / 424.51 / 422.55 (median 422.55). Prefill
  +54.0 percent median.
- Decode: off 13.37 / 13.53 / 13.47 tok/s (median 13.47), on 17.30 / 16.89 /
  16.89 (median 16.89), +25.4 percent. First token 0.074 -> 0.058 s.
- The GB10's decode gain is smaller than the 4070's (+89 percent): the
  attention was a smaller share of its decode step to begin with.

The previous round's GB10 numbers, for continuity: prefill 258.53 ->
351.77 tok/s (+36.1 percent) and the `VQ_VERIFY` gate/up relL2 at or below
the reference's own n/2-vs-n band (2.234e-3 against 2.381e-3), 1990 down
comparisons p50 1.43e-5 / p99 3.54e-3 / max 3.76e-3. That run also
produced the determinism finding above.
