# K2-Horizon hVQ optimization, 2026-11-25

Five adopted rounds after the family landed (`56c66463`): two prefill
kernels, one prefill setting and two decode changes. Host: one GB10,
CUDA 13.5.37 (`sm_121a`), driver 620.70.06, 300–2,200 MHz clock policy
(busy SM clocks 2,184–2,190 MHz in every run below). Speeds are fresh
`ds4-bench` processes under the host memory guard after `clear_cache`;
8K + 64 is the MQ87 protocol (the first 8,192 tokens of
`speed-bench/promessi_sposi.txt`, 64 greedy tokens).

## Starting profile

8K cold prefill (nsys, 2,048-row chunks, 10.06 s of kernels): routed
experts (`mw_kernel`) 42%, Q8_0 MMQ 27.5%, prefill attention 9%, RMSNorm
4.6%, Q8_1 quantization 4.4%, the routed reduce 3.4%, MMQ's output scrub
2%, the F32 router GEMM 1.8%. Decode at 8K (PDL off) 77.9 ms per token:
Q8_0 GEMVs 48.7 ms (63%, ~212 GB/s), expert gate/up 10.6 ms and down 6.1
ms, decode attention 5.3 ms (107 GB/s), RMSNorm 3.0 ms (24 µs a call in
one block), router GEMV 1.7 ms. The bytes a token must read (10.35 GB of
Q8_0, 3.11 GB of experts, 0.58 GB of K/V at 8K) set a decode floor near
58 ms at 240 GB/s.

## Rounds

| Round | Target | Change | Result (fresh pairs) | Numerics | Switch |
|---|---|---|---|---|---|
| P1 | Q8_0 MMQ: each of q/k/v and gate/up quantized its own input (508 quantizations per chunk), every output scrubbed, down outputs rounded in their own pass | One Q8_1 quantization per input; no scrub (finite F16 scales, checked by the host); BF16 rounding in the scrub's place | 8K prefill 946.5–948.3 → 987.8–990.0 tok/s (+4.5%), decode unchanged | byte-identical logits and tokens | — |
| D1 | Decode attention LDS-bound (a lane per key, Q from shared memory); RMSNorm one load round trip per element | Tensor-core split (`k2a_split`): the six query heads of a KV head are the m16 rows of a BF16 MMA over 64-key tiles staged by cp.async; vector partial stores; a parallel merge; all norm loads before the first store | Attention per layer, cold L2: 2K 30.5 → 22.5 µs, 8K 79.7 → 57.1, 32K 288 → 173, 64K 578 → 336 (224 GB/s). 8K decode 14.45–14.76 → 15.31–15.41 tok/s | decode path, 3,072 held-out positions: KL to the reference student 0.01376 both ways, NLL −0.0007 (s.e. 0.0033) | `DS4_K2H_ATTN_MMA=0` |
| P2 | Prefill attention 31% of a 32K prefill at 62 TFLOPS: eight warps decoded, scored and summed in step, idling the tensor pipe | Ping-pong (`k2a_prefill_pp`): two decoded tiles; one head's warps decode the next keys then consume, the other's consume then decode the values; softmax in raw dot units, no mask below the diagonal, rescale skipped while maxima hold | Harness 62.3 → 81.0 TFLOPS at 32K, 63.1 → 82.5 at 64K (tensor pipe 77.5%). 32K prefill 770.4–772.3 → 826.5–827.2 tok/s | 32K frontier KL to P1 0.044, against 0.031 for 4,096-row chunks; held-out prefill (64 × 512): NLL +0.0007 (s.e. 0.0018), KL to the reference +0.0007 (s.e. 0.0006) | `DS4_K2H_ATTN_PP=0` |
| D2 | Q8_0 GEMVs of the 1,024-row k/v and 1,792-row shared gate/up ran at 187 and 204 GB/s alone (launch ramp and tail), the larger matrices at 233–269 | One launch per shared input: q/k/v, shared gate/up, dense gate/up | 8K + 256, three pairs: decode 15.31–15.41 → 15.52–15.86 tok/s (+1.9%) | byte-identical logits and tokens | `DS4_K2H_Q8_GROUP=0` |
| P3 | Expert bytes per token halve with rows per chunk | 4,096-row chunks by default (the engine still halves rows before the context when memory is short) | 8K 1,010–1,013 → 1,064–1,065 tok/s, 32K 827 → 861 | the chunk size is itself the prefill noise reference | `DS4_K2H_PREFILL_CHUNK=2048` |

P3 costs 0.67 GiB of workspace (1,478,571,520 B at 4,096 rows). D1's
merge and RMSNorm parts kept their operation order except the merge's
denominator (a tree sum).

## After the rounds

The [2K–64K sweep](../speed-bench/gb10_spark_k2h_hvq_aee84ba4.meta.txt)
(three fresh runs per arm, interleaved, 2,048 new tokens and 128 greedy
tokens per frontier) records `56c66463` at 619.4 / 12.26 and `c089901f` at
711.9 / 13.96 tok/s: **+14.9% prefill, +13.9% decode**. The incremental
rates gain most at long frontiers, where attention dominates a chunk (64K:
404.6 → 485.5 / 10.02 → 12.29); P3's larger chunks do not reach this
protocol, whose frontiers add one 2,048-row chunk. Run means span
618.9–619.9 / 12.26–12.34 and 711.7–712.1 / 13.94–14.09.

32K cold prefill (nsys, P2 build, 2,048-row chunks, 42.3 s of kernels):
routed experts 32.1%, attention 31.0%, Q8_0 MMQ 21.6%, RMSNorm 4.1%, the
routed reduce 2.9%, Q8_1 quantization 2.1%, the router GEMM 1.6%. A
frontier's rate in the sweep is its last 2,048-token chunk's: at 64K
attention is about two thirds of that chunk. Decode at 32K (PDL off) 73.8
ms per token: Q8_0 GEMVs 45.2 ms (229 GB/s), attention 9.7, expert
gate/up 9.7 and down 5.5, router 1.3, RMSNorm 0.4.

## Measured and not adopted

- **BF16 GEMMs for the Q8_0 projections.** MMQ runs at ~59 TOPS (27% of
  the INT8 peak) and quantizes activations to Q8_1, while the reference
  uses `bf16(d · q)` weights and BF16 activations. Widening per call (Hy4's
  path) adds two bytes written and read per weight and a 200 MB scratch the
  512K plan does not have; on the same six held-out sequences the prefill
  path's KL to the reference (0.0184) is close to the exact-GEMV decode
  path's (0.0138). A producer-dequantizing BF16 GEMM is the lever.
- **F16 attention operands.** F16 K/V decode in two instructions per pair
  instead of six, but Q has no bound that keeps it inside F16's range.
- **A BF16 router copy** would read half the bytes (≈1 ms per token) for
  137 MB of device memory the 512K plan cannot spare.

## Next levers

- Decode is the Q8_0 trunk stream: 10.35 GB per token at 229 GB/s is ~45
  ms of a ~65 ms step. Expert decode runs at 197 GB/s (gate/up) and 170
  GB/s (down, 364-byte rows).
- Prefill: the routed experts (FP8 MMA at ~60 TFLOPS on real routes) and
  the Q8_0 GEMMs above; attention past 64K at 82 TFLOPS.
