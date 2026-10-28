# Ling-3.1-flash-1019 VQ on one DGX Spark: optimization round, October 28

The all-12-bit [1019 release](https://huggingface.co/Baekpica/Ling-3.1-flash-1019-VQ-GGUF)
on the branch `feat/add-ling31-1019`, base `73f3aa0e`. Each adopted change is
its own commit with a kill switch; this page keeps the measured ladder, the
bottleneck each change answered, and what was rejected.
[Ling-3.1-flash VQ](ling31-flash-vq.md) describes the family.

## Protocol

- One model process at a time under `tools/host_memory_guard.py` (2 GiB
  floor, PSI 40%), page cache dropped before each run, `ds4-bench` on
  `speed-bench/promessi_sposi.txt`.
- Prefill: 8K prompt after one 8K warm-up prefill, 32 decode tokens,
  alternating fresh-process pairs (A = kill switch or previous build,
  B = change). Numerics: 2K–32K frontier logits (2,048-token steps) against
  the previous path; the reference is the chunk-size noise of the same build
  (`DS4_LING31_PREFILL_CHUNK=1024` against 4,096: mean KL 0.0085–0.0092).
- Decode: 8K context, 128 greedy tokens, at least three pairs. Plain decode
  is bimodal per process (about ±8% from physical placement), so a single
  pair proves nothing. Value changes on the decode path are judged by
  `tests/ling31_quality` over six held-out cases (9,113 teacher-forced decode
  steps) against `DS4_LING31_GAIN_DOT=0`, an accepted arithmetic change of
  the same path (mean |ΔNLL| 0.033, top-1 98.2%).
- Kernels: Nsight Systems inside the `ds4.prefill` / `ds4.decode` ranges
  (`DS4_GLMVQ_PDL=0` for decode attribution) and model-free harnesses at the
  production shapes.

## Ladder

| Commit | Change | Numerical class | 8K prefill tok/s | Decode tok/s |
|---|---|---|---:|---:|
| `29abaabb` | 4,096-row default chunk; records-only KDA scratch | chunk noise | 1039 | 17.4–18.0 |
| `2f7cd355` | Beam's warp-specialized experts on 12-bit layers | KL 0.0103 vs noise 0.0101 | 1039 → 1123 | — |
| `faa7f3fb` | KDA decay joins the q/k/v/gate q4p group | no value change | 1123 → 1130 | — |
| `42daaef6` | residual RMS: weights before the PDL wait, warp-0 tail | no value change | — | 10.3 → 4.5 µs per call, ~1.4% |
| `99e7707b` | router GEMM on TF32 tensor cores | KL 0.0082 vs 0.0085 | 1128 → 1147 | — |
| `786182af`, `45940fbe` | one grouped launch for KDA's five decode projections | F32 order (split 2), then each matrix's split | — | 18.31 → 18.91 |
| `806abba3` | FFN norm as BF16 rows; router on BF16 tensor cores | KL 0.0078 vs 0.0092 | 1147 → 1171 | — |
| `25544c26` | attention norm as BF16 rows | no value change | 1159 → 1172 | — |
| `7a0231b0` | KDA state in registers across MTP verify rows | no value change | — | MTP2 24.40 → 25.51 |

Each pair of numbers is the A/B median of its own runs; absolute levels
drift a few tok/s between sessions. After the round, 8K prefill runs at
1,170–1,188 tok/s, plain decode at 18.6–19.1 and two-draft MTP decode at
25.4–25.6.

## What each change answered

- **Experts (P1).** At 4,096 rows each of 512 experts sees about 64 rows,
  so every chunk streams all expert weights (1.5 GB per layer). The FP8 MMA
  kernel rebuilt codewords per 64-token item; Beam's warp-specialized gate,
  up and down kernels take 128-token items with producer warps. The routed
  experts are now weight-stream bound (about 80% of DRAM bandwidth per
  layer-chunk); larger chunks, not kernels, would lower their per-token cost.
- **Router (P3, P4).** The F32 SGEMM (PEDANTIC) took 1.46 ms per chunk.
  Its operands are BF16 values (norm rows rounded at the boundary, BF16
  weights), which TF32 holds exactly: products stay exact, only the
  accumulation order moves (harness relRMS against FP64: F32 6.9e-7, TF32
  8.1e-6, BF16 4.6e-6). With BF16 norm rows the router reads them on BF16
  tensor cores without the widened weight copy (0.36 ms).
- **BF16 norm rows (P4, P6).** Both norms per layer were BF16 values stored
  as F32 and read by up to three kernels. BF16 rows halve the residual's
  write and every read, and remove the F32 → BF16 input pass of the BF16
  beta and head-gate GEMMs (0.40 ms per call, 57 ms per 8K prefill). A
  deterministic predicate tells the writer (the previous residual) which
  rows the next reader takes; decode and verify keep F32.
- **Grouped decode projections (D2).** One token's KDA projections are five
  18.9 MB Q4_K matrices sharing one input row. Five launches streamed at
  ~229 GB/s; Beam's grouped stream reached 245–254 GB/s in a harness
  (410 → 380 µs per layer). Beam's split 2 changes the F32 order; val6 stayed
  within the accepted reference (|ΔNLL| 0.035 against 0.033), but the MTP
  gate's fixed 0.5 tie flagged one token at 0.625 on the final build.
  `45940fbe` groups at the split each matrix picks alone (split 1 for 8192
  rows), which runs at the same speed; the gate passes and decode gains 3.3%
  (18.31 → 18.91 tok/s, three pairs).
- **Verify KDA (D3).** At three rows the recurrence read and wrote the
  state twice per row through L2 (127 µs per layer). Thread d now holds
  column d in registers (255 registers, 160 bytes of spill): 97 µs, same
  operations and order. One decode row keeps the memory loops: there the
  register kernel measured 1.5% slower end to end.

## Rejected, with the evidence

- **Grouped q4m launch at verify widths.** One launch over the five KDA
  matrices was 7% faster in a harness (three tokens:
  408–424 against 441–454 µs) but moved MTP2 decode 24.41 → 24.42: under
  PDL the per-matrix launches already overlap.
- **Decode experts.** Next-task register prefetch (no gain), L2 prefetch of
  rows 1–4 tasks ahead (122 → 129–138 µs per layer), two down rows in flight
  per warp (−4% on down only). A read-only probe of the same access pattern
  sets the ceiling: gate/up 240 GB/s (65 µs), down 191 GB/s (41 µs), against
  72 and 46 µs measured; 24 or 16 warps lose to 32.
- **Shared expert at one token.** Grouped gate/up plus a SwiGLU kernel lost
  to the fused q4m SwiGLU (33.5 against 30–32 µs); STAGE for the 4096 × 1280
  down was within noise of the scalar kernel.
- **Prefill attention.** A 64-key softmax variant (62 against 65 TFLOPS) and
  issuing the next QK step early: no gain.

## Profile after the round

8K prefill (6.85 s of kernels): q4p 42.9%, warp-specialized experts 22.9%,
KDA chunk kernels 13.9% (local 4.9 ms and scan 2.6 ms per 4,096-row call,
against DRAM floors near 2.6 and 2.1 ms), residual RMS 4.3%, routed reduce
3.5%, prefill attention 3.0%, q4p preparation 2.6%. Plain decode (PDL off,
55 ms per token): grouped and single Q4_K streams 63%, experts 17%
(near their read-only ceilings), shared expert 6.8%, KDA recurrence 4.1%.

## Lessons

- TF32 is exact for BF16 values. An F32 GEMM whose operands were rounded at
  a BF16 boundary can run on TF32 tensor cores without truncating anything.
- Check grouped launches in the model, not only in a harness: PDL hides most
  launch boundaries of back-to-back kernels that already fill the GPU.
- A fixed logit tie in a gate is not a noise floor: an F32-order change that
  stays within the accepted output reference can still cross it.
- Measure a decode kernel's read-only ceiling for its own access pattern
  before tuning it; small rows (240 bytes per warp) cap the stream well below
  the DRAM peak even without compute.
