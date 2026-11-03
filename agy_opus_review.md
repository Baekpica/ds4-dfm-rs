# Gemma-4 26B A4B VQ prefill review (RTX 4070 SUPER, sm_89)

Reviewed on 2026-11-03 at commit `c09cfbf9` (`feat/gemma4-family`). This is a
code reading plus arithmetic only: nothing was run on the GPU, so every figure
below is an estimate to be confirmed by measurement.

**Short version:** the measurements point somewhere different from the
framing. The restore path is not limited by decode. Its limit is writing and
re-reading bf16 weights in DRAM. The direct kernel is limited by the latency of
byte-wide codebook reads that miss L1. So the direct kernel is worth rescuing,
and a working tensor-core template already exists in this repo. But the two
cheapest wins are elsewhere: the Q8_0 down and attention.

## The numbers that decide it

Per layer at 256 tokens, all 128 experts get hit (the chance an expert is
unused is about e^-16). That is 128 x (2x704x2816 + 2816x704) ~ **761 M
weights**.

- **Restore + cuBLAS:** 761 M x 2 B ~ 1.52 GB of bf16 written per layer, then
  read back. That alone is about 3 GB per layer of DRAM traffic, a floor of
  roughly **6-7 ms/layer, ~190 ms per chunk**. It matches the profile: restore
  takes 207 ms (30%), and the unattributed ~18% (~124 ms) is very likely the
  cuBLAS expert GEMMs.
  - By this count restore runs at about 110 G weights/s, not 66. If the 66
    counts something else, re-check the comparison.
- **Direct kernel:** the R13+8 bytes are about 0.25 GB per layer, a DRAM floor
  near 0.5 ms. At roughly 3-4 instructions per weight, decode alone would allow
  several T weights/s. It reaches about 74 G/s (10.3 ms/layer). DRAM sits at
  3.8% and L2 at 60%: the time goes to waiting on L2, not to decoding.

## 1. Rescue the direct kernel? Yes

Its ceiling is about 10x above the restore path's. Two causes are visible in
[gemma_vq_prefill.cuh](cuda/gemma_vq_prefill.cuh):

- **Byte-wide codeword loads.** [`g4_vqp_pair`](cuda/gemma_vq_prefill.cuh#L74-L87)
  reads each codeword as `c1[0]`, `c1[1]`, ... That is 8 single-byte loads for
  the main codeword and 8 for the residual, per 8 weights. They are serialised
  behind the index read.
- **The codebook misses L1.** Ada has 128 KB of L1 plus shared memory per SM.
  Two blocks x 39 KB of shared leaves at most about 50 KB of L1, less than the
  64 KB codebook, so the lookups fall through to L2. That fits the 61% L1TEX
  stall and 60% L2 load.

The tensor-core design already exists in
[ds41_vq_prefill_mma.cuh](cuda/ds41_vq_prefill_mma.cuh):

- `vqm`: 512 threads, codebook in shared, 128 rows x up to 128 tokens, 64 K
  columns per round, bf16 `m16n8k16` MMA.
- `vqs`: decodes straight into MMA fragments with `cp.async`.
- It already compiles for sm_89.

Make a Gemma copy with the exact Gemma decode, `bf16(gain*(main+res))`. The
products stay exact in f32; only the K accumulation order changes, so it is
gated by the chunk-width noise floor. MMA output is deterministic.

Expected: 310 ms -> roughly 60-100 ms. That is the largest single saving
(~200+ ms), but also the most work.

## 2. Q8_0 down or attention? The Q8_0 down first, then attention, both before the MMA port

**A. Send the Q8_0 dense down to MMQ (lowest cost, ~-135 ms).**

- At 0.63 TFLOP/s this matmul (3.05 GFLOP/layer) costs 4.8 ms per layer.
- [`ds4_mmq_q8_0_dense`](ds4_cuda.cu#L25608) (the vendored llama.cpp MMQ) and
  the `_d2r_preq` variant that takes pre-quantized input already exist and are
  used by other families. Wire a Gemma branch to them.
- Determinism: it uses stream-K with a fixup pass
  ([ds4_mmq.cu#L636](cuda/mmq/ds4_mmq.cu#L636)), which runs in a fixed order.
  Still confirm two runs are byte-identical, or turn stream-K off for this
  shape.
- Numerics: the same int8 activation quantisation; only the order of the float
  scale sums changes.
- Expect well under 0.5 ms per layer.

**B. Attention (~-130 ms at ctx 256, growing with context on the global layers).**

- Step one, a few minutes: [g4_attn_kernel](cuda/gemma4_attn.cuh#L83-L85)
  computes **two double-precision `exp` per key**, redundantly on all 32
  lanes. On GeForce, FP64 runs at 1/64 the FP32 rate.
  - Replace them with `exp2f` on log2e-prescaled scores.
  - This is a tiny numerical change, below the PyTorch f32 softmax the
    reference uses; gate it like any other.
  - Check first with ncu (FP64 pipe utilisation). This alone is expected to
    remove a large part of the 145 ms.
- Step two: a group-shared tile kernel.
  - One block per (KV head, 16-32 query rows), K/V tiles staged once in shared
    for all 16 query heads.
  - Process keys in chunks with one online-softmax update per chunk, not per
    key.
  - Then `mma` for QK^T and PV. The landed decode fast kernel
    ([gemma4_attn_fast.cuh](cuda/gemma4_attn_fast.cuh)) is the starting point.

## 3. Fixing the L1TEX stalls without a rewrite

The first four keep every result bit-identical:

1. **Load whole codewords.** Fetch each codeword as one `uint2` (8 B aligned,
   `index*8`) for main and residual: 16 byte loads become 2.
2. **Hoist and batch loads.** Read `gain` once per row instead of once per
   vector. Issue all of a thread's index loads, then all its codebook loads,
   then do the math and stores. That puts 8-16 loads in flight instead of one
   dependent chain.
3. **Codebook into shared.** Put the 64 KB + 2 KB codebook in dynamic shared
   memory with `__launch_bounds__(512,1)`: 16 warps, <=128 registers,
   persistent grid, items walked in a fixed order (deterministic). GLU at
   BK=64 needs 105 KB, which doesn't fit, so do one of:
   - BK=32 (about 86 KB total), or
   - decode gate and up through one shared weight buffer in turn.
4. **Vectorise the activation tile.**
   [`g4_vqp_activation_tile`](cuda/gemma_vq_prefill.cuh#L148-L157) loads one
   float at a time and re-converts the same tokens for each of the 6 (gate/up)
   or 22 (down) row tiles. Use `float4` loads, or convert to bf16 once per
   layer.
5. **Double-buffer.** Decode round k+1 while computing round k. Today
   decode -> sync -> compute -> sync leaves nothing to hide latency with 2
   blocks per SM.

Fixes 1-4 could plausibly bring the scalar kernel to 1.5-2x and make it a clear
win over restore before any MMA work. To confirm the cause in one ncu run:
Source page, sort by L1TEX stall, and check whether the stalled loads are the
`c1[]`/`c2[]` codeword reads.

## Suggested order

| # | Change | Est. saving per 256-token chunk | Effort | Numerical class |
|---|---|---|---|---|
| 1 | VQ load fixes (3.1-3.4) | 50-150 ms | hours | bit-identical |
| 2 | Q8_0 down -> MMQ | ~135 ms | hours | accumulation order |
| 3 | Attention: double `exp` out, then group-shared tile kernel | 40-80 ms, then ~130 ms total | short, then days | exp ulp, then order |
| 4 | Gemma copy of `vqm`/`vqs` MMA, then remove restore | ~200+ ms total vs today | days | accumulation order |

After that, the remaining ~10% (rope, router, GELU and other small kernels,
~69 ms) is probably launch fragmentation and worth an nsys look. Re-measure the
whole prefill after each step, per the playbook.
