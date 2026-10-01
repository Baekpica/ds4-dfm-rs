# Prism Bonsai (qwen35) prefill: the bf16 gated delta-net scalars — 2026-09-30

The family's two gated-delta-net scalar projections are the only bf16 weights in
the artifact (`ssm_alpha`, `ssm_beta`, 48 layers x 2, each 48 x 5120).  A prefill
chunk ran them through a one-warp-per-row matvec that walked the whole token
range inside each warp and re-read its weight row per token.  Tiling the token
range (one block owns 8 tokens; the row is read once per tile) cut that kernel's
GPU time in a 463-token chunk from 259.0 ms to 13.7 ms, and on a 2140-token
served prompt took ttft from 3291.9 ms to 2154.0 ms — prefill 663.2 to 1023.3
tokens/s — with every generated id unchanged.

Host: RTX 4070 SUPER (sm_89), `/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf`,
clocks 2775-2790 MHz of a 3105 MHz maximum, 119-123 W, 49-53 C.

Branch `feature/qwen35-port`.  No sibling commit corresponds to this: the
sibling's four unported perf commits (9251222 GEMV dispatch, d5a1113 fold-launch
collapse, a6c7bdd split-K attention, 758926b device-resident weights) are
already in this tree — the first two directly, the third as
`DS4_CUDA_COPY_MODEL=1` which the launcher sets on every CUDA run, and the
fourth is moot here because this tree's graph never selects the token-tile
attention kernel.  This unit came out of profiling this tree, not out of the
port list.

## The measurement (baseline)

`nsys profile`, session path (the chunked prefill the server uses), a 463-token
prompt and one step, so the window is prefill plus a single decode.  GPU time by
kernel:

    matvec_bf16                      96 calls   2,697,939 ns each   259.0 ms   39.8%
    mul_mat_q<PQ2_0,128>            400 calls     582,092 ns each   232.8 ms   35.8%
    gdn_scan<4,128>                  48 calls   1,176,962 ns each    56.5 ms    8.7%
    fold_rotate                     498 calls      93,668 ns each    46.6 ms    7.2%
    conv / swiglu / the rest                                       ~56.1 ms    8.5%
    window total                                                          ~651 ms

`matvec_bf16` was the largest single kernel and the 96 calls are exactly the 48
gdn layers x the two scalars.  Why the existing path was inefficient, from the
code and the launch geometry rather than from a guess:

- the kernel assigns one warp per output row, so M=48 gives `grid ((M+3)/4) = 12`
  blocks of 128 threads — 48 warps on a 56-SM card, and each warp then iterates
  the whole token range in registers, `for (t = 0; t < T; t++)`, reading its
  weight row again on every pass.  On a 463-token chunk that is 463 passes over
  480 KB of weights per call, from 12 blocks.
- ncu could not be used: this host answers `ERR_NVGPUCTRPERM` for performance
  counters without root, so the kernel's own DRAM traffic is inferred from the
  geometry and from the A/B below, not counter-measured.  Recorded as a limit.

## What landed

- `cuda/qwen35_attn_gdn.cuh`: `matvec_bf16_tiled` next to the existing
  `matvec_bf16`.  Each block owns `MATVEC_BF16_TT` (8) consecutive tokens of its
  4 rows and keeps 8 accumulators in registers, so one weight load feeds all 8
  tokens and the token range is spread over `gridDim.y`
  (`ceil(463/8) = 58` token tiles for a 463-token chunk, instead of 1).
- The dispatch in `ds4_gpu_qwen35_matvec_bf16_tensor`: `T > MATVEC_BF16_TT`
  takes the tiled kernel, everything else keeps the untiled one, which is
  already the right shape for a single decode row.  So decode is untouched by
  construction, and only the chunked path — the only path with the re-read —
  moves.  `DS4_QWEN35_BF16_MATVEC_TILED=0` is the kill switch back to the old
  path (used as the base arm of the A/B below).  The tail is safe by
  construction: `gridDim.y = ceil(T/8)`, `nt = min(8, T - t0)` and every access
  is behind `j < nt`, so `t0 + nt <= T` for any width.
- `tests/test_qwen35_cuda.cu`: test 6, `test_bf16_matvec_tile`, at the real
  (48, 5120) shape and the real widths (1, 8, 9, 11, 486 — the last is the width
  a session run on this host prefilled): both arms must agree bit-for-bit and
  both must sit inside the float-rounding band of a double-precision dot.  The
  numerical contract is bit-identity: the tile only interleaves the loops, so
  every token keeps the lane-strided sum order over i and the same warp
  reduction.

## Evidence

Parity, `make test-qwen35-cuda CUDA_ARCH=sm_89` (the new group, then the whole
suite):

    bf16 tile T=9:   bit-identical to the untiled kernel, max abs 1.91e-05, rel L2 2.42e-07
    bf16 tile T=11:  bit-identical to the untiled kernel, max abs 1.81e-05, rel L2 2.32e-07
    bf16 tile T=486: bit-identical to the untiled kernel, max abs 2.67e-05, rel L2 2.36e-07
    bf16 matvec tile parity: PASS
    PQ2_0 CUDA parity: PASS

Speed, served prompt of 2140 tokens with 64 decode tokens, three interleaved
pairs, a fresh server process per sample, one binary for both arms (the kill
switch selects the path):

    arm  base (TILED=0)            tiled (default)
    1    ttft 3332.4 ms  654.7 t/s     2138.8 ms  1030.2 t/s
    2    ttft 3243.3 ms  673.1 t/s     2199.2 ms  1000.9 t/s
    3    ttft 3300.0 ms  661.8 t/s     2123.9 ms  1038.7 t/s
    mean ttft 3291.9 ms  663.2 t/s     2154.0 ms  1023.3 t/s     -34.6% / +54.3%

    decode in the same runs: 17.80/17.80/17.90 t/s base, 18.00/17.50/18.00 t/s
    tiled — unchanged within noise, as the dispatch intends.

The kernel mix in the same 463-token session window after the change, which is
where the prediction is checked: `matvec_bf16_tiled` 96 calls x 142,967 ns =
13.7 ms (3.4%), against 259.0 ms before, and the window total fell about 651 ms
to 400 ms.  The 245 ms the kernel gave up accounts for essentially all of the
251 ms explained by the ttft drop at 2140 tokens.

Unchanged gates: `./run-bonsai.sh ids` IDENTICAL on both backends,
`./run-bonsai.sh session` IDENTICAL (16 ids), `make test-qwen35-session`
PASS on CUDA and on the CPU reference, `make test-qwen35-session-multichunk`
PASS, `make test-qwen35-rust-host` PASS on both backends.

## What the retained baseline now costs

Prefill is now the PQ2_0 tile GEMM (`mul_mat_q<PQ2_0,128>`, 57.2% of the
window), then `gdn_scan` (13.5%) and `fold_rotate` (12.2%).  Decode is a
different story and the next measured target: 17.8 tokens/s at a 2140-token
context against 36.8 tokens/s on a 50-token prompt, so roughly 29 ms of each
56 ms token is context work — the row-exact attention, which this tree's graph
still runs with `splits = 1` because it hands the kernel no partial buffer.
That is split-K attention (the sibling's a6c7bdd), and its blocker is the
partial buffer's size, which must be sized for the widest call the graph can
make.

## Limits

- The tile is a fixed 8 tokens; nothing sweeps the tile size or the block shape.
- ncu counters are unavailable on this host (see above), so no DRAM/L2 traffic
  reading backs the geometry argument.
- Decode is untouched by design, so the long-context decode gap above remains.
- The 2140-token served prompt is one workload; the family's caps still carry no
  qualified limit.
