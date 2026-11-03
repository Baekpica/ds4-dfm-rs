# Gemma-4 direct VQ prefill — 2026-11-03

Round starts from `c09cfbf9` on `feat/gemma4-family`. Artifact
`/data/vq-artifacts/gguf-family-it-r8` (R13+8 dense-VQ family build);
`release.json` SHA256
`0bd8a1abd260909735431d3d4da1925648228e07a0d97bd66fd07573448bcb36`.

One RTX 4070 SUPER, 56 SM, `sm_89`, driver 610.43.03, CUDA 13.3.73,
12282 MiB, max SM clock 3105 MHz, kernel 6.12.113+deb13-amd64.
One model process at a time; the model is device-resident
(`DS4_CUDA_COPY_MODEL=1`, 9.37 GiB copied, 0.05 GiB KV at ctx 256).
No MTP, no drafter, no CPU weight offload.

## Adopted round

| Round | Env | Change | Paired median gain |
|---|---|---|---:|
| P1 | `DS4_GEMMA4_VQ_PREFILL=1` | Direct VQ expert prefill GEMM, one work-item launch per layer | Prefill +30.8% |

The round replaces the per-expert restore-to-bf16 plus cuBLAS path for the
routed experts with a kernel that decodes the GVQ1 bytes into a shared tile
and multiplies there. The container format is unchanged; the records are the
existing R13+8 ones. Gate and up are fused with the GELU epilogue, the down
projection is the same kernel with one matrix.

## Round protocol

Fresh processes, same executable, artifact and prompt; three consecutive
off runs followed by three consecutive on runs at ctx 256 with one 256-token
prefill and one generated token. The prefill rate is the bench's own
`prefill_tps`; the ranges do not overlap.

```sh
DS4_CUDA_COPY_MODEL=1 ./ds4-bench -m /data/vq-artifacts/gguf-family-it-r8 \
  --cuda --prompt-file /tmp/g4-bench-256.txt --ctx-start 256 --ctx-max 256 \
  --gen-tokens 1
# Repeat with DS4_GEMMA4_VQ_PREFILL=1 in fresh processes.
```

Prompt SHA256:
`ecde80544a343f191c655ab6591d71db415d68aa9fce6945fe9bc1c9dab44d26`.

## Results

| Run | Off (tok/s) | On (tok/s) |
|---|---:|---:|
| 1 | 338.14 | 425.49 |
| 2 | 329.50 | 450.61 |
| 3 | 351.22 | 442.21 |
| Median | 338.14 | 442.21 |
| Mean | 339.62 | 439.44 |

Median gain +30.8%, mean +29.4%. The off range is 329.50–351.22, the on
range 425.49–450.61.

Kernel attribution, nsys, same binary and protocol:

| Kernel | Before the load fixes | After |
|---|---:|---:|
| `g4_vq_gemm_items_kernel<GLU>` | 7.50 ms/call (225.0 ms, 30 calls) | 3.24 ms/call (97.2 ms, 30 calls) |
| `g4_vq_gemm_items_kernel<single>` | 3.66 ms/call (109.8 ms, 30 calls) | 1.38 ms/call (41.5 ms, 30 calls) |

The per-layer expert cost fell from 11.16 ms to 4.62 ms. The reference path
pays the restore (207 ms per chunk in the earlier profile) plus the cuBLAS
expert GEMMs; the direct path pays one weight decode plus scalar FMA and no
bf16 expansion.

## ncu

Root-side capture, one launch, 606 blocks x 256 threads, `--clock-control`
left at default:

```sh
sudo env DS4_LOCK_FILE=/tmp/g4-ncu.lock DS4_CUDA_COPY_MODEL=1 \
  DS4_GEMMA4_VQ_PREFILL=1 /usr/local/cuda/bin/ncu \
  --kernel-name regex:g4_vq_gemm_items --launch-count 1 \
  --section SpeedOfLight --section Occupancy --section WarpStateStats \
  --target-processes all ./ds4-bench -m /data/vq-artifacts/gguf-family-it-r8 \
  --cuda --prompt-file /tmp/g4-bench-256.txt --ctx-start 256 --ctx-max 256 \
  --gen-tokens 1
```

The first capture, before the load fixes, measured: duration 6.66 ms,
occupancy 33.3% theoretical / 32.2% achieved (register- and shared-limited,
111 registers and 39 KB shared, 2 blocks/SM), warp cycles per issued
instruction 13.54 of which 8.3 (61.3%) are L1TEX scoreboard stalls, DRAM
throughput 3.78%, L2 60.54%, L1/TEX 53.65%, SM compute 28.0%, SM clock
2.49 GHz. That is the evidence that the kernel was waiting on codebook
lookups that missed L1, not on DRAM: two blocks of 39 KB shared leave about
50 KB of L1, less than the 64 KB main codebook.

The applied fixes, all bit-identical in arithmetic:

- fetch each E4M3 codeword as one 8-byte `uint2` instead of eight byte loads;
- stage the 2 KB residual book in shared memory once per block (2 KB for the
  single matrix, 4 KB for the fused pair) so the main book stays in L1;
- map two threads per weight row, four contiguous vectors each, so the four
  residual indices are one 32-bit load;
- hoist the row gain out of the K loop.

Registers after the fixes: 128 for the fused kernel, 80 for the single
matrix; shared 43.3 KB and 23.8 KB. Occupancy is unchanged at 2 blocks/SM;
the win came from the load pattern, not from more warps.

## Numerical class

`DS4_GEMMA4_VQ_VERIFY=1` compares the direct GEMM against the restore plus
cuBLAS path on the live expert records. The gate and up GEMMs differ by
relL2 1.9e-3 to 2.9e-3 (max 7.8e-3 to 1.6e-2); the down GEMM by 1e-5 to
1e-9. The same harness runs the reference against itself at half the token
count (`ref gate n/2 vs n`), which measures cuBLAS's own shape dependence:
relL2 0.0 to 3.3e-3. The direct kernel therefore sits inside the reference's
own chunk-width noise band; the residual difference is the K accumulation
order that cuBLAS splits by shape, not a decode error.

The output is rounded to bf16 at the same boundary `glmvq_gemm` applies
(cuBLAS writes bf16 and `glmvq_convert` widens it). Omitting that boundary
moved the 256-token frontier logits by 23% relL2, so it is part of the
contract.

Eight greedy tokens are identical with the option off and on:
`[6113, 1698, 236761, 28333, 506, 877, 6910, 3136]`. Frontier logits at
ctx 256 differ by relL2 2.14e-1 (max 8.88), the accumulated gate/up
accumulation-order difference over 30 layers; the acceptance gate is the
token sequence, as for every other VQ family.

## Limits

- Measured at ctx 256 only, device-resident. The 1024-token ctx does not fit
  with the model copied (9.37 GiB weights plus the lazy session graph exceed
  12 GiB); a longer prefill needs zero-copy, which changes the comparison.
- The wall time of each bench run is dominated by the Rust-side identify and
  load, so the GPU duty cycle is about 5%; the prefill rate is the bench's
  own phase timing, and the kernel attribution above is the GPU-side check.
- The reference's own n/2-vs-n band is measured on the same records at the
  same shapes; it is the right noise floor for this comparison but not a
  general claim about cuBLAS.
- The Spark (GB10, sm_121a) is unmeasured for this round.

## Reviews

Two advisor reviews were taken on the same evidence, from the Antigravity
sessions: Gemini 4 Argon (tmux session 14) and Claude Opus 5.5 (session 15,
saved as `agy_opus_review.md`). Both rank the next units the same way:

1. Q8_0 dense down (2112x2816 at 256 tokens, 4.8 ms/layer on the naive
   `matmul_q8_0_preq_kernel`, 0.63 TFLOP/s) to a tiled MMQ path: about
   135-140 ms per chunk. The shape is `K = 8*256 + 64`; the existing fused
   tail path handles `K = 256n + 128` only, so it needs a 64-element tail
   (two 32-value blocks) or a Gemma-side tiled kernel.
2. Prefill attention (`g4_attn_kernel`, 2.4 ms/layer at ctx 256): first
   replace the two double-precision `exp` per key with `exp2f`, then a
   group-shared tile kernel that stages K/V once for the 16 query heads of a
   KV head, starting from `cuda/gemma4_attn_fast.cuh`.
3. The remaining direct-VQ levers: drop the second weight tile (decode gate
   and up through one shared buffer in turn, freeing 17 KB for L1) and
   double-buffer the decode against the compute.

The Q8_0 down is the next unit; the attention work follows it.
