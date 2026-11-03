#ifndef DS4_GEMMA4_HEAD_CUH
#define DS4_GEMMA4_HEAD_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "gemma4_primitives.cuh"

/* Gemma-4 vocabulary head, batch 1. The tied token table is BF16
 * (ds4_gemma4_bind.inc: token_embd is DS4_TENSOR_BF16), so the head is a
 * 262144 x 2816 BF16 GEMV. The arithmetic is the shared runtime's
 * glmvq_bf16_dot split==1 sequence (cuda/glm_vq_gemv.cuh:34-80), which
 * measures about 466 GB/s on this card: each lane owns the columns
 * lane*8 + k*256, accumulates the same eight products per uint4 in the same
 * order, reduces through the same XOR tree and rounds once. This file is the
 * family's own copy so the head does not run sibling code; the input row is
 * staged once per block in shared and the softcap is fused into the epilogue.
 *
 * g4_bf16_lo/hi are the half-word widenings the shared kernel calls
 * glmvq_bf16_lo/hi (cuda/glm_vq_gemv.cuh:17-18). */

/* One warp per row, eight warps per block, a persistent grid over the rows:
 * the block stages the input row once and every warp then streams its rows'
 * weights. SOFTCAP applies g4_softcap_kernel's exact three roundings to the
 * bf16-rounded sum before the store. */
template <uint32_t COLS, bool SOFTCAP>
__global__ void __launch_bounds__(256, 4) g4_head_gemv_kernel(
        const __nv_bfloat16 *__restrict__ weight, const float *__restrict__ input,
        float *__restrict__ output, uint32_t rows, float cap) {
    static_assert(COLS % 256u == 0u, "the lane stride covers 256 columns per step");
    __shared__ __align__(16) float sx[COLS];
    for (uint32_t c = threadIdx.x; c < COLS; c += 256u) { sx[c] = g4_round(input[c]); }
    __syncthreads();
    const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint32_t base = lane * 8u;
    const uint32_t step = gridDim.x * 8u;
    for (uint32_t row = blockIdx.x * 8u + warp; row < rows; row += step) {
        const __nv_bfloat16 *w = weight + (uint64_t)row * COLS + base;
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t k = 0; k < COLS / 256u; k++) {
            const uint4 wv = __ldcs((const uint4 *)(w + k * 256u));
            const float4 a = *(const float4 *)(sx + base + k * 256u);
            const float4 b = *(const float4 *)(sx + base + k * 256u + 4u);
            sum = fmaf(g4_bf16_lo(wv.x), a.x, sum);
            sum = fmaf(g4_bf16_hi(wv.x), a.y, sum);
            sum = fmaf(g4_bf16_lo(wv.y), a.z, sum);
            sum = fmaf(g4_bf16_hi(wv.y), a.w, sum);
            sum = fmaf(g4_bf16_lo(wv.z), b.x, sum);
            sum = fmaf(g4_bf16_hi(wv.z), b.y, sum);
            sum = fmaf(g4_bf16_lo(wv.w), b.z, sum);
            sum = fmaf(g4_bf16_hi(wv.w), b.w, sum);
        }
        #pragma unroll
        for (unsigned s = 16u; s; s >>= 1u) { sum += __shfl_xor_sync(0xffffffffu, sum, s); }
        if (lane == 0u) {
            float y = g4_round(sum);
            if constexpr (SOFTCAP) {
                const float divided = g4_round(y / cap);
                const float squashed = g4_round((float)tanh((double)divided));
                y = g4_round(squashed * cap);
            }
            output[row] = y;
        }
    }
}

#endif
