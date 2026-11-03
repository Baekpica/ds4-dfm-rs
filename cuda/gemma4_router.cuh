#ifndef DS4_GEMMA4_ROUTER_CUH
#define DS4_GEMMA4_ROUTER_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "gemma4_primitives.cuh"

/* Gemma-4 router projection, one token: 128 experts x 2816 inputs, BF16.
 * The arithmetic is the shared BF16 dot's split==8 sequence
 * (cuda/glm_vq_gemv.cuh:37-80 with GLMVQ_DOT_SPLIT = 8, chosen there because
 * rows 128 < 4096 and cols 2816 >= 2048): block (expert) owns one row, warp
 * part p covers the columns (p*32 + lane)*8 + k*2048, each lane accumulates
 * its eight products per uint4 in order, the warp reduces through
 * __shfl_xor_sync, and thread 0 sums partial[0..7] left to right and rounds
 * once. The family keeps its own copy so the router does not run sibling
 * code; the mapping is unchanged, so the logits are the shared kernel's bit
 * for bit. */
__global__ void __launch_bounds__(256) g4_router_dot_kernel(
        const __nv_bfloat16 *__restrict__ weight, const float *__restrict__ input,
        float *__restrict__ output, uint32_t rows, uint32_t cols) {
    const uint32_t lane = threadIdx.x & 31u, part = threadIdx.x >> 5u;
    const uint32_t row = blockIdx.x;
    if (row >= rows) { return; }
    const __nv_bfloat16 *w = weight + (uint64_t)row * cols;
    float sum = 0.0f;
    for (uint32_t col = (part * 32u + lane) * 8u; col < cols; col += 8u * 32u * 8u) {
        const uint4 wv = *(const uint4 *)(w + col);
        const float4 a = *(const float4 *)(input + col);
        const float4 b = *(const float4 *)(input + col + 4u);
        sum = fmaf(g4_bf16_lo(wv.x), g4_round(a.x), sum);
        sum = fmaf(g4_bf16_hi(wv.x), g4_round(a.y), sum);
        sum = fmaf(g4_bf16_lo(wv.y), g4_round(a.z), sum);
        sum = fmaf(g4_bf16_hi(wv.y), g4_round(a.w), sum);
        sum = fmaf(g4_bf16_lo(wv.z), g4_round(b.x), sum);
        sum = fmaf(g4_bf16_hi(wv.z), g4_round(b.y), sum);
        sum = fmaf(g4_bf16_lo(wv.w), g4_round(b.z), sum);
        sum = fmaf(g4_bf16_hi(wv.w), g4_round(b.w), sum);
    }
    #pragma unroll
    for (unsigned stride = 16u; stride; stride >>= 1u) {
        sum += __shfl_xor_sync(0xffffffffu, sum, stride);
    }
    __shared__ float partial[8];
    if (!lane) { partial[part] = sum; }
    __syncthreads();
    if (threadIdx.x == 0u) {
        float value = 0.0f;
        for (uint32_t i = 0; i < 8u; i++) { value += partial[i]; }
        output[row] = g4_round(value);
    }
}

#endif
