#ifndef DS4_AXK2_NORM_CUH
#define DS4_AXK2_NORM_CUH

#include "glm_vq_ops.cuh"
#include "../ds4_axk2.h"
#include <cuda_pipeline.h>

static const float AX_EPS = 1e-6f;
enum { AX_ROW_THREADS = 256, AX_ROW_VALUES = DS4_AXK2_HIDDEN / AX_ROW_THREADS, AX_ROW_WARPS = AX_ROW_THREADS / 32 };
static_assert(DS4_AXK2_HIDDEN % (AX_ROW_THREADS * 8) == 0 || DS4_AXK2_HIDDEN % AX_ROW_THREADS == 0, "whole rows per block");

// Sum of `value` over the 256-thread block; every thread gets it.
__device__ __forceinline__ float ax_block_sum(float value, float *scratch) {
    for (uint32_t stride = 16; stride; stride >>= 1) { value += __shfl_xor_sync(0xffffffffu, value, stride); }
    if ((threadIdx.x & 31) == 0) { scratch[threadIdx.x / 32] = value; }
    __syncthreads();
    float sum = 0.0f;
    for (uint32_t w = 0; w < AX_ROW_WARPS; w++) { sum += scratch[w]; }
    __syncthreads();
    return sum;
}

enum class ax_norm_load { reference, staged };

enum { AX_GNORM_MID = DS4_AXK2_GATE_RANK + 1, AX_GNORM_ROWS = 4 };

template <ax_norm_load LOAD>
__global__ void __launch_bounds__(AX_ROW_THREADS) ax_gnorm_down_kernel(float *mid, const float *cur,
        const float *branch, const __nv_bfloat16 *weight, const __nv_bfloat16 *down) {
    __shared__ float scratch[AX_ROW_WARPS];
    __shared__ __align__(16) __nv_bfloat16 y_row[DS4_AXK2_HIDDEN];
    __shared__ __align__(16) __nv_bfloat16 w_row[LOAD == ax_norm_load::staged ? DS4_AXK2_HIDDEN : 1];
    glmvq_pdl_open();
    const uint32_t t = threadIdx.x, lane = t & 31, k = blockIdx.x, r = blockIdx.y;
    if constexpr (LOAD == ax_norm_load::staged) {
        // Stage immutable weights while the preceding kernel finishes the row.
        for (uint32_t c = t; c < DS4_AXK2_HIDDEN / 8; c += AX_ROW_THREADS) {
            __pipeline_memcpy_async((uint4 *)w_row + c, (const uint4 *)(down + (uint64_t)k * DS4_AXK2_HIDDEN) + c, 16);
        }
        __pipeline_commit();
    }
    glmvq_pdl_wait();
    const uint64_t base = (uint64_t)r * DS4_AXK2_HIDDEN;
    float x[AX_ROW_VALUES];
    float sum = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < AX_ROW_VALUES; j++) {
        const uint32_t col = t + j * AX_ROW_THREADS;
        float v = glmvq_round(cur[base + col]);
        if (branch) { v = glmvq_round(glmvq_add(v, glmvq_round(branch[base + col]))); }
        x[j] = v;
        sum = fmaf(v, v, sum);
    }
    const float inv = __frsqrt_rn(glmvq_add(__fdiv_rn(ax_block_sum(sum, scratch), (float)DS4_AXK2_HIDDEN), AX_EPS));
    #pragma unroll
    for (uint32_t j = 0; j < AX_ROW_VALUES; j++) {
        const uint32_t col = t + j * AX_ROW_THREADS;
        y_row[col] = __float2bfloat16_rn(glmvq_round(glmvq_mul(__bfloat162float(weight[col]),
            glmvq_round(glmvq_mul(x[j], inv)))));
    }
    if constexpr (LOAD == ax_norm_load::staged) { __pipeline_wait_prior(0); }
    __syncthreads();
    if (t >= 32) { return; }
    const uint4 *w = LOAD == ax_norm_load::staged ? (const uint4 *)w_row :
        (const uint4 *)(down + (uint64_t)k * DS4_AXK2_HIDDEN);
    const uint4 *y = (const uint4 *)y_row;
    float acc = 0.0f;
    #pragma unroll 4
    for (uint32_t c = lane; c < DS4_AXK2_HIDDEN / 8; c += 32) {
        const uint4 wv = w[c], yv = y[c];
        const uint32_t ws[4] = {wv.x, wv.y, wv.z, wv.w}, ys[4] = {yv.x, yv.y, yv.z, yv.w};
        #pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
            acc = fmaf(__uint_as_float(ws[i] << 16), __uint_as_float(ys[i] << 16), acc);
            acc = fmaf(__uint_as_float(ws[i] & 0xffff0000u), __uint_as_float(ys[i] & 0xffff0000u), acc);
        }
    }
    for (uint32_t stride = 16; stride; stride >>= 1) { acc += __shfl_xor_sync(0xffffffffu, acc, stride); }
    if (lane) { return; }
    const float g = glmvq_round(acc);
    mid[r * AX_GNORM_MID + k] = glmvq_round(__fdiv_rn(g, glmvq_add(1.0f, expf(-g))));
    if (!k) { mid[r * AX_GNORM_MID + DS4_AXK2_GATE_RANK] = inv; }
}

#endif
