#ifndef DS4_MINIMAX_ROWS_CUH
#define DS4_MINIMAX_ROWS_CUH

#include "../ds4_minimax.h"
#include "glm_vq_q4p.cuh"

enum { MM_ROW_THREADS = 256, MM_ROW_VALUES = DS4_MINIMAX_HIDDEN / MM_ROW_THREADS };

// Sum of `value` over the block (256 threads); every thread gets it.
__device__ __forceinline__ float mm_block_sum(float value, float *scratch) {
    for (uint32_t stride = 16; stride; stride >>= 1) { value += __shfl_xor_sync(0xffffffffu, value, stride); }
    if ((threadIdx.x & 31) == 0) { scratch[threadIdx.x / 32] = value; }
    __syncthreads();
    float sum = 0.0f;
    for (uint32_t w = 0; w < MM_ROW_THREADS / 32; w++) { sum += scratch[w]; }
    __syncthreads();
    return sum;
}

// A norm's 24 weights per thread (index tid + j * 256), as F32. They are
// constant: kernels load them before glmvq_pdl_wait.
__device__ __forceinline__ void mm_row_weights(const __nv_bfloat16 *weight, float *w) {
    #pragma unroll
    for (uint32_t j = 0; j < MM_ROW_VALUES; j++) { w[j] = __bfloat162float(weight[threadIdx.x + j * MM_ROW_THREADS]); }
}

// Gemma RMSNorm of one 6144-wide row held as 24 values per thread:
// x * rsqrt(mean(x^2) + eps), times (1 + w) in F32, rounded once.
__device__ __forceinline__ void mm_row_rms(const float *x, const float *w, float *out, float *scratch) {
    float sum = 0.0f;
    for (uint32_t j = 0; j < MM_ROW_VALUES; j++) { sum = fmaf(x[j], x[j], sum); }
    const float inv = __frsqrt_rn(glmvq_add(__fdiv_rn(mm_block_sum(sum, scratch), (float)DS4_MINIMAX_HIDDEN), 1e-6f));
    for (uint32_t j = 0; j < MM_ROW_VALUES; j++) {
        out[j] = glmvq_round(glmvq_mul(glmvq_add(1.0f, w[j]), glmvq_mul(x[j], inv)));
    }
}

// q4p consumes FP16 x*2^-e in quad order (0,2,1,3), with scale 2^(e-6).
// Keep e=0 until the row's peak exceeds 2^15, as glmvq_q4p_prep does.
template <uint32_t VALUES>
__device__ __forceinline__ void mm_row_q4p(const float *x, uint32_t cols, __half *xh, float *xs, float *scratch) {
    float peak = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < VALUES; j++) { peak = fmaxf(peak, fabsf(x[j])); }
    for (uint32_t stride = 16; stride; stride >>= 1) { peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, stride)); }
    if ((threadIdx.x & 31) == 0) { scratch[threadIdx.x / 32] = peak; }
    __syncthreads();
    peak = 0.0f;
    for (uint32_t w = 0; w < MM_ROW_THREADS / 32; w++) { peak = fmaxf(peak, scratch[w]); }
    __syncthreads();
    int e = 0;
    if (peak > GLMVQ_Q4P_RANGE) { frexpf(peak / GLMVQ_Q4P_RANGE, &e); }
    const float scale = ldexpf(1.0f, -e);
    __half *row = xh + (uint64_t)blockIdx.x * cols;
    #pragma unroll
    for (uint32_t j = 0; j < VALUES; j++) {
        const uint32_t c = threadIdx.x + j * MM_ROW_THREADS, k = c & 3u;
        row[(c & ~3u) | (k == 1u ? 2u : k == 2u ? 1u : k)] = __float2half_rn(x[j] * scale);
    }
    if (!threadIdx.x) { xs[blockIdx.x] = ldexpf(1.0f, e - GLMVQ_Q4P_SHIFT); }
}

// With q4h, the norm row also lands as q4p's prepared input (the q4p pass
// over it read 100 MB and wrote 50 MB per 4,096-row call).
template <typename CT, typename BT>
__global__ void mm_residual_kernel(CT *cur, const BT *branch, float *norm, const __nv_bfloat16 *next,
        __half *q4h, float *q4s) {
    __shared__ float scratch[MM_ROW_THREADS / 32];
    glmvq_pdl_open();
    float w[MM_ROW_VALUES];
    if (next) { mm_row_weights(next, w); }
    glmvq_pdl_wait();
    const uint64_t base = (uint64_t)blockIdx.x * DS4_MINIMAX_HIDDEN;
    float x[MM_ROW_VALUES], y[MM_ROW_VALUES];
    // Every load before the first store: cur may alias branch for the
    // compiler, and storing each sum before the next loads serialized 48 L2
    // round trips (9.4 us of a decode row's 24).
    #pragma unroll
    for (uint32_t j = 0; j < MM_ROW_VALUES; j++) {
        const uint64_t at = base + threadIdx.x + j * MM_ROW_THREADS;
        x[j] = glmvq_value(cur[at]);
        y[j] = glmvq_value(branch[at]);
    }
    #pragma unroll
    for (uint32_t j = 0; j < MM_ROW_VALUES; j++) {
        x[j] = glmvq_round(glmvq_add(x[j], glmvq_round(y[j])));
        cur[base + threadIdx.x + j * MM_ROW_THREADS] = glmvq_cast<CT>(x[j]);
    }
    if (!next) { return; }
    mm_row_rms(x, w, y, scratch);
    for (uint32_t j = 0; j < MM_ROW_VALUES; j++) { norm[base + threadIdx.x + j * MM_ROW_THREADS] = y[j]; }
    if (q4h) { mm_row_q4p<MM_ROW_VALUES>(y, DS4_MINIMAX_HIDDEN, q4h, q4s, scratch); }
}

// Prefill MoE output is consumed only by this residual: keep its four
// expert rows in registers instead of materializing a BF16 branch.
template <typename CT, typename ST>
__global__ void mm_moe_residual_kernel(CT *cur, const __nv_bfloat16 *down, const float *weights,
        const uint32_t *pairs, const uint32_t *order, const ST *shared, float *norm,
        const __nv_bfloat16 *next, __half *q4h, float *q4s) {
    __shared__ float scratch[MM_ROW_THREADS / 32];
    const uint64_t base = (uint64_t)blockIdx.x * DS4_MINIMAX_HIDDEN;
    uint64_t expert[DS4_MINIMAX_USED];
    float scale[DS4_MINIMAX_USED];
    #pragma unroll
    for (uint32_t k = 0; k < DS4_MINIMAX_USED; k++) {
        const uint32_t r = order[blockIdx.x * DS4_MINIMAX_USED + k];
        expert[k] = (uint64_t)r * DS4_MINIMAX_HIDDEN;
        scale[k] = weights[pairs[r]];
    }
    float w[MM_ROW_VALUES], x[MM_ROW_VALUES], y[MM_ROW_VALUES];
    if (next) { mm_row_weights(next, w); }
    #pragma unroll
    for (uint32_t j = 0; j < MM_ROW_VALUES; j++) {
        const uint32_t col = threadIdx.x + j * MM_ROW_THREADS;
        x[j] = glmvq_value(cur[base + col]);
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t k = 0; k < DS4_MINIMAX_USED; k++) {
            sum = glmvq_add(sum, glmvq_mul(__bfloat162float(down[expert[k] + col]), scale[k]));
        }
        y[j] = glmvq_round(glmvq_add(sum, glmvq_round(glmvq_value(shared[base + col]))));
    }
    // Retain the separate branch's rounding before the residual add.
    #pragma unroll
    for (uint32_t j = 0; j < MM_ROW_VALUES; j++) {
        x[j] = glmvq_round(glmvq_add(x[j], glmvq_round(y[j])));
        cur[base + threadIdx.x + j * MM_ROW_THREADS] = glmvq_cast<CT>(x[j]);
    }
    if (!next) { return; }
    mm_row_rms(x, w, y, scratch);
    for (uint32_t j = 0; j < MM_ROW_VALUES; j++) { norm[base + threadIdx.x + j * MM_ROW_THREADS] = y[j]; }
    if (q4h) { mm_row_q4p<MM_ROW_VALUES>(y, DS4_MINIMAX_HIDDEN, q4h, q4s, scratch); }
}

#endif
