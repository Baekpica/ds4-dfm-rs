#ifndef DS4_GLM_VQ_Q4_CUH
#define DS4_GLM_VQ_Q4_CUH

#include "q4_k.cuh"

enum { GLMVQ_Q4_BLOCK = DS4_Q4_K_BLOCK, GLMVQ_Q4_BYTES = DS4_Q4_K_BYTES, GLMVQ_Q4_WARPS = 8 };

// Q4_K layout/dot mapping: YoungAi 7fb26cc, cuda_v41_q4k.inc.cu:21-70.
// Unlike GVQ1 these weights use local affine groups, without a codebook.
template <typename T>
__global__ void glmvq_q4_restore(const uint8_t *weight, T *out, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        out[i] = glmvq_cast<T>(ds4_q4_k_value(weight + i / GLMVQ_Q4_BLOCK * GLMVQ_Q4_BYTES,
            i % GLMVQ_Q4_BLOCK));
    }
}

template <uint32_t NT>
__global__ void glmvq_q4_small(const uint8_t *weight, const float *input,
        float *out, uint32_t cols, uint32_t rows, uint32_t tokens, uint32_t split) {
    __shared__ float partial[GLMVQ_Q4_WARPS * NT];
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint32_t row = blockIdx.x * (GLMVQ_Q4_WARPS / split) + warp / split;
    const uint32_t part = warp % split, blocks = cols / GLMVQ_Q4_BLOCK;
    float sum[NT] = {};
    if (row < rows) {
        for (uint32_t b = part; b < blocks; b += split) {
            const uint8_t *packed = weight + ((uint64_t)row * blocks + b) * GLMVQ_Q4_BYTES;
            // Each warp covers one block; independent rows expose enough
            // parallelism for a single token without global BF16 restoration.
            #pragma unroll
            for (uint32_t j = 0; j < GLMVQ_Q4_BLOCK; j += 32) {
                const float w = ds4_q4_k_value(packed, j + lane);
                #pragma unroll
                for (uint32_t t = 0; t < NT; t++) {
                    if (t < tokens) {
                        sum[t] = fmaf(w, input[(uint64_t)t * cols + b * GLMVQ_Q4_BLOCK + j + lane], sum[t]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        for (int stride = 16; stride; stride >>= 1) {
            sum[t] += __shfl_xor_sync(0xffffffffu, sum[t], stride);
        }
        if (!lane) { partial[warp * NT + t] = sum[t]; }
    }
    __syncthreads();
    if (row >= rows || lane || part) { return; }
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        if (t >= tokens) { continue; }
        float value = 0;
        for (uint32_t p = 0; p < split; p++) { value += partial[(warp + p) * NT + t]; }
        out[(uint64_t)t * rows + row] = __bfloat162float(__float2bfloat16_rn(value));
    }
}

static cudaError_t glmvq_q4_decode(const void *weight, const float *input, float *out,
        uint32_t rows, uint32_t cols, uint32_t tokens, cudaStream_t stream) {
    if (!weight || !input || !out || !rows || !cols || cols % GLMVQ_Q4_BLOCK ||
        !tokens || tokens > GLMVQ_Q4_WARPS || (uintptr_t)weight % 16) { return cudaErrorInvalidValue; }
    uint32_t split = 1;
    constexpr uint32_t target_warps = 8192;
    while (split < GLMVQ_Q4_WARPS && split * rows < target_warps &&
        split * 2 <= cols / GLMVQ_Q4_BLOCK) { split *= 2; }
    const uint32_t rpb = GLMVQ_Q4_WARPS / split, grid = (rows + rpb - 1) / rpb;
    if (tokens == 1) {
        glmvq_q4_small<1><<<grid, GLMVQ_Q4_WARPS * 32, 0, stream>>>(
            (const uint8_t *)weight, input, out, cols, rows, tokens, split);
    } else {
        glmvq_q4_small<8><<<grid, GLMVQ_Q4_WARPS * 32, 0, stream>>>(
            (const uint8_t *)weight, input, out, cols, rows, tokens, split);
    }
    return cudaGetLastError();
}
#endif
