#ifndef DS4_GLM_VQ_FP8_CUH
#define DS4_GLM_VQ_FP8_CUH

#include "glm_vq_ops.cuh"

enum glmvq_fp8_feed { GLMVQ_FP8_GLOBAL, GLMVQ_FP8_SHARED };
enum { GLMVQ_FP8_WARPS = 8, GLMVQ_FP8_VECTOR = 8, GLMVQ_FP8_MAX_COLS = 16384 };
enum { GLMVQ_FP8_HIDDEN = 4096, GLMVQ_FP8_SHARED_FFN = 2048, GLMVQ_FP8_DENSE_FFN = 12288,
       GLMVQ_FP8_Q_RANK = 1536, GLMVQ_FP8_KV_RANK = 512, GLMVQ_FP8_Q_WIDTH = 16384,
       GLMVQ_FP8_CONCAT = 2 * GLMVQ_FP8_HIDDEN, GLMVQ_FP8_TOKENS = 4 };
// Ling-3.1: KDA f/g [8192,4096]; MTP q_b [12288,1536], kv_a [576,4096],
// head gate [64,4096] and shared experts [1280,4096] / [4096,1280].
enum { LING31_FP8_KDA = 8192, LING31_FP8_Q_WIDTH = 12288, LING31_FP8_KV_ROW = 576,
       LING31_FP8_GATE = 64, LING31_FP8_FFN = 1280 };

static uint32_t glmvq_fp8_split(uint32_t rows, uint32_t cols) {
    // Only measured original-artifact shapes use this N1 specialization.
    if (rows == GLMVQ_FP8_Q_WIDTH && cols == GLMVQ_FP8_Q_RANK) { return 2; }
    if (cols == GLMVQ_FP8_HIDDEN && (rows == GLMVQ_FP8_KV_RANK || rows == GLMVQ_FP8_Q_RANK ||
        rows == GLMVQ_FP8_SHARED_FFN || rows == GLMVQ_FP8_DENSE_FFN)) { return GLMVQ_FP8_WARPS; }
    // CONCAT: the MTP layer's eh_proj over [embedding norm, hidden norm].
    if (rows == GLMVQ_FP8_HIDDEN && (cols == GLMVQ_FP8_SHARED_FFN || cols == GLMVQ_FP8_DENSE_FFN ||
        cols == GLMVQ_FP8_Q_WIDTH || cols == GLMVQ_FP8_CONCAT)) { return GLMVQ_FP8_WARPS; }
    if (rows == LING31_FP8_Q_WIDTH && cols == GLMVQ_FP8_Q_RANK) { return 2; }
    if (cols == GLMVQ_FP8_HIDDEN && (rows == LING31_FP8_KDA || rows == LING31_FP8_KV_ROW ||
        rows == LING31_FP8_GATE || rows == LING31_FP8_FFN)) { return GLMVQ_FP8_WARPS; }
    if (rows == GLMVQ_FP8_HIDDEN && cols == LING31_FP8_FFN) { return GLMVQ_FP8_WARPS; }
    return 0;
}

__device__ __forceinline__ uint32_t glmvq_fp8_pack(float a, float b) {
    const __nv_bfloat162_raw value = __floats2bfloat162_rn(a, b);
    return (uint32_t)value.x | ((uint32_t)value.y << 16);
}

__device__ __forceinline__ float glmvq_fp8_pair(uint32_t word, uint32_t x,
        float scale, float sum) {
    // Original FP8 scales multiply in F32 before the BF16 weight boundary.
    const uint32_t w = glmvq_scaled_pair(word, scale);
    sum = fmaf(__uint_as_float(w << 16), __uint_as_float(x << 16), sum);
    return fmaf(__uint_as_float(w & 0xffff0000u), __uint_as_float(x & 0xffff0000u), sum);
}

// YoungAi 7fb26cc cuda_v41_q4k.inc.cu:12-15 motivates complete coalesced
// lines. GLM uses contiguous FP8 bytes and F32 128x128 block scales here.
// MTP verify rows (NT up to 4, global feed) reuse each scaled weight pair;
// every token keeps the one-token FMA and reduction order.
template <glmvq_fp8_feed FEED, uint32_t NT>
__global__ void glmvq_fp8_dot(const uint8_t *weight, const float *scales,
        const float *input, float *output, uint32_t rows, uint32_t cols, uint32_t split, uint32_t tokens) {
    extern __shared__ __align__(16) uint8_t fp8_input[];
    __shared__ float partial[GLMVQ_FP8_WARPS][NT];
    auto *staged = (__nv_bfloat16 *)fp8_input;
    if constexpr (FEED == GLMVQ_FP8_SHARED) {
        for (uint32_t i = threadIdx.x * 4; i < cols; i += blockDim.x * 4) {
            const float4 x = *(const float4 *)(input + i);
            *(uint2 *)(staged + i) = make_uint2(glmvq_fp8_pack(x.x, x.y), glmvq_fp8_pack(x.z, x.w));
        }
        __syncthreads();
    }
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const uint32_t local = warp / split, part = warp % split;
    const uint32_t row = blockIdx.x * (GLMVQ_FP8_WARPS / split) + local;
    float sum[NT] = {};
    if (row < rows) {
        const uint8_t *w = weight + (uint64_t)row * cols;
        const float *s = scales + (uint64_t)(row / GLMVQ_SCALE_BLOCK) * (cols / GLMVQ_SCALE_BLOCK);
        for (uint32_t col = part * 32 * GLMVQ_FP8_VECTOR + lane * GLMVQ_FP8_VECTOR;
                col < cols; col += split * 32 * GLMVQ_FP8_VECTOR) {
            const uint2 words = *(const uint2 *)(w + col);
            const float scale = s[col / GLMVQ_SCALE_BLOCK];
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                if (t >= tokens) { continue; }
                uint4 x;
                if constexpr (FEED == GLMVQ_FP8_SHARED) {
                    x = *(const uint4 *)(staged + col);
                } else {
                    const float *in = input + (uint64_t)t * cols + col;
                    const float4 a = *(const float4 *)in, b = *(const float4 *)(in + 4);
                    x = make_uint4(glmvq_fp8_pack(a.x, a.y), glmvq_fp8_pack(a.z, a.w),
                        glmvq_fp8_pack(b.x, b.y), glmvq_fp8_pack(b.z, b.w));
                }
                sum[t] = glmvq_fp8_pair(words.x, x.x, scale, sum[t]);
                sum[t] = glmvq_fp8_pair(words.x >> 16, x.y, scale, sum[t]);
                sum[t] = glmvq_fp8_pair(words.y, x.z, scale, sum[t]);
                sum[t] = glmvq_fp8_pair(words.y >> 16, x.w, scale, sum[t]);
            }
        }
    }
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        for (uint32_t stride = 16; stride; stride >>= 1) { sum[t] += __shfl_xor_sync(0xffffffffu, sum[t], stride); }
        if (!lane) { partial[warp][t] = sum[t]; }
    }
    __syncthreads();
    if (row >= rows || part || lane) { return; }
    for (uint32_t t = 0; t < tokens; t++) {
        float value = 0;
        for (uint32_t i = 0; i < split; i++) { value += partial[local * split + i][t]; }
        output[(uint64_t)t * rows + row] = glmvq_round(value);
    }
}

static bool glmvq_fp8_fits(const void *weight, const void *scale,
        const void *input, const void *output, uint32_t rows, uint32_t cols) {
    // Rows index ceil-division scale rows; columns stay whole 128-blocks.
    return weight && scale && input && output && rows && cols &&
        cols % (32 * GLMVQ_FP8_VECTOR) == 0 &&
        cols <= GLMVQ_FP8_MAX_COLS && (uintptr_t)weight % sizeof(uint2) == 0 &&
        (uintptr_t)scale % sizeof(float) == 0 && (uintptr_t)input % sizeof(float4) == 0 &&
        (uintptr_t)output % sizeof(float) == 0;
}

static cudaError_t glmvq_fp8_decode(const void *weight, const float *scale,
        const float *input, float *output, uint32_t rows, uint32_t cols,
        uint32_t split, glmvq_fp8_feed feed, cudaStream_t stream, uint32_t tokens = 1) {
    if (!glmvq_fp8_fits(weight, scale, input, output, rows, cols) ||
        !split || split > GLMVQ_FP8_WARPS || (split & (split - 1)) ||
        feed > GLMVQ_FP8_SHARED || !tokens || tokens > GLMVQ_FP8_TOKENS ||
        (tokens > 1 && feed != GLMVQ_FP8_GLOBAL)) { return cudaErrorInvalidValue; }
    const uint32_t grid = (rows + GLMVQ_FP8_WARPS / split - 1) / (GLMVQ_FP8_WARPS / split);
    if (feed == GLMVQ_FP8_SHARED) {
        glmvq_fp8_dot<GLMVQ_FP8_SHARED, 1><<<grid, GLMVQ_THREADS,
            cols * sizeof(__nv_bfloat16), stream>>>((const uint8_t *)weight, scale, input, output, rows, cols, split, tokens);
    } else if (tokens == 1) {
        glmvq_fp8_dot<GLMVQ_FP8_GLOBAL, 1><<<grid, GLMVQ_THREADS, 0, stream>>>(
            (const uint8_t *)weight, scale, input, output, rows, cols, split, tokens);
    } else {
        glmvq_fp8_dot<GLMVQ_FP8_GLOBAL, GLMVQ_FP8_TOKENS><<<grid, GLMVQ_THREADS, 0, stream>>>(
            (const uint8_t *)weight, scale, input, output, rows, cols, split, tokens);
    }
    return cudaGetLastError();
}

#endif
