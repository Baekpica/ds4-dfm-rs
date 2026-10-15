#ifndef DS4_GLM_VQ_FP8_CUH
#define DS4_GLM_VQ_FP8_CUH

#include "glm_vq_ops.cuh"

enum glmvq_fp8_feed { GLMVQ_FP8_GLOBAL, GLMVQ_FP8_SHARED };
enum { GLMVQ_FP8_WARPS = 8, GLMVQ_FP8_VECTOR = 8, GLMVQ_FP8_MAX_COLS = 16384 };
enum { GLMVQ_FP8_HIDDEN = 4096, GLMVQ_FP8_SHARED_FFN = 2048, GLMVQ_FP8_DENSE_FFN = 12288,
       GLMVQ_FP8_Q_RANK = 1536, GLMVQ_FP8_KV_RANK = 512, GLMVQ_FP8_Q_WIDTH = 16384 };

static uint32_t glmvq_fp8_split(uint32_t rows, uint32_t cols) {
    // Only measured original-artifact shapes use this N1 specialization.
    if (rows == GLMVQ_FP8_Q_WIDTH && cols == GLMVQ_FP8_Q_RANK) { return 2; }
    if (cols == GLMVQ_FP8_HIDDEN && (rows == GLMVQ_FP8_KV_RANK || rows == GLMVQ_FP8_Q_RANK ||
        rows == GLMVQ_FP8_SHARED_FFN || rows == GLMVQ_FP8_DENSE_FFN)) { return GLMVQ_FP8_WARPS; }
    if (rows == GLMVQ_FP8_HIDDEN && (cols == GLMVQ_FP8_SHARED_FFN || cols == GLMVQ_FP8_DENSE_FFN ||
        cols == GLMVQ_FP8_Q_WIDTH)) { return GLMVQ_FP8_WARPS; }
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
template <glmvq_fp8_feed FEED>
__global__ void glmvq_fp8_dot(const uint8_t *weight, const float *scales,
        const float *input, float *output, uint32_t rows, uint32_t cols, uint32_t split) {
    extern __shared__ __align__(16) uint8_t fp8_input[];
    __shared__ float partial[GLMVQ_FP8_WARPS];
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
    float sum = 0;
    if (row < rows) {
        const uint8_t *w = weight + (uint64_t)row * cols;
        const float *s = scales + (uint64_t)(row / GLMVQ_SCALE_BLOCK) * (cols / GLMVQ_SCALE_BLOCK);
        for (uint32_t col = part * 32 * GLMVQ_FP8_VECTOR + lane * GLMVQ_FP8_VECTOR;
                col < cols; col += split * 32 * GLMVQ_FP8_VECTOR) {
            const uint2 words = *(const uint2 *)(w + col);
            const float scale = s[col / GLMVQ_SCALE_BLOCK];
            uint4 x;
            if constexpr (FEED == GLMVQ_FP8_SHARED) {
                x = *(const uint4 *)(staged + col);
            } else {
                const float4 a = *(const float4 *)(input + col), b = *(const float4 *)(input + col + 4);
                x = make_uint4(glmvq_fp8_pack(a.x, a.y), glmvq_fp8_pack(a.z, a.w),
                    glmvq_fp8_pack(b.x, b.y), glmvq_fp8_pack(b.z, b.w));
            }
            sum = glmvq_fp8_pair(words.x, x.x, scale, sum);
            sum = glmvq_fp8_pair(words.x >> 16, x.y, scale, sum);
            sum = glmvq_fp8_pair(words.y, x.z, scale, sum);
            sum = glmvq_fp8_pair(words.y >> 16, x.w, scale, sum);
        }
    }
    for (uint32_t stride = 16; stride; stride >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, stride); }
    if (!lane) { partial[warp] = sum; }
    __syncthreads();
    if (row >= rows || part || lane) { return; }
    sum = 0;
    for (uint32_t i = 0; i < split; i++) { sum += partial[local * split + i]; }
    output[row] = glmvq_round(sum);
}

static bool glmvq_fp8_fits(const void *weight, const void *scale,
        const void *input, const void *output, uint32_t rows, uint32_t cols) {
    return weight && scale && input && output && rows && cols &&
        rows % GLMVQ_SCALE_BLOCK == 0 && cols % (32 * GLMVQ_FP8_VECTOR) == 0 &&
        cols <= GLMVQ_FP8_MAX_COLS && (uintptr_t)weight % sizeof(uint2) == 0 &&
        (uintptr_t)scale % sizeof(float) == 0 && (uintptr_t)input % sizeof(float4) == 0 &&
        (uintptr_t)output % sizeof(float) == 0;
}

static cudaError_t glmvq_fp8_decode(const void *weight, const float *scale,
        const float *input, float *output, uint32_t rows, uint32_t cols,
        uint32_t split, glmvq_fp8_feed feed, cudaStream_t stream) {
    if (!glmvq_fp8_fits(weight, scale, input, output, rows, cols) ||
        !split || split > GLMVQ_FP8_WARPS || (split & (split - 1)) ||
        feed > GLMVQ_FP8_SHARED) { return cudaErrorInvalidValue; }
    const uint32_t grid = (rows + GLMVQ_FP8_WARPS / split - 1) / (GLMVQ_FP8_WARPS / split);
    if (feed == GLMVQ_FP8_SHARED) {
        glmvq_fp8_dot<GLMVQ_FP8_SHARED><<<grid, GLMVQ_THREADS,
            cols * sizeof(__nv_bfloat16), stream>>>((const uint8_t *)weight, scale, input, output, rows, cols, split);
    } else {
        glmvq_fp8_dot<GLMVQ_FP8_GLOBAL><<<grid, GLMVQ_THREADS, 0, stream>>>(
            (const uint8_t *)weight, scale, input, output, rows, cols, split);
    }
    return cudaGetLastError();
}

#endif
