#ifndef DS4_GLM_VQ_INDEX_CUH
#define DS4_GLM_VQ_INDEX_CUH

#include "glm_vq_linear.cuh"
#include "../ds4_glm53_compact.h"
#include <float.h>

enum { GLMVQ_INDEX_HEADS = 32 };
static constexpr float GLMVQ_INDEX_EPS = 1.0e-6f;

/* Source modeling_glm5_next.py:835-837,998-1004: LayerNorm, probabilities,
 * products and pooled keys each cross a BF16 boundary. The MXQ pool stores
 * FP16 and omits these intermediate roundings, so it is not this contract. */
__global__ void glmvq_pool_key(__nv_bfloat16 *cache, float *tail_k,
        float *tail_gate, const float *raw, const float *gate,
        const __nv_bfloat16 *norm, const __nv_bfloat16 *bias,
        const __nv_bfloat16 *ape, uint32_t first, uint32_t rows) {
    constexpr uint32_t width = DS4_GLM53_POOL_DIM, ratio = DS4_GLM53_POOL_SIZE;
    const uint32_t d = threadIdx.x, pool = first / ratio + blockIdx.x;
    const uint32_t start = pool * ratio, end = first + rows;
    const bool complete = start + ratio <= end;
    __shared__ float keys[ratio * width], mean[ratio], inv[ratio];
    float logits[ratio], maximum = -INFINITY;
    for (uint32_t r = 0; r < ratio; r++) {
        const uint32_t pos = start + r;
        float key = 0.0f, score = 0.0f;
        if (pos >= first && pos < end) {
            key = raw[(uint64_t)(pos - first) * width + d];
            score = gate[(uint64_t)(pos - first) * width + d];
            if (!complete) {
                tail_k[r * width + d] = key;
                tail_gate[r * width + d] = score;
            }
        } else if (pos < first) {
            key = tail_k[r * width + d];
            score = tail_gate[r * width + d];
        }
        keys[r * width + d] = key;
        logits[r] = glmvq_add(score, __bfloat162float(ape[r * width + d]));
        maximum = fmaxf(maximum, logits[r]);
    }
    __syncthreads();
    if (!complete) { return; }
    if (d < ratio) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < width; j++) { sum += keys[d * width + j]; }
        mean[d] = sum / width;
        float variance = 0.0f;
        for (uint32_t j = 0; j < width; j++) {
            const float delta = keys[d * width + j] - mean[d];
            variance = fmaf(delta, delta, variance);
        }
        inv[d] = rsqrtf(variance / width + GLMVQ_INDEX_EPS);
    }
    __syncthreads();
    float total = 0.0f;
    for (uint32_t r = 0; r < ratio; r++) {
        logits[r] = expf(glmvq_add(logits[r], -maximum));
        total = glmvq_add(total, logits[r]);
    }
    float value = 0.0f;
    for (uint32_t r = 0; r < ratio; r++) {
        const float normalized = glmvq_round(fmaf(
            (keys[r * width + d] - mean[r]) * inv[r],
            __bfloat162float(norm[d]), __bfloat162float(bias[d])));
        const float probability = glmvq_round(__fdiv_rn(logits[r], total));
        value = glmvq_add(value, glmvq_round(glmvq_mul(probability, normalized)));
    }
    cache[(uint64_t)pool * width + d] = __float2bfloat16_rn(value);
}

static cudaError_t glmvq_pool_update(__nv_bfloat16 *cache, float *tail_k,
        float *tail_gate, const float *raw, const float *gate,
        const __nv_bfloat16 *norm, const __nv_bfloat16 *bias,
        const __nv_bfloat16 *ape, uint32_t first, uint32_t rows, cudaStream_t stream) {
    if (!cache || !tail_k || !tail_gate || !raw || !gate || !norm || !bias || !ape ||
        !rows || first > UINT32_MAX - rows) { return cudaErrorInvalidValue; }
    constexpr uint32_t width = DS4_GLM53_POOL_DIM, ratio = DS4_GLM53_POOL_SIZE;
    // A leading carry may read the same tail a later partial pool writes.
    // Stream-ordered launches give that four-row state a single writer.
    uint32_t done = 0;
    if (first % ratio) {
        done = min(rows, ratio - first % ratio);
        glmvq_pool_key<<<1, width, 0, stream>>>(cache, tail_k, tail_gate,
            raw, gate, norm, bias, ape, first, done);
        if (cudaGetLastError() != cudaSuccess) { return cudaErrorLaunchFailure; }
    }
    const uint32_t full = (rows - done) / ratio * ratio;
    if (full) {
        glmvq_pool_key<<<full / ratio, width, 0, stream>>>(cache, tail_k, tail_gate,
            raw + (uint64_t)done * width, gate + (uint64_t)done * width,
            norm, bias, ape, first + done, full);
        if (cudaGetLastError() != cudaSuccess) { return cudaErrorLaunchFailure; }
        done += full;
    }
    if (done < rows) {
        glmvq_pool_key<<<1, width, 0, stream>>>(cache, tail_k, tail_gate,
            raw + (uint64_t)done * width, gate + (uint64_t)done * width,
            norm, bias, ape, first + done, rows - done);
    }
    return cudaGetLastError();
}

static uint64_t glmvq_index_bytes(uint32_t rows, uint32_t pools) {
    const uint64_t queries = (uint64_t)rows * GLMVQ_INDEX_HEADS * DS4_GLM53_POOL_DIM;
    return queries * sizeof(__nv_bfloat16) +
        (uint64_t)rows * GLMVQ_INDEX_HEADS * pools * sizeof(float);
}

/* Source :862-881 scales the per-head dot before ReLU, then combines heads
 * with separately scaled BF16 projection outputs. Invisible pools remain
 * unselectable; the common integer expansion appends the incomplete tail. */
__global__ void glmvq_index_reduce(float *out, const float *dots,
        const float *weights, uint32_t pools, uint32_t rows, uint32_t first) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * pools) { return; }
    const uint32_t row = i / pools, pool = i % pools;
    if (pool >= (first + row + 1u) / DS4_GLM53_POOL_SIZE) {
        out[i] = -FLT_MAX;
        return;
    }
    const float q_scale = 1.0f / sqrtf((float)DS4_GLM53_POOL_DIM);
    const float w_scale = 1.0f / sqrtf((float)GLMVQ_INDEX_HEADS);
    float sum = 0.0f;
    for (uint32_t h = 0; h < GLMVQ_INDEX_HEADS; h++) {
        const uint64_t head = (uint64_t)row * GLMVQ_INDEX_HEADS + h;
        const float dot = fmaxf(glmvq_mul(dots[head * pools + pool], q_scale), 0.0f);
        sum = fmaf(dot, glmvq_mul(weights[head], w_scale), sum);
    }
    out[i] = sum;
}

static cublasStatus_t glmvq_index_score(cublasHandle_t handle, float *out,
        const float *query, const float *weights, const __nv_bfloat16 *pool,
        uint32_t pools, uint32_t rows, uint32_t first,
        void *scratch, uint64_t bytes, cudaStream_t stream) {
    if (!handle || !out || !query || !weights || !pool || !scratch || !pools || !rows ||
        pools > INT_MAX || rows > INT_MAX / GLMVQ_INDEX_HEADS || first > UINT32_MAX - rows ||
        bytes < glmvq_index_bytes(rows, pools)) { return CUBLAS_STATUS_INVALID_VALUE; }
    const uint64_t nq = (uint64_t)rows * GLMVQ_INDEX_HEADS * DS4_GLM53_POOL_DIM;
    auto *q = (__nv_bfloat16 *)scratch;
    auto *dots = (float *)(q + nq);
    glmvq_convert<<<(nq + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, stream>>>(query, q, nq);
    if (cudaGetLastError() != cudaSuccess) { return CUBLAS_STATUS_EXECUTION_FAILED; }
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        pools, rows * GLMVQ_INDEX_HEADS, DS4_GLM53_POOL_DIM,
        &alpha, pool, CUDA_R_16BF, DS4_GLM53_POOL_DIM,
        q, CUDA_R_16BF, DS4_GLM53_POOL_DIM, &beta, dots, CUDA_R_32F, pools,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) { return status; }
    const uint64_t count = (uint64_t)rows * pools;
    glmvq_index_reduce<<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, stream>>>(
        out, dots, weights, pools, rows, first);
    return cudaGetLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

/* Prefill index scores without the [rows][heads][pools] F32 dot buffer that
 * the cuBLAS path writes and rereads (4.3 GB per DSA layer for a 2K chunk at
 * 64K context). A CTA scores 4 query rows x 128 pools: BF16 m16n8k16 MMAs
 * over the 128-dim keys, then each warp folds its 32 heads in registers.
 * F32 summation order across heads differs from glmvq_index_reduce. */
enum { GLMVQ_IDX_ROWS = 4, GLMVQ_IDX_POOLS = 128, GLMVQ_IDX_PITCH = DS4_GLM53_POOL_DIM + 8,
       GLMVQ_IDX_THREADS = 256 };
static constexpr uint32_t GLMVQ_IDX_SMEM = (GLMVQ_IDX_ROWS * GLMVQ_INDEX_HEADS + GLMVQ_IDX_POOLS) *
    GLMVQ_IDX_PITCH * sizeof(__nv_bfloat16) + GLMVQ_IDX_ROWS * GLMVQ_INDEX_HEADS * sizeof(float);

__device__ __forceinline__ uint32_t glmvq_idx_pack(float lo, float hi) {
    const __nv_bfloat162_raw v = __floats2bfloat162_rn(lo, hi);
    return (uint32_t)v.x | ((uint32_t)v.y << 16);
}

__device__ __forceinline__ void glmvq_idx_mma(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(GLMVQ_IDX_THREADS) glmvq_index_fused(float *out,
        const float *query, const float *weights, const __nv_bfloat16 *pool,
        uint32_t pools, uint32_t rows, uint32_t first) {
    extern __shared__ __align__(16) uint8_t idx_shared[];
    // Rows padded by 8 BF16 (68 words): fragment loads hit distinct banks.
    auto *qs = (__nv_bfloat16 *)idx_shared;
    auto *ks = qs + GLMVQ_IDX_ROWS * GLMVQ_INDEX_HEADS * GLMVQ_IDX_PITCH;
    auto *ws = (float *)(ks + GLMVQ_IDX_POOLS * GLMVQ_IDX_PITCH);
    constexpr uint32_t D = DS4_GLM53_POOL_DIM;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t row0 = blockIdx.y * GLMVQ_IDX_ROWS, pool0 = blockIdx.x * GLMVQ_IDX_POOLS;
    for (uint32_t i = tid; i < GLMVQ_IDX_ROWS * GLMVQ_INDEX_HEADS * D / 4; i += blockDim.x) {
        const uint32_t m = i / (D / 4), c = i % (D / 4) * 4, r = row0 + m / GLMVQ_INDEX_HEADS;
        const float4 v = r < rows ? *(const float4 *)(query + ((uint64_t)row0 * GLMVQ_INDEX_HEADS + m) * D + c)
                                  : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        *(uint2 *)(qs + m * GLMVQ_IDX_PITCH + c) = make_uint2(glmvq_idx_pack(v.x, v.y), glmvq_idx_pack(v.z, v.w));
    }
    for (uint32_t i = tid; i < GLMVQ_IDX_POOLS * D / 8; i += blockDim.x) {
        const uint32_t n = i / (D / 8), c = i % (D / 8) * 8;
        const uint4 v = pool0 + n < pools ? *(const uint4 *)(pool + (uint64_t)(pool0 + n) * D + c) : make_uint4(0, 0, 0, 0);
        *(uint4 *)(ks + n * GLMVQ_IDX_PITCH + c) = v;
    }
    const float w_scale = 1.0f / sqrtf((float)GLMVQ_INDEX_HEADS);
    if (tid < GLMVQ_IDX_ROWS * GLMVQ_INDEX_HEADS) {
        const uint32_t r = row0 + tid / GLMVQ_INDEX_HEADS;
        ws[tid] = r < rows ? glmvq_mul(weights[(uint64_t)row0 * GLMVQ_INDEX_HEADS + tid], w_scale) : 0.0f;
    }
    __syncthreads();

    // Warp w: query row w/2 (both head m16 tiles), pools (w%2)*64 .. +64.
    const uint32_t g = lane >> 2, t = lane & 3u, local = warp >> 1, nbase = (warp & 1u) * 64;
    float c[2][8][4] = {};
    #pragma unroll
    for (uint32_t k = 0; k < D; k += 16) {
        uint32_t a[2][4];
        #pragma unroll
        for (uint32_t m = 0; m < 2; m++) {
            const __nv_bfloat16 *base = qs + (local * GLMVQ_INDEX_HEADS + m * 16 + g) * GLMVQ_IDX_PITCH + k + 2 * t;
            a[m][0] = *(const uint32_t *)base;
            a[m][1] = *(const uint32_t *)(base + 8 * GLMVQ_IDX_PITCH);
            a[m][2] = *(const uint32_t *)(base + 8);
            a[m][3] = *(const uint32_t *)(base + 8 * GLMVQ_IDX_PITCH + 8);
        }
        #pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            const __nv_bfloat16 *base = ks + (nbase + j * 8 + g) * GLMVQ_IDX_PITCH + k + 2 * t;
            const uint32_t b0 = *(const uint32_t *)base, b1 = *(const uint32_t *)(base + 8);
            glmvq_idx_mma(c[0][j], a[0], b0, b1);
            glmvq_idx_mma(c[1][j], a[1], b0, b1);
        }
    }
    const float q_scale = 1.0f / sqrtf((float)D);
    const uint32_t row = row0 + local;
    const float *w = ws + local * GLMVQ_INDEX_HEADS;
    const uint32_t visible = (first + row + 1u) / DS4_GLM53_POOL_SIZE;
    #pragma unroll
    for (uint32_t j = 0; j < 8; j++) {
        float sum[2];
        #pragma unroll
        for (uint32_t e = 0; e < 2; e++) {
            // Heads g, g+8, 16+g, 24+g of this lane, then the other 28 by shuffle.
            sum[e] = fmaxf(glmvq_mul(c[0][j][e], q_scale), 0.0f) * w[g] +
                fmaxf(glmvq_mul(c[0][j][2 + e], q_scale), 0.0f) * w[g + 8] +
                fmaxf(glmvq_mul(c[1][j][e], q_scale), 0.0f) * w[16 + g] +
                fmaxf(glmvq_mul(c[1][j][2 + e], q_scale), 0.0f) * w[24 + g];
            for (uint32_t stride = 4; stride < 32; stride <<= 1) { sum[e] += __shfl_xor_sync(0xffffffffu, sum[e], stride); }
        }
        if (g || row >= rows) { continue; }
        #pragma unroll
        for (uint32_t e = 0; e < 2; e++) {
            const uint32_t p = pool0 + nbase + j * 8 + 2 * t + e;
            if (p < pools) { out[(uint64_t)row * pools + p] = p < visible ? sum[e] : -FLT_MAX; }
        }
    }
}

#endif
