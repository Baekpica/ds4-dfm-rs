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

#endif
