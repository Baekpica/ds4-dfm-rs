/* Adapted from antirez/ds4 0aaea5a, MIT. Dense prefill only: FP16 query
 * and probability, FP16 compact KV, FP32 dot/value accumulation. */
#ifndef DS4_GLM53_DENSE_ATTN_CUH
#define DS4_GLM53_DENSE_ATTN_CUH
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <float.h>
#include <math.h>
#include "../ds4_glm53_attn.h"
__global__ static void glm53_dense_pack_q_f16_kernel(
        __half *packed,
        const float *q,
        uint32_t n_q,
        uint32_t n_head,
        uint32_t head0,
        uint32_t group_heads,
        uint32_t dim) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t count = (uint64_t)group_heads * n_q * dim;
    if (i >= count) { return; }
    const uint32_t d = (uint32_t)(i % dim);
    const uint64_t row = i / dim;
    const uint32_t token = (uint32_t)(row % n_q);
    const uint32_t head = head0 + (uint32_t)(row / n_q);
    packed[i] = __float2half(q[((uint64_t)token * n_head + head) * dim + d]);
}

__global__ static void glm53_dense_causal_softmax_f16_kernel(
        __half *prob,
        const float *scores,
        uint32_t n_q,
        uint32_t n_kv,
        uint32_t q_row0,
        float scale) {
    const uint32_t token = blockIdx.x;
    const uint32_t group_head = blockIdx.y;
    if (token >= n_q) { return; }
    const uint32_t visible = min(n_kv, q_row0 + token + 1u);
    const uint64_t row = ((uint64_t)group_head * n_q + token) * n_kv;

    float local_max = -FLT_MAX;
    for (uint32_t col = threadIdx.x; col < visible; col += blockDim.x) {
        local_max = fmaxf(local_max, scores[row + col] * scale);
    }
    __shared__ float reduce[256];
    reduce[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x],
                                         reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t col = threadIdx.x; col < visible; col += blockDim.x) {
        local_sum += expf(scores[row + col] * scale - max_score);
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_sum = reduce[0] > 0.0f ? 1.0f / reduce[0] : 0.0f;
    for (uint32_t col = threadIdx.x; col < n_kv; col += blockDim.x) {
        const float p = col < visible
            ? expf(scores[row + col] * scale - max_score) * inv_sum : 0.0f;
        prob[row + col] = __float2half(p);
    }
}

__global__ static void glm53_dense_unpack_f32_kernel(
        float *out,
        const float *packed,
        uint32_t n_q,
        uint32_t n_head,
        uint32_t head0,
        uint32_t group_heads,
        uint32_t dim) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t count = (uint64_t)group_heads * n_q * dim;
    if (i >= count) { return; }
    const uint32_t d = (uint32_t)(i % dim);
    const uint64_t row = i / dim;
    const uint32_t token = (uint32_t)(row % n_q);
    const uint32_t head = head0 + (uint32_t)(row / n_q);
    out[((uint64_t)token * n_head + head) * dim + d] = packed[i];
}


static int glm53_dense_run(cublasHandle_t handle, cudaStream_t stream,
        float *out, const float *q, const __half *kv, void *scratch,
        uint32_t rows, uint32_t pos0, uint32_t heads, uint32_t dim,
        uint32_t head_dim) {
    if (pos0 > DS4_GLM53_MAX_SELECTED || rows > DS4_GLM53_MAX_SELECTED - pos0 ||
        !glm53_dense_shape(rows, pos0 + rows, heads, dim, GLM53_ATTN_ALL) ||
        !handle || !out || !q || !kv || !scratch || !head_dim) { return 0; }
    const uint32_t keys = pos0 + rows;
    const uint64_t q_count = (uint64_t)GLM53_DENSE_GROUP * rows * dim;
    const uint64_t score_count = (uint64_t)GLM53_DENSE_GROUP * rows * keys;
    const uint64_t score_off = glm53_dense_align(q_count * sizeof(__half));
    const uint64_t prob_off = glm53_dense_align(score_off + score_count * sizeof(float));
    const uint64_t out_off = glm53_dense_align(prob_off + score_count * sizeof(__half));
    auto *base = (unsigned char *)scratch;
    auto *qh = (__half *)base;
    auto *scores = (float *)(base + score_off);
    auto *prob = (__half *)(base + prob_off);
    auto *packed = (float *)(base + out_off);
    const float alpha = 1.0f, beta = 0.0f;
    cudaStream_t current;
    if (cublasGetStream(handle, &current) != CUBLAS_STATUS_SUCCESS) { return 0; }
    // Setting an unchanged stream would discard the engine's cuBLAS slab.
    if (current != stream && cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) { return 0; }
    for (uint32_t h = 0; h < heads; h += GLM53_DENSE_GROUP) {
        glm53_dense_pack_q_f16_kernel<<<(q_count + 255u) / 256u, 256, 0, stream>>>(
            qh, q, rows, heads, h, GLM53_DENSE_GROUP, dim);
        if (cudaGetLastError() != cudaSuccess) { return 0; }
        const cublasStatus_t score_status = cublasGemmStridedBatchedEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N, keys, rows, dim, &alpha,
            kv, CUDA_R_16F, dim, 0, qh, CUDA_R_16F, dim, (long long)rows * dim,
            &beta, scores, CUDA_R_32F, keys, (long long)rows * keys,
            GLM53_DENSE_GROUP, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (score_status != CUBLAS_STATUS_SUCCESS) { return 0; }
        glm53_dense_causal_softmax_f16_kernel<<<dim3(rows, GLM53_DENSE_GROUP), 256, 0, stream>>>(
            prob, scores, rows, keys, pos0, 1.0f / sqrtf((float)head_dim));
        if (cudaGetLastError() != cudaSuccess) { return 0; }
        const cublasStatus_t value_status = cublasGemmStridedBatchedEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N, dim, rows, keys, &alpha,
            kv, CUDA_R_16F, dim, 0, prob, CUDA_R_16F, keys, (long long)rows * keys,
            &beta, packed, CUDA_R_32F, dim, (long long)rows * dim,
            GLM53_DENSE_GROUP, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (value_status != CUBLAS_STATUS_SUCCESS) { return 0; }
        glm53_dense_unpack_f32_kernel<<<(q_count + 255u) / 256u, 256, 0, stream>>>(
            out, packed, rows, heads, h, GLM53_DENSE_GROUP, dim);
        if (cudaGetLastError() != cudaSuccess) { return 0; }
    }
    return 1;
}
#endif
