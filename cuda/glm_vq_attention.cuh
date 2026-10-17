#ifndef DS4_GLM_VQ_ATTENTION_CUH
#define DS4_GLM_VQ_ATTENTION_CUH

#include "glm_vq_linear.cuh"
#include <type_traits>

struct glmvq_attn_layout { uint64_t q, kv, scores, output, mask, bytes; };

static glmvq_attn_layout glmvq_attn_bytes(uint32_t rows, uint32_t keys,
        uint32_t heads, uint32_t dim) {
    enum { ALIGN = 128 };
    uint64_t cursor = 0;
    auto take = [&](uint64_t bytes) {
        const uint64_t offset = cursor;
        cursor += (bytes + ALIGN - 1) / ALIGN * ALIGN;
        return offset;
    };
    glmvq_attn_layout layout;
    layout.q = take((uint64_t)rows * heads * dim * sizeof(__nv_bfloat16));
    layout.kv = take((uint64_t)keys * heads * 2 * dim * sizeof(__nv_bfloat16));
    layout.scores = take((uint64_t)heads * rows * keys * sizeof(__nv_bfloat16));
    layout.output = take((uint64_t)rows * heads * dim * sizeof(__nv_bfloat16));
    layout.mask = take((uint64_t)rows * keys * sizeof(uint32_t));
    layout.bytes = cursor;
    return layout;
}

__global__ void glmvq_attn_mask(uint32_t *mask, const int32_t *selected,
        uint32_t keys, uint32_t stride) {
    const uint64_t row = blockIdx.x;
    for (uint32_t k = threadIdx.x; k < keys; k += blockDim.x) { mask[row * keys + k] = 0; }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < stride; i += blockDim.x) {
        const int32_t key = selected[row * stride + i];
        if (key >= 0 && (uint32_t)key < keys) { atomicExch(mask + row * keys + key, 1u); }
    }
}

/* Source eager_attention_forward:1076-1099. QK and scaling round to BF16
 * separately; softmax is FP32 and its probabilities round before PV.
 * Positions are live device values, even when GEMM capacity is captured. */
__global__ void glmvq_attn_softmax(__nv_bfloat16 *scores, const uint32_t *mask,
        const uint32_t *positions, uint32_t rows, uint32_t keys, float scale) {
    const uint32_t row = blockIdx.x, head = blockIdx.y;
    __nv_bfloat16 *values = scores + ((uint64_t)head * rows + row) * keys;
    __shared__ float partial[GLMVQ_THREADS];
    float maximum = -INFINITY;
    for (uint32_t k = threadIdx.x; k < keys; k += blockDim.x) {
        const float value = k <= positions[row] && (!mask || mask[(uint64_t)row * keys + k])
            ? glmvq_round(glmvq_mul(__bfloat162float(values[k]), scale)) : -INFINITY;
        values[k] = __float2bfloat16_rn(value);
        maximum = fmaxf(maximum, value);
    }
    partial[threadIdx.x] = maximum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) { partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]); }
        __syncthreads();
    }
    maximum = partial[0];
    float sum = 0.0f;
    for (uint32_t k = threadIdx.x; k < keys; k += blockDim.x) {
        const float value = __bfloat162float(values[k]);
        if (isfinite(maximum)) { sum = glmvq_add(sum, expf(glmvq_add(value, -maximum))); }
    }
    __syncthreads();
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) { partial[threadIdx.x] = glmvq_add(partial[threadIdx.x], partial[threadIdx.x + stride]); }
        __syncthreads();
    }
    const float total = partial[0];
    for (uint32_t k = threadIdx.x; k < keys; k += blockDim.x) {
        const float p = total > 0.0f ? __fdiv_rn(expf(glmvq_add(__bfloat162float(values[k]), -maximum)), total) : 0.0f;
        values[k] = __float2bfloat16_rn(p);
    }
}

__global__ void glmvq_attn_output(float *output, const __nv_bfloat16 *input,
        uint32_t rows, uint32_t heads, uint32_t dim, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const uint32_t col = i % dim, head = (i / dim) % heads;
    const uint64_t row = i / ((uint64_t)heads * dim);
    output[i] = __bfloat162float(input[((uint64_t)head * rows + row) * dim + col]);
}

template<typename KV>
static cublasStatus_t glmvq_attention(cublasHandle_t handle, const float *query,
        const KV *kv, float *output, const uint32_t *positions,
        const int32_t *selected, uint32_t stride, uint32_t rows, uint32_t keys,
        uint32_t heads, uint32_t dim, void *scratch, uint64_t bytes, cudaStream_t stream) {
    if (!handle || !query || !kv || !output || !positions || !scratch ||
        !rows || !keys || !heads || !dim || (selected && !stride) ||
        rows > INT_MAX || keys > INT_MAX || (uint64_t)heads * 2 * dim > INT_MAX) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const glmvq_attn_layout layout = glmvq_attn_bytes(rows, keys, heads, dim);
    if (bytes < layout.bytes) { return CUBLAS_STATUS_INVALID_VALUE; }
    uint8_t *base = (uint8_t *)scratch;
    static_assert(std::is_same<KV, float>::value || std::is_same<KV, __nv_bfloat16>::value);
    __nv_bfloat16 *q = (__nv_bfloat16 *)(base + layout.q);
    const __nv_bfloat16 *k;
    __nv_bfloat16 *scores = (__nv_bfloat16 *)(base + layout.scores), *y = (__nv_bfloat16 *)(base + layout.output);
    uint32_t *mask = selected ? (uint32_t *)(base + layout.mask) : nullptr;
    const uint64_t nq = (uint64_t)rows * heads * dim, nkv = (uint64_t)keys * heads * 2 * dim;
    glmvq_convert<<<(nq + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, stream>>>(query, q, nq);
    if constexpr (std::is_same<KV, float>::value) {
        auto *packed = (__nv_bfloat16 *)(base + layout.kv);
        glmvq_convert<<<(nkv + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, stream>>>(kv, packed, nkv);
        k = packed;
    } else {
        // KV_B already rounds to BF16; consuming it directly preserves that boundary.
        k = kv;
    }
    if (selected) { glmvq_attn_mask<<<rows, GLMVQ_THREADS, 0, stream>>>(mask, selected, keys, stride); }
    if (cudaGetLastError() != cudaSuccess) { return CUBLAS_STATUS_EXECUTION_FAILED; }
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        keys, rows, dim, &alpha, k, CUDA_R_16BF, heads * 2 * dim, 2 * dim,
        q, CUDA_R_16BF, heads * dim, dim, &beta, scores, CUDA_R_16BF, keys,
        (uint64_t)rows * keys, heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) { return status; }
    glmvq_attn_softmax<<<dim3(rows, heads), GLMVQ_THREADS, 0, stream>>>(
        scores, mask, positions, rows, keys, 1.0f / sqrtf((float)dim));
    if (cudaGetLastError() != cudaSuccess) { return CUBLAS_STATUS_EXECUTION_FAILED; }
    status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        dim, rows, keys, &alpha, k + dim, CUDA_R_16BF, heads * 2 * dim, 2 * dim,
        scores, CUDA_R_16BF, keys, (uint64_t)rows * keys, &beta,
        y, CUDA_R_16BF, dim, (uint64_t)rows * dim, heads,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) { return status; }
    glmvq_attn_output<<<(nq + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, stream>>>(output, y, rows, heads, dim, nq);
    return cudaGetLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

/* Source modeling_glm5_next.py:1076-1199 expands every cached latent row.
 * Factor no-RoPE attention as (Q Wk) C^T, then (P C) Wv^T for N=1.
 * BF16 rounding moves from expanded K/V to Q-low and output-low; this is
 * approximate arithmetic, qualified separately from the expanded path. */
static cublasStatus_t glmvq_latent_attn(cublasHandle_t handle,
        const __nv_bfloat16 *weight, const __nv_bfloat16 *cache,
        const float *query, float *output, const uint32_t *positions,
        uint32_t rows, uint32_t keys, uint32_t heads, uint32_t dim,
        uint32_t latent, void *scratch, uint64_t bytes, cudaStream_t stream,
        const int32_t *selected = nullptr, uint32_t stride = 0) {
    if (!handle || !weight || !cache || !query || !output || !positions || !scratch ||
        rows != 1 || !keys || !heads || !dim || !latent || (selected && !stride) ||
        (uint64_t)rows * heads > INT_MAX || keys > INT_MAX ||
        (uint64_t)heads * dim > INT_MAX || latent > INT_MAX) { return CUBLAS_STATUS_INVALID_VALUE; }
    const glmvq_attn_layout l = glmvq_attn_bytes(rows, keys, heads, dim);
    const uint64_t nq = (uint64_t)rows * heads * dim;
    const uint64_t nl = (uint64_t)rows * heads * latent;
    if (bytes < l.bytes || nl * sizeof(__nv_bfloat16) * 2 > l.scores - l.kv) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    auto *base = (uint8_t *)scratch;
    auto *q = (__nv_bfloat16 *)(base + l.q);
    auto *qlow = (__nv_bfloat16 *)(base + l.kv), *low = qlow + nl;
    auto *scores = (__nv_bfloat16 *)(base + l.scores);
    auto *y = (__nv_bfloat16 *)(base + l.output);
    auto *mask = selected ? (uint32_t *)(base + l.mask) : nullptr;
    const float alpha = 1.0f, beta = 0.0f;
    const long long wstride = (long long)2 * dim * latent;
    glmvq_convert<<<(nq + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, stream>>>(query, q, nq);
    if (cudaGetLastError() != cudaSuccess) { return CUBLAS_STATUS_EXECUTION_FAILED; }
    cublasStatus_t status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        latent, rows, dim, &alpha, weight, CUDA_R_16BF, latent, wstride,
        q, CUDA_R_16BF, heads * dim, dim, &beta, qlow, CUDA_R_16BF,
        latent, (uint64_t)rows * latent, heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) { return status; }
    // All heads share C. Fold heads into N instead of replicating C or using zero strides.
    status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        keys, rows * heads, latent, &alpha, cache, CUDA_R_16BF, latent,
        qlow, CUDA_R_16BF, latent, &beta, scores, CUDA_R_16BF, keys,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) { return status; }
    if (selected) { glmvq_attn_mask<<<rows, GLMVQ_THREADS, 0, stream>>>(mask, selected, keys, stride); }
    glmvq_attn_softmax<<<dim3(rows, heads), GLMVQ_THREADS, 0, stream>>>(
        scores, mask, positions, rows, keys, 1.0f / sqrtf((float)dim));
    if (cudaGetLastError() != cudaSuccess) { return CUBLAS_STATUS_EXECUTION_FAILED; }
    // cuBLAS may reduce split-K partials in the output type. Keep only this
    // long PV reduction in FP32 without changing the caller's math policy.
    cublasMath_t math;
    status = cublasGetMathMode(handle, &math);
    if (status != CUBLAS_STATUS_SUCCESS) { return status; }
    status = cublasSetMathMode(handle,
        (cublasMath_t)(math | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    if (status != CUBLAS_STATUS_SUCCESS) { return status; }
    status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        latent, rows * heads, keys, &alpha, cache, CUDA_R_16BF, latent,
        scores, CUDA_R_16BF, keys, &beta, low, CUDA_R_16BF, latent,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    const cublasStatus_t restored = cublasSetMathMode(handle, math);
    if (status != CUBLAS_STATUS_SUCCESS) { return status; }
    if (restored != CUBLAS_STATUS_SUCCESS) { return restored; }
    status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        dim, rows, latent, &alpha, weight + (uint64_t)dim * latent,
        CUDA_R_16BF, latent, wstride, low, CUDA_R_16BF, latent,
        (uint64_t)rows * latent, &beta, y, CUDA_R_16BF, dim,
        (uint64_t)rows * dim, heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) { return status; }
    glmvq_attn_output<<<(nq + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, stream>>>(
        output, y, rows, heads, dim, nq);
    return cudaGetLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

#endif
