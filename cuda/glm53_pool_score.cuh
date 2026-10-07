#ifndef DS4_GLM53_POOL_SCORE_CUH
#define DS4_GLM53_POOL_SCORE_CUH

#include "../ds4_glm53_compact.h"

enum { GLM53_POOL_HEADS = 32, GLM53_POOL_WARP_SIZE = 32,
       GLM53_POOL_WARPS = DS4_GLM53_POOL_DIM / GLM53_POOL_WARP_SIZE };

__global__ static void glm53_pool_score_kernel(float *scores,
        const float *q, const float *weights, const __half *pool_cache,
        uint32_t n_pools, uint32_t pos0, uint32_t heads, float scale) {
    const uint32_t pool = blockIdx.x;
    const uint32_t token = blockIdx.y;
    const uint32_t d = threadIdx.x;
    if (pool >= (pos0 + token + 1u) / DS4_GLM53_POOL_SIZE) {
        if (d == 0u) { scores[(uint64_t)token * n_pools + pool] = -INFINITY; }
        return;
    }
    __shared__ float partial[DS4_GLM53_POOL_DIM];
    float total = 0.0f;
    const float key = __half2float(pool_cache[(uint64_t)pool * DS4_GLM53_POOL_DIM + d]);
    for (uint32_t h = 0; h < heads; h++) {
        partial[d] = q[((uint64_t)token * heads + h) * DS4_GLM53_POOL_DIM + d] * key;
        __syncthreads();
        for (uint32_t step = DS4_GLM53_POOL_DIM / 2u; step; step >>= 1u) {
            if (d < step) { partial[d] += partial[d + step]; }
            __syncthreads();
        }
        if (d == 0u) { total = fmaf(fmaxf(partial[0], 0.0f), weights[(uint64_t)token * heads + h], total); }
        __syncthreads();
    }
    if (d == 0u) { scores[(uint64_t)token * n_pools + pool] = total * scale; }
}

__global__ static void glm53_pool_score_warp(float *scores,
        const float *q, const float *weights, const __half *pool_cache,
        uint32_t n_pools, uint32_t pos0, float scale) {
    const uint32_t pool = blockIdx.x, token = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (pool >= (pos0 + token + 1u) / DS4_GLM53_POOL_SIZE) {
        if (tid == 0u) { scores[(uint64_t)token * n_pools + pool] = -INFINITY; }
        return;
    }
    const uint32_t lane = tid % GLM53_POOL_WARP_SIZE;
    const uint32_t warp = tid / GLM53_POOL_WARP_SIZE;
    const __half *key = pool_cache + (uint64_t)pool * DS4_GLM53_POOL_DIM + lane;
    float keys[GLM53_POOL_WARPS];
    #pragma unroll
    for (uint32_t i = 0; i < GLM53_POOL_WARPS; i++) {
        keys[i] = __half2float(key[i * GLM53_POOL_WARP_SIZE]);
    }
    __shared__ float dots[GLM53_POOL_HEADS];
    for (uint32_t h = warp; h < GLM53_POOL_HEADS; h += GLM53_POOL_WARPS) {
        const float *query = q + ((uint64_t)token * GLM53_POOL_HEADS + h) * DS4_GLM53_POOL_DIM + lane;
        float product[GLM53_POOL_WARPS];
        #pragma unroll
        for (uint32_t i = 0; i < GLM53_POOL_WARPS; i++) {
            product[i] = __fmul_rn(query[i * GLM53_POOL_WARP_SIZE], keys[i]);
        }
        // Reproduce the shared tree's +64, +32, +16... additions. Explicit
        // rounding prevents contraction into FMAs across those boundaries.
        float dot = __fadd_rn(__fadd_rn(product[0], product[2]),
                             __fadd_rn(product[1], product[3]));
        #pragma unroll
        for (uint32_t step = GLM53_POOL_WARP_SIZE / 2u; step; step >>= 1u) {
            const float other = __shfl_down_sync(0xffffffffu, dot, step);
            if (lane < step) { dot = __fadd_rn(dot, other); }
        }
        if (lane == 0u) { dots[h] = dot; }
    }
    __syncthreads();
    if (tid != 0u) { return; }

    // Heads are independent until this ordered weighted sum.
    float total = 0.0f;
    for (uint32_t h = 0; h < GLM53_POOL_HEADS; h++) {
        total = fmaf(fmaxf(dots[h], 0.0f), weights[(uint64_t)token * GLM53_POOL_HEADS + h], total);
    }
    scores[(uint64_t)token * n_pools + pool] = total * scale;
}

static int glm53_pool_warp(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *value = getenv("DS4_GLM53_POOL_WARP");
        enabled = !value || atoi(value) > 0;
    }
    return enabled;
}

extern "C" int ds4_gpu_glm53_pool_score(ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *pool_cache, uint32_t n_pools, uint32_t rows,
        uint32_t pos0, uint32_t heads, float scale) {
    if (ds4_capture_active() || !glm53_compact_shape(rows, heads, 32u, 32u) ||
        !n_pools || n_pools > UINT32_MAX / DS4_GLM53_POOL_SIZE ||
        pos0 > UINT32_MAX - rows ||
        (pos0 + rows) / DS4_GLM53_POOL_SIZE > n_pools ||
        !isfinite(scale) || scale <= 0.0f ||
        !glm53_compact_has(scores, (uint64_t)rows * n_pools, sizeof(float)) ||
        !glm53_compact_has(q, (uint64_t)rows * heads * DS4_GLM53_POOL_DIM, sizeof(float)) ||
        !glm53_compact_has(weights, (uint64_t)rows * heads, sizeof(float)) ||
        !glm53_compact_has(pool_cache, (uint64_t)n_pools * DS4_GLM53_POOL_DIM, sizeof(__half))) {
        return 0;
    }
    if (heads == GLM53_POOL_HEADS && glm53_pool_warp()) {
        glm53_pool_score_warp<<<dim3(n_pools, rows), DS4_GLM53_POOL_DIM,
            0, ds4_current_stream()>>>(
                (float *)scores->ptr, (const float *)q->ptr,
                (const float *)weights->ptr, (const __half *)pool_cache->ptr,
                n_pools, pos0, scale);
        return cuda_ok(cudaGetLastError(), "GLM-5.3 warp pooled index scores");
    }
    glm53_pool_score_kernel<<<dim3(n_pools, rows), DS4_GLM53_POOL_DIM,
        0, ds4_current_stream()>>>(
            (float *)scores->ptr, (const float *)q->ptr,
            (const float *)weights->ptr, (const __half *)pool_cache->ptr,
            n_pools, pos0, heads, scale);
    return cuda_ok(cudaGetLastError(), "GLM-5.3 pooled index scores");
}

#endif
