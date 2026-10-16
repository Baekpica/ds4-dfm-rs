#include "cuda/glm_vq_index.cuh"

/* Test-only allocation/handle ownership. No framework CUDA libraries. */
extern "C" int test_gvq_pool(const float *raw, const float *gate,
        const uint16_t *norm, const uint16_t *bias, const uint16_t *ape,
        uint16_t *output, float *tail, const uint32_t *chunks, uint32_t count,
        uint32_t rows) {
    constexpr uint32_t width = DS4_GLM53_POOL_DIM, ratio = DS4_GLM53_POOL_SIZE;
    if (!rows || !count) { return -1; }
    uint32_t total = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (!chunks[i] || chunks[i] > rows - total) { return -1; }
        total += chunks[i];
    }
    if (total != rows) { return -1; }
    const size_t xb = (uint64_t)rows * width * sizeof(float);
    const size_t pb = (uint64_t)((rows + ratio - 1) / ratio) * width * sizeof(uint16_t);
    const size_t tb = 2u * ratio * width * sizeof(float);
    float *x = nullptr, *g = nullptr, *state = nullptr;
    __nv_bfloat16 *params = nullptr, *pool = nullptr;
    cudaError_t err = cudaMalloc(&x, xb);
    if (err == cudaSuccess) { err = cudaMalloc(&g, xb); }
    if (err == cudaSuccess) { err = cudaMalloc(&state, tb); }
    if (err == cudaSuccess) { err = cudaMalloc(&params, (2u + ratio) * width * sizeof(uint16_t)); }
    if (err == cudaSuccess) { err = cudaMalloc(&pool, pb); }
    if (err == cudaSuccess) { err = cudaMemcpy(x, raw, xb, cudaMemcpyHostToDevice); }
    if (err == cudaSuccess) { err = cudaMemcpy(g, gate, xb, cudaMemcpyHostToDevice); }
    if (err == cudaSuccess) { err = cudaMemcpy(params, norm, width * sizeof(uint16_t), cudaMemcpyHostToDevice); }
    if (err == cudaSuccess) { err = cudaMemcpy(params + width, bias, width * sizeof(uint16_t), cudaMemcpyHostToDevice); }
    if (err == cudaSuccess) { err = cudaMemcpy(params + 2u * width, ape, ratio * width * sizeof(uint16_t), cudaMemcpyHostToDevice); }
    if (err == cudaSuccess) { err = cudaMemset(state, 0, tb); }
    if (err == cudaSuccess) { err = cudaMemset(pool, 0, pb); }
    uint32_t pos = 0;
    for (uint32_t i = 0; i < count && err == cudaSuccess; i++) {
        err = glmvq_pool_update(pool, state, state + ratio * width,
            x + (uint64_t)pos * width, g + (uint64_t)pos * width,
            params, params + width, params + 2u * width, pos, chunks[i], 0);
        pos += chunks[i];
    }
    if (err == cudaSuccess) { err = cudaMemcpy(output, pool, pb, cudaMemcpyDeviceToHost); }
    if (err == cudaSuccess) { err = cudaMemcpy(tail, state, tb, cudaMemcpyDeviceToHost); }
    cudaFree(pool); cudaFree(params); cudaFree(state); cudaFree(g); cudaFree(x);
    return (int)err;
}

extern "C" int test_gvq_scores(const float *query, const float *weights,
        const uint16_t *cache, float *output, uint32_t rows, uint32_t pools,
        uint32_t first) {
    constexpr uint32_t width = DS4_GLM53_POOL_DIM, heads = GLMVQ_INDEX_HEADS;
    const size_t qb = (uint64_t)rows * heads * width * sizeof(float);
    const size_t wb = (uint64_t)rows * heads * sizeof(float);
    const size_t pb = (uint64_t)pools * width * sizeof(uint16_t);
    const size_t ob = (uint64_t)rows * pools * sizeof(float);
    const uint64_t bytes = glmvq_index_bytes(rows, pools);
    float *q = nullptr, *w = nullptr, *out = nullptr;
    __nv_bfloat16 *pool = nullptr;
    void *scratch = nullptr;
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return -1; }
    cudaError_t err = cudaMalloc(&q, qb);
    if (err == cudaSuccess) { err = cudaMalloc(&w, wb); }
    if (err == cudaSuccess) { err = cudaMalloc(&pool, pb); }
    if (err == cudaSuccess) { err = cudaMalloc(&out, ob); }
    if (err == cudaSuccess) { err = cudaMalloc(&scratch, bytes); }
    if (err == cudaSuccess) { err = cudaMemcpy(q, query, qb, cudaMemcpyHostToDevice); }
    if (err == cudaSuccess) { err = cudaMemcpy(w, weights, wb, cudaMemcpyHostToDevice); }
    if (err == cudaSuccess) { err = cudaMemcpy(pool, cache, pb, cudaMemcpyHostToDevice); }
    if (err == cudaSuccess && glmvq_index_score(handle, out, q, w, pool,
            pools, rows, first, scratch, bytes, 0) != CUBLAS_STATUS_SUCCESS) {
        err = cudaErrorUnknown;
    }
    if (err == cudaSuccess) { err = cudaMemcpy(output, out, ob, cudaMemcpyDeviceToHost); }
    cudaFree(scratch); cudaFree(out); cudaFree(pool); cudaFree(w); cudaFree(q);
    cublasDestroy(handle);
    return (int)err;
}
