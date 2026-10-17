#include "cuda/glm_vq_attention.cuh"

extern "C" int test_glmvq_attention(const float *query, const float *kv, float *output,
        const uint32_t *positions, const int32_t *selected, uint32_t stride,
        uint32_t rows, uint32_t keys, uint32_t heads, uint32_t dim, cudaStream_t stream) {
    cublasHandle_t handle = NULL;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return -1; }
    if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) { cublasDestroy(handle); return -1; }
    const uint64_t bytes = glmvq_attn_bytes(rows, keys, heads, dim).bytes;
    void *scratch = NULL;
    if (cudaMalloc(&scratch, bytes) != cudaSuccess) { cublasDestroy(handle); return -1; }
    const cublasStatus_t result = glmvq_attention(handle, query, kv, output, positions,
        selected, stride, rows, keys, heads, dim, scratch, bytes, stream);
    const cudaError_t sync = cudaStreamSynchronize(stream);
    cudaFree(scratch);
    cublasDestroy(handle);
    return sync == cudaSuccess ? (int)result : -1;
}
