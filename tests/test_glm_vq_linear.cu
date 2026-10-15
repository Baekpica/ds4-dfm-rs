/* Test-only stream/handle ownership; the production graph owns its scratch. */
#include "cuda/glm_vq_linear.cuh"

extern "C" int test_glmvq_linear(const void *data, const float *scale,
        const uint8_t *main, const uint8_t *plane, const void *book, const __half *gain,
        uint32_t rows, uint32_t cols, uint32_t precision, uint32_t storage,
        const float *input, float *output, uint32_t tokens, uint32_t compute,
        cudaStream_t stream) {
    if (storage > GLMVQ_WEIGHT_VQ || compute > GLMVQ_COMPUTE_F32) { return -1; }
    const glmvq_source source = {data, scale,
        {main, plane, book, gain, rows, cols, (glmvq_precision)precision},
        rows, cols, (glmvq_storage)storage};
    cublasHandle_t handle = NULL;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return -1; }
    if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) { cublasDestroy(handle); return -1; }
    const uint64_t bytes = glmvq_linear_bytes(source, tokens, (glmvq_compute)compute);
    void *scratch = NULL;
    if (cudaMalloc(&scratch, bytes) != cudaSuccess) { cublasDestroy(handle); return -1; }
    const cublasStatus_t result = glmvq_gemm(handle, source, input, output, tokens,
        (glmvq_compute)compute, scratch, bytes, stream);
    const cudaError_t sync = cudaStreamSynchronize(stream);
    cudaFree(scratch);
    cublasDestroy(handle);
    return sync == cudaSuccess ? (int)result : -1;
}
