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

/* Host buffers keep this Q4_K gate independent of Torch's bundled cuBLAS. */
extern "C" int test_glmvq_q4(const uint8_t *packed, const float *input, float *output,
        uint32_t rows, uint32_t cols, uint32_t tokens) {
    if (!packed || !input || !output || !rows || !cols || cols % GLMVQ_Q4_BLOCK ||
        !tokens || tokens > GLMVQ_Q4_WARPS) { return -1; }
    void *w = nullptr;
    float *x = nullptr, *y = nullptr;
    const uint64_t wb = (uint64_t)rows * cols / GLMVQ_Q4_BLOCK * GLMVQ_Q4_BYTES;
    const uint64_t xb = (uint64_t)tokens * cols * sizeof(float), yb = (uint64_t)tokens * rows * sizeof(float);
    cudaError_t err = cudaMalloc(&w, wb);
    if (err == cudaSuccess) { err = cudaMalloc(&x, xb); }
    if (err == cudaSuccess) { err = cudaMalloc(&y, yb); }
    if (err == cudaSuccess) { err = cudaMemcpy(w, packed, wb, cudaMemcpyHostToDevice); }
    if (err == cudaSuccess) { err = cudaMemcpy(x, input, xb, cudaMemcpyHostToDevice); }
    if (err == cudaSuccess) { err = glmvq_q4_decode(w, x, y, rows, cols, tokens, 0); }
    if (err == cudaSuccess) { err = cudaMemcpy(output, y, yb, cudaMemcpyDeviceToHost); }
    cudaFree(y); cudaFree(x); cudaFree(w);
    return (int)err;
}
