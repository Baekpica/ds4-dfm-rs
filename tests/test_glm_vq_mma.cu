#include "cuda/glm_vq_mma.cuh"

extern "C" int mma_fits(const glmvq_source *source, uint32_t rows,
        uint32_t cols, const void *book) {
    return glmvq_mma_fits(*source, rows, cols, book);
}

extern "C" int mma_mapping(__nv_bfloat16 *gathered, float *output,
        const float *input, const float *down, const float *weights,
        const uint32_t *pairs, const uint32_t *order, uint32_t rows,
        uint32_t width, uint32_t used) {
    const uint64_t count = (uint64_t)rows * width;
    glmvq_mma_gather<<<(count * used + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS>>>(
        gathered, input, pairs, used, width, count * used);
    glmvq_mma_reduce<<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS>>>(
        output, down, weights, pairs, order, used, width, count);
    return (int)cudaGetLastError();
}

template<glmvq_mma_stage MODE>
static int trial_resources(int *blocks, int *registers, size_t *local) {
    cudaFuncAttributes attr{};
    cudaError_t error = cudaFuncGetAttributes(&attr, glmvq_mma_kernel<256, 32, MODE>);
    if (error != cudaSuccess) { return (int)error; }
    *registers = attr.numRegs; *local = attr.localSizeBytes;
    return (int)glmvq_mma_prepare<256, 32, MODE>(blocks);
}

extern "C" int mma_resources(int stage, int *blocks, int *registers, size_t *local) {
    if (stage == GLMVQ_MMA_GATE) { return trial_resources<GLMVQ_MMA_GATE>(blocks, registers, local); }
    if (stage == GLMVQ_MMA_UP) { return trial_resources<GLMVQ_MMA_UP>(blocks, registers, local); }
    if (stage == GLMVQ_MMA_DOWN) { return trial_resources<GLMVQ_MMA_DOWN>(blocks, registers, local); }
    return (int)cudaErrorInvalidValue;
}

template<glmvq_mma_stage MODE>
static int trial_launch(const glmvq_source *sources, const glmvq_mma_item *items,
        uint32_t nitems, const uint8_t *book, const __nv_bfloat16 *input,
        float *gate, __nv_bfloat16 *mid, float *output, uint32_t rows,
        uint32_t cols, float limit, uint32_t blocks) {
    if (!sources || !items || !nitems || !book || !input || !rows || !cols ||
        rows % 256 || cols % 128 || !blocks) { return (int)cudaErrorInvalidValue; }
    const uint32_t grid = min(nitems * (rows / 256), blocks);
    glmvq_mma_kernel<256, 32, MODE><<<grid, 512, glmvq_mma_bytes<256, 32>()>>>(
        sources, items, nitems, book, input, gate, mid, output, rows, cols, limit);
    return (int)cudaGetLastError();
}

extern "C" int mma_launch(int stage, const glmvq_source *sources, const glmvq_mma_item *items,
        uint32_t nitems, const uint8_t *book, const __nv_bfloat16 *input,
        float *gate, __nv_bfloat16 *mid, float *output, uint32_t rows,
        uint32_t cols, float limit, uint32_t blocks) {
#define TRIAL_LAUNCH(MODE) trial_launch<MODE>(sources, items, nitems, book, input, gate, mid, output, rows, cols, limit, blocks)
    if (stage == GLMVQ_MMA_GATE) { return TRIAL_LAUNCH(GLMVQ_MMA_GATE); }
    if (stage == GLMVQ_MMA_UP) { return TRIAL_LAUNCH(GLMVQ_MMA_UP); }
    if (stage == GLMVQ_MMA_DOWN) { return TRIAL_LAUNCH(GLMVQ_MMA_DOWN); }
    return (int)cudaErrorInvalidValue;
#undef TRIAL_LAUNCH
}

extern "C" void *mma_ref_create(void *workspace) {
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return nullptr; }
    int version = 0;
    cublasGetVersion(handle, &version);
    if (version != CUBLAS_VERSION ||
        cublasSetStream(handle, nullptr) != CUBLAS_STATUS_SUCCESS ||
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS ||
        cublasSetWorkspace(handle, workspace, 32u * 1024 * 1024) != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle); return nullptr;
    }
    return handle;
}

extern "C" int mma_ref(void *handle, const glmvq_source *source, uint32_t tokens,
        const float *input, float *output, void *scratch, uint64_t bytes) {
    return (int)glmvq_gemm((cublasHandle_t)handle, *source, input, output, tokens,
        GLMVQ_COMPUTE_BF16, scratch, bytes, nullptr, GLMVQ_RESTORE_VECTOR);
}

extern "C" int mma_ref_free(void *handle) { return (int)cublasDestroy((cublasHandle_t)handle); }

extern "C" int mma_swiglu(const float *gate, const float *up, float *output, uint64_t count) {
    constexpr float GLM_LIMIT = 10.0f;
    glmvq_swiglu<<<(count + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS>>>(
        gate, up, output, count, GLM_LIMIT);
    return (int)cudaGetLastError();
}

