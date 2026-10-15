/* Actual FP8 matrix replay; no model owner or framework CUDA libraries. */
#include "cuda/glm_vq_linear.cuh"
#include "cuda/glm_vq_fp8.cuh"
#include <cuda_profiler_api.h>
#include <vector>

enum fp8_trace { FP8_PLAIN, FP8_PROFILE };
enum fp8_path { FP8_BASE, FP8_GLOBAL, FP8_SHARED };
static constexpr uint64_t FP8_ROTATION = 64ull << 20;
static constexpr uint64_t FP8_WORKSPACE = 32ull << 20;
static constexpr uint32_t FP8_WARMUP = 16;

struct fp8_replay {
    std::vector<void *> allocations;
    cublasHandle_t handle = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;

    ~fp8_replay() {
        if (begin) { cudaEventDestroy(begin); }
        if (end) { cudaEventDestroy(end); }
        if (handle) { cublasDestroy(handle); }
        for (void *p : allocations) { cudaFree(p); }
    }

    bool alloc(void **p, uint64_t bytes, const void *host = nullptr) {
        if (cudaMalloc(p, bytes) != cudaSuccess) { return false; }
        allocations.push_back(*p);
        return !host || cudaMemcpy(*p, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    }
};

extern "C" int bench_glmvq_fp8(const uint8_t *weight, const float *scale,
        const float *input, float *output, uint32_t rows, uint32_t cols,
        uint32_t tokens, uint32_t repeats, fp8_trace trace, fp8_path path, uint32_t split, float *ms) {
    if (!weight || !scale || !input || !output || !ms || !rows || !cols ||
        rows % GLMVQ_SCALE_BLOCK || cols % GLMVQ_SCALE_BLOCK || !tokens ||
        !repeats || trace > FP8_PROFILE || path > FP8_SHARED ||
        (path != FP8_BASE && tokens != 1)) { return -1; }
    fp8_replay c;
    uint8_t *w = nullptr;
    float *s = nullptr, *x = nullptr, *y = nullptr;
    void *scratch = nullptr, *workspace = nullptr;
    const uint64_t bytes = (uint64_t)rows * cols;
    const uint32_t copies = (uint32_t)((FP8_ROTATION + bytes - 1) / bytes);
    const uint64_t nx = (uint64_t)tokens * cols, ny = (uint64_t)tokens * rows;
    const uint64_t scratch_bytes = (bytes + nx + ny) * sizeof(__nv_bfloat16);
    if (!c.alloc((void **)&w, bytes * copies) ||
        !c.alloc((void **)&s, bytes / GLMVQ_SCALE_BLOCK / GLMVQ_SCALE_BLOCK * sizeof(float), scale) ||
        !c.alloc((void **)&x, nx * sizeof(float), input) ||
        !c.alloc((void **)&y, ny * sizeof(float)) ||
        !c.alloc(&scratch, scratch_bytes) || !c.alloc(&workspace, FP8_WORKSPACE) ||
        cublasCreate(&c.handle) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(c.handle, nullptr) != CUBLAS_STATUS_SUCCESS ||
        cublasSetMathMode(c.handle, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS ||
        cublasSetWorkspace(c.handle, workspace, FP8_WORKSPACE) != CUBLAS_STATUS_SUCCESS) { return -1; }
    int version = 0;
    if (cublasGetVersion(c.handle, &version) != CUBLAS_STATUS_SUCCESS || version != CUBLAS_VERSION) { return -1; }
    for (uint32_t i = 0; i < copies; i++) {
        if (cudaMemcpy(w + i * bytes, weight, bytes, cudaMemcpyHostToDevice) != cudaSuccess) { return -1; }
    }
    auto run = [&](uint32_t i) {
        const glmvq_source source = {w + i % copies * bytes, s, {}, rows, cols, GLMVQ_WEIGHT_FP8};
        if (path != FP8_BASE) {
            const auto feed = path == FP8_GLOBAL ? GLMVQ_FP8_GLOBAL : GLMVQ_FP8_SHARED;
            return glmvq_fp8_decode(source.data, s, x, y, rows, cols, split, feed, nullptr) == cudaSuccess
                ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
        }
        return glmvq_gemm(c.handle, source, x, y, tokens, GLMVQ_COMPUTE_BF16,
            scratch, scratch_bytes, nullptr);
    };
    for (uint32_t i = 0; i < FP8_WARMUP; i++) {
        if (run(i) != CUBLAS_STATUS_SUCCESS) { return -1; }
    }
    if (cudaDeviceSynchronize() != cudaSuccess || cudaEventCreate(&c.begin) != cudaSuccess ||
        cudaEventCreate(&c.end) != cudaSuccess) { return -1; }
    if (trace == FP8_PROFILE && cudaProfilerStart() != cudaSuccess) { return -1; }
    if (cudaEventRecord(c.begin) != cudaSuccess) { return -1; }
    for (uint32_t i = 0; i < repeats; i++) {
        if (run(i + FP8_WARMUP) != CUBLAS_STATUS_SUCCESS) { return -1; }
    }
    if (cudaEventRecord(c.end) != cudaSuccess || cudaEventSynchronize(c.end) != cudaSuccess ||
        cudaEventElapsedTime(ms, c.begin, c.end) != cudaSuccess) { return -1; }
    if (trace == FP8_PROFILE && cudaProfilerStop() != cudaSuccess) { return -1; }
    *ms /= repeats;
    return (int)cudaMemcpy(output, y, ny * sizeof(float), cudaMemcpyDeviceToHost);
}
