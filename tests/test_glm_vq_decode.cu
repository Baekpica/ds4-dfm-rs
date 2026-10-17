/* Actual selected expert replay, without a model owner or Torch libraries. */
#include "cuda/glm_vq_linear.cuh"
#include "cuda/glm_vq_decode.cuh"
#include <vector>
#include <cstring>

enum { TEST_H = GLMVQ_DEC_HIDDEN, TEST_F = GLMVQ_DEC_FFN, TEST_E = GLMVQ_DEC_USED };
static constexpr uint64_t TEST_WORKSPACE = 32ull << 20;
static constexpr float TEST_CLAMP = 10.0f;

struct decode_test {
    std::vector<void *> allocations;
    cublasHandle_t handle = nullptr;
    glmvq_decode_item host[TEST_E];
    glmvq_decode_item *items = nullptr;
    void *book = nullptr, *scratch = nullptr;
    float *input = nullptr, *gate = nullptr, *up = nullptr, *wide_mid = nullptr;
    float *partial = nullptr, *output = nullptr, *weights = nullptr;
    __nv_bfloat16 *mid = nullptr;
    uint64_t scratch_bytes = ((uint64_t)TEST_H * TEST_F + TEST_H + TEST_F) * 2;

    ~decode_test() {
        if (handle) { cublasDestroy(handle); }
        for (void *p : allocations) { cudaFree(p); }
    }

    bool alloc(void **dst, uint64_t bytes, const void *src = nullptr) {
        if (cudaMalloc(dst, bytes) != cudaSuccess) { return false; }
        allocations.push_back(*dst);
        return !src || cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    }
};

static int decode_base(decode_test &c) {
    for (uint32_t e = 0; e < TEST_E; e++) {
        const glmvq_matrix matrices[] = {c.host[e].gate, c.host[e].up, c.host[e].down};
        float *outputs[] = {c.gate + e * TEST_F, c.up + e * TEST_F, c.partial + e * TEST_H};
        for (uint32_t j = 0; j < 3; j++) {
            const glmvq_matrix &m = matrices[j];
            const glmvq_source source = {nullptr, nullptr, m, m.rows, m.cols, GLMVQ_WEIGHT_VQ};
            if (j == 2) {
                glmvq_swiglu<<<TEST_F / GLMVQ_THREADS, GLMVQ_THREADS>>>(
                    outputs[0], outputs[1], c.wide_mid + e * TEST_F, TEST_F, TEST_CLAMP);
            }
            const float *x = j == 2 ? c.wide_mid + e * TEST_F : c.input;
            if (glmvq_gemm(c.handle, source, x, outputs[j], 1, GLMVQ_COMPUTE_BF16,
                c.scratch, c.scratch_bytes, nullptr) != CUBLAS_STATUS_SUCCESS) { return -1; }
        }
    }
    glmvq_decode_reduce<<<TEST_H / GLMVQ_THREADS, GLMVQ_THREADS>>>(c.output, c.partial, c.weights, c.items);
    return (int)cudaGetLastError();
}

template <uint32_t BITS, uint32_t WARPS>
static int decode_case(decode_test &c, uint32_t repeats, float *ms,
        float *gate, float *up, uint16_t *mid, float *partial, float *output, float *baseline) {
    uint32_t blocks = 0;
    cudaError_t err = glmvq_decode_prepare<BITS, WARPS, GLMVQ_DEC_NORMAL>(&blocks);
    if (err == cudaSuccess) { err = glmvq_decode_prepare<BITS, WARPS, GLMVQ_DEC_PROOF>(&blocks); }
    if (err != cudaSuccess) { return (int)err; }
    cudaEvent_t begin = nullptr, end = nullptr;
    err = cudaEventCreate(&begin);
    if (err == cudaSuccess) { err = cudaEventCreate(&end); }
    for (uint32_t mode = 0; mode < 2 && err == cudaSuccess; mode++) {
        for (uint32_t i = 0; i < 4 && err == cudaSuccess; i++) {
            err = mode ? glmvq_decode_launch<BITS, WARPS, GLMVQ_DEC_NORMAL>(c.items, c.book,
                c.input, c.mid, c.partial, c.output, c.weights, TEST_CLAMP, blocks, nullptr)
                : (cudaError_t)decode_base(c);
        }
        if (err == cudaSuccess) { err = cudaEventRecord(begin); }
        for (uint32_t i = 0; i < repeats && err == cudaSuccess; i++) {
            err = mode ? glmvq_decode_launch<BITS, WARPS, GLMVQ_DEC_NORMAL>(c.items, c.book,
                c.input, c.mid, c.partial, c.output, c.weights, TEST_CLAMP, blocks, nullptr)
                : (cudaError_t)decode_base(c);
        }
        if (err == cudaSuccess) { err = cudaEventRecord(end); }
        if (err == cudaSuccess) { err = cudaEventSynchronize(end); }
        if (err == cudaSuccess) { err = cudaEventElapsedTime(ms + mode, begin, end); }
        if (err == cudaSuccess) { ms[mode] /= repeats; }
        if (!mode && err == cudaSuccess) {
            err = cudaMemcpy(baseline, c.output, TEST_H * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
    if (begin) { cudaEventDestroy(begin); }
    if (end) { cudaEventDestroy(end); }
    if (err != cudaSuccess) { return (int)err; }
    err = glmvq_decode_launch<BITS, WARPS, GLMVQ_DEC_PROOF>(c.items, c.book,
        c.input, c.mid, c.partial, c.output, c.weights, TEST_CLAMP, blocks, nullptr, c.gate, c.up);
    if (err != cudaSuccess) { return (int)err; }
    // Isolate activation fusion from the changed dot-product order.
    glmvq_swiglu<<<TEST_E * TEST_F / GLMVQ_THREADS, GLMVQ_THREADS>>>(
        c.gate, c.up, c.wide_mid, TEST_E * TEST_F, TEST_CLAMP);
    std::vector<float> activation(TEST_E * TEST_F);
    if ((err = cudaMemcpy(activation.data(), c.wide_mid, activation.size() * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess ||
        (err = cudaMemcpy(gate, c.gate, TEST_E * TEST_F * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess ||
        (err = cudaMemcpy(up, c.up, TEST_E * TEST_F * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess ||
        (err = cudaMemcpy(mid, c.mid, TEST_E * TEST_F * sizeof(uint16_t), cudaMemcpyDeviceToHost)) != cudaSuccess ||
        (err = cudaMemcpy(partial, c.partial, TEST_E * TEST_H * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess ||
        (err = cudaMemcpy(output, c.output, TEST_H * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) { return (int)err; }
    for (uint32_t i = 0; i < activation.size(); i++) {
        uint32_t bits;
        memcpy(&bits, &activation[i], sizeof(bits));
        if (bits != (uint32_t)mid[i] << 16) { return -2; }
    }
    return 0;
}

extern "C" int test_glmvq_decode(const uint8_t *const *main, const uint8_t *const *plane,
        const uint16_t *const *gain, const uint8_t *book, const float *input,
        const float *weights, uint32_t bits, uint32_t warps, uint32_t repeats,
        float *ms, float *gate, float *up, uint16_t *mid, float *partial, float *output, float *baseline) {
    if ((bits != 12 && bits != 13) || (warps != 16 && warps != 32) || !repeats) { return -1; }
    decode_test c;
    void *workspace = nullptr;
    if (!c.alloc(&c.book, (1u << bits) * 8, book) ||
        !c.alloc((void **)&c.input, TEST_H * sizeof(float), input) ||
        !c.alloc((void **)&c.weights, TEST_E * sizeof(float), weights) ||
        !c.alloc((void **)&c.gate, TEST_E * TEST_F * sizeof(float)) ||
        !c.alloc((void **)&c.up, TEST_E * TEST_F * sizeof(float)) ||
        !c.alloc((void **)&c.wide_mid, TEST_E * TEST_F * sizeof(float)) ||
        !c.alloc((void **)&c.mid, TEST_E * TEST_F * sizeof(__nv_bfloat16)) ||
        !c.alloc((void **)&c.partial, TEST_E * TEST_H * sizeof(float)) ||
        !c.alloc((void **)&c.output, TEST_H * sizeof(float)) ||
        !c.alloc(&c.scratch, c.scratch_bytes) || !c.alloc(&workspace, TEST_WORKSPACE) ||
        cublasCreate(&c.handle) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(c.handle, nullptr) != CUBLAS_STATUS_SUCCESS ||
        cublasSetMathMode(c.handle, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS ||
        cublasSetWorkspace(c.handle, workspace, TEST_WORKSPACE) != CUBLAS_STATUS_SUCCESS) { return -1; }
    int version = 0;
    if (cublasGetVersion(c.handle, &version) != CUBLAS_STATUS_SUCCESS || version != CUBLAS_VERSION) { return -1; }
    for (uint32_t i = 0; i < TEST_E * 3; i++) {
        const uint32_t rows = i % 3 == 2 ? TEST_H : TEST_F, cols = i % 3 == 2 ? TEST_F : TEST_H;
        const uint64_t count = (uint64_t)rows * cols;
        void *m = nullptr, *p = nullptr, *g = nullptr;
        if (!c.alloc(&m, count * 3 / 16, main[i]) ||
            (bits == 13 && !c.alloc(&p, count / 64, plane[i])) ||
            !c.alloc(&g, rows * sizeof(__half), gain[i])) { return -1; }
        const glmvq_matrix matrix = {(const uint8_t *)m, (const uint8_t *)p, c.book,
            (const __half *)g, rows, cols, GLMVQ_E4M3};
        auto &item = c.host[i / 3];
        if (i % 3 == 0) { item.gate = matrix; }
        else if (i % 3 == 1) { item.up = matrix; }
        else { item.down = matrix; }
        item.pair = TEST_E - 1 - i / 3;
    }
    if (!c.alloc((void **)&c.items, sizeof(c.host), c.host)) { return -1; }
    if (bits == 12 && warps == 16) { return decode_case<12, 16>(c, repeats, ms, gate, up, mid, partial, output, baseline); }
    if (bits == 12 && warps == 32) { return decode_case<12, 32>(c, repeats, ms, gate, up, mid, partial, output, baseline); }
    if (warps == 16) { return decode_case<13, 16>(c, repeats, ms, gate, up, mid, partial, output, baseline); }
    return decode_case<13, 32>(c, repeats, ms, gate, up, mid, partial, output, baseline);
}
