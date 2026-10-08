/* Shared preparation must preserve both raw and padded expert dot products. */
#include "../cuda/mmq/ds4_mmq.h"
#define GGML_COMMON_DECL_CPP
#include "../cuda/mmq/ggml-common.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" int ds4_mmq_glm_pair(uint32_t, const void *, const void *,
    const float *, const int32_t *, float *, float *, int, int, int, int,
    int, uint64_t, cudaStream_t);

enum { WIDTH = 256, HIDDEN = 256, EXPERTS = 288, USED = 8, MAX_ROWS = 2048,
       IQ2_XXS = 16, IQ2_XS = 17, Q4_K = 12 };
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GLM pair FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static void *device(size_t bytes, const void *data = nullptr) {
    void *ptr = nullptr;
    CHECK(cudaMalloc(&ptr, bytes) == cudaSuccess);
    if (data) { CHECK(cudaMemcpy(ptr, data, bytes, cudaMemcpyHostToDevice) == cudaSuccess); }
    return ptr;
}

static void run(uint32_t type, size_t block, int padding) {
    const size_t stride = (HIDDEN + padding) * block;
    std::vector<unsigned char> weights(EXPERTS * stride, 0xff);
    unsigned rng = 71;
    for (int e = 0; e < EXPERTS; e++) {
        for (int r = 0; r < HIDDEN; r++) {
            auto *p = weights.data() + e * stride + r * block;
            for (size_t i = 0; i < block; i++) {
                rng = rng * 1664525u + 1013904223u;
                p[i] = rng >> 24;
            }
            const uint16_t scale = 0x1400 + e % 8 * 128;
            memcpy(p, &scale, sizeof(scale));
            if (type == Q4_K) { memcpy(p + sizeof(scale), &scale, sizeof(scale)); }
        }
    }
    std::vector<float> x(MAX_ROWS * WIDTH);
    std::vector<int32_t> ids(MAX_ROWS * USED);
    for (size_t i = 0; i < x.size(); i++) { x[i] = sinf((float)i * 0.17f); }
    for (size_t i = 0; i < ids.size(); i++) { ids[i] = (i * 37 + 287) % EXPERTS; }
    void *w = device(weights.size(), weights.data());
    auto *dx = (float *)device(x.size() * sizeof(float), x.data());
    auto *di = (int32_t *)device(ids.size() * sizeof(int32_t), ids.data());
    const size_t count = (MAX_ROWS * USED + 1) * HIDDEN;
    auto *a = (float *)device(count * sizeof(float));
    auto *b = (float *)device(count * sizeof(float));
    auto *pa = (float *)device(count * sizeof(float));
    auto *pb = (float *)device(count * sizeof(float));
    std::vector<float> ref(count), got(count);
    for (int rows : {17, 128, 2048}) {
        CHECK(cudaMemset(a, 0x5a, count * sizeof(float)) == cudaSuccess);
        CHECK(cudaMemset(pa, 0x5a, count * sizeof(float)) == cudaSuccess);
        CHECK(ds4_mmq_glm_moe(type, w, dx, di, a, HIDDEN, WIDTH,
            rows, EXPERTS, USED, stride, nullptr) == 0);
        CHECK(ds4_mmq_glm_moe(type, w, dx, di, b, HIDDEN, WIDTH,
            rows, EXPERTS, USED, stride, nullptr) == 0);
        CHECK(ds4_mmq_glm_pair(type, w, w, dx, di, pa, pb, HIDDEN, WIDTH,
            rows, EXPERTS, USED, stride, nullptr) == 0);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        CHECK(cudaMemcpy(ref.data(), a, count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(cudaMemcpy(got.data(), pa, count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(memcmp(ref.data(), got.data(), count * sizeof(float)) == 0);
        CHECK(cudaMemcpy(ref.data(), b, rows * USED * HIDDEN * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(cudaMemcpy(got.data(), pb, rows * USED * HIDDEN * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(memcmp(ref.data(), got.data(), rows * USED * HIDDEN * sizeof(float)) == 0);
        for (int i = 0; i < rows * USED * HIDDEN; i++) { CHECK(std::isfinite(got[i])); }
        printf("GLM pair type=%u padding=%d rows=%d byte exact\n", type, padding, rows);
    }
    for (void *p : {w, (void *)dx, (void *)di, (void *)a, (void *)b, (void *)pa, (void *)pb}) {
        CHECK(cudaFree(p) == cudaSuccess);
    }
}

int main() {
    CHECK(ds4_mmq_init(0) == 0);
    for (int padding : {0, 3}) {
        run(IQ2_XXS, sizeof(block_iq2_xxs), padding);
        run(IQ2_XS, sizeof(block_iq2_xs), padding);
        run(Q4_K, sizeof(block_q4_K), padding);
    }
}
