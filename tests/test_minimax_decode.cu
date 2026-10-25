/* MiniMax's routed-expert decode kernels (cuda/minimax_decode.cuh) at
 * production shapes, no model: top-4 experts of 6144 -> 3072 -> 6144 on a
 * random 13-bit book, with and without the GVR8 residual stage, rotating
 * over DS4_MD_SETS distinct expert sets (default 32, ~1.5 GB) so every
 * launch streams its weights from DRAM. Prints each kernel's time and
 * weight-read rate; DS4_MD_ONLY=gu|down runs that kernel once (for ncu),
 * DS4_MD_RES=0|1 picks the stage (default both). Exit 0 = launches ran. */
#include "cuda/minimax_decode.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)

static uint64_t g_state = 0x9e3779b97f4a7c15ull;
static uint64_t next_u64() {
    g_state ^= g_state << 13; g_state ^= g_state >> 7; g_state ^= g_state << 17;
    return g_state;
}

static void *dev_random(uint64_t bytes) {
    std::vector<uint8_t> h(bytes);
    for (uint64_t i = 0; i < bytes; i += 8) {
        const uint64_t r = next_u64();
        memcpy(h.data() + i, &r, bytes - i < 8 ? bytes - i : 8);
    }
    void *p = nullptr;
    CK(cudaMalloc(&p, bytes));
    CK(cudaMemcpy(p, h.data(), bytes, cudaMemcpyHostToDevice));
    return p;
}

// Finite E4M3 codewords (no 0x7f / 0xff NaN bytes) and F16 gains near 1/64.
static void *dev_book(uint32_t words) {
    std::vector<uint8_t> b((size_t)words * GLMVQ_VECTOR);
    for (auto &x : b) { x = (uint8_t)(next_u64() & 0x77u); }
    void *p = nullptr;
    CK(cudaMalloc(&p, b.size()));
    CK(cudaMemcpy(p, b.data(), b.size(), cudaMemcpyHostToDevice));
    return p;
}

static glmvq_matrix make_matrix(uint32_t rows, uint32_t cols, const void *book, const void *res_book) {
    glmvq_matrix m = {};
    m.main = (const uint8_t *)dev_random((uint64_t)rows * cols * 3 / 16);
    m.plane = (const uint8_t *)dev_random((uint64_t)rows * cols / 64);
    std::vector<__half> gain(rows, __float2half(1.0f / 64.0f));
    void *g = nullptr;
    CK(cudaMalloc(&g, rows * sizeof(__half)));
    CK(cudaMemcpy(g, gain.data(), rows * sizeof(__half), cudaMemcpyHostToDevice));
    m.gain = (const __half *)g;
    m.book = book;
    m.rows = rows;
    m.cols = cols;
    m.precision = GLMVQ_E4M3;
    if (res_book) {
        m.res = (const uint8_t *)dev_random((uint64_t)rows * cols / 8);
        m.res_book = (const uint8_t *)res_book;
    }
    return m;
}

template <bool RES>
static void run(int sets, const char *only) {
    constexpr uint32_t books = glmvq_decode_books<13, RES>();
    CK(cudaFuncSetAttribute(mm_decode_gu<RES>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        books + MM_HIDDEN * sizeof(__nv_bfloat16)));
    CK(cudaFuncSetAttribute(mm_decode_down<RES>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        books + MM_USED * MM_FFN * sizeof(__nv_bfloat16)));
    int device = 0, sms = 0;
    CK(cudaGetDevice(&device));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    void *book = dev_book(MM_WORDS), *book2 = RES ? dev_book(GLMVQ_RES_WORDS) : nullptr;
    std::vector<glmvq_decode_item *> items(sets);
    for (int s = 0; s < sets; s++) {
        glmvq_decode_item h[MM_USED];
        for (uint32_t e = 0; e < MM_USED; e++) {
            h[e] = {make_matrix(MM_FFN, MM_HIDDEN, book, book2), make_matrix(MM_FFN, MM_HIDDEN, book, book2),
                make_matrix(MM_HIDDEN, MM_FFN, book, book2), e};
        }
        CK(cudaMalloc(&items[s], sizeof(h)));
        CK(cudaMemcpy(items[s], h, sizeof(h), cudaMemcpyHostToDevice));
    }
    std::vector<float> x(MM_HIDDEN);
    for (auto &v : x) { v = __bfloat162float(__float2bfloat16_rn((float)((int)(next_u64() % 2001) - 1000) / 1000.0f)); }
    float *input = nullptr, *partial = nullptr;
    __nv_bfloat16 *mid = nullptr;
    CK(cudaMalloc(&input, MM_HIDDEN * sizeof(float)));
    CK(cudaMemcpy(input, x.data(), MM_HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&mid, MM_USED * MM_FFN * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&partial, MM_USED * MM_HIDDEN * sizeof(float)));
    auto gu = [&](int s) {
        mm_decode_gu<RES><<<sms, MM_WARPS * 32, books + MM_HIDDEN * sizeof(__nv_bfloat16)>>>(items[s],
            (const uint4 *)book, (const uint4 *)book2, input, mid);
    };
    auto dn = [&](int s) {
        mm_decode_down<RES><<<sms, MM_WARPS * 32, books + MM_USED * MM_FFN * sizeof(__nv_bfloat16)>>>(items[s],
            (const uint4 *)book, (const uint4 *)book2, mid, partial);
    };
    if (only) {
        if (!strcmp(only, "gu")) { gu(0); } else { dn(0); }
        CK(cudaDeviceSynchronize());
        return;
    }
    // Bytes per launch: 13 bits (+8 residual) per 8 weights, gains.
    const double per = (double)MM_FFN * MM_HIDDEN * (RES ? 21.0 : 13.0) / 64.0;
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (int pass = 0; pass < 2; pass++) {   // the first pass warms up
        float t_gu = 0.0f, t_dn = 0.0f;
        for (int s = 0; s < sets; s++) {
            float ms = 0.0f;
            CK(cudaEventRecord(a));
            gu(s);
            CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b));
            CK(cudaEventElapsedTime(&ms, a, b));
            t_gu += ms;
            CK(cudaEventRecord(a));
            dn(s);
            CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b));
            CK(cudaEventElapsedTime(&ms, a, b));
            t_dn += ms;
        }
        CK(cudaGetLastError());
        if (pass) {
            const double gu_us = t_gu * 1e3 / sets, dn_us = t_dn * 1e3 / sets;
            printf("%s: gate/up %.1f us (%.0f GB/s), down %.1f us (%.0f GB/s)\n", RES ? "R13+8" : "R13", gu_us,
                MM_USED * 2 * per / gu_us / 1e3, dn_us, MM_USED * per / dn_us / 1e3);
        }
    }
}

int main() {
    const char *s = getenv("DS4_MD_SETS"), *only = getenv("DS4_MD_ONLY"), *res = getenv("DS4_MD_RES");
    const int sets = s ? atoi(s) : 32;
    if (!res || !strcmp(res, "0")) { run<false>(sets, only); }
    if (!res || !strcmp(res, "1")) { run<true>(sets, only); }
    return 0;
}
