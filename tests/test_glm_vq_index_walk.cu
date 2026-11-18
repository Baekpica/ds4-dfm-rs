// glmvq_index_walk against glmvq_index_fused: the same scores bit for bit on
// random BF16 pools and F32 queries (ragged rows and pools, a causal frontier),
// then both kernels' time at prefill shapes. No model.
#include "cuda/glm_vq_index.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static bool check(const char *what, cudaError_t err) {
    if (err != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err)); }
    return err == cudaSuccess;
}

struct Case {
    float *q = nullptr, *w = nullptr, *a = nullptr, *b = nullptr;
    __nv_bfloat16 *pool = nullptr;
    uint32_t rows, pools, first;
    ~Case() { cudaFree(q); cudaFree(w); cudaFree(a); cudaFree(b); cudaFree(pool); }
};

__global__ void fill_kernel(float *q, float *w, __nv_bfloat16 *pool, uint64_t nq, uint64_t nw, uint64_t np, uint32_t seed) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    // A cheap hash to a signed value around 1: enough spread for the ReLU and the BF16 rounding.
    auto value = [seed](uint64_t n) {
        uint32_t x = (uint32_t)n * 2654435761u + (uint32_t)(n >> 32) * 40503u + seed;
        x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
        return ((float)(x & 0xffffu) - 32768.0f) / 16384.0f;
    };
    if (i < nq) { q[i] = value(i); }
    if (i < nw) { w[i] = value(i + nq); }
    if (i < np) { pool[i] = __float2bfloat16_rn(value(i + nq + nw)); }
}

static bool make(Case &c, uint32_t rows, uint32_t pools, uint32_t first, uint32_t seed) {
    c.rows = rows; c.pools = pools; c.first = first;
    const uint64_t nq = (uint64_t)rows * GLMVQ_INDEX_HEADS * DS4_GLM53_POOL_DIM, nw = (uint64_t)rows * GLMVQ_INDEX_HEADS;
    const uint64_t np = (uint64_t)pools * DS4_GLM53_POOL_DIM, no = (uint64_t)rows * pools;
    if (!check("alloc", cudaMalloc(&c.q, nq * 4)) || !check("alloc", cudaMalloc(&c.w, nw * 4)) ||
        !check("alloc", cudaMalloc(&c.pool, np * 2)) || !check("alloc", cudaMalloc(&c.a, no * 4)) ||
        !check("alloc", cudaMalloc(&c.b, no * 4))) { return false; }
    const uint64_t n = nq > np ? nq : np;
    fill_kernel<<<(uint32_t)((n + 255) / 256), 256>>>(c.q, c.w, c.pool, nq, nw, np, seed);
    return check("fill", cudaDeviceSynchronize());
}

static void fused(const Case &c) {
    const dim3 grid((c.pools + GLMVQ_IDX_POOLS - 1) / GLMVQ_IDX_POOLS, (c.rows + GLMVQ_IDX_ROWS - 1) / GLMVQ_IDX_ROWS);
    glmvq_index_fused<<<grid, GLMVQ_IDX_THREADS, GLMVQ_IDX_SMEM>>>(c.a, c.q, c.w, c.pool, c.pools, c.rows, c.first);
}

static void walk(const Case &c) {
    glmvq_index_walk<<<(c.rows + GLMVQ_IDW_ROWS - 1) / GLMVQ_IDW_ROWS, GLMVQ_IDW_THREADS, GLMVQ_IDW_SMEM>>>(c.b, c.q, c.w,
        c.pool, c.pools, c.rows, c.first);
}

static bool equal(uint32_t rows, uint32_t pools, uint32_t first, uint32_t seed) {
    Case c;
    if (!make(c, rows, pools, first, seed)) { return false; }
    const uint64_t n = (uint64_t)rows * pools;
    cudaMemset(c.a, 0x5a, n * 4);
    cudaMemset(c.b, 0xa5, n * 4);
    fused(c);
    walk(c);
    std::vector<uint32_t> a(n), b(n);
    if (!check("run", cudaDeviceSynchronize()) || !check("copy", cudaMemcpy(a.data(), c.a, n * 4, cudaMemcpyDeviceToHost)) ||
        !check("copy", cudaMemcpy(b.data(), c.b, n * 4, cudaMemcpyDeviceToHost))) { return false; }
    uint64_t diff = 0, visible = 0;
    for (uint64_t i = 0; i < n; i++) {
        diff += a[i] != b[i];
        visible += a[i] != 0xff7fffffu;   // -FLT_MAX marks an invisible pool
    }
    if (diff || !visible) {
        fprintf(stderr, "FAIL rows=%u pools=%u first=%u: %llu of %llu differ (%llu visible)\n", rows, pools, first,
            (unsigned long long)diff, (unsigned long long)n, (unsigned long long)visible);
        return false;
    }
    return true;
}

template <typename F> static float time_ms(F &&run, int reps) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    run();
    cudaEventRecord(a);
    for (int i = 0; i < reps; i++) { run(); }
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / reps;
}

int main() {
    if (!check("fused smem", cudaFuncSetAttribute(glmvq_index_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GLMVQ_IDX_SMEM)) ||
        !check("walk smem", cudaFuncSetAttribute(glmvq_index_walk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GLMVQ_IDW_SMEM))) {
        return 1;
    }
    // rows, pools, first: ragged tails, a frontier inside the rows, a one-tile cache.
    const uint32_t cases[][3] = {{4, 1, 0}, {37, 9, 0}, {130, 300, 1071}, {2048, 512, 0}, {2048, 1024, 2048},
                                 {515, 4099, 15881}, {64, 16384, 65472}};
    bool ok = true;
    uint32_t seed = 1;
    for (const auto &k : cases) { ok = equal(k[0], k[1], k[2], seed++) && ok; }
    if (!ok) { return 1; }
    printf("glm vq index walk: scores equal the fused kernel's\n");

    const char *big = getenv("DS4_IDX_BIG");
    const uint32_t rows = 4096;
    for (uint32_t pools : {2048u, 8192u, 16384u, 65536u, 262144u}) {
        if (pools > 16384u && !(big && !strcmp(big, "1"))) { continue; }
        // A 1M context prefills 2,048-row chunks by default.
        const uint32_t n = pools > 65536u ? 2048u : rows;
        Case c;
        if (!make(c, n, pools, pools * DS4_GLM53_POOL_SIZE - n, 99)) { return 1; }
        const int reps = pools > 16384u ? 3 : 10;
        const float f = time_ms([&] { fused(c); }, reps), w = time_ms([&] { walk(c); }, reps);
        if (!check("time", cudaDeviceSynchronize())) { return 1; }
        const double tflop = 2.0 * n * GLMVQ_INDEX_HEADS * (double)pools * DS4_GLM53_POOL_DIM / 1e12;
        printf("  %u rows x %6u pools: fused %8.2f ms (%5.1f TFLOPS)  walk %8.2f ms (%5.1f TFLOPS)\n",
            n, pools, f, tflop / f * 1e3, w, tflop / w * 1e3);
    }
    return 0;
}
