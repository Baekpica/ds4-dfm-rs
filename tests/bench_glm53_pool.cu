/* Production pooled scores; synthetic Q, head weights and FP16 pool keys. */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

struct ds4_gpu_tensor { void *ptr; uint64_t bytes; };
static int capture;
static int ds4_capture_active(void) { return capture; }
static cudaStream_t ds4_current_stream(void) { return 0; }
static int cuda_ok(cudaError_t status, const char *label) {
    if (status == cudaSuccess) { return 1; }
    fprintf(stderr, "%s: %s\n", label, cudaGetErrorString(status));
    return 0;
}
static bool glm53_compact_has(const ds4_gpu_tensor *t, uint64_t n, uint64_t elem) {
    return t && t->ptr && n <= UINT64_MAX / elem && t->bytes >= n * elem;
}
static bool glm53_compact_shape(uint32_t rows, uint32_t heads,
        uint32_t latent_dim, uint32_t head_dim) {
    return rows && rows <= 65535u && heads && heads <= 256u &&
        latent_dim && latent_dim <= 1024u && latent_dim % 32u == 0u &&
        head_dim && head_dim <= 1024u && head_dim % 32u == 0u;
}
#include "../cuda/glm53_pool_score.cuh"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "pool FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

enum { HEADS = 32, WARM_CALLS = 3, TIME_CALLS = 12, GUARD = 16 };
enum Run { NUMERIC, TIMING, PROFILE };
static constexpr float SCALE = 1.0f / 64.0f;
static constexpr float SENTINEL = 12345.0f;
static constexpr double CPU_MAX_ABS = 3.0e-5;

static ds4_gpu_tensor upload(const void *src, uint64_t bytes) {
    ds4_gpu_tensor t = {NULL, bytes};
    CHECK(cuda_ok(cudaMalloc(&t.ptr, (size_t)bytes), "alloc"));
    CHECK(cuda_ok(cudaMemcpy(t.ptr, src, (size_t)bytes, cudaMemcpyHostToDevice), "upload"));
    return t;
}

static uint32_t random_step(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return x;
}

static void cpu_check(uint32_t rows, uint32_t pools, uint32_t pos0,
        const std::vector<__half> &key, const std::vector<float> &q,
        const std::vector<float> &weights, const std::vector<float> &out) {
    const uint32_t tokens[] = {0u, rows / 3u, rows / 2u, rows - 1u};
    const uint32_t sample_pools[] = {0u, 17u, 511u, 512u, pools / 2u, pools - 40u,
        pools - 33u, pools - 32u, pools - 31u, pools - 17u, pools - 2u, pools - 1u};
    double worst = 0.0;
    for (uint32_t t : tokens) {
        for (uint32_t p : sample_pools) {
            const float actual = out[(uint64_t)t * pools + p];
            if (p >= (pos0 + t + 1u) / DS4_GLM53_POOL_SIZE) {
                CHECK(actual == -INFINITY);
                continue;
            }
            double total = 0.0;
            for (uint32_t h = 0; h < HEADS; h++) {
                double dot = 0.0;
                for (uint32_t d = 0; d < DS4_GLM53_POOL_DIM; d++) {
                    dot += (double)q[((uint64_t)t * HEADS + h) * DS4_GLM53_POOL_DIM + d] *
                        (double)__half2float(key[(uint64_t)p * DS4_GLM53_POOL_DIM + d]);
                }
                total += fmax(dot, 0.0) * weights[(uint64_t)t * HEADS + h];
            }
            const double error = fabs(actual - total * SCALE);
            CHECK(error <= CPU_MAX_ABS);
            worst = fmax(worst, error);
        }
    }
    printf("CPU samples=48 max_abs=%.9g bound=%.9g PASS\n", worst, CPU_MAX_ABS);
}

static void launch(ds4_gpu_tensor &out, const ds4_gpu_tensor &q,
        const ds4_gpu_tensor &weights, const ds4_gpu_tensor &key,
        uint32_t pools, uint32_t rows, uint32_t pos0) {
    CHECK(ds4_gpu_glm53_pool_score(&out, &q, &weights, &key,
        pools, rows, pos0, HEADS, SCALE));
}

int main(int argc, char **argv) {
    CHECK(argc == 5);
    const uint32_t rows = (uint32_t)strtoul(argv[1], NULL, 10);
    const uint32_t pools = (uint32_t)strtoul(argv[2], NULL, 10);
    CHECK(rows == 1u || rows == 128u || rows == 2048u);
    CHECK(pools == 2048u || pools == 16384u || pools == 262144u);
    const uint32_t pos0 = pools * DS4_GLM53_POOL_SIZE - (rows == 1u ? 0u : rows);
    Run mode = NUMERIC;
    if (!strcmp(argv[3], "time")) { mode = TIMING; }
    else if (!strcmp(argv[3], "ncu")) { mode = PROFILE; }
    else { CHECK(!strcmp(argv[3], "numeric")); }
    std::vector<__half> key((uint64_t)pools * DS4_GLM53_POOL_DIM);
    std::vector<float> q((uint64_t)rows * HEADS * DS4_GLM53_POOL_DIM);
    std::vector<float> weights((uint64_t)rows * HEADS), out((uint64_t)rows * pools + GUARD, SENTINEL);
    uint32_t state = 0x58137426u;
    for (auto &x : key) { x = __float2half_rn((float)((int)(random_step(&state) % 769u) - 384) / 512.0f); }
    for (auto &x : q) { x = (float)((int)(random_step(&state) % 1025u) - 512) / 1024.0f; }
    for (auto &x : weights) { x = (float)((int)(random_step(&state) % 2049u) - 1024) / 2048.0f; }
    ds4_gpu_tensor dk = upload(key.data(), key.size() * sizeof(__half));
    ds4_gpu_tensor dq = upload(q.data(), q.size() * sizeof(float));
    ds4_gpu_tensor dw = upload(weights.data(), weights.size() * sizeof(float));
    ds4_gpu_tensor dout = upload(out.data(), out.size() * sizeof(float));
    printf("synthetic=true rows=%u heads=%u pool_dim=%u pools=%u pos0=%u scale=%.9g bytes=%llu\n",
        rows, HEADS, DS4_GLM53_POOL_DIM, pools, pos0, SCALE,
        (unsigned long long)(dk.bytes + dq.bytes + dw.bytes + dout.bytes));
    for (int i = 0; i < WARM_CALLS; i++) { launch(dout, dq, dw, dk, pools, rows, pos0); }
    CHECK(cuda_ok(cudaDeviceSynchronize(), "warm"));
    if (mode == PROFILE) {
        CHECK(cuda_ok(cudaProfilerStart(), "profile start"));
        launch(dout, dq, dw, dk, pools, rows, pos0);
        CHECK(cuda_ok(cudaDeviceSynchronize(), "profile sync"));
        CHECK(cuda_ok(cudaProfilerStop(), "profile stop"));
    }
    if (mode == TIMING) {
        cudaEvent_t start, stop;
        CHECK(cuda_ok(cudaEventCreate(&start), "event start"));
        CHECK(cuda_ok(cudaEventCreate(&stop), "event stop"));
        for (int i = 0; i < TIME_CALLS; i++) {
            CHECK(cuda_ok(cudaEventRecord(start), "record start"));
            launch(dout, dq, dw, dk, pools, rows, pos0);
            CHECK(cuda_ok(cudaEventRecord(stop), "record stop"));
            CHECK(cuda_ok(cudaEventSynchronize(stop), "event sync"));
            float ms = 0.0f;
            CHECK(cuda_ok(cudaEventElapsedTime(&ms, start, stop), "elapsed"));
            printf("call=%d ms=%.9g\n", i, ms);
        }
        CHECK(cuda_ok(cudaEventDestroy(start), "destroy start"));
        CHECK(cuda_ok(cudaEventDestroy(stop), "destroy stop"));
    }
    CHECK(cuda_ok(cudaMemcpy(out.data(), dout.ptr, (size_t)dout.bytes, cudaMemcpyDeviceToHost), "read"));
    uint64_t masked = 0;
    for (uint32_t t = 0; t < rows; t++) {
        for (uint32_t p = 0; p < pools; p++) {
            const float value = out[(uint64_t)t * pools + p];
            if (p >= (pos0 + t + 1u) / DS4_GLM53_POOL_SIZE) { CHECK(value == -INFINITY); masked++; continue; }
            CHECK(isfinite(value) && value != SENTINEL);
        }
    }
    cpu_check(rows, pools, pos0, key, q, weights, out);
    for (uint64_t i = (uint64_t)rows * pools; i < out.size(); i++) {
        CHECK(out[i] == SENTINEL);
    }
    // Refusals must leave a valid frontier untouched; no captured position arguments.
    CHECK(!ds4_gpu_glm53_pool_score(&dout, &dq, &dw, &dk, pools, rows, UINT32_MAX, HEADS, SCALE));
    CHECK(!ds4_gpu_glm53_pool_score(&dout, &dq, &dw, &dk, pools, rows, pos0, HEADS, -SCALE));
    ds4_gpu_tensor short_key = dk;
    short_key.bytes--;
    CHECK(!ds4_gpu_glm53_pool_score(&dout, &dq, &dw, &short_key, pools, rows, pos0, HEADS, SCALE));
    capture = 1;
    CHECK(!ds4_gpu_glm53_pool_score(&dout, &dq, &dw, &dk, pools, rows, pos0, HEADS, SCALE));
    capture = 0;
    FILE *file = fopen(argv[4], "wb");
    CHECK(file);
    const size_t count = out.size() - GUARD;
    CHECK(fwrite(out.data(), sizeof(float), count, file) == count);
    CHECK(fclose(file) == 0);
    printf("all_checked=%zu masked=%llu guard=%u output=%s PASS\n", count, (unsigned long long)masked, GUARD, argv[4]);
    CHECK(cuda_ok(cudaFree(dout.ptr), "out free"));
    CHECK(cuda_ok(cudaFree(dw.ptr), "weights free"));
    CHECK(cuda_ok(cudaFree(dq.ptr), "q free"));
    CHECK(cuda_ok(cudaFree(dk.ptr), "key free"));
    return 0;
}
