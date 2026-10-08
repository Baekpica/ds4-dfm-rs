/* Production kernels, measured 8K geometry, synthetic pooled routing. */
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
enum { GLM53_COMPACT_THREADS = 256, GLM53_COMPACT_MAX_DIM = 1024 };
static int ds4_capture_active(void) { return 0; }
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
        latent_dim && latent_dim <= GLM53_COMPACT_MAX_DIM && latent_dim % 32u == 0u &&
        head_dim && head_dim <= GLM53_COMPACT_MAX_DIM && head_dim % 32u == 0u;
}
#include "../cuda/glm53_low_attn.cuh"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "attention FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

enum { HEADS = 64, DIM = 512, HEAD_DIM = 256, PREFILL_POS = 8064,
       DECODE_POS = 8192, WARM_CALLS = 3, TIME_CALLS = 12 };
enum Run { NUMERIC, TIMING, PROFILE };
static constexpr float SENTINEL = 12345.0f;
static constexpr double CPU_MAX_ABS = 3.0e-4;

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

static void inputs(uint32_t rows, uint32_t pos0, std::vector<__half> &kv,
        std::vector<float> &q, std::vector<uint32_t> &picked) {
    uint32_t state = 0x58137426u;
    for (auto &x : kv) {
        x = __float2half_rn((float)((int)(random_step(&state) % 769u) - 384) / 512.0f);
    }
    for (auto &x : q) {
        x = (float)((int)(random_step(&state) % 1025u) - 512) / 1024.0f;
    }
    for (uint32_t t = 0; t < rows; t++) {
        const uint32_t visible = pos0 + t + 1u;
        const uint32_t pools = visible / DS4_GLM53_POOL_SIZE;
        std::vector<uint32_t> order(pools);
        for (uint32_t i = 0; i < pools; i++) { order[i] = i; }
        for (uint32_t i = pools; i > 1u; i--) {
            const uint32_t j = random_step(&state) % i;
            const uint32_t tmp = order[i - 1u];
            order[i - 1u] = order[j];
            order[j] = tmp;
        }
        // Ranked pools expand to four adjacent rows; the partial tail stays causal.
        for (uint32_t s = 0; s < DS4_GLM53_INDEX_TOPK; s++) {
            picked[(uint64_t)t * DS4_GLM53_MAX_SELECTED + s] =
                order[s / DS4_GLM53_POOL_SIZE] * DS4_GLM53_POOL_SIZE + s % DS4_GLM53_POOL_SIZE;
        }
        for (uint32_t s = DS4_GLM53_INDEX_TOPK; s < DS4_GLM53_MAX_SELECTED; s++) {
            const uint32_t tail = s - DS4_GLM53_INDEX_TOPK;
            picked[(uint64_t)t * DS4_GLM53_MAX_SELECTED + s] = tail < visible % DS4_GLM53_POOL_SIZE
                ? visible - visible % DS4_GLM53_POOL_SIZE + tail : UINT32_MAX;
        }
    }
}

static void cpu_check(uint32_t rows, uint32_t pos0, const std::vector<__half> &kv,
        const std::vector<float> &q, const std::vector<uint32_t> &picked,
        const std::vector<float> &out) {
    const uint32_t tokens[] = {0u, rows / 2u, rows - 1u};
    const uint32_t heads[] = {0u, 17u, 42u, HEADS - 1u};
    const uint32_t dims[] = {0u, 127u, 256u, DIM - 1u};
    double worst = 0.0;
    for (uint32_t t : tokens) {
        for (uint32_t h : heads) {
            std::vector<double> scores(DS4_GLM53_MAX_SELECTED);
            double maximum = -INFINITY;
            for (uint32_t s = 0; s < DS4_GLM53_MAX_SELECTED; s++) {
                const uint32_t r = picked[(uint64_t)t * DS4_GLM53_MAX_SELECTED + s];
                double dot = 0.0;
                if (r < pos0 + t + 1u) {
                    for (uint32_t d = 0; d < DIM; d++) {
                        dot += (double)q[((uint64_t)t * HEADS + h) * DIM + d] *
                            (double)__half2float(kv[(uint64_t)r * DIM + d]);
                    }
                }
                scores[s] = r < pos0 + t + 1u ? dot / sqrt((double)HEAD_DIM) : -INFINITY;
                maximum = fmax(maximum, scores[s]);
            }
            double denominator = 0.0;
            for (auto &x : scores) { x = exp(x - maximum); denominator += x; }
            for (uint32_t d : dims) {
                double expected = 0.0;
                for (uint32_t s = 0; s < DS4_GLM53_MAX_SELECTED; s++) {
                    const uint32_t r = picked[(uint64_t)t * DS4_GLM53_MAX_SELECTED + s];
                    if (r >= pos0 + t + 1u) { continue; }
                    expected += scores[s] / denominator * (double)__half2float(kv[(uint64_t)r * DIM + d]);
                }
                const double err = fabs(out[((uint64_t)t * HEADS + h) * DIM + d] - expected);
                CHECK(err <= CPU_MAX_ABS);
                worst = fmax(worst, err);
            }
        }
    }
    printf("CPU samples=48 max_abs=%.9g bound=%.9g PASS\n", worst, CPU_MAX_ABS);
}

static void launch(ds4_gpu_tensor &out, const ds4_gpu_tensor &q,
        const ds4_gpu_tensor &kv, const ds4_gpu_tensor &picked,
        uint32_t rows, uint32_t pos0) {
    CHECK(ds4_gpu_glm53_attn_low(&out, &q, &kv, &picked, DS4_GLM53_MAX_SELECTED,
        rows, pos0, pos0 + rows, HEADS, DIM, HEAD_DIM));
}

int main(int argc, char **argv) {
    CHECK(argc == 4);
    const uint32_t rows = (uint32_t)strtoul(argv[1], NULL, 10);
    CHECK(rows == 1u || rows == 128u || rows == 2048u);
    const uint32_t pos0 = rows == 1u ? DECODE_POS : DECODE_POS - rows;
    Run mode = NUMERIC;
    if (strcmp(argv[2], "time") == 0) { mode = TIMING; }
    else if (strcmp(argv[2], "ncu") == 0) { mode = PROFILE; }
    else { CHECK(strcmp(argv[2], "numeric") == 0); }
    std::vector<__half> kv((uint64_t)(pos0 + rows) * DIM);
    std::vector<float> q((uint64_t)rows * HEADS * DIM), out(q.size(), SENTINEL);
    std::vector<uint32_t> picked((uint64_t)rows * DS4_GLM53_MAX_SELECTED);
    inputs(rows, pos0, kv, q, picked);
    ds4_gpu_tensor dk = upload(kv.data(), kv.size() * sizeof(__half));
    ds4_gpu_tensor dq = upload(q.data(), q.size() * sizeof(float));
    ds4_gpu_tensor dp = upload(picked.data(), picked.size() * sizeof(uint32_t));
    ds4_gpu_tensor dout = upload(out.data(), out.size() * sizeof(float));
    printf("synthetic=true rows=%u heads=%u latent=%u head_dim=%u pos0=%u selected=%u bytes=%llu\n",
        rows, HEADS, DIM, HEAD_DIM, pos0, DS4_GLM53_MAX_SELECTED,
        (unsigned long long)(dk.bytes + dq.bytes + dp.bytes + dout.bytes));
    for (int i = 0; i < WARM_CALLS; i++) { launch(dout, dq, dk, dp, rows, pos0); }
    CHECK(cuda_ok(cudaDeviceSynchronize(), "warm"));
    if (mode == PROFILE) {
        CHECK(cuda_ok(cudaProfilerStart(), "profile start"));
        launch(dout, dq, dk, dp, rows, pos0);
        CHECK(cuda_ok(cudaDeviceSynchronize(), "profile sync"));
        CHECK(cuda_ok(cudaProfilerStop(), "profile stop"));
    }
    if (mode == TIMING) {
        cudaEvent_t start, stop;
        CHECK(cuda_ok(cudaEventCreate(&start), "event start"));
        CHECK(cuda_ok(cudaEventCreate(&stop), "event stop"));
        for (int i = 0; i < TIME_CALLS; i++) {
            CHECK(cuda_ok(cudaEventRecord(start), "record start"));
            launch(dout, dq, dk, dp, rows, pos0);
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
    for (float x : out) { CHECK(isfinite(x) && x != SENTINEL); }
    cpu_check(rows, pos0, kv, q, picked, out);
    FILE *file = fopen(argv[3], "wb");
    CHECK(file);
    CHECK(fwrite(out.data(), sizeof(float), out.size(), file) == out.size());
    CHECK(fclose(file) == 0);
    printf("all_finite=%zu output=%s PASS\n", out.size(), argv[3]);
    CHECK(cuda_ok(cudaFree(dout.ptr), "out free"));
    CHECK(cuda_ok(cudaFree(dp.ptr), "picked free"));
    CHECK(cuda_ok(cudaFree(dq.ptr), "q free"));
    CHECK(cuda_ok(cudaFree(dk.ptr), "kv free"));
    return 0;
}
