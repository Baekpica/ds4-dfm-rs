// Drives ds4_gpu_mimo2_attention for sliding-window prefill.
// Default rows>=32 uses the tensor-core tile. DS4_MIMO2_NO_SWA_HMMA=1
// restores the walk. HMMA changes summation order; its FP64 error must
// stay within max(2e-5, 1.25x) of the walk on the sampled rows.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool m2_hmma_available = false;
static int cuda_ok(cudaError_t err, const char *) { return err == cudaSuccess; }
static inline cudaStream_t ds4_current_stream(void) { return 0; }
static inline int ds4_capture_active(void) { return 0; }
static const char *cuda_model_range_ptr(const void *map, uint64_t offset, uint64_t, const char *) {
    return map ? (const char *)map + offset : nullptr;
}
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; int owner; int memc; };
#include "../ds4_mimo2_gpu.cuh"

enum { HEADS = 64, KEY = 192, VALUE = 128, KV = 8, ROWS = 65, WINDOW = 128,
       STRIDE = KV * (KEY + VALUE) };

static double sampled_fp64(const std::vector<float> &got, const std::vector<float> &q,
                           const std::vector<__half> &cache, const std::vector<unsigned> &positions,
                           unsigned capacity) {
    double error = 0;
    for (unsigned row : {0u, 31u, 32u, 64u}) {
        const unsigned pos = positions[row];
        const unsigned first = pos + 1 > WINDOW ? pos + 1 - WINDOW : 0;
        for (unsigned h : {0u, 7u, 8u, 63u}) {
            const unsigned kh = h / (HEADS / KV);
            std::vector<double> score(pos - first + 1);
            double maximum = -INFINITY;
            for (unsigned p = first; p <= pos; p++) {
                double dot = 0;
                const __half *key = cache.data() + (uint64_t)(p % capacity) * STRIDE + kh * KEY;
                for (unsigned d = 0; d < KEY; d++) {
                    dot += (double)q[(row * HEADS + h) * KEY + d] * __half2float(key[d]);
                }
                score[p - first] = dot / sqrt((double)KEY);
                maximum = std::max(maximum, score[p - first]);
            }
            double denominator = 0;
            for (double &s : score) {
                s = exp(s - maximum);
                denominator += s;
            }
            for (unsigned d = 0; d < VALUE; d++) {
                double value = 0;
                for (unsigned p = first; p <= pos; p++) {
                    const __half *col = cache.data() + (uint64_t)(p % capacity) * STRIDE +
                        KV * KEY + kh * VALUE;
                    value += score[p - first] * __half2float(col[d]);
                }
                error = std::max(error, fabs(got[(row * HEADS + h) * VALUE + d] - value / denominator));
            }
        }
    }
    return error;
}

static int launch(float *dout, float *dq, __half *dc, unsigned *dp, unsigned rows,
                  unsigned capacity, unsigned pos0, int kill, std::vector<float> &host) {
    if (kill) { setenv("DS4_MIMO2_NO_SWA_HMMA", "1", 1); }
    else { unsetenv("DS4_MIMO2_NO_SWA_HMMA"); }
    unsetenv("DS4_MIMO2_SWA_HMMA");
    m2_attn_path = -1;
    ds4_gpu_tensor out{dout, host.size() * sizeof(float), 0, 0};
    ds4_gpu_tensor query{dq, (uint64_t)rows * HEADS * KEY * sizeof(float), 0, 0};
    ds4_gpu_tensor kv{dc, (uint64_t)capacity * STRIDE * sizeof(__half), 0, 0};
    ds4_gpu_tensor pos{dp, (uint64_t)rows * sizeof(unsigned), 0, 0};
    const int rc = ds4_gpu_mimo2_attention(
        &out, &query, &kv, &pos, nullptr, 0, 0, KV, capacity, rows, WINDOW, pos0);
    if (rc != 1 || cudaMemcpy(host.data(), dout, host.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 0;
    }
    return 1;
}

static int one_case(unsigned start) {
    const unsigned capacity = start + ROWS;
    std::vector<__half> cache((uint64_t)capacity * STRIDE);
    std::vector<float> q((uint64_t)ROWS * HEADS * KEY);
    std::vector<unsigned> positions(ROWS);
    std::vector<float> walked(ROWS * HEADS * VALUE), scored(walked.size());
    for (unsigned row = 0; row < ROWS; row++) { positions[row] = start + row; }
    for (size_t i = 0; i < q.size(); i++) { q[i] = sinf((float)i * 0.013f); }
    for (unsigned p = 0; p < capacity; p++) {
        for (unsigned d = 0; d < STRIDE; d++) {
            cache[(uint64_t)p * STRIDE + d] = __float2half_rn(sinf(p * 0.047f + d * 0.017f));
        }
    }
    float *dq = nullptr, *dout = nullptr;
    __half *dc = nullptr;
    unsigned *dp = nullptr;
    if (cudaMalloc(&dq, q.size() * sizeof(float)) || cudaMalloc(&dout, walked.size() * sizeof(float)) ||
        cudaMalloc(&dc, cache.size() * sizeof(__half)) || cudaMalloc(&dp, positions.size() * sizeof(unsigned))) {
        return 1;
    }
    cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, cache.data(), cache.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dp, positions.data(), positions.size() * sizeof(unsigned), cudaMemcpyHostToDevice);
    if (!launch(dout, dq, dc, dp, ROWS, capacity, start, 1, walked) || m2_attn_path != 0) {
        fprintf(stderr, "kill switch missed the walk start=%u path=%d\n", start, m2_attn_path);
        return 2;
    }
    if (!launch(dout, dq, dc, dp, ROWS, capacity, start, 0, scored) || m2_attn_path != M2_PATH_SWA_HMMA) {
        fprintf(stderr, "default missed SWA HMMA start=%u path=%d\n", start, m2_attn_path);
        return 3;
    }
    const double walk_fp64 = sampled_fp64(walked, q, cache, positions, capacity);
    const double hmma_fp64 = sampled_fp64(scored, q, cache, positions, capacity);
    const double limit = std::max(2e-5, walk_fp64 * 1.25);
    printf("swa start=%u path=%d walk_fp64=%.9g hmma_fp64=%.9g limit=%.9g\n",
           start, m2_attn_path, walk_fp64, hmma_fp64, limit);
    cudaFree(dq); cudaFree(dout); cudaFree(dc); cudaFree(dp);
    if (!std::isfinite(hmma_fp64) || hmma_fp64 > limit) { return 4; }
    return 0;
}

int main() {
    m2_hmma_init();
    if (!m2_hmma_available) {
        fprintf(stderr, "tensor-core prefill is not supported on this device\n");
        return 1;
    }
    for (unsigned start : {0u, 200u, 4096u}) {
        const int rc = one_case(start);
        if (rc) { return rc; }
    }
    puts("swa_hmma=fp64_bound kill=walk");
    return 0;
}
