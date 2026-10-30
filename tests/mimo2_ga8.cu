// Full-attention FP8 prefill: mimo2_ga8::prefill against the scalar-staged
// mimo2_hmma::prefill<Fp8> and an FP64 reference over the decoded rows on
// sampled (row, head) pairs, then both timed.
// Usage: mimo2_ga8 [frontier [rows [reps]]]; no argument runs the checks.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
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

enum { HEADS = 64, KEY = 192, VALUE = 128, KV = 4 };

// Normal-ish values; every eighth row is twice as large so head scales differ.
__global__ static void fill(float *x, uint64_t n, unsigned width, unsigned seed) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    unsigned h = (unsigned)i * 2654435761u ^ seed;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    const float u1 = ((h & 0xffffu) + 0.5f) / 65536.0f, u2 = ((h >> 16) + 0.5f) / 65536.0f;
    const float g = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    x[i] = g * ((i / width) % 8 == 0 ? 2.0f : 1.0f);
}

static void random(float *x, uint64_t n, unsigned width, unsigned seed) {
    fill<<<(unsigned)((n + 255) / 256), 256>>>(x, n, width, seed);
}

static float e4m3(uint8_t code) {
    __nv_fp8_e4m3 v;
    v.__x = code;
    return (float)v;
}

// FP64 attention over the stored codes times their f16 scales.
static double reference_error(const std::vector<float> &got, const std::vector<float> &q,
                              const std::vector<uint8_t> &cache, unsigned frontier, unsigned rows,
                              bool all_heads = false) {
    double error = 0;
    for (unsigned row : {0u, 1u, 63u, 64u, rows / 2, rows - 1}) {
        if (row >= rows) { continue; }
        const unsigned pos = frontier + row;
        for (unsigned h = 0; h < HEADS; h += all_heads ? 1 : (h == 0 ? 1 : h == 1 ? 16 : h == 17 ? 29 : h == 46 ? 17 : 64)) {
            const unsigned kh = h / (HEADS / KV);
            std::vector<double> score(pos + 1);
            double maximum = -INFINITY;
            for (unsigned p = 0; p <= pos; p++) {
                const uint8_t *r = cache.data() + (uint64_t)p * M2_FP8_ROW;
                const double s = __half2float(reinterpret_cast<const __half *>(r + M2_FP8_SCALES)[kh]);
                double dot = 0;
                for (unsigned d = 0; d < KEY; d++) {
                    dot += (double)q[((uint64_t)row * HEADS + h) * KEY + d] * e4m3(r[kh * KEY + d]) * s;
                }
                score[p] = dot / sqrt((double)KEY);
                maximum = std::max(maximum, score[p]);
            }
            double denominator = 0;
            for (double &s : score) {
                s = exp(s - maximum);
                denominator += s;
            }
            for (unsigned d = 0; d < VALUE; d++) {
                double value = 0;
                for (unsigned p = 0; p <= pos; p++) {
                    const uint8_t *r = cache.data() + (uint64_t)p * M2_FP8_ROW;
                    const double s = __half2float(reinterpret_cast<const __half *>(r + M2_FP8_SCALES)[KV + kh]);
                    value += score[p] * e4m3(r[M2_FP8_KEYS + kh * VALUE + d]) * s;
                }
                const double want = value / denominator;
                error = std::max(error, fabs(got[((uint64_t)row * HEADS + h) * VALUE + d] - want));
            }
        }
    }
    return error;
}

static void launch(int ga8, float *out, const float *q, const uint8_t *cache,
                   const unsigned *positions, unsigned rows, unsigned capacity) {
    if (ga8) {
        mimo2_ga8::prefill<<<dim3((rows + mimo2_ga8::ROWS - 1) / mimo2_ga8::ROWS, HEADS / 2),
            mimo2_ga8::THREADS, mimo2_ga8::SMEM>>>(out, q, cache, positions, rows, capacity);
        return;
    }
    mimo2_hmma::prefill<mimo2_hmma::Fp8><<<dim3((rows + mimo2_hmma::TQ - 1) / mimo2_hmma::TQ, HEADS),
        32 * mimo2_hmma::WARPS>>>(out, q, cache, nullptr, positions, rows, KV, capacity);
}

// Returns 0 when ga8 stays within the FP64 bound; prints both timings.
static int run(unsigned frontier, unsigned rows, int reps, int check) {
    const unsigned capacity = frontier + rows;
    float *q, *k, *v, *out;
    uint8_t *cache;
    unsigned *positions;
    const uint64_t nq = (uint64_t)rows * HEADS * KEY, no = (uint64_t)rows * HEADS * VALUE;
    if (cudaMalloc(&q, nq * 4) || cudaMalloc(&out, no * 4) ||
        cudaMalloc(&k, (uint64_t)capacity * M2_FP8_KEYS * 4) ||
        cudaMalloc(&v, (uint64_t)capacity * KV * VALUE * 4) ||
        cudaMalloc(&cache, (uint64_t)capacity * M2_FP8_ROW) ||
        cudaMalloc(&positions, (uint64_t)capacity * 4)) { return 1; }
    std::vector<unsigned> all(capacity);
    for (unsigned i = 0; i < capacity; i++) { all[i] = i; }
    cudaMemcpy(positions, all.data(), capacity * 4, cudaMemcpyHostToDevice);
    random(q, nq, KEY, 1);
    random(k, (uint64_t)capacity * M2_FP8_KEYS, M2_FP8_KEYS, 2);
    random(v, (uint64_t)capacity * KV * VALUE, KV * VALUE, 3);
    ds4_gpu_tensor tc{cache, (uint64_t)capacity * M2_FP8_ROW, 0, 0};
    ds4_gpu_tensor tk{k, (uint64_t)capacity * M2_FP8_KEYS * 4, 0, 0};
    ds4_gpu_tensor tv{v, (uint64_t)capacity * KV * VALUE * 4, 0, 0};
    ds4_gpu_tensor tp{positions, (uint64_t)capacity * 4, 0, 0};
    if (!ds4_gpu_mimo2_kv_store_fp8(&tc, &tk, &tv, &tp, capacity, capacity)) { return 2; }
    const unsigned *prow = positions + frontier;

    std::vector<float> old(no), ga8(no);
    launch(0, out, q, cache, prow, rows, capacity);
    cudaMemcpy(old.data(), out, no * 4, cudaMemcpyDeviceToHost);
    cudaMemset(out, 0xff, no * 4);
    launch(1, out, q, cache, prow, rows, capacity);
    if (cudaMemcpy(ga8.data(), out, no * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "ga8 launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 3;
    }
    double num = 0, den = 0, worst = 0;
    for (uint64_t i = 0; i < no; i++) {
        const double d = (double)ga8[i] - old[i];
        num += d * d;
        den += (double)old[i] * old[i];
        worst = std::max(worst, fabs(d));
    }
    int rc = std::isfinite(num) ? 0 : 4;
    printf("frontier=%u rows=%u ga8_vs_old rel_rms=%.3g max_abs=%.3g", frontier, rows, sqrt(num / den), worst);
    if (check) {
        std::vector<float> hq(nq);
        std::vector<uint8_t> hc((uint64_t)capacity * M2_FP8_ROW);
        cudaMemcpy(hq.data(), q, nq * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hc.data(), cache, hc.size(), cudaMemcpyDeviceToHost);
        const double e_old = reference_error(old, hq, hc, frontier, rows);
        const double e_new = reference_error(ga8, hq, hc, frontier, rows);
        // Single f16 Q and P against the old kernel's two components.
        const double limit = std::max(2e-3, 4 * e_old);
        printf(" fp64 old=%.3g ga8=%.3g limit=%.3g", e_old, e_new, limit);
        if (!(e_new <= limit)) { rc = 5; }
    }
    for (int ga : {0, 1}) {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        launch(ga, out, q, cache, prow, rows, capacity);
        cudaEventRecord(a);
        for (int i = 0; i < reps; i++) { launch(ga, out, q, cache, prow, rows, capacity); }
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        const double flop = 2.0 * rows * HEADS * (KEY + VALUE) * (frontier + (rows + 1) / 2.0);
        printf(" %s=%.3fms(%.1fTF)", ga ? "ga8" : "old", ms / reps, flop / (ms / reps * 1e-3) / 1e12);
    }
    printf("\n");
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(out); cudaFree(cache); cudaFree(positions);
    return rc;
}

// Decode/verify widths: the dispatcher's ga8 decode against the split walk.
static int run_decode(unsigned frontier, unsigned rows, int reps, int check) {
    const unsigned capacity = frontier + rows;
    float *q, *k, *v, *out;
    uint8_t *cache;
    unsigned *positions;
    const uint64_t nq = (uint64_t)rows * HEADS * KEY, no = (uint64_t)rows * HEADS * VALUE;
    if (cudaMalloc(&q, nq * 4) || cudaMalloc(&out, no * 4) ||
        cudaMalloc(&k, (uint64_t)capacity * M2_FP8_KEYS * 4) ||
        cudaMalloc(&v, (uint64_t)capacity * KV * VALUE * 4) ||
        cudaMalloc(&cache, (uint64_t)capacity * M2_FP8_ROW) ||
        cudaMalloc(&positions, (uint64_t)capacity * 4)) { return 1; }
    std::vector<unsigned> all(capacity);
    for (unsigned i = 0; i < capacity; i++) { all[i] = i; }
    cudaMemcpy(positions, all.data(), capacity * 4, cudaMemcpyHostToDevice);
    random(q, nq, KEY, 7);
    random(k, (uint64_t)capacity * M2_FP8_KEYS, M2_FP8_KEYS, 8);
    random(v, (uint64_t)capacity * KV * VALUE, KV * VALUE, 9);
    ds4_gpu_tensor tc{cache, (uint64_t)capacity * M2_FP8_ROW, 0, 0};
    ds4_gpu_tensor tk{k, (uint64_t)capacity * M2_FP8_KEYS * 4, 0, 0};
    ds4_gpu_tensor tv{v, (uint64_t)capacity * KV * VALUE * 4, 0, 0};
    ds4_gpu_tensor tp{positions, (uint64_t)capacity * 4, 0, 0};
    if (!ds4_gpu_mimo2_kv_store_fp8(&tc, &tk, &tv, &tp, capacity, capacity)) { return 2; }
    ds4_gpu_tensor to{out, no * 4, 0, 0}, tq{q, nq * 4, 0, 0};
    ds4_gpu_tensor tr{positions + frontier, (uint64_t)rows * 4, 0, 0};
    auto launch = [&](bool ga8) {
        m2_ga8_available = ga8;
        const int ok = ds4_gpu_mimo2_attention_sinks(&to, &tq, &tc, &tr, nullptr, 0, KV, capacity, rows, 0, 0, 1);
        return ok && m2_attn_path == (ga8 ? M2_PATH_GA8 : 3);
    };
    std::vector<float> old(no), ga8(no);
    if (!launch(false)) { return 6; }
    cudaMemcpy(old.data(), out, no * 4, cudaMemcpyDeviceToHost);
    cudaMemset(out, 0xff, no * 4);
    if (!launch(true)) { return 7; }
    if (cudaMemcpy(ga8.data(), out, no * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { return 3; }
    double num = 0, den = 0, worst = 0;
    for (uint64_t i = 0; i < no; i++) {
        const double d = (double)ga8[i] - old[i];
        num += d * d;
        den += (double)old[i] * old[i];
        worst = std::max(worst, fabs(d));
    }
    int rc = std::isfinite(num) ? 0 : 4;
    printf("decode frontier=%u rows=%u ga8_vs_split rel_rms=%.3g max_abs=%.3g", frontier, rows, sqrt(num / den), worst);
    if (check) {
        std::vector<float> hq(nq);
        std::vector<uint8_t> hc((uint64_t)capacity * M2_FP8_ROW);
        cudaMemcpy(hq.data(), q, nq * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hc.data(), cache, hc.size(), cudaMemcpyDeviceToHost);
        const double e_old = reference_error(old, hq, hc, frontier, rows, true);
        const double e_new = reference_error(ga8, hq, hc, frontier, rows, true);
        const double limit = std::max(2e-3, 4 * e_old);
        printf(" fp64 split=%.3g ga8=%.3g limit=%.3g", e_old, e_new, limit);
        if (!(e_new <= limit)) { rc = 5; }
    }
    for (int ga : {0, 1}) {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        launch(ga);
        cudaEventRecord(a);
        for (int i = 0; i < reps; i++) { launch(ga); }
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        const double bytes = (double)(frontier + rows) * M2_FP8_ROW;
        printf(" %s=%.4fms(%.0fGB/s)", ga ? "ga8" : "split", ms / reps, bytes / (ms / reps * 1e-3) / 1e9);
    }
    printf("\n");
    m2_ga8_available = true;
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(out); cudaFree(cache); cudaFree(positions);
    return rc;
}

int main(int argc, char **argv) {
    m2_hmma_init();
    if (!m2_hmma_available || !mimo2_ga8::init()) {
        fprintf(stderr, "tensor-core prefill is not supported on this device\n");
        return 1;
    }
    if (argc > 1) {
        const unsigned rows = argc > 2 ? (unsigned)atoi(argv[2]) : 2048;
        const int reps = argc > 3 ? atoi(argv[3]) : 5;
        if (rows <= 8) { return run_decode((unsigned)atoi(argv[1]), rows, reps, 0); }
        return run((unsigned)atoi(argv[1]), rows, reps, 0);
    }
    for (unsigned frontier : {0u, 5u, 100u, 4100u}) {
        for (unsigned rows : {1u, 2u, 3u, 8u}) {
            const int rc = run_decode(frontier, rows, 3, 1);
            if (rc) { return rc; }
        }
    }
    for (unsigned frontier : {0u, 37u, 4096u}) {
        for (unsigned rows : {33u, 200u, 2048u}) {
            const int rc = run(frontier, rows, 1, 1);
            if (rc) { return rc; }
        }
    }
    puts("ga8=fp64_bound");
    return 0;
}
