/* Kernel-level A/B for Bonsai's dense PQ2_0 matmul: the family-owned INT8
 * tensor-core file (cuda/bonsai_pq2_gemm.cuh) against the vendored PQ2_0
 * MMQ/MMVQ pair, on the model's real shapes and row counts.
 *
 * Model-free: the weights are random blocks in device memory, so the number
 * measures the kernel, not the artifact.  Every run also compares the two
 * outputs (relative L2) so a speed reading is never taken from a wrong result.
 *
 * Build: make bonsai-dense-bench
 * Run:   ./tests/bonsai_dense_bench [iterations] */

#include "ds4_gpu.h"
#include "cuda/mmq/ds4_mmq.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

/* Defined by cuda/bonsai_pq2_gemm.cuh (extern "C", so this matches). */
extern "C" int ds4_gpu_bonsai_pq2_matmul(const void *w, const float *x,
                                         int8_t *xq, float *xs, float *out,
                                         uint32_t out_dim, uint32_t in_dim,
                                         uint32_t rows, cudaStream_t stream);

/* Defined in ds4.c under DS4_TEST_HOOKS: the double-precision row dot. */
extern "C" int ds4_test_pq2_0_ref_matvec(const void *blocks, uint64_t out_dim,
                                         uint64_t in_dim, const float *x,
                                         float *out);

namespace {

constexpr uint32_t kBlockValues = 128;
constexpr uint32_t kBlockBytes  = 34;

struct shape {
    uint32_t m;   /* output rows */
    uint32_t k;   /* input columns (multiple of 128) */
    const char *name;
};

const shape kShapes[] = {
    {17408, 5120, "ffn_gate/up   "},
    { 5120, 17408, "ffn_down      "},
    {10240, 5120, "attn_qkv      "},
    { 6144, 5120, "attn_gate     "},
    { 5120,  6144, "ssm_out       "},
    {248320, 5120, "output head   "},
};

const uint32_t kRows[] = {1, 8, 9, 64, 512};

void fill_blocks(std::vector<uint8_t> &w) {
    std::mt19937 rng(1234);
    for (size_t b = 0; b < w.size(); b += kBlockBytes) {
        const uint32_t scale = 0x3c003c00u + (rng() & 0x7u);   /* ~1.0 in fp16 */
        std::memcpy(&w[b], &scale, 2);
        for (uint32_t i = 2; i < kBlockBytes; i++) {
            w[b + i] = (uint8_t) (rng() & 0xFFu);
        }
    }
}

float rel_l2(const std::vector<float> &a, const std::vector<float> &b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double) a[i] - (double) b[i];
        num += d * d;
        den += (double) b[i] * (double) b[i];
    }
    return den > 0.0 ? (float) std::sqrt(num / den) : 0.0f;
}

int run_one(const shape &s, uint32_t rows, int iters) {
    const uint64_t w_blocks = (uint64_t) s.m * (s.k / kBlockValues);
    const uint64_t w_bytes  = w_blocks * kBlockBytes;
    const uint64_t x_bytes  = (uint64_t) rows * s.k * sizeof(float);
    const uint64_t o_bytes  = (uint64_t) rows * s.m * sizeof(float);

    std::vector<uint8_t> host_w((size_t) w_bytes);
    std::vector<float>   host_x((size_t) rows * s.k);
    fill_blocks(host_w);
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (float &v : host_x) {
        v = dist(rng);
    }

    const uint64_t q_bytes = (uint64_t) rows * s.k;
    const uint64_t s_bytes = (uint64_t) rows * (s.k / kBlockValues) * sizeof(float);
    void *d_w = nullptr, *d_x = nullptr, *d_a = nullptr, *d_b = nullptr;
    int8_t *d_xq = nullptr;
    float *d_xs = nullptr;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess ||
        cudaMalloc(&d_w, w_bytes) != cudaSuccess ||
        cudaMalloc(&d_x, x_bytes) != cudaSuccess ||
        cudaMalloc(&d_xq, q_bytes) != cudaSuccess ||
        cudaMalloc(&d_xs, s_bytes) != cudaSuccess ||
        cudaMalloc(&d_a, o_bytes) != cudaSuccess ||
        cudaMalloc(&d_b, o_bytes) != cudaSuccess) {
        std::fprintf(stderr, "%s N=%u: allocation failed\n", s.name, rows);
        return 1;
    }
    cudaMemcpyAsync(d_w, host_w.data(), w_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_x, host_x.data(), x_bytes, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    /* Both arms write the same [rows][m] row-major layout. */
    const int use_gemv = rows <= 8;

    auto launch_family = [&]() {
        return ds4_gpu_bonsai_pq2_matmul(d_w, (const float *) d_x, d_xq, d_xs,
                                         (float *) d_a, s.m, s.k, rows, stream);
    };
    auto launch_mmq = [&]() {
        return use_gemv
            ? ds4_mmq_pq2_0_dense_vec(d_w, (const float *) d_x, (float *) d_b,
                                      (int) s.m, (int) rows, (int) s.k, stream)
            : ds4_mmq_pq2_0_dense(d_w, (const float *) d_x, (float *) d_b,
                                  (int) s.m, (int) rows, (int) s.k, stream);
    };

    if (!launch_family() || launch_mmq() != 0) {
        std::fprintf(stderr, "%s N=%u: launch refused\n", s.name, rows);
        return 1;
    }
    cudaStreamSynchronize(stream);

    std::vector<float> out_a((size_t) rows * s.m), out_b((size_t) rows * s.m);
    cudaMemcpy(out_a.data(), d_a, o_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_b.data(), d_b, o_bytes, cudaMemcpyDeviceToHost);
    const float diff = rel_l2(out_a, out_b);

    /* The GEMM tiles cover four 16-row groups; probe one row from each so a
     * wrong warp group cannot hide behind its neighbours. */
    const uint32_t probes[] = {0u, 16u, 32u, 48u};
    double num_a = 0.0, num_b = 0.0, den = 0.0;
    std::vector<float> ref((size_t) s.m);
    std::vector<float> got((size_t) s.m);
    for (uint32_t p : probes) {
        if (p >= rows) {
            continue;
        }
        ds4_test_pq2_0_ref_matvec(host_w.data(), s.m, s.k,
                                  host_x.data() + (size_t) p * s.k, ref.data());
        for (int arm = 0; arm < 2; arm++) {
            std::memcpy(got.data(), (arm ? out_b : out_a).data() + (size_t) p * s.m,
                        (size_t) s.m * sizeof(float));
            const float e = rel_l2(got, ref);
            (arm ? num_b : num_a) += (double) e * e;
            if (arm == 0) {
                den += 1.0;
            }
        }
    }
    const double ref_a = den > 0.0 ? std::sqrt(num_a / den) : 0.0;
    const double ref_b = den > 0.0 ? std::sqrt(num_b / den) : 0.0;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    auto time_arm = [&](bool family) {
        for (int i = 0; i < 3; i++) {
            (void) (family ? launch_family() : launch_mmq());
        }
        cudaStreamSynchronize(stream);
        cudaEventRecord(t0, stream);
        for (int i = 0; i < iters; i++) {
            (void) (family ? launch_family() : launch_mmq());
        }
        cudaEventRecord(t1, stream);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        return ms / (float) iters;
    };

    const float ms_family = time_arm(true);
    const float ms_mmq    = time_arm(false);

    const double flops = 2.0 * (double) s.m * s.k * rows;
    const double bytes = (double) w_bytes;
    std::printf("%s N=%-4u  family %7.3f ms %7.1f TF  %6.1f GB/s | "
                "mmq %7.3f ms %7.1f TF  %6.1f GB/s | %+6.1f%% | vs ref %.1e/%.1e | fam-vs-mmq %.1e\n",
                s.name, rows, ms_family, flops / (ms_family * 1e9),
                bytes / (ms_family * 1e6), ms_mmq, flops / (ms_mmq * 1e9),
                bytes / (ms_mmq * 1e6),
                100.0 * (ms_mmq - ms_family) / ms_mmq, ref_a, ref_b, diff);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_xq);
    cudaFree(d_xs);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaStreamDestroy(stream);
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 20;
    if (ds4_mmq_init(0) != 0) {
        std::fprintf(stderr, "ds4_mmq_init failed\n");
        return 1;
    }
    int rc = 0;
    std::printf("shapes: %zu, iterations: %d\n", sizeof(kShapes) / sizeof(kShapes[0]), iters);
    for (const shape &s : kShapes) {
        for (uint32_t rows : kRows) {
            /* The 248k-row head at 512 tokens would need a 508 MB output. */
            if (s.m > 100000u && rows > 64u) {
                continue;
            }
            rc |= run_one(s, rows, iters);
        }
    }
    return rc;
}
