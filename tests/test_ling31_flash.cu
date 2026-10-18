/* Ling-3.1 tensor-core latent attention against an FP64 reference of the
 * absorbed MLA: s = (q_abs·c + q_pe·k_pe) / sqrt(192) over keys 0..pos0+row,
 * softmax, O = Σ p c. Q and P round to BF16 in the kernel, so the gate is a
 * relative error bound, at single- and multi-split, partial-tile, decode and
 * verify shapes. */
#include "cuda/l31_flash.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static bool check(cudaError_t err, const char *what) {
    if (err == cudaSuccess) { return true; }
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    return false;
}

static bool run(uint32_t rows, uint32_t pos0, std::mt19937 &rng) {
    constexpr uint32_t H = GLMVQ_FA_HEADS, L = GLMVQ_FA_LATENT, R = L31_FA_ROPE, K = L31_FA_KEY;
    const uint32_t keys = pos0 + rows;
    std::normal_distribution<float> unit(0.0f, 1.0f);
    std::vector<float> q((uint64_t)rows * H * K), q_abs((uint64_t)rows * H * L);
    std::vector<__nv_bfloat16> latent((uint64_t)keys * L), k_pe((uint64_t)keys * R);
    for (auto &v : q) { v = unit(rng); }
    for (auto &v : q_abs) { v = 0.5f * unit(rng); }
    for (auto &v : latent) { v = __float2bfloat16_rn(0.5f * unit(rng)); }
    for (auto &v : k_pe) { v = __float2bfloat16_rn(unit(rng)); }
    const float scale = 1.0f / sqrtf((float)K);

    float *d_q = nullptr, *d_abs = nullptr, *d_out = nullptr;
    __nv_bfloat16 *d_lat = nullptr, *d_pe = nullptr;
    void *d_scratch = nullptr;
    const uint64_t out_n = (uint64_t)rows * H * L;
    bool ok = check(cudaMalloc(&d_q, q.size() * 4), "alloc") && check(cudaMalloc(&d_abs, q_abs.size() * 4), "alloc") &&
        check(cudaMalloc(&d_out, out_n * 4), "alloc") && check(cudaMalloc(&d_lat, latent.size() * 2), "alloc") &&
        check(cudaMalloc(&d_pe, k_pe.size() * 2), "alloc") && check(cudaMalloc(&d_scratch, l31_fa_scratch_bytes()), "alloc") &&
        check(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "copy") &&
        check(cudaMemcpy(d_abs, q_abs.data(), q_abs.size() * 4, cudaMemcpyHostToDevice), "copy") &&
        check(cudaMemcpy(d_lat, latent.data(), latent.size() * 2, cudaMemcpyHostToDevice), "copy") &&
        check(cudaMemcpy(d_pe, k_pe.data(), k_pe.size() * 2, cudaMemcpyHostToDevice), "copy") &&
        check(l31_fa_launch(d_out, d_scratch, d_q, d_abs, d_lat, d_pe, rows, pos0, scale, GLMVQ_LAUNCH_PLAIN, 0), "launch") &&
        check(cudaDeviceSynchronize(), "kernel");
    std::vector<float> out(out_n);
    ok = ok && check(cudaMemcpy(out.data(), d_out, out_n * 4, cudaMemcpyDeviceToHost), "read");

    double worst = 0.0;
    for (uint32_t r = 0; ok && r < rows; r++) {
        const uint32_t visible = pos0 + r + 1;
        for (uint32_t h = 0; h < H; h++) {
            const float *qa = &q_abs[((uint64_t)r * H + h) * L], *qp = &q[((uint64_t)r * H + h) * K + L31_FA_NOPE];
            std::vector<double> s(visible);
            double peak = -INFINITY;
            for (uint32_t k = 0; k < visible; k++) {
                double dot = 0.0;
                for (uint32_t d = 0; d < L; d++) { dot += (double)qa[d] * __bfloat162float(latent[(uint64_t)k * L + d]); }
                for (uint32_t d = 0; d < R; d++) { dot += (double)qp[d] * __bfloat162float(k_pe[(uint64_t)k * R + d]); }
                s[k] = dot * scale;
                peak = std::max(peak, s[k]);
            }
            double sum = 0.0;
            for (auto &v : s) { v = std::exp(v - peak); sum += v; }
            double norm = 0.0, err = 0.0;
            for (uint32_t d = 0; d < L; d++) {
                double ref = 0.0;
                for (uint32_t k = 0; k < visible; k++) { ref += s[k] * __bfloat162float(latent[(uint64_t)k * L + d]); }
                ref /= sum;
                const double got = out[((uint64_t)r * H + h) * L + d];
                norm += ref * ref;
                err += (got - ref) * (got - ref);
            }
            worst = std::max(worst, std::sqrt(err / std::max(norm, 1e-30)));
        }
    }
    // BF16 Q and P carry ~2^-9 relative rounding each.
    const bool pass = ok && worst < 1.0e-2;
    printf("rows=%u pos0=%u keys=%u rel_rms %.3e %s\n", rows, pos0, keys, worst, pass ? "ok" : "FAIL");
    cudaFree(d_q); cudaFree(d_abs); cudaFree(d_out); cudaFree(d_lat); cudaFree(d_pe); cudaFree(d_scratch);
    return pass;
}

// `bench`: decode-row time per call against the bytes of every cached key.
static void bench(uint32_t rows, uint32_t pos0) {
    constexpr uint32_t H = GLMVQ_FA_HEADS, L = GLMVQ_FA_LATENT, R = L31_FA_ROPE, K = L31_FA_KEY;
    const uint64_t keys = pos0 + rows;
    float *d_q = nullptr, *d_abs = nullptr, *d_out = nullptr;
    __nv_bfloat16 *d_lat = nullptr, *d_pe = nullptr;
    void *d_scratch = nullptr;
    cudaMalloc(&d_q, (uint64_t)rows * H * K * 4); cudaMalloc(&d_abs, (uint64_t)rows * H * L * 4);
    cudaMalloc(&d_out, (uint64_t)rows * H * L * 4); cudaMalloc(&d_lat, keys * L * 2);
    cudaMalloc(&d_pe, keys * R * 2); cudaMalloc(&d_scratch, l31_fa_scratch_bytes());
    cudaMemset(d_q, 0, (uint64_t)rows * H * K * 4); cudaMemset(d_abs, 0, (uint64_t)rows * H * L * 4);
    cudaMemset(d_lat, 0, keys * L * 2); cudaMemset(d_pe, 0, keys * R * 2);
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    const float scale = 1.0f / sqrtf((float)K);
    for (int i = 0; i < 3; i++) { l31_fa_launch(d_out, d_scratch, d_q, d_abs, d_lat, d_pe, rows, pos0, scale, GLMVQ_LAUNCH_PLAIN, 0); }
    cudaEventRecord(a);
    const int iters = 20;
    for (int i = 0; i < iters; i++) { l31_fa_launch(d_out, d_scratch, d_q, d_abs, d_lat, d_pe, rows, pos0, scale, GLMVQ_LAUNCH_PLAIN, 0); }
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    const double us = 1000.0 * ms / iters, gb = (double)keys * (L + R) * 2 / 1e9;
    printf("bench rows=%u keys=%llu %.1f us %.1f GB/s\n", rows, (unsigned long long)keys, us, gb / (us * 1e-6));
    cudaFree(d_q); cudaFree(d_abs); cudaFree(d_out); cudaFree(d_lat); cudaFree(d_pe); cudaFree(d_scratch);
}

int main(int argc, char **argv) {
    if (!check(cudaFuncSetAttribute(l31_fa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L31_FA_SMEM),
            "shared memory")) { return 1; }
    if (argc > 1 && !strcmp(argv[1], "bench")) {
        for (uint32_t pos : {8191u, 32767u, 65535u, 131071u}) { bench(1, pos); }
        bench(3, 32765u);
        return 0;
    }
    std::mt19937 rng(31);
    bool ok = true;
    const uint32_t cases[][2] = {{1, 0}, {1, 30}, {1, 31}, {1, 1000}, {1, 4099}, {3, 4097}, {4, 70}, {2, 9000}};
    for (const auto &c : cases) { ok = run(c[0], c[1], rng) && ok; }
    return ok ? 0 : 1;
}
