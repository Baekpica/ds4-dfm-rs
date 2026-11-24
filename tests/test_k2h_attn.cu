// K2-Horizon attention (cuda/k2h_attn.cuh) against a double-precision CPU
// reference over random NVFP4 pages: the prefill tile kernel at a chunk
// offset that starts and ends inside blocks, and the decode split kernel.
// Tolerance: the kernels round probabilities to BF16 before P.V.
#include "cuda/k2h_attn.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static float e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const float v = e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp(m / 8.0f, -6);
    return s ? -v : v;
}

static float e2m1(uint32_t n) {
    static const float t[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return n & 8 ? -t[n & 7] : t[n & 7];
}

static float bf16(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    memcpy(&v, &u, 4);
    return v;
}

static uint32_t rng = 12345u;
static uint32_t next() { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

// Value d of K (half 0) or V (half 1) of position p, KV head h.
static float kv_value(const std::vector<uint8_t> &pages, uint32_t p, uint32_t h, uint32_t half, uint32_t d) {
    const uint8_t *page = pages.data() + (uint64_t)(p / INK2A_KB) * INK2A_BLOCK + h * INK2A_PAGE +
        half * INK2A_KB * INK2A_FP4_ROW;
    const uint32_t slot = p % INK2A_KB;
    const uint8_t byte = page[slot * INK2A_FP4 + d / 2];
    const uint8_t scale = page[INK2A_KB * INK2A_FP4 + slot * 8 + d / 16];
    return e2m1(d % 2 ? byte >> 4 : byte & 15) * e4m3(scale);
}

static int check(const char *name, const std::vector<float> &got, const std::vector<float> &q,
        const std::vector<uint8_t> &pages, uint32_t rows, uint32_t base) {
    double worst = 0.0;
    for (uint32_t r = 0; r < rows; r++) {
        const uint32_t pos = base + r;
        for (uint32_t h = 0; h < K2A_HEADS; h++) {
            const uint32_t kv = h / K2A_GROUP;
            std::vector<double> s(pos + 1);
            double m = -INFINITY;
            for (uint32_t p = 0; p <= pos; p++) {
                double dot = 0.0;
                for (uint32_t d = 0; d < K2A_DIM; d++) {
                    dot += (double)q[((uint64_t)r * K2A_HEADS + h) * K2A_DIM + d] * kv_value(pages, p, kv, 0, d);
                }
                s[p] = dot * K2A_SCALE;
                m = std::max(m, s[p]);
            }
            double l = 0.0;
            for (uint32_t p = 0; p <= pos; p++) { s[p] = std::exp(s[p] - m); l += s[p]; }
            for (uint32_t d = 0; d < K2A_DIM; d++) {
                double o = 0.0, scale = 0.0;
                for (uint32_t p = 0; p <= pos; p++) {
                    const double v = kv_value(pages, p, kv, 1, d);
                    o += s[p] * v;
                    scale += s[p] * std::fabs(v);
                }
                const double want = o / l, err = std::fabs(got[((uint64_t)r * K2A_HEADS + h) * K2A_DIM + d] - want);
                // BF16 P and output: relative to the magnitude of the terms.
                worst = std::max(worst, err / (scale / l + 1e-3));
            }
        }
    }
    printf("%-8s rows=%u base=%u worst relative error %.2e\n", name, rows, base, worst);
    return worst < 1.5e-2 ? 0 : 1;
}

int main() {
    const uint32_t keys = 1400, blocks = (keys + INK2A_KB - 1) / INK2A_KB;
    std::vector<uint8_t> pages((uint64_t)blocks * INK2A_BLOCK);
    for (uint64_t i = 0; i < pages.size(); i++) { pages[i] = (uint8_t)next(); }
    // Scales: modest E4M3 values (exponent 4..8), never NaN.
    for (uint32_t b = 0; b < blocks; b++) {
        for (uint32_t h = 0; h < INK2A_KV_HEADS; h++) {
            for (uint32_t half = 0; half < 2; half++) {
                uint8_t *scales = pages.data() + (uint64_t)b * INK2A_BLOCK + h * INK2A_PAGE +
                    half * INK2A_KB * INK2A_FP4_ROW + INK2A_KB * INK2A_FP4;
                for (uint32_t i = 0; i < INK2A_KB * 8; i++) { scales[i] = (uint8_t)((4 + next() % 5) << 3 | (next() & 7)); }
            }
        }
    }
    uint8_t *d_pages;
    cudaMalloc(&d_pages, pages.size());
    cudaMemcpy(d_pages, pages.data(), pages.size(), cudaMemcpyHostToDevice);
    int failed = 0;
    struct { uint32_t rows, base; } cases[] = {{100, 37}, {257, 1100}, {1, 1300}, {3, 999}, {4, 0}};
    for (const auto &c : cases) {
        std::vector<float> q((uint64_t)c.rows * K2A_HEADS * K2A_DIM);
        for (auto &v : q) { v = bf16(((float)(next() % 2001) - 1000.0f) / 4000.0f); }
        float *d_q, *d_out, *d_part;
        cudaMalloc(&d_q, q.size() * 4);
        cudaMalloc(&d_out, q.size() * 4);
        cudaMalloc(&d_part, (uint64_t)4 * K2A_HEADS * K2A_SPLITS * K2A_PARTIAL * 4);
        cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
        const char *name;
        if (c.rows <= 4) {
            name = "decode";
            const uint32_t splits = ink2a_splits(c.base + c.rows);
            k2a_decode<<<dim3(splits, INK2A_KV_HEADS, c.rows), INK2A_DTHREADS>>>(d_part, d_q, d_pages, c.base, splits);
            k2a_merge<<<c.rows * K2A_HEADS, K2A_DIM>>>(d_out, d_part, splits);
        } else {
            name = "prefill";
            cudaFuncSetAttribute(k2a_prefill, cudaFuncAttributeMaxDynamicSharedMemorySize, INK2A_SMEM_FP4);
            const dim3 grid((c.rows + INK2A_TQ - 1) / INK2A_TQ, K2A_HEADS / INK2A_CTA_HEADS);
            k2a_prefill<<<grid, INK2A_THREADS, INK2A_SMEM_FP4>>>(d_out, d_q, d_pages, c.rows, c.base);
        }
        const cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { printf("%s: %s\n", name, cudaGetErrorString(err)); return 1; }
        std::vector<float> got(q.size());
        cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost);
        failed |= check(name, got, q, pages, c.rows, c.base);
        cudaFree(d_q);
        cudaFree(d_out);
        cudaFree(d_part);
    }
    cudaFree(d_pages);
    printf(failed ? "FAIL\n" : "PASS\n");
    return failed;
}
