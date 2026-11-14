// Inkling-2 ViT flash attention (cuda/inkling2_vit_attn.cuh) against an F64
// softmax over the same BF16 q, k, v: image lengths around the 64-row tiles,
// query ranges starting off a tile, and no write past the range.
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

#include "cuda/inkling2_vit_attn.cuh"

static float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

static int run(uint32_t n, uint32_t q0, uint32_t m, uint32_t seed) {
    const uint32_t W = INK2_FA_ROW, HD = INK2_FA_HEADS * INK2_FA_HEAD;
    std::vector<__nv_bfloat16> qkv((size_t)n * W);
    std::vector<float> f((size_t)n * W);
    uint32_t s = seed;
    for (size_t i = 0; i < qkv.size(); i++) {
        s = s * 1664525u + 1013904223u;
        f[i] = bf(((float)(s >> 8) / 16777216.0f - 0.5f) * 4.0f);
        qkv[i] = __float2bfloat16_rn(f[i]);
    }
    __nv_bfloat16 *dq;
    float *dout;
    const size_t out_floats = (size_t)(m + 1) * HD;
    cudaMalloc(&dq, qkv.size() * sizeof(qkv[0]));
    cudaMalloc(&dout, out_floats * sizeof(float));
    cudaMemcpy(dq, qkv.data(), qkv.size() * sizeof(qkv[0]), cudaMemcpyHostToDevice);
    std::vector<float> out(out_floats, 12345.0f);
    cudaMemcpy(dout, out.data(), out_floats * sizeof(float), cudaMemcpyHostToDevice);
    if (ink2_vit_flash(dout, dq, n, q0, m, 0) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        printf("n %u q0 %u m %u: launch failed  FAIL\n", n, q0, m);
        return 0;
    }
    cudaMemcpy(out.data(), dout, out_floats * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dq);
    cudaFree(dout);

    double worst = 0.0;
    std::vector<double> p(n);
    for (uint32_t r = 0; r < m; r++) {
        for (uint32_t h = 0; h < INK2_FA_HEADS; h++) {
            const float *q = &f[(size_t)(q0 + r) * W + h * INK2_FA_HEAD];
            double top = -INFINITY, sum = 0.0;
            for (uint32_t j = 0; j < n; j++) {
                const float *k = &f[(size_t)j * W + HD + h * INK2_FA_HEAD];
                double dot = 0.0;
                for (uint32_t d = 0; d < INK2_FA_HEAD; d++) { dot += (double)q[d] * k[d]; }
                p[j] = dot / 8.0;
                top = std::fmax(top, p[j]);
            }
            for (uint32_t j = 0; j < n; j++) { sum += p[j] = std::exp(p[j] - top); }
            for (uint32_t d = 0; d < INK2_FA_HEAD; d++) {
                double want = 0.0;
                for (uint32_t j = 0; j < n; j++) { want += p[j] * f[(size_t)j * W + 2 * HD + h * INK2_FA_HEAD + d]; }
                want /= sum;
                // P rounds to BF16 before P V, the output after the division.
                const double err = std::fabs(out[(size_t)r * HD + h * INK2_FA_HEAD + d] - want) /
                    (0.02 + 0.01 * std::fabs(want));
                worst = std::fmax(worst, err);
            }
        }
    }
    bool untouched = true;
    for (uint32_t i = 0; i < HD; i++) { untouched &= out[(size_t)m * HD + i] == 12345.0f; }
    const bool ok = worst <= 1.0 && untouched;
    printf("n %u q0 %u m %u: worst error %.3f of tolerance%s%s\n", n, q0, m, worst,
        untouched ? "" : ", wrote past the range", ok ? "" : "  FAIL");
    return ok;
}

int main() {
    int ok = 1;
    const uint32_t lengths[] = {1, 5, 63, 64, 65, 129, 1001};
    for (uint32_t i = 0; i < sizeof lengths / sizeof lengths[0]; i++) {
        const uint32_t n = lengths[i];
        ok &= run(n, 0, n, 7u + i);
        if (n > 3) { ok &= run(n, n / 3, n - n / 3 - 1, 99u + i); }
    }
    return ok ? 0 : 1;
}
