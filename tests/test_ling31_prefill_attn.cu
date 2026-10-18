/* Ling-3.1's pipelined expanded-MLA range attention against the Ling-3.0
 * TK=64 kernel it replaces: same per-warp math, so outputs and LSE must be
 * byte-identical, including partial query and key tiles, causal segments
 * and prefix segments that end before the first query. */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "cuda/mmq/ds4_mmq.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "Ling-3.1 prefill attention FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

enum { HEADS = 64, QK = 192, VD = 128 };

static float pseudo(size_t i, unsigned salt) {
    uint32_t x = (uint32_t)(i * 2654435761u) ^ (salt * 40503u);
    x ^= x >> 13;
    x *= 0x5bd1e995u;
    x ^= x >> 15;
    return ((int)(x % 2001u) - 1000) / 500.0f;
}

struct Case { const char *name; int rows, pos, kv, kv0; };

int main() {
    const Case cases[] = {
        {"prefix tail", 200, 1000, 300, 0},
        {"causal tail", 200, 1000, 200, 1000},
        {"causal straddle", 130, 4096, 194, 4032},
        {"prefix full", 512, 2048, 512, 512},
    };
    const float scale = 1.0f / sqrtf((float)QK);
    for (const Case &c : cases) {
        std::vector<float> hq((size_t)c.rows * HEADS * QK);
        std::vector<__nv_bfloat16> hk((size_t)c.kv * HEADS * QK), hv((size_t)c.kv * HEADS * VD);
        for (size_t i = 0; i < hq.size(); i++) hq[i] = pseudo(i, 1);
        for (size_t i = 0; i < hk.size(); i++) hk[i] = __float2bfloat16(pseudo(i, 2));
        for (size_t i = 0; i < hv.size(); i++) hv[i] = __float2bfloat16(pseudo(i, 3));
        float *q, *out[2], *lse[2];
        void *k, *v;
        const size_t out_n = (size_t)c.rows * HEADS * VD, lse_n = (size_t)c.rows * HEADS;
        CHECK(cudaMalloc(&q, hq.size() * 4) == cudaSuccess);
        CHECK(cudaMalloc(&k, hk.size() * 2) == cudaSuccess);
        CHECK(cudaMalloc(&v, hv.size() * 2) == cudaSuccess);
        for (int i = 0; i < 2; i++) {
            CHECK(cudaMalloc(&out[i], out_n * 4) == cudaSuccess);
            CHECK(cudaMalloc(&lse[i], lse_n * 4) == cudaSuccess);
            CHECK(cudaMemset(out[i], 0xff, out_n * 4) == cudaSuccess);
        }
        CHECK(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
        CHECK(cudaMemcpy(k, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
        CHECK(cudaMemcpy(v, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
        CHECK(ds4_mmq_ling3vl_prefill_attn_hmma(out[0], lse[0], q, k, v, c.rows, c.pos, c.kv, c.kv0,
            HEADS, QK, VD, scale, 64, 0) == 0);
        CHECK(ds4_mmq_ling31_prefill_attn_pipe(out[1], lse[1], q, k, v, c.rows, c.pos, c.kv, c.kv0,
            HEADS, QK, VD, scale, 0) == 0);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        std::vector<float> ho[2], hl[2];
        for (int i = 0; i < 2; i++) {
            ho[i].resize(out_n);
            hl[i].resize(lse_n);
            CHECK(cudaMemcpy(ho[i].data(), out[i], out_n * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
            CHECK(cudaMemcpy(hl[i].data(), lse[i], lse_n * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
            cudaFree(out[i]);
            cudaFree(lse[i]);
        }
        CHECK(memcmp(ho[0].data(), ho[1].data(), out_n * 4) == 0);
        CHECK(memcmp(hl[0].data(), hl[1].data(), lse_n * 4) == 0);
        for (size_t i = 0; i < lse_n; i++) { CHECK(std::isfinite(hl[1][i])); }
        cudaFree(q);
        cudaFree(k);
        cudaFree(v);
        printf("Ling-3.1 prefill attention %s: %d rows x %d keys byte-identical\n", c.name, c.rows, c.kv);
    }
    return 0;
}
