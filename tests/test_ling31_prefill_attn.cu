/* Ling-3.1's pipelined expanded-MLA range attention against the Ling-3.0
 * TK=64 kernel it replaces: same per-warp math, so outputs and LSE must be
 * byte-identical, including partial query and key tiles, causal segments
 * and prefix segments that end before the first query. The in-place merge
 * must equal a separate Motif-3 state merge (its kernel, copied below). */
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

// ds4_cuda.cu motif3_merge_attention_states_kernel, verbatim.
__global__ static void merge_states(float *attention, float *lse, const float *prefix_attention,
        const float *prefix_lse, uint32_t rows, uint32_t q_heads, uint32_t value_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t head = blockIdx.y;
    if (row >= rows || head >= q_heads) return;
    const uint64_t state = (uint64_t)row * q_heads + head;
    __shared__ float suffix_weight;
    __shared__ float prefix_weight;
    __shared__ float merged_lse;
    if (threadIdx.x == 0u) {
        const float suffix_lse = lse[state];
        const float prior_lse = prefix_lse[state];
        const float m = fmaxf(suffix_lse, prior_lse);
        if (isinf(m) && m < 0.0f) {
            suffix_weight = 0.0f;
            prefix_weight = 0.0f;
            merged_lse = -INFINITY;
        } else {
            const float sw = expf(suffix_lse - m);
            const float pw = expf(prior_lse - m);
            const float inv = 1.0f / (sw + pw);
            suffix_weight = sw * inv;
            prefix_weight = pw * inv;
            merged_lse = m + logf(sw + pw);
        }
    }
    __syncthreads();
    const uint64_t base = state * value_dim;
    for (uint32_t d = threadIdx.x; d < value_dim; d += blockDim.x) {
        attention[base + d] =
            attention[base + d] * suffix_weight +
            prefix_attention[base + d] * prefix_weight;
    }
    if (threadIdx.x == 0u) lse[state] = merged_lse;
}

static void upload(std::vector<__nv_bfloat16> &host, void **dev, size_t n, unsigned salt) {
    host.resize(n);
    for (size_t i = 0; i < n; i++) host[i] = __float2bfloat16(pseudo(i, salt));
    CHECK(cudaMalloc(dev, n * 2) == cudaSuccess);
    CHECK(cudaMemcpy(*dev, host.data(), n * 2, cudaMemcpyHostToDevice) == cudaSuccess);
}

// Two segments (prefix, then the causal chunk) through each path.
static void merge_case(float scale) {
    const int rows = 200, pos = 1000, prefix = 300;
    std::vector<float> hq((size_t)rows * HEADS * QK);
    for (size_t i = 0; i < hq.size(); i++) hq[i] = pseudo(i, 7);
    std::vector<__nv_bfloat16> hk[2], hv[2];
    void *k[2], *v[2];
    const int kv[2] = {prefix, rows}, kv0[2] = {0, pos};
    for (int s = 0; s < 2; s++) {
        upload(hk[s], &k[s], (size_t)kv[s] * HEADS * QK, 8 + s);
        upload(hv[s], &v[s], (size_t)kv[s] * HEADS * VD, 10 + s);
    }
    float *q, *out[3], *lse[3];
    const size_t out_n = (size_t)rows * HEADS * VD, lse_n = (size_t)rows * HEADS;
    CHECK(cudaMalloc(&q, hq.size() * 4) == cudaSuccess);
    CHECK(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    for (int i = 0; i < 3; i++) {
        CHECK(cudaMalloc(&out[i], out_n * 4) == cudaSuccess);
        CHECK(cudaMalloc(&lse[i], lse_n * 4) == cudaSuccess);
    }
    // Separate: base kernel per segment into out[0] and the partial out[2].
    CHECK(ds4_mmq_ling3vl_prefill_attn_hmma(out[0], lse[0], q, k[0], v[0], rows, pos, kv[0], kv0[0],
        HEADS, QK, VD, scale, 64, 0) == 0);
    CHECK(ds4_mmq_ling3vl_prefill_attn_hmma(out[2], lse[2], q, k[1], v[1], rows, pos, kv[1], kv0[1],
        HEADS, QK, VD, scale, 64, 0) == 0);
    merge_states<<<dim3(rows, HEADS), VD>>>(out[0], lse[0], out[2], lse[2], rows, HEADS, VD);
    // Fused: the second segment merges into out[1] in place.
    CHECK(ds4_mmq_ling31_prefill_attn_pipe(out[1], lse[1], q, k[0], v[0], rows, pos, kv[0], kv0[0],
        HEADS, QK, VD, scale, 0, 0) == 0);
    CHECK(ds4_mmq_ling31_prefill_attn_pipe(out[1], lse[1], q, k[1], v[1], rows, pos, kv[1], kv0[1],
        HEADS, QK, VD, scale, 1, 0) == 0);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<float> ho[2], hl[2];
    for (int i = 0; i < 2; i++) {
        ho[i].resize(out_n);
        hl[i].resize(lse_n);
        CHECK(cudaMemcpy(ho[i].data(), out[i], out_n * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(cudaMemcpy(hl[i].data(), lse[i], lse_n * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    }
    CHECK(memcmp(ho[0].data(), ho[1].data(), out_n * 4) == 0);
    CHECK(memcmp(hl[0].data(), hl[1].data(), lse_n * 4) == 0);
    for (int i = 0; i < 3; i++) { cudaFree(out[i]); cudaFree(lse[i]); }
    for (int s = 0; s < 2; s++) { cudaFree(k[s]); cudaFree(v[s]); }
    cudaFree(q);
    printf("Ling-3.1 prefill attention in-place merge: %d rows x %d + %d keys byte-identical\n",
           rows, prefix, rows);
}

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
            HEADS, QK, VD, scale, 0, 0) == 0);
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
    merge_case(scale);
    return 0;
}
