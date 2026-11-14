#ifndef DS4_INKLING2_RES_CUH
#define DS4_INKLING2_RES_CUH
#include "../ds4_inkling2.h"
#include "glm_vq_ops.cuh"

enum { INK2_HIDDEN = DS4_INK2_HIDDEN, INK2_THREADS = 256, INK2_PER_THREAD = INK2_HIDDEN / INK2_THREADS,
       INK2_MAX_STACK = DS4_INK2_BLOCKS + 1, INK2_MAX_QUERIES = 3 };

// Block-wide sums of `n` per-thread values (n <= 40), in place.
template <uint32_t N>
__device__ __forceinline__ void ink2_block_sum(float (&v)[N], uint32_t n, float *scratch) {
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    #pragma unroll
    for (uint32_t i = 0; i < n; i++) {
        float x = v[i];
        for (uint32_t o = 16; o; o >>= 1) { x += __shfl_xor_sync(0xffffffffu, x, o); }
        if (!lane) { scratch[warp * N + i] = x; }
    }
    __syncthreads();
    #pragma unroll
    for (uint32_t i = 0; i < n; i++) {
        float x = 0.0f;
        for (uint32_t w = 0; w < warps; w++) { x += scratch[w * N + i]; }
        v[i] = x;
    }
    __syncthreads();
}

__device__ __forceinline__ float ink2_bf(uint16_t bits) { return __uint_as_float((uint32_t)bits << 16); }

/* One CTA per token. Pass 1: each stacked state's sum of squares and its dot
 * with every query; pass 2: the attention-weighted sum of the normalized
 * states; then the weighted RMS norm of each output. */
template <int BLOCKS = -1, int QUERIES = -1>
__global__ void __launch_bounds__(INK2_THREADS) ink2_attn_res_kernel(float *out0, float *out1, float *out2,
        const uint16_t *blocks, uint32_t nblocks, uint32_t cap, const float *partial, const float *query,
        uint32_t nq, const uint16_t *norm0, const uint16_t *norm1, const uint16_t *norm2) {
    constexpr uint32_t STACK = BLOCKS < 0 ? INK2_MAX_STACK : BLOCKS + 1;
    constexpr uint32_t QUERIES_MAX = QUERIES < 0 ? INK2_MAX_QUERIES : (QUERIES ? QUERIES : 1);
    constexpr uint32_t STATS = STACK * (QUERIES_MAX + 1);
    __shared__ float scratch[(INK2_THREADS / 32) * (STATS > INK2_MAX_QUERIES ? STATS : INK2_MAX_QUERIES)];
    if constexpr (BLOCKS >= 0) { nblocks = BLOCKS; }
    if constexpr (QUERIES >= 0) { nq = QUERIES; }
    const uint32_t row = blockIdx.x, n = nblocks + 1;
    const float *x = partial + (uint64_t)row * INK2_HIDDEN;
    auto state = [&](uint32_t j, uint32_t i) {
        return j < nblocks ? ink2_bf(blocks[((uint64_t)j * cap + row) * INK2_HIDDEN + i]) : x[i];
    };
    float *outs[INK2_MAX_QUERIES] = {out0, out1, out2};
    const uint16_t *norms[INK2_MAX_QUERIES] = {norm0, norm1, norm2};
    float y[QUERIES_MAX][INK2_PER_THREAD];
    const uint32_t outputs = QUERIES < 0 ? (query ? nq : 1u) : QUERIES_MAX;
    if (QUERIES == 0 || (QUERIES < 0 && !query)) {
        // No attention: the partial state itself is normalized.
        #pragma unroll
        for (uint32_t k = 0; k < INK2_PER_THREAD; k++) { y[0][k] = x[threadIdx.x + k * INK2_THREADS]; }
    } else {
        float stats[STATS];
        #pragma unroll
        for (uint32_t i = 0; i < STATS; i++) { stats[i] = 0.0f; }
        #pragma unroll
        for (uint32_t j = 0; j < n; j++) {
            #pragma unroll
            for (uint32_t k = 0; k < INK2_PER_THREAD; k++) {
                const uint32_t i = threadIdx.x + k * INK2_THREADS;
                const float v = state(j, i);
                stats[j] = fmaf(v, v, stats[j]);
                #pragma unroll
                for (uint32_t qi = 0; qi < nq; qi++) {
                    stats[STACK + qi * STACK + j] =
                        fmaf(query[(uint64_t)qi * INK2_HIDDEN + i], v,
                            stats[STACK + qi * STACK + j]);
                }
            }
        }
        ink2_block_sum<STATS>(stats, STATS, scratch);
        float inv[STACK], p[QUERIES_MAX][STACK];
        #pragma unroll
        for (uint32_t j = 0; j < n; j++) {
            inv[j] = rsqrtf(__fdiv_rn(stats[j], (float)INK2_HIDDEN) + 1e-6f);
        }
        const float scale = rsqrtf((float)INK2_HIDDEN);
        #pragma unroll
        for (uint32_t qi = 0; qi < nq; qi++) {
            float top = -INFINITY, total = 0.0f;
            #pragma unroll
            for (uint32_t j = 0; j < n; j++) {
                p[qi][j] = stats[STACK + qi * STACK + j] * inv[j] * scale;
                top = fmaxf(top, p[qi][j]);
            }
            #pragma unroll
            for (uint32_t j = 0; j < n; j++) {
                p[qi][j] = expf(p[qi][j] - top);
                total += p[qi][j];
            }
            #pragma unroll
            for (uint32_t j = 0; j < n; j++) { p[qi][j] = __fdiv_rn(p[qi][j], total); }
        }
        #pragma unroll
        for (uint32_t qi = 0; qi < nq; qi++) {
            #pragma unroll
            for (uint32_t k = 0; k < INK2_PER_THREAD; k++) { y[qi][k] = 0.0f; }
        }
        #pragma unroll
        for (uint32_t j = 0; j < n; j++) {
            #pragma unroll
            for (uint32_t k = 0; k < INK2_PER_THREAD; k++) {
                const float v = state(j, threadIdx.x + k * INK2_THREADS) * inv[j];
                #pragma unroll
                for (uint32_t qi = 0; qi < nq; qi++) { y[qi][k] = fmaf(p[qi][j], v, y[qi][k]); }
            }
        }
    }
    float sq[INK2_MAX_QUERIES] = {0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (uint32_t qi = 0; qi < outputs; qi++) {
        #pragma unroll
        for (uint32_t k = 0; k < INK2_PER_THREAD; k++) { sq[qi] = fmaf(y[qi][k], y[qi][k], sq[qi]); }
    }
    ink2_block_sum<INK2_MAX_QUERIES>(sq, outputs, scratch);
    #pragma unroll
    for (uint32_t qi = 0; qi < outputs; qi++) {
        const float r = rsqrtf(__fdiv_rn(sq[qi], (float)INK2_HIDDEN) + 1e-6f);
        float *o = outs[qi] + (uint64_t)row * INK2_HIDDEN;
        #pragma unroll
        for (uint32_t k = 0; k < INK2_PER_THREAD; k++) {
            const uint32_t i = threadIdx.x + k * INK2_THREADS;
            o[i] = glmvq_round(glmvq_mul(ink2_bf(norms[qi][i]), glmvq_mul(y[qi][k], r)));
        }
    }
}

// Runtime indexing put 592 bytes of per-thread arrays on the stack. The
// model's fixed stack/query widths keep those arrays in registers.
template <uint32_t BLOCKS = 0>
static void ink2_res_launch(float *out0, float *out1, float *out2, const uint16_t *blocks,
        uint32_t nblocks, uint32_t cap, const float *partial, const float *query, uint32_t nq,
        const uint16_t *norm0, const uint16_t *norm1, const uint16_t *norm2,
        uint32_t rows, cudaStream_t stream) {
    if (!query) {
        ink2_attn_res_kernel<0, 0><<<rows, INK2_THREADS, 0, stream>>>(out0, out1, out2,
            blocks, nblocks, cap, partial, query, nq, norm0, norm1, norm2);
        return;
    }
    if constexpr (BLOCKS < DS4_INK2_BLOCKS) {
        if (nblocks != BLOCKS) {
            ink2_res_launch<BLOCKS + 1>(out0, out1, out2, blocks, nblocks, cap, partial, query, nq,
                norm0, norm1, norm2, rows, stream);
            return;
        }
    }
    if (nq == 3) {
        ink2_attn_res_kernel<BLOCKS, 3><<<rows, INK2_THREADS, 0, stream>>>(out0, out1, out2,
            blocks, nblocks, cap, partial, query, nq, norm0, norm1, norm2);
    } else {
        ink2_attn_res_kernel<BLOCKS, 1><<<rows, INK2_THREADS, 0, stream>>>(out0, out1, out2,
            blocks, nblocks, cap, partial, query, nq, norm0, norm1, norm2);
    }
}

#endif
