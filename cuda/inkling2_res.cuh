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

/* Decode and verify rows: one row over INK2_RES_CTAS CTAs, a thread per
 * element. One CTA moved a row's ~190 KB (states, queries, norms, outputs)
 * at a single SM's rate: 16-42 us a call against 8-14 us here, 97 calls per
 * token. `ink2_res_stats` reduces each CTA's slice of the
 * stacked states' Gram matrix and of their dots with the queries;
 * `ink2_res_apply` sums the slices, weighs its slice of the states and scales
 * by the output norm, whose sum of squares is the weights' quadratic form of
 * the Gram matrix: no pass over the finished outputs. The sums run in a
 * different F32 order than the one-CTA kernel's. */
enum { INK2_RES_SPAN = 192, INK2_RES_CTAS = INK2_HIDDEN / INK2_RES_SPAN, INK2_RES_ROWS = 8,
       INK2_RES_STATS = INK2_MAX_STACK * (INK2_MAX_STACK + 1) / 2 + INK2_MAX_QUERIES * INK2_MAX_STACK,
       INK2_RES_SCRATCH = INK2_RES_ROWS * INK2_RES_CTAS * INK2_RES_STATS };
static_assert(INK2_RES_SPAN * INK2_RES_CTAS == INK2_HIDDEN && INK2_RES_SPAN % 32 == 0, "whole warps tile a row");
static_assert(INK2_RES_STATS <= INK2_RES_SPAN, "a thread per statistic");

// Gram entry (j, k), j <= k, of n stacked states.
__device__ __forceinline__ constexpr uint32_t ink2_res_gram(uint32_t n, uint32_t j, uint32_t k) {
    return j * n - j * (j - 1) / 2 + (k - j);
}

template <int BLOCKS>
__device__ __forceinline__ void ink2_res_states(float (&v)[BLOCKS + 1], const uint16_t *blocks, uint32_t cap,
        const float *partial, uint32_t row, uint32_t i) {
    #pragma unroll
    for (uint32_t j = 0; j < BLOCKS; j++) { v[j] = ink2_bf(blocks[((uint64_t)j * cap + row) * INK2_HIDDEN + i]); }
    v[BLOCKS] = partial[(uint64_t)row * INK2_HIDDEN + i];
}

template <int BLOCKS, int QUERIES>
__global__ void __launch_bounds__(INK2_RES_SPAN) ink2_res_stats(float *part, const uint16_t *blocks, uint32_t cap,
        const float *partial, const float *query) {
    constexpr uint32_t N = BLOCKS + 1, GRAM = N * (N + 1) / 2, STATS = GRAM + QUERIES * N;
    constexpr uint32_t WARPS = INK2_RES_SPAN / 32;
    __shared__ float scratch[WARPS * STATS];
    const uint32_t row = blockIdx.y, i = blockIdx.x * INK2_RES_SPAN + threadIdx.x;
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    float v[N], qv[QUERIES], stats[STATS];
    ink2_res_states<BLOCKS>(v, blocks, cap, partial, row, i);
    #pragma unroll
    for (uint32_t qi = 0; qi < QUERIES; qi++) { qv[qi] = query[(uint64_t)qi * INK2_HIDDEN + i]; }
    #pragma unroll
    for (uint32_t j = 0; j < N; j++) {
        #pragma unroll
        for (uint32_t k = j; k < N; k++) { stats[ink2_res_gram(N, j, k)] = v[j] * v[k]; }
        #pragma unroll
        for (uint32_t qi = 0; qi < QUERIES; qi++) { stats[GRAM + qi * N + j] = qv[qi] * v[j]; }
    }
    #pragma unroll
    for (uint32_t s = 0; s < STATS; s++) {
        float x = stats[s];
        for (uint32_t o = 16; o; o >>= 1) { x += __shfl_xor_sync(0xffffffffu, x, o); }
        if (!lane) { scratch[warp * STATS + s] = x; }
    }
    __syncthreads();
    if (threadIdx.x >= STATS) { return; }
    float x = 0.0f;
    #pragma unroll
    for (uint32_t w = 0; w < WARPS; w++) { x += scratch[w * STATS + threadIdx.x]; }
    part[((uint64_t)row * INK2_RES_CTAS + blockIdx.x) * INK2_RES_STATS + threadIdx.x] = x;
}

template <int BLOCKS, int QUERIES>
__global__ void __launch_bounds__(INK2_RES_SPAN) ink2_res_apply(float *out0, float *out1, float *out2,
        const float *part, const uint16_t *blocks, uint32_t cap, const float *partial,
        const uint16_t *norm0, const uint16_t *norm1, const uint16_t *norm2) {
    constexpr uint32_t N = BLOCKS + 1, GRAM = N * (N + 1) / 2, STATS = GRAM + QUERIES * N;
    __shared__ float total[STATS];
    const uint32_t row = blockIdx.y, i = blockIdx.x * INK2_RES_SPAN + threadIdx.x;
    float *outs[INK2_MAX_QUERIES] = {out0, out1, out2};
    const uint16_t *norms[INK2_MAX_QUERIES] = {norm0, norm1, norm2};
    // The element's loads fly while the slices' statistics are summed.
    float v[N], weight[QUERIES];
    ink2_res_states<BLOCKS>(v, blocks, cap, partial, row, i);
    #pragma unroll
    for (uint32_t qi = 0; qi < QUERIES; qi++) { weight[qi] = ink2_bf(norms[qi][i]); }
    if (threadIdx.x < STATS) {
        const float *src = part + (uint64_t)row * INK2_RES_CTAS * INK2_RES_STATS + threadIdx.x;
        float x = 0.0f;
        #pragma unroll
        for (uint32_t c = 0; c < INK2_RES_CTAS; c++) { x += src[c * INK2_RES_STATS]; }
        total[threadIdx.x] = x;
    }
    __syncthreads();
    float inv[N];
    #pragma unroll
    for (uint32_t j = 0; j < N; j++) {
        inv[j] = rsqrtf(__fdiv_rn(total[ink2_res_gram(N, j, j)], (float)INK2_HIDDEN) + 1e-6f);
    }
    const float scale = rsqrtf((float)INK2_HIDDEN);
    #pragma unroll
    for (uint32_t qi = 0; qi < QUERIES; qi++) {
        float p[N], top = -INFINITY, sum = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < N; j++) {
            p[j] = total[GRAM + qi * N + j] * inv[j] * scale;
            top = fmaxf(top, p[j]);
        }
        #pragma unroll
        for (uint32_t j = 0; j < N; j++) {
            p[j] = expf(p[j] - top);
            sum += p[j];
        }
        float y = 0.0f, sq = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < N; j++) {
            p[j] = __fdiv_rn(p[j], sum);
            y = fmaf(p[j], v[j] * inv[j], y);
        }
        // |y|^2 over the row: sum_jk (p_j inv_j) (p_k inv_k) <state_j, state_k>.
        #pragma unroll
        for (uint32_t j = 0; j < N; j++) {
            #pragma unroll
            for (uint32_t k = j; k < N; k++) {
                const float pair = p[j] * inv[j] * (p[k] * inv[k]) * total[ink2_res_gram(N, j, k)];
                sq += j == k ? pair : 2.0f * pair;
            }
        }
        const float r = rsqrtf(__fdiv_rn(fmaxf(sq, 0.0f), (float)INK2_HIDDEN) + 1e-6f);
        outs[qi][(uint64_t)row * INK2_HIDDEN + i] = glmvq_round(glmvq_mul(weight[qi], glmvq_mul(y, r)));
    }
}

// `scratch` holds INK2_RES_SCRATCH floats. Returns false for a shape the
// caller keeps on the one-CTA kernel.
template <uint32_t BLOCKS = 0>
static bool ink2_res_split(float *scratch, float *out0, float *out1, float *out2, const uint16_t *blocks,
        uint32_t nblocks, uint32_t cap, const float *partial, const float *query, uint32_t nq,
        const uint16_t *norm0, const uint16_t *norm1, const uint16_t *norm2, uint32_t rows, cudaStream_t stream) {
    if (!query || rows > INK2_RES_ROWS || (nq != 1 && nq != 3)) { return false; }
    if constexpr (BLOCKS < DS4_INK2_BLOCKS) {
        if (nblocks != BLOCKS) {
            return ink2_res_split<BLOCKS + 1>(scratch, out0, out1, out2, blocks, nblocks, cap, partial, query, nq,
                norm0, norm1, norm2, rows, stream);
        }
    }
    const dim3 grid(INK2_RES_CTAS, rows);
    if (nq == 3) {
        ink2_res_stats<BLOCKS, 3><<<grid, INK2_RES_SPAN, 0, stream>>>(scratch, blocks, cap, partial, query);
        ink2_res_apply<BLOCKS, 3><<<grid, INK2_RES_SPAN, 0, stream>>>(out0, out1, out2, scratch, blocks, cap,
            partial, norm0, norm1, norm2);
    } else {
        ink2_res_stats<BLOCKS, 1><<<grid, INK2_RES_SPAN, 0, stream>>>(scratch, blocks, cap, partial, query);
        ink2_res_apply<BLOCKS, 1><<<grid, INK2_RES_SPAN, 0, stream>>>(out0, out1, out2, scratch, blocks, cap,
            partial, norm0, norm1, norm2);
    }
    return true;
}

#endif
