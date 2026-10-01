#pragma once
#include "naive_primitives.cuh"

/* One CTA/head shares rounded scores across four key walks. Softmax and V
 * still visit ascending IDs. The complete XOR reduction yields the same
 * finite sum in every lane, so storing lane zero preserves all lanes. */
template<NaiveCache CACHE = NaiveCache::Ring>
__global__ static void naive_sparse_tile(
        float *out, const float *q, const __nv_bfloat16 *cache,
        const unsigned *positions, const unsigned *selected, unsigned capacity) {
    enum { WARP = 32, WARPS = 4, THREADS = WARP * WARPS, KV_HEADS = 4 };
    const unsigned tid = threadIdx.x, lane = tid % WARP, warp = tid / WARP;
    const unsigned head = blockIdx.x, row = blockIdx.y;
    const unsigned pos = positions[row], count = min(pos + 1, (unsigned)N05_TOP_K);
    const unsigned kh = head / (N05_HEADS / KV_HEADS);
    const unsigned stride = KV_HEADS * (N05_KEY + N05_VALUE);
    const unsigned *ids = selected + (uint64_t)row * N05_TOP_K;
    const uint64_t qbase = ((uint64_t)row * N05_HEADS + head) * N05_KEY;
    const uint64_t obase = ((uint64_t)row * N05_HEADS + head) * N05_VALUE;
    __shared__ __nv_bfloat16 scores[N05_TOP_K];
    __shared__ float maximum, denominator;

    float query[N05_KEY / WARP];
    for (unsigned d = 0; d < N05_KEY / WARP; d++) { query[d] = q[qbase + lane + d * WARP]; }
    for (unsigned i = warp; i < count; i += WARPS) {
        const unsigned key = ids[i];
        if (key > pos) { continue; }
        const __nv_bfloat16 *slot = cache + (uint64_t)naive_cache_slot<CACHE>(key, capacity) * stride;
        float dot = 0;
        for (unsigned d = 0; d < N05_KEY / WARP; d++) {
            dot = __fmaf_rn(query[d], __bfloat162float(slot[kh * N05_KEY + lane + d * WARP]), dot);
        }
        for (unsigned step = WARP / 2; step; step /= 2) { dot += __shfl_xor_sync(0xffffffff, dot, step); }
        const float score = naive_bf16(naive_bf16(dot) * N05_QK_SCALE);
        if (!lane) { scores[i] = __float2bfloat16_rn(score); }
    }
    __syncthreads();

    if (!tid) {
        float max = -INFINITY, denom = 0;
        for (unsigned i = 0; i < count; i++) {
            if (ids[i] > pos) { continue; }
            const float score = __bfloat162float(scores[i]);
            // One exponent is exactly exp(0); elide it without changing order.
            if (score > max) {
                denom = __fmaf_rn(denom, expf(max - score), 1.0f);
                max = score;
            } else {
                denom += expf(score - max);
            }
        }
        maximum = max; denominator = denom;
    }
    __syncthreads();
    for (unsigned i = tid; i < count; i += THREADS) {
        if (ids[i] > pos) { continue; }
        const float score = __bfloat162float(scores[i]);
        scores[i] = __float2bfloat16_rn(__fdiv_rn(expf(score - maximum), denominator));
    }
    __syncthreads();

    float acc = 0;
    for (unsigned i = 0; i < count; i++) {
        const unsigned key = ids[i];
        if (key > pos) { continue; }
        const __nv_bfloat16 *slot = cache + (uint64_t)naive_cache_slot<CACHE>(key, capacity) * stride;
        const float value = naive_bf16(__bfloat162float(slot[KV_HEADS * N05_KEY + kh * N05_VALUE + tid]) * N05_V_SCALE);
        acc = __fmaf_rn(__bfloat162float(scores[i]), value, acc);
    }
    out[obase + tid] = naive_bf16(acc);
}
