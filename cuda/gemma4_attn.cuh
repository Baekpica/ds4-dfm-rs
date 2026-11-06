#ifndef DS4_GEMMA4_ATTN_CUH
#define DS4_GEMMA4_ATTN_CUH

/* Causal attention for Gemma-4's two geometries: scale 1.0 (the q/k RMS
 * norms replace head_dim**-0.5, semantics 3.3), grouped KV (16 query heads
 * over 8 or 2 KV heads), an optional sliding window of 1024 that includes
 * the query (semantics 3.4), and softmax in float32.
 *
 * Layout: queries are [rows][heads][head_dim] with row r at absolute
 * position first + r; keys and values are [capacity][kv_heads][head_dim]
 * stored as bf16, which is the reference's rounding of the projection.
 *
 * Correctness first: one warp per (row, head), an online softmax, and one
 * warp reduction per key for the dot product. This is not the speed path. */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../ds4_gemma4_ref.h"

__device__ __forceinline__ static float g4_warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

/* `window` 0 means full causal; otherwise the visible set is
 * [position - window + 1, position] (semantics 3.4). `ring` is the cache's
 * row count when rows wrap (a sliding layer keeps window + prefill_cap rows,
 * so a prefill batch never overwrites a slot its own rows still read); 0
 * means the cache is indexed by absolute position.
 *
 * `keys` is how many keys the cache actually holds: 0 means first + rows, the
 * ordinary forward's case. The drafter passes a shorter count, because its
 * query sits at the position of the token being committed while the trunk has
 * written only the rows before it (candidate_generator.py:1316-1319 truncates
 * the shared KV to the committed length; candidate_generator.py:1327 puts the
 * query at `input_ids.shape[1] - 1`). The window's begin is still relative to
 * the QUERY's position, as the reference's flipped sliding mask is. */
__global__ static void g4_attn_kernel(float *out, const float *query,
        const __nv_bfloat16 *key, const __nv_bfloat16 *value,
        uint32_t rows, uint32_t first, uint32_t heads, uint32_t kv_heads,
        uint32_t dim, uint32_t window, uint32_t ring, uint32_t keys,
        uint64_t kv_stride) {
    const uint32_t warp = threadIdx.x / 32u;
    const uint32_t lane = threadIdx.x % 32u;
    const uint64_t job = (uint64_t)blockIdx.x * (blockDim.x / 32u) + warp;
    if (job >= (uint64_t)rows * heads) { return; }
    const uint32_t row = (uint32_t)(job / heads);
    const uint32_t head = (uint32_t)(job % heads);
    const uint32_t group = heads / kv_heads;
    const uint32_t kv_head = head / group;
    const uint32_t position = first + row;
    const float *q = query + job * dim;
    float *o = out + job * dim;

    const uint32_t per_lane = (dim + 31u) / 32u;
    float acc[16];
    float qv[16];
    for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
        const uint32_t d = lane + i * 32u;
        qv[i] = d < dim ? q[d] : 0.0f;
        acc[i] = 0.0f;
    }
    float maximum = -INFINITY, total = 0.0f;
    const uint32_t begin = window && position + 1 > window ? position + 1 - window : 0u;
    const uint32_t last = keys && keys <= position ? keys - 1u : position;
    for (uint32_t k = begin; k <= last; k++) {
        const uint64_t slot = ring ? (uint64_t)(k % ring) : k;
        const __nv_bfloat16 *kv = key + slot * kv_stride + (uint64_t)kv_head * dim;
        float dot = 0.0f;
        for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
            const uint32_t d = lane + i * 32u;
            if (d < dim) { dot += qv[i] * __bfloat162float(kv[d]); }
        }
        dot = g4_warp_sum(dot);
        dot = __shfl_sync(0xffffffffu, dot, 0);
        /* Softmax in float32, which is the reference's own arithmetic:
         * modeling_gemma4.py:839 calls F.softmax(attn_weights, dim=-1,
         * dtype=torch.float32), so the exponent is a float32 exp. This
         * kernel used a double exp until 2026-11-03; on the consumer cards
         * the FP64 pipe runs at 1/64 of FP32 and two double exps per key
         * dominated the prefill attention (139.8 ms of the 511 ms GPU time
         * at ctx 256 on the 4070 SUPER). expf keeps the reference's
         * precision class and moves the work to the FP32 pipe. */
        const float m_new = fmaxf(maximum, dot);
        const float alpha = expf(maximum - m_new);
        const float p = expf(dot - m_new);
        total = total * alpha + p;
        maximum = m_new;
        const __nv_bfloat16 *vv = value + slot * kv_stride + (uint64_t)kv_head * dim;
        for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
            const uint32_t d = lane + i * 32u;
            if (d < dim) { acc[i] = acc[i] * alpha + p * __bfloat162float(vv[d]); }
        }
    }
    const float inverse = 1.0f / total;
    for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
        const uint32_t d = lane + i * 32u;
        if (d < dim) { o[d] = acc[i] * inverse; }
    }
}

#endif
