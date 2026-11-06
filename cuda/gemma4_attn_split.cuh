#ifndef DS4_GEMMA4_ATTN_SPLIT_CUH
#define DS4_GEMMA4_ATTN_SPLIT_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>

/* Split-K decode attention for Gemma-4.
 *
 * The decode fast kernel (cuda/gemma4_attn_fast.cuh) is one CTA per KV head:
 * two CTAs on the five full layers and eight 64-thread CTAs on the sliding
 * ones. At 13,447 tokens the oracle is 2.09 tok/s and the fast kernel 4.82,
 * so the kernel is the dominant long-context decode cost and the machine is
 * mostly idle during it. This header splits each KV head's visible key range
 * across `splits` CTAs; each CTA runs the fast kernel's own per-key dot and
 * online softmax over its range, writes the unnormalized O plus (m, l), and
 * a combine kernel joins the partials:
 *
 *   m = max_s m_s
 *   l = sum_s l_s * exp(m_s - m)
 *   o = sum_s o_s * exp(m_s - m) / l
 *
 * The per-key arithmetic is the fast kernel's; only the softmax combine
 * order differs, so this is a parity path (the split boundaries change the
 * floating-point summation) rather than a bit-identical one. The combine is
 * exact when a single split covers the range, and the gate compares against
 * the fast kernel on the model.
 *
 * The key span per CTA is a host knob (DS4_GEMMA4_ATTN_SPLIT_SPAN, default
 * 64): the host divides the visible range by it and passes the resulting
 * `splits`. There is no fixed ceiling on that count: the combine kernel's
 * weight array is dynamic shared memory sized by the launch, and the host
 * asks for the larger dynamic-shared limit when the count needs it. */

enum { G4_SPLIT_TILE = 16u };

__device__ __forceinline__ float g4_split_warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

/* One (KV head, key range) CTA. `k0..k1` are absolute keys; `ring` wraps the
 * slot, `keys` caps the producer's written rows. Writes the unnormalized
 * accumulator and the (m, l) stats for this split. */
template <uint32_t TILE>
__global__ static void g4_attn_split_kernel(float *partial, float *stats,
        const float *query, const __nv_bfloat16 *key, const __nv_bfloat16 *value,
        uint32_t first, uint32_t heads, uint32_t kv_heads, uint32_t dim,
        uint32_t window, uint32_t ring, uint32_t keys, uint64_t kv_stride,
        uint32_t splits) {
    extern __shared__ __align__(16) __nv_bfloat16 tile[];
    const uint32_t kv_head = blockIdx.x, split = blockIdx.y;
    const uint32_t group = heads / kv_heads;
    const uint32_t warp = threadIdx.x / 32u;
    const uint32_t lane = threadIdx.x % 32u;
    const uint32_t head = kv_head * group + warp;
    const uint32_t position = first;
    const uint32_t begin = window && position + 1 > window ? position + 1 - window : 0u;
    const uint32_t last = keys && keys <= position ? keys - 1u : position;
    const uint32_t span = (last >= begin) ? (last - begin + splits) / splits : 0u;
    const uint32_t k0 = begin + split * span;
    const uint32_t k1 = k0 + span - 1u < last ? k0 + span - 1u : last;
    const uint32_t head_at = split * heads + head;
    if (span == 0u || k0 > last || begin > last) {
        if (lane == 0) {
            stats[head_at * 2u + 0u] = -INFINITY;
            stats[head_at * 2u + 1u] = 0.0f;
        }
        return;
    }
    const float *q = query + (uint64_t)head * dim;
    const uint32_t per_lane = (dim + 31u) / 32u;
    float qv[16], acc[16];
    for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
        const uint32_t d = lane + i * 32u;
        qv[i] = d < dim ? q[d] : 0.0f;
        acc[i] = 0.0f;
    }
    float maximum = -INFINITY, total = 0.0f;
    const uint32_t tile_values = TILE * dim;
    for (uint32_t base = k0; base <= k1; base += TILE) {
        const uint32_t count = k1 - base + 1u < TILE ? k1 - base + 1u : TILE;
        const uint32_t vecs = dim / 8u;
        for (uint32_t t = 0; t < count; t++) {
            const uint32_t k = base + t;
            const uint64_t slot = ring ? (uint64_t)(k % ring) : k;
            const uint4 *ks = (const uint4 *)(key + slot * kv_stride + (uint64_t)kv_head * dim);
            const uint4 *vs = (const uint4 *)(value + slot * kv_stride + (uint64_t)kv_head * dim);
            uint4 *kt = (uint4 *)(tile + (size_t)t * dim);
            uint4 *vt = (uint4 *)(tile + tile_values + (size_t)t * dim);
            for (uint32_t v = threadIdx.x; v < vecs; v += blockDim.x) {
                kt[v] = ks[v];
                vt[v] = vs[v];
            }
        }
        __syncthreads();
        for (uint32_t t = 0; t < count; t++) {
            float dot = 0.0f;
            for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
                const uint32_t d = lane + i * 32u;
                if (d < dim) { dot += qv[i] * __bfloat162float(tile[t * dim + d]); }
            }
            dot = g4_split_warp_sum(dot);
            dot = __shfl_sync(0xffffffffu, dot, 0);
            const float m_new = fmaxf(maximum, dot);
            const float alpha = expf(maximum - m_new);
            const float p = expf(dot - m_new);
            total = total * alpha + p;
            maximum = m_new;
            for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
                const uint32_t d = lane + i * 32u;
                if (d < dim) {
                    acc[i] = acc[i] * alpha +
                        p * __bfloat162float(tile[tile_values + t * dim + d]);
                }
            }
        }
        __syncthreads();
    }
    for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
        const uint32_t d = lane + i * 32u;
        if (d < dim) { partial[(uint64_t)head_at * dim + d] = acc[i]; }
    }
    if (lane == 0) {
        stats[head_at * 2u + 0u] = maximum;
        stats[head_at * 2u + 1u] = total;
    }
}

/* Join the splits: one CTA per head, one thread per output column. The
 * per-split softmax weight lives in dynamic shared memory, one float per
 * split; the host sizes the launch. */
__global__ static void g4_attn_split_combine_kernel(float *out, const float *partial,
        const float *stats, uint32_t heads, uint32_t dim, uint32_t splits) {
    extern __shared__ float weight[];
    const uint32_t head = blockIdx.x;
    float m = -INFINITY;
    for (uint32_t s = 0; s < splits; s++) {
        m = fmaxf(m, stats[((uint64_t)s * heads + head) * 2u + 0u]);
    }
    const float ref = m == -INFINITY ? 0.0f : m;
    for (uint32_t s = threadIdx.x; s < splits; s += blockDim.x) {
        weight[s] = expf(stats[((uint64_t)s * heads + head) * 2u + 0u] - ref);
    }
    __syncthreads();
    float l = 0.0f;
    if (threadIdx.x == 0) {
        for (uint32_t s = 0; s < splits; s++) {
            l += stats[((uint64_t)s * heads + head) * 2u + 1u] * weight[s];
        }
    }
    __shared__ float inv_shared;
    if (threadIdx.x == 0) { inv_shared = l > 0.0f ? 1.0f / l : 0.0f; }
    __syncthreads();
    const float inv = inv_shared;
    float *dst = out + (uint64_t)head * dim;
    for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < splits; s++) {
            acc += partial[((uint64_t)s * heads + head) * dim + d] * weight[s];
        }
        dst[d] = acc * inv;
    }
}

#endif
