#ifndef DS4_GEMMA4_ATTN_FAST_CUH
#define DS4_GEMMA4_ATTN_FAST_CUH

/* Decode-only fast attention for Gemma-4: one CTA per KV head, the group's
 * query heads sharing one shared-memory K/V tile, instead of one warp per
 * (row, head) each reading the same keys from global memory
 * (cuda/gemma4_attn.cuh:41-96). The technique is the head-grouping and tile
 * reuse of llama.cpp's flash attention (ggml/src/ggml-cuda/fattn.cu), kept
 * to CUDA cores so the arithmetic stays the one the oracle uses: the same
 * per-lane dot order, the same warp reduction, the same online softmax and
 * the same float32 accumulators. Only the K/V load path changes, so a fast
 * run is expected to match the oracle bit for bit.
 *
 * The block covers one KV head: group = heads / kv_heads query heads, one
 * warp each. The 25 sliding layers have group 2 (16 query heads, 8 KV
 * heads) and the 5 full layers group 8 (16 query heads, 2 KV heads); the
 * full layers are where the oracle's 8x redundant key reads dominate. The
 * kernel is used only when rows == 1 (decode) and keys == 0 (the trunk, not
 * the MTP drafter); the oracle stays for prefill and for the drafter. */

#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ static float g4_fa_warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

/* TILE keys are staged per iteration; dim 512 gives 2 * 16 * 512 * 2 = 32 KiB
 * of shared memory, dim 256 gives 16 KiB. */
template <uint32_t TILE>
__global__ static void g4_attn_fast_kernel(float *out, const float *query,
        const __nv_bfloat16 *key, const __nv_bfloat16 *value,
        uint32_t first, uint32_t heads, uint32_t kv_heads, uint32_t dim,
        uint32_t window, uint32_t ring, uint32_t keys, uint64_t kv_stride) {
    extern __shared__ __align__(16) __nv_bfloat16 tile[];
    const uint32_t kv_head = blockIdx.x;
    const uint32_t group = heads / kv_heads;
    const uint32_t warp = threadIdx.x / 32u;
    const uint32_t lane = threadIdx.x % 32u;
    const uint32_t head = kv_head * group + warp;
    const uint32_t position = first;
    const uint32_t begin = window && position + 1 > window ? position + 1 - window : 0u;
    const uint32_t last = keys && keys <= position ? keys - 1u : position;
    const float *q = query + (uint64_t)head * dim;
    float *o = out + (uint64_t)head * dim;
    const uint32_t per_lane = (dim + 31u) / 32u;
    float qv[16], acc[16];
    for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
        const uint32_t d = lane + i * 32u;
        qv[i] = d < dim ? q[d] : 0.0f;
        acc[i] = 0.0f;
    }
    float maximum = -INFINITY, total = 0.0f;
    const uint32_t tile_values = TILE * dim;
    for (uint32_t base = begin; base <= last; base += TILE) {
        const uint32_t count = last - base + 1 < TILE ? last - base + 1 : TILE;
        /* Vectorized staging (uint4 = 8 bf16, and dim and kv_stride are
         * multiples of 8): the 2-byte copy this loop used to do cost more
         * than the arithmetic at ctx 256. */
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
                if (d < dim) {
                    dot += qv[i] * __bfloat162float(tile[t * dim + d]);
                }
            }
            dot = g4_fa_warp_sum(dot);
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
    const float inverse = 1.0f / total;
    for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
        const uint32_t d = lane + i * 32u;
        if (d < dim) { o[d] = acc[i] * inverse; }
    }
}

#endif
