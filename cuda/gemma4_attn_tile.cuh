#ifndef DS4_GEMMA4_ATTN_TILE_CUH
#define DS4_GEMMA4_ATTN_TILE_CUH

/* Prefill (rows > 1) fast attention for Gemma-4: one CTA per (row tile, KV
 * head), so the group's query heads and the tile's rows share one staged
 * K/V tile. The arithmetic is the oracle's, unchanged: the same per-lane
 * dot order, the same warp reduction, the same float32 online softmax and
 * the same accumulators (cuda/gemma4_attn.cuh:41-96), so a fast run is
 * expected to match the oracle bit for bit. Only where the K/V bytes come
 * from changes. The oracle gives every (row, head) warp its own global
 * reads, which re-reads each KV row once per query head -- 16 times per
 * layer, measured at 1.12 ms per layer at ctx 256 on the 4070 SUPER with
 * 512 MB of L1/L2 traffic. The tile stages a KV row once for the block's
 * 8 warps: 8x less traffic, and the staging is vectorized (uint4 = 8 bf16)
 * instead of one 2-byte copy per element.
 *
 * Geometry: the sliding layers have 16 query heads over 8 KV heads (group
 * 2) and dim 256; the full layers 16 over 2 (group 8) and dim 512. A CTA is
 * always 8 warps (256 threads): 4 rows x 2 heads for the sliding layers,
 * 1 row x 8 heads for the full ones. TILE keys are staged per iteration;
 * TILE is 16 for dim 256 and 8 for dim 512 so the tile is 16 KiB either
 * way, which keeps three blocks per SM under the 48 KiB default carveout.
 *
 * The tile range is the union of the block's rows ([begin of the first row,
 * last key of the last row]); each warp then walks only its own
 * [begin, last] inside it, so the causal and sliding windows are exactly
 * the oracle's. */

#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ static float g4_tile_warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

/* `rtile` is the row tile (4 sliding, 1 full); `group` is heads / kv_heads.
 * The block has rtile * group warps. */
template <uint32_t TILE>
__global__ static void g4_attn_tile_kernel(float *out, const float *query,
        const __nv_bfloat16 *key, const __nv_bfloat16 *value,
        uint32_t rows, uint32_t first, uint32_t heads, uint32_t kv_heads,
        uint32_t dim, uint32_t window, uint32_t ring, uint32_t keys,
        uint64_t kv_stride, uint32_t rtile) {
    extern __shared__ __align__(16) __nv_bfloat16 tile[];
    __nv_bfloat16 *ktile = tile;
    __nv_bfloat16 *vtile = tile + (size_t)TILE * dim;
    const uint32_t group = heads / kv_heads;
    const uint32_t kv_head = blockIdx.x;
    const uint32_t warp = threadIdx.x / 32u;
    const uint32_t lane = threadIdx.x % 32u;
    const uint32_t row_in = warp / group;
    const uint32_t head_in = warp % group;
    const uint32_t row0 = blockIdx.y * rtile;
    const uint32_t row_last = row0 + rtile - 1u < rows ? row0 + rtile - 1u : rows - 1u;
    /* Block-uniform tile range: begin is monotone in the row and last is
     * monotone in the row, so the first row's begin and the last row's last
     * key bound every warp in the block. */
    const uint32_t pos0 = first + row0;
    const uint32_t begin_all = window && pos0 + 1u > window ? pos0 + 1u - window : 0u;
    const uint32_t pos_last = first + row_last;
    const uint32_t last_all = keys && keys <= pos_last ? keys - 1u : pos_last;

    const uint32_t row = row0 + row_in;
    const bool valid = row < rows;
    const uint32_t position = first + row;
    const uint32_t head = kv_head * group + head_in;
    const uint32_t begin = window && position + 1u > window ? position + 1u - window : 0u;
    const uint32_t last = keys && keys <= position ? keys - 1u : position;
    const float *q = query + (uint64_t)row * heads * dim + (uint64_t)head * dim;
    float *o = out + (uint64_t)row * heads * dim + (uint64_t)head * dim;

    const uint32_t per_lane = (dim + 31u) / 32u;
    float qv[16], acc[16];
    for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
        const uint32_t d = lane + i * 32u;
        qv[i] = valid && d < dim ? q[d] : 0.0f;
        acc[i] = 0.0f;
    }
    float maximum = -INFINITY, total = 0.0f;

    const uint32_t vecs = dim / 8u;   /* uint4 = 8 bf16 */
    for (uint32_t base = begin_all; base <= last_all; base += TILE) {
        const uint32_t count = last_all - base + 1u < TILE ? last_all - base + 1u : TILE;
        const uint32_t vec_total = count * vecs;
        for (uint32_t i = threadIdx.x; i < vec_total; i += blockDim.x) {
            const uint32_t t = i / vecs;
            const uint32_t v = i - t * vecs;
            const uint32_t k = base + t;
            const uint64_t slot = ring ? (uint64_t)(k % ring) : k;
            const uint64_t off = slot * kv_stride + (uint64_t)kv_head * dim + (uint64_t)v * 8u;
            ((uint4 *)(ktile + (size_t)t * dim))[v] = *(const uint4 *)(key + off);
            ((uint4 *)(vtile + (size_t)t * dim))[v] = *(const uint4 *)(value + off);
        }
        __syncthreads();
        const uint32_t k_first = base > begin ? base : begin;
        const uint32_t k_end = base + count - 1u < last ? base + count - 1u : last;
        for (uint32_t k = k_first; valid && k <= k_end; k++) {
            const uint32_t t = k - base;
            float dot = 0.0f;
            for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
                const uint32_t d = lane + i * 32u;
                if (d < dim) { dot += qv[i] * __bfloat162float(ktile[(size_t)t * dim + d]); }
            }
            dot = g4_tile_warp_sum(dot);
            dot = __shfl_sync(0xffffffffu, dot, 0);
            /* The oracle's float32 online softmax (cuda/gemma4_attn.cuh). */
            const float m_new = fmaxf(maximum, dot);
            const float alpha = expf(maximum - m_new);
            const float p = expf(dot - m_new);
            total = total * alpha + p;
            maximum = m_new;
            for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
                const uint32_t d = lane + i * 32u;
                if (d < dim) {
                    acc[i] = acc[i] * alpha +
                        p * __bfloat162float(vtile[(size_t)t * dim + d]);
                }
            }
        }
        __syncthreads();
    }
    const float inverse = 1.0f / total;
    for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
        const uint32_t d = lane + i * 32u;
        if (valid && d < dim) { o[d] = acc[i] * inverse; }
    }
}

#endif
