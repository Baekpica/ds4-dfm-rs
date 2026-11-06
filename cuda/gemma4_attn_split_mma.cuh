#ifndef DS4_GEMMA4_ATTN_SPLIT_MMA_CUH
#define DS4_GEMMA4_ATTN_SPLIT_MMA_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "gemma4_attn_mma.cuh"

/* Full-layer split-K decode attention on tensor cores.
 *
 * The split path (cuda/gemma4_attn_split.cuh) computes one scalar dot per
 * (head, key) on CUDA cores; at 13,447 keys the five full layers are about
 * 11.3 ms of the 19.3 ms decode attention. This kernel keeps the split
 * structure exactly (one CTA per KV head and key range, the same partial and
 * stats layout, the same combine kernel) and moves the per-key arithmetic to
 * m16n8k16: the eight query heads of a KV head sit in M rows 0-7, the key
 * tile in N, and head_dim = 512 in K. Q K^T and P V both run on tensor
 * cores, and the online softmax rescales once per 64-key tile instead of
 * once per key.
 *
 * Geometry: D = 512, group = 8, TK = 64 keys, 8 warps (256 threads).
 *   - Q is staged once as 16 rows x 512 bf16 (rows 8-15 zero) and read
 *     through ldmatrix. The query is already bf16-rounded by
 *     g4_rms_rope_kernel, so the conversion is 0 ULP.
 *   - Each warp owns one n8 key tile for Q K^T (8 warps = 64 keys) and 64
 *     output dims for P V.
 *   - K and V share one 64-key staging buffer: K is consumed by Q K^T, then
 *     V overwrites it for P V.
 *   - P is rounded to bf16 before P V, as in the prefill MMA kernel, so this
 *     is a parity path (the split combine already differs from the fast
 *     kernel). The gate is a tolerance against the f32 reference.
 *
 * Chunk invariance: the key loop starts at (k0 / TK) * TK and masks every
 * key outside [k0, k1]; split boundaries depend only on absolute key
 * indices, so the same split sees the same tiles at every context length.
 *
 * Shared memory: Q 16 KiB + K/V 64 KiB + P 1 KiB + two 8x8 float reductions
 * = 81.5 KiB, above the 48 KiB default and under the 99 KiB sm_89 limit, so
 * the host grants the dynamic-shared limit once. */

enum { G4_SMMA_TK = 64u, G4_SMMA_THREADS = 256u, G4_SMMA_D = 512u,
       G4_SMMA_GROUP = 8u };

__device__ __forceinline__ void g4_smma_ldsm2(uint32_t *r, const void *p) {
    /* Two 8x8 bf16 matrices: lanes 0-7 address matrix 0, lanes 8-15 matrix 1
     * (the B fragment pair of one m16n8k16). */
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

/* One (KV head, key split) CTA. `k0..k1` are the split's absolute keys; the
 * host divides the visible range with DS4_GEMMA4_ATTN_SPLIT_SPAN and passes
 * the resulting `splits`, and the kernel re-derives the same even division
 * the scalar split kernel uses.
 *
 * FP8 = true reads the combined E4M3 rows of ds4_gemma4_full_kv_row instead
 * of the bf16 pair (K at kv_head * D bytes, V at kv_dim + kv_head * D, the
 * two f16 scale arrays at 2 * kv_dim) and decodes each 16-byte shared slot
 * with g4_fp8_decode8; the split division, the partial/stats layout, the
 * combine kernel and the softmax are the bf16 path's. */
template <bool FP8>
__global__ __launch_bounds__(G4_SMMA_THREADS, 1) void g4_attn_split_mma_kernel(
        float *partial, float *stats, const float *query,
        const __nv_bfloat16 *key, const __nv_bfloat16 *value,
        const uint8_t *fp8, uint64_t row_bytes,
        uint32_t first, uint32_t heads, uint32_t kv_heads, uint32_t dim,
        uint32_t window, uint32_t ring, uint32_t keys, uint64_t kv_stride,
        uint32_t splits) {
    constexpr uint32_t D = G4_SMMA_D, TK = G4_SMMA_TK, GROUP = G4_SMMA_GROUP;
    using E = typename g4_mma_elem<FP8>::type;
    const uint64_t kv_dim = (uint64_t)kv_heads * D;
    extern __shared__ __align__(16) uint8_t smem[];
    E *qs = (E *)smem;                           /* 16 x D */
    E *kv = qs + 16u * D;                        /* TK x D, K then V */
    E *ps = kv + TK * D;                         /* GROUP x TK */
    float *red_max = (float *)(ps + GROUP * TK);
    float *red_sum = red_max + 8u * 8u;

    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, q = lane & 3u;
    const uint32_t kv_head = blockIdx.x, split = blockIdx.y;
    const uint32_t position = first;
    const uint32_t begin = window && position + 1u > window ? position + 1u - window : 0u;
    const uint32_t last = keys && keys <= position ? keys - 1u : position;
    const uint32_t span = (last >= begin) ? (last - begin + splits) / splits : 0u;
    const uint32_t k0 = begin + split * span;
    const uint32_t k1 = k0 + span - 1u < last ? k0 + span - 1u : last;
    if (span == 0u || k0 > last || begin > last) {
        if (lane == 0u) {
            const uint32_t head = kv_head * GROUP + warp;
            const uint64_t at = ((uint64_t)split * heads + head) * 2u;
            stats[at + 0u] = -INFINITY;
            stats[at + 1u] = 0.0f;
        }
        return;
    }

    /* Q: rows 0-7 are this KV head's query heads, rows 8-15 zero (the M
     * rows the 8-head packing leaves unused). */
    for (uint32_t i = tid; i < 16u * (D / 8u); i += G4_SMMA_THREADS) {
        const uint32_t row = i / (D / 8u), chunk = i % (D / 8u);
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (row < GROUP) {
            const float *src = query + (uint64_t)(kv_head * GROUP + row) * D;
            uint32_t *w = reinterpret_cast<uint32_t *>(&v);
            #pragma unroll
            for (uint32_t e = 0; e < 4u; e++) {
                w[e] = g4_mma_pack_t<FP8>(src[chunk * 8u + 2u * e],
                                          src[chunk * 8u + 2u * e + 1u]);
            }
        }
        *(uint4 *)(qs + g4_mma_swz(row, chunk, D)) = v;
    }
    __syncthreads();

    float o[8][4] = {};
    float m = -INFINITY, l = 0.0f;
    const uint32_t base_start = (k0 / TK) * TK;
    for (uint32_t base = base_start; base <= k1; base += TK) {
        /* Stage K (a key outside this split's range is zero-filled and
         * masked below). */
        if constexpr (FP8) {
            for (uint32_t i = tid; i < TK * (D / 8u); i += G4_SMMA_THREADS) {
                const uint32_t t = i / (D / 8u), c = i % (D / 8u);
                const uint32_t k_abs = base + t;
                const bool valid = k_abs >= k0 && k_abs <= k1;
                const uint32_t slot = valid ? (ring ? k_abs % ring : k_abs) : 0u;
                const uint8_t *row = fp8 + (uint64_t)slot * row_bytes;
                const __half kscale = ((const __half *)(row + 2u * kv_dim))[kv_head];
                const uint8_t *src = row + (uint64_t)kv_head * D;
                const uint2 raw = valid ? __ldg((const uint2 *)(src + c * 8u))
                                        : make_uint2(0u, 0u);
                *(uint4 *)(kv + g4_mma_swz(t, c, D)) = g4_fp8_decode8(raw, kscale);
            }
        } else {
            for (uint32_t i = tid; i < TK * (D / 8u); i += G4_SMMA_THREADS) {
                const uint32_t t = i / (D / 8u), c = i % (D / 8u);
                const uint32_t k_abs = base + t;
                const bool valid = k_abs >= k0 && k_abs <= k1;
                const uint32_t slot = valid ? (ring ? k_abs % ring : k_abs) : 0u;
                const __nv_bfloat16 *src = key + (uint64_t)slot * kv_stride + (uint64_t)kv_head * D;
                g4_mma_cp16(kv + g4_mma_swz(t, c, D), src + c * 8u, valid ? 16u : 0u);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();

        /* S = Q K^T: this warp's 8 keys, 32 k16 steps of D = 512. */
        float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (uint32_t k = 0; k < D / 16u; k++) {
            uint32_t qa[4];
            const uint32_t mat = lane >> 3;
            const uint32_t qrow = (mat & 1u) * 8u + (lane & 7u);
            const uint32_t qchunk = k * 2u + (mat >> 1);
            g4_mma_ldsm4(qa, qs + g4_mma_swz(qrow, qchunk, D), false);
            uint32_t b[2];
            const uint32_t brow = warp * 8u + (lane & 7u);
            const uint32_t bchunk = k * 2u + ((lane >> 3) & 1u);
            g4_smma_ldsm2(b, kv + g4_mma_swz(brow, bchunk, D));
            g4_mma_mma_t<FP8>(s, qa, b[0], b[1]);
        }

        /* Mask and online softmax; only row g (heads 0-7) is valid. The
         * tile max and sum cross the 8 warps through shared memory. */
        const uint32_t kk0 = base + warp * 8u + 2u * q;
        float peak = -INFINITY;
        #pragma unroll
        for (uint32_t e = 0; e < 2u; e++) {
            const uint32_t k_abs = kk0 + e;
            if (k_abs < k0 || k_abs > k1) { s[e] = -INFINITY; }
            peak = fmaxf(peak, s[e]);
        }
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, 1));
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, 2));
        if (q == 0u) { red_max[warp * 8u + g] = peak; }
        __syncthreads();
        float tile_max = -INFINITY;
        #pragma unroll
        for (uint32_t w = 0; w < 8u; w++) { tile_max = fmaxf(tile_max, red_max[w * 8u + g]); }
        const bool any = tile_max > -INFINITY;
        const float mn = any ? fmaxf(m, tile_max) : m;
        const float alpha = any ? (m == -INFINITY ? 0.0f : expf(m - mn)) : 1.0f;
        const float p0 = any ? expf(s[0] - mn) : 0.0f;
        const float p1 = any ? expf(s[1] - mn) : 0.0f;
        ps[g * TK + warp * 8u + 2u * q] = g4_mma_store_t<FP8>(p0);
        ps[g * TK + warp * 8u + 2u * q + 1u] = g4_mma_store_t<FP8>(p1);
        float lsum = p0 + p1;
        lsum += __shfl_xor_sync(0xffffffffu, lsum, 1);
        lsum += __shfl_xor_sync(0xffffffffu, lsum, 2);
        if (q == 0u) { red_sum[warp * 8u + g] = lsum; }
        __syncthreads();
        float tile_sum = 0.0f;
        #pragma unroll
        for (uint32_t w = 0; w < 8u; w++) { tile_sum += red_sum[w * 8u + g]; }
        l = l * alpha + tile_sum;
        m = mn;
        #pragma unroll
        for (uint32_t t = 0; t < 8u; t++) {
            o[t][0] *= alpha;
            o[t][1] *= alpha;
        }

        /* Stage V over K; P is already in shared. */
        if constexpr (FP8) {
            for (uint32_t i = tid; i < TK * (D / 8u); i += G4_SMMA_THREADS) {
                const uint32_t t = i / (D / 8u), c = i % (D / 8u);
                const uint32_t k_abs = base + t;
                const bool valid = k_abs >= k0 && k_abs <= k1;
                const uint32_t slot = valid ? (ring ? k_abs % ring : k_abs) : 0u;
                const uint8_t *row = fp8 + (uint64_t)slot * row_bytes;
                const __half vscale = ((const __half *)(row + 2u * kv_dim))[kv_heads + kv_head];
                const uint8_t *src = row + kv_dim + (uint64_t)kv_head * D;
                const uint2 raw = valid ? __ldg((const uint2 *)(src + c * 8u))
                                        : make_uint2(0u, 0u);
                *(uint4 *)(kv + g4_mma_swz(t, c, D)) = g4_fp8_decode8(raw, vscale);
            }
        } else {
            for (uint32_t i = tid; i < TK * (D / 8u); i += G4_SMMA_THREADS) {
                const uint32_t t = i / (D / 8u), c = i % (D / 8u);
                const uint32_t k_abs = base + t;
                const bool valid = k_abs >= k0 && k_abs <= k1;
                const uint32_t slot = valid ? (ring ? k_abs % ring : k_abs) : 0u;
                const __nv_bfloat16 *src = value + (uint64_t)slot * kv_stride + (uint64_t)kv_head * D;
                g4_mma_cp16(kv + g4_mma_swz(t, c, D), src + c * 8u, valid ? 16u : 0u);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();

        /* O += P V: four k16 steps over the 64 keys, 8 n8 output tiles per
         * warp (warp w owns dims [64w, 64w + 64)). */
        #pragma unroll
        for (uint32_t kk = 0; kk < 4u; kk++) {
            const uint32_t pa[4] = {
                g4_mma_pack_t<FP8>(g4_mma_load_t<FP8>(ps[g * TK + kk * 16u + 2u * q]),
                                   g4_mma_load_t<FP8>(ps[g * TK + kk * 16u + 2u * q + 1u])),
                0u,
                g4_mma_pack_t<FP8>(g4_mma_load_t<FP8>(ps[g * TK + kk * 16u + 2u * q + 8u]),
                                   g4_mma_load_t<FP8>(ps[g * TK + kk * 16u + 2u * q + 9u])),
                0u};
            #pragma unroll
            for (uint32_t t = 0; t < 8u; t += 2u) {
                uint32_t b[4];
                const uint32_t mat = lane >> 3;
                const uint32_t krow = kk * 16u + (mat & 1u) * 8u + (lane & 7u);
                const uint32_t chunk = warp * 8u + t + (mat >> 1);
                g4_mma_ldsm4(b, kv + g4_mma_swz(krow, chunk, D), true);
                g4_mma_mma_t<FP8>(o[t], pa, b[0], b[1]);
                g4_mma_mma_t<FP8>(o[t + 1u], pa, b[2], b[3]);
            }
        }
        __syncthreads();
    }

    const uint32_t head = kv_head * GROUP + g;
    float *dst = partial + ((uint64_t)split * heads + head) * D;
    #pragma unroll
    for (uint32_t t = 0; t < 8u; t++) {
        const uint32_t d = warp * 64u + t * 8u + 2u * q;
        dst[d] = o[t][0];
        dst[d + 1u] = o[t][1];
    }
    if (q == 0u) {
        stats[((uint64_t)split * heads + head) * 2u + 0u] = m;
        stats[((uint64_t)split * heads + head) * 2u + 1u] = l;
    }
}

#endif
