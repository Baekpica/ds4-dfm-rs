#ifndef DS4_K2H_ATTN_CUH
#define DS4_K2H_ATTN_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <stdint.h>

#include "inkling2_attn.cuh"

/* K2-Horizon attention (k2h_ref.attention: SDPA, causal, scale 128^-1/2):
 * 48 query heads over 8 KV heads of 128, every layer over every key from 0.
 * Scores stay F32 (flash attention keeps S in F32), the softmax runs online
 * in F32, BF16 probabilities multiply BF16 V, out = BF16(acc / l).
 *
 * K/V live in Inkling-2's NVFP4 page layout (cuda/inkling2_attn.cuh) and
 * decode to BF16 exactly in shared memory. Prefill rows run on BF16 tensor
 * cores, Inkling-2's tile: a CTA owns 64 queries of two heads sharing a KV
 * head, 64-key tiles through a two-stage cp.async ring of raw pages. Key
 * tiles start at multiples of 64 in absolute position and masked keys add
 * exact zeros, so a query's result does not depend on the chunk. Decode
 * rows split their keys over CTAs (split, KV head, row; a warp per query
 * head) and a merge folds the partials. */

enum {
    K2A_HEADS = INK2A_HEADS, K2A_GROUP = INK2A_GROUP, K2A_DIM = INK2A_DIM,
    K2A_SPLITS = INK2A_SPLITS, K2A_PARTIAL = INK2A_PARTIAL, K2A_MIN_SPLIT = INK2A_MIN_SPLIT,
};

// SDPA's scale 128^-1/2, applied to the F32 dot.
#define K2A_SCALE 0.08838834764831845f

__global__ __launch_bounds__(INK2A_THREADS, 1)
void k2a_prefill(float *out, const float *q, const uint8_t *pages, uint32_t rows, uint32_t base) {
    extern __shared__ __align__(16) uint16_t k2a_smem[];
    // A two-stage ring of raw NVFP4 tiles after one decoded BF16 tile.
    uint8_t *raw = (uint8_t *)(k2a_smem + INK2A_STAGE_ELEMS);
    auto s_k = reinterpret_cast<uint16_t (*)[INK2A_SROW]>(k2a_smem);
    auto s_v = s_k + INK2A_TK;
    const uint32_t tq0 = blockIdx.x * INK2A_TQ, h0 = blockIdx.y * INK2A_CTA_HEADS, kv_head = h0 / INK2A_GROUP;
    const uint32_t h = h0 + threadIdx.x / (INK2A_WARPS * 32);
    const uint32_t local = threadIdx.x % (INK2A_WARPS * 32), warp = local / 32, lane = local % 32;
    if (tq0 >= rows) { return; }

    // C-fragment rows lane/4 and lane/4 + 8 of this warp's 16-query tile.
    uint32_t qrow[2], qpos[2];
    bool alive[2];
    float row_m[2] = {-INFINITY, -INFINITY}, row_l[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        qrow[r] = warp * INK2A_WQ + lane / 4 + r * 8;
        alive[r] = tq0 + qrow[r] < rows;
        qpos[r] = base + (alive[r] ? tq0 + qrow[r] : 0);
    }
    uint32_t qa[INK2A_DIM / 16][4];
    #pragma unroll
    for (int kc = 0; kc < INK2A_DIM / 16; kc++) {
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            const int r = l % 2, j = (l / 2) * 4 + (int)(lane % 4);
            const uint32_t t = alive[r] ? tq0 + qrow[r] : 0;
            const float2 xy = *reinterpret_cast<const float2 *>(
                q + ((uint64_t)t * INK2A_HEADS + h) * INK2A_DIM + kc * 16 + 2 * j);
            qa[kc][l] = ink2a_pack(xy.x, xy.y);
        }
    }

    const uint32_t key_hi = base + (tq0 + INK2A_TQ - 1 < rows ? tq0 + INK2A_TQ - 1 : rows - 1);
    const uint32_t tiles = key_hi / INK2A_TK + 1u;
    float output[INK2A_DIM / 8][4];
    #pragma unroll
    for (int cb = 0; cb < INK2A_DIM / 8; cb++) {
        #pragma unroll
        for (int l = 0; l < 4; l++) { output[cb][l] = 0.0f; }
    }

    // A 64-key tile is half of one block: four contiguous segments.
    auto issue_tile = [&](int stage, uint32_t kt0) {
        const uint8_t *page = ink2a_page(pages, kt0, kv_head);
        const uint32_t slot = kt0 % INK2A_KB;
        uint8_t *dst = raw + stage * INK2A_RAW;
        for (uint32_t c = threadIdx.x; c < INK2A_RAW / 16; c += INK2A_THREADS) {
            const uint32_t at = c * 16, half = at / INK2A_RAW_HALF, in = at % INK2A_RAW_HALF;
            const uint32_t data = INK2A_TK * INK2A_FP4;
            const uint8_t *src = page + half * INK2A_KB * INK2A_FP4_ROW +
                (in < data ? slot * INK2A_FP4 + in : INK2A_KB * INK2A_FP4 + slot * 8 + (in - data));
            __pipeline_memcpy_async(dst + at, src, 16);
        }
    };
    // Raw NVFP4 tile -> BF16 rows; keys past key_hi are zeros (their bytes
    // may be stale or never written).
    auto decode_tile = [&](int stage, uint32_t kt0) {
        const uint8_t *src = raw + stage * INK2A_RAW;
        for (uint32_t g = threadIdx.x; g < 2 * INK2A_TK * 8; g += INK2A_THREADS) {
            const uint32_t value = g / (INK2A_TK * 8), key = (g / 8) % INK2A_TK, grp = g % 8;
            const uint8_t *half = src + value * INK2A_RAW_HALF;
            uint32_t packed[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (kt0 + key <= key_hi) {
                const uint2 data = *reinterpret_cast<const uint2 *>(half + key * INK2A_FP4 + grp * 8);
                ink2a_fp4x16(data, half[INK2A_TK * INK2A_FP4 + key * 8 + grp], packed);
            }
            uint16_t *dst = (value ? s_v[key] : s_k[key]) + grp * 16;
            *reinterpret_cast<uint4 *>(dst) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
            *reinterpret_cast<uint4 *>(dst + 8) = make_uint4(packed[4], packed[5], packed[6], packed[7]);
        }
    };
    auto consume = [&](uint32_t kt0) {
        float s[INK2A_NB][4];
        const uint32_t k_row = lane & 7u, k_col = (lane >> 3) * 8u;
        #pragma unroll
        for (int nb = 0; nb < INK2A_NB; nb++) {
            #pragma unroll
            for (int l = 0; l < 4; l++) { s[nb][l] = 0.0f; }
            #pragma unroll
            for (int kc = 0; kc < INK2A_DIM / 16; kc += 2) {
                uint32_t b0, b1, b2, b3;
                ink2a_ldsm4(b0, b1, b2, b3, &s_k[nb * 8 + k_row][kc * 16 + k_col]);
                ink2a_mma(s[nb], qa[kc], b0, b1);
                ink2a_mma(s[nb], qa[kc + 1], b2, b3);
            }
        }
        float tile_max[2] = {-INFINITY, -INFINITY};
        #pragma unroll
        for (int nb = 0; nb < INK2A_NB; nb++) {
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const int r = l / 2;
                const uint32_t p = kt0 + nb * 8 + (lane % 4) * 2 + (l % 2);
                const float x = alive[r] && p <= qpos[r] ? s[nb][l] * K2A_SCALE : -INFINITY;
                s[nb][l] = x;
                tile_max[r] = fmaxf(tile_max[r], x);
            }
        }
        float rescale[2], tile_sum[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 1));
            tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 2));
            const float next = fmaxf(row_m[r], tile_max[r]);
            rescale[r] = row_m[r] == -INFINITY ? 0.0f : __expf(row_m[r] - next);
            row_m[r] = next;
        }
        uint32_t pa[INK2A_KS][4];
        #pragma unroll
        for (int nb = 0; nb < INK2A_NB; nb++) {
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const int r = l / 2;
                const float w = s[nb][l] == -INFINITY ? 0.0f : __expf(s[nb][l] - row_m[r]);
                s[nb][l] = w;
                tile_sum[r] += w;
            }
        }
        #pragma unroll
        for (int ks = 0; ks < INK2A_KS; ks++) {
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const float *src = s[2 * ks + l / 2];
                pa[ks][l] = ink2a_pack(src[(l % 2) * 2], src[(l % 2) * 2 + 1]);
            }
        }
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            tile_sum[r] += __shfl_xor_sync(0xffffffffu, tile_sum[r], 1);
            tile_sum[r] += __shfl_xor_sync(0xffffffffu, tile_sum[r], 2);
            row_l[r] = row_l[r] * rescale[r] + tile_sum[r];
        }
        #pragma unroll
        for (int cb = 0; cb < INK2A_DIM / 8; cb++) {
            #pragma unroll
            for (int l = 0; l < 4; l++) { output[cb][l] *= rescale[l / 2]; }
        }
        const uint32_t v_row = ((lane >> 3) & 1u) * 8u + (lane & 7u), v_col = (lane >> 4) * 8u;
        #pragma unroll
        for (int ks = 0; ks < INK2A_KS; ks++) {
            #pragma unroll
            for (int cb = 0; cb < INK2A_DIM / 8; cb += 2) {
                uint32_t b0, b1, b2, b3;
                ink2a_ldsm4t(b0, b1, b2, b3, &s_v[ks * 16 + v_row][cb * 8 + v_col]);
                ink2a_mma(output[cb], pa[ks], b0, b1);
                ink2a_mma(output[cb + 1], pa[ks], b2, b3);
            }
        }
    };

    issue_tile(0, 0);
    __pipeline_commit();
    int stage = 0;
    #pragma unroll 1
    for (uint32_t t = 0; t < tiles; t++) {
        const uint32_t kt0 = t * INK2A_TK;
        if (t + 1 < tiles) {
            issue_tile(stage ^ 1, kt0 + INK2A_TK);
            __pipeline_commit();
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
        decode_tile(stage, kt0);
        __syncthreads();
        consume(kt0);
        __syncthreads();
        stage ^= 1;
    }

    #pragma unroll
    for (int cb = 0; cb < INK2A_DIM / 8; cb++) {
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (!alive[r]) { continue; }
            const uint32_t t = tq0 + qrow[r];
            const uint64_t at = ((uint64_t)t * INK2A_HEADS + h) * INK2A_DIM + cb * 8 + (lane % 4) * 2;
            *reinterpret_cast<float2 *>(out + at) = make_float2(
                ink2a_bf16(__fdiv_rn(output[cb][2 * r], row_l[r])),
                ink2a_bf16(__fdiv_rn(output[cb][2 * r + 1], row_l[r])));
        }
    }
}

/* Decode partials: CTA (split, KV head, row); warp w scores head 6h + w over
 * the split's keys, a lane per key of each 32-key tile staged in shared
 * memory (lane j reads K row j from word (i + j) % 64 on, so the 32 rows'
 * words fall in distinct banks). partial = (m, l, acc[128]). */
__global__ __launch_bounds__(INK2A_DTHREADS)
void k2a_decode(float *partials, const float *q, const uint8_t *pages, uint32_t base, uint32_t splits) {
    __shared__ __align__(16) uint32_t sk[INK2A_DTK][INK2A_DIM / 2], sv[INK2A_DTK][INK2A_DIM / 2];
    __shared__ float sq[INK2A_GROUP][INK2A_DIM];
    const uint32_t split = blockIdx.x, kv = blockIdx.y, row = blockIdx.z;
    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32, h = kv * INK2A_GROUP + warp;
    const uint32_t keys = base + row + 1;
    const uint32_t span = ((keys + splits - 1) / splits + INK2A_DTK - 1) / INK2A_DTK * INK2A_DTK;
    const uint32_t k0 = split * span, k1 = min(keys, k0 + span);
    for (uint32_t i = lane; i < INK2A_DIM; i += 32) {
        sq[warp][i] = ink2a_bf16(q[((uint64_t)row * INK2A_HEADS + h) * INK2A_DIM + i]);
    }
    float m = -INFINITY, l = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t t0 = k0; t0 < k1; t0 += INK2A_DTK) {
        __syncthreads();
        // Issue the tile's raw loads before any conversion consumes them.
        constexpr int batches = (INK2A_DTK * 2 * 8 + INK2A_DTHREADS - 1) / INK2A_DTHREADS;
        uint2 data[batches];
        uint32_t scales[batches];
        #pragma unroll
        for (int batch = 0; batch < batches; batch++) {
            const uint32_t i = threadIdx.x + batch * INK2A_DTHREADS;
            if (i >= INK2A_DTK * 2 * 8) { continue; }
            const uint32_t j = i / 16, value = (i / 8) % 2, grp = i % 8;
            const uint32_t p = min(t0 + j, k1 - 1), slot = p % INK2A_KB;
            const uint8_t *half = ink2a_page(pages, p, kv) + value * INK2A_KB * INK2A_FP4_ROW;
            asm volatile("ld.global.v2.u32 {%0, %1}, [%2];"
                : "=r"(data[batch].x), "=r"(data[batch].y) : "l"(half + slot * INK2A_FP4 + grp * 8) : "memory");
            asm volatile("ld.global.u8 %0, [%1];" : "=r"(scales[batch])
                : "l"(half + INK2A_KB * INK2A_FP4 + slot * 8 + grp) : "memory");
        }
        #pragma unroll
        for (int batch = 0; batch < batches; batch++) {
            const uint32_t i = threadIdx.x + batch * INK2A_DTHREADS;
            if (i >= INK2A_DTK * 2 * 8) { continue; }
            const uint32_t j = i / 16, value = (i / 8) % 2, grp = i % 8;
            uint32_t *dst = (value ? sv[j] : sk[j]) + grp * 8;
            ink2a_fp4x16(data[batch], scales[batch], dst);
        }
        __syncthreads();
        const uint32_t p = t0 + lane;
        float score = -INFINITY;
        if (p < k1) {
            float dot = 0.0f;
            #pragma unroll 8
            for (uint32_t i = 0; i < INK2A_DIM / 2; i++) {
                const uint32_t w = (i + lane) & (INK2A_DIM / 2 - 1);
                const uint32_t bits = sk[lane][w];
                dot = fmaf(__uint_as_float(bits << 16), sq[warp][2 * w], dot);
                dot = fmaf(__uint_as_float(bits & 0xffff0000u), sq[warp][2 * w + 1], dot);
            }
            score = dot * K2A_SCALE;
        }
        float tmax = score;
        for (uint32_t o = 16; o; o >>= 1) { tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o)); }
        const float next = fmaxf(m, tmax);
        const float scale = m == -INFINITY ? 0.0f : __expf(m - next);
        const float w = score == -INFINITY ? 0.0f : __expf(score - next);
        float sum = w;
        for (uint32_t o = 16; o; o >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, o); }
        l = l * scale + sum;
        m = next;
        #pragma unroll
        for (int d = 0; d < 4; d++) { acc[d] *= scale; }
        for (uint32_t j = 0; j < INK2A_DTK; j++) {
            const float pj = __shfl_sync(0xffffffffu, w, j);
            if (pj == 0.0f) { continue; }
            const uint2 vv = *reinterpret_cast<const uint2 *>(&sv[j][2 * lane]);
            acc[0] = fmaf(pj, __uint_as_float(vv.x << 16), acc[0]);
            acc[1] = fmaf(pj, __uint_as_float(vv.x & 0xffff0000u), acc[1]);
            acc[2] = fmaf(pj, __uint_as_float(vv.y << 16), acc[2]);
            acc[3] = fmaf(pj, __uint_as_float(vv.y & 0xffff0000u), acc[3]);
        }
    }
    float *dst = partials + (((uint64_t)row * INK2A_HEADS + h) * INK2A_SPLITS + split) * INK2A_PARTIAL;
    if (lane == 0) { dst[0] = m; dst[1] = l; }
    #pragma unroll
    for (int d = 0; d < 4; d++) { dst[2 + 4 * lane + d] = acc[d]; }
}

// out = BF16(sum of the partials' acc over the sum of their l).
__global__ void k2a_merge(float *out, const float *partials, uint32_t splits) {
    const uint32_t rh = blockIdx.x, d = threadIdx.x;
    const float *src = partials + (uint64_t)rh * INK2A_SPLITS * INK2A_PARTIAL;
    float top = -INFINITY;
    for (uint32_t s = 0; s < splits; s++) { top = fmaxf(top, src[s * INK2A_PARTIAL]); }
    float total = 0.0f, value = 0.0f;
    for (uint32_t s = 0; s < splits; s++) {
        const float ms = src[s * INK2A_PARTIAL];
        if (ms == -INFINITY) { continue; }
        const float e = __expf(ms - top);
        total = fmaf(src[s * INK2A_PARTIAL + 1], e, total);
        value = fmaf(src[s * INK2A_PARTIAL + 2 + d], e, value);
    }
    out[(uint64_t)rh * INK2A_DIM + d] = ink2a_bf16(__fdiv_rn(value, total));
}

#endif
