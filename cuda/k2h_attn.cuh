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
 * rows split their keys over CTAs (k2a_split: the KV head's six query heads
 * as MMA rows; k2a_decode: a warp per query head) and a merge folds the
 * partials. */

/* A decode partial: (m, l, two pad floats, acc[128]); the pad keeps acc
 * 16-byte aligned for vector stores. */
enum {
    K2A_HEADS = INK2A_HEADS, K2A_GROUP = INK2A_GROUP, K2A_DIM = INK2A_DIM,
    K2A_SPLITS = INK2A_SPLITS, K2A_ACC = 4, K2A_PARTIAL = K2A_ACC + INK2A_DIM, K2A_MIN_SPLIT = INK2A_MIN_SPLIT,
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

/* Prefill, ping-pong form of k2a_prefill: two decoded tiles, so the two
 * heads' warp groups run out of phase on each SM sub-core. Head h0's group
 * decodes tile t + 1's keys and then consumes tile t; head h0 + 1's group
 * consumes tile t and then decodes tile t + 1's values, so one group's
 * decode and softmax overlap the other's MMAs (k2a_prefill's eight warps
 * decoded, scored and summed in step, idling the tensor pipe in between:
 * 62 TFLOPS). One barrier per tile; the next raw tile loads a tile ahead.
 * The softmax works in raw dot units, p = 2^(s c - m c) with c the SDPA
 * scale times log2 e; tiles below every query skip the causal mask, and the
 * output rescale is skipped while every row's maximum holds. */
enum { K2A_PP_SMEM = 2 * INK2A_STAGE_ELEMS * (int)sizeof(uint16_t) + 2 * INK2A_RAW };

__global__ __launch_bounds__(INK2A_THREADS, 1)
void k2a_prefill_pp(float *out, const float *q, const uint8_t *pages, uint32_t rows, uint32_t base) {
    extern __shared__ __align__(16) uint16_t k2a_pp_smem[];
    // Two decoded tiles (K rows then V rows), then a two-stage ring of raw tiles.
    uint8_t *raw = (uint8_t *)(k2a_pp_smem + 2 * INK2A_STAGE_ELEMS);
    auto s_k = [&](uint32_t b) { return reinterpret_cast<uint16_t (*)[INK2A_SROW]>(k2a_pp_smem + b * INK2A_STAGE_ELEMS); };
    const uint32_t tq0 = blockIdx.x * INK2A_TQ, h0 = blockIdx.y * INK2A_CTA_HEADS, kv_head = h0 / INK2A_GROUP;
    const uint32_t group = threadIdx.x / (INK2A_WARPS * 32), h = h0 + group;
    const uint32_t local = threadIdx.x % (INK2A_WARPS * 32), warp = local / 32, lane = local % 32;
    if (tq0 >= rows) { return; }
    constexpr float C = K2A_SCALE * 1.4426950408889634f;

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
    const uint32_t tiles = key_hi / INK2A_TK + 1u, qmin = base + tq0;
    float output[INK2A_DIM / 8][4];
    #pragma unroll
    for (int cb = 0; cb < INK2A_DIM / 8; cb++) {
        #pragma unroll
        for (int l = 0; l < 4; l++) { output[cb][l] = 0.0f; }
    }

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
    // One group decodes K (value 0) or V (value 1) of a raw tile into buffer b;
    // keys past key_hi are zeros.
    auto decode_half = [&](uint32_t value, uint32_t b, int stage, uint32_t kt0) {
        const uint8_t *half = raw + stage * INK2A_RAW + value * INK2A_RAW_HALF;
        auto rows_b = s_k(b) + value * INK2A_TK;
        #pragma unroll
        for (uint32_t g = local; g < INK2A_TK * 8; g += INK2A_WARPS * 32) {
            const uint32_t key = g / 8, grp = g % 8;
            uint32_t packed[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (kt0 + key <= key_hi) {
                const uint2 data = *reinterpret_cast<const uint2 *>(half + key * INK2A_FP4 + grp * 8);
                ink2a_fp4x16(data, half[INK2A_TK * INK2A_FP4 + key * 8 + grp], packed);
            }
            uint16_t *dst = rows_b[key] + grp * 16;
            *reinterpret_cast<uint4 *>(dst) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
            *reinterpret_cast<uint4 *>(dst + 8) = make_uint4(packed[4], packed[5], packed[6], packed[7]);
        }
    };
    auto consume = [&](uint32_t b, uint32_t kt0) {
        auto sk = s_k(b);
        auto sv = sk + INK2A_TK;
        float s[INK2A_NB][4];
        const uint32_t k_row = lane & 7u, k_col = (lane >> 3) * 8u;
        #pragma unroll
        for (int nb = 0; nb < INK2A_NB; nb++) {
            #pragma unroll
            for (int l = 0; l < 4; l++) { s[nb][l] = 0.0f; }
            #pragma unroll
            for (int kc = 0; kc < INK2A_DIM / 16; kc += 2) {
                uint32_t b0, b1, b2, b3;
                ink2a_ldsm4(b0, b1, b2, b3, &sk[nb * 8 + k_row][kc * 16 + k_col]);
                ink2a_mma(s[nb], qa[kc], b0, b1);
                ink2a_mma(s[nb], qa[kc + 1], b2, b3);
            }
        }
        float tile_max[2] = {-INFINITY, -INFINITY};
        if (kt0 + INK2A_TK - 1 > qmin) {
            // Diagonal and dead rows: causal mask (raw units; the scale is positive).
            #pragma unroll
            for (int nb = 0; nb < INK2A_NB; nb++) {
                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    const int r = l / 2;
                    const uint32_t p = kt0 + nb * 8 + (lane % 4) * 2 + (l % 2);
                    if (!(alive[r] && p <= qpos[r])) { s[nb][l] = -INFINITY; }
                    tile_max[r] = fmaxf(tile_max[r], s[nb][l]);
                }
            }
        } else {
            #pragma unroll
            for (int nb = 0; nb < INK2A_NB; nb++) {
                #pragma unroll
                for (int l = 0; l < 4; l++) { tile_max[l / 2] = fmaxf(tile_max[l / 2], s[nb][l]); }
            }
        }
        float rescale[2], tile_sum[2] = {0.0f, 0.0f}, mc[2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 1));
            tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 2));
            const float next = fmaxf(row_m[r], tile_max[r]);
            rescale[r] = row_m[r] == next ? 1.0f : exp2f((row_m[r] - next) * C);
            row_m[r] = next;
            mc[r] = next == -INFINITY ? 0.0f : next * C;
        }
        uint32_t pa[INK2A_KS][4];
        #pragma unroll
        for (int nb = 0; nb < INK2A_NB; nb++) {
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                // exp2(-inf) = 0 for masked keys.
                const float w = exp2f(fmaf(s[nb][l], C, -mc[l / 2]));
                s[nb][l] = w;
                tile_sum[l / 2] += w;
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
        if (__any_sync(0xffffffffu, rescale[0] != 1.0f || rescale[1] != 1.0f)) {
            #pragma unroll
            for (int cb = 0; cb < INK2A_DIM / 8; cb++) {
                #pragma unroll
                for (int l = 0; l < 4; l++) { output[cb][l] *= rescale[l / 2]; }
            }
        }
        const uint32_t v_row = ((lane >> 3) & 1u) * 8u + (lane & 7u), v_col = (lane >> 4) * 8u;
        #pragma unroll
        for (int ks = 0; ks < INK2A_KS; ks++) {
            #pragma unroll
            for (int cb = 0; cb < INK2A_DIM / 8; cb += 2) {
                uint32_t b0, b1, b2, b3;
                ink2a_ldsm4t(b0, b1, b2, b3, &sv[ks * 16 + v_row][cb * 8 + v_col]);
                ink2a_mma(output[cb], pa[ks], b0, b1);
                ink2a_mma(output[cb + 1], pa[ks], b2, b3);
            }
        }
    };

    // Raw tile j lands in stage j % 2; decoded tile j in buffer j % 2.
    issue_tile(0, 0);
    __pipeline_commit();
    if (tiles > 1) { issue_tile(1, INK2A_TK); }
    __pipeline_commit();
    __pipeline_wait_prior(1);
    __syncthreads();
    decode_half(group, 0, 0, 0);
    #pragma unroll 1
    for (uint32_t t = 0; t < tiles; t++) {
        const uint32_t kt0 = t * INK2A_TK, b = t % 2;
        const bool more = t + 1 < tiles;
        // Raw tile t + 1 has landed; tile t is decoded; buffer b ^ 1 and raw
        // stage b are free.
        __pipeline_wait_prior(0);
        __syncthreads();
        if (t + 2 < tiles) { issue_tile(b, kt0 + 2 * INK2A_TK); }
        __pipeline_commit();
        if (group == 0) {
            if (more) { decode_half(0, b ^ 1, b ^ 1, kt0 + INK2A_TK); }
            consume(b, kt0);
        } else {
            consume(b, kt0);
            if (more) { decode_half(1, b ^ 1, b ^ 1, kt0 + INK2A_TK); }
        }
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
    float *dst = partials + (((uint64_t)row * INK2A_HEADS + h) * INK2A_SPLITS + split) * K2A_PARTIAL;
    if (lane == 0) { dst[0] = m; dst[1] = l; }
    #pragma unroll
    for (int d = 0; d < 4; d++) { dst[K2A_ACC + 4 * lane + d] = acc[d]; }
}

/* Programmatic dependent launch (cuda/glm_vq.cuh's contract): open first,
 * wait before reading a predecessor's output. No-ops for plain launches. */
__device__ __forceinline__ void k2a_pdl_open() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

__device__ __forceinline__ void k2a_pdl_wait() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

/* Decode attention on tensor cores (MiMo's GQA form): the six query heads
 * of a KV head are the m16 rows of a BF16 MMA (rows 6-15 zero), so one
 * decoded K fragment scores all six. CTA (slice, KV head, row): the slice's
 * 64-key tiles (9 KB raw: K data, K scales, V data, V scales) stream through
 * a three-stage cp.async ring; warp w takes keys 16w..16w+15 of each tile
 * with its own online softmax and writes partial 4 * slice + w. The k2a_decode
 * walk (a lane per key, Q from shared memory) ran at 107 GB/s.
 *
 * Fragment order: lane t of a score MMA holds dims 32t..32t+31 over the
 * eight k steps (step c: dims 32t + 4c..+3), so it reads 16 contiguous code
 * bytes of its key and Q follows the same order; for P.V lane g holds output
 * dims 16g..16g+15 (n block j: dim 16g + j) of its four keys. E2M1 x E4M3
 * is a BF16, so the decoded operands are exact. */
enum { K2A_DWARPS = 4, K2A_DSTAGES = 3, K2A_DSUB = INK2A_TK / K2A_DWARPS,
       K2A_DSLICES = INK2A_SPLITS / K2A_DWARPS };

__device__ __forceinline__ float k2a_e4m3(uint32_t byte) {
    __nv_fp8_e4m3 s;
    s.__x = (__nv_fp8_storage_t)byte;
    return (float)s;
}

// The two E2M1 codes of `byte` (low nibble first) times `scale`, as BF16.
__device__ __forceinline__ uint32_t k2a_pair(uint32_t byte, float scale) {
    const float2 v = ink2a_e2m1x2(byte);
    return ink2a_pack(v.x * scale, v.y * scale);
}

__global__ __launch_bounds__(K2A_DWARPS * 32)
void k2a_split(float *partials, const float *q, const uint8_t *pages, uint32_t base, uint32_t slices) {
    __shared__ __align__(16) uint8_t ring[K2A_DSTAGES][INK2A_RAW];
    k2a_pdl_open();
    const uint32_t slice = blockIdx.x, kv = blockIdx.y, row = blockIdx.z;
    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const uint32_t last = base + row, tiles = last / INK2A_TK + 1, per = (tiles + slices - 1) / slices;
    const uint32_t first = min(slice * per, tiles), n = min(first + per, tiles) - first;
    k2a_pdl_wait();

    // Q fragments of k step c: dims 32t + 4c..+3 of head 6 kv + g (zero past the group).
    uint32_t qa[INK2A_DIM / 16][2];
    const float4 *qrow = reinterpret_cast<const float4 *>(q +
        ((uint64_t)row * INK2A_HEADS + kv * INK2A_GROUP + min(g, (uint32_t)INK2A_GROUP - 1)) * INK2A_DIM + 32 * t);
    #pragma unroll
    for (int c = 0; c < INK2A_DIM / 16; c++) {
        const float4 x = g < INK2A_GROUP ? qrow[c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        qa[c][0] = ink2a_pack(x.x, x.y);
        qa[c][1] = ink2a_pack(x.z, x.w);
    }

    // Tile `tile` (64 keys from position 64 tile) into a ring stage.
    auto issue = [&](uint32_t stage, uint32_t tile) {
        const uint32_t kt0 = tile * INK2A_TK, slot = kt0 % INK2A_KB, data = INK2A_TK * INK2A_FP4;
        const uint8_t *page = ink2a_page(pages, kt0, kv);
        for (uint32_t c = threadIdx.x; c < INK2A_RAW / 16; c += K2A_DWARPS * 32) {
            const uint32_t at = c * 16, half = at / INK2A_RAW_HALF, in = at % INK2A_RAW_HALF;
            const uint8_t *src = page + half * INK2A_KB * INK2A_FP4_ROW +
                (in < data ? slot * INK2A_FP4 + in : INK2A_KB * INK2A_FP4 + slot * 8 + (in - data));
            __pipeline_memcpy_async(ring[stage] + at, src, 16);
        }
    };

    float o[INK2A_DIM / 8][4], m = -INFINITY, l = 0.0f;
    #pragma unroll
    for (int j = 0; j < INK2A_DIM / 8; j++) { o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f; }
    #pragma unroll
    for (uint32_t s = 0; s + 1 < K2A_DSTAGES; s++) {
        if (s < n) { issue(s, first + s); }
        __pipeline_commit();
    }
    #pragma unroll 1
    for (uint32_t i = 0; i < n; i++) {
        // Tile i has landed; every warp is done with the stage refilled here.
        __pipeline_wait_prior(K2A_DSTAGES - 2);
        __syncthreads();
        if (i + K2A_DSTAGES - 1 < n) { issue((i + K2A_DSTAGES - 1) % K2A_DSTAGES, first + i + K2A_DSTAGES - 1); }
        __pipeline_commit();
        const uint8_t *st = ring[i % K2A_DSTAGES];
        const uint32_t tile0 = (first + i) * INK2A_TK, sub = K2A_DSUB * warp;

        // Scores of keys sub + 8 jj + g (B column g of n block jj).
        float s[2][4];
        #pragma unroll
        for (int jj = 0; jj < 2; jj++) {
            const uint32_t key = sub + 8 * jj + g;
            const uint4 d = *reinterpret_cast<const uint4 *>(st + key * INK2A_FP4 + 16 * t);
            const uint32_t sc = *reinterpret_cast<const uint16_t *>(st + INK2A_TK * INK2A_FP4 + key * 8 + 2 * t);
            const float lo = k2a_e4m3(sc & 0xffu), hi = k2a_e4m3(sc >> 8);
            const uint32_t w[4] = {d.x, d.y, d.z, d.w};
            s[jj][0] = s[jj][1] = s[jj][2] = s[jj][3] = 0.0f;
            #pragma unroll
            for (int c = 0; c < INK2A_DIM / 16; c++) {
                const uint32_t bytes = w[c / 2] >> (16 * (c % 2));
                const float scale = c < 4 ? lo : hi;
                const uint32_t a[4] = {qa[c][0], 0u, qa[c][1], 0u};
                ink2a_mma(s[jj], a, k2a_pair(bytes & 0xffu, scale), k2a_pair((bytes >> 8) & 0xffu, scale));
            }
        }

        // Online softmax of row g over this lane's keys sub + 8 jj + 2t + e.
        float top = -INFINITY;
        #pragma unroll
        for (int jj = 0; jj < 2; jj++) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                const bool in = tile0 + sub + 8 * jj + 2 * t + e <= last;
                s[jj][e] = in ? s[jj][e] * K2A_SCALE : -INFINITY;
                top = fmaxf(top, s[jj][e]);
            }
        }
        top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
        top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
        const float next = fmaxf(m, top), rescale = m == -INFINITY ? 0.0f : __expf(m - next);
        m = next;
        float sum = 0.0f;
        #pragma unroll
        for (int jj = 0; jj < 2; jj++) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                s[jj][e] = s[jj][e] == -INFINITY ? 0.0f : __expf(s[jj][e] - m);
                sum += s[jj][e];
            }
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        l = l * rescale + sum;
        #pragma unroll
        for (int j = 0; j < INK2A_DIM / 8; j++) {
            o[j][0] *= rescale;
            o[j][1] *= rescale;
        }

        // P.V: k rows 2t, 2t + 1, 2t + 8, 2t + 9 are keys sub + 2t + {0, 1, 8, 9};
        // masked keys take a zero scale so stale bytes cannot reach the sum.
        const uint32_t pa[4] = {ink2a_pack(s[0][0], s[0][1]), 0u, ink2a_pack(s[1][0], s[1][1]), 0u};
        const uint8_t *vd = st + INK2A_RAW_HALF, *vsc = vd + INK2A_TK * INK2A_FP4;
        uint2 vw[4];
        float vs[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t key = sub + 2 * t + k % 2 + 8 * (k / 2);
            vw[k] = *reinterpret_cast<const uint2 *>(vd + key * INK2A_FP4 + 8 * g);
            vs[k] = tile0 + key <= last ? k2a_e4m3(vsc[key * 8 + g]) : 0.0f;
        }
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            // Byte b: output dims 16g + 2b and 16g + 2b + 1 (n blocks 2b, 2b + 1).
            float2 f[4];
            #pragma unroll
            for (int k = 0; k < 4; k++) { f[k] = ink2a_e2m1x2(((b < 4 ? vw[k].x : vw[k].y) >> (8 * (b % 4))) & 0xffu); }
            ink2a_mma(o[2 * b], pa, ink2a_pack(f[0].x * vs[0], f[1].x * vs[1]),
                ink2a_pack(f[2].x * vs[2], f[3].x * vs[3]));
            ink2a_mma(o[2 * b + 1], pa, ink2a_pack(f[0].y * vs[0], f[1].y * vs[1]),
                ink2a_pack(f[2].y * vs[2], f[3].y * vs[3]));
        }
    }

    // C column n of n block j is dim 16n + j: this lane holds dims 32t..32t+31,
    // stored as eight 16-byte pieces (scalar stores cost ~30 us a call).
    if (g >= INK2A_GROUP) { return; }
    float *dst = partials + (((uint64_t)row * INK2A_HEADS + kv * INK2A_GROUP + g) * INK2A_SPLITS +
        slice * K2A_DWARPS + warp) * K2A_PARTIAL;
    if (t == 0) {
        dst[0] = m;
        dst[1] = l;
    }
    float4 *acc = reinterpret_cast<float4 *>(dst + K2A_ACC + 32 * t);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        acc[i] = make_float4(o[4 * i][0], o[4 * i + 1][0], o[4 * i + 2][0], o[4 * i + 3][0]);
        acc[4 + i] = make_float4(o[4 * i][1], o[4 * i + 1][1], o[4 * i + 2][1], o[4 * i + 3][1]);
    }
}

/* out = BF16(sum of the partials' acc over the sum of their l). Thread s
 * loads split s's (m, l) so the 64 splits' maxima arrive together (a serial
 * walk cost ~0.5 us per split), then each thread sums its dim over the
 * splits with independent loads. */
__global__ void __launch_bounds__(INK2A_DIM) k2a_merge(float *out, const float *partials, uint32_t splits) {
    static_assert(INK2A_SPLITS <= INK2A_DIM, "a thread per split");
    __shared__ float weight[INK2A_SPLITS], red[INK2A_DIM / 32];
    k2a_pdl_open();
    k2a_pdl_wait();
    const uint32_t rh = blockIdx.x, d = threadIdx.x, lane = d % 32, warp = d / 32;
    const float *src = partials + (uint64_t)rh * INK2A_SPLITS * K2A_PARTIAL;
    const float ms = d < splits ? src[d * K2A_PARTIAL] : -INFINITY;
    const float ls = d < splits ? src[d * K2A_PARTIAL + 1] : 0.0f;
    float top = ms;
    for (uint32_t o = 16; o; o >>= 1) { top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, o)); }
    if (!lane) { red[warp] = top; }
    __syncthreads();
    top = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    const float e = ms == -INFINITY ? 0.0f : __expf(ms - top);
    if (d < INK2A_SPLITS) { weight[d] = e; }
    float den = ls * e;
    for (uint32_t o = 16; o; o >>= 1) { den += __shfl_xor_sync(0xffffffffu, den, o); }
    __syncthreads();
    if (!lane) { red[warp] = den; }
    __syncthreads();
    den = (red[0] + red[1]) + (red[2] + red[3]);
    float value = 0.0f;
    #pragma unroll 16
    for (uint32_t s = 0; s < splits; s++) { value = fmaf(src[s * K2A_PARTIAL + K2A_ACC + d], weight[s], value); }
    out[(uint64_t)rh * INK2A_DIM + d] = ink2a_bf16(__fdiv_rn(value, den));
}

#endif
