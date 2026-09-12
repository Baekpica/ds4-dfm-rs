/* Prefill attention on tensor cores (flash-attention style).
 *
 * Layout (head_dim 128 only): grid (ceil(n_tokens/64), n_head), block 128
 * threads. Each warp owns 16 query rows. The block stages 64 Q rows and
 * walks 32-key K/V tiles (two 16-key HMMA consume steps). Q.K^T and P.V
 * use m16n8k16 HMMA with online softmax between them. Compressed Solar
 * K/V is decoded once per shared tile with the per-row scale reused
 * across the 128 dims, so all 64 query rows share one dequant.
 *
 * When n_head/n_head_kv is even and at least 2, the GQA-pair kernel
 * launches instead: grid.y = n_head/2, block 256. Warps 0-3 and 4-7
 * own consecutive Q heads that share one KV head. That kernel walks
 * 64-key tiles (four 16-key consume steps) so the deep K walk syncs
 * half as often as the one-head 32-key path. Q stays in qa[] registers
 * so the 64-key K/V staging stays under 48 KiB. Eight Q heads in one
 * block still does not fit registers or shared memory.
 *
 * The GQA-pair kernel reads its K/V fragments with ldmatrix and, for
 * BF16 K/V, stages the next 64-key tile in registers while the current
 * one is consumed (K2 round 6; DS4_FATTN_HMMA_LDSM=0 restores the scalar
 * loads and the direct fill, bit-identical).
 *
 * With DS4_SOLAR_FATTN_WS=1, K-FP8/V-FP4 pairs take
 * ds4_fattn_hmma_solar_ws_kernel instead: producer warps stream and decode
 * the tiles while the eight consumer warps walk them (Solar round 5,
 * bit-identical; opt-in because of its power draw on GB10 hosts).
 */
#include "common.cuh"
#include "mma.cuh"
#include "ds4_mmq.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

using namespace ggml_cuda_mma;

#include "inkling_attention.cuh"

namespace {

enum {
    FA_HD      = 128,
    FA_WQ      = 16,
    FA_WARPS   = 4,
    FA_TQ      = FA_WQ * FA_WARPS,
    FA_TK      = 32,
    FA_CONSUME = 16,
    FA_PAD     = 8,
    FA_ROW     = FA_HD + FA_PAD,
    SOLAR_KV_BF16 = 0,
    SOLAR_KV_FP8 = 1,
    SOLAR_KV_FP4 = 2,
    SOLAR_KV_KFP8_VFP4 = 3,
};

typedef tile<16, 8, half2> tile_a;
typedef tile< 8, 8, half2> tile_b;
typedef tile<16, 8, float> tile_c;

__device__ __forceinline__ float solar_fattn_e4m3(uint8_t code) {
    __nv_fp8_e4m3 value;
    value.__x = code;
    return (float)value;
}

__device__ __forceinline__ float solar_fattn_e2m1(uint8_t code) {
    __nv_fp4_e2m1 value;
    value.__x = code & 0x0fu;
    return (float)value;
}

template <int FORMAT, bool VALUE>
__device__ __forceinline__ float solar_fattn_kv_load(
        const uint8_t *row,
        uint32_t       n_head_kv,
        uint32_t       kvh,
        uint32_t       dim) {
    const uint64_t kv_dim = (uint64_t)n_head_kv * FA_HD;
    const uint64_t elem = (uint64_t)kvh * FA_HD + dim;
    const uint64_t k_bytes = FORMAT == SOLAR_KV_FP4 ? kv_dim / 2u : kv_dim;
    const uint64_t v_bytes = FORMAT == SOLAR_KV_FP8 ? kv_dim : kv_dim / 2u;
    const __half *scales = (const __half *)(row + k_bytes + v_bytes);
    const float scale = __half2float(
        scales[(VALUE ? n_head_kv : 0u) + kvh]);
    const uint8_t *data = VALUE ? row + k_bytes : row;
    if constexpr ((VALUE && FORMAT == SOLAR_KV_FP8) ||
                  (!VALUE && FORMAT != SOLAR_KV_FP4)) {
        return solar_fattn_e4m3(data[elem]) * scale;
    } else {
        const uint8_t packed = data[elem >> 1u];
        const uint8_t code = (elem & 1u) ? packed >> 4u : packed & 0x0fu;
        return solar_fattn_e2m1(code) * scale;
    }
}

template <int FORMAT>
__device__ __forceinline__ void solar_fattn_kv_bytes(
        uint32_t  n_head_kv,
        uint64_t *k_bytes,
        uint64_t *v_bytes) {
    const uint64_t kv_dim = (uint64_t)n_head_kv * FA_HD;
    *k_bytes = FORMAT == SOLAR_KV_FP4 ? kv_dim / 2u : kv_dim;
    *v_bytes = FORMAT == SOLAR_KV_FP8 ? kv_dim : kv_dim / 2u;
}

/* Decode one TK-key (or shorter) K/V tile.  Per-row K/V scales are read
 * once and reused across the 128 dims.  Compressed formats convert four
 * consecutive dims per thread so the packed row is touched with aligned
 * 4-byte / 2-byte loads.  TK is 32 on the one-head kernel and 64 on the
 * GQA-pair kernel. */
template <int FORMAT, int TK = FA_TK>
__device__ __forceinline__ void solar_fattn_fill_kv_tile(
        __half        s_k[][FA_ROW],
        __half        s_v[][FA_ROW],
        const void   *kv,
        uint64_t      row_bytes,
        uint32_t      n_head_kv,
        uint32_t      kvh,
        uint32_t      kv_dim,
        uint32_t      kv_cap,
        uint32_t      kt0,
        uint32_t      tile_len) {
    __shared__ float scale_k[TK];
    __shared__ float scale_v[TK];
    if (threadIdx.x < (uint32_t)TK) {
        const uint32_t r = threadIdx.x;
        const uint32_t src = r < tile_len ? kt0 + r : kt0;
        if constexpr (FORMAT == SOLAR_KV_BF16) {
            scale_k[r] = 1.0f;
            scale_v[r] = 1.0f;
        } else {
            uint64_t k_bytes = 0, v_bytes = 0;
            solar_fattn_kv_bytes<FORMAT>(n_head_kv, &k_bytes, &v_bytes);
            const uint8_t *row = (const uint8_t *)kv +
                (uint64_t)(src % kv_cap) * row_bytes;
            const __half *scales = (const __half *)(row + k_bytes + v_bytes);
            scale_k[r] = __half2float(scales[kvh]);
            scale_v[r] = __half2float(scales[n_head_kv + kvh]);
        }
    }
    __syncthreads();

    constexpr uint32_t NGRP = FA_HD / 4u;
    for (uint32_t idx = threadIdx.x; idx < (uint32_t)TK * NGRP;
         idx += blockDim.x) {
        const uint32_t r = idx / NGRP;
        const uint32_t g = idx - r * NGRP;
        const uint32_t c = g * 4u;
        const uint32_t src = r < tile_len ? kt0 + r : kt0;
        if constexpr (FORMAT == SOLAR_KV_BF16) {
            const __half *row = (const __half *)kv +
                (size_t)(src % kv_cap) * kv_dim * 2u +
                (size_t)kvh * FA_HD;
            const float2 k2 = *reinterpret_cast<const float2 *>(row + c);
            const float2 v2 = *reinterpret_cast<const float2 *>(
                row + kv_dim + c);
            *reinterpret_cast<float2 *>(&s_k[r][c]) = k2;
            *reinterpret_cast<float2 *>(&s_v[r][c]) = v2;
        } else if constexpr (FORMAT == SOLAR_KV_KFP8_VFP4) {
            const uint64_t k_bytes = (uint64_t)n_head_kv * FA_HD;
            const uint8_t *row = (const uint8_t *)kv +
                (uint64_t)(src % kv_cap) * row_bytes;
            const float sk = scale_k[r];
            const float sv = scale_v[r];
            const uint64_t kbase = (uint64_t)kvh * FA_HD + c;
            const uint32_t kpack = *reinterpret_cast<const uint32_t *>(
                row + kbase);
            const uint16_t vpack = *reinterpret_cast<const uint16_t *>(
                row + k_bytes + (kbase >> 1u));
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const uint8_t kc = (uint8_t)(kpack >> (8 * i));
                const uint8_t vc = (uint8_t)((vpack >> (4 * i)) & 0x0fu);
                s_k[r][c + (uint32_t)i] = __float2half(
                    solar_fattn_e4m3(kc) * sk);
                s_v[r][c + (uint32_t)i] = __float2half(
                    solar_fattn_e2m1(vc) * sv);
            }
        } else if constexpr (FORMAT == SOLAR_KV_FP8) {
            const uint64_t kv_elems = (uint64_t)n_head_kv * FA_HD;
            const uint8_t *row = (const uint8_t *)kv +
                (uint64_t)(src % kv_cap) * row_bytes;
            const float sk = scale_k[r];
            const float sv = scale_v[r];
            const uint64_t e = (uint64_t)kvh * FA_HD + c;
            const uint32_t kpack = *reinterpret_cast<const uint32_t *>(
                row + e);
            const uint32_t vpack = *reinterpret_cast<const uint32_t *>(
                row + kv_elems + e);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                s_k[r][c + (uint32_t)i] = __float2half(
                    solar_fattn_e4m3((uint8_t)(kpack >> (8 * i))) * sk);
                s_v[r][c + (uint32_t)i] = __float2half(
                    solar_fattn_e4m3((uint8_t)(vpack >> (8 * i))) * sv);
            }
        } else {
            const uint8_t *row = (const uint8_t *)kv +
                (uint64_t)(src % kv_cap) * row_bytes;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                s_k[r][c + (uint32_t)i] = __float2half(
                    solar_fattn_kv_load<FORMAT, false>(
                        row, n_head_kv, kvh, c + (uint32_t)i));
                s_v[r][c + (uint32_t)i] = __float2half(
                    solar_fattn_kv_load<FORMAT, true>(
                        row, n_head_kv, kvh, c + (uint32_t)i));
            }
        }
    }
}

__device__ __forceinline__ void solar_fattn_consume_16(
        tile_c          output[FA_HD / 8],
        float           row_m[2],
        float           row_l[2],
        const tile_a    qa[FA_HD / 16],
        const __half  (*s_k)[FA_ROW],
        const __half  (*s_v)[FA_ROW],
        const bool      alive[2],
        const uint32_t  qpos[2],
        const uint32_t  qfirst[2],
        uint32_t        kt0,
        uint32_t        tile_len,
        uint32_t        lane,
        float           scale) {
    if (tile_len == 0u) return;
    tile_c scores[2];
#pragma unroll
    for (int nb = 0; nb < 2; nb++) {
        tile_c zero;
        scores[nb] = zero;
#pragma unroll
        for (int kc = 0; kc < FA_HD / 16; kc++) {
            tile_b keys;
#pragma unroll
            for (int l = 0; l < tile_b::ne; l++) {
                const int i = (int)(lane / 4);
                const int j = l * 4 + (int)(lane % 4);
                keys.x[l] = *(const half2 *)&s_k[
                    nb * 8 + i][kc * 16 + 2 * j];
            }
            mma(scores[nb], qa[kc], keys);
        }
    }

    float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nb = 0; nb < 2; nb++) {
#pragma unroll
        for (int l = 0; l < tile_c::ne; l++) {
            const int r = l / 2;
            const uint32_t p =
                kt0 + nb * 8u + (lane % 4u) * 2u + (l % 2u);
            float score = scores[nb].x[l] * scale;
            if (!alive[r] || p > qpos[r] || p < qfirst[r] ||
                p >= kt0 + tile_len) {
                score = -INFINITY;
            }
            scores[nb].x[l] = score;
            tile_max[r] = fmaxf(tile_max[r], score);
        }
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        tile_max[r] = fmaxf(
            tile_max[r],
            __shfl_xor_sync(0xffffffffu, tile_max[r], 1));
        tile_max[r] = fmaxf(
            tile_max[r],
            __shfl_xor_sync(0xffffffffu, tile_max[r], 2));
    }

    float rescale[2];
    float tile_sum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const float next_max = fmaxf(row_m[r], tile_max[r]);
        rescale[r] = row_m[r] == -INFINITY
            ? 0.0f : __expf(row_m[r] - next_max);
        row_m[r] = next_max;
    }
#pragma unroll
    for (int nb = 0; nb < 2; nb++) {
#pragma unroll
        for (int l = 0; l < tile_c::ne; l++) {
            const int r = l / 2;
            const float weight =
                scores[nb].x[l] == -INFINITY || row_m[r] == -INFINITY
                    ? 0.0f : __expf(scores[nb].x[l] - row_m[r]);
            scores[nb].x[l] = weight;
            tile_sum[r] += weight;
        }
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        tile_sum[r] +=
            __shfl_xor_sync(0xffffffffu, tile_sum[r], 1);
        tile_sum[r] +=
            __shfl_xor_sync(0xffffffffu, tile_sum[r], 2);
        row_l[r] = row_l[r] * rescale[r] + tile_sum[r];
    }

    tile_a probabilities;
#pragma unroll
    for (int l = 0; l < tile_a::ne; l++) {
        probabilities.x[l] = __floats2half2_rn(
            scores[l / 2].x[(l % 2) * 2],
            scores[l / 2].x[(l % 2) * 2 + 1]);
    }
#pragma unroll
    for (int cb = 0; cb < FA_HD / 8; cb++) {
#pragma unroll
        for (int l = 0; l < tile_c::ne; l++) {
            output[cb].x[l] *= rescale[l / 2];
        }
        tile_b values;
#pragma unroll
        for (int l = 0; l < tile_b::ne; l++) {
            const int i = (int)(lane / 4);
            const int j = l * 4 + (int)(lane % 4);
            values.x[l] = __halves2half2(
                s_v[2 * j][cb * 8 + i],
                s_v[2 * j + 1][cb * 8 + i]);
        }
        mma(output[cb], probabilities, values);
    }
}

/* Four 8x8 b16 matrices from shared memory into mma.sync B-fragment
 * order.  Lanes 8m..8m+7 supply the row addresses of matrix m; every row
 * address must be 16-byte aligned, which the padded FA_ROW rows are
 * (272 bytes) whenever the column offset is a multiple of 8 halves. */
__device__ __forceinline__ void solar_fattn_ldsm_x4(
        uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
        const __half *row_ptr) {
#ifdef TURING_MMA_AVAILABLE
    const uint32_t addr = (uint32_t)__cvta_generic_to_shared(row_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
#else
    GGML_UNUSED_VARS(r0, r1, r2, r3, row_ptr);
    NO_DEVICE_CODE;
#endif
}

__device__ __forceinline__ void solar_fattn_ldsm_x4_trans(
        uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
        const __half *row_ptr) {
#ifdef TURING_MMA_AVAILABLE
    const uint32_t addr = (uint32_t)__cvta_generic_to_shared(row_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
#else
    GGML_UNUSED_VARS(r0, r1, r2, r3, row_ptr);
    NO_DEVICE_CODE;
#endif
}

/* solar_fattn_consume_16 with ldmatrix fragment loads (K2 round 6).
 *
 * ncu on a 1024-token synthetic prefill (64 heads, 8 KV heads, head_dim
 * 128; K2 itself has 48 heads over the same 8 KV
 * heads) put the GQA-pair kernel at 164 registers, issue slots active 31 %,
 * the LSU pipe at 54 % of peak with 13.4 M shared-load wavefronts against
 * 1.1 M shared stores, and the top stalls long_scoreboard 1.20 / wait 0.99
 * / mio_throttle 0.76 / lg_throttle 0.70 per issued instruction: the
 * kernel is bound by fragment loads, not by the tensor pipe.  The scalar
 * consume builds every B fragment by hand -- 32 half2 loads for the 16 K
 * fragments and 64 half loads plus 32 packs for the 16 V fragments per
 * 16-key step and lane.  ldmatrix delivers the same fragments in mma.sync
 * order: one .x4 covers the two K fragments of a kc pair (matrices =
 * dims kc*16 + {0, 8, 16, 24}, eight key rows each) and one .x4.trans the
 * two V fragments of a cb pair (matrices = keys {0-7, 8-15} x dims cb*8
 * and (cb+1)*8), so a step issues 8 + 8 shared loads instead of 96.
 *
 * Bit-identity: ldmatrix returns thread t the elements (t/4, 2*(t%4)) and
 * (t/4, 2*(t%4)+1) of each 8x8 matrix, low column in the low half -- the
 * exact half2 the scalar loop read from s_k[nb*8 + lane/4][kc*16 +
 * 2*(l*4 + lane%4)]; the .trans form returns (2*(t%4), t/4) and
 * (2*(t%4)+1, t/4), i.e. the same even/odd key pair the scalar loop packed
 * with __halves2half2.  The mma sequence (kc ascending per nb, cb
 * ascending with the rescale applied just before each PV mma), the masked
 * online softmax and the row sums are the scalar function's, so the
 * accumulators see the same operands in the same order. */
__device__ __forceinline__ void solar_fattn_consume_16_ldsm(
        tile_c          output[FA_HD / 8],
        float           row_m[2],
        float           row_l[2],
        const tile_a    qa[FA_HD / 16],
        const __half  (*s_k)[FA_ROW],
        const __half  (*s_v)[FA_ROW],
        const bool      alive[2],
        const uint32_t  qpos[2],
        const uint32_t  qfirst[2],
        uint32_t        kt0,
        uint32_t        tile_len,
        uint32_t        lane,
        float           scale) {
    if (tile_len == 0u) return;
    static_assert(tile_b::ne == 2, "B fragment is two b32 registers");
    static_assert((FA_ROW * sizeof(__half)) % 16u == 0u,
                  "ldmatrix rows must stay 16-byte aligned");
    tile_c scores[2];
    /* Lane l addresses row (l % 8) of matrix (l / 8): keys nb*8 + (l % 8),
     * dims kc*16 + (l / 8) * 8. */
    const uint32_t k_row = lane & 7u;
    const uint32_t k_col = (lane >> 3) * 8u;
#pragma unroll
    for (int nb = 0; nb < 2; nb++) {
        tile_c zero;
        scores[nb] = zero;
#pragma unroll
        for (int kc = 0; kc < FA_HD / 16; kc += 2) {
            tile_b keys0, keys1;
            uint32_t *k0 = reinterpret_cast<uint32_t *>(keys0.x);
            uint32_t *k1 = reinterpret_cast<uint32_t *>(keys1.x);
            solar_fattn_ldsm_x4(
                k0[0], k0[1], k1[0], k1[1],
                &s_k[nb * 8 + k_row][kc * 16 + k_col]);
            mma(scores[nb], qa[kc], keys0);
            mma(scores[nb], qa[kc + 1], keys1);
        }
    }

    float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nb = 0; nb < 2; nb++) {
#pragma unroll
        for (int l = 0; l < tile_c::ne; l++) {
            const int r = l / 2;
            const uint32_t p =
                kt0 + nb * 8u + (lane % 4u) * 2u + (l % 2u);
            float score = scores[nb].x[l] * scale;
            if (!alive[r] || p > qpos[r] || p < qfirst[r] ||
                p >= kt0 + tile_len) {
                score = -INFINITY;
            }
            scores[nb].x[l] = score;
            tile_max[r] = fmaxf(tile_max[r], score);
        }
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        tile_max[r] = fmaxf(
            tile_max[r],
            __shfl_xor_sync(0xffffffffu, tile_max[r], 1));
        tile_max[r] = fmaxf(
            tile_max[r],
            __shfl_xor_sync(0xffffffffu, tile_max[r], 2));
    }

    float rescale[2];
    float tile_sum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const float next_max = fmaxf(row_m[r], tile_max[r]);
        rescale[r] = row_m[r] == -INFINITY
            ? 0.0f : __expf(row_m[r] - next_max);
        row_m[r] = next_max;
    }
#pragma unroll
    for (int nb = 0; nb < 2; nb++) {
#pragma unroll
        for (int l = 0; l < tile_c::ne; l++) {
            const int r = l / 2;
            const float weight =
                scores[nb].x[l] == -INFINITY || row_m[r] == -INFINITY
                    ? 0.0f : __expf(scores[nb].x[l] - row_m[r]);
            scores[nb].x[l] = weight;
            tile_sum[r] += weight;
        }
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        tile_sum[r] +=
            __shfl_xor_sync(0xffffffffu, tile_sum[r], 1);
        tile_sum[r] +=
            __shfl_xor_sync(0xffffffffu, tile_sum[r], 2);
        row_l[r] = row_l[r] * rescale[r] + tile_sum[r];
    }

    tile_a probabilities;
#pragma unroll
    for (int l = 0; l < tile_a::ne; l++) {
        probabilities.x[l] = __floats2half2_rn(
            scores[l / 2].x[(l % 2) * 2],
            scores[l / 2].x[(l % 2) * 2 + 1]);
    }
    /* Lane l addresses key row ((l / 8) % 2) * 8 + (l % 8) of matrix
     * (l / 8): matrices 0/1 are keys 0-7 / 8-15 at dims cb*8, matrices 2/3
     * the same keys at dims (cb+1)*8. */
    const uint32_t v_row = ((lane >> 3) & 1u) * 8u + (lane & 7u);
    const uint32_t v_col = (lane >> 4) * 8u;
#pragma unroll
    for (int cb = 0; cb < FA_HD / 8; cb += 2) {
        tile_b values0, values1;
        uint32_t *v0 = reinterpret_cast<uint32_t *>(values0.x);
        uint32_t *v1 = reinterpret_cast<uint32_t *>(values1.x);
        solar_fattn_ldsm_x4_trans(
            v0[0], v0[1], v1[0], v1[1],
            &s_v[v_row][cb * 8 + v_col]);
#pragma unroll
        for (int l = 0; l < tile_c::ne; l++) {
            output[cb].x[l] *= rescale[l / 2];
        }
        mma(output[cb], probabilities, values0);
#pragma unroll
        for (int l = 0; l < tile_c::ne; l++) {
            output[cb + 1].x[l] *= rescale[l / 2];
        }
        mma(output[cb + 1], probabilities, values1);
    }
}

template <int KV_FORMAT>
__global__ void ds4_fattn_hmma_kernel(
        float * __restrict__ heads,
        const float * __restrict__ q,
        const void * __restrict__ kv,
        const uint64_t row_bytes,
        const uint32_t n_tokens,
        const uint32_t pos0,
        const uint32_t n_head,
        const uint32_t n_head_kv,
        const uint32_t kv_cap,
        const uint32_t window,
        const float scale) {
    const uint32_t tq0 = blockIdx.x * FA_TQ;
    const uint32_t h = blockIdx.y;
    if (tq0 >= n_tokens || h >= n_head) return;
    const uint32_t group = n_head / n_head_kv;
    const uint32_t kvh = h / group;
    const uint32_t kv_dim = n_head_kv * FA_HD;

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;

    __shared__ __half s_q[FA_TQ][FA_ROW];
    __shared__ __half s_k[FA_TK][FA_ROW];
    __shared__ __half s_v[FA_TK][FA_ROW];

    for (uint32_t idx = threadIdx.x; idx < FA_TQ * FA_HD;
         idx += blockDim.x) {
        const uint32_t r = idx / FA_HD;
        const uint32_t c = idx - r * FA_HD;
        const uint32_t t = tq0 + r < n_tokens ? tq0 + r : 0u;
        s_q[r][c] = __float2half(
            q[((size_t)t * n_head + h) * FA_HD + c]);
    }
    __syncthreads();

    const uint32_t qrow[2] = {
        warp * FA_WQ + lane / 4,
        warp * FA_WQ + lane / 4 + 8u,
    };
    uint32_t qpos[2], qfirst[2];
    bool alive[2];
    float row_m[2], row_l[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        alive[r] = tq0 + qrow[r] < n_tokens;
        qpos[r] = alive[r] ? pos0 + tq0 + qrow[r] : pos0;
        qfirst[r] = window && qpos[r] + 1u > window
            ? qpos[r] + 1u - window : 0u;
        row_m[r] = -INFINITY;
        row_l[r] = 0.0f;
    }

    tile_a qa[FA_HD / 16];
#pragma unroll
    for (int kc = 0; kc < FA_HD / 16; kc++) {
#pragma unroll
        for (int l = 0; l < tile_a::ne; l++) {
            const int i = (l % 2) * 8 + (int)(lane / 4);
            const int j = (l / 2) * 4 + (int)(lane % 4);
            qa[kc].x[l] = *(const half2 *)&s_q[
                warp * FA_WQ + i][kc * 16 + 2 * j];
        }
    }

    tile_c output[FA_HD / 8];
    uint32_t block_first = 0xffffffffu;
    uint32_t block_last = 0u;
#pragma unroll
    for (int r = 0; r < 2; r++) {
        if (!alive[r]) continue;
        block_first = min(block_first, qfirst[r]);
        block_last = max(block_last, qpos[r]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        block_first = min(
            block_first,
            __shfl_xor_sync(0xffffffffu, block_first, offset));
        block_last = max(
            block_last,
            __shfl_xor_sync(0xffffffffu, block_last, offset));
    }
    __shared__ uint32_t shared_first;
    __shared__ uint32_t shared_last;
    if (threadIdx.x == 0u) {
        shared_first = 0xffffffffu;
        shared_last = 0u;
    }
    __syncthreads();
    if (lane == 0u) {
        atomicMin(&shared_first, block_first);
        atomicMax(&shared_last, block_last);
    }
    __syncthreads();
    block_first = shared_first;
    block_last = shared_last;
    if (block_first == 0xffffffffu) return;

    for (uint32_t kt0 = block_first; kt0 <= block_last; kt0 += FA_TK) {
        const uint32_t remaining = block_last - kt0 + 1u;
        const uint32_t tile_len = remaining < (uint32_t)FA_TK
            ? remaining : (uint32_t)FA_TK;
        solar_fattn_fill_kv_tile<KV_FORMAT>(
            s_k, s_v, kv, row_bytes, n_head_kv, kvh, kv_dim, kv_cap,
            kt0, tile_len);
        __syncthreads();
        const uint32_t first_len = tile_len < (uint32_t)FA_CONSUME
            ? tile_len : (uint32_t)FA_CONSUME;
        solar_fattn_consume_16(
            output, row_m, row_l, qa, s_k, s_v, alive, qpos, qfirst,
            kt0, first_len, lane, scale);
        if (tile_len > (uint32_t)FA_CONSUME) {
            solar_fattn_consume_16(
                output, row_m, row_l, qa, s_k + FA_CONSUME, s_v + FA_CONSUME,
                alive, qpos, qfirst, kt0 + (uint32_t)FA_CONSUME,
                tile_len - (uint32_t)FA_CONSUME, lane, scale);
        }
        __syncthreads();
    }

#pragma unroll
    for (int cb = 0; cb < FA_HD / 8; cb++) {
#pragma unroll
        for (int l = 0; l < tile_c::ne; l++) {
            const int r = l / 2;
            if (!alive[r] || row_l[r] <= 0.0f) continue;
            const uint32_t token = tq0 + qrow[r];
            const int col = (int)(lane % 4) * 2 + (l % 2);
            heads[((size_t)token * n_head + h) * FA_HD + cb * 8 + col] =
                output[cb].x[l] / row_l[r];
        }
    }
}

__device__ __forceinline__ void solar_fattn_load_qa(
        tile_a        qa[FA_HD / 16],
        const float  *q,
        uint32_t      tq0,
        uint32_t      n_tokens,
        uint32_t      n_head,
        uint32_t      h,
        uint32_t      warp,
        uint32_t      lane) {
#pragma unroll
    for (int kc = 0; kc < FA_HD / 16; kc++) {
#pragma unroll
        for (int l = 0; l < tile_a::ne; l++) {
            const int i = (l % 2) * 8 + (int)(lane / 4);
            const int j = (l / 2) * 4 + (int)(lane % 4);
            const uint32_t row = warp * FA_WQ + (uint32_t)i;
            const uint32_t col = (uint32_t)kc * 16u + 2u * (uint32_t)j;
            const uint32_t t = tq0 + row < n_tokens ? tq0 + row : 0u;
            const float2 xy = *reinterpret_cast<const float2 *>(
                q + ((size_t)t * n_head + h) * FA_HD + col);
            qa[kc].x[l] = __floats2half2_rn(xy.x, xy.y);
        }
    }
}

/* BF16 K/V tile staged in registers (K2 round 6).  The pair kernel is
 * register-limited to one block per SM, so no second block hides the L2
 * round trip of solar_fattn_fill_kv_tile: ncu showed long_scoreboard as
 * the top stall with the block idle between tiles.  With LDSM the next
 * 64-key tile is fetched into PF float2 pairs per thread right after the
 * barrier that publishes the current tile, and lands while the four
 * consume steps run.  Same source rows, same bytes, same duplicated row
 * past tile_len as the direct fill, so the staged tile is identical. */
template <uint32_t PF>
__device__ __forceinline__ void solar_fattn_bf16_tile_fetch(
        float2      pk[PF],
        float2      pv[PF],
        const void *kv,
        uint32_t    kv_dim,
        uint32_t    kvh,
        uint32_t    kv_cap,
        uint32_t    kt0,
        uint32_t    tile_len) {
    constexpr uint32_t NGRP = FA_HD / 4u;
    constexpr uint32_t THREADS = 2u * (uint32_t)FA_WARPS * 32u;
#pragma unroll
    for (uint32_t p = 0; p < PF; p++) {
        const uint32_t idx = threadIdx.x + p * THREADS;
        const uint32_t r = idx / NGRP;
        const uint32_t c = (idx - r * NGRP) * 4u;
        const uint32_t src = r < tile_len ? kt0 + r : kt0;
        const __half *row = (const __half *)kv +
            (size_t)(src % kv_cap) * kv_dim * 2u +
            (size_t)kvh * FA_HD;
        pk[p] = *reinterpret_cast<const float2 *>(row + c);
        pv[p] = *reinterpret_cast<const float2 *>(row + kv_dim + c);
    }
}

template <uint32_t PF>
__device__ __forceinline__ void solar_fattn_bf16_tile_store(
        const float2 pk[PF],
        const float2 pv[PF],
        __half       s_k[][FA_ROW],
        __half       s_v[][FA_ROW]) {
    constexpr uint32_t NGRP = FA_HD / 4u;
    constexpr uint32_t THREADS = 2u * (uint32_t)FA_WARPS * 32u;
#pragma unroll
    for (uint32_t p = 0; p < PF; p++) {
        const uint32_t idx = threadIdx.x + p * THREADS;
        const uint32_t r = idx / NGRP;
        const uint32_t c = (idx - r * NGRP) * 4u;
        *reinterpret_cast<float2 *>(&s_k[r][c]) = pk[p];
        *reinterpret_cast<float2 *>(&s_v[r][c]) = pv[p];
    }
}

/* Two even-grouped Q heads share one dequantized K/V tile.  Eight warps:
 * 0-3 own head 2*by, 4-7 own head 2*by+1.  Fragment loads use the local
 * lane (threadIdx.x % 32), not raw threadIdx.x / 4, so warps 4-7 keep
 * the same HMMA layout as warps 0-3.
 *
 * LDSM (default, DS4_FATTN_HMMA_LDSM=0 restores the scalar path) swaps the
 * scalar fragment loads for solar_fattn_consume_16_ldsm and, for the BF16
 * K/V format, overlaps the next tile fill with the consume steps through
 * solar_fattn_bf16_tile_fetch; compressed Solar formats keep the direct
 * fill and only take the ldmatrix consume.  Both variants write the same
 * bytes (tests/test_exaone_kernels test_attention_ldsm memcmps them). */
template <int KV_FORMAT, bool LDSM>
__global__ void ds4_fattn_hmma_gqa2_kernel(
        float * __restrict__ heads,
        const float * __restrict__ q,
        const void * __restrict__ kv,
        const uint64_t row_bytes,
        const uint32_t n_tokens,
        const uint32_t pos0,
        const uint32_t n_head,
        const uint32_t n_head_kv,
        const uint32_t kv_cap,
        const uint32_t window,
        const float scale) {
    constexpr uint32_t N_Q = 2u;
    constexpr uint32_t FA_TK2 = 64u;
    constexpr uint32_t GROUP_THREADS = (uint32_t)FA_WARPS * 32u;
    const uint32_t tq0 = blockIdx.x * FA_TQ;
    const uint32_t h0 = blockIdx.y * N_Q;
    if (tq0 >= n_tokens || h0 + 1u >= n_head) return;
    const uint32_t group = n_head / n_head_kv;
    const uint32_t kvh = h0 / group;
    const uint32_t kv_dim = n_head_kv * FA_HD;
    const uint32_t local = threadIdx.x % GROUP_THREADS;
    const uint32_t hi = threadIdx.x / GROUP_THREADS;
    const uint32_t h = h0 + hi;
    const uint32_t warp = local >> 5;
    const uint32_t lane = local & 31u;

    /* 16-byte aligned for ldmatrix row addresses (LDSM). */
    __shared__ __align__(16) __half s_k[FA_TK2][FA_ROW];
    __shared__ __align__(16) __half s_v[FA_TK2][FA_ROW];
    static_assert(sizeof(s_k) + sizeof(s_v) + 2u * sizeof(uint32_t) <=
                      49152u,
                  "GQA-pair FATTN 64-key tile must stay under 48 KiB");

    tile_a qa[FA_HD / 16];
    solar_fattn_load_qa(qa, q, tq0, n_tokens, n_head, h, warp, lane);

    const uint32_t qrow[2] = {
        warp * FA_WQ + lane / 4,
        warp * FA_WQ + lane / 4 + 8u,
    };
    uint32_t qpos[2], qfirst[2];
    bool alive[2];
    float row_m[2], row_l[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        alive[r] = tq0 + qrow[r] < n_tokens;
        qpos[r] = alive[r] ? pos0 + tq0 + qrow[r] : pos0;
        qfirst[r] = window && qpos[r] + 1u > window
            ? qpos[r] + 1u - window : 0u;
        row_m[r] = -INFINITY;
        row_l[r] = 0.0f;
    }

    tile_c output[FA_HD / 8];
    uint32_t block_first = 0xffffffffu;
    uint32_t block_last = 0u;
#pragma unroll
    for (int r = 0; r < 2; r++) {
        if (!alive[r]) continue;
        block_first = min(block_first, qfirst[r]);
        block_last = max(block_last, qpos[r]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        block_first = min(
            block_first,
            __shfl_xor_sync(0xffffffffu, block_first, offset));
        block_last = max(
            block_last,
            __shfl_xor_sync(0xffffffffu, block_last, offset));
    }
    __shared__ uint32_t shared_first;
    __shared__ uint32_t shared_last;
    if (threadIdx.x == 0u) {
        shared_first = 0xffffffffu;
        shared_last = 0u;
    }
    __syncthreads();
    if (lane == 0u) {
        atomicMin(&shared_first, block_first);
        atomicMax(&shared_last, block_last);
    }
    __syncthreads();
    block_first = shared_first;
    block_last = shared_last;
    if (block_first == 0xffffffffu) return;

    if constexpr (LDSM && KV_FORMAT == SOLAR_KV_BF16) {
        constexpr uint32_t PF =
            FA_TK2 * (FA_HD / 4u) / (2u * GROUP_THREADS);
        static_assert(PF * 2u * GROUP_THREADS == FA_TK2 * (FA_HD / 4u),
                      "tile fill must split evenly over the block");
        float2 pk[PF], pv[PF];
        {
            const uint32_t remaining = block_last - block_first + 1u;
            solar_fattn_bf16_tile_fetch<PF>(
                pk, pv, kv, kv_dim, kvh, kv_cap, block_first,
                remaining < FA_TK2 ? remaining : FA_TK2);
        }
        for (uint32_t kt0 = block_first; kt0 <= block_last; kt0 += FA_TK2) {
            const uint32_t remaining = block_last - kt0 + 1u;
            const uint32_t tile_len = remaining < FA_TK2 ? remaining : FA_TK2;
            /* The previous iteration's trailing barrier released the tile. */
            solar_fattn_bf16_tile_store<PF>(pk, pv, s_k, s_v);
            __syncthreads();
            const uint32_t kt1 = kt0 + FA_TK2;
            if (kt1 <= block_last) {
                const uint32_t remaining1 = block_last - kt1 + 1u;
                solar_fattn_bf16_tile_fetch<PF>(
                    pk, pv, kv, kv_dim, kvh, kv_cap, kt1,
                    remaining1 < FA_TK2 ? remaining1 : FA_TK2);
            }
            for (uint32_t step = 0; step < FA_TK2;
                 step += (uint32_t)FA_CONSUME) {
                if (step >= tile_len) break;
                const uint32_t step_len =
                    tile_len - step < (uint32_t)FA_CONSUME
                        ? tile_len - step : (uint32_t)FA_CONSUME;
                solar_fattn_consume_16_ldsm(
                    output, row_m, row_l, qa, s_k + step, s_v + step,
                    alive, qpos, qfirst, kt0 + step, step_len, lane, scale);
            }
            __syncthreads();
        }
    } else {
        for (uint32_t kt0 = block_first; kt0 <= block_last; kt0 += FA_TK2) {
            const uint32_t remaining = block_last - kt0 + 1u;
            const uint32_t tile_len = remaining < FA_TK2 ? remaining : FA_TK2;
            solar_fattn_fill_kv_tile<KV_FORMAT, (int)FA_TK2>(
                s_k, s_v, kv, row_bytes, n_head_kv, kvh, kv_dim, kv_cap,
                kt0, tile_len);
            __syncthreads();
            for (uint32_t step = 0; step < FA_TK2;
                 step += (uint32_t)FA_CONSUME) {
                if (step >= tile_len) break;
                const uint32_t step_len =
                    tile_len - step < (uint32_t)FA_CONSUME
                        ? tile_len - step : (uint32_t)FA_CONSUME;
                if constexpr (LDSM) {
                    solar_fattn_consume_16_ldsm(
                        output, row_m, row_l, qa, s_k + step, s_v + step,
                        alive, qpos, qfirst, kt0 + step, step_len, lane,
                        scale);
                } else {
                    solar_fattn_consume_16(
                        output, row_m, row_l, qa, s_k + step, s_v + step,
                        alive, qpos, qfirst, kt0 + step, step_len, lane,
                        scale);
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int cb = 0; cb < FA_HD / 8; cb++) {
#pragma unroll
        for (int l = 0; l < tile_c::ne; l++) {
            const int r = l / 2;
            if (!alive[r] || row_l[r] <= 0.0f) continue;
            const uint32_t token = tq0 + qrow[r];
            const int col = (int)(lane % 4) * 2 + (l % 2);
            heads[((size_t)token * n_head + h) * FA_HD + cb * 8 + col] =
                output[cb].x[l] / row_l[r];
        }
    }
}

/* Warp-specialized K-FP8/V-FP4 GQA-pair prefill (Solar round 5).
 *
 * The pair kernel above fills each 64-key tile synchronously: scale
 * loads, a barrier, packed K/V loads, decode, another barrier, then the
 * HMMA walk.  At 131 registers it runs one CTA per SM, so nothing hides
 * the global round trip or the decode; at 64K context the tail chunk
 * spent 292 ms per layer in it and the tensor pipe idled three quarters
 * of the time.  This kernel splits the roles:
 *
 *   warps 0-7   consumers  two Q heads x 64 queries, same HMMA layout,
 *                          same 16-key online-softmax steps
 *   warps 8-11  producers  cp.async raw rows -> two-stage raw ring,
 *                          decode -> double-buffered half tiles
 *
 *   producers:  raw[0] raw[1] raw[0] ...      (cp.async, 2 tiles ahead)
 *               dec->half[0] dec->half[1] ... (FULL[b] arrive)
 *   consumers:  FULL[0] wait, walk half[0], EMPTY[0] arrive, FULL[1] ...
 *
 * Named barriers 1-4 (count 384) pair one producer arrive with one
 * consumer sync (FULL) or the reverse (EMPTY), so consumers never wait
 * for a decode and producers never wait for HMMA work.  One CTA per SM
 * (95 KB dynamic shared memory, 384 threads, <=168 registers, no spills).
 *
 * Numerical contract: byte-identical to the pair kernel.  Every element
 * is decoded with the same conversion and the same fp32 multiply and
 * half rounding (the x2 hardware conversions are exact); each query row
 * consumes the same 16-key steps in the same order with the same
 * operations.  Interior tiles (warp-uniform: no causal or window edge
 * inside the 64 keys) skip only the mask chain, which can change no
 * finite score, and the output rescale is skipped only when every factor
 * in the warp is exactly 1.0f (an identity multiply).
 * DS4_SOLAR_FATTN_WS=1 selects it; see solar_fattn_ws_enabled. */
enum {
    WS_TK        = 64,
    WS_KRAW      = FA_HD,                 /* K bytes per row per KV head */
    WS_VRAW      = FA_HD / 2,             /* V bytes per row per KV head */
    WS_CONSUMERS = 2 * FA_WARPS,
    WS_PRODUCERS = 4,
    WS_CTHREADS  = WS_CONSUMERS * 32,
    WS_PTHREADS  = WS_PRODUCERS * 32,
    WS_THREADS   = WS_CTHREADS + WS_PTHREADS,
    WS_BAR_FULL0  = 1,
    WS_BAR_EMPTY0 = 3,
    WS_BAR_PROD   = 5,
};

__device__ __forceinline__ void solar_ws_cp_async_16(void *dst, const void *src) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(src));
}

__device__ __forceinline__ void solar_ws_cp_async_8(void *dst, const void *src) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(s), "l"(src));
}

__device__ __forceinline__ void solar_ws_cp_async_4(void *dst, const void *src) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s), "l"(src));
}

__device__ __forceinline__ void solar_ws_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void solar_ws_cp_wait_1() {
    asm volatile("cp.async.wait_group 1;\n" ::);
}

__device__ __forceinline__ void solar_ws_bar_sync(uint32_t id, uint32_t count) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(count) : "memory");
}

__device__ __forceinline__ void solar_ws_bar_arrive(uint32_t id, uint32_t count) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(count) : "memory");
}

/* One KV head's raw bytes of a 64-key tile plus, per row, the 4-byte
 * words holding its K and V scales. */
struct solar_ws_raw {
    uint8_t  k[WS_TK][WS_KRAW];
    uint8_t  v[WS_TK][WS_VRAW];
    uint32_t s[WS_TK][2];
};

struct solar_ws_smem {
    __half       s_k[2][WS_TK][FA_ROW];
    __half       s_v[2][WS_TK][FA_ROW];
    solar_ws_raw raw[2];
};

/* Raw copies of tile kt0 into one ring stage.  Two producer threads share
 * a row (one address computation each) and alternate its CPW-byte K/V
 * chunks and the two scale words.  Rows past tile_len duplicate row kt0
 * like the direct fill, so the staged tile is identical. */
template <int CPW>
__device__ __forceinline__ void solar_ws_issue_tile(
        solar_ws_raw  &raw,
        const uint8_t *kv,
        uint64_t       row_bytes,
        uint32_t       n_head_kv,
        uint32_t       kvh,
        uint32_t       kv_cap,
        uint32_t       kt0,
        uint32_t       tile_len,
        uint32_t       ptid) {
    constexpr uint32_t KCH = WS_KRAW / CPW;
    constexpr uint32_t VCH = WS_VRAW / CPW;
    constexpr uint32_t PARTS = KCH + VCH + 2u;
    const uint64_t k_bytes = (uint64_t)n_head_kv * FA_HD;
    const uint64_t v_bytes = k_bytes / 2u;
    const uint32_t r = ptid >> 1;
    const uint32_t src = r < tile_len ? kt0 + r : kt0;
    const uint8_t *row = kv + (uint64_t)(src % kv_cap) * row_bytes;
    const uint8_t *krow = row + (uint64_t)kvh * FA_HD;
    const uint8_t *vrow = row + k_bytes + (uint64_t)kvh * WS_VRAW;
    const uint8_t *srow = row + k_bytes + v_bytes;
#pragma unroll
    for (uint32_t part = ptid & 1u; part < PARTS; part += 2u) {
        if (part < KCH) {
            if constexpr (CPW == 16) {
                solar_ws_cp_async_16(&raw.k[r][part * CPW], krow + part * CPW);
            } else {
                solar_ws_cp_async_8(&raw.k[r][part * CPW], krow + part * CPW);
            }
        } else if (part < KCH + VCH) {
            const uint32_t vp = part - KCH;
            if constexpr (CPW == 16) {
                solar_ws_cp_async_16(&raw.v[r][vp * CPW], vrow + vp * CPW);
            } else {
                solar_ws_cp_async_8(&raw.v[r][vp * CPW], vrow + vp * CPW);
            }
        } else if (part == KCH + VCH) {
            solar_ws_cp_async_4(&raw.s[r][0], srow + ((kvh * 2u) & ~3u));
        } else {
            solar_ws_cp_async_4(
                &raw.s[r][1], srow + (((n_head_kv + kvh) * 2u) & ~3u));
        }
    }
}

/* raw -> half tiles.  Element pairs go through the x2 hardware
 * conversions (e4m3x2 / e2m1x2 -> half2, exact) and half -> float
 * (exact); the fp32 product and the single rn half rounding are those of
 * solar_fattn_fill_kv_tile, so the bytes match while the instruction
 * count halves. */
__device__ __forceinline__ void solar_ws_decode_tile(
        const solar_ws_raw &raw,
        __half              s_k[][FA_ROW],
        __half              s_v[][FA_ROW],
        uint32_t            n_head_kv,
        uint32_t            kvh,
        uint32_t            ptid) {
    constexpr uint32_t NGRP = FA_HD / 4u;
    const uint32_t k_sel = kvh & 1u;
    const uint32_t v_sel = (n_head_kv + kvh) & 1u;
#pragma unroll 4
    for (uint32_t i = 0; i < (WS_TK * NGRP) / WS_PTHREADS; i++) {
        const uint32_t idx = ptid + i * WS_PTHREADS;
        const uint32_t r = idx / NGRP;
        const uint32_t g = idx - r * NGRP;
        const uint32_t c = g * 4u;
        const uint32_t kpack = *reinterpret_cast<const uint32_t *>(&raw.k[r][c]);
        const uint16_t vpack = *reinterpret_cast<const uint16_t *>(&raw.v[r][c >> 1u]);
        const uint32_t kw = raw.s[r][0];
        const uint32_t vw = raw.s[r][1];
        const float sk = __half2float(__ushort_as_half(
            (uint16_t)(k_sel ? kw >> 16u : kw & 0xffffu)));
        const float sv = __half2float(__ushort_as_half(
            (uint16_t)(v_sel ? vw >> 16u : vw & 0xffffu)));
        const __half2_raw k01 = __nv_cvt_fp8x2_to_halfraw2(
            (__nv_fp8x2_storage_t)(kpack & 0xffffu), __NV_E4M3);
        const __half2_raw k23 = __nv_cvt_fp8x2_to_halfraw2(
            (__nv_fp8x2_storage_t)(kpack >> 16u), __NV_E4M3);
        const __half2_raw v01 = __nv_cvt_fp4x2_to_halfraw2(
            (__nv_fp4x2_storage_t)(vpack & 0xffu), __NV_E2M1);
        const __half2_raw v23 = __nv_cvt_fp4x2_to_halfraw2(
            (__nv_fp4x2_storage_t)(vpack >> 8u), __NV_E2M1);
        const float2 fk01 = __half22float2(__half2(k01));
        const float2 fk23 = __half22float2(__half2(k23));
        const float2 fv01 = __half22float2(__half2(v01));
        const float2 fv23 = __half22float2(__half2(v23));
        const __half2 hk01 = __floats2half2_rn(fk01.x * sk, fk01.y * sk);
        const __half2 hk23 = __floats2half2_rn(fk23.x * sk, fk23.y * sk);
        const __half2 hv01 = __floats2half2_rn(fv01.x * sv, fv01.y * sv);
        const __half2 hv23 = __floats2half2_rn(fv23.x * sv, fv23.y * sv);
        uint2 kq, vq;
        kq.x = *reinterpret_cast<const uint32_t *>(&hk01);
        kq.y = *reinterpret_cast<const uint32_t *>(&hk23);
        vq.x = *reinterpret_cast<const uint32_t *>(&hv01);
        vq.y = *reinterpret_cast<const uint32_t *>(&hv23);
        *reinterpret_cast<uint2 *>(&s_k[r][c]) = kq;
        *reinterpret_cast<uint2 *>(&s_v[r][c]) = vq;
    }
}

/* Q.K^T of one 16-key step: kc ascending per nb, from zero, as in
 * solar_fattn_consume_16_ldsm. */
__device__ __forceinline__ void solar_ws_qk_step(
        tile_c          scores[2],
        const tile_a    qa[FA_HD / 16],
        const __half  (*s_k)[FA_ROW],
        uint32_t        lane) {
    const uint32_t k_row = lane & 7u;
    const uint32_t k_col = (lane >> 3) * 8u;
#pragma unroll
    for (int nb = 0; nb < 2; nb++) {
        tile_c zero;
        scores[nb] = zero;
#pragma unroll
        for (int kc = 0; kc < FA_HD / 16; kc += 2) {
            tile_b keys0, keys1;
            uint32_t *k0 = reinterpret_cast<uint32_t *>(keys0.x);
            uint32_t *k1 = reinterpret_cast<uint32_t *>(keys1.x);
            solar_fattn_ldsm_x4(
                k0[0], k0[1], k1[0], k1[1],
                &s_k[nb * 8 + k_row][kc * 16 + k_col]);
            mma(scores[nb], qa[kc], keys0);
            mma(scores[nb], qa[kc + 1], keys1);
        }
    }
}

/* Online softmax of one interior step: the tracked sequence without the
 * mask chain.  __fmul_rn keeps the score scaling a plain FMUL; in the
 * masked path the select between the multiply and the subtraction
 * already blocks FMA contraction, and the bytes must not diverge here. */
__device__ __forceinline__ void solar_ws_softmax_step(
        tile_c  scores[2],
        float   row_m[2],
        float   row_l[2],
        float   rescale[2],
        float   scale) {
    float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nb = 0; nb < 2; nb++) {
#pragma unroll
        for (int l = 0; l < tile_c::ne; l++) {
            const int r = l / 2;
            const float score = __fmul_rn(scores[nb].x[l], scale);
            scores[nb].x[l] = score;
            tile_max[r] = fmaxf(tile_max[r], score);
        }
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        tile_max[r] = fmaxf(
            tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 1));
        tile_max[r] = fmaxf(
            tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 2));
    }
    float tile_sum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const float next_max = fmaxf(row_m[r], tile_max[r]);
        rescale[r] = row_m[r] == -INFINITY
            ? 0.0f : __expf(row_m[r] - next_max);
        row_m[r] = next_max;
    }
#pragma unroll
    for (int nb = 0; nb < 2; nb++) {
#pragma unroll
        for (int l = 0; l < tile_c::ne; l++) {
            const int r = l / 2;
            const float weight = __expf(scores[nb].x[l] - row_m[r]);
            scores[nb].x[l] = weight;
            tile_sum[r] += weight;
        }
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        tile_sum[r] += __shfl_xor_sync(0xffffffffu, tile_sum[r], 1);
        tile_sum[r] += __shfl_xor_sync(0xffffffffu, tile_sum[r], 2);
        row_l[r] = row_l[r] * rescale[r] + tile_sum[r];
    }
}

/* P.V of one 16-key step.  The accumulator rescale is an identity when
 * every factor in the warp is exactly 1.0f (the running max did not move
 * for any of the warp's 16 rows, the common case deep in a walk), so it
 * is skipped only then. */
__device__ __forceinline__ void solar_ws_pv_step(
        tile_c          output[FA_HD / 8],
        const tile_c    scores[2],
        const float     rescale[2],
        const __half  (*s_v)[FA_ROW],
        uint32_t        lane) {
    tile_a probabilities;
#pragma unroll
    for (int l = 0; l < tile_a::ne; l++) {
        probabilities.x[l] = __floats2half2_rn(
            scores[l / 2].x[(l % 2) * 2],
            scores[l / 2].x[(l % 2) * 2 + 1]);
    }
    const uint32_t v_row = ((lane >> 3) & 1u) * 8u + (lane & 7u);
    const uint32_t v_col = (lane >> 4) * 8u;
    const bool apply = !__all_sync(
        0xffffffffu, rescale[0] == 1.0f && rescale[1] == 1.0f);
    if (apply) {
#pragma unroll
        for (int cb = 0; cb < FA_HD / 8; cb++) {
#pragma unroll
            for (int l = 0; l < tile_c::ne; l++) {
                output[cb].x[l] *= rescale[l / 2];
            }
        }
    }
#pragma unroll
    for (int cb = 0; cb < FA_HD / 8; cb += 2) {
        tile_b values0, values1;
        uint32_t *v0 = reinterpret_cast<uint32_t *>(values0.x);
        uint32_t *v1 = reinterpret_cast<uint32_t *>(values1.x);
        solar_fattn_ldsm_x4_trans(
            v0[0], v0[1], v1[0], v1[1], &s_v[v_row][cb * 8 + v_col]);
        mma(output[cb], probabilities, values0);
        mma(output[cb + 1], probabilities, values1);
    }
}

template <int CPW>
__global__ void __launch_bounds__(WS_THREADS, 1)
ds4_fattn_hmma_solar_ws_kernel(
        float * __restrict__ heads,
        const float * __restrict__ q,
        const void * __restrict__ kv,
        const uint64_t row_bytes,
        const uint32_t n_tokens,
        const uint32_t pos0,
        const uint32_t n_head,
        const uint32_t n_head_kv,
        const uint32_t kv_cap,
        const uint32_t window,
        const float scale) {
    constexpr uint32_t N_Q = 2u;
    constexpr uint32_t GROUP_THREADS = (uint32_t)FA_WARPS * 32u;
    extern __shared__ __align__(16) uint8_t solar_ws_dyn[];
    solar_ws_smem &sm = *reinterpret_cast<solar_ws_smem *>(solar_ws_dyn);
    __shared__ uint32_t shared_first;
    __shared__ uint32_t shared_last;

    const uint32_t tq0 = blockIdx.x * FA_TQ;
    const uint32_t h0 = blockIdx.y * N_Q;
    if (tq0 >= n_tokens || h0 + 1u >= n_head) { return; }
    const uint32_t group = n_head / n_head_kv;
    const uint32_t kvh = h0 / group;
    const bool producer = (threadIdx.x >> 5) >= (uint32_t)WS_CONSUMERS;
    const uint32_t local = threadIdx.x % GROUP_THREADS;
    const uint32_t hi = producer ? 0u : threadIdx.x / GROUP_THREADS;
    const uint32_t h = h0 + hi;
    const uint32_t warp = local >> 5;
    const uint32_t lane = local & 31u;

    /* Block key range from the consumer rows; producers hold no rows. */
    const uint32_t qrow[2] = {
        warp * FA_WQ + lane / 4,
        warp * FA_WQ + lane / 4 + 8u,
    };
    uint32_t qpos[2], qfirst[2];
    bool alive[2];
    float row_m[2], row_l[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        alive[r] = !producer && tq0 + qrow[r] < n_tokens;
        qpos[r] = alive[r] ? pos0 + tq0 + qrow[r] : pos0;
        qfirst[r] = window && qpos[r] + 1u > window
            ? qpos[r] + 1u - window : 0u;
        row_m[r] = -INFINITY;
        row_l[r] = 0.0f;
    }
    uint32_t block_first = 0xffffffffu;
    uint32_t block_last = 0u;
#pragma unroll
    for (int r = 0; r < 2; r++) {
        if (!alive[r]) { continue; }
        block_first = min(block_first, qfirst[r]);
        block_last = max(block_last, qpos[r]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        block_first = min(
            block_first,
            __shfl_xor_sync(0xffffffffu, block_first, offset));
        block_last = max(
            block_last,
            __shfl_xor_sync(0xffffffffu, block_last, offset));
    }
    if (threadIdx.x == 0u) {
        shared_first = 0xffffffffu;
        shared_last = 0u;
    }
    __syncthreads();
    if (lane == 0u && !producer) {
        atomicMin(&shared_first, block_first);
        atomicMax(&shared_last, block_last);
    }
    __syncthreads();
    block_first = shared_first;
    block_last = shared_last;
    if (block_first == 0xffffffffu) { return; }
    const uint32_t n_tiles = (block_last - block_first) / WS_TK + 1u;

    if (producer) {
        const uint32_t ptid = threadIdx.x - WS_CTHREADS;
        const uint8_t *kvb = (const uint8_t *)kv;
        auto tile_len_of = [&](uint32_t t) {
            const uint32_t remaining =
                block_last - (block_first + t * WS_TK) + 1u;
            return remaining < (uint32_t)WS_TK ? remaining : (uint32_t)WS_TK;
        };
        /* Two tiles in flight; an empty commit keeps the group count
         * uniform so wait_group 1 always means "tile t landed". */
        solar_ws_issue_tile<CPW>(
            sm.raw[0], kvb, row_bytes, n_head_kv, kvh, kv_cap, block_first,
            tile_len_of(0), ptid);
        solar_ws_cp_commit();
        if (n_tiles > 1u) {
            solar_ws_issue_tile<CPW>(
                sm.raw[1], kvb, row_bytes, n_head_kv, kvh, kv_cap,
                block_first + WS_TK, tile_len_of(1), ptid);
        }
        solar_ws_cp_commit();
        for (uint32_t t = 0; t < n_tiles; t++) {
            const uint32_t b = t & 1u;
            solar_ws_cp_wait_1();
            /* Every producer's copies of tile t are visible. */
            solar_ws_bar_sync(WS_BAR_PROD, WS_PTHREADS);
            /* Consumers released half[b] (tile t-2). */
            if (t >= 2u) { solar_ws_bar_sync(WS_BAR_EMPTY0 + b, WS_THREADS); }
            solar_ws_decode_tile(sm.raw[b], sm.s_k[b], sm.s_v[b],
                                 n_head_kv, kvh, ptid);
            __threadfence_block();
            solar_ws_bar_arrive(WS_BAR_FULL0 + b, WS_THREADS);
            /* raw[b] fully read before tile t+2 overwrites it. */
            solar_ws_bar_sync(WS_BAR_PROD, WS_PTHREADS);
            if (t + 2u < n_tiles) {
                solar_ws_issue_tile<CPW>(
                    sm.raw[b], kvb, row_bytes, n_head_kv, kvh, kv_cap,
                    block_first + (t + 2u) * WS_TK, tile_len_of(t + 2u), ptid);
            }
            solar_ws_cp_commit();
        }
        /* Balance the consumers' last EMPTY arrives. */
        for (uint32_t t = n_tiles > 2u ? n_tiles - 2u : 0u; t < n_tiles; t++) {
            solar_ws_bar_sync(WS_BAR_EMPTY0 + (t & 1u), WS_THREADS);
        }
        return;
    }

    tile_a qa[FA_HD / 16];
    solar_fattn_load_qa(qa, q, tq0, n_tokens, n_head, h, warp, lane);
    tile_c output[FA_HD / 8];
    /* Warp-uniform interior bounds: row 0 of the warp has the smallest
     * qpos (dead rows only trail alive rows); qfirst grows with qpos. */
    const uint32_t warp_qmin = __shfl_sync(0xffffffffu, qpos[0], 0);
    uint32_t warp_fmax = max(qfirst[0], qfirst[1]);
#pragma unroll
    for (int offset = 4; offset < 32; offset <<= 1) {
        warp_fmax = max(
            warp_fmax, __shfl_xor_sync(0xffffffffu, warp_fmax, offset));
    }
    for (uint32_t t = 0; t < n_tiles; t++) {
        const uint32_t b = t & 1u;
        const uint32_t kt0 = block_first + t * WS_TK;
        const uint32_t remaining = block_last - kt0 + 1u;
        const uint32_t tile_len =
            remaining < (uint32_t)WS_TK ? remaining : (uint32_t)WS_TK;
        solar_ws_bar_sync(WS_BAR_FULL0 + b, WS_THREADS);
        const __half (*s_k)[FA_ROW] = sm.s_k[b];
        const __half (*s_v)[FA_ROW] = sm.s_v[b];
        const bool interior = tile_len == (uint32_t)WS_TK &&
            kt0 >= warp_fmax && kt0 + (WS_TK - 1u) <= warp_qmin;
        if (interior) {
#pragma unroll
            for (uint32_t step = 0; step < (uint32_t)WS_TK;
                 step += (uint32_t)FA_CONSUME) {
                tile_c scores[2];
                float rescale[2];
                solar_ws_qk_step(scores, qa, s_k + step, lane);
                solar_ws_softmax_step(scores, row_m, row_l, rescale, scale);
                solar_ws_pv_step(output, scores, rescale, s_v + step, lane);
            }
        } else {
            for (uint32_t step = 0; step < (uint32_t)WS_TK;
                 step += (uint32_t)FA_CONSUME) {
                if (step >= tile_len) { break; }
                const uint32_t step_len =
                    tile_len - step < (uint32_t)FA_CONSUME
                        ? tile_len - step : (uint32_t)FA_CONSUME;
                solar_fattn_consume_16_ldsm(
                    output, row_m, row_l, qa, s_k + step, s_v + step,
                    alive, qpos, qfirst, kt0 + step, step_len, lane, scale);
            }
        }
        solar_ws_bar_arrive(WS_BAR_EMPTY0 + b, WS_THREADS);
    }

#pragma unroll
    for (int cb = 0; cb < FA_HD / 8; cb++) {
#pragma unroll
        for (int l = 0; l < tile_c::ne; l++) {
            const int r = l / 2;
            if (!alive[r] || row_l[r] <= 0.0f) { continue; }
            const uint32_t token = tq0 + qrow[r];
            const int col = (int)(lane % 4) * 2 + (l % 2);
            heads[((size_t)token * n_head + h) * FA_HD + cb * 8 + col] =
                output[cb].x[l] / row_l[r];
        }
    }
}

/* Opt-in: DS4_SOLAR_FATTN_WS=1 selects the warp-specialized kernel for the
 * K-FP8/V-FP4 format (bit-identical, 2.6x faster at 64K depth).  It stays
 * off by default because on the GB10 hosts measured so far it draws about
 * 105 W against the pair kernel's 64 W at 64K depth, and sustained draw
 * above roughly 90 W hard-freezes those hosts without a log line (a known
 * platform fault; cap the SM clock with nvidia-smi -lgc before enabling). */
static int solar_fattn_ws_enabled(void) {
    const char *value = getenv("DS4_SOLAR_FATTN_WS");
    return value && value[0] == '1';
}

/* cp.async copy width the cache supports: the K/V regions sit at
 * multiples of 64 bytes inside a row, so only the row stride and the
 * cache base decide.  0 means the pair kernel must run. */
static int solar_fattn_ws_width(const void *kv, uint64_t row_bytes) {
    const uintptr_t base = (uintptr_t)kv;
    if ((row_bytes % 16u) == 0u && (base % 16u) == 0u) { return 16; }
    if ((row_bytes % 8u) == 0u && (base % 8u) == 0u) { return 8; }
    return 0;
}

/* Opt the kernel into its 95 KB dynamic shared memory; 0 when the device
 * cannot provide it. */
template <int CPW>
static int solar_fattn_ws_device_ok(void) {
    const auto &device = ggml_cuda_info().devices[ggml_cuda_get_device()];
    const int smem = (int)sizeof(solar_ws_smem);
    if (device.smpbo < (size_t)smem) { return 0; }
    if (cudaFuncSetAttribute(ds4_fattn_hmma_solar_ws_kernel<CPW>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

/* Launch the warp-specialized kernel when the device allows it; 0 means
 * the caller keeps the pair path. */
template <int CPW>
static int solar_fattn_ws_launch(
        float *heads, const float *q, const void *kv, uint64_t row_bytes,
        int n_tokens, int pos0, int n_head, int n_head_kv, int kv_cap,
        int window, float scale, cudaStream_t stream) {
    if (!solar_fattn_ws_device_ok<CPW>()) { return 0; }
    const int tiles = (n_tokens + FA_TQ - 1) / FA_TQ;
    const dim3 grid(tiles, n_head / 2, 1);
    ds4_fattn_hmma_solar_ws_kernel<CPW>
        <<<grid, WS_THREADS, (int)sizeof(solar_ws_smem), stream>>>(
            heads, q, kv, row_bytes, (uint32_t)n_tokens, (uint32_t)pos0,
            (uint32_t)n_head, (uint32_t)n_head_kv, (uint32_t)kv_cap,
            (uint32_t)window, scale);
    return 1;
}

static int solar_fattn_ws_try(
        float *heads, const float *q, const void *kv, uint64_t row_bytes,
        int n_tokens, int pos0, int n_head, int n_head_kv, int kv_cap,
        int window, float scale, cudaStream_t stream) {
    if (!solar_fattn_ws_enabled()) { return 0; }
    switch (solar_fattn_ws_width(kv, row_bytes)) {
    case 16:
        return solar_fattn_ws_launch<16>(
            heads, q, kv, row_bytes, n_tokens, pos0, n_head, n_head_kv,
            kv_cap, window, scale, stream);
    case 8:
        return solar_fattn_ws_launch<8>(
            heads, q, kv, row_bytes, n_tokens, pos0, n_head, n_head_kv,
            kv_cap, window, scale, stream);
    default:
        return 0;
    }
}

enum {
    M3_FA_QK     = 192,
    M3_FA_V      = 128,
    M3_FA_WQ     = 16,
    M3_FA_WARPS  = 4,
    M3_FA_TQ     = M3_FA_WQ * M3_FA_WARPS,
    M3_FA_TK     = 32,
    M3_FA_PAD    = 8,
    M3_FA_QK_ROW = M3_FA_QK + M3_FA_PAD,
    M3_FA_V_ROW  = M3_FA_V + M3_FA_PAD,
};

typedef tile<16, 8, nv_bfloat162> motif_tile_a;
typedef tile< 8, 8, nv_bfloat162> motif_tile_b;
typedef tile<16, 8, float> motif_tile_c;

/* Motif-3 full-attention prefill follows the compute-friendly MLA path from
 * the official Motif vLLM port: W_UK/W_UV are materialized for one bounded KV
 * chunk and attention runs at QK=192, V=128.  A caller can merge several KV
 * chunks exactly from the returned log-sum-exp values, so no expanded 256K KV
 * cache is required. */
/* GB10 has 100 KiB shared / SM. Staging Q+K+V was 36 KiB and capped the
 * kernel at two CTAs; Q already lives in qa[] for the whole K walk, so
 * loading it once from global frees enough shared memory for three CTAs.
 * TK=32 (two 16-key consume steps) stays under the 3-CTA budget (~21 KiB);
 * TK=64 drops to two CTAs and is slower on the late-chunk Motif walk. */
__global__ __launch_bounds__(M3_FA_WARPS * 32, 3)
void motif3_fattn_hmma_kernel(
        float * __restrict__ heads,
        float * __restrict__ lse,
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const uint32_t n_query,
        const uint32_t query_pos0,
        const uint32_t n_kv,
        const uint32_t kv_pos0,
        const uint32_t n_head,
        const uint32_t n_head_kv,
        const float scale,
        const uint32_t window) {
    const uint32_t tq0 = blockIdx.x * M3_FA_TQ;
    const uint32_t h = blockIdx.y;
    if (tq0 >= n_query || h >= n_head) return;
    const uint32_t group = n_head / n_head_kv;
    const uint32_t kvh = h / group;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;

    __shared__ __nv_bfloat16 s_k[M3_FA_TK][M3_FA_QK_ROW];
    __shared__ __nv_bfloat16 s_v[M3_FA_TK][M3_FA_V_ROW];
    static_assert(3u * (sizeof(s_k) + sizeof(s_v) + sizeof(uint32_t)) <=
                      102400u,
                  "Motif FATTN shared memory no longer fits three GB10 CTAs");

    const uint32_t qrow[2] = {
        warp * M3_FA_WQ + lane / 4,
        warp * M3_FA_WQ + lane / 4 + 8u,
    };
    uint32_t qpos[2];
    bool alive[2];
    float row_m[2], row_l[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        alive[r] = tq0 + qrow[r] < n_query;
        qpos[r] = alive[r] ? query_pos0 + tq0 + qrow[r] : query_pos0;
        row_m[r] = -INFINITY;
        row_l[r] = 0.0f;
    }

    motif_tile_a qa[M3_FA_QK / 16];
#pragma unroll
    for (int kc = 0; kc < M3_FA_QK / 16; kc++) {
#pragma unroll
        for (int l = 0; l < motif_tile_a::ne; l++) {
            const int i = (l % 2) * 8 + (int)(lane / 4);
            const int j = (l / 2) * 4 + (int)(lane % 4);
            const uint32_t row = warp * M3_FA_WQ + (uint32_t)i;
            const uint32_t col = (uint32_t)kc * 16u + 2u * (uint32_t)j;
            const uint32_t t = tq0 + row < n_query ? tq0 + row : 0u;
            const float2 xy = *reinterpret_cast<const float2 *>(
                q + ((size_t)t * n_head + h) * M3_FA_QK + col);
            qa[kc].x[l] = __floats2bfloat162_rn(xy.x, xy.y);
        }
    }

    motif_tile_c output[M3_FA_V / 8];
    const uint32_t kv_last = kv_pos0 + n_kv - 1u;
    uint32_t block_last = 0u;
#pragma unroll
    for (int r = 0; r < 2; r++) {
        if (alive[r]) block_last = max(block_last, min(qpos[r], kv_last));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        block_last = max(
            block_last,
            __shfl_xor_sync(0xffffffffu, block_last, offset));
    __shared__ uint32_t shared_last;
    if (threadIdx.x == 0u) shared_last = 0u;
    __syncthreads();
    if (lane == 0u) atomicMax(&shared_last, block_last);
    __syncthreads();
    block_last = shared_last;

    uint32_t block_first = kv_pos0;
    if (window > 0u) {
        const uint32_t first_qpos = query_pos0 + tq0;
        const uint32_t history = window - 1u;
        if (first_qpos > history)
            block_first = max(block_first, first_qpos - history);
    }
    if (block_last >= block_first) {
        for (uint32_t kt0 = block_first; kt0 <= block_last; kt0 += M3_FA_TK) {
            const uint32_t remaining = block_last - kt0 + 1u;
            const uint32_t tile_len = remaining < (uint32_t)M3_FA_TK
                ? remaining : (uint32_t)M3_FA_TK;
            for (uint32_t idx = threadIdx.x * 4u;
                 idx < M3_FA_TK * M3_FA_QK; idx += blockDim.x * 4u) {
                const uint32_t r = idx / M3_FA_QK;
                const uint32_t c = idx - r * M3_FA_QK;
                const uint32_t src = r < tile_len ? kt0 + r : kt0;
                const uint32_t local = src - kv_pos0;
                const float4 x = *reinterpret_cast<const float4 *>(
                    k + ((size_t)local * n_head_kv + kvh) * M3_FA_QK + c);
                *reinterpret_cast<nv_bfloat162 *>(&s_k[r][c]) =
                    __floats2bfloat162_rn(x.x, x.y);
                *reinterpret_cast<nv_bfloat162 *>(&s_k[r][c + 2]) =
                    __floats2bfloat162_rn(x.z, x.w);
            }
            for (uint32_t idx = threadIdx.x * 4u;
                 idx < M3_FA_TK * M3_FA_V; idx += blockDim.x * 4u) {
                const uint32_t r = idx / M3_FA_V;
                const uint32_t c = idx - r * M3_FA_V;
                const uint32_t src = r < tile_len ? kt0 + r : kt0;
                const uint32_t local = src - kv_pos0;
                const float4 x = *reinterpret_cast<const float4 *>(
                    v + ((size_t)local * n_head_kv + kvh) * M3_FA_V + c);
                *reinterpret_cast<nv_bfloat162 *>(&s_v[r][c]) =
                    __floats2bfloat162_rn(x.x, x.y);
                *reinterpret_cast<nv_bfloat162 *>(&s_v[r][c + 2]) =
                    __floats2bfloat162_rn(x.z, x.w);
            }
            __syncthreads();

#pragma unroll
            for (int step = 0; step < M3_FA_TK / 16; step++) {
            const uint32_t kbase = (uint32_t)step * 16u;
            motif_tile_c scores[2];
#pragma unroll
            for (int nb = 0; nb < 2; nb++) {
                motif_tile_c zero;
                scores[nb] = zero;
#pragma unroll
                for (int kc = 0; kc < M3_FA_QK / 16; kc++) {
                    motif_tile_b keys;
#pragma unroll
                    for (int l = 0; l < motif_tile_b::ne; l++) {
                        const int i = (int)(lane / 4);
                        const int j = l * 4 + (int)(lane % 4);
                        keys.x[l] = *(const nv_bfloat162 *)&s_k[
                            kbase + nb * 8 + i][kc * 16 + 2 * j];
                    }
                    mma(scores[nb], qa[kc], keys);
                }
            }

            float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
            for (int nb = 0; nb < 2; nb++) {
#pragma unroll
                for (int l = 0; l < motif_tile_c::ne; l++) {
                    const int r = l / 2;
                    const uint32_t p =
                        kt0 + kbase + nb * 8u + (lane % 4u) * 2u + (l % 2u);
                    float score = scores[nb].x[l] * scale;
                    if (!alive[r] || p > qpos[r] ||
                        (window > 0u && p + window <= qpos[r]) ||
                        p >= kt0 + tile_len || p > kv_last) {
                        score = -INFINITY;
                    }
                    scores[nb].x[l] = score;
                    tile_max[r] = fmaxf(tile_max[r], score);
                }
            }
#pragma unroll
            for (int r = 0; r < 2; r++) {
                tile_max[r] = fmaxf(
                    tile_max[r],
                    __shfl_xor_sync(0xffffffffu, tile_max[r], 1));
                tile_max[r] = fmaxf(
                    tile_max[r],
                    __shfl_xor_sync(0xffffffffu, tile_max[r], 2));
            }

            float rescale[2];
            float tile_sum[2] = {0.0f, 0.0f};
#pragma unroll
            for (int r = 0; r < 2; r++) {
                const float next_max = fmaxf(row_m[r], tile_max[r]);
                rescale[r] = row_m[r] == -INFINITY
                    ? 0.0f : __expf(row_m[r] - next_max);
                row_m[r] = next_max;
            }
#pragma unroll
            for (int nb = 0; nb < 2; nb++) {
#pragma unroll
                for (int l = 0; l < motif_tile_c::ne; l++) {
                    const int r = l / 2;
                    const float weight =
                        scores[nb].x[l] == -INFINITY || row_m[r] == -INFINITY
                            ? 0.0f : __expf(scores[nb].x[l] - row_m[r]);
                    scores[nb].x[l] = weight;
                    tile_sum[r] += weight;
                }
            }
#pragma unroll
            for (int r = 0; r < 2; r++) {
                tile_sum[r] +=
                    __shfl_xor_sync(0xffffffffu, tile_sum[r], 1);
                tile_sum[r] +=
                    __shfl_xor_sync(0xffffffffu, tile_sum[r], 2);
                row_l[r] = row_l[r] * rescale[r] + tile_sum[r];
            }

            motif_tile_a probabilities;
#pragma unroll
            for (int l = 0; l < motif_tile_a::ne; l++) {
                probabilities.x[l] = __floats2bfloat162_rn(
                    scores[l / 2].x[(l % 2) * 2],
                    scores[l / 2].x[(l % 2) * 2 + 1]);
            }
#pragma unroll
            for (int cb = 0; cb < M3_FA_V / 8; cb++) {
#pragma unroll
                for (int l = 0; l < motif_tile_c::ne; l++)
                    output[cb].x[l] *= rescale[l / 2];
                motif_tile_b values;
#pragma unroll
                for (int l = 0; l < motif_tile_b::ne; l++) {
                    const int i = (int)(lane / 4);
                    const int j = l * 4 + (int)(lane % 4);
                    values.x[l] = __halves2bfloat162(
                        s_v[kbase + 2 * j][cb * 8 + i],
                        s_v[kbase + 2 * j + 1][cb * 8 + i]);
                }
                mma(output[cb], probabilities, values);
            }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int cb = 0; cb < M3_FA_V / 8; cb++) {
#pragma unroll
        for (int l = 0; l < motif_tile_c::ne; l++) {
            const int r = l / 2;
            if (!alive[r]) continue;
            const uint32_t token = tq0 + qrow[r];
            const int col = (int)(lane % 4) * 2 + (l % 2);
            /* A query row that saw no keys (an out-of-window SWA prefix
             * segment) must still publish a neutral partial: zero output
             * with -inf LSE, so the state merge weighs it out instead of
             * blending whatever the scratch buffer held before. */
            heads[((size_t)token * n_head + h) * M3_FA_V +
                  cb * 8 + col] = row_l[r] > 0.0f
                ? output[cb].x[l] / row_l[r] : 0.0f;
        }
    }
    if (lse && (lane & 3u) == 0u) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            if (!alive[r]) continue;
            const uint32_t token = tq0 + qrow[r];
            lse[(size_t)token * n_head + h] = row_l[r] > 0.0f
                ? row_m[r] + logf(row_l[r]) : -INFINITY;
        }
    }
}


/* dots3-note tensor-core operands are FP16, not BF16: the latent cache is
 * BF16 (every bf16 value inside the fp16 range converts exactly), and fp16
 * keeps three more mantissa bits for the rounded Q, P, activation and
 * dequantized-weight operands (2^-11 instead of 2^-8).  Normed latents and
 * absorbed queries stay far below the fp16 range; the conversions saturate
 * instead of producing inf so an outlier can never poison an MMA. */
__device__ __forceinline__ half2 dots3_f2_to_h2(float a, float b) {
    const float lim = 65504.0f;
    return __floats2half2_rn(fminf(fmaxf(a, -lim), lim), fminf(fmaxf(b, -lim), lim));
}

__device__ __forceinline__ uint32_t dots3_bf162_to_h2(uint32_t packed) {
    __nv_bfloat162 b;
    memcpy(&b, &packed, sizeof(b));
    const float2 f = __bfloat1622float2(b);
    const half2 h = dots3_f2_to_h2(f.x, f.y);
    uint32_t out;
    memcpy(&out, &h, sizeof(out));
    return out;
}

__device__ __forceinline__ uint4 dots3_bf16x8_to_h8(uint4 v) {
    return make_uint4(dots3_bf162_to_h2(v.x), dots3_bf162_to_h2(v.y),
                      dots3_bf162_to_h2(v.z), dots3_bf162_to_h2(v.w));
}

/* dots3-note latent attention on tensor cores (prefill widths).
 *
 * The absorbed MLA form keeps one BF16 latent row per key that every query
 * head shares, so for one token the 128 (full) or 64 (SWA) heads form a
 * proper GEMM against that token's key set: S = Q_abs . [latent | k_pe]^T
 * over 576 / 1088 dims, then O = P . latent over 512 / 1024 dims.  The
 * per-token key set is either the DSA top-k list (gathered rows, full
 * layers) or the causal / sliding window (contiguous, SWA layers), which is
 * why the block owns one token: grid (heads / (16 * MT), tokens).
 *
 * Block = 8 warps.  Each key tile of TK rows is staged once in shared
 * memory (latent + k_pe side by side, BF16, padded rows) and read twice:
 * as the B operand of S (keys on n) and as the B operand of O (keys on k,
 * latent columns on n).  Q for the block's MT*16 heads is staged once as
 * BF16.  Shared memory (GB10: 100 KiB / SM) picks the tile shape:
 *   full  latent 512:  MT=2 (32 heads), TK=32, ~78 KiB, one CTA;
 *   SWA   latent 1024: MT=1 (16 heads), TK=16, ~75 KiB, one CTA.
 * QK work is split across warps by (m-tile, n-tile, k-slice); k-slices are
 * summed through shared memory.  PV splits the latent columns across the
 * warps of each m-tile so every warp keeps 16 n8 accumulators (64 regs).
 * The next tile's rows are fetched into registers while the current tile
 * is consumed (the fill latency was the whole per-tile budget when it sat
 * behind the barrier), and every fragment comes from ldmatrix.
 *
 * Numerics: Q and P are rounded to FP16 (the BF16 cache rows convert
 * exactly), the dot products accumulate in FP32 on the MMA, softmax runs in
 * FP32 with __expf; the scalar kernel keeps Q and P in FP32.  Same masking as the
 * scalar kernel: DSA filler ids (< 0 or > qpos) and slots beyond the cache
 * are dropped. */
enum {
    D3_FA_WARPS   = 8,
    D3_FA_THREADS = D3_FA_WARPS * 32,
    D3_FA_ROPE    = 64,
    D3_FA_PAD     = 8,
};

/* Two 8x8 b16 matrices (lanes 0-15 supply the row addresses). */
__device__ __forceinline__ void dots3_ldsm_x2(
        uint32_t &r0, uint32_t &r1, const __half *row_ptr) {
#ifdef TURING_MMA_AVAILABLE
    const uint32_t addr = (uint32_t)__cvta_generic_to_shared(row_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
#else
    GGML_UNUSED_VARS(r0, r1, row_ptr);
    NO_DEVICE_CODE;
#endif
}

template <uint32_t LATENT, uint32_t MT, uint32_t TK>
struct dots3_fattn_shape {
    static constexpr uint32_t heads     = MT * 16u;
    static constexpr uint32_t dim       = LATENT + D3_FA_ROPE;
    static constexpr uint32_t row       = dim + D3_FA_PAD;        /* fp16 */
    static constexpr uint32_t ksteps    = dim / 16u;
    static constexpr uint32_t nt        = TK / 8u;                 /* n8 tiles per key tile */
    static constexpr uint32_t ksplit    = D3_FA_WARPS / (MT * nt);
    static constexpr uint32_t kper      = ksteps / ksplit;
    static constexpr uint32_t pvw       = D3_FA_WARPS / MT;       /* PV warps per m-tile */
    static constexpr uint32_t cols      = LATENT / pvw;            /* latent cols per PV warp */
    static constexpr uint32_t cb        = cols / 8u;               /* n8 tiles per PV warp */
    static constexpr uint32_t prow      = TK + D3_FA_PAD;          /* P tile row, fp16 */
    static constexpr uint32_t chunks    = dim / 8u;                /* uint4 per key row */
    static constexpr uint32_t pf        = (TK * chunks + D3_FA_THREADS - 1u) / D3_FA_THREADS;
    static constexpr size_t q_bytes     = (size_t)heads * row * 2u;
    static constexpr size_t k_bytes     = (size_t)TK * row * 2u;
    static constexpr size_t p_bytes     = (size_t)MT * 16u * prow * 2u;
    static constexpr size_t part_bytes  = ksplit > 1u ? (size_t)ksplit * MT * nt * 128u * 4u : 0u;
    static constexpr size_t stat_bytes  = (size_t)MT * nt * 16u * 4u;
    static constexpr size_t valid_bytes = (size_t)TK * 4u;
    static constexpr size_t smem        = q_bytes + k_bytes + p_bytes + part_bytes + 2u * stat_bytes + valid_bytes;
    static_assert(MT * nt * ksplit == D3_FA_WARPS, "warp split must cover the block");
    static_assert(ksteps % ksplit == 0u, "k slices must be whole MMA steps");
    static_assert(cols % 16u == 0u, "PV column span must be n8 tile pairs");
    static_assert(TK % 16u == 0u, "P tile must be whole MMA k steps");
    static_assert((row * 2u) % 16u == 0u && (prow * 2u) % 16u == 0u, "ldmatrix rows must stay 16-byte aligned");
    static_assert(pf <= 32u, "row validity travels in one 32-bit mask");
    static_assert(smem <= 100u * 1024u, "dots3 FATTN tile no longer fits one GB10 CTA");
};

/* Fetch one key tile into registers: latent row then k_pe row per key,
 * zeros for masked (DSA filler, out of range) or absent keys.  Bit r of
 * *vmask says whether this thread's r-th chunk belongs to a valid key. */
template <uint32_t LATENT, uint32_t MT, uint32_t TK, bool SEL, bool WINDOW>
__device__ __forceinline__ void dots3_fattn_tile_fetch(
        uint4 regs[dots3_fattn_shape<LATENT, MT, TK>::pf], uint32_t *vmask,
        const __nv_bfloat16 *latent_cache, const __nv_bfloat16 *k_pe_cache,
        const int32_t *selected, uint32_t sel_stride, uint32_t token,
        uint32_t qpos, uint32_t first, uint32_t cache_cap,
        uint32_t kt0, uint32_t tile_len) {
    typedef dots3_fattn_shape<LATENT, MT, TK> S;
    uint32_t mask = 0u;
#pragma unroll
    for (uint32_t p = 0; p < S::pf; p++) {
        const uint32_t idx = threadIdx.x + p * D3_FA_THREADS;
        const uint32_t r = idx / S::chunks;
        const uint32_t c = idx - r * S::chunks;
        bool valid = idx < TK * S::chunks && r < tile_len;
        uint32_t slot = 0u;
        if (valid) {
            uint32_t logical;
            if constexpr (SEL) {
                const int32_t id = selected[(size_t)token * sel_stride + kt0 + r];
                valid = id >= 0 && (uint32_t)id <= qpos;
                logical = valid ? (uint32_t)id : 0u;
            } else {
                logical = first + kt0 + r;
            }
            slot = WINDOW ? logical % cache_cap : logical;
            valid = valid && slot < cache_cap;
        }
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (valid) {
            const __nv_bfloat16 *src = c < LATENT / 8u
                ? latent_cache + (size_t)slot * LATENT + c * 8u
                : k_pe_cache + (size_t)slot * D3_FA_ROPE + (c - LATENT / 8u) * 8u;
            v = *reinterpret_cast<const uint4 *>(src);
            mask |= 1u << p;
        }
        regs[p] = v;
    }
    *vmask = mask;
}

template <uint32_t LATENT, uint32_t MT, uint32_t TK>
__device__ __forceinline__ void dots3_fattn_tile_store(
        const uint4 regs[dots3_fattn_shape<LATENT, MT, TK>::pf], uint32_t vmask,
        __half *s_k, uint32_t *s_valid) {
    typedef dots3_fattn_shape<LATENT, MT, TK> S;
#pragma unroll
    for (uint32_t p = 0; p < S::pf; p++) {
        const uint32_t idx = threadIdx.x + p * D3_FA_THREADS;
        if (idx >= TK * S::chunks) break;
        const uint32_t r = idx / S::chunks;
        const uint32_t c = idx - r * S::chunks;
        *reinterpret_cast<uint4 *>(s_k + (size_t)r * S::row + c * 8u) = dots3_bf16x8_to_h8(regs[p]);
        if (c == 0u) s_valid[r] = (vmask >> p) & 1u;
    }
}

template <uint32_t LATENT, uint32_t MT, uint32_t TK, bool SEL, bool WINDOW>
__global__ __launch_bounds__(D3_FA_THREADS, 1)
void dots3_fattn_hmma_kernel(
        float * __restrict__ out,
        const float * __restrict__ q,
        const float * __restrict__ q_absorbed,
        const __nv_bfloat16 * __restrict__ latent_cache,
        const __nv_bfloat16 * __restrict__ k_pe_cache,
        const int32_t * __restrict__ selected,
        const uint32_t sel_stride,
        const uint32_t pos0,
        const uint32_t cache_cap,
        const uint32_t window,
        const uint32_t q_heads,
        const uint32_t qk_nope,
        const float scale) {
    typedef dots3_fattn_shape<LATENT, MT, TK> S;
    extern __shared__ __align__(16) unsigned char d3_smem[];
    __half *s_q = (__half *)d3_smem;
    __half *s_k = (__half *)(d3_smem + S::q_bytes);
    __half *s_p = (__half *)(d3_smem + S::q_bytes + S::k_bytes);
    float *s_part = (float *)(d3_smem + S::q_bytes + S::k_bytes + S::p_bytes);
    float *s_max = (float *)((unsigned char *)s_part + S::part_bytes);
    float *s_sum = s_max + MT * S::nt * 16u;
    uint32_t *s_valid = (uint32_t *)(s_sum + MT * S::nt * 16u);

    const uint32_t token = blockIdx.y;
    const uint32_t head0 = blockIdx.x * S::heads;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    if (head0 + S::heads > q_heads) return;

    /* Warp roles.  QK: (m-tile, n-tile, k-slice).  PV: (m-tile, column
     * span).  Both roles of one warp sit on the same m-tile, so the warp's
     * softmax statistics serve both. */
    const uint32_t mt = warp / (S::nt * S::ksplit);
    const uint32_t qk_rem = warp - mt * (S::nt * S::ksplit);
    const uint32_t nt = qk_rem % S::nt;
    const uint32_t ks = qk_rem / S::nt;
    const uint32_t pv_col0 = (warp - mt * S::pvw) * S::cols;
    const uint32_t key_dim = qk_nope + D3_FA_ROPE;
    const uint32_t qpos = pos0 + token;
    const uint32_t end = qpos + 1u;
    const uint32_t visible = WINDOW ? min(end, window) : end;
    const uint32_t first = end - visible;
    const uint32_t count = SEL ? sel_stride : visible;
    /* ldmatrix lane addressing: A/P fragments (x4: rows 0-7, 8-15 x k
     * 0-7, 8-15), K fragments (x2: keys 0-7 x k 0-7, 8-15), V fragments
     * (x4.trans: keys 0-7, 8-15 x two 8-column blocks). */
    const uint32_t a_row = ((lane >> 3) & 1u) * 8u + (lane & 7u);
    const uint32_t a_col = (lane >> 4) * 8u;
    const uint32_t b_row = lane & 7u;
    const uint32_t b_col = ((lane >> 3) & 1u) * 8u;

    /* Stage Q: absorbed latent part then the rotated rope tail, BF16. */
    for (uint32_t idx = tid; idx < S::heads * (LATENT / 4u); idx += D3_FA_THREADS) {
        const uint32_t h = idx / (LATENT / 4u);
        const uint32_t c = (idx - h * (LATENT / 4u)) * 4u;
        const float4 x = *reinterpret_cast<const float4 *>(
            q_absorbed + ((size_t)token * q_heads + head0 + h) * LATENT + c);
        half2 *dst = reinterpret_cast<half2 *>(s_q + (size_t)h * S::row + c);
        dst[0] = dots3_f2_to_h2(x.x, x.y);
        dst[1] = dots3_f2_to_h2(x.z, x.w);
    }
    for (uint32_t idx = tid; idx < S::heads * (D3_FA_ROPE / 4u); idx += D3_FA_THREADS) {
        const uint32_t h = idx / (D3_FA_ROPE / 4u);
        const uint32_t c = (idx - h * (D3_FA_ROPE / 4u)) * 4u;
        const float4 x = *reinterpret_cast<const float4 *>(
            q + ((size_t)token * q_heads + head0 + h) * key_dim + qk_nope + c);
        half2 *dst = reinterpret_cast<half2 *>(s_q + (size_t)h * S::row + LATENT + c);
        dst[0] = dots3_f2_to_h2(x.x, x.y);
        dst[1] = dots3_f2_to_h2(x.z, x.w);
    }

    float row_m[2] = {-INFINITY, -INFINITY};
    float row_l[2] = {0.0f, 0.0f};
    tile_c output[S::cb];
    uint4 regs[S::pf];
    uint32_t vmask = 0u;
    if (count > 0u) {
        dots3_fattn_tile_fetch<LATENT, MT, TK, SEL, WINDOW>(
            regs, &vmask, latent_cache, k_pe_cache, selected, sel_stride,
            token, qpos, first, cache_cap, 0u, min((uint32_t)TK, count));
    }
    __syncthreads();

    for (uint32_t kt0 = 0; kt0 < count; kt0 += TK) {
        /* The previous iteration's trailing barrier released s_k. */
        dots3_fattn_tile_store<LATENT, MT, TK>(regs, vmask, s_k, s_valid);
        __syncthreads();
        const uint32_t kt1 = kt0 + TK;
        if (kt1 < count) {
            dots3_fattn_tile_fetch<LATENT, MT, TK, SEL, WINDOW>(
                regs, &vmask, latent_cache, k_pe_cache, selected, sel_stride,
                token, qpos, first, cache_cap, kt1, min((uint32_t)TK, count - kt1));
        }

        /* S partial for this warp's (m-tile, n-tile) over its k-slice. */
        tile_c acc;
#pragma unroll 4
        for (uint32_t kc = ks * S::kper; kc < (ks + 1u) * S::kper; kc++) {
            tile_a qa;
            tile_b kb;
            uint32_t *qa_x = reinterpret_cast<uint32_t *>(qa.x);
            uint32_t *kb_x = reinterpret_cast<uint32_t *>(kb.x);
            solar_fattn_ldsm_x4(
                qa_x[0], qa_x[1], qa_x[2], qa_x[3],
                s_q + (size_t)(mt * 16u + a_row) * S::row + kc * 16u + a_col);
            dots3_ldsm_x2(
                kb_x[0], kb_x[1],
                s_k + (size_t)(nt * 8u + b_row) * S::row + kc * 16u + b_col);
            mma(acc, qa, kb);
        }
        if constexpr (S::ksplit > 1u) {
            float4 *part = reinterpret_cast<float4 *>(s_part) +
                ((size_t)(ks * MT + mt) * S::nt + nt) * 32u + lane;
            *part = make_float4(acc.x[0], acc.x[1], acc.x[2], acc.x[3]);
            __syncthreads();
            if (ks == 0u) {
#pragma unroll
                for (uint32_t s = 1; s < S::ksplit; s++) {
                    const float4 p = reinterpret_cast<const float4 *>(s_part)[
                        ((size_t)(s * MT + mt) * S::nt + nt) * 32u + lane];
                    acc.x[0] += p.x; acc.x[1] += p.y; acc.x[2] += p.z; acc.x[3] += p.w;
                }
            }
        }

        /* Scale, mask, tile max (the k-slice 0 warps own the scores). */
        float tile_max[2] = {-INFINITY, -INFINITY};
        if (ks == 0u) {
#pragma unroll
            for (int l = 0; l < tile_c::ne; l++) {
                const uint32_t kk = nt * 8u + (lane % 4u) * 2u + (uint32_t)(l % 2);
                const float score = s_valid[kk] ? acc.x[l] * scale : -INFINITY;
                acc.x[l] = score;
                tile_max[l / 2] = fmaxf(tile_max[l / 2], score);
            }
#pragma unroll
            for (int r = 0; r < 2; r++) {
                tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 1));
                tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 2));
            }
            if ((lane & 3u) == 0u) {
                s_max[(mt * S::nt + nt) * 16u + lane / 4u] = tile_max[0];
                s_max[(mt * S::nt + nt) * 16u + lane / 4u + 8u] = tile_max[1];
            }
        }
        __syncthreads();

        /* Every warp of the m-tile folds the tile max into its running
         * statistics; the score owners then emit P (BF16) and row sums. */
        float rescale[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            float m = -INFINITY;
#pragma unroll
            for (uint32_t n = 0; n < S::nt; n++)
                m = fmaxf(m, s_max[(mt * S::nt + n) * 16u + lane / 4u + 8u * (uint32_t)r]);
            const float next_max = fmaxf(row_m[r], m);
            rescale[r] = row_m[r] == -INFINITY ? 0.0f : __expf(row_m[r] - next_max);
            row_m[r] = next_max;
        }
        if (ks == 0u) {
            float tile_sum[2] = {0.0f, 0.0f};
#pragma unroll
            for (int l = 0; l < tile_c::ne; l++) {
                const int r = l / 2;
                const float w = acc.x[l] == -INFINITY || row_m[r] == -INFINITY
                    ? 0.0f : __expf(acc.x[l] - row_m[r]);
                acc.x[l] = w;
                tile_sum[r] += w;
            }
#pragma unroll
            for (int r = 0; r < 2; r++) {
                tile_sum[r] += __shfl_xor_sync(0xffffffffu, tile_sum[r], 1);
                tile_sum[r] += __shfl_xor_sync(0xffffffffu, tile_sum[r], 2);
            }
            if ((lane & 3u) == 0u) {
                s_sum[(mt * S::nt + nt) * 16u + lane / 4u] = tile_sum[0];
                s_sum[(mt * S::nt + nt) * 16u + lane / 4u + 8u] = tile_sum[1];
            }
#pragma unroll
            for (int r = 0; r < 2; r++) {
                const uint32_t prow = mt * 16u + lane / 4u + 8u * (uint32_t)r;
                const uint32_t pcol = nt * 8u + (lane % 4u) * 2u;
                *reinterpret_cast<half2 *>(s_p + (size_t)prow * S::prow + pcol) =
                    __floats2half2_rn(acc.x[r * 2], acc.x[r * 2 + 1]);
            }
        }
        __syncthreads();

        /* O = O * rescale + P . latent over this warp's column span. */
#pragma unroll
        for (int r = 0; r < 2; r++) {
            float s = 0.0f;
#pragma unroll
            for (uint32_t n = 0; n < S::nt; n++)
                s += s_sum[(mt * S::nt + n) * 16u + lane / 4u + 8u * (uint32_t)r];
            row_l[r] = row_l[r] * rescale[r] + s;
        }
#pragma unroll
        for (uint32_t c = 0; c < S::cb; c++) {
#pragma unroll
            for (int l = 0; l < tile_c::ne; l++) output[c].x[l] *= rescale[l / 2];
        }
#pragma unroll
        for (uint32_t step = 0; step < TK / 16u; step++) {
            tile_a pa;
            uint32_t *pa_x = reinterpret_cast<uint32_t *>(pa.x);
            solar_fattn_ldsm_x4(
                pa_x[0], pa_x[1], pa_x[2], pa_x[3],
                s_p + (size_t)(mt * 16u + a_row) * S::prow + step * 16u + a_col);
            const __half *v_base = s_k + (size_t)(step * 16u + a_row) * S::row + pv_col0 + a_col;
#pragma unroll
            for (uint32_t c = 0; c < S::cb; c += 2u) {
                tile_b v0, v1;
                uint32_t *v0_x = reinterpret_cast<uint32_t *>(v0.x);
                uint32_t *v1_x = reinterpret_cast<uint32_t *>(v1.x);
                solar_fattn_ldsm_x4_trans(
                    v0_x[0], v0_x[1], v1_x[0], v1_x[1],
                    v_base + c * 8u);
                mma(output[c], pa, v0);
                mma(output[c + 1u], pa, v1);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 2; r++) {
        const float inv = row_l[r] > 0.0f ? 1.0f / row_l[r] : 0.0f;
        const uint32_t head = head0 + mt * 16u + lane / 4u + 8u * (uint32_t)r;
        float *dst = out + ((size_t)token * q_heads + head) * LATENT + pv_col0 + (lane % 4u) * 2u;
#pragma unroll
        for (uint32_t c = 0; c < S::cb; c++) {
            *reinterpret_cast<float2 *>(dst + c * 8u) =
                make_float2(output[c].x[r * 2] * inv, output[c].x[r * 2 + 1] * inv);
        }
    }
}

template <uint32_t LATENT, uint32_t MT, uint32_t TK, bool SEL, bool WINDOW>
static int dots3_fattn_launch(
        float *out, const float *q, const float *q_absorbed,
        const __nv_bfloat16 *latent, const __nv_bfloat16 *k_pe,
        const int32_t *selected, uint32_t sel_stride, uint32_t rows,
        uint32_t pos0, uint32_t cache_cap, uint32_t window,
        uint32_t q_heads, uint32_t qk_nope, float scale,
        cudaStream_t stream) {
    typedef dots3_fattn_shape<LATENT, MT, TK> S;
    auto kernel = dots3_fattn_hmma_kernel<LATENT, MT, TK, SEL, WINDOW>;
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)S::smem) != cudaSuccess) {
        return -2;
    }
    const dim3 grid(q_heads / S::heads, rows, 1);
    kernel<<<grid, D3_FA_THREADS, S::smem, stream>>>(
        out, q, q_absorbed, latent, k_pe, selected, sel_stride, pos0,
        cache_cap, window, q_heads, qk_nope, scale);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

template <uint32_t LATENT, uint32_t MT, uint32_t TK>
static int dots3_fattn_dispatch(
        float *out, const float *q, const float *q_absorbed,
        const __nv_bfloat16 *latent, const __nv_bfloat16 *k_pe,
        const int32_t *selected, uint32_t sel_stride, uint32_t rows,
        uint32_t pos0, uint32_t cache_cap, uint32_t window,
        uint32_t q_heads, uint32_t qk_nope, float scale,
        cudaStream_t stream) {
#define D3_FA_CALL(sel_, win_)                                               \
    dots3_fattn_launch<LATENT, MT, TK, sel_, win_>(                          \
        out, q, q_absorbed, latent, k_pe, selected, sel_stride, rows, pos0,  \
        cache_cap, window, q_heads, qk_nope, scale, stream)
    if (selected && window) return D3_FA_CALL(true, true);
    if (selected) return D3_FA_CALL(true, false);
    if (window) return D3_FA_CALL(false, true);
    return D3_FA_CALL(false, false);
#undef D3_FA_CALL
}

/* dots3-note latent value projection on tensor cores (prefill widths).
 *
 * attention[t, h, v] = sum_j latent_out[t, h, j] * W_UV[h][v][j] is one
 * [tokens x latent] . [latent x 128] GEMM per head.  The owner's transposed
 * Q8_0 artifact (ds4_repack_build_motif3_kv_b_value) stores, per head and
 * per 32-wide j block, the 128 value columns contiguously: a k-major
 * [j][v] layout that feeds the MMA B operand straight through
 * ldmatrix.trans.  Block = (head, 64 tokens), four warps of 16 tokens,
 * walking one 32-j block per step with the activations and the int8 codes
 * double-buffered in shared memory.  The int8 codes are exact in BF16, so
 * each block's product is accumulated separately and folded into the
 * output with its FP32 per-column scale; only the activations are rounded
 * (to FP16), which is the same contract as the attention kernel's Q. */
enum {
    D3_VP_TOKENS  = 64,
    D3_VP_WARPS   = 4,
    D3_VP_THREADS = D3_VP_WARPS * 32,
    D3_VP_KB      = 32,                      /* j per step (one Q8_0 block) */
    D3_VP_VALUES  = 128,
    D3_VP_A_ROW   = D3_VP_KB + D3_FA_PAD,    /* fp16 */
    D3_VP_B_ROW   = D3_VP_VALUES + D3_FA_PAD,
    D3_VP_CB      = D3_VP_VALUES / 8,        /* n8 tiles per warp */
};

struct dots3_vp_stage {
    float4 a[4];        /* 16 activations: token = tid / 2, j half = tid % 2 */
    uint4 b[2];         /* 32 codes: j = tid / 4, v quarter = tid % 4 */
    __half scale;       /* column tid */
};

__device__ __forceinline__ void dots3_vp_fetch(
        dots3_vp_stage &st, const float *latent, const __half *scale,
        const int8_t *code, uint32_t token0, uint32_t rows, uint32_t q_heads,
        uint32_t head, uint32_t latent_dim, uint32_t k_blocks, uint32_t b) {
    const uint32_t tid = threadIdx.x;
    const uint32_t t = token0 + tid / 2u;
    const uint32_t src_t = t < rows ? t : rows - 1u;
    const float *a_src = latent + ((size_t)src_t * q_heads + head) * latent_dim +
                         b * D3_VP_KB + (tid & 1u) * 16u;
#pragma unroll
    for (uint32_t i = 0; i < 4u; i++)
        st.a[i] = *reinterpret_cast<const float4 *>(a_src + i * 4u);
    const size_t hb = (size_t)head * k_blocks + b;
    const int8_t *b_src = code + (hb * D3_VP_KB + tid / 4u) * D3_VP_VALUES + (tid & 3u) * 32u;
    st.b[0] = *reinterpret_cast<const uint4 *>(b_src);
    st.b[1] = *reinterpret_cast<const uint4 *>(b_src + 16u);
    st.scale = scale[hb * D3_VP_VALUES + tid];
}

__device__ __forceinline__ void dots3_vp_store(
        const dots3_vp_stage &st, __half *s_a, __half *s_b,
        float *s_scale) {
    const uint32_t tid = threadIdx.x;
    half2 *a_dst = reinterpret_cast<half2 *>(
        s_a + (size_t)(tid / 2u) * D3_VP_A_ROW + (tid & 1u) * 16u);
#pragma unroll
    for (uint32_t i = 0; i < 4u; i++) {
        a_dst[2u * i] = dots3_f2_to_h2(st.a[i].x, st.a[i].y);
        a_dst[2u * i + 1u] = dots3_f2_to_h2(st.a[i].z, st.a[i].w);
    }
    half2 *b_dst = reinterpret_cast<half2 *>(
        s_b + (size_t)(tid / 4u) * D3_VP_B_ROW + (tid & 3u) * 32u);
#pragma unroll
    for (uint32_t i = 0; i < 2u; i++) {
        const uint32_t w[4] = {st.b[i].x, st.b[i].y, st.b[i].z, st.b[i].w};
#pragma unroll
        for (uint32_t k = 0; k < 4u; k++) {
            const int8_t *c = reinterpret_cast<const int8_t *>(&w[k]);
            b_dst[i * 8u + k * 2u] = __floats2half2_rn((float)c[0], (float)c[1]);
            b_dst[i * 8u + k * 2u + 1u] = __floats2half2_rn((float)c[2], (float)c[3]);
        }
    }
    s_scale[tid] = __half2float(st.scale);
}

__global__ __launch_bounds__(D3_VP_THREADS, 2)
void dots3_value_project_hmma_kernel(
        float * __restrict__ heads,
        const float * __restrict__ latent,
        const __half * __restrict__ scale,
        const int8_t * __restrict__ code,
        const float * __restrict__ gate_logits,
        const uint32_t rows,
        const uint32_t q_heads,
        const uint32_t latent_dim) {
    __shared__ __align__(16) __half s_a[2][D3_VP_TOKENS * D3_VP_A_ROW];
    __shared__ __align__(16) __half s_b[2][D3_VP_KB * D3_VP_B_ROW];
    __shared__ float s_scale[2][D3_VP_VALUES];
    const uint32_t head = blockIdx.x;
    const uint32_t token0 = blockIdx.y * D3_VP_TOKENS;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t k_blocks = latent_dim / D3_VP_KB;
    if (head >= q_heads || token0 >= rows) return;
    const uint32_t a_row = ((lane >> 3) & 1u) * 8u + (lane & 7u);
    const uint32_t a_col = (lane >> 4) * 8u;

    tile_c output[D3_VP_CB];
    dots3_vp_stage st;
    dots3_vp_fetch(st, latent, scale, code, token0, rows, q_heads, head,
                   latent_dim, k_blocks, 0u);
    for (uint32_t b = 0; b < k_blocks; b++) {
        const uint32_t buf = b & 1u;
        dots3_vp_store(st, s_a[buf], s_b[buf], s_scale[buf]);
        __syncthreads();
        if (b + 1u < k_blocks) {
            dots3_vp_fetch(st, latent, scale, code, token0, rows, q_heads,
                           head, latent_dim, k_blocks, b + 1u);
        }
        tile_c acc[D3_VP_CB];
#pragma unroll
        for (uint32_t kk = 0; kk < D3_VP_KB / 16u; kk++) {
            tile_a a;
            uint32_t *a_x = reinterpret_cast<uint32_t *>(a.x);
            solar_fattn_ldsm_x4(
                a_x[0], a_x[1], a_x[2], a_x[3],
                s_a[buf] + (size_t)(warp * 16u + a_row) * D3_VP_A_ROW + kk * 16u + a_col);
            const __half *b_base =
                s_b[buf] + (size_t)(kk * 16u + a_row) * D3_VP_B_ROW + a_col;
#pragma unroll
            for (uint32_t c = 0; c < D3_VP_CB; c += 2u) {
                tile_b b0, b1;
                uint32_t *b0_x = reinterpret_cast<uint32_t *>(b0.x);
                uint32_t *b1_x = reinterpret_cast<uint32_t *>(b1.x);
                solar_fattn_ldsm_x4_trans(
                    b0_x[0], b0_x[1], b1_x[0], b1_x[1],
                    b_base + c * 8u);
                mma(acc[c], a, b0);
                mma(acc[c + 1u], a, b1);
            }
        }
        /* Fold this block's exact int8 product in with its column scales. */
#pragma unroll
        for (uint32_t c = 0; c < D3_VP_CB; c++) {
            const float2 sc = *reinterpret_cast<const float2 *>(
                s_scale[buf] + c * 8u + (lane & 3u) * 2u);
            output[c].x[0] += acc[c].x[0] * sc.x;
            output[c].x[1] += acc[c].x[1] * sc.y;
            output[c].x[2] += acc[c].x[2] * sc.x;
            output[c].x[3] += acc[c].x[3] * sc.y;
        }
    }

    /* Headwise sigmoid gate folded in (the separate gate pass multiplied the
     * finished sums by the same factor). */
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const uint32_t token = token0 + warp * 16u + lane / 4u + 8u * (uint32_t)r;
        if (token >= rows) continue;
        float g = 1.0f;
        if (gate_logits) {
            const float x = gate_logits[(size_t)token * q_heads + head];
            g = x >= 0.0f ? 1.0f / (1.0f + __expf(-x))
                          : __expf(x) / (1.0f + __expf(x));
        }
        float *dst = heads + ((size_t)token * q_heads + head) * D3_VP_VALUES + (lane & 3u) * 2u;
#pragma unroll
        for (uint32_t c = 0; c < D3_VP_CB; c++) {
            *reinterpret_cast<float2 *>(dst + c * 8u) =
                make_float2(output[c].x[r * 2] * g, output[c].x[r * 2 + 1] * g);
        }
    }
}

/* dots3-note Q/K absorption on tensor cores (prefill widths).
 *
 * q_absorbed[t, h, j] = sum_d q_nope[t, h, d] * W_UK[h][d][j] is one
 * [tokens x nope] . [nope x latent] GEMM per head over the raw Q8_0
 * attn_kv_b rows (row h*(nope+128)+d holds latent_dim columns in 34-byte
 * blocks).  Block = (head, 64 tokens), four warps of 16 tokens; the whole
 * nope-wide A operand lives in registers (8 or 12 k-steps) and the kernel
 * walks the latent columns one 32-wide Q8_0 block at a time, dequantizing
 * that [nope x 32] slab to FP16 in a double-buffered shared tile.  The
 * scale varies along the reduction here (per (d, block)), so the weights
 * are rounded to FP16 as well as the activations. */
enum {
    D3_AB_TOKENS  = 64,
    D3_AB_WARPS   = 4,
    D3_AB_THREADS = D3_AB_WARPS * 32,
    D3_AB_NB      = 32,                    /* latent columns per step */
    D3_AB_B_ROW   = D3_AB_NB + D3_FA_PAD,  /* fp16 */
    D3_AB_MAX_K   = 192,
    D3_AB_ROWS_PT = (D3_AB_MAX_K + D3_AB_THREADS - 1) / D3_AB_THREADS,
    D3_AB_WORDS   = 9,                     /* 4-byte words covering one 34-byte block */
};

/* Raw Q8_0 block (scale + 32 codes) for row d, column block b, fetched as
 * nine aligned words.  The block starts at byte 34*b of the row: even b
 * starts on a word, odd b two bytes into one, so `shift` says where the
 * scale sits inside the first word. */
struct dots3_ab_stage {
    uint32_t w[D3_AB_ROWS_PT][D3_AB_WORDS];
};

__device__ __forceinline__ void dots3_ab_fetch(
        dots3_ab_stage &st, const unsigned char *weight, uint64_t row_bytes,
        uint32_t row0, uint32_t nope, uint32_t b) {
#pragma unroll
    for (uint32_t p = 0; p < D3_AB_ROWS_PT; p++) {
        const uint32_t d = threadIdx.x + p * D3_AB_THREADS;
        if (d >= nope) break;
        const unsigned char *blk = weight + (size_t)(row0 + d) * row_bytes + (size_t)b * 34u;
        const unsigned char *span = reinterpret_cast<const unsigned char *>(
            reinterpret_cast<uintptr_t>(blk) & ~(uintptr_t)3u);
        const uint32_t *words = reinterpret_cast<const uint32_t *>(span);
#pragma unroll
        for (uint32_t i = 0; i + 1u < D3_AB_WORDS; i++) st.w[p][i] = words[i];
        /* Bytes 32..35 of the span: an odd block (span starts two bytes
         * early) owns all four, an even block ends at byte 33 -- never read
         * the neighbour's bytes past the tensor's last row. */
        const uint16_t *tail = reinterpret_cast<const uint16_t *>(span + 32);
        st.w[p][D3_AB_WORDS - 1u] =
            (uint32_t)tail[0] | ((b & 1u) ? (uint32_t)tail[1] << 16 : 0u);
    }
}

__device__ __forceinline__ void dots3_ab_store(
        const dots3_ab_stage &st, __half *s_b, uint32_t nope, uint32_t b) {
    const uint32_t shift = (b & 1u) ? 2u : 0u;   /* bytes before the scale */
#pragma unroll
    for (uint32_t p = 0; p < D3_AB_ROWS_PT; p++) {
        const uint32_t d = threadIdx.x + p * D3_AB_THREADS;
        if (d >= nope) break;
        const unsigned char *bytes = reinterpret_cast<const unsigned char *>(st.w[p]);
        uint16_t sh;
        memcpy(&sh, bytes + shift, 2u);
        const float scale = __half2float(__ushort_as_half(sh));
        const int8_t *codes = reinterpret_cast<const int8_t *>(bytes + shift + 2u);
        half2 *dst = reinterpret_cast<half2 *>(s_b + (size_t)d * D3_AB_B_ROW);
#pragma unroll
        for (uint32_t k = 0; k < D3_AB_NB / 2u; k++) {
            dst[k] = dots3_f2_to_h2(scale * (float)codes[2u * k],
                                    scale * (float)codes[2u * k + 1u]);
        }
    }
}

template <uint32_t NOPE>
__global__ __launch_bounds__(D3_AB_THREADS, 2)
void dots3_absorb_hmma_kernel(
        float * __restrict__ out,
        const float * __restrict__ q,
        const unsigned char * __restrict__ weight,
        const uint32_t rows,
        const uint32_t q_heads,
        const uint32_t latent_dim,
        const uint32_t key_dim,
        const uint32_t value_dim,
        const uint64_t row_bytes) {
    __shared__ __align__(16) __half s_b[2][NOPE * D3_AB_B_ROW];
    const uint32_t head = blockIdx.x;
    const uint32_t token0 = blockIdx.y * D3_AB_TOKENS;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    if (head >= q_heads || token0 >= rows) return;
    const uint32_t row0 = head * (NOPE + value_dim);
    const uint32_t n_blocks = latent_dim / D3_AB_NB;
    const uint32_t a_row = ((lane >> 3) & 1u) * 8u + (lane & 7u);
    const uint32_t a_col = (lane >> 4) * 8u;

    /* A: this warp's 16 tokens x NOPE, straight from the F32 rows. */
    tile_a qa[NOPE / 16u];
#pragma unroll
    for (uint32_t kc = 0; kc < NOPE / 16u; kc++) {
#pragma unroll
        for (int l = 0; l < tile_a::ne; l++) {
            const uint32_t i = (uint32_t)((l % 2) * 8) + lane / 4u;
            const uint32_t j = (uint32_t)((l / 2) * 4) + (lane % 4u);
            uint32_t t = token0 + warp * 16u + i;
            if (t >= rows) t = rows - 1u;
            const float2 xy = *reinterpret_cast<const float2 *>(
                q + ((size_t)t * q_heads + head) * key_dim + kc * 16u + 2u * j);
            qa[kc].x[l] = dots3_f2_to_h2(xy.x, xy.y);
        }
    }

    dots3_ab_stage st;
    dots3_ab_fetch(st, weight, row_bytes, row0, NOPE, 0u);
    for (uint32_t b = 0; b < n_blocks; b++) {
        const uint32_t buf = b & 1u;
        dots3_ab_store(st, s_b[buf], NOPE, b);
        __syncthreads();
        if (b + 1u < n_blocks) dots3_ab_fetch(st, weight, row_bytes, row0, NOPE, b + 1u);
        tile_c acc[D3_AB_NB / 8u];
#pragma unroll
        for (uint32_t kc = 0; kc < NOPE / 16u; kc++) {
            const __half *b_base =
                s_b[buf] + (size_t)(kc * 16u + a_row) * D3_AB_B_ROW + a_col;
#pragma unroll
            for (uint32_t c = 0; c < D3_AB_NB / 8u; c += 2u) {
                tile_b b0, b1;
                uint32_t *b0_x = reinterpret_cast<uint32_t *>(b0.x);
                uint32_t *b1_x = reinterpret_cast<uint32_t *>(b1.x);
                solar_fattn_ldsm_x4_trans(
                    b0_x[0], b0_x[1], b1_x[0], b1_x[1],
                    b_base + c * 8u);
                mma(acc[c], qa[kc], b0);
                mma(acc[c + 1u], qa[kc], b1);
            }
        }
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const uint32_t token = token0 + warp * 16u + lane / 4u + 8u * (uint32_t)r;
            if (token >= rows) continue;
            float *dst = out + ((size_t)token * q_heads + head) * latent_dim +
                         b * D3_AB_NB + (lane & 3u) * 2u;
#pragma unroll
            for (uint32_t c = 0; c < D3_AB_NB / 8u; c++) {
                *reinterpret_cast<float2 *>(dst + c * 8u) =
                    make_float2(acc[c].x[r * 2], acc[c].x[r * 2 + 1]);
            }
        }
    }
}

}  // namespace

static int solar_fattn_gqa_pair(int n_head, int n_head_kv) {
    if (n_head_kv <= 0 || n_head % n_head_kv != 0) return 0;
    const int group = n_head / n_head_kv;
    if (group < 2 || (group % 2) != 0) return 0;
    /* Diagnostic only: DS4_SOLAR_FATTN_GQA2=0 restores the one-head
     * kernel so tests can compare the pair path against it. */
    const char *value = getenv("DS4_SOLAR_FATTN_GQA2");
    if (value && value[0] == '0') return 0;
    return 1;
}

/* Diagnostic only: DS4_FATTN_HMMA_LDSM=0 restores the scalar fragment
 * loads and the direct tile fill in the GQA-pair kernel (bit-identical,
 * slower).  Read per call so the kernel test can toggle it. */
static int solar_fattn_ldsm_enabled(void) {
    const char *value = getenv("DS4_FATTN_HMMA_LDSM");
    if (value && value[0] == '0') return 0;
    return 1;
}

template <int FORMAT>
static void solar_fattn_launch(
        float *heads, const float *q, const void *kv, uint64_t row_bytes,
        int n_tokens, int pos0, int n_head, int n_head_kv, int kv_cap,
        int window, float scale, cudaStream_t stream) {
    const int tiles = (n_tokens + FA_TQ - 1) / FA_TQ;
    if (solar_fattn_gqa_pair(n_head, n_head_kv)) {
        const dim3 grid(tiles, n_head / 2, 1);
        if constexpr (FORMAT == SOLAR_KV_KFP8_VFP4) {
            if (solar_fattn_ldsm_enabled() &&
                solar_fattn_ws_try(heads, q, kv, row_bytes, n_tokens, pos0,
                                   n_head, n_head_kv, kv_cap, window, scale,
                                   stream)) {
                return;
            }
        }
        if (solar_fattn_ldsm_enabled()) {
            ds4_fattn_hmma_gqa2_kernel<FORMAT, true>
                <<<grid, FA_WARPS * 32 * 2, 0, stream>>>(
                    heads, q, kv, row_bytes, (uint32_t)n_tokens,
                    (uint32_t)pos0, (uint32_t)n_head, (uint32_t)n_head_kv,
                    (uint32_t)kv_cap, (uint32_t)window, scale);
        } else {
            ds4_fattn_hmma_gqa2_kernel<FORMAT, false>
                <<<grid, FA_WARPS * 32 * 2, 0, stream>>>(
                    heads, q, kv, row_bytes, (uint32_t)n_tokens,
                    (uint32_t)pos0, (uint32_t)n_head, (uint32_t)n_head_kv,
                    (uint32_t)kv_cap, (uint32_t)window, scale);
        }
        return;
    }
    const dim3 grid(tiles, n_head, 1);
    ds4_fattn_hmma_kernel<FORMAT>
        <<<grid, FA_WARPS * 32, 0, stream>>>(
            heads, q, kv, row_bytes, (uint32_t)n_tokens, (uint32_t)pos0,
            (uint32_t)n_head, (uint32_t)n_head_kv, (uint32_t)kv_cap,
            (uint32_t)window, scale);
}

extern "C" int ds4_mmq_exaone_prefill_attn_hmma(
        float *heads, const float *q, const void *kv,
        int n_tokens, int pos0, int n_head, int n_head_kv, int head_dim,
        int kv_cap, int window, float scale, cudaStream_t stream) {
    if (!heads || !q || !kv || n_tokens <= 0 || pos0 < 0 || n_head <= 0 ||
        n_head_kv <= 0 || head_dim != FA_HD || kv_cap <= 0 ||
        n_head % n_head_kv != 0) {
        return -1;
    }
    const int device = ggml_cuda_get_device();
    if (ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_AMPERE) return -1;
    solar_fattn_launch<SOLAR_KV_BF16>(
        heads, q, kv, 0u, n_tokens, pos0, n_head, n_head_kv, kv_cap,
        window, scale, stream);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

extern "C" int ds4_mmq_solar_prefill_attn_ws_available(
        const void *kv, size_t row_bytes) {
    if (!kv || row_bytes == 0u) { return 0; }
    const int device = ggml_cuda_get_device();
    if (ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_AMPERE) { return 0; }
    switch (solar_fattn_ws_width(kv, (uint64_t)row_bytes)) {
    case 16: return solar_fattn_ws_device_ok<16>();
    case 8: return solar_fattn_ws_device_ok<8>();
    default: return 0;
    }
}

extern "C" int ds4_mmq_solar_prefill_attn_hmma(
        float *heads, const float *q, const void *kv,
        int format, size_t row_bytes,
        int n_tokens, int pos0, int n_head, int n_head_kv, int head_dim,
        int kv_cap, int window, float scale, cudaStream_t stream) {
    if (!heads || !q || !kv || format < SOLAR_KV_FP8 ||
        format > SOLAR_KV_KFP8_VFP4 || row_bytes == 0u || n_tokens <= 0 ||
        pos0 < 0 || n_head <= 0 || n_head_kv <= 0 || head_dim != FA_HD ||
        kv_cap <= 0 || n_head % n_head_kv != 0) {
        return -1;
    }
    const int device = ggml_cuda_get_device();
    if (ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_AMPERE) return -1;
    switch (format) {
    case SOLAR_KV_FP8:
        solar_fattn_launch<SOLAR_KV_FP8>(
            heads, q, kv, (uint64_t)row_bytes, n_tokens, pos0, n_head,
            n_head_kv, kv_cap, window, scale, stream);
        break;
    case SOLAR_KV_FP4:
        solar_fattn_launch<SOLAR_KV_FP4>(
            heads, q, kv, (uint64_t)row_bytes, n_tokens, pos0, n_head,
            n_head_kv, kv_cap, window, scale, stream);
        break;
    case SOLAR_KV_KFP8_VFP4:
        solar_fattn_launch<SOLAR_KV_KFP8_VFP4>(
            heads, q, kv, (uint64_t)row_bytes, n_tokens, pos0, n_head,
            n_head_kv, kv_cap, window, scale, stream);
        break;
    default:
        return -1;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

extern "C" int ds4_mmq_motif3_prefill_attn_hmma(
        float *heads, float *lse, const float *q,
        const float *k, const float *v,
        int n_query, int query_pos0, int n_kv, int kv_pos0,
        int n_head, int n_head_kv, int qk_dim, int v_dim,
        float scale, int window, cudaStream_t stream) {
    if (!heads || !q || !k || !v || n_query <= 0 || query_pos0 < 0 ||
        n_kv <= 0 || kv_pos0 < 0 || n_head <= 0 || n_head_kv <= 0 ||
        n_head % n_head_kv != 0 || qk_dim != M3_FA_QK ||
        v_dim != M3_FA_V || kv_pos0 > query_pos0 || window < 0) {
        return -1;
    }
    const int device = ggml_cuda_get_device();
    if (ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_AMPERE) return -1;
    const dim3 grid((n_query + M3_FA_TQ - 1) / M3_FA_TQ, n_head, 1);
    motif3_fattn_hmma_kernel<<<grid, M3_FA_WARPS * 32, 0, stream>>>(
        heads, lse, q, k, v,
        (uint32_t)n_query, (uint32_t)query_pos0,
        (uint32_t)n_kv, (uint32_t)kv_pos0,
        (uint32_t)n_head, (uint32_t)n_head_kv, scale, (uint32_t)window);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

extern "C" int ds4_mmq_dots3_prefill_attn_hmma(
        float *out, const float *q, const float *q_absorbed,
        const void *latent_cache, const void *k_pe_cache,
        const int32_t *selected, int sel_stride,
        int rows, int pos0, int cache_cap, int window,
        int q_heads, int latent_dim, int qk_nope, int qk_rope,
        float scale, cudaStream_t stream) {
    if (!out || !q || !q_absorbed || !latent_cache || !k_pe_cache ||
        rows <= 0 || pos0 < 0 || cache_cap <= 0 || window < 0 ||
        q_heads <= 0 || qk_nope <= 0 || qk_rope != D3_FA_ROPE ||
        (selected && sel_stride <= 0)) {
        return -1;
    }
    const int device = ggml_cuda_get_device();
    if (ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_AMPERE) return -1;
    const __nv_bfloat16 *latent = (const __nv_bfloat16 *)latent_cache;
    const __nv_bfloat16 *k_pe = (const __nv_bfloat16 *)k_pe_cache;
    if (latent_dim == 512 && q_heads % 32 == 0) {
        return dots3_fattn_dispatch<512u, 2u, 32u>(
            out, q, q_absorbed, latent, k_pe, selected, (uint32_t)sel_stride,
            (uint32_t)rows, (uint32_t)pos0, (uint32_t)cache_cap,
            (uint32_t)window, (uint32_t)q_heads, (uint32_t)qk_nope, scale,
            stream);
    }
    if (latent_dim == 1024 && q_heads % 16 == 0) {
        return dots3_fattn_dispatch<1024u, 1u, 16u>(
            out, q, q_absorbed, latent, k_pe, selected, (uint32_t)sel_stride,
            (uint32_t)rows, (uint32_t)pos0, (uint32_t)cache_cap,
            (uint32_t)window, (uint32_t)q_heads, (uint32_t)qk_nope, scale,
            stream);
    }
    return -1;
}

extern "C" int ds4_mmq_dots3_value_project_hmma(
        float *heads, const float *latent, const void *scale,
        const void *code, const float *gate_logits, int rows, int q_heads,
        int latent_dim, cudaStream_t stream) {
    if (!heads || !latent || !scale || !code || rows <= 0 || q_heads <= 0 ||
        latent_dim <= 0 || latent_dim % D3_VP_KB != 0) {
        return -1;
    }
    const int device = ggml_cuda_get_device();
    if (ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_AMPERE) return -1;
    const dim3 grid((unsigned)q_heads,
                    (unsigned)((rows + D3_VP_TOKENS - 1) / D3_VP_TOKENS), 1);
    dots3_value_project_hmma_kernel<<<grid, D3_VP_THREADS, 0, stream>>>(
        heads, latent, (const __half *)scale, (const int8_t *)code,
        gate_logits, (uint32_t)rows, (uint32_t)q_heads, (uint32_t)latent_dim);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

extern "C" int ds4_mmq_dots3_absorb_hmma(
        float *out, const float *q, const void *weight,
        int rows, int q_heads, int latent_dim, int qk_nope, int key_dim,
        int value_dim, size_t row_bytes, cudaStream_t stream) {
    if (!out || !q || !weight || rows <= 0 || q_heads <= 0 ||
        latent_dim <= 0 || latent_dim % D3_AB_NB != 0 ||
        (qk_nope != 128 && qk_nope != 192) || key_dim < qk_nope ||
        value_dim <= 0 || row_bytes == 0u ||
        row_bytes != (size_t)(latent_dim / 32) * 34u) {
        return -1;
    }
    const int device = ggml_cuda_get_device();
    if (ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_AMPERE) return -1;
    const dim3 grid((unsigned)q_heads,
                    (unsigned)((rows + D3_AB_TOKENS - 1) / D3_AB_TOKENS), 1);
    if (qk_nope == 128) {
        dots3_absorb_hmma_kernel<128u><<<grid, D3_AB_THREADS, 0, stream>>>(
            out, q, (const unsigned char *)weight, (uint32_t)rows,
            (uint32_t)q_heads, (uint32_t)latent_dim, (uint32_t)key_dim,
            (uint32_t)value_dim, (uint64_t)row_bytes);
    } else {
        dots3_absorb_hmma_kernel<192u><<<grid, D3_AB_THREADS, 0, stream>>>(
            out, q, (const unsigned char *)weight, (uint32_t)rows,
            (uint32_t)q_heads, (uint32_t)latent_dim, (uint32_t)key_dim,
            (uint32_t)value_dim, (uint64_t)row_bytes);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
