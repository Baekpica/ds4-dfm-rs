#ifndef DS4_INKLING2_ATTN_CUH
#define DS4_INKLING2_ATTN_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <stdint.h>

/* Inkling 2 attention (transformers 109a7d19 eager_attention_forward): 48
 * query heads over 8 KV heads of 128, scores BF16(BF16(q . k) * 128^-1/2)
 * plus the BF16 relative bias of distance d < 1024, a per-head sink logit in
 * the F32 softmax denominator, BF16 probabilities times BF16 V, then the
 * sigmoid output gate. Sliding layers see a 1,024-key window of a BF16 ring
 * (row p at slot p % cap); full layers every key from 0 (row p at p).
 *
 * Full layers may store K/V as NVFP4 pages (ds4_inkling2.h): their tiles
 * stream raw and decode to BF16 in shared memory (exact: an E2M1 value times
 * its E4M3 scale is a BF16), keys outside the CTA's range decoded as zeros.
 *
 * Prefill rows run on BF16 tensor cores: a CTA owns 64 queries of two heads
 * sharing a KV head (four warps of 16 queries per head); K/V tiles of 64
 * keys stream through a two-stage cp.async ring; the softmax runs online per
 * tile, initialized with the sink (m = sink, l = 1). Key tiles start at
 * multiples of 64 in absolute position and masked keys add exact zeros, so a
 * query's result does not depend on the chunk. Decode and verify rows split
 * their keys over CTAs (one per (split, KV head, row), a warp per query
 * head) and a merge adds the sink and the gate. */

enum {
    INK2A_HEADS = 48, INK2A_KV_HEADS = 8, INK2A_GROUP = INK2A_HEADS / INK2A_KV_HEADS, INK2A_DIM = 128,
    INK2A_KV = INK2A_KV_HEADS * INK2A_DIM, INK2A_ROW = 2 * INK2A_KV, INK2A_EXTENT = 1024,
    // Prefill tile geometry.
    INK2A_WQ = 16, INK2A_WARPS = 4, INK2A_TQ = INK2A_WQ * INK2A_WARPS, INK2A_CTA_HEADS = 2,
    INK2A_THREADS = INK2A_CTA_HEADS * INK2A_WARPS * 32, INK2A_TK = 64, INK2A_NB = INK2A_TK / 8,
    INK2A_KS = INK2A_TK / 16, INK2A_PAD = 8, INK2A_SROW = INK2A_DIM + INK2A_PAD, INK2A_CHUNK = 8,
    INK2A_ROW_CHUNKS = 2 * INK2A_DIM / INK2A_CHUNK, INK2A_TILE_CHUNKS = INK2A_TK * INK2A_ROW_CHUNKS,
    INK2A_CPT = INK2A_TILE_CHUNKS / INK2A_THREADS, INK2A_STAGE_ELEMS = 2 * INK2A_TK * INK2A_SROW,
    INK2A_SMEM = 2 * INK2A_STAGE_ELEMS * (int)sizeof(uint16_t),
    // Decode geometry: a 32-key tile per step, one warp per query head.
    INK2A_DTK = 32, INK2A_DTHREADS = INK2A_GROUP * 32, INK2A_PARTIAL = INK2A_DIM + 2,
    INK2A_SPLITS = 64, INK2A_SPLIT_ROWS = 8, INK2A_MIN_SPLIT = 128,
    // NVFP4 pages: 128 keys of one KV head, K data/scales then V data/scales.
    INK2A_FMT_BF16 = 0, INK2A_FMT_NVFP4 = 1,
    INK2A_KB = 128, INK2A_FP4 = INK2A_DIM / 2, INK2A_FP4_ROW = INK2A_FP4 + 8,
    INK2A_PAGE = 2 * INK2A_KB * INK2A_FP4_ROW, INK2A_BLOCK = INK2A_KV_HEADS * INK2A_PAGE,
    // A 64-key raw tile: K data, K scales, V data, V scales.
    INK2A_RAW_HALF = INK2A_TK * INK2A_FP4_ROW, INK2A_RAW = 2 * INK2A_RAW_HALF,
    INK2A_SMEM_FP4 = 2 * INK2A_RAW + INK2A_STAGE_ELEMS * (int)sizeof(uint16_t),
    /* Key segments (a full layer's keys attended in pieces):
     * ONE attends every key and writes the output; FIRST starts the state
     * (m, l, acc[128] per row and head, the sink as m = sink, l = 1); NEXT
     * continues it; LAST continues it and writes the output. */
    INK2A_ONE = 0, INK2A_FIRST = 1, INK2A_NEXT = 2, INK2A_LAST = 3, INK2A_STATE = INK2A_DIM + 2,
};
static_assert(INK2A_CPT * INK2A_THREADS == INK2A_TILE_CHUNKS, "tile copy splits evenly over the block");
static_assert(INK2A_GROUP % INK2A_CTA_HEADS == 0, "a CTA's heads share a KV head");

// The source's scaling: 128^-1/2 as a Python float, applied in F32.
#define INK2A_SCALE 0.08838834764831845f

__device__ __forceinline__ float ink2a_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

__device__ __forceinline__ uint32_t ink2a_pack(float a, float b) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t *>(&h);
}

__device__ __forceinline__ float ink2a_sigmoid(float x) { return __fdiv_rn(1.0f, 1.0f + expf(-x)); }

// A byte's two E2M1 codes (low nibble first, sign in bit 3) as floats:
// cvt.rn.f16x2.e2m1x2 is exact; older targets use the 0, 0.5, 1, 1.5, 2,
// 3, 4, 6 table.
__device__ __forceinline__ float2 ink2a_e2m1x2(uint32_t byte) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t h;
    asm("{ .reg .b8 lo, hi;\n mov.b16 {lo, hi}, %1;\n cvt.rn.f16x2.e2m1x2 %0, lo; }"
        : "=r"(h) : "h"((unsigned short)byte));
    return __half22float2(*reinterpret_cast<const __half2 *>(&h));
#else
    const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float lo = table[byte & 7u], hi = table[(byte >> 4) & 7u];
    return make_float2(byte & 8u ? -lo : lo, byte & 0x80u ? -hi : hi);
#endif
}

__device__ __forceinline__ void ink2a_mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ink2a_ldsm4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
        const uint16_t *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

__device__ __forceinline__ void ink2a_ldsm4t(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
        const uint16_t *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

// Cache row of key position p: sliding layers keep a ring of `cap` rows.
__device__ __forceinline__ const uint16_t *ink2a_row(const uint16_t *cache, uint32_t p, uint32_t cap,
        uint32_t window) {
    return cache + (uint64_t)(window ? p % cap : p) * INK2A_ROW;
}

// NVFP4 page of key position p and KV head h: block p / 128 of a store
// whose first block is block0.
__device__ __forceinline__ const uint8_t *ink2a_page(const uint8_t *cache, uint32_t p, uint32_t h,
        uint32_t block0 = 0) {
    return cache + (uint64_t)(p / INK2A_KB - block0) * INK2A_BLOCK + h * INK2A_PAGE;
}

// 16 NVFP4 values (two data words, low nibble first) times their E4M3 scale
// as eight BF16 pairs.
__device__ __forceinline__ void ink2a_fp4x16(uint2 data, uint8_t scale, uint32_t *out) {
    __nv_fp8_e4m3 s8;
    s8.__x = scale;
    const float s = (float)s8;
    const uint32_t words[2] = {data.x, data.y};
    #pragma unroll
    for (int w = 0; w < 2; w++) {
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            const float2 v = ink2a_e2m1x2((words[w] >> (8 * b)) & 0xffu);
            out[w * 4 + b] = ink2a_pack(v.x * s, v.y * s);
        }
    }
}

// One score: the source's BF16 roundings around the scale and the bias.
__device__ __forceinline__ float ink2a_score(float dot, float bias) {
    return ink2a_bf16(ink2a_bf16(ink2a_bf16(dot) * INK2A_SCALE) + bias);
}

template <int FMT, int MODE = INK2A_ONE>
__global__ __launch_bounds__(INK2A_THREADS, 1)
void ink2a_prefill(float *out, const float *q, const uint16_t *rel, const float *sinks, const float *gate,
        const void *cache_raw, uint32_t cap, uint32_t window, uint32_t rows, uint32_t base,
        float *state = nullptr, uint32_t seg_lo = 0, uint32_t seg_hi = UINT32_MAX, uint32_t block0 = 0) {
    extern __shared__ __align__(16) uint16_t ink2a_smem[];
    const uint16_t *cache = (const uint16_t *)cache_raw;
    const uint8_t *pages = (const uint8_t *)cache_raw;
    // BF16: a two-stage ring of K/V tiles. NVFP4: a two-stage ring of raw
    // tiles, then one decoded tile (stage 0 of the BF16 layout).
    uint8_t *raw = (uint8_t *)(ink2a_smem + (FMT == INK2A_FMT_NVFP4 ? INK2A_STAGE_ELEMS : 0));
    auto s_k = [&](int stage) {
        return reinterpret_cast<uint16_t (*)[INK2A_SROW]>(ink2a_smem +
            (FMT == INK2A_FMT_NVFP4 ? 0 : (size_t)stage * INK2A_STAGE_ELEMS));
    };
    auto s_v = [&](int stage) { return s_k(stage) + INK2A_TK; };
    const uint32_t tq0 = blockIdx.x * INK2A_TQ, h0 = blockIdx.y * INK2A_CTA_HEADS, kv_head = h0 / INK2A_GROUP;
    const uint32_t h = h0 + threadIdx.x / (INK2A_WARPS * 32);
    const uint32_t local = threadIdx.x % (INK2A_WARPS * 32), warp = local / 32, lane = local % 32;
    if (tq0 >= rows) { return; }

    // C-fragment rows lane/4 and lane/4 + 8 of this warp's 16-query tile.
    uint32_t qrow[2], qpos[2], qfirst[2];
    bool alive[2];
    float row_m[2], row_l[2];
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        qrow[r] = warp * INK2A_WQ + lane / 4 + r * 8;
        alive[r] = tq0 + qrow[r] < rows;
        const uint32_t t = alive[r] ? tq0 + qrow[r] : 0;
        qpos[r] = base + t;
        qfirst[r] = window && qpos[r] >= window ? qpos[r] - (window - 1) : 0;
        qfirst[r] = max(qfirst[r], seg_lo);
        // The sink is one more logit with a zero value.
        row_m[r] = ink2a_bf16(sinks[(uint64_t)t * INK2A_HEADS + h]);
        row_l[r] = 1.0f;
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

    const uint32_t q_lo = base + tq0;
    const uint32_t q_hi = base + (tq0 + INK2A_TQ - 1 < rows ? tq0 + INK2A_TQ - 1 : rows - 1);
    const uint32_t key_lo = max(window && q_lo >= window ? q_lo - (window - 1) : 0u, seg_lo);
    const uint32_t key_hi = min(q_hi, seg_hi - 1u);
    const uint32_t kt_start = key_lo & ~(uint32_t)(INK2A_TK - 1);
    const uint32_t tiles = key_hi >= key_lo ? (key_hi - kt_start) / INK2A_TK + 1 : 0u;
    float output[INK2A_DIM / 8][4];
    #pragma unroll
    for (int cb = 0; cb < INK2A_DIM / 8; cb++) {
        #pragma unroll
        for (int l = 0; l < 4; l++) { output[cb][l] = 0.0f; }
    }
    // A continued state: this quad's columns of the row's accumulator.
    if constexpr (MODE == INK2A_NEXT || MODE == INK2A_LAST) {
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (!alive[r]) { continue; }
            const float *st = state + ((uint64_t)(tq0 + qrow[r]) * INK2A_HEADS + h) * INK2A_STATE;
            row_m[r] = st[0];
            row_l[r] = st[1];
            #pragma unroll
            for (int cb = 0; cb < INK2A_DIM / 8; cb++) {
                const float2 a = *reinterpret_cast<const float2 *>(st + 2 + cb * 8 + (lane % 4) * 2);
                output[cb][2 * r] = a.x;
                output[cb][2 * r + 1] = a.y;
            }
        }
    }

    // Rows outside [key_lo, key_hi] repeat key_lo so masked keys stay finite.
    auto issue_tile = [&](int stage, uint32_t kt0) {
        if constexpr (FMT == INK2A_FMT_NVFP4) {
            // A 64-key tile is half of one block: four contiguous segments.
            const uint8_t *page = ink2a_page(pages, kt0, kv_head, block0);
            const uint32_t slot = kt0 % INK2A_KB;
            uint8_t *dst = raw + stage * INK2A_RAW;
            for (uint32_t c = threadIdx.x; c < INK2A_RAW / 16; c += INK2A_THREADS) {
                const uint32_t at = c * 16, half = at / INK2A_RAW_HALF, in = at % INK2A_RAW_HALF;
                const uint32_t data = INK2A_TK * INK2A_FP4;
                const uint8_t *src = page + half * INK2A_KB * INK2A_FP4_ROW +
                    (in < data ? slot * INK2A_FP4 + in : INK2A_KB * INK2A_FP4 + slot * 8 + (in - data));
                __pipeline_memcpy_async(dst + at, src, 16);
            }
        } else {
            #pragma unroll
            for (uint32_t p = 0; p < INK2A_CPT; p++) {
                const uint32_t idx = threadIdx.x + p * INK2A_THREADS;
                const uint32_t r = idx / INK2A_ROW_CHUNKS, j = idx % INK2A_ROW_CHUNKS;
                uint32_t key = kt0 + r;
                if (key < key_lo || key > key_hi) { key = key_lo; }
                const uint16_t *row = ink2a_row(cache, key, cap, window);
                const bool value = j >= INK2A_ROW_CHUNKS / 2;
                const uint32_t c = (j % (INK2A_ROW_CHUNKS / 2)) * INK2A_CHUNK;
                const uint16_t *src = row + (value ? INK2A_KV : 0) + kv_head * INK2A_DIM + c;
                uint16_t *dst = (value ? s_v(stage)[r] : s_k(stage)[r]) + c;
                __pipeline_memcpy_async(dst, src, INK2A_CHUNK * sizeof(uint16_t));
            }
        }
    };
    // Raw NVFP4 tile -> BF16 rows; keys outside [key_lo, key_hi] are zeros.
    auto decode_tile = [&](int stage, uint32_t kt0) {
        const uint8_t *src = raw + stage * INK2A_RAW;
        for (uint32_t g = threadIdx.x; g < 2 * INK2A_TK * 8; g += INK2A_THREADS) {
            const uint32_t value = g / (INK2A_TK * 8), key = (g / 8) % INK2A_TK, grp = g % 8;
            const uint8_t *half = src + value * INK2A_RAW_HALF;
            uint32_t packed[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            const uint32_t p = kt0 + key;
            if (p >= key_lo && p <= key_hi) {
                const uint2 data = *reinterpret_cast<const uint2 *>(half + key * INK2A_FP4 + grp * 8);
                ink2a_fp4x16(data, half[INK2A_TK * INK2A_FP4 + key * 8 + grp], packed);
            }
            uint16_t *dst = (value ? s_v(0)[key] : s_k(0)[key]) + grp * 16;
            *reinterpret_cast<uint4 *>(dst) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
            *reinterpret_cast<uint4 *>(dst + 8) = make_uint4(packed[4], packed[5], packed[6], packed[7]);
        }
    };
    auto consume = [&](int stage, uint32_t kt0) {
        const uint16_t (*sk)[INK2A_SROW] = s_k(stage);
        const uint16_t (*sv)[INK2A_SROW] = s_v(stage);
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
        #pragma unroll
        for (int nb = 0; nb < INK2A_NB; nb++) {
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const int r = l / 2;
                const uint32_t p = kt0 + nb * 8 + (lane % 4) * 2 + (l % 2);
                float x = -INFINITY;
                if (alive[r] && p <= qpos[r] && p >= qfirst[r] && p < seg_hi) {
                    const uint32_t d = qpos[r] - p;
                    const uint32_t t = tq0 + qrow[r];
                    const float bias = d < INK2A_EXTENT
                        ? __uint_as_float((uint32_t)rel[((uint64_t)t * INK2A_HEADS + h) * INK2A_EXTENT + d] << 16)
                        : 0.0f;
                    x = ink2a_score(s[nb][l], bias);
                }
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
            rescale[r] = __expf(row_m[r] - next);
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
                ink2a_ldsm4t(b0, b1, b2, b3, &sv[ks * 16 + v_row][cb * 8 + v_col]);
                ink2a_mma(output[cb], pa[ks], b0, b1);
                ink2a_mma(output[cb + 1], pa[ks], b2, b3);
            }
        }
    };

    // A key segment past this CTA's queries leaves it no tile to load.
    if (tiles) { issue_tile(0, kt_start); }
    __pipeline_commit();
    int stage = 0;
    #pragma unroll 1
    for (uint32_t t = 0; t < tiles; t++) {
        const uint32_t kt0 = kt_start + t * INK2A_TK;
        if (t + 1 < tiles) {
            issue_tile(stage ^ 1, kt0 + INK2A_TK);
            __pipeline_commit();
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
        if constexpr (FMT == INK2A_FMT_NVFP4) {
            decode_tile(stage, kt0);
            __syncthreads();
        }
        consume(stage, kt0);
        __syncthreads();
        stage ^= 1;
    }

    if constexpr (MODE == INK2A_FIRST || MODE == INK2A_NEXT) {
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (!alive[r]) { continue; }
            float *st = state + ((uint64_t)(tq0 + qrow[r]) * INK2A_HEADS + h) * INK2A_STATE;
            if (lane % 4 == 0) {
                st[0] = row_m[r];
                st[1] = row_l[r];
            }
            #pragma unroll
            for (int cb = 0; cb < INK2A_DIM / 8; cb++) {
                *reinterpret_cast<float2 *>(st + 2 + cb * 8 + (lane % 4) * 2) =
                    make_float2(output[cb][2 * r], output[cb][2 * r + 1]);
            }
        }
    } else {
        #pragma unroll
        for (int cb = 0; cb < INK2A_DIM / 8; cb++) {
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (!alive[r]) { continue; }
                const uint32_t t = tq0 + qrow[r];
                const uint64_t at = ((uint64_t)t * INK2A_HEADS + h) * INK2A_DIM + cb * 8 + (lane % 4) * 2;
                const float2 g = *reinterpret_cast<const float2 *>(gate + at);
                const float o0 = ink2a_bf16(__fdiv_rn(output[cb][2 * r], row_l[r]));
                const float o1 = ink2a_bf16(__fdiv_rn(output[cb][2 * r + 1], row_l[r]));
                *reinterpret_cast<float2 *>(out + at) = make_float2(
                    ink2a_bf16(o0 * ink2a_sigmoid(ink2a_bf16(g.x))), ink2a_bf16(o1 * ink2a_sigmoid(ink2a_bf16(g.y))));
            }
        }
    }
}

/* Decode partials: CTA (split, KV head, row); warp w scores head 6h + w over
 * the split's keys, a lane per key of each 32-key tile staged in shared
 * memory (lane j reads K row j from word (i + j) % 64 on, so the 32 rows'
 * words fall in distinct banks). partial = (m, l, acc[128]). */
template <int FMT>
__global__ __launch_bounds__(INK2A_DTHREADS)
void ink2a_decode(float *partials, const float *q, const uint16_t *rel, const void *cache_raw, uint32_t cap,
        uint32_t window, uint32_t base, uint32_t splits, uint32_t seg_lo = 0, uint32_t seg_hi = UINT32_MAX,
        uint32_t block0 = 0) {
    const uint16_t *cache = (const uint16_t *)cache_raw;
    const uint8_t *pages = (const uint8_t *)cache_raw;
    __shared__ __align__(16) uint32_t sk[INK2A_DTK][INK2A_DIM / 2], sv[INK2A_DTK][INK2A_DIM / 2];
    __shared__ float sq[INK2A_GROUP][INK2A_DIM];
    const uint32_t split = blockIdx.x, kv = blockIdx.y, row = blockIdx.z;
    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32, h = kv * INK2A_GROUP + warp;
    const uint32_t pos = base + row;
    const uint32_t lo = max(window && pos >= window ? pos - (window - 1) : 0u, seg_lo);
    const uint32_t hi = min(pos + 1, seg_hi), keys = hi > lo ? hi - lo : 0u;
    const uint32_t span = ((keys + splits - 1) / splits + INK2A_DTK - 1) / INK2A_DTK * INK2A_DTK;
    const uint32_t k0 = lo + split * span, k1 = min(hi, k0 + span);
    for (uint32_t i = lane; i < INK2A_DIM; i += 32) {
        sq[warp][i] = ink2a_bf16(q[((uint64_t)row * INK2A_HEADS + h) * INK2A_DIM + i]);
    }
    float m = -INFINITY, l = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint16_t *bias = rel + ((uint64_t)row * INK2A_HEADS + h) * INK2A_EXTENT;
    for (uint32_t t0 = k0; t0 < k1; t0 += INK2A_DTK) {
        __syncthreads();
        if constexpr (FMT == INK2A_FMT_NVFP4) {
            // 32 keys x (K, V) x 8 groups of 16 values, decoded to BF16 pairs.
            for (uint32_t i = threadIdx.x; i < INK2A_DTK * 2 * 8; i += INK2A_DTHREADS) {
                const uint32_t j = i / 16, value = (i / 8) % 2, grp = i % 8;
                const uint32_t p = min(t0 + j, k1 - 1), slot = p % INK2A_KB;
                const uint8_t *half = ink2a_page(pages, p, kv, block0) + value * INK2A_KB * INK2A_FP4_ROW;
                const uint2 data = *reinterpret_cast<const uint2 *>(half + slot * INK2A_FP4 + grp * 8);
                uint32_t *dst = (value ? sv[j] : sk[j]) + grp * 8;
                ink2a_fp4x16(data, half[INK2A_KB * INK2A_FP4 + slot * 8 + grp], dst);
            }
        } else {
            for (uint32_t i = threadIdx.x; i < INK2A_DTK * INK2A_DIM / 4; i += INK2A_DTHREADS) {
                const uint32_t j = i / (INK2A_DIM / 4), c = i % (INK2A_DIM / 4);
                const uint32_t p = min(t0 + j, k1 - 1);
                const uint16_t *src = ink2a_row(cache, p, cap, window) + kv * INK2A_DIM;
                const uint2 kk = *reinterpret_cast<const uint2 *>(src + 4 * c);
                const uint2 vv = *reinterpret_cast<const uint2 *>(src + INK2A_KV + 4 * c);
                sk[j][2 * c] = kk.x; sk[j][2 * c + 1] = kk.y;
                sv[j][2 * c] = vv.x; sv[j][2 * c + 1] = vv.y;
            }
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
            const uint32_t d = pos - p;
            score = ink2a_score(dot, d < INK2A_EXTENT ? __uint_as_float((uint32_t)bias[d] << 16) : 0.0f);
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

// out = BF16(BF16(sum of partials over (sink + sum of l)) * sigmoid(gate)).
__global__ void ink2a_merge(float *out, const float *partials, const float *sinks, const float *gate,
        uint32_t splits) {
    const uint32_t rh = blockIdx.x, d = threadIdx.x;
    const float *src = partials + (uint64_t)rh * INK2A_SPLITS * INK2A_PARTIAL;
    const float sink = ink2a_bf16(sinks[rh]);
    float top = sink;
    for (uint32_t s = 0; s < splits; s++) { top = fmaxf(top, src[s * INK2A_PARTIAL]); }
    float total = __expf(sink - top), value = 0.0f;
    for (uint32_t s = 0; s < splits; s++) {
        const float ms = src[s * INK2A_PARTIAL];
        if (ms == -INFINITY) { continue; }
        const float e = __expf(ms - top);
        total = fmaf(src[s * INK2A_PARTIAL + 1], e, total);
        value = fmaf(src[s * INK2A_PARTIAL + 2 + d], e, value);
    }
    const uint64_t at = (uint64_t)rh * INK2A_DIM + d;
    out[at] = ink2a_bf16(ink2a_bf16(__fdiv_rn(value, total)) * ink2a_sigmoid(ink2a_bf16(gate[at])));
}

static uint32_t ink2a_splits(uint32_t keys) {
    const uint32_t want = (keys + INK2A_MIN_SPLIT - 1) / INK2A_MIN_SPLIT;
    return want < 1 ? 1 : want > INK2A_SPLITS ? INK2A_SPLITS : want;
}

#endif
