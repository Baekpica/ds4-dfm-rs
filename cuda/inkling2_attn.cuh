#ifndef DS4_INKLING2_ATTN_CUH
#define DS4_INKLING2_ATTN_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
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
    INK2A_DTK = 32, INK2A_DTHREADS = INK2A_GROUP * 32,
    // A decode partial: (m, l, two pad floats, acc[128]); the pad keeps acc
    // 16-byte aligned for vector stores.
    INK2A_ACC = 4, INK2A_PARTIAL = INK2A_ACC + INK2A_DIM,
    INK2A_SPLITS = 64, INK2A_SPLIT_ROWS = 8, INK2A_MIN_SPLIT = 128, INK2A_SWA_SPLIT = 64,
    INK2A_FULL_STAGE_MIN_KEYS = 32768,
    // NVFP4 pages: 128 keys of one KV head, K data/scales then V data/scales.
    INK2A_FMT_BF16 = 0, INK2A_FMT_NVFP4 = 1,
    INK2A_KB = 128, INK2A_FP4 = INK2A_DIM / 2, INK2A_FP4_ROW = INK2A_FP4 + 8,
    INK2A_PAGE = 2 * INK2A_KB * INK2A_FP4_ROW, INK2A_BLOCK = INK2A_KV_HEADS * INK2A_PAGE,
    // A 64-key raw tile: K data, K scales, V data, V scales.
    INK2A_RAW_HALF = INK2A_TK * INK2A_FP4_ROW, INK2A_RAW = 2 * INK2A_RAW_HALF,
    INK2A_SMEM_FP4 = 2 * INK2A_RAW + INK2A_STAGE_ELEMS * (int)sizeof(uint16_t),
    /* Key segments (the SSD tier attends a full layer's keys in pieces):
     * ONE attends every key and writes the output; FIRST starts the state
     * (m, l, acc[128] per row and head, the sink as m = sink, l = 1); NEXT
     * continues it; LAST continues it and writes the output. */
    INK2A_ONE = 0, INK2A_FIRST = 1, INK2A_NEXT = 2, INK2A_LAST = 3, INK2A_STATE = INK2A_DIM + 2,
    /* Where a prefill tile's relative bias comes from: the table
     * rel[row][head][distance] loaded per score (LOAD) or ahead of Q.K
     * (AHEAD), or its rank-16 factors (RANK). The table is 2 KB per row and
     * head, read once: its DRAM traffic and load latency were over half of a
     * sliding layer's attention. RANK's `rel` holds the projection by
     * distance in B-fragment order ([1024][lane % 4][k 2q, 2q + 1, 8 + 2q,
     * 9 + 2q]), then every (row, head)'s 16 r values; a warp computes its 16
     * queries' biases over a 64-key tile's 79 distances with ten BF16 MMAs
     * and keeps them in shared memory. Rows of 87 entries start odd rows on
     * an odd entry, so each (key + 1, key) pair is an aligned 32-bit load,
     * and put a warp's 32 lanes on 32 banks in every store and load. */
    INK2A_BIAS_LOAD = 0, INK2A_BIAS_AHEAD = 1, INK2A_BIAS_RANK = 2,
    INK2A_RANK = 16, INK2A_RANK_PROJ = INK2A_EXTENT * INK2A_RANK,
    INK2A_RANK_BLOCKS = (INK2A_TK + INK2A_WQ + 7) / 8, INK2A_RANK_ROW = 87,
    INK2A_RANK_SMEM = INK2A_CTA_HEADS * INK2A_WARPS * INK2A_WQ * INK2A_RANK_ROW * (int)sizeof(uint16_t),
};
static_assert(INK2A_CPT * INK2A_THREADS == INK2A_TILE_CHUNKS, "tile copy splits evenly over the block");
static_assert(INK2A_RANK_ROW >= INK2A_RANK_BLOCKS * 8 && INK2A_RANK_ROW % 2 == 1 && INK2A_WQ % 2 == 0,
    "a warp's bias rows hold a tile's distances and pair up on 32-bit words");
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

template <int FMT, int MODE = INK2A_ONE, int BIAS = INK2A_BIAS_LOAD, bool FAR_TILES = false>
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

    // Rank bias: this warp's r rows as an A fragment, its tile biases after
    // the K/V stages.
    uint32_t ra[4] = {0, 0, 0, 0};
    uint16_t *srank = ink2a_smem + 2 * INK2A_STAGE_ELEMS +
        (threadIdx.x / 32) * INK2A_WQ * INK2A_RANK_ROW;
    if constexpr (BIAS == INK2A_BIAS_RANK) {
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            const int r = l % 2, j = (l / 2) * 4 + (int)(lane % 4);
            const uint32_t t = alive[r] ? tq0 + qrow[r] : 0;
            ra[l] = *reinterpret_cast<const uint32_t *>(
                rel + INK2A_RANK_PROJ + ((uint64_t)t * INK2A_HEADS + h) * INK2A_RANK + 2 * j);
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
        // Keep raw bias loads ahead of QK; immediate scalar use serialized
        // L1TEX waits. Volatile loads prevent the compiler from sinking them.
        uint32_t biases[INK2A_NB][4];
        if constexpr (BIAS == INK2A_BIAS_RANK) {
            // S[row][j] = r_row . proj[:, dlo + j]; the tile's key c of the
            // warp's query `row` is at distance dlo + row - c + 63.
            const int dlo = (int)(base + tq0 + warp * INK2A_WQ) - (int)kt0 - (INK2A_TK - 1);
            #pragma unroll
            for (int b = 0; b < INK2A_RANK_BLOCKS; b++) {
                const int d = min(max(dlo + 8 * b + (int)(lane / 4), 0), INK2A_EXTENT - 1);
                const uint2 w = *reinterpret_cast<const uint2 *>(rel + ((uint64_t)d * 4 + lane % 4) * 4);
                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                ink2a_mma(c, ra, w.x, w.y);
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    const uint32_t row = lane / 4 + r * 8;
                    uint16_t *dst = srank + row * INK2A_RANK_ROW + 8 * b + 2 * (lane % 4);
                    const uint32_t pair = ink2a_pack(c[2 * r], c[2 * r + 1]);
                    dst[0] = (uint16_t)pair;
                    dst[1] = (uint16_t)(pair >> 16);
                }
            }
            __syncwarp();
            #pragma unroll
            for (int nb = 0; nb < INK2A_NB; nb++) {
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    const uint32_t row = lane / 4 + r * 8, c0 = nb * 8 + (lane % 4) * 2;
                    const uint32_t pair = *reinterpret_cast<const uint32_t *>(
                        srank + row * INK2A_RANK_ROW + row + (INK2A_TK - 2) - c0);
                    biases[nb][2 * r] = pair >> 16;
                    biases[nb][2 * r + 1] = pair & 0xffffu;
                }
            }
        }
        if constexpr (BIAS == INK2A_BIAS_AHEAD) {
            #pragma unroll
            for (int nb = 0; nb < INK2A_NB; nb++) {
                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    const int r = l / 2;
                    const uint32_t p = kt0 + nb * 8 + (lane % 4) * 2 + (l % 2);
                    const uint32_t d = qpos[r] - p;
                    biases[nb][l] = 0;
                    if (alive[r] && p <= qpos[r] && p >= qfirst[r] && p < seg_hi && d < INK2A_EXTENT) {
                        const uint16_t *src = rel + ((uint64_t)(tq0 + qrow[r]) * INK2A_HEADS + h) * INK2A_EXTENT + d;
                        asm volatile("ld.global.u16 %0, [%1];" : "=r"(biases[nb][l]) : "l"(src) : "memory");
                    }
                }
            }
        }
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
        // Old complete tiles have zero bias and no masked scores.
        const bool far = FAR_TILES && FMT == INK2A_FMT_NVFP4 && MODE == INK2A_ONE &&
            tq0 + INK2A_TQ <= rows && q_lo >= kt0 + INK2A_TK + INK2A_EXTENT - 1 &&
            kt0 >= seg_lo && kt0 + INK2A_TK <= seg_hi;
        if (far) {
            #pragma unroll
            for (int nb = 0; nb < INK2A_NB; nb++) {
                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    const float x = ink2a_score(s[nb][l], 0.0f);
                    s[nb][l] = x;
                    tile_max[l / 2] = fmaxf(tile_max[l / 2], x);
                }
            }
        } else {
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
                        const float bias = d >= INK2A_EXTENT ? 0.0f
                            : BIAS != INK2A_BIAS_LOAD ? __uint_as_float(biases[nb][l] << 16)
                            : __uint_as_float((uint32_t)rel[((uint64_t)t * INK2A_HEADS + h) * INK2A_EXTENT + d] << 16);
                        x = ink2a_score(s[nb][l], bias);
                    }
                    s[nb][l] = x;
                    tile_max[r] = fmaxf(tile_max[r], x);
                }
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

/* Full-layer prefill over NVFP4 pages, ping-pong form (K2-Horizon's
 * k2a_prefill_pp with Inkling 2's sink, relative bias, key segments and
 * gate). ink2a_prefill's eight warps decode, score and sum in step, so the
 * tensor pipe idles through every decode and softmax. Here two decoded
 * tiles let the two heads' warp groups run out of phase:
 *
 *   group 0 (head h0)      decode K of tile t + 1 ─▶ consume tile t
 *   group 1 (head h0 + 1)  consume tile t ─▶ decode V of tile t + 1
 *
 * One barrier per tile; the raw tile after next loads meanwhile. Tiles
 * wholly below every query's bias span (the bulk of a long context) take no
 * mask and no bias, and the output rescale is skipped while every row's
 * maximum holds. Scores keep the source's BF16 roundings. */
enum { INK2A_PP_SMEM = 2 * INK2A_STAGE_ELEMS * (int)sizeof(uint16_t) + 2 * INK2A_RAW };

template <int MODE = INK2A_ONE>
__global__ __launch_bounds__(INK2A_THREADS, 1)
void ink2a_prefill_pp(float *out, const float *q, const uint16_t *rel, const float *sinks, const float *gate,
        const uint8_t *pages, uint32_t rows, uint32_t base, float *state = nullptr, uint32_t seg_lo = 0,
        uint32_t seg_hi = UINT32_MAX, uint32_t block0 = 0) {
    extern __shared__ __align__(16) uint16_t ink2a_pp_smem[];
    // Two decoded tiles (K rows then V rows), then a two-stage ring of raw tiles.
    uint8_t *raw = (uint8_t *)(ink2a_pp_smem + 2 * INK2A_STAGE_ELEMS);
    auto s_k = [&](uint32_t b) {
        return reinterpret_cast<uint16_t (*)[INK2A_SROW]>(ink2a_pp_smem + b * INK2A_STAGE_ELEMS);
    };
    const uint32_t tq0 = blockIdx.x * INK2A_TQ, h0 = blockIdx.y * INK2A_CTA_HEADS, kv_head = h0 / INK2A_GROUP;
    const uint32_t group = threadIdx.x / (INK2A_WARPS * 32), h = h0 + group;
    const uint32_t local = threadIdx.x % (INK2A_WARPS * 32), warp = local / 32, lane = local % 32;
    if (tq0 >= rows) { return; }
    constexpr float LOG2E = 1.4426950408889634f;

    // C-fragment rows lane/4 and lane/4 + 8 of this warp's 16-query tile; a
    // dead row repeats row 0 and is never stored.
    uint32_t qrow[2], qpos[2];
    bool alive[2];
    float row_m[2], row_l[2];
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        qrow[r] = warp * INK2A_WQ + lane / 4 + r * 8;
        alive[r] = tq0 + qrow[r] < rows;
        const uint32_t t = alive[r] ? tq0 + qrow[r] : 0;
        qpos[r] = base + t;
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
    const uint32_t key_lo = seg_lo, key_hi = min(q_hi, seg_hi - 1u);
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

    // A 64-key tile is half of one block: four contiguous segments.
    auto issue_tile = [&](uint32_t stage, uint32_t kt0) {
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
    };
    // One group decodes K (value 0) or V (value 1) of a raw tile into buffer
    // b; keys outside [key_lo, key_hi] are zeros (their bytes may be stale).
    auto decode_half = [&](uint32_t value, uint32_t b, uint32_t stage, uint32_t kt0) {
        const uint8_t *half = raw + stage * INK2A_RAW + value * INK2A_RAW_HALF;
        auto rows_b = s_k(b) + value * INK2A_TK;
        #pragma unroll
        for (uint32_t g = local; g < INK2A_TK * 8; g += INK2A_WARPS * 32) {
            const uint32_t key = g / 8, grp = g % 8, p = kt0 + key;
            uint32_t packed[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (p >= key_lo && p <= key_hi) {
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
        // Old complete tiles have zero bias and no masked scores.
        const bool far = q_lo >= kt0 + INK2A_TK + INK2A_EXTENT - 1 && kt0 >= seg_lo && kt0 + INK2A_TK <= seg_hi;
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
        if (far) {
            #pragma unroll
            for (int nb = 0; nb < INK2A_NB; nb++) {
                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    const float x = ink2a_bf16(ink2a_bf16(s[nb][l]) * INK2A_SCALE);
                    s[nb][l] = x;
                    tile_max[l / 2] = fmaxf(tile_max[l / 2], x);
                }
            }
        } else {
            #pragma unroll
            for (int nb = 0; nb < INK2A_NB; nb++) {
                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    const int r = l / 2;
                    const uint32_t p = kt0 + nb * 8 + (lane % 4) * 2 + (l % 2);
                    float x = -INFINITY;
                    if (alive[r] && p <= qpos[r] && p >= key_lo && p < seg_hi) {
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
        }
        float rescale[2], tile_sum[2] = {0.0f, 0.0f}, mc[2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 1));
            tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 2));
            const float next = fmaxf(row_m[r], tile_max[r]);
            rescale[r] = row_m[r] == next ? 1.0f : exp2f((row_m[r] - next) * LOG2E);
            row_m[r] = next;
            mc[r] = next * LOG2E;
        }
        uint32_t pa[INK2A_KS][4];
        #pragma unroll
        for (int nb = 0; nb < INK2A_NB; nb++) {
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                // exp2(-inf) = 0 for masked keys; the sink keeps m finite.
                const float w = exp2f(fmaf(s[nb][l], LOG2E, -mc[l / 2]));
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

    // Raw tile j lands in stage j % 2; decoded tile j in buffer j % 2. A key
    // segment past this CTA's queries leaves it no tile.
    if (tiles) { issue_tile(0, kt_start); }
    __pipeline_commit();
    if (tiles > 1) { issue_tile(1, kt_start + INK2A_TK); }
    __pipeline_commit();
    __pipeline_wait_prior(1);
    __syncthreads();
    if (tiles) { decode_half(group, 0, 0, kt_start); }
    #pragma unroll 1
    for (uint32_t t = 0; t < tiles; t++) {
        const uint32_t kt0 = kt_start + t * INK2A_TK, b = t % 2;
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

__device__ __forceinline__ float ink2a_e4m3(uint32_t byte) {
    __nv_fp8_e4m3 s;
    s.__x = (__nv_fp8_storage_t)byte;
    return (float)s;
}

#ifdef DS4_CUDA_HAVE_MXF4
// C += A (16 x 64 E2M1, a scale per row and 16 values) . B (64 x 8, a scale per
// column and 16 values): NVFP4's own MMA, four times the BF16 rate.
__device__ __forceinline__ void ink2a_mma4(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
        uint32_t as, uint32_t bs) {
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(as), "r"(bs));
}

/* One 128-wide query by a warp (lane l: dims 4l..4l+3, scale group l / 4)
 * as an NVFP4 row: 64 code bytes, then 8 scales (the K/V encoder's rule:
 * scale = E4M3(max(amax / 6, 2^-9))). v becomes what the codes miss, the
 * next row's input. */
__device__ __forceinline__ void ink2a_q4_encode(float (&v)[4], uint8_t *dst) {
    const uint32_t lane = threadIdx.x & 31;
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const uint8_t code = __nv_cvt_float_to_fp8(fmaxf(__fdiv_rn(amax, 6.0f), 0.001953125f), __NV_SATFINITE, __NV_E4M3);
    const float scale = ink2a_e4m3(code);
    const uint8_t a = __nv_cvt_float2_to_fp4x2(make_float2(__fdiv_rn(v[0], scale), __fdiv_rn(v[1], scale)),
        __NV_E2M1, cudaRoundNearest);
    const uint8_t b = __nv_cvt_float2_to_fp4x2(make_float2(__fdiv_rn(v[2], scale), __fdiv_rn(v[3], scale)),
        __NV_E2M1, cudaRoundNearest);
    dst[lane * 2] = a;
    dst[lane * 2 + 1] = b;
    if (!(lane & 3)) { dst[INK2A_FP4 + lane / 4] = code; }
    const float2 lo = ink2a_e2m1x2(a), hi = ink2a_e2m1x2(b);
    v[0] -= lo.x * scale; v[1] -= lo.y * scale;
    v[2] -= hi.x * scale; v[3] -= hi.y * scale;
}

/* Full-layer prefill, ping-pong form with Q.K on NVFP4 tensor cores. The
 * stored keys are the MMA's B operand as they are (no decode); a query is
 * INK2A_Q4_COMPS NVFP4 rows, each the codes of what the rows before it
 * miss (two rows left 4.6e-2 of the output's RMS as error, three 3.9e-3,
 * beside 1.3e-3 to 1.8e-3 for the BF16 score roundings). Q.K takes six k64
 * MMAs per eight keys where BF16 takes eight k16 ones over decoded keys.
 * P.V stays BF16 over decoded values.
 *
 *   group 0 (head h0)      decode V of keys 0-31 of tile t + 1 ─▶ consume tile t
 *   group 1 (head h0 + 1)  consume tile t ─▶ decode V of keys 32-63 of tile t + 1
 *
 * Raw tiles ride a three-stage ring (tile t's keys are read while tile
 * t + 1's values decode and tile t + 2 loads); a key row is padded to 80
 * bytes so a fragment quad's 32 lanes read distinct banks. Far tiles keep
 * F32 scores; tiles within the bias span take the source's roundings, bias
 * and mask. */
enum { INK2A_Q4_COMPS = 3, INK2A_Q4_STAGES = 3, INK2A_Q4_KROW = 80, INK2A_Q4_KS = INK2A_TK * INK2A_Q4_KROW,
       INK2A_Q4_V = INK2A_Q4_KS + INK2A_TK * 8, INK2A_Q4_STAGE = INK2A_Q4_V + INK2A_RAW_HALF,
       INK2A_Q4_VBUF = INK2A_TK * INK2A_SROW * (int)sizeof(uint16_t),
       INK2A_Q4_SMEM = 2 * INK2A_Q4_VBUF + INK2A_Q4_STAGES * INK2A_Q4_STAGE };
static_assert(INK2A_Q4_COMPS * INK2A_TQ * INK2A_CTA_HEADS * INK2A_FP4_ROW <= 2 * INK2A_Q4_VBUF,
    "query rows borrow the value buffers");

template <int MODE = INK2A_ONE>
__global__ __launch_bounds__(INK2A_THREADS, 1)
void ink2a_prefill_q4(float *out, const float *q, const uint16_t *rel, const float *sinks, const float *gate,
        const uint8_t *pages, uint32_t rows, uint32_t base, float *state = nullptr, uint32_t seg_lo = 0,
        uint32_t seg_hi = UINT32_MAX, uint32_t block0 = 0) {
    extern __shared__ __align__(16) uint8_t ink2a_q4_smem[];
    // Two decoded value tiles, then the ring of raw tiles.
    uint8_t *ring = ink2a_q4_smem + 2 * INK2A_Q4_VBUF;
    auto s_v = [&](uint32_t b) {
        return reinterpret_cast<uint16_t (*)[INK2A_SROW]>(ink2a_q4_smem + b * INK2A_Q4_VBUF);
    };
    const uint32_t tq0 = blockIdx.x * INK2A_TQ, h0 = blockIdx.y * INK2A_CTA_HEADS, kv_head = h0 / INK2A_GROUP;
    const uint32_t group = threadIdx.x / (INK2A_WARPS * 32), h = h0 + group;
    const uint32_t local = threadIdx.x % (INK2A_WARPS * 32), warp = local / 32, lane = local % 32;
    const uint32_t g = lane / 4, t4 = lane % 4;
    if (tq0 >= rows) { return; }
    constexpr float LOG2E = 1.4426950408889634f;

    uint32_t qrow[2], qpos[2];
    bool alive[2];
    float row_m[2], row_l[2];
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        qrow[r] = warp * INK2A_WQ + lane / 4 + r * 8;
        alive[r] = tq0 + qrow[r] < rows;
        const uint32_t t = alive[r] ? tq0 + qrow[r] : 0;
        qpos[r] = base + t;
        row_m[r] = ink2a_bf16(sinks[(uint64_t)t * INK2A_HEADS + h]);
        row_l[r] = 1.0f;
    }

    // This warp's 16 queries as NVFP4 rows (codes, then each residual's) in
    // the value buffers, which hold nothing yet.
    uint8_t *rows4 = ink2a_q4_smem + (group * INK2A_WARPS + warp) * INK2A_Q4_COMPS * INK2A_WQ * INK2A_FP4_ROW;
    #pragma unroll 1
    for (uint32_t i = 0; i < INK2A_WQ; i++) {
        const uint32_t t = tq0 + warp * INK2A_WQ + i < rows ? tq0 + warp * INK2A_WQ + i : 0;
        const float4 x = *reinterpret_cast<const float4 *>(q + ((uint64_t)t * INK2A_HEADS + h) * INK2A_DIM + 4 * lane);
        float v[4] = {x.x, x.y, x.z, x.w};
        #pragma unroll
        for (int c = 0; c < INK2A_Q4_COMPS; c++) { ink2a_q4_encode(v, rows4 + (c * INK2A_WQ + i) * INK2A_FP4_ROW); }
    }
    __syncwarp();
    // Fragments of k64 half k: rows g and g + 8, k values 8t..8t+7 and
    // 32+8t..; a lane carries the scales of row g (t even) or g + 8.
    uint32_t qa[INK2A_Q4_COMPS][2][4], qs[INK2A_Q4_COMPS][2];
    #pragma unroll
    for (int c = 0; c < INK2A_Q4_COMPS; c++) {
        const uint8_t *comp = rows4 + c * INK2A_WQ * INK2A_FP4_ROW;
        const uint32_t *ra = reinterpret_cast<const uint32_t *>(comp + g * INK2A_FP4_ROW);
        const uint32_t *rb = reinterpret_cast<const uint32_t *>(comp + (g + 8) * INK2A_FP4_ROW);
        #pragma unroll
        for (int k = 0; k < 2; k++) {
            qa[c][k][0] = ra[k * 8 + t4];
            qa[c][k][1] = rb[k * 8 + t4];
            qa[c][k][2] = ra[k * 8 + t4 + 4];
            qa[c][k][3] = rb[k * 8 + t4 + 4];
            qs[c][k] = (t4 % 2 ? rb : ra)[INK2A_FP4 / 4 + k];
        }
    }

    const uint32_t q_lo = base + tq0;
    const uint32_t q_hi = base + (tq0 + INK2A_TQ - 1 < rows ? tq0 + INK2A_TQ - 1 : rows - 1);
    const uint32_t key_lo = seg_lo, key_hi = min(q_hi, seg_hi - 1u);
    const uint32_t kt_start = key_lo & ~(uint32_t)(INK2A_TK - 1);
    const uint32_t tiles = key_hi >= key_lo ? (key_hi - kt_start) / INK2A_TK + 1 : 0u;
    float output[INK2A_DIM / 8][4];
    #pragma unroll
    for (int cb = 0; cb < INK2A_DIM / 8; cb++) {
        #pragma unroll
        for (int l = 0; l < 4; l++) { output[cb][l] = 0.0f; }
    }
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

    // A 64-key tile is half of one block: K data (padded rows), K scales,
    // then the V half as it is stored.
    auto issue_tile = [&](uint32_t stage, uint32_t kt0) {
        const uint8_t *page = ink2a_page(pages, kt0, kv_head, block0);
        const uint32_t slot = kt0 % INK2A_KB;
        const uint32_t kdata = INK2A_TK * INK2A_FP4 / 16, kscale = INK2A_TK * 8 / 16;
        uint8_t *dst = ring + stage * INK2A_Q4_STAGE;
        for (uint32_t c = threadIdx.x; c < 2 * (kdata + kscale); c += INK2A_THREADS) {
            if (c < kdata) {
                __pipeline_memcpy_async(dst + c / 4 * INK2A_Q4_KROW + c % 4 * 16,
                    page + (slot + c / 4) * INK2A_FP4 + c % 4 * 16, 16);
            } else if (c < kdata + kscale) {
                __pipeline_memcpy_async(dst + INK2A_Q4_KS + (c - kdata) * 16,
                    page + INK2A_KB * INK2A_FP4 + slot * 8 + (c - kdata) * 16, 16);
            } else {
                const uint32_t at = (c - kdata - kscale) * 16, data = INK2A_TK * INK2A_FP4;
                const uint8_t *half = page + INK2A_KB * INK2A_FP4_ROW;
                __pipeline_memcpy_async(dst + INK2A_Q4_V + at,
                    half + (at < data ? slot * INK2A_FP4 + at : INK2A_KB * INK2A_FP4 + slot * 8 + (at - data)), 16);
            }
        }
    };
    // A group decodes half the keys' values (part 0: keys 0-31) of a raw
    // tile into buffer b; keys outside [key_lo, key_hi] are zeros.
    auto decode_values = [&](uint32_t part, uint32_t b, uint32_t stage, uint32_t kt0) {
        const uint8_t *half = ring + stage * INK2A_Q4_STAGE + INK2A_Q4_V;
        auto rows_b = s_v(b);
        #pragma unroll
        for (uint32_t u = local; u < INK2A_TK / 2 * 8; u += INK2A_WARPS * 32) {
            const uint32_t key = part * (INK2A_TK / 2) + u / 8, grp = u % 8, p = kt0 + key;
            uint32_t packed[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (p >= key_lo && p <= key_hi) {
                const uint2 data = *reinterpret_cast<const uint2 *>(half + key * INK2A_FP4 + grp * 8);
                ink2a_fp4x16(data, half[INK2A_TK * INK2A_FP4 + key * 8 + grp], packed);
            }
            uint16_t *dst = rows_b[key] + grp * 16;
            *reinterpret_cast<uint4 *>(dst) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
            *reinterpret_cast<uint4 *>(dst + 8) = make_uint4(packed[4], packed[5], packed[6], packed[7]);
        }
    };
    auto consume = [&](uint32_t b, uint32_t stage, uint32_t kt0) {
        const uint8_t *kd = ring + stage * INK2A_Q4_STAGE, *ks = kd + INK2A_Q4_KS;
        auto sv = s_v(b);
        const bool far = q_lo >= kt0 + INK2A_TK + INK2A_EXTENT - 1 && kt0 >= seg_lo && kt0 + INK2A_TK <= seg_hi;
        // Eight key blocks' accumulators stay independent between MMAs.
        float s[INK2A_NB][4];
        #pragma unroll
        for (int nb = 0; nb < INK2A_NB; nb++) {
            #pragma unroll
            for (int l = 0; l < 4; l++) { s[nb][l] = 0.0f; }
        }
        #pragma unroll
        for (int k = 0; k < 2; k++) {
            uint32_t b0[INK2A_NB], b1[INK2A_NB], bs[INK2A_NB];
            #pragma unroll
            for (int nb = 0; nb < INK2A_NB; nb++) {
                const uint32_t key = nb * 8 + g;
                const uint32_t *kw = reinterpret_cast<const uint32_t *>(kd + key * INK2A_Q4_KROW);
                b0[nb] = kw[k * 8 + t4];
                b1[nb] = kw[k * 8 + t4 + 4];
                bs[nb] = reinterpret_cast<const uint32_t *>(ks + key * 8)[k];
            }
            #pragma unroll
            for (int c = 0; c < INK2A_Q4_COMPS; c++) {
                #pragma unroll
                for (int nb = 0; nb < INK2A_NB; nb++) { ink2a_mma4(s[nb], qa[c][k], b0[nb], b1[nb], qs[c][k], bs[nb]); }
            }
        }
        float tile_max[2] = {-INFINITY, -INFINITY};
        if (far) {
            #pragma unroll
            for (int nb = 0; nb < INK2A_NB; nb++) {
                #pragma unroll
                for (int l = 0; l < 4; l++) { tile_max[l / 2] = fmaxf(tile_max[l / 2], s[nb][l]); }
            }
            tile_max[0] *= INK2A_SCALE;
            tile_max[1] *= INK2A_SCALE;
        } else {
            #pragma unroll
            for (int nb = 0; nb < INK2A_NB; nb++) {
                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    const int r = l / 2;
                    const uint32_t p = kt0 + nb * 8 + (lane % 4) * 2 + (l % 2);
                    float x = -INFINITY;
                    if (alive[r] && p <= qpos[r] && p >= key_lo && p < seg_hi) {
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
        }
        float rescale[2], tile_sum[2] = {0.0f, 0.0f}, mc[2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 1));
            tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 2));
            const float next = fmaxf(row_m[r], tile_max[r]);
            rescale[r] = row_m[r] == next ? 1.0f : exp2f((row_m[r] - next) * LOG2E);
            row_m[r] = next;
            mc[r] = next * LOG2E;
        }
        uint32_t pa[INK2A_KS][4];
        const float unit = far ? INK2A_SCALE * LOG2E : LOG2E;
        #pragma unroll
        for (int nb = 0; nb < INK2A_NB; nb++) {
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const float w = exp2f(fmaf(s[nb][l], unit, -mc[l / 2]));
                s[nb][l] = w;
                tile_sum[l / 2] += w;
            }
        }
        #pragma unroll
        for (int ks4 = 0; ks4 < INK2A_KS; ks4++) {
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const float *src = s[2 * ks4 + l / 2];
                pa[ks4][l] = ink2a_pack(src[(l % 2) * 2], src[(l % 2) * 2 + 1]);
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
        for (int ks4 = 0; ks4 < INK2A_KS; ks4++) {
            #pragma unroll
            for (int cb = 0; cb < INK2A_DIM / 8; cb += 2) {
                uint32_t b0, b1, b2, b3;
                ink2a_ldsm4t(b0, b1, b2, b3, &sv[ks4 * 16 + v_row][cb * 8 + v_col]);
                ink2a_mma(output[cb], pa[ks4], b0, b1);
                ink2a_mma(output[cb + 1], pa[ks4], b2, b3);
            }
        }
    };

    // Raw tile j rides stage j % 3; its decoded values buffer j % 2.
    if (tiles) { issue_tile(0, kt_start); }
    __pipeline_commit();
    if (tiles > 1) { issue_tile(1, kt_start + INK2A_TK); }
    __pipeline_commit();
    __pipeline_wait_prior(1);
    __syncthreads();
    if (tiles) { decode_values(group, 0, 0, kt_start); }
    #pragma unroll 1
    for (uint32_t t = 0; t < tiles; t++) {
        const uint32_t kt0 = kt_start + t * INK2A_TK, b = t % 2, stage = t % INK2A_Q4_STAGES;
        const uint32_t next = (t + 1) % INK2A_Q4_STAGES;
        const bool more = t + 1 < tiles;
        // Raw tile t + 1 has landed; tile t's values are decoded; the stage
        // of tile t - 1 is free.
        __pipeline_wait_prior(0);
        __syncthreads();
        if (t + 2 < tiles) { issue_tile((t + 2) % INK2A_Q4_STAGES, kt0 + 2 * INK2A_TK); }
        __pipeline_commit();
        if (group == 0) {
            if (more) { decode_values(0, b ^ 1, next, kt0 + INK2A_TK); }
            consume(b, stage, kt0);
        } else {
            consume(b, stage, kt0);
            if (more) { decode_values(1, b ^ 1, next, kt0 + INK2A_TK); }
        }
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
                const float2 gt = *reinterpret_cast<const float2 *>(gate + at);
                const float o0 = ink2a_bf16(__fdiv_rn(output[cb][2 * r], row_l[r]));
                const float o1 = ink2a_bf16(__fdiv_rn(output[cb][2 * r + 1], row_l[r]));
                *reinterpret_cast<float2 *>(out + at) = make_float2(
                    ink2a_bf16(o0 * ink2a_sigmoid(ink2a_bf16(gt.x))), ink2a_bf16(o1 * ink2a_sigmoid(ink2a_bf16(gt.y))));
            }
        }
    }
}
#endif

/* Decode partials: CTA (split, KV head, row); warp w scores head 6h + w over
 * the split's keys, a lane per key of each 32-key tile staged in shared
 * memory (lane j reads K row j from word (i + j) % 64 on, so the 32 rows'
 * words fall in distinct banks). partial = (m, l, acc[128]). */
template <int FMT, bool STAGED = false>
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
        // Keep the bias load ahead of the tile wait and QK work.
        uint16_t prefetched_bias = 0;
        if constexpr (STAGED) {
            const uint32_t p = t0 + lane;
            if (p < k1 && pos - p < INK2A_EXTENT) {
                asm volatile("ld.global.u16 %0, [%1];" : "=h"(prefetched_bias) : "l"(bias + pos - p));
            }
        }
        if constexpr (FMT == INK2A_FMT_NVFP4 && STAGED) {
            // Issue independent raw loads before any conversion consumes them.
            constexpr int batches = (INK2A_DTK * 2 * 8 + INK2A_DTHREADS - 1) / INK2A_DTHREADS;
            uint2 data[batches];
            uint32_t scales[batches];
            #pragma unroll
            for (int batch = 0; batch < batches; batch++) {
                const uint32_t i = threadIdx.x + batch * INK2A_DTHREADS;
                if (i >= INK2A_DTK * 2 * 8) { continue; }
                const uint32_t j = i / 16, value = (i / 8) % 2, grp = i % 8;
                const uint32_t p = min(t0 + j, k1 - 1), slot = p % INK2A_KB;
                const uint8_t *half = ink2a_page(pages, p, kv, block0) + value * INK2A_KB * INK2A_FP4_ROW;
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
        } else if constexpr (FMT == INK2A_FMT_NVFP4) {
            // 32 keys x (K, V) x 8 groups of 16 values, decoded to BF16 pairs.
            for (uint32_t i = threadIdx.x; i < INK2A_DTK * 2 * 8; i += INK2A_DTHREADS) {
                const uint32_t j = i / 16, value = (i / 8) % 2, grp = i % 8;
                const uint32_t p = min(t0 + j, k1 - 1), slot = p % INK2A_KB;
                const uint8_t *half = ink2a_page(pages, p, kv, block0) + value * INK2A_KB * INK2A_FP4_ROW;
                const uint2 data = *reinterpret_cast<const uint2 *>(half + slot * INK2A_FP4 + grp * 8);
                uint32_t *dst = (value ? sv[j] : sk[j]) + grp * 8;
                ink2a_fp4x16(data, half[INK2A_KB * INK2A_FP4 + slot * 8 + grp], dst);
            }
        } else if constexpr (STAGED) {
            for (uint32_t i = threadIdx.x; i < INK2A_DTK * INK2A_DIM / 8; i += INK2A_DTHREADS) {
                const uint32_t j = i / (INK2A_DIM / 8), c = i % (INK2A_DIM / 8);
                const uint32_t p = min(t0 + j, k1 - 1);
                const uint16_t *src = ink2a_row(cache, p, cap, window) + kv * INK2A_DIM;
                __pipeline_memcpy_async(&sk[j][4 * c], src + 8 * c, 16);
                __pipeline_memcpy_async(&sv[j][4 * c], src + INK2A_KV + 8 * c, 16);
            }
            __pipeline_commit();
            __pipeline_wait_prior(0);
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
            const float relative = STAGED ? __uint_as_float((uint32_t)prefetched_bias << 16)
                : d < INK2A_EXTENT ? __uint_as_float((uint32_t)bias[d] << 16) : 0.0f;
            score = ink2a_score(dot, relative);
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
    *reinterpret_cast<float4 *>(dst + INK2A_ACC + 4 * lane) = make_float4(acc[0], acc[1], acc[2], acc[3]);
}

// The two E2M1 codes of `byte` (low nibble first) times `scale`, as BF16.
__device__ __forceinline__ uint32_t ink2a_pair(uint32_t byte, float scale) {
    const float2 v = ink2a_e2m1x2(byte);
    return ink2a_pack(v.x * scale, v.y * scale);
}

/* One decode row over NVFP4 pages on tensor cores (K2-Horizon's k2a_split
 * with Inkling 2's score roundings, relative bias and key segments): the six
 * query heads of a KV head are the m16 rows of a BF16 MMA (rows 6-15 zero),
 * so one decoded K fragment scores all six. CTA (slice, KV head, row): the
 * slice's 64-key tiles (9 KB raw: K data, K scales, V data, V scales) stream
 * through a three-stage cp.async ring; warp w takes keys 16w..16w+15 of each
 * tile with its own online softmax and writes partial 4 * slice + w.
 * ink2a_decode's walk (a lane per key, Q from shared memory) reads 135 GB/s.
 *
 * Fragment order: lane t of a score MMA holds dims 32t..32t+31 over the
 * eight k steps (step c: dims 32t + 4c..+3), so it reads 16 contiguous code
 * bytes of its key and Q follows the same order; for P.V lane g holds output
 * dims 16g..16g+15 (n block j: dim 16g + j) of its four keys. E2M1 x E4M3
 * is a BF16, so the decoded operands are exact; P rounds to BF16 as in
 * prefill and verify. */
enum { INK2A_SWARPS = 4, INK2A_SSTAGES = 3, INK2A_SSUB = INK2A_TK / INK2A_SWARPS,
       INK2A_SSLICES = INK2A_SPLITS / INK2A_SWARPS };

__global__ __launch_bounds__(INK2A_SWARPS * 32)
void ink2a_split(float *partials, const float *q, const uint16_t *rel, const uint8_t *pages, uint32_t base,
        uint32_t slices, uint32_t seg_lo = 0, uint32_t seg_hi = UINT32_MAX, uint32_t block0 = 0) {
    __shared__ __align__(16) uint8_t ring[INK2A_SSTAGES][INK2A_RAW];
    const uint32_t slice = blockIdx.x, kv = blockIdx.y, row = blockIdx.z;
    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const uint32_t pos = base + row, end = min(pos + 1, seg_hi);
    // The segment's 64-key tiles (seg_lo is a block boundary), this slice's share.
    const uint32_t tile_lo = seg_lo / INK2A_TK, tiles = end > seg_lo ? (end - 1) / INK2A_TK + 1 - tile_lo : 0u;
    const uint32_t per = (tiles + slices - 1) / slices;
    const uint32_t first = min(slice * per, tiles), n = min(first + per, tiles) - first;
    const uint32_t head = kv * INK2A_GROUP + min(g, (uint32_t)INK2A_GROUP - 1);

    // Q fragments of k step c: dims 32t + 4c..+3 of head 6 kv + g (zero past the group).
    uint32_t qa[INK2A_DIM / 16][2];
    const float4 *qrow = reinterpret_cast<const float4 *>(q + ((uint64_t)row * INK2A_HEADS + head) * INK2A_DIM + 32 * t);
    #pragma unroll
    for (int c = 0; c < INK2A_DIM / 16; c++) {
        const float4 x = g < INK2A_GROUP ? qrow[c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        qa[c][0] = ink2a_pack(x.x, x.y);
        qa[c][1] = ink2a_pack(x.z, x.w);
    }
    const uint16_t *bias = rel + ((uint64_t)row * INK2A_HEADS + head) * INK2A_EXTENT;

    // Tile `tile` (64 keys from position 64 tile) into a ring stage.
    auto issue = [&](uint32_t stage, uint32_t tile) {
        const uint32_t kt0 = tile * INK2A_TK, slot = kt0 % INK2A_KB, data = INK2A_TK * INK2A_FP4;
        const uint8_t *page = ink2a_page(pages, kt0, kv, block0);
        for (uint32_t c = threadIdx.x; c < INK2A_RAW / 16; c += INK2A_SWARPS * 32) {
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
    for (uint32_t s = 0; s + 1 < INK2A_SSTAGES; s++) {
        if (s < n) { issue(s, tile_lo + first + s); }
        __pipeline_commit();
    }
    #pragma unroll 1
    for (uint32_t i = 0; i < n; i++) {
        // Tile i has landed; every warp is done with the stage refilled here.
        __pipeline_wait_prior(INK2A_SSTAGES - 2);
        __syncthreads();
        if (i + INK2A_SSTAGES - 1 < n) {
            issue((i + INK2A_SSTAGES - 1) % INK2A_SSTAGES, tile_lo + first + i + INK2A_SSTAGES - 1);
        }
        __pipeline_commit();
        const uint8_t *st = ring[i % INK2A_SSTAGES];
        const uint32_t tile0 = (tile_lo + first + i) * INK2A_TK, sub = INK2A_SSUB * warp;

        // Scores of keys sub + 8 jj + g (B column g of n block jj).
        float s[2][4];
        #pragma unroll
        for (int jj = 0; jj < 2; jj++) {
            const uint32_t key = sub + 8 * jj + g;
            const uint4 d = *reinterpret_cast<const uint4 *>(st + key * INK2A_FP4 + 16 * t);
            const uint32_t sc = *reinterpret_cast<const uint16_t *>(st + INK2A_TK * INK2A_FP4 + key * 8 + 2 * t);
            const float lo = ink2a_e4m3(sc & 0xffu), hi = ink2a_e4m3(sc >> 8);
            const uint32_t w[4] = {d.x, d.y, d.z, d.w};
            s[jj][0] = s[jj][1] = s[jj][2] = s[jj][3] = 0.0f;
            #pragma unroll
            for (int c = 0; c < INK2A_DIM / 16; c++) {
                const uint32_t bytes = w[c / 2] >> (16 * (c % 2));
                const float scale = c < 4 ? lo : hi;
                const uint32_t a[4] = {qa[c][0], 0u, qa[c][1], 0u};
                ink2a_mma(s[jj], a, ink2a_pair(bytes & 0xffu, scale), ink2a_pair((bytes >> 8) & 0xffu, scale));
            }
        }

        // Online softmax of row g over this lane's keys sub + 8 jj + 2t + e;
        // only the tiles that reach the bias span load a bias.
        const bool near = tile0 + INK2A_TK + INK2A_EXTENT > pos + 1;
        float top = -INFINITY;
        #pragma unroll
        for (int jj = 0; jj < 2; jj++) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                const uint32_t p = tile0 + sub + 8 * jj + 2 * t + e;
                const bool in = p < end;
                float relative = 0.0f;
                if (near && in && pos - p < INK2A_EXTENT) { relative = __uint_as_float((uint32_t)bias[pos - p] << 16); }
                s[jj][e] = in ? ink2a_score(s[jj][e], relative) : -INFINITY;
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
            vs[k] = tile0 + key < end ? ink2a_e4m3(vsc[key * 8 + g]) : 0.0f;
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
    // stored as eight 16-byte pieces.
    if (g >= INK2A_GROUP) { return; }
    float *dst = partials + (((uint64_t)row * INK2A_HEADS + kv * INK2A_GROUP + g) * INK2A_SPLITS +
        slice * INK2A_SWARPS + warp) * INK2A_PARTIAL;
    if (t == 0) {
        dst[0] = m;
        dst[1] = l;
    }
    float4 *acc = reinterpret_cast<float4 *>(dst + INK2A_ACC + 32 * t);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        acc[i] = make_float4(o[4 * i][0], o[4 * i + 1][0], o[4 * i + 2][0], o[4 * i + 3][0]);
        acc[4 + i] = make_float4(o[4 * i][1], o[4 * i + 1][1], o[4 * i + 2][1], o[4 * i + 3][1]);
    }
}

/* Verify rows (2..8) on BF16 tensor cores: a CTA owns one (split, KV head)
 * and the rows' queries of that head group as m16 tiles (slot i = row i / 6,
 * head 6 kv + i % 6; four warps cover 64 slots, 48 used at eight rows). The
 * split's 64-key tiles stream through the prefill kernel's ring once for
 * every row. Writes the decode kernel's partials (m, l, acc without the
 * sink), P rounded to BF16 as in prefill. */
enum { INK2A_VTHREADS = INK2A_WARPS * 32, INK2A_VERIFY_SPLITS = 6 };

template <int FMT>
__global__ __launch_bounds__(INK2A_VTHREADS)
void ink2a_verify(float *partials, const float *q, const uint16_t *rel, const void *cache_raw, uint32_t cap,
        uint32_t window, uint32_t base, uint32_t rows, uint32_t span, uint32_t seg_lo, uint32_t seg_hi,
        uint32_t block0) {
    extern __shared__ __align__(16) uint16_t ink2a_smem[];
    const uint16_t *cache = (const uint16_t *)cache_raw;
    const uint8_t *pages = (const uint8_t *)cache_raw;
    uint8_t *raw = (uint8_t *)(ink2a_smem + (FMT == INK2A_FMT_NVFP4 ? INK2A_STAGE_ELEMS : 0));
    auto s_k = [&](int stage) {
        return reinterpret_cast<uint16_t (*)[INK2A_SROW]>(ink2a_smem +
            (FMT == INK2A_FMT_NVFP4 ? 0 : (size_t)stage * INK2A_STAGE_ELEMS));
    };
    auto s_v = [&](int stage) { return s_k(stage) + INK2A_TK; };
    const uint32_t split = blockIdx.x, kv_head = blockIdx.y;
    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    // The rows' common range, this split's share of it.
    const uint32_t lo = max(window && base >= window ? base - (window - 1) : 0u, seg_lo);
    const uint32_t hi = min(base + rows, seg_hi);
    const uint32_t key_lo = lo + split * span, key_end = min(hi, key_lo + span);

    uint32_t qrow[2], qh[2], qpos[2], qfirst[2];
    bool alive[2];
    float row_m[2], row_l[2];
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        const uint32_t slot = warp * INK2A_WQ + lane / 4 + r * 8;
        qrow[r] = slot / INK2A_GROUP;
        qh[r] = kv_head * INK2A_GROUP + slot % INK2A_GROUP;
        alive[r] = qrow[r] < rows;
        qpos[r] = base + qrow[r];
        qfirst[r] = max(window && qpos[r] >= window ? qpos[r] - (window - 1) : 0u, key_lo);
        row_m[r] = -INFINITY;
        row_l[r] = 0.0f;
    }
    uint32_t qa[INK2A_DIM / 16][4];
    #pragma unroll
    for (int kc = 0; kc < INK2A_DIM / 16; kc++) {
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            const int r = l % 2, j = (l / 2) * 4 + (int)(lane % 4);
            float2 xy = make_float2(0.0f, 0.0f);
            if (alive[r]) {
                xy = *reinterpret_cast<const float2 *>(
                    q + ((uint64_t)qrow[r] * INK2A_HEADS + qh[r]) * INK2A_DIM + kc * 16 + 2 * j);
            }
            qa[kc][l] = ink2a_pack(xy.x, xy.y);
        }
    }
    float output[INK2A_DIM / 8][4];
    #pragma unroll
    for (int cb = 0; cb < INK2A_DIM / 8; cb++) {
        #pragma unroll
        for (int l = 0; l < 4; l++) { output[cb][l] = 0.0f; }
    }
    const uint32_t key_hi = key_end > key_lo ? key_end - 1 : key_lo;
    const uint32_t kt_start = key_lo & ~(uint32_t)(INK2A_TK - 1);
    const uint32_t tiles = key_end > key_lo ? (key_hi - kt_start) / INK2A_TK + 1 : 0u;

    // Rows outside [key_lo, key_hi] repeat key_lo so masked keys stay finite.
    auto issue_tile = [&](int stage, uint32_t kt0) {
        if constexpr (FMT == INK2A_FMT_NVFP4) {
            const uint8_t *page = ink2a_page(pages, kt0, kv_head, block0);
            const uint32_t slot = kt0 % INK2A_KB;
            uint8_t *dst = raw + stage * INK2A_RAW;
            for (uint32_t c = threadIdx.x; c < INK2A_RAW / 16; c += INK2A_VTHREADS) {
                const uint32_t at = c * 16, half = at / INK2A_RAW_HALF, in = at % INK2A_RAW_HALF;
                const uint32_t data = INK2A_TK * INK2A_FP4;
                const uint8_t *src = page + half * INK2A_KB * INK2A_FP4_ROW +
                    (in < data ? slot * INK2A_FP4 + in : INK2A_KB * INK2A_FP4 + slot * 8 + (in - data));
                __pipeline_memcpy_async(dst + at, src, 16);
            }
        } else {
            for (uint32_t idx = threadIdx.x; idx < INK2A_TILE_CHUNKS; idx += INK2A_VTHREADS) {
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
    auto decode_tile = [&](int stage, uint32_t kt0) {
        const uint8_t *src = raw + stage * INK2A_RAW;
        for (uint32_t g = threadIdx.x; g < 2 * INK2A_TK * 8; g += INK2A_VTHREADS) {
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
    // Two or three rows fill 12 or 18 of the 64 query slots: a warp whose
    // slots are all past the rows stages tiles but runs no MMA.
    const bool live = warp * INK2A_WQ < rows * INK2A_GROUP;
    auto consume = [&](int stage, uint32_t kt0) {
        if (!live) { return; }
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
                if (alive[r] && p <= qpos[r] && p >= qfirst[r] && p < key_end) {
                    const uint32_t d = qpos[r] - p;
                    const float bias = d < INK2A_EXTENT
                        ? __uint_as_float((uint32_t)rel[((uint64_t)qrow[r] * INK2A_HEADS + qh[r]) * INK2A_EXTENT + d] << 16)
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
            // No sink here: a row with no live key yet keeps m = -inf.
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
                ink2a_ldsm4t(b0, b1, b2, b3, &sv[ks * 16 + v_row][cb * 8 + v_col]);
                ink2a_mma(output[cb], pa[ks], b0, b1);
                ink2a_mma(output[cb + 1], pa[ks], b2, b3);
            }
        }
    };

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

    #pragma unroll
    for (int r = 0; r < 2; r++) {
        if (!alive[r]) { continue; }
        float *dst = partials + (((uint64_t)qrow[r] * INK2A_HEADS + qh[r]) * INK2A_SPLITS + split) * INK2A_PARTIAL;
        if (lane % 4 == 0) {
            dst[0] = row_m[r];
            dst[1] = row_l[r];
        }
        #pragma unroll
        for (int cb = 0; cb < INK2A_DIM / 8; cb++) {
            *reinterpret_cast<float2 *>(dst + INK2A_ACC + cb * 8 + (lane % 4) * 2) =
                make_float2(output[cb][2 * r], output[cb][2 * r + 1]);
        }
    }
}

/* Splits' maxima and weights for a merge: thread s loads split s's (m, l),
 * so the loads arrive together (a serial walk waited ~0.5 us per split);
 * `weight[s]` = exp(m_s - top) with top the largest of the splits' maxima and
 * `m0`. Returns the sum of l_s weight[s]; `top` comes back through `m0`. */
__device__ __forceinline__ float ink2a_split_weights(const float *src, uint32_t splits, float *weight, float *red,
        float &m0) {
    static_assert(INK2A_SPLITS <= INK2A_DIM, "a thread per split");
    const uint32_t d = threadIdx.x, lane = d % 32, warp = d / 32;
    const float ms = d < splits ? src[d * INK2A_PARTIAL] : -INFINITY;
    const float ls = d < splits ? src[d * INK2A_PARTIAL + 1] : 0.0f;
    float top = ms;
    for (uint32_t o = 16; o; o >>= 1) { top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, o)); }
    if (!lane) { red[warp] = top; }
    __syncthreads();
    top = fmaxf(m0, fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3])));
    const float e = ms == -INFINITY ? 0.0f : __expf(ms - top);
    if (d < INK2A_SPLITS) { weight[d] = e; }
    float sum = ls * e;
    for (uint32_t o = 16; o; o >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, o); }
    __syncthreads();
    if (!lane) { red[warp] = sum; }
    __syncthreads();
    m0 = top;
    return (red[0] + red[1]) + (red[2] + red[3]);
}

// out = BF16(BF16(sum of partials over (sink + sum of l)) * sigmoid(gate)).
__global__ void __launch_bounds__(INK2A_DIM) ink2a_merge(float *out, const float *partials, const float *sinks,
        const float *gate, uint32_t splits) {
    __shared__ float weight[INK2A_SPLITS], red[INK2A_DIM / 32];
    const uint32_t rh = blockIdx.x, d = threadIdx.x;
    const float *src = partials + (uint64_t)rh * INK2A_SPLITS * INK2A_PARTIAL;
    const float sink = ink2a_bf16(sinks[rh]);
    float top = sink;
    const float sum = ink2a_split_weights(src, splits, weight, red, top);
    const float total = __expf(sink - top) + sum;
    float value = 0.0f;
    // Each thread sums its dim over the splits with independent loads.
    #pragma unroll 16
    for (uint32_t s = 0; s < splits; s++) { value = fmaf(src[s * INK2A_PARTIAL + INK2A_ACC + d], weight[s], value); }
    const uint64_t at = (uint64_t)rh * INK2A_DIM + d;
    out[at] = ink2a_bf16(ink2a_bf16(__fdiv_rn(value, total)) * ink2a_sigmoid(ink2a_bf16(gate[at])));
}

/* Decode key segments: fold a segment's split partials into the state (FIRST
 * starts it with the sink), then finish: out = BF16(BF16(acc / l) *
 * sigmoid(gate)). */
__global__ void __launch_bounds__(INK2A_DIM) ink2a_fold(float *state, const float *partials, const float *sinks,
        uint32_t splits, int first) {
    __shared__ float weight[INK2A_SPLITS], red[INK2A_DIM / 32];
    const uint32_t rh = blockIdx.x, d = threadIdx.x;
    const float *src = partials + (uint64_t)rh * INK2A_SPLITS * INK2A_PARTIAL;
    float *st = state + (uint64_t)rh * INK2A_STATE;
    const float m0 = first ? ink2a_bf16(sinks[rh]) : st[0], l0 = first ? 1.0f : st[1];
    const float a0 = first ? 0.0f : st[2 + d];
    float top = m0;
    const float sum = ink2a_split_weights(src, splits, weight, red, top);
    const float e0 = __expf(m0 - top);
    float value = a0 * e0;
    #pragma unroll 16
    for (uint32_t s = 0; s < splits; s++) { value = fmaf(src[s * INK2A_PARTIAL + INK2A_ACC + d], weight[s], value); }
    // Every thread has read the state it replaces.
    __syncthreads();
    if (d == 0) {
        st[0] = top;
        st[1] = l0 * e0 + sum;
    }
    st[2 + d] = value;
}

__global__ void ink2a_finish(float *out, const float *state, const float *gate) {
    const uint32_t rh = blockIdx.x, d = threadIdx.x;
    const float *st = state + (uint64_t)rh * INK2A_STATE;
    const uint64_t at = (uint64_t)rh * INK2A_DIM + d;
    out[at] = ink2a_bf16(ink2a_bf16(__fdiv_rn(st[2 + d], st[1])) * ink2a_sigmoid(ink2a_bf16(gate[at])));
}

static uint32_t ink2a_splits(uint32_t keys) {
    const uint32_t want = (keys + INK2A_MIN_SPLIT - 1) / INK2A_MIN_SPLIT;
    return want < 1 ? 1 : want > INK2A_SPLITS ? INK2A_SPLITS : want;
}

#endif
