#ifndef DS4_GEMMA4_ATTN_MMA_CUH
#define DS4_GEMMA4_ATTN_MMA_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

/* Gemma-4 tensor-core prefill attention (Phase 8).
 *
 * The scalar tile kernel (cuda/gemma4_attn_tile.cuh) is 72 percent of the
 * 13,447-token prefill on the GB10: 22.95 s of 43.2 s in the full-layer
 * geometry and 8.25 s in the sliding one, at about 171 G MAC/s. This header
 * is the tensor-core form the Phase 8 review specifies
 * (Agy_phase8_review.md, sections 2 and 3): one CTA per (KV head, row tile),
 * 8 warps, 32-key tiles, Q kept in registers, online softmax in f32, P
 * rounded to bf16 for the P*V mma. The two specializations share the
 * per-warp inner engine:
 *
 *   sliding <256, 2>  8 tokens x 2 query heads per warp, full D = 256,
 *                     RTILE = 64 tokens per CTA, 32 KiB shared (K + V).
 *   full    <512, 8>  2 tokens x 8 query heads per warp, each of two warps
 *                     takes a 256-dim half, RTILE = 8 tokens per CTA,
 *                     48 KiB shared (32 KiB K/V + 16 KiB score exchange).
 *
 * The arithmetic is not the scalar kernel's: the QK dot is an m16n8k16
 * tensor-core tree, the softmax rescales once per 32-key tile, and P is
 * rounded to bf16 before P*V. The gate is therefore a parity gate, not cmp
 * (Agy_phase8_review.md section 4: max_abs <= 2.5e-3, relL2 <= 1.5e-3 on
 * synthetic shapes; greedy-token parity and frontier KL on the model).
 * Q is already rounded to bf16 by g4_rms_rope_kernel, so converting the f32
 * query to bf16 here is 0 ULP.
 *
 * Chunk invariance: the key loop starts at (begin_all / 32) * 32, so every
 * tile boundary depends only on absolute key indices; a fully masked tile
 * leaves m, l and o untouched. A token therefore sees the same tile
 * sequence at every chunk size and row tile. */

enum { G4_MMA_TK = 32u, G4_MMA_THREADS = 256u };

__device__ __forceinline__ uint32_t g4_mma_swz(uint32_t key, uint32_t chunk, uint32_t dim) {
    /* 16-byte chunks: dim/8 per key row, XOR the low three chunk bits with
     * the key index so the eight rows one ldmatrix reads land on eight
     * distinct bank groups (Agy_phase8_review.md section 2.4). */
    return key * dim + ((chunk ^ (key & 7u)) << 3);
}

__device__ __forceinline__ void g4_mma_cp16(void *dst, const void *src, uint32_t bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
        "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void g4_mma_ldsm4(uint32_t *r, const void *p, bool trans) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    if (trans) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    }
}

__device__ __forceinline__ void g4_mma_mma(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t g4_mma_pack(float lo, float hi) {
    const __nv_bfloat162_raw v = __floats2bfloat162_rn(lo, hi);
    return (uint32_t)v.x | ((uint32_t)v.y << 16);
}

/* The same two rows for the FP8 lane's f16 tiles. A query value is already
 * bf16-rounded by g4_rms_rope_kernel and every bf16 with a half-range
 * exponent is exactly representable in f16, so this conversion is 0 ULP;
 * rounding a softmax probability to f16 keeps 11 significand bits against
 * bf16's 8, so the P V product here is strictly closer to the f32 reference
 * than the bf16 path's. */
__device__ __forceinline__ uint32_t g4_mma_pack_h(float lo, float hi) {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&v);
}

__device__ __forceinline__ void g4_mma_mma_h(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/* The staged-tile element: bf16 for the bf16 rows, f16 for the E4M3 rows.
 * ldmatrix and the swizzle are format-agnostic (both types are 2 bytes), so
 * only the mma instruction and the staging decode differ. */
template <bool FP8> struct g4_mma_elem { using type = __nv_bfloat16; };
template <> struct g4_mma_elem<true> { using type = __half; };

template <bool FP8>
__device__ __forceinline__ uint32_t g4_mma_pack_t(float lo, float hi) {
    if constexpr (FP8) { return g4_mma_pack_h(lo, hi); }
    return g4_mma_pack(lo, hi);
}

template <bool FP8>
__device__ __forceinline__ void g4_mma_mma_t(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    if constexpr (FP8) { g4_mma_mma_h(d, a, b0, b1); }
    else { g4_mma_mma(d, a, b0, b1); }
}

/* Rounding a softmax probability into the staged element type, and reading
 * it back. The bf16 arm is the split kernel's original pair
 * (__float2bfloat16_rn / __bfloat162float). */
template <bool FP8>
__device__ __forceinline__ typename g4_mma_elem<FP8>::type g4_mma_store_t(float x) {
    if constexpr (FP8) { return __float2half_rn(x); }
    else { return __float2bfloat16_rn(x); }
}

template <bool FP8>
__device__ __forceinline__ float g4_mma_load_t(typename g4_mma_elem<FP8>::type x) {
    if constexpr (FP8) { return __half2float(x); }
    else { return __bfloat162float(x); }
}

/* Eight E4M3 codes (one uint2) times one f16 row scale, as eight f16 values
 * in a uint4. This is the decode stage of the Solar warp-specialized FP8
 * kernel, cuda/mmq/ds4_fattn.cu:1557 (fp8_ws_scale4): the exact hardware
 * e4m3x2 conversion, the product in fp32, then rn rounding to f16, which is
 * the correctly rounded half of the exact product. The scalar kernel this
 * lane used before (gemma4_attn_fp8.cuh, g4_fp8_e4m3) decodes through the
 * same conversion and keeps the product in fp32, so the two agree to the
 * f16 rounding of K and V; the parity gate below measures that. */
__device__ __forceinline__ uint4 g4_fp8_decode8(uint2 codes, __half scale) {
    const float s = __half2float(scale);
    const float2 f0 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(codes.x & 0xffffu), __NV_E4M3)));
    const float2 f1 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(codes.x >> 16u), __NV_E4M3)));
    const float2 f2 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(codes.y & 0xffffu), __NV_E4M3)));
    const float2 f3 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(codes.y >> 16u), __NV_E4M3)));
    const __half2 h0 = __floats2half2_rn(f0.x * s, f0.y * s);
    const __half2 h1 = __floats2half2_rn(f1.x * s, f1.y * s);
    const __half2 h2 = __floats2half2_rn(f2.x * s, f2.y * s);
    const __half2 h3 = __floats2half2_rn(f3.x * s, f3.y * s);
    return make_uint4(*reinterpret_cast<const uint32_t *>(&h0),
                      *reinterpret_cast<const uint32_t *>(&h1),
                      *reinterpret_cast<const uint32_t *>(&h2),
                      *reinterpret_cast<const uint32_t *>(&h3));
}

/* One 32-key tile of the sliding geometry: D = 256, group = 2, one warp per
 * 8 tokens x 2 query heads. `warp` picks the token group, `kv_head` the
 * shared K/V head. */
__global__ void __launch_bounds__(G4_MMA_THREADS, 1) g4_attn_mma_slide_kernel(
        float *out, const float *query, const __nv_bfloat16 *key,
        const __nv_bfloat16 *value, uint32_t rows, uint32_t first,
        uint32_t heads, uint32_t kv_heads, uint32_t window, uint32_t ring,
        uint32_t keys, uint64_t kv_stride) {
    constexpr uint32_t D = 256u, GROUP = 2u, RTILE = 64u, TK = G4_MMA_TK;
    extern __shared__ __align__(16) uint8_t smem[];
    __nv_bfloat16 *ks = (__nv_bfloat16 *)smem;
    __nv_bfloat16 *vs = ks + TK * D;

    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, q = lane & 3u;
    const uint32_t kv_head = blockIdx.x;
    const uint32_t row0 = blockIdx.y * RTILE;
    const uint32_t row_end = row0 + RTILE < rows ? row0 + RTILE : rows;
    const uint32_t pos_first = first + row0;
    const uint32_t pos_last = first + row_end - 1u;
    const uint32_t begin_all = window && pos_first + 1u > window ? pos_first + 1u - window : 0u;
    const uint32_t last_all = keys && keys <= pos_last ? keys - 1u : pos_last;
    if (keys && begin_all >= keys) { return; }
    const uint32_t base_start = (begin_all / TK) * TK;

    /* Q fragments: 16 M rows (8 tokens x 2 heads) x 16 k16 chunks of D.
     * The A fragment order is [row g, row g+8] x [col, col+8], the same
     * order glm_vq_flash.cuh:100-107 uses. */
    uint32_t qa[16][4];
    #pragma unroll
    for (uint32_t k = 0; k < 16u; k++) {
        const uint32_t col = k * 16u + 2u * q;
        #pragma unroll
        for (uint32_t m = 0; m < 2u; m++) {
            const uint32_t row_g = g + 8u * m;
            const uint32_t token = row0 + warp * 8u + (row_g >> 1);
            const bool valid = token < rows;
            const uint32_t head = kv_head * GROUP + (row_g & 1u);
            const float *qr = query + ((uint64_t)token * heads + head) * D;
            const float lo = valid ? qr[col] : 0.0f;
            const float hi = valid ? qr[col + 1u] : 0.0f;
            const float lo8 = valid ? qr[col + 8u] : 0.0f;
            const float hi8 = valid ? qr[col + 9u] : 0.0f;
            /* m = 0 fills the g row pair, m = 1 the g+8 pair. */
            if (m == 0u) {
                qa[k][0] = g4_mma_pack(lo, hi);
                qa[k][2] = g4_mma_pack(lo8, hi8);
            } else {
                qa[k][1] = g4_mma_pack(lo, hi);
                qa[k][3] = g4_mma_pack(lo8, hi8);
            }
        }
    }

    float o[32][4] = {};
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};

    for (uint32_t base = base_start; base <= last_all; base += TK) {
        /* Stage this KV head's 32 key rows. A key outside the CTA's visible
         * range is hardware-zeroed (src_size 0); its score is masked below. */
        for (uint32_t t = warp * 4u; t < warp * 4u + 4u; t++) {
            const uint32_t k_abs = base + t;
            const bool valid = k_abs >= begin_all && k_abs <= last_all;
            const uint32_t slot = valid ? (ring ? (k_abs % ring) : k_abs) : 0u;
            const __nv_bfloat16 *src = key + (uint64_t)slot * kv_stride + (uint64_t)kv_head * D;
            const __nv_bfloat16 *vsrc = value + (uint64_t)slot * kv_stride + (uint64_t)kv_head * D;
            for (uint32_t c = lane; c < D / 8u; c += 32u) {
                g4_mma_cp16(ks + g4_mma_swz(t, c, D), src + c * 8u, valid ? 16u : 0u);
                g4_mma_cp16(vs + g4_mma_swz(t, c, D), vsrc + c * 8u, valid ? 16u : 0u);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();

        /* S = Q K^T over the 32-key tile: four n8 tiles, 16 k16 steps. */
        float s[4][4] = {};
        #pragma unroll
        for (uint32_t k = 0; k < 16u; k++) {
            #pragma unroll
            for (uint32_t j = 0; j < 4u; j += 2u) {
                uint32_t b[4];
                const uint32_t mat = lane >> 3;
                const uint32_t krow = (j + (mat >> 1)) * 8u + (lane & 7u);
                const uint32_t chunk = k * 2u + (mat & 1u);
                g4_mma_ldsm4(b, ks + g4_mma_swz(krow, chunk, D), false);
                g4_mma_mma(s[j], qa[k], b[0], b[1]);
                g4_mma_mma(s[j + 1u], qa[k], b[2], b[3]);
            }
        }

        /* Mask and online softmax, per row h = 0 (g) and h = 1 (g + 8). */
        #pragma unroll
        for (uint32_t h = 0; h < 2u; h++) {
            const uint32_t mrow = g + 8u * h;
            const uint32_t token = row0 + warp * 8u + (mrow >> 1);
            const bool rv = token < rows;
            const uint32_t pos = first + token;
            const uint32_t rb = window && pos + 1u > window ? pos + 1u - window : 0u;
            const uint32_t rl = keys && keys <= pos ? keys - 1u : pos;
            float peak = -INFINITY;
            #pragma unroll
            for (uint32_t nt = 0; nt < 4u; nt++) {
                #pragma unroll
                for (uint32_t e = 0; e < 2u; e++) {
                    const uint32_t kk = base + nt * 8u + 2u * q + e;
                    if (!rv || kk < rb || kk > rl) { s[nt][h * 2u + e] = -INFINITY; }
                    peak = fmaxf(peak, s[nt][h * 2u + e]);
                }
            }
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, 1));
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, 2));
            /* A fully masked tile keeps m, l and o unchanged; the branchless
             * form below is the same thing without a divergent shuffle. */
            const bool tv = peak > -INFINITY;
            const float mn = tv ? fmaxf(m[h], peak) : m[h];
            const float alpha = tv ? (m[h] == -INFINITY ? 0.0f : expf(m[h] - mn)) : 1.0f;
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t nt = 0; nt < 4u; nt++) {
                #pragma unroll
                for (uint32_t e = 0; e < 2u; e++) {
                    const float p = tv ? expf(s[nt][h * 2u + e] - mn) : 0.0f;
                    s[nt][h * 2u + e] = p;
                    sum += p;
                }
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            l[h] = l[h] * alpha + sum;
            m[h] = mn;
            #pragma unroll
            for (uint32_t t = 0; t < 32u; t++) {
                o[t][h * 2u + 0u] *= alpha;
                o[t][h * 2u + 1u] *= alpha;
            }
        }

        /* O += P V: two k16 steps over the 32-key tile, 32 n8 output tiles. */
        #pragma unroll
        for (uint32_t kk = 0; kk < 2u; kk++) {
            const uint32_t pa[4] = {
                g4_mma_pack(s[2u * kk][0], s[2u * kk][1]),
                g4_mma_pack(s[2u * kk][2], s[2u * kk][3]),
                g4_mma_pack(s[2u * kk + 1u][0], s[2u * kk + 1u][1]),
                g4_mma_pack(s[2u * kk + 1u][2], s[2u * kk + 1u][3])};
            #pragma unroll
            for (uint32_t t = 0; t < 32u; t += 2u) {
                uint32_t b[4];
                const uint32_t mat = lane >> 3;
                const uint32_t krow = kk * 16u + (mat & 1u) * 8u + (lane & 7u);
                const uint32_t chunk = t + (mat >> 1);
                g4_mma_ldsm4(b, vs + g4_mma_swz(krow, chunk, D), true);
                g4_mma_mma(o[t], pa, b[0], b[1]);
                g4_mma_mma(o[t + 1u], pa, b[2], b[3]);
            }
        }
        __syncthreads();
    }

    const float inv[2] = {l[0] > 0.0f ? 1.0f / l[0] : 0.0f,
                          l[1] > 0.0f ? 1.0f / l[1] : 0.0f};
    #pragma unroll
    for (uint32_t h = 0; h < 2u; h++) {
        const uint32_t mrow = g + 8u * h;
        const uint32_t token = row0 + warp * 8u + (mrow >> 1);
        if (token >= rows) { continue; }
        const uint32_t head = kv_head * GROUP + (mrow & 1u);
        float *dst = out + ((uint64_t)token * heads + head) * D;
        #pragma unroll
        for (uint32_t t = 0; t < 32u; t++) {
            const uint32_t d = t * 8u + 2u * q;
            dst[d] = o[t][h * 2u + 0u] * inv[h];
            dst[d + 1u] = o[t][h * 2u + 1u] * inv[h];
        }
    }
}

/* One 32-key tile of the full geometry: D = 512, group = 8, 8 tokens per
 * CTA. Each warp owns 2 tokens x 8 query heads (16 M rows) and one 256-dim
 * half; the two halves of an M group exchange their partial 16x32 scores in
 * shared memory and add in half order before the softmax.
 *
 * FP8 = true runs the same walk over the full layers' combined E4M3 rows
 * (ds4_gemma4.h, ds4_gemma4_full_kv_row): K at kv_head * D bytes, V at
 * kv_dim + kv_head * D, the two f16 scale arrays at 2 * kv_dim, and the tile
 * is decoded to f16 on the way into shared memory (g4_fp8_decode8) instead
 * of being cp.async'd. Everything after the staging step is the bf16 path:
 * the same swizzle, ldmatrix, mask, online softmax and half-order exchange. */
template <bool FP8>
__global__ void __launch_bounds__(G4_MMA_THREADS, 1) g4_attn_mma_full_kernel(
        float *out, const float *query, const __nv_bfloat16 *key,
        const __nv_bfloat16 *value, const uint8_t *fp8, uint32_t rows,
        uint32_t first, uint32_t heads, uint32_t kv_heads, uint32_t window,
        uint32_t ring, uint32_t keys, uint64_t kv_stride, uint64_t row_bytes) {
    constexpr uint32_t D = 512u, GROUP = 8u, RTILE = 8u, TK = G4_MMA_TK;
    constexpr uint32_t HALF = 256u;
    using E = typename g4_mma_elem<FP8>::type;
    extern __shared__ __align__(16) uint8_t smem[];
    E *kv = (E *)smem;
    float *exchange = (float *)(kv + TK * D);
    const uint64_t kv_dim = (uint64_t)kv_heads * D;

    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, q = lane & 3u;
    const uint32_t kv_head = blockIdx.x;
    const uint32_t row0 = blockIdx.y * RTILE;
    const uint32_t row_end = row0 + RTILE < rows ? row0 + RTILE : rows;
    const uint32_t pos_first = first + row0;
    const uint32_t pos_last = first + row_end - 1u;
    const uint32_t begin_all = window && pos_first + 1u > window ? pos_first + 1u - window : 0u;
    const uint32_t last_all = keys && keys <= pos_last ? keys - 1u : pos_last;
    if (keys && begin_all >= keys) { return; }
    const uint32_t base_start = (begin_all / TK) * TK;
    const uint32_t mgroup = warp & 3u, half = warp >> 2;

    /* Q fragments for this 256-dim half; M row = 2 tokens x 8 heads. */
    uint32_t qa[16][4];
    #pragma unroll
    for (uint32_t k = 0; k < 16u; k++) {
        const uint32_t col = half * HALF + k * 16u + 2u * q;
        #pragma unroll
        for (uint32_t m = 0; m < 2u; m++) {
            const uint32_t row_g = g + 8u * m;
            const uint32_t token = row0 + mgroup * 2u + (row_g >> 3);
            const bool valid = token < rows;
            const uint32_t head = kv_head * GROUP + (row_g & 7u);
            const float *qr = query + ((uint64_t)token * heads + head) * D;
            const float lo = valid ? qr[col] : 0.0f;
            const float hi = valid ? qr[col + 1u] : 0.0f;
            const float lo8 = valid ? qr[col + 8u] : 0.0f;
            const float hi8 = valid ? qr[col + 9u] : 0.0f;
            if (m == 0u) {
                qa[k][0] = g4_mma_pack_t<FP8>(lo, hi);
                qa[k][2] = g4_mma_pack_t<FP8>(lo8, hi8);
            } else {
                qa[k][1] = g4_mma_pack_t<FP8>(lo, hi);
                qa[k][3] = g4_mma_pack_t<FP8>(lo8, hi8);
            }
        }
    }

    float o[32][4] = {};
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};

    for (uint32_t base = base_start; base <= last_all; base += TK) {
        /* Stage K: 32 keys x 512 dims, 64 chunks per key, 2 per lane. The
         * bf16 lane cp.asyncs each 16-byte chunk; the E4M3 lane decodes 8
         * codes into the same 16-byte shared slot, so the tile layout and
         * the ldmatrix addressing below are identical for both. */
        if constexpr (FP8) {
            for (uint32_t t = warp * 4u; t < warp * 4u + 4u; t++) {
                const uint32_t k_abs = base + t;
                const bool valid = k_abs >= begin_all && k_abs <= last_all;
                const uint32_t slot = valid ? (ring ? (k_abs % ring) : k_abs) : 0u;
                const uint8_t *row = fp8 + (uint64_t)slot * row_bytes;
                const __half kscale = ((const __half *)(row + 2u * kv_dim))[kv_head];
                const uint8_t *src = row + (uint64_t)kv_head * D;
                for (uint32_t c = lane; c < D / 8u; c += 32u) {
                    const uint2 raw = valid ? __ldg((const uint2 *)(src + c * 8u))
                                            : make_uint2(0u, 0u);
                    *(uint4 *)(kv + g4_mma_swz(t, c, D)) = g4_fp8_decode8(raw, kscale);
                }
            }
        } else {
            for (uint32_t t = warp * 4u; t < warp * 4u + 4u; t++) {
                const uint32_t k_abs = base + t;
                const bool valid = k_abs >= begin_all && k_abs <= last_all;
                const uint32_t slot = valid ? (ring ? (k_abs % ring) : k_abs) : 0u;
                const __nv_bfloat16 *src = key + (uint64_t)slot * kv_stride + (uint64_t)kv_head * D;
                for (uint32_t c = lane; c < D / 8u; c += 32u) {
                    g4_mma_cp16(kv + g4_mma_swz(t, c, D), src + c * 8u, valid ? 16u : 0u);
                }
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();

        /* Partial S over this warp's half: four n8 tiles, 16 k16 steps. */
        float s[4][4] = {};
        #pragma unroll
        for (uint32_t k = 0; k < 16u; k++) {
            #pragma unroll
            for (uint32_t j = 0; j < 4u; j += 2u) {
                uint32_t b[4];
                const uint32_t mat = lane >> 3;
                const uint32_t krow = (j + (mat >> 1)) * 8u + (lane & 7u);
                const uint32_t chunk = half * 32u + k * 2u + (mat & 1u);
                g4_mma_ldsm4(b, kv + g4_mma_swz(krow, chunk, D), false);
                g4_mma_mma_t<FP8>(s[j], qa[k], b[0], b[1]);
                g4_mma_mma_t<FP8>(s[j + 1u], qa[k], b[2], b[3]);
            }
        }

        /* Publish the half's scores and add the two halves in half order. */
        float *mine = exchange + (mgroup * 2u + half) * 16u * 32u;
        #pragma unroll
        for (uint32_t nt = 0; nt < 4u; nt++) {
            *(float2 *)(mine + g * 32u + nt * 8u + 2u * q) = make_float2(s[nt][0], s[nt][1]);
            *(float2 *)(mine + (g + 8u) * 32u + nt * 8u + 2u * q) = make_float2(s[nt][2], s[nt][3]);
        }
        __syncthreads();
        const float *ex0 = exchange + (mgroup * 2u + 0u) * 16u * 32u;
        const float *ex1 = exchange + (mgroup * 2u + 1u) * 16u * 32u;
        #pragma unroll
        for (uint32_t nt = 0; nt < 4u; nt++) {
            const float2 a0 = *(const float2 *)(ex0 + g * 32u + nt * 8u + 2u * q);
            const float2 b0 = *(const float2 *)(ex1 + g * 32u + nt * 8u + 2u * q);
            const float2 a1 = *(const float2 *)(ex0 + (g + 8u) * 32u + nt * 8u + 2u * q);
            const float2 b1 = *(const float2 *)(ex1 + (g + 8u) * 32u + nt * 8u + 2u * q);
            s[nt][0] = a0.x + b0.x; s[nt][1] = a0.y + b0.y;
            s[nt][2] = a1.x + b1.x; s[nt][3] = a1.y + b1.y;
        }

        /* Mask and online softmax, per row h = 0 (g) and h = 1 (g + 8). */
        #pragma unroll
        for (uint32_t h = 0; h < 2u; h++) {
            const uint32_t row_g = g + 8u * h;
            const uint32_t token = row0 + mgroup * 2u + (row_g >> 3);
            const bool rv = token < rows;
            const uint32_t pos = first + token;
            const uint32_t rb = window && pos + 1u > window ? pos + 1u - window : 0u;
            const uint32_t rl = keys && keys <= pos ? keys - 1u : pos;
            float peak = -INFINITY;
            #pragma unroll
            for (uint32_t nt = 0; nt < 4u; nt++) {
                #pragma unroll
                for (uint32_t e = 0; e < 2u; e++) {
                    const uint32_t kk = base + nt * 8u + 2u * q + e;
                    if (!rv || kk < rb || kk > rl) { s[nt][h * 2u + e] = -INFINITY; }
                    peak = fmaxf(peak, s[nt][h * 2u + e]);
                }
            }
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, 1));
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, 2));
            const bool tv = peak > -INFINITY;
            const float mn = tv ? fmaxf(m[h], peak) : m[h];
            const float alpha = tv ? (m[h] == -INFINITY ? 0.0f : expf(m[h] - mn)) : 1.0f;
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t nt = 0; nt < 4u; nt++) {
                #pragma unroll
                for (uint32_t e = 0; e < 2u; e++) {
                    const float p = tv ? expf(s[nt][h * 2u + e] - mn) : 0.0f;
                    s[nt][h * 2u + e] = p;
                    sum += p;
                }
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            l[h] = l[h] * alpha + sum;
            m[h] = mn;
            #pragma unroll
            for (uint32_t t = 0; t < 32u; t++) {
                o[t][h * 2u + 0u] *= alpha;
                o[t][h * 2u + 1u] *= alpha;
            }
        }

        /* Stage V over the K tile (K is dead after the QK barrier) and
         * accumulate O += P V over this half's 256 output dims. */
        __syncthreads();
        if constexpr (FP8) {
            for (uint32_t t = warp * 4u; t < warp * 4u + 4u; t++) {
                const uint32_t k_abs = base + t;
                const bool valid = k_abs >= begin_all && k_abs <= last_all;
                const uint32_t slot = valid ? (ring ? (k_abs % ring) : k_abs) : 0u;
                const uint8_t *row = fp8 + (uint64_t)slot * row_bytes;
                const __half vscale = ((const __half *)(row + 2u * kv_dim))[kv_heads + kv_head];
                const uint8_t *src = row + kv_dim + (uint64_t)kv_head * D;
                for (uint32_t c = lane; c < D / 8u; c += 32u) {
                    const uint2 raw = valid ? __ldg((const uint2 *)(src + c * 8u))
                                            : make_uint2(0u, 0u);
                    *(uint4 *)(kv + g4_mma_swz(t, c, D)) = g4_fp8_decode8(raw, vscale);
                }
            }
        } else {
            for (uint32_t t = warp * 4u; t < warp * 4u + 4u; t++) {
                const uint32_t k_abs = base + t;
                const bool valid = k_abs >= begin_all && k_abs <= last_all;
                const uint32_t slot = valid ? (ring ? (k_abs % ring) : k_abs) : 0u;
                const __nv_bfloat16 *src = value + (uint64_t)slot * kv_stride + (uint64_t)kv_head * D;
                for (uint32_t c = lane; c < D / 8u; c += 32u) {
                    g4_mma_cp16(kv + g4_mma_swz(t, c, D), src + c * 8u, valid ? 16u : 0u);
                }
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();

        #pragma unroll
        for (uint32_t kk = 0; kk < 2u; kk++) {
            const uint32_t pa[4] = {
                g4_mma_pack_t<FP8>(s[2u * kk][0], s[2u * kk][1]),
                g4_mma_pack_t<FP8>(s[2u * kk][2], s[2u * kk][3]),
                g4_mma_pack_t<FP8>(s[2u * kk + 1u][0], s[2u * kk + 1u][1]),
                g4_mma_pack_t<FP8>(s[2u * kk + 1u][2], s[2u * kk + 1u][3])};
            #pragma unroll
            for (uint32_t t = 0; t < 32u; t += 2u) {
                uint32_t b[4];
                const uint32_t mat = lane >> 3;
                const uint32_t krow = kk * 16u + (mat & 1u) * 8u + (lane & 7u);
                const uint32_t chunk = half * 32u + t + (mat >> 1);
                g4_mma_ldsm4(b, kv + g4_mma_swz(krow, chunk, D), true);
                g4_mma_mma_t<FP8>(o[t], pa, b[0], b[1]);
                g4_mma_mma_t<FP8>(o[t + 1u], pa, b[2], b[3]);
            }
        }
        __syncthreads();
    }

    const float inv[2] = {l[0] > 0.0f ? 1.0f / l[0] : 0.0f,
                          l[1] > 0.0f ? 1.0f / l[1] : 0.0f};
    #pragma unroll
    for (uint32_t h = 0; h < 2u; h++) {
        const uint32_t row_g = g + 8u * h;
        const uint32_t token = row0 + mgroup * 2u + (row_g >> 3);
        if (token >= rows) { continue; }
        const uint32_t head = kv_head * GROUP + (row_g & 7u);
        float *dst = out + ((uint64_t)token * heads + head) * D + half * HALF;
        #pragma unroll
        for (uint32_t t = 0; t < 32u; t++) {
            const uint32_t d = t * 8u + 2u * q;
            dst[d] = o[t][h * 2u + 0u] * inv[h];
            dst[d + 1u] = o[t][h * 2u + 1u] * inv[h];
        }
    }
}

#endif
