#ifndef DS4_GLM_VQ_Q4P_CUH
#define DS4_GLM_VQ_Q4P_CUH

#include <cuda_fp16.h>
#include "glm_vq_ops.cuh"
#include "glm_vq_q4.cuh"

/* Prefill Q4_K GEMM on FP16 tensor cores.
 *
 * The vendored Q4_K x Q8_1 MMQ scales every k32 int32 partial by its weight
 * and activation scales: 4 I2F, 4 FMUL and 8 FFMA per IMMA were 74% of its
 * instructions and its tensor pipe ran at 29% (8192 x 4096 x 4096 tokens:
 * 4.48 ms plus 0.38 ms of Q8_1 quantization).
 *
 * Here producer warps dequantize each weight once into the MMA's A fragment
 * order in shared memory, and consumer warps run a plain FP16 GEMM:
 *
 *   y = 2^(e-6) * sum_k fp16(fp16(q_k S_s) - M_s) x'_k
 *   S_s = fp16(64 d sc_s), M_s = fp16(64 dmin m_s)
 *   x' = fp16(x 2^-e), e = 0 unless the token's max |x| exceeds 2^15
 *
 * q S comes from one HFMA2 on the 1024+q magic: (1024+q) S - 1024 S. MMQ also
 * rounds d sc and dmin m to FP16; here the weight rounds twice more (q S,
 * then - M), and activations round to FP16 (11 bits) instead of Q8_1 (8 bits
 * per 32 values). The 2^6 keeps the GLM artifact's smallest weights normal;
 * glmvq_q4p_check rejects matrices whose 1024 S or M would overflow.
 *
 *   Tile: 128 rows x 128 tokens; one persistent CTA per SM walks the tiles.
 *   Warps 0-7 compute 32 rows x 64 tokens each; warps 8-11 load raw rows one
 *   stage early, write dequantized A fragments and copy token rows into a
 *   three-slot ring (full/empty mbarriers). A stage is one 32-byte qs chunk
 *   of every row (64 values: sub-blocks 2p and 2p+1 in the low and high
 *   nibbles). Groups of eight token tiles walk the rows, so the CTAs in
 *   flight share weight and token tiles in L2. */
enum { GLMVQ_Q4P_BM = 128, GLMVQ_Q4P_BN = 128, GLMVQ_Q4P_BK = 64, GLMVQ_Q4P_STAGES = 3,
       GLMVQ_Q4P_CONSUMERS = 256, GLMVQ_Q4P_PRODUCERS = 128,
       GLMVQ_Q4P_THREADS = GLMVQ_Q4P_CONSUMERS + GLMVQ_Q4P_PRODUCERS,
       GLMVQ_Q4P_NT = 8, GLMVQ_Q4P_GROUP = 8, GLMVQ_Q4P_PREP = 128,
       GLMVQ_Q4P_COLS = GLMVQ_Q4P_PREP * 4, GLMVQ_Q4P_SHIFT = 6 };
// A: [m tile 8][sub-block 2][k16 half 2][lane 32] fragments of 16 bytes.
static constexpr uint32_t GLMVQ_Q4P_A = GLMVQ_Q4P_BM * GLMVQ_Q4P_BK * sizeof(__half);
static constexpr uint32_t GLMVQ_Q4P_B = GLMVQ_Q4P_BN * GLMVQ_Q4P_BK * sizeof(__half);
static constexpr uint32_t GLMVQ_Q4P_SMEM = GLMVQ_Q4P_STAGES * (GLMVQ_Q4P_A + GLMVQ_Q4P_B) +
    2 * GLMVQ_Q4P_STAGES * sizeof(uint64_t);
static constexpr float GLMVQ_Q4P_RANGE = 32768.0f;
// 1024 S and M stay below the FP16 maximum.
static constexpr float GLMVQ_Q4P_SCALE_MAX = 63.96f / 64.0f, GLMVQ_Q4P_MIN_MAX = 1023.0f;

__device__ __forceinline__ uint32_t glmvq_q4p_h2(__half lo, __half hi) {
    return (uint32_t)__half_as_ushort(lo) | ((uint32_t)__half_as_ushort(hi) << 16);
}

__device__ __forceinline__ float4 glmvq_q4p_quad(const float *x, uint32_t i) { return ((const float4 *)x)[i]; }

// BF16 rows hold the same BF16-rounded values in half the bytes.
__device__ __forceinline__ float4 glmvq_q4p_quad(const __nv_bfloat16 *x, uint32_t i) {
    const uint2 v = ((const uint2 *)x)[i];
    return make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
        __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
}

// x' for `scale`; returns the row's max |x| (before scaling).
template <typename T>
__device__ __forceinline__ float glmvq_q4p_row(const T *x, __half *out, uint32_t quads, float scale) {
    float peak = 0.0f;
    for (uint32_t i = threadIdx.x; i < quads; i += GLMVQ_Q4P_PREP) {
        const float4 v = glmvq_q4p_quad(x, i);
        peak = fmaxf(peak, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        const __half h0 = __float2half_rn(v.x * scale), h1 = __float2half_rn(v.y * scale);
        const __half h2 = __float2half_rn(v.z * scale), h3 = __float2half_rn(v.w * scale);
        *(uint2 *)(out + 4 * i) = make_uint2(glmvq_q4p_h2(h0, h2), glmvq_q4p_h2(h1, h3));
    }
    return peak;
}

/* One CTA per token: x' as FP16 with each four values stored (0, 2, 1, 3),
 * so one 32-bit load is an MMA register pair, and the 2^(e-6) output scale.
 * A token beyond 2^15 (none in GLM so far) runs a second pass. */
template <typename T>
__global__ void __launch_bounds__(GLMVQ_Q4P_PREP) glmvq_q4p_prep(const T *input,
        __half *xh, float *xscale, uint32_t cols) {
    __shared__ float peaks[GLMVQ_Q4P_PREP / 32];
    const uint32_t token = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const T *x = input + (uint64_t)token * cols;
    __half *out = xh + (uint64_t)token * cols;
    float peak = glmvq_q4p_row(x, out, cols / 4, 1.0f);
    for (uint32_t s = 16; s; s >>= 1) { peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, s)); }
    if (!lane) { peaks[warp] = peak; }
    __syncthreads();
    peak = fmaxf(fmaxf(peaks[0], peaks[1]), fmaxf(peaks[2], peaks[3]));
    // Power-of-two scaling is exact; FP16 overflows at 65520.
    int e = 0;
    if (peak > GLMVQ_Q4P_RANGE) {
        frexpf(peak / GLMVQ_Q4P_RANGE, &e);
        glmvq_q4p_row(x, out, cols / 4, ldexpf(1.0f, -e));
    }
    if (!threadIdx.x) { xscale[token] = ldexpf(1.0f, e - GLMVQ_Q4P_SHIFT); }
}

// Sets *bad when a Q4_K block's scales leave the folded FP16 range.
__global__ void glmvq_q4p_check(const uint8_t *weight, uint64_t blocks, uint32_t *bad) {
    const uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) { return; }
    const uint4 h = *(const uint4 *)(weight + b * GLMVQ_Q4_BYTES);
    const float d = __half2float(__ushort_as_half(h.x & 65535u)), dmin = __half2float(__ushort_as_half(h.x >> 16));
    float worst = 0.0f, minimum = 0.0f;
    for (uint32_t s = 0; s < 8; s++) {
        float sc, m;
        glmvq_q4_scales(h, s, &sc, &m);
        worst = fmaxf(worst, fabsf(d) * sc);
        minimum = fmaxf(minimum, fabsf(dmin) * m);
    }
    if (!(worst <= GLMVQ_Q4P_SCALE_MAX) || !(minimum <= GLMVQ_Q4P_MIN_MAX)) { atomicOr(bad, 1u); }
}

__device__ __forceinline__ uint32_t glmvq_q4p_smem(const void *p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void glmvq_q4p_wait(const uint64_t *bar, uint32_t parity) {
    asm volatile("{\n .reg .pred done;\n Q4P_WAIT%=:\n"
        " mbarrier.try_wait.parity.shared.b64 done, [%0], %1;\n"
        " @!done bra Q4P_WAIT%=;\n}" :: "r"(glmvq_q4p_smem(bar)), "r"(parity) : "memory");
}

__device__ __forceinline__ void glmvq_q4p_arrive(uint64_t *bar) {
    asm volatile("{\n .reg .b64 state;\n mbarrier.arrive.shared.b64 state, [%0];\n}"
        :: "r"(glmvq_q4p_smem(bar)) : "memory");
}

// Token rows of 8 16-byte chunks: tokens 2a and 2a+1 use opposite chunk
// halves, so the four chunks a half-warp phase reads hit distinct banks.
__device__ __forceinline__ uint32_t glmvq_q4p_swz(uint32_t token) {
    return ((token & 1u) << 2) ^ ((token >> 1) & 3u);
}

// Row scales for sub-block s: S, -1024 S and M as replicated FP16 pairs.
struct glmvq_q4p_scale { __half2 s, bias, m; };

__device__ __forceinline__ glmvq_q4p_scale glmvq_q4p_scales(const uint4 &h, uint32_t s) {
    float sc, m;
    glmvq_q4_scales(h, s, &sc, &m);
    const float d = __half2float(__ushort_as_half(h.x & 65535u)), dmin = __half2float(__ushort_as_half(h.x >> 16));
    const __half2 scale = __float2half2_rn(d * sc * (1u << GLMVQ_Q4P_SHIFT));
    return {scale, __hmul2(scale, __float2half2_rn(-1024.0f)), __float2half2_rn(dmin * m * (1u << GLMVQ_Q4P_SHIFT))};
}

// Dequantized pair for the nibbles at bits 0-3 and 16-19: ((1024+q) S - 1024 S) - M.
__device__ __forceinline__ uint32_t glmvq_q4p_weight(uint32_t v, const glmvq_q4p_scale &k) {
    uint32_t magic;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(magic) : "r"(v), "r"(0x000F000Fu), "r"(0x64006400u));
    const __half2 w = __hsub2(__hfma2(*(const __half2 *)&magic, k.s, k.bias), k.m);
    return *(const uint32_t *)&w;
}

/* FP8 weights (Ling-3.1's KDA decay/gate): row-major E4M3 bytes with F32
 * 128 x 128 block scales. The producer writes fp16(e4m3 * scale * 2^6), one
 * rounding of the exact F32 product, into the Q4_K fragment slots, so the
 * consumers and the 2^-6 epilogue are shared. */
enum glmvq_q4p_kind { GLMVQ_Q4P_Q4K, GLMVQ_Q4P_FP8 };
static constexpr float GLMVQ_Q4P_FP8_SCALE_MAX = 65504.0f / (448.0f * (1u << GLMVQ_Q4P_SHIFT));

// Sets *bad when a block scale would overflow the folded FP16 weight.
__global__ void glmvq_q4p_check_fp8(const float *scales, uint64_t blocks, uint32_t *bad) {
    const uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b < blocks && !(fabsf(scales[b]) <= GLMVQ_Q4P_FP8_SCALE_MAX)) { atomicOr(bad, 1u); }
}

// Bytes `sel` of u (two E4M3 values) as an FP16 pair times `scale`.
__device__ __forceinline__ uint32_t glmvq_q4p_fp8(uint32_t u, uint32_t sel, float scale) {
    const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)__byte_perm(u, 0u, sel), __NV_E4M3);
    const float2 f = __half22float2(*(const __half2 *)&raw);
    const __half2 w = __floats2half2_rn(f.x * scale, f.y * scale);
    return *(const uint32_t *)&w;
}

__device__ __forceinline__ void glmvq_q4p_mma(float c[4], const uint4 &a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

/* y[token][row] for rows multiple of 128 and cols of 512, as F32 or as the
 * BF16 rounding decode and verify store. Fragment lane (g, c) owns elements
 * 8c..8c+7 of every sub-block: bytes 8c..8c+7 of a row's qs chunk and one
 * 16-byte token load, in the q4m element order. */
template <bool TAIL, typename OUT, glmvq_q4p_kind KIND = GLMVQ_Q4P_Q4K>
__global__ void __launch_bounds__(GLMVQ_Q4P_THREADS, 1) glmvq_q4p(const uint8_t *weight,
        const __half *xh, const float *xscale, OUT *out, uint32_t rows, uint32_t cols, uint32_t tokens,
        const float *scales) {
    constexpr uint32_t BM = GLMVQ_Q4P_BM, BN = GLMVQ_Q4P_BN, STAGES = GLMVQ_Q4P_STAGES;
    constexpr uint32_t NT = GLMVQ_Q4P_NT, CONSUMERS = GLMVQ_Q4P_CONSUMERS, PRODUCERS = GLMVQ_Q4P_PRODUCERS;
    extern __shared__ __align__(16) uint8_t q4p_shared[];
    uint8_t *sa = q4p_shared, *sb = sa + STAGES * GLMVQ_Q4P_A;
    uint64_t *full = (uint64_t *)(sb + STAGES * GLMVQ_Q4P_B), *empty = full + STAGES;
    const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    const uint32_t m_tiles = rows / BM, n_tiles = (tokens + BN - 1) / BN, tiles = m_tiles * n_tiles;
    const uint32_t steps = cols / GLMVQ_Q4P_BK;
    const uint64_t row_bytes = KIND == GLMVQ_Q4P_FP8 ? cols : (uint64_t)cols / GLMVQ_Q4_BLOCK * GLMVQ_Q4_BYTES;
    // Tile t: groups of GROUP token tiles walk the rows.
    auto origin = [&](uint32_t t, uint32_t &m0, uint32_t &n0) {
        const uint32_t per = GLMVQ_Q4P_GROUP * m_tiles, group = t / per, local = t % per;
        const uint32_t first = group * GLMVQ_Q4P_GROUP, width = min((uint32_t)GLMVQ_Q4P_GROUP, n_tiles - first);
        m0 = local / width * BM;
        n0 = (first + local % width) * BN;
    };

    if (!tid) {
        for (uint32_t s = 0; s < STAGES; s++) {
            asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(glmvq_q4p_smem(full + s)), "r"(2u * PRODUCERS) : "memory");
            asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(glmvq_q4p_smem(empty + s)), "r"(CONSUMERS / 32) : "memory");
        }
    }
    __syncthreads();

    // Persistent CTAs: the ring counter runs on across tiles, so the next
    // tile's first stages load while the consumers finish the last one.
    if (tid >= CONSUMERS) {
        // Producer lane pl owns fragment items pl and pl + 128: item = (m tile,
        // g, c), rows 16mt + g and + 8, qs bytes 8c..8c+7 of each stage.
        const uint32_t pl = tid - CONSUMERS, qs = 16 + 8 * (pl & 3);
        uint32_t slot[2];
        #pragma unroll
        for (uint32_t i = 0; i < 2; i++) { slot[i] = (pl + PRODUCERS * i) / 32 * 4 * 512 + (pl & 31) * 16; }
        // Token chunk pl & 7 of tokens pl / 8 + 16i.
        const uint32_t chunk = pl & 7, t0 = pl >> 3;
        const uint32_t b_dst = glmvq_q4p_smem(sb + t0 * 128 + ((chunk ^ glmvq_q4p_swz(t0)) << 4));
        uint32_t ring = 0;
        for (uint32_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
            uint32_t m0, n0;
            origin(tile, m0, n0);
            const uint8_t *rowp[2][2];
            #pragma unroll
            for (uint32_t i = 0; i < 2; i++) {
                const uint32_t item = pl + PRODUCERS * i, mt = item >> 5, g = (item >> 2) & 7;
                rowp[i][0] = weight + (uint64_t)(m0 + 16 * mt + g) * row_bytes;
                rowp[i][1] = rowp[i][0] + 8 * row_bytes;
            }
            const __half *b_src = xh + (uint64_t)(n0 + t0) * cols + chunk * 8;
            if constexpr (KIND == GLMVQ_Q4P_FP8) {
                // Stage kt is bytes 64 kt.. of each row; lane c takes 8 bytes at
                // 32 n + 8 c, the elements the Q4_K nibbles of sub-block n give it.
                const float *block = scales + (uint64_t)(m0 / GLMVQ_SCALE_BLOCK) * (cols / GLMVQ_SCALE_BLOCK);
                const uint32_t c8 = 8 * (pl & 3);
                auto raw8 = [&](uint32_t kt, uint2 v[2][2][2]) {
                    #pragma unroll
                    for (uint32_t i = 0; i < 2; i++) {
                        #pragma unroll
                        for (uint32_t n = 0; n < 2; n++) {
                            v[i][0][n] = __ldg((const uint2 *)(rowp[i][0] + kt * GLMVQ_Q4P_BK + 32 * n + c8));
                            v[i][1][n] = __ldg((const uint2 *)(rowp[i][1] + kt * GLMVQ_Q4P_BK + 32 * n + c8));
                        }
                    }
                };
                uint2 v[2][2][2], vn[2][2][2];
                raw8(0, v);
                for (uint32_t kt = 0; kt < steps; kt++, ring++) {
                    const uint32_t s = ring % STAGES, use = ring / STAGES;
                    if (kt + 1 < steps) { raw8(kt + 1, vn); }
                    if (use) { glmvq_q4p_wait(empty + s, (use - 1) & 1); }
                    #pragma unroll
                    for (uint32_t i = 0; i < BN * 8 / PRODUCERS; i++) {
                        const bool live = !TAIL || n0 + t0 + 16 * i < tokens;
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
                            "r"(b_dst + s * GLMVQ_Q4P_B + i * 16 * 128),
                            "l"(live ? b_src + (uint64_t)16 * i * cols + kt * GLMVQ_Q4P_BK : xh), "r"(live ? 16 : 0) : "memory");
                    }
                    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(glmvq_q4p_smem(full + s)) : "memory");
                    const float scale = block[kt * GLMVQ_Q4P_BK / GLMVQ_SCALE_BLOCK] * (1u << GLMVQ_Q4P_SHIFT);
                    // Pairs (e0, e2) and (e1, e3) of each four bytes, as the nibbles pair.
                    #pragma unroll
                    for (uint32_t i = 0; i < 2; i++) {
                        #pragma unroll
                        for (uint32_t n = 0; n < 2; n++) {
                            #pragma unroll
                            for (uint32_t kh = 0; kh < 2; kh++) {
                                const uint32_t u0 = kh ? v[i][0][n].y : v[i][0][n].x;
                                const uint32_t u1 = kh ? v[i][1][n].y : v[i][1][n].x;
                                uint4 a;
                                a.x = glmvq_q4p_fp8(u0, 0x0020u, scale);
                                a.z = glmvq_q4p_fp8(u0, 0x0031u, scale);
                                a.y = glmvq_q4p_fp8(u1, 0x0020u, scale);
                                a.w = glmvq_q4p_fp8(u1, 0x0031u, scale);
                                *(uint4 *)(sa + s * GLMVQ_Q4P_A + slot[i] + (n * 2 + kh) * 512) = a;
                            }
                        }
                    }
                    glmvq_q4p_arrive(full + s);
                    #pragma unroll
                    for (uint32_t i = 0; i < 2; i++) {
                        #pragma unroll
                        for (uint32_t n = 0; n < 2; n++) { v[i][0][n] = vn[i][0][n]; v[i][1][n] = vn[i][1][n]; }
                    }
                }
                continue;
            }
            auto raw = [&](uint32_t kt, uint2 w[2][2]) {
                const uint32_t off = (kt >> 2) * GLMVQ_Q4_BYTES + qs + 32 * (kt & 3);
                #pragma unroll
                for (uint32_t i = 0; i < 2; i++) {
                    w[i][0] = __ldg((const uint2 *)(rowp[i][0] + off));
                    w[i][1] = __ldg((const uint2 *)(rowp[i][1] + off));
                }
            };
            auto header = [&](uint32_t block, uint4 h[2][2]) {
                #pragma unroll
                for (uint32_t i = 0; i < 2; i++) {
                    h[i][0] = __ldg((const uint4 *)(rowp[i][0] + block * GLMVQ_Q4_BYTES));
                    h[i][1] = __ldg((const uint4 *)(rowp[i][1] + block * GLMVQ_Q4_BYTES));
                }
            };
            uint2 w[2][2], wn[2][2];
            uint4 h[2][2], hn[2][2];
            raw(0, w);
            header(0, h);
            for (uint32_t kt = 0; kt < steps; kt++, ring++) {
                const uint32_t s = ring % STAGES, use = ring / STAGES, p = kt & 3;
                if (kt + 1 < steps) { raw(kt + 1, wn); }
                if (p == 3 && kt + 1 < steps) { header((kt + 1) >> 2, hn); }
                if (use) { glmvq_q4p_wait(empty + s, (use - 1) & 1); }
                #pragma unroll
                for (uint32_t i = 0; i < BN * 8 / PRODUCERS; i++) {
                    const bool live = !TAIL || n0 + t0 + 16 * i < tokens;
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
                        "r"(b_dst + s * GLMVQ_Q4P_B + i * 16 * 128),
                        "l"(live ? b_src + (uint64_t)16 * i * cols + kt * GLMVQ_Q4P_BK : xh), "r"(live ? 16 : 0) : "memory");
                }
                asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(glmvq_q4p_smem(full + s)) : "memory");
                // Sub-block 2p + n sits in the low (n = 0) or high (n = 1) nibbles.
                #pragma unroll
                for (uint32_t i = 0; i < 2; i++) {
                    #pragma unroll
                    for (uint32_t n = 0; n < 2; n++) {
                        const glmvq_q4p_scale k0 = glmvq_q4p_scales(h[i][0], 2 * p + n);
                        const glmvq_q4p_scale k1 = glmvq_q4p_scales(h[i][1], 2 * p + n);
                        #pragma unroll
                        for (uint32_t kh = 0; kh < 2; kh++) {
                            const uint32_t u0 = kh ? w[i][0].y : w[i][0].x, u1 = kh ? w[i][1].y : w[i][1].x;
                            uint4 a;
                            a.x = glmvq_q4p_weight(u0 >> (4 * n), k0);
                            a.z = glmvq_q4p_weight(u0 >> (8 + 4 * n), k0);
                            a.y = glmvq_q4p_weight(u1 >> (4 * n), k1);
                            a.w = glmvq_q4p_weight(u1 >> (8 + 4 * n), k1);
                            *(uint4 *)(sa + s * GLMVQ_Q4P_A + slot[i] + (n * 2 + kh) * 512) = a;
                        }
                    }
                }
                glmvq_q4p_arrive(full + s);
                #pragma unroll
                for (uint32_t i = 0; i < 2; i++) {
                    w[i][0] = wn[i][0];
                    w[i][1] = wn[i][1];
                    if (p == 3) { h[i][0] = hn[i][0]; h[i][1] = hn[i][1]; }
                }
            }
        }
        asm volatile("cp.async.wait_all;" ::: "memory");
        return;
    }

    const uint32_t g = lane >> 2, c = lane & 3, wm = warp & 3, wn = warp >> 2;
    uint32_t ring = 0;
    for (uint32_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        uint32_t m0, n0;
        origin(tile, m0, n0);
        float acc[2][NT][4] = {};
        for (uint32_t kt = 0; kt < steps; kt++, ring++) {
            const uint32_t stage = ring % STAGES;
            glmvq_q4p_wait(full + stage, (ring / STAGES) & 1);
            const uint8_t *as = sa + stage * GLMVQ_Q4P_A + 2 * wm * 4 * 512 + lane * 16;
            #pragma unroll
            for (uint32_t n = 0; n < 2; n++) {
                uint4 a[2][2];
                #pragma unroll
                for (uint32_t mt = 0; mt < 2; mt++) {
                    #pragma unroll
                    for (uint32_t kh = 0; kh < 2; kh++) {
                        a[mt][kh] = *(const uint4 *)(as + (mt * 4 + n * 2 + kh) * 512);
                    }
                }
                uint32_t b[NT][4];
                #pragma unroll
                for (uint32_t j = 0; j < NT; j++) {
                    const uint32_t t = wn * NT * 8 + 8 * j + g;
                    *(uint4 *)b[j] = *(const uint4 *)(sb + stage * GLMVQ_Q4P_B + t * 128 +
                        (((4 * n + c) ^ glmvq_q4p_swz(t)) << 4));
                }
                #pragma unroll
                for (uint32_t kh = 0; kh < 2; kh++) {
                    #pragma unroll
                    for (uint32_t j = 0; j < NT; j++) {
                        #pragma unroll
                        for (uint32_t mt = 0; mt < 2; mt++) {
                            glmvq_q4p_mma(acc[mt][j], a[mt][kh], b[j][2 * kh], b[j][2 * kh + 1]);
                        }
                    }
                }
            }
            // The stage's fragments are in registers: free the slot.
            __syncwarp();
            if (!lane) { glmvq_q4p_arrive(empty + stage); }
        }

        #pragma unroll
        for (uint32_t j = 0; j < NT; j++) {
            #pragma unroll
            for (uint32_t e = 0; e < 2; e++) {
                const uint32_t token = n0 + wn * NT * 8 + 8 * j + 2 * c + e;
                if (TAIL && token >= tokens) { continue; }
                const float s = xscale[token];
                #pragma unroll
                for (uint32_t mt = 0; mt < 2; mt++) {
                    OUT *dst = out + (uint64_t)token * rows + m0 + wm * 32 + 16 * mt + g;
                    dst[0] = glmvq_cast<OUT>(acc[mt][j][e] * s);
                    dst[8] = glmvq_cast<OUT>(acc[mt][j][e + 2] * s);
                }
            }
        }
    }
}

static bool glmvq_q4p_fits(uint32_t rows, uint32_t cols) {
    return rows % GLMVQ_Q4P_BM == 0 && cols % GLMVQ_Q4P_COLS == 0;
}

// Prepared-input bytes for `tokens` rows of `cols`: FP16 values and scales.
static uint64_t glmvq_q4p_bytes(uint32_t tokens, uint32_t cols) {
    return ((uint64_t)tokens * cols * sizeof(__half) + 255) / 256 * 256 + (uint64_t)tokens * sizeof(float);
}

struct glmvq_q4p_input { __half *xh; float *xscale; };

static glmvq_q4p_input glmvq_q4p_layout(void *scratch, uint32_t tokens, uint32_t cols) {
    uint8_t *base = (uint8_t *)scratch;
    return {(__half *)base, (float *)(base + ((uint64_t)tokens * cols * sizeof(__half) + 255) / 256 * 256)};
}

template <typename T>
static cudaError_t glmvq_q4p_prepare(const glmvq_q4p_input &in, const T *input,
        uint32_t tokens, uint32_t cols, cudaStream_t stream) {
    glmvq_q4p_prep<<<tokens, GLMVQ_Q4P_PREP, 0, stream>>>(input, in.xh, in.xscale, cols);
    return cudaGetLastError();
}

template <typename OUT>
static cudaError_t glmvq_q4p_gemm(const uint8_t *weight, const glmvq_q4p_input &in, OUT *out,
        uint32_t rows, uint32_t cols, uint32_t tokens, cudaStream_t stream) {
    // One persistent CTA per SM.
    int device = 0, sms = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) { return cudaErrorUnknown; }
    const uint32_t tiles = rows / GLMVQ_Q4P_BM * ((tokens + GLMVQ_Q4P_BN - 1) / GLMVQ_Q4P_BN);
    const uint32_t ctas = min(tiles, (uint32_t)sms);
    if (tokens % GLMVQ_Q4P_BN) {
        glmvq_q4p<true, OUT><<<ctas, GLMVQ_Q4P_THREADS, GLMVQ_Q4P_SMEM, stream>>>(
            weight, in.xh, in.xscale, out, rows, cols, tokens, nullptr);
    } else {
        glmvq_q4p<false, OUT><<<ctas, GLMVQ_Q4P_THREADS, GLMVQ_Q4P_SMEM, stream>>>(
            weight, in.xh, in.xscale, out, rows, cols, tokens, nullptr);
    }
    return cudaGetLastError();
}

// FP8 rows x 128 x 128 block `scales` into BF16 rows (Ling-3.1 decay/gate).
static cudaError_t glmvq_q4p_gemm_fp8(const uint8_t *weight, const float *scales, const glmvq_q4p_input &in,
        __nv_bfloat16 *out, uint32_t rows, uint32_t cols, uint32_t tokens, cudaStream_t stream) {
    int device = 0, sms = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) { return cudaErrorUnknown; }
    const uint32_t tiles = rows / GLMVQ_Q4P_BM * ((tokens + GLMVQ_Q4P_BN - 1) / GLMVQ_Q4P_BN);
    const uint32_t ctas = min(tiles, (uint32_t)sms);
    if (tokens % GLMVQ_Q4P_BN) {
        glmvq_q4p<true, __nv_bfloat16, GLMVQ_Q4P_FP8><<<ctas, GLMVQ_Q4P_THREADS, GLMVQ_Q4P_SMEM, stream>>>(
            weight, in.xh, in.xscale, out, rows, cols, tokens, scales);
    } else {
        glmvq_q4p<false, __nv_bfloat16, GLMVQ_Q4P_FP8><<<ctas, GLMVQ_Q4P_THREADS, GLMVQ_Q4P_SMEM, stream>>>(
            weight, in.xh, in.xscale, out, rows, cols, tokens, scales);
    }
    return cudaGetLastError();
}

static cudaError_t glmvq_q4p_init(void) {
    const void *kernels[] = {(const void *)glmvq_q4p<true, float>, (const void *)glmvq_q4p<false, float>,
        (const void *)glmvq_q4p<true, __nv_bfloat16>, (const void *)glmvq_q4p<false, __nv_bfloat16>,
        (const void *)glmvq_q4p<true, __nv_bfloat16, GLMVQ_Q4P_FP8>,
        (const void *)glmvq_q4p<false, __nv_bfloat16, GLMVQ_Q4P_FP8>};
    for (const void *kernel : kernels) {
        const cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GLMVQ_Q4P_SMEM);
        if (e != cudaSuccess) { return e; }
    }
    return cudaSuccess;
}

#endif
