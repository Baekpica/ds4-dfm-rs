#ifndef DS4_GEMMA_VQ_PREFILL_CUH
#define DS4_GEMMA_VQ_PREFILL_CUH

/* Gemma-4 direct VQ prefill GEMM (tokens > 1).
 *
 * Why this header exists: the grouped expert prefill restores each VQ matrix
 * to bf16 and calls cuBLAS (cuda/glm_vq_linear.cuh:80-110). For the R13+8
 * expert set that is a six-times expansion per chunk — 30 layers x 128
 * experts x 3 matrices x 1.98M weights restored to 45.7 GB of bf16, written
 * and read back by the GEMM — while the VQ stream itself is 7.5 GB. This
 * kernel reads the GVQ1 bytes once into a shared tile and multiplies there,
 * so the bf16 expansion never reaches global memory. The container format is
 * unchanged; the records are the existing R13+8 ones.
 *
 * The technique (decode a weight tile into shared memory once and reuse it
 * across the token batch) is llama.cpp's MMQ tile loader applied to the GLU
 * pair (commit c49edf096, "CUDA: fuse FFN gate/up matmuls and GLU in MMQ
 * with interleaved gate/up rows") and the sibling family's direct VQ prefill
 * (cuda/ds41_vq_prefill_mma.cuh, which carries the same idea on tensor
 * cores). This header is the scalar-FMA form.
 *
 * One launch per layer, not one per expert: the per-expert gate GEMM is
 * 704 rows, which is six 128-row tiles — six blocks, a tenth of the SMs. The
 * layer pass builds a work-item list (expert, row tile, token tile) over
 * every routed expert and launches it as one grid, so the card is filled
 * across experts. Each item writes its own rows, so the result does not
 * depend on the order items run in.
 *
 * The arithmetic is the runtime's own, not a new design:
 *   - weight: w = bf16(gain * (C1[index] + C2[res])) — the same expression
 *     cuda/glm_vq.cuh:120-129 (glmvq_restore) and cuda/gemma_vq_decode.cuh:74-93
 *     (g4_vq_dot_vector) use;
 *   - activation: x = bf16(x), the conversion glmvq_convert performs before
 *     the BF16 GEMM (cuda/glm_vq_linear.cuh:97-110);
 *   - accumulation: f32, products of two bf16 values are exact in f32;
 *   - output: bf16, the boundary glmvq_gemm applies (cuBLAS writes bf16 and
 *     glmvq_convert widens it, cuda/glm_vq_linear.cuh:97-110). Omitting it
 *     moved the 256-token frontier logits by 23 percent relL2, so it is part
 *     of the contract, not a detail.
 * The remaining difference from the restore + cuBLAS path is the K
 * accumulation order, which cuBLAS itself does not expose (it splits K for
 * these shapes, cublasLt::splitKreduce_kernel); the gate is generated tokens
 * within the measured noise, not bit identity. */

#include "gemma_vq_decode.cuh"
#include "gemma4_primitives.cuh"

/* Tile: BM weight rows x BN tokens, BK columns per round. The padded row
 * stride keeps the shared loads conflict-free: (BK + 4) bf16 = 136 B = 34
 * words, so 32 consecutive rows touch 16 banks twice, and every row base is
 * 8-byte aligned for the uint2 reads the compute loop uses. */
enum {
    G4_VQP_BM = 128,
    G4_VQP_BN = 32,
    G4_VQP_BK = 64,
    G4_VQP_PAD = 4,
    G4_VQP_RT = 4,      /* rows per compute thread */
    G4_VQP_TT = 4,      /* tokens per compute thread */
    G4_VQP_THREADS = (G4_VQP_BM / G4_VQP_RT) * (G4_VQP_BN / G4_VQP_TT)
};

/* One work item: a row tile of one expert's matrix for one token tile.
 * `a` and `b` index the layer's uploaded glmvq_matrix array; `token0` and
 * `end` are global pair rows in the gathered input (m->in), so a block
 * never touches another expert's rows. */
struct g4_vqp_item {
    uint32_t a, b, row0, token0, end;
};

/* One E4M3 pair word + one residual pair word, scaled and rounded to bf16:
 * the exact per-element expression of g4_vq_dot_vector, two weights at a
 * time. The codeword arrives as a whole 32-bit word (two E4M3 bytes) instead
 * of two byte loads, which is what the 61 percent L1TEX stall in the ncu
 * profile was waiting on; the arithmetic is unchanged, so the result stays
 * bit-identical. The E4M3 -> f16 -> f32 route is exact (ds4_beam_gpu.cuh:781-786
 * uses the same conversion for its own FP8 rows). */
__device__ __forceinline__ uint32_t g4_vqp_word(uint32_t w1, uint32_t w2, float gain) {
    float2 value = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(w1 & 0xffffu), __NV_E4M3)));
    const float2 res = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(w2 & 0xffffu), __NV_E4M3)));
    value.x += res.x;
    value.y += res.y;
    const __nv_bfloat162 pair = __floats2bfloat162_rn(glmvq_mul(gain, value.x),
        glmvq_mul(gain, value.y));
    return (uint32_t)__bfloat16_as_ushort(pair.x) | ((uint32_t)__bfloat16_as_ushort(pair.y) << 16);
}

/* Scalar pair for the F16 codeword precision (the decode kernel supports it;
 * the Gemma R13+8 artifact is E4M3). Kept next to the fast path so the two
 * cannot drift. */
__device__ __forceinline__ uint32_t g4_vqp_pair_f16(const __half *c1, const uint8_t *c2,
        float gain) {
    float2 value = __half22float2(*(const __half2 *)c1);
    if (c2) {
        const __nv_fp8x2_storage_t w2 = (__nv_fp8x2_storage_t)((uint32_t)c2[0] | ((uint32_t)c2[1] << 8));
        const float2 res = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(w2, __NV_E4M3)));
        value.x += res.x;
        value.y += res.y;
    }
    const __nv_bfloat162 pair = __floats2bfloat162_rn(glmvq_mul(gain, value.x),
        glmvq_mul(gain, value.y));
    return (uint32_t)__bfloat16_as_ushort(pair.x) | ((uint32_t)__bfloat16_as_ushort(pair.y) << 16);
}

/* Decode BM rows x BK columns of one matrix into shared bf16.
 *
 * Mapping (both advisors flagged the old one): two threads per weight row,
 * four contiguous vectors each, so the four 12-bit indices of a thread are
 * adjacent in the main stream and the four residual indices are one 32-bit
 * load. The codeword is fetched as one 8-byte uint2 from the main book; the
 * residual codeword comes from the shared copy the kernel stages once per
 * block (2 KB), which keeps the 64 KB main book in the L1 the 39 KB of
 * weight tiles leave behind. Vector v of row r is the flat vector
 * r*(cols/8) + k0/8 + v, the same index the restore walks. Rows past m.rows
 * are zero-filled and never read back. */
template <uint32_t BM, uint32_t BK, uint32_t THREADS>
__device__ __forceinline__ void g4_vqp_weight_tile(const glmvq_matrix &m, uint32_t r0,
        uint32_t k0, float gain, const uint2 *s_res, __nv_bfloat16 *ws) {
    static_assert(THREADS == BM * 2u, "the decode maps two threads per weight row");
    static_assert(BK / G4_VQ_VECTOR == 8u, "the decode maps four contiguous vectors per thread");
    constexpr uint32_t STRIDE = BK + G4_VQP_PAD;
    const uint32_t nvec = m.cols / G4_VQ_VECTOR;
    const uint32_t base = k0 / G4_VQ_VECTOR;
    const uint32_t r = threadIdx.x / 2u;
    const uint32_t v0 = (threadIdx.x % 2u) * 4u;
    const uint32_t row = r0 + r;
    uint32_t *dst = (uint32_t *)(ws + r * STRIDE + v0 * G4_VQ_VECTOR);
    if (row >= m.rows) {
        #pragma unroll
        for (uint32_t i = 0; i < 16u; i++) { dst[i] = 0u; }
        return;
    }
    const uint64_t vector = (uint64_t)row * nvec + base + v0;
    if (m.precision == GLMVQ_E4M3) {
        uint2 w1[4];
        #pragma unroll
        for (uint32_t j = 0; j < 4u; j++) {
            w1[j] = __ldg((const uint2 *)m.book + g4_vq_index(m, vector + j));
        }
        uint32_t res = 0u;
        if (m.res) { res = __ldg((const uint32_t *)(m.res + vector)); }
        #pragma unroll
        for (uint32_t j = 0; j < 4u; j++) {
            const uint2 w2 = s_res ? s_res[(res >> (8u * j)) & 0xffu] : make_uint2(0u, 0u);
            dst[j * 4u + 0u] = g4_vqp_word(w1[j].x, w2.x, gain);
            dst[j * 4u + 1u] = g4_vqp_word(w1[j].x >> 16, w2.x >> 16, gain);
            dst[j * 4u + 2u] = g4_vqp_word(w1[j].y, w2.y, gain);
            dst[j * 4u + 3u] = g4_vqp_word(w1[j].y >> 16, w2.y >> 16, gain);
        }
    } else {
        #pragma unroll
        for (uint32_t j = 0; j < 4u; j++) {
            const uint32_t index = g4_vq_index(m, vector + j);
            const __half *c1 = (const __half *)m.book + (uint64_t)index * G4_VQ_VECTOR;
            const uint8_t *c2 = m.res ? m.res_book + (uint64_t)m.res[vector + j] * G4_VQ_VECTOR
                : nullptr;
            dst[j * 4u + 0u] = g4_vqp_pair_f16(c1 + 0, c2 ? c2 + 0 : nullptr, gain);
            dst[j * 4u + 1u] = g4_vqp_pair_f16(c1 + 2, c2 ? c2 + 2 : nullptr, gain);
            dst[j * 4u + 2u] = g4_vqp_pair_f16(c1 + 4, c2 ? c2 + 4 : nullptr, gain);
            dst[j * 4u + 3u] = g4_vqp_pair_f16(c1 + 6, c2 ? c2 + 6 : nullptr, gain);
        }
    }
}

/* Stage BN tokens x BK columns of the float activation rows as bf16. */
template <uint32_t BN, uint32_t BK, uint32_t THREADS>
__device__ __forceinline__ void g4_vqp_activation_tile(const float *x, uint32_t xstride,
        uint32_t t0, uint32_t tokens, uint32_t k0, __nv_bfloat16 *xs) {
    constexpr uint32_t STRIDE = BK + G4_VQP_PAD;
    for (uint32_t i = threadIdx.x; i < BN * BK; i += THREADS) {
        const uint32_t t = i / BK, k = i % BK;
        const uint32_t token = t0 + t;
        xs[t * STRIDE + k] = token < tokens
            ? __float2bfloat16_rn(x[(uint64_t)token * xstride + k0 + k])
            : __float2bfloat16_rn(0.0f);
    }
}

/* The compute micro-tile: G4_VQP_RT rows x G4_VQP_TT tokens per thread, K
 * walked in rounds of four with 2 x uint2 shared loads per row and per
 * token. The thread's accumulators live in registers across the whole K. */
template <uint32_t BK>
__device__ __forceinline__ void g4_vqp_compute(const __nv_bfloat16 *ws, const __nv_bfloat16 *xs,
        uint32_t r, uint32_t t, float acc[G4_VQP_RT][G4_VQP_TT]) {
    constexpr uint32_t STRIDE = BK + G4_VQP_PAD;
    for (uint32_t k = 0; k < BK; k += 4) {
        float w[G4_VQP_RT][4];
        float xv[G4_VQP_TT][4];
        #pragma unroll
        for (uint32_t i = 0; i < G4_VQP_RT; i++) {
            const uint2 v = *(const uint2 *)(ws + (r + i) * STRIDE + k);
            const float2 a = __bfloat1622float2(*(const __nv_bfloat162 *)&v.x);
            const float2 b = __bfloat1622float2(*(const __nv_bfloat162 *)&v.y);
            w[i][0] = a.x; w[i][1] = a.y; w[i][2] = b.x; w[i][3] = b.y;
        }
        #pragma unroll
        for (uint32_t j = 0; j < G4_VQP_TT; j++) {
            const uint2 v = *(const uint2 *)(xs + (t + j) * STRIDE + k);
            const float2 a = __bfloat1622float2(*(const __nv_bfloat162 *)&v.x);
            const float2 b = __bfloat1622float2(*(const __nv_bfloat162 *)&v.y);
            xv[j][0] = a.x; xv[j][1] = a.y; xv[j][2] = b.x; xv[j][3] = b.y;
        }
        #pragma unroll
        for (uint32_t i = 0; i < G4_VQP_RT; i++) {
            #pragma unroll
            for (uint32_t j = 0; j < G4_VQP_TT; j++) {
                #pragma unroll
                for (uint32_t d = 0; d < 4; d++) {
                    acc[i][j] = fmaf(w[i][d], xv[j][d], acc[i][j]);
                }
            }
        }
    }
}

/* Whether the tile kernel can run this geometry: the layout's own checks
 * (cols a multiple of the 8-weight vector), the tile's (cols a multiple of
 * BK), and the precision the decode supports. */
static int g4_vq_prefill_ready(const glmvq_matrix &m) {
    if (!m.main || !m.book || !m.gain || !m.rows || !m.cols) { return 0; }
    if ((m.cols % G4_VQP_BK) || (m.cols % G4_VQ_VECTOR)) { return 0; }
    if (m.precision != GLMVQ_E4M3 && m.precision != GLMVQ_F16) { return 0; }
    if (m.res && !m.res_book) { return 0; }
    return 1;
}

/* The layer pass: one block per work item. GLU decodes both matrices and
 * writes gelu(a) * b (the g4_gelu_kernel expression, cuda/gemma4_primitives.cuh:125-130);
 * otherwise it writes the rounded single product. */
template <bool GLU, uint32_t BM, uint32_t BN, uint32_t BK, uint32_t THREADS>
__global__ __launch_bounds__(THREADS, GLU ? 1 : 2) void g4_vq_gemm_items_kernel(
        const g4_vqp_item *items, uint32_t count, const glmvq_matrix *qmat,
        const float *x, uint32_t xstride, float *y, uint32_t ystride) {
    if (blockIdx.x >= count) { return; }
    const g4_vqp_item it = items[blockIdx.x];
    const glmvq_matrix a = qmat[it.a];
    const glmvq_matrix b = GLU ? qmat[it.b] : a;
    __shared__ __nv_bfloat16 wa[BM * (BK + G4_VQP_PAD)];
    __shared__ __nv_bfloat16 wb[GLU ? BM * (BK + G4_VQP_PAD) : 1];
    __shared__ __nv_bfloat16 xs[BN * (BK + G4_VQP_PAD)];
    __shared__ uint2 s_res[GLU ? 2 : 1][256];
    constexpr uint32_t GROUPS = BN / G4_VQP_TT;
    const uint32_t r = (threadIdx.x / GROUPS) * G4_VQP_RT;
    const uint32_t t = (threadIdx.x % GROUPS) * G4_VQP_TT;
    if (threadIdx.x < 256u) {
        if (a.res) { s_res[0][threadIdx.x] = __ldg((const uint2 *)a.res_book + threadIdx.x); }
        if (GLU && b.res) { s_res[1][threadIdx.x] = __ldg((const uint2 *)b.res_book + threadIdx.x); }
    }
    __syncthreads();
    const uint32_t drow = it.row0 + threadIdx.x / 2u;
    const float gain_a = drow < a.rows ? __half2float(a.gain[drow]) : 0.0f;
    const float gain_b = GLU ? (drow < b.rows ? __half2float(b.gain[drow]) : 0.0f) : 0.0f;
    float g[G4_VQP_RT][G4_VQP_TT], u[G4_VQP_RT][G4_VQP_TT];
    #pragma unroll
    for (uint32_t i = 0; i < G4_VQP_RT; i++) {
        #pragma unroll
        for (uint32_t j = 0; j < G4_VQP_TT; j++) { g[i][j] = 0.0f; u[i][j] = 0.0f; }
    }
    for (uint32_t k0 = 0; k0 < a.cols; k0 += BK) {
        g4_vqp_weight_tile<BM, BK, THREADS>(a, it.row0, k0, gain_a,
            a.res ? s_res[0] : nullptr, wa);
        if (GLU) {
            g4_vqp_weight_tile<BM, BK, THREADS>(b, it.row0, k0, gain_b,
                b.res ? s_res[1] : nullptr, wb);
        }
        g4_vqp_activation_tile<BN, BK, THREADS>(x, xstride, it.token0, it.end, k0, xs);
        __syncthreads();
        g4_vqp_compute<BK>(wa, xs, r, t, g);
        if (GLU) { g4_vqp_compute<BK>(wb, xs, r, t, u); }
        __syncthreads();
    }
    #pragma unroll
    for (uint32_t i = 0; i < G4_VQP_RT; i++) {
        const uint32_t row = it.row0 + r + i;
        if (row >= a.rows) { continue; }
        #pragma unroll
        for (uint32_t j = 0; j < G4_VQP_TT; j++) {
            const uint32_t token = it.token0 + t + j;
            if (token < it.end) {
                /* GLU writes the reference's epilogue exactly; the single
                 * matrix rounds to the BF16 boundary glmvq_gemm applies. */
                y[(uint64_t)token * ystride + row] = GLU
                    ? g4_round(g4_gelu(g[i][j]) * g4_round(u[i][j]))
                    : g4_round(g[i][j]);
            }
        }
    }
}

/* One direct VQ GEMM for a single matrix and the whole token range: the
 * verification entry (DS4_GEMMA4_VQ_VERIFY) and the one-matrix shape. */
template <uint32_t BM, uint32_t BN, uint32_t BK, uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 2) void g4_vq_gemm_kernel(glmvq_matrix m,
        const float *x, uint32_t xstride, float *y, uint32_t ystride, uint32_t tokens) {
    __shared__ __nv_bfloat16 ws[BM * (BK + G4_VQP_PAD)];
    __shared__ __nv_bfloat16 xs[BN * (BK + G4_VQP_PAD)];
    __shared__ uint2 s_res[256];
    const uint32_t r0 = blockIdx.x * BM;
    const uint32_t t0 = blockIdx.y * BN;
    constexpr uint32_t GROUPS = BN / G4_VQP_TT;
    const uint32_t r = (threadIdx.x / GROUPS) * G4_VQP_RT;
    const uint32_t t = (threadIdx.x % GROUPS) * G4_VQP_TT;
    if (m.res && threadIdx.x < 256u) {
        s_res[threadIdx.x] = __ldg((const uint2 *)m.res_book + threadIdx.x);
    }
    __syncthreads();
    const uint32_t drow = r0 + threadIdx.x / 2u;
    const float gain = drow < m.rows ? __half2float(m.gain[drow]) : 0.0f;
    float acc[G4_VQP_RT][G4_VQP_TT];
    #pragma unroll
    for (uint32_t i = 0; i < G4_VQP_RT; i++) {
        #pragma unroll
        for (uint32_t j = 0; j < G4_VQP_TT; j++) { acc[i][j] = 0.0f; }
    }
    for (uint32_t k0 = 0; k0 < m.cols; k0 += BK) {
        g4_vqp_weight_tile<BM, BK, THREADS>(m, r0, k0, gain, m.res ? s_res : nullptr, ws);
        g4_vqp_activation_tile<BN, BK, THREADS>(x, xstride, t0, tokens, k0, xs);
        __syncthreads();
        g4_vqp_compute<BK>(ws, xs, r, t, acc);
        __syncthreads();
    }
    #pragma unroll
    for (uint32_t i = 0; i < G4_VQP_RT; i++) {
        const uint32_t row = r0 + r + i;
        if (row >= m.rows) { continue; }
        #pragma unroll
        for (uint32_t j = 0; j < G4_VQP_TT; j++) {
            const uint32_t token = t0 + t + j;
            if (token < tokens) { y[(uint64_t)token * ystride + row] = g4_round(acc[i][j]); }
        }
    }
}

/* Launch one direct VQ GEMM over a single matrix. Returns 0 (nothing
 * written) when the geometry or the arguments are outside the tile. */
static int g4_vq_gemm(const glmvq_matrix &m, const float *x, uint32_t xstride,
        float *y, uint32_t ystride, uint32_t tokens, cudaStream_t stream) {
    if (!x || !y || !tokens || !g4_vq_prefill_ready(m)) { return 0; }
    if (xstride < m.cols || ystride < m.rows) { return 0; }
    const dim3 grid((m.rows + G4_VQP_BM - 1u) / G4_VQP_BM,
        (tokens + G4_VQP_BN - 1u) / G4_VQP_BN);
    g4_vq_gemm_kernel<G4_VQP_BM, G4_VQP_BN, G4_VQP_BK, G4_VQP_THREADS>
        <<<grid, G4_VQP_THREADS, 0, stream>>>(m, x, xstride, y, ystride, tokens);
    return cudaGetLastError() == cudaSuccess;
}

/* Launch the layer's work-item list. GLU selects the gate/up pair with the
 * fused epilogue; otherwise each item is one matrix (the down projection).
 * The caller owns the item and matrix buffers. */
static int g4_vq_gemm_items(const g4_vqp_item *items, uint32_t count,
        const glmvq_matrix *qmat, bool glu, const float *x, uint32_t xstride,
        float *y, uint32_t ystride, cudaStream_t stream) {
    if (!items || !count || !qmat || !x || !y) { return 0; }
    if (glu) {
        g4_vq_gemm_items_kernel<true, G4_VQP_BM, G4_VQP_BN, G4_VQP_BK, G4_VQP_THREADS>
            <<<count, G4_VQP_THREADS, 0, stream>>>(items, count, qmat, x, xstride, y, ystride);
    } else {
        g4_vq_gemm_items_kernel<false, G4_VQP_BM, G4_VQP_BN, G4_VQP_BK, G4_VQP_THREADS>
            <<<count, G4_VQP_THREADS, 0, stream>>>(items, count, qmat, x, xstride, y, ystride);
    }
    return cudaGetLastError() == cudaSuccess;
}

#endif
