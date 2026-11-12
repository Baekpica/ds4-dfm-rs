/* ds41_q4k.cuh — DeepSeek V4.1 (ds41) q4_K skeleton decode family, ported from
 * the C engine at /data/YoungAi (commit 3946dbc), file
 * src/cuda/cuda_v41_q4k.inc.cu (2026-09-19, the 100 GB recipe's skeleton).
 *
 * On-disk q4_K: 144 B / 256 elements — f16 d + f16 dmin + 12 B of 8 packed
 * 6-bit (scale, min) pairs + 128 B nibbles.  The golden decoder is
 * src/common/ds4_quantfmt.c ds4_deq_q4_K (byte-exact fixture in the engine's
 * unit tests); the kernels here mirror it expression for expression — the
 * block header packs 8 six-bit pairs into 12 B and a one-bit packing error
 * silently shifts a whole 32-element group by a constant.
 *
 * Lane layout (the engine's measured 64 B DRAM granule): lane l reads
 * qs[4l..4l+3] so one warp step is 128 contiguous bytes; byte p feeds
 * sub-block 2(p/32) (low nibble) and 2(p/32)+1 (high nibble), 32 elements
 * apart.  The GEMV pair (stage/pipe) stages the CTA's rows through shared
 * first — byte-identical to the retired per-warp reader (benchmarked on the
 * real shapes) — and selects between them by the group's row bytes.
 *
 * Scope: the n <= 8 GEMV arm, the embedding row fetch and the grouped (wo_a)
 * arm (P4-4), plus the n > 8 prefill GEMM (P4-5: decode to bf16 in row tiles,
 * then cuBLAS — cuda_v41_q4k.inc.cu:183-205 the to-bf16 kernel, :330-390 the
 * GEMM).  The head colnorm is not ported yet and refuses by name. */
#pragma once
#include <cuda_pipeline.h>   /* the pipe variant's cp.async intrinsics */

#define V41_Q4K_BLK 256u          /* elements per block */
#define V41_Q4K_BYTES 144u        /* bytes per block: 2(d) + 2(dmin) + 12(scales) + 128(qs) */
/* The GEMV's warp budget (`cuda_v41_4.inc.cu:17`); the q4k launchers were
 * tuned around it (parallelism target 8192, stage/pipe selection). */
#define V41_GEMV_WARPS 8u

/* The j-th (scale, min) pair of a block header; expression-identical to
 * ds4_quantfmt.c q4k_scale_min (cuda_v41_q4k.inc.cu:21-27). */
__device__ __forceinline__ static void v41_q4k_sm(const uint8_t *sc, int j, float *s, float *m) {
    uint8_t a, b;
    if (j < 4) { a = sc[j] & 63u; b = sc[j + 4] & 63u; }
    else { a = (uint8_t)((sc[j + 4] & 0xFu) | ((sc[j - 4] >> 6) << 4));
           b = (uint8_t)((sc[j + 4] >> 4)  | ((sc[j - 0] >> 6) << 4)); }
    *s = (float)a; *m = (float)b;
}

/* Header already in registers (one uint4): scale byte k of the 12-byte table.
 * NOT ((const uint8_t *)&h)[k] — k varies per lane and a dynamic index into a
 * register array drops it into local memory; this compiles to SEL/SHF. */
__device__ __forceinline__ static uint32_t v41_q4k_scb(const uint4 &h, uint32_t k) {
    const uint32_t w = k < 4u ? h.y : (k < 8u ? h.z : h.w);
    return (w >> ((k & 3u) * 8u)) & 0xFFu;
}
__device__ __forceinline__ static void v41_q4k_sm_reg(const uint4 &h, uint32_t j, float *s, float *m) {
    uint32_t a, b;
    if (j < 4u) { a = v41_q4k_scb(h, j) & 63u; b = v41_q4k_scb(h, j + 4u) & 63u; }
    else { a = (v41_q4k_scb(h, j + 4u) & 0xFu) | ((v41_q4k_scb(h, j - 4u) >> 6) << 4);
           b = (v41_q4k_scb(h, j + 4u) >> 4)   | ((v41_q4k_scb(h, j) >> 6) << 4); }
    *s = (float)a; *m = (float)b;
}

/* One warp dot-products one q4_K block against NT activations, accumulating
 * into acc[NT]; the header and qs word come from the caller (shared staging).
 * lane l: gidx = l>>3 selects a 64-element group, q0 = (l&7)*4 four
 * consecutive elements inside it; low half -> gidx*64 + q0 + i (sub-block
 * 2*gidx), high half -> +32 (sub-block 2*gidx+1). */
template <uint32_t NT>
__device__ __forceinline__ static void v41_q4k_blk_acc_reg(const uint4 &h, uint32_t qw, const float *x, uint32_t x_stride,
                                                           uint32_t base, float *acc) {
    const uint32_t lane = threadIdx.x & 31u, gidx = lane >> 3, q0 = (lane & 7u) * 4u;
    const float d = __half2float(__ushort_as_half((unsigned short)(h.x & 0xFFFFu)));
    const float dmin = __half2float(__ushort_as_half((unsigned short)(h.x >> 16)));
    float s_lo, m_lo, s_hi, m_hi;
    v41_q4k_sm_reg(h, gidx * 2u, &s_lo, &m_lo);
    v41_q4k_sm_reg(h, gidx * 2u + 1u, &s_hi, &m_hi);
    const float dl = d * s_lo, ml = dmin * m_lo, dh = d * s_hi, mh = dmin * m_hi;
    float wl[4], wh[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t byte = (qw >> (8 * i)) & 0xFFu;
        wl[i] = dl * (float)(byte & 0xFu) - ml;
        wh[i] = dh * (float)(byte >> 4) - mh;
    }
    const uint32_t e_lo = base + gidx * 64u + q0, e_hi = e_lo + 32u;
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        const float *xt = x + (uint64_t)t * x_stride;
        const float4 xl = *(const float4 *)(xt + e_lo), xh = *(const float4 *)(xt + e_hi);
        acc[t] += wl[0] * xl.x + wl[1] * xl.y + wl[2] * xl.z + wl[3] * xl.w
                + wh[0] * xh.x + wh[1] * xh.y + wh[2] * xh.z + wh[3] * xh.w;
    }
}

/* The stage/pipe pair (cuda_v41_q4k.inc.cu:72-179): byte-identical to the
 * retired per-warp reader on all nine real shapes (the engine's benchmark);
 * selection by the group's row bytes — stage for large groups (18-23 KB),
 * pipe (cp.async double buffer) for 4-12 KB groups. */
#define V41_Q4K_PIPE_MIN 4096u
#define V41_Q4K_PIPE_MAX 12288u

template <uint32_t NT>
__device__ __forceinline__ static void v41_q4k_rown_smem(const uint8_t *wr, uint32_t nblk, uint32_t ksplit, uint32_t kpart,
                                                         const float *x, uint32_t x_stride, float *acc) {
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t b = kpart; b < nblk; b += ksplit) {
        const uint8_t *blk = wr + b * V41_Q4K_BYTES;
        v41_q4k_blk_acc_reg<NT>(*(const uint4 *)blk, *(const uint32_t *)(blk + 16u + 4u * lane), x, x_stride, b * V41_Q4K_BLK, acc);
    }
}
template <uint32_t NT>
__device__ __forceinline__ static void v41_q4k_finishn(float *acc, float *out, uint32_t out_stride, uint32_t r, uint32_t out_dim,
                                                       uint32_t ksplit, uint32_t rloc, uint32_t kpart, int round_out, float *red) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        for (int o = 16; o; o >>= 1) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], o);
        if (lane == 0) red[warp * NT + t] = acc[t];
    }
    __syncthreads();
    if (r < out_dim && kpart == 0 && lane == 0) {
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            float sum = 0.f;
            for (uint32_t k = 0; k < ksplit; k++) sum += red[(rloc * ksplit + k) * NT + t];
            out[(uint64_t)t * out_stride + r] = round_out ? v41_bf16r(sum) : sum;
        }
    }
}
template <uint32_t NT>
__global__ static void __launch_bounds__(256, 4) v41_q4k_gemv1_stage_kernel(float *out, const uint8_t *w, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t ksplit, uint64_t w_gstride, uint32_t x_gstride, uint32_t out_gstride, int round_out,
        uint32_t x_stride, uint32_t out_stride) {
    extern __shared__ uint4 v41_q4k_st[];
    __shared__ float red[V41_GEMV_WARPS * NT];
    const uint32_t g = blockIdx.y;
    x += (uint64_t)g * x_gstride; out += (uint64_t)g * out_gstride; w += (uint64_t)g * w_gstride;
    const uint32_t warp = threadIdx.x >> 5, rpb = V41_GEMV_WARPS / ksplit, rloc = warp / ksplit, kpart = warp % ksplit;
    const uint32_t nblk = in_dim / V41_Q4K_BLK, r0 = blockIdx.x * rpb, r = r0 + rloc;
    const uint32_t nrows = out_dim - r0 < rpb ? out_dim - r0 : rpb, n16 = nrows * nblk * (V41_Q4K_BYTES / 16u);
    const uint4 *src = (const uint4 *)(w + (uint64_t)r0 * nblk * V41_Q4K_BYTES);
    for (uint32_t i = threadIdx.x; i < n16; i += blockDim.x) v41_q4k_st[i] = __ldcs(src + i);   /* streaming: weights read once per step */
    v41_pdl_wait();   /* PDL: weights (constants) staged first, then wait for the upstream activations */
    __syncthreads();
    float acc[NT];
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) acc[t] = 0.f;
    if (r < out_dim) v41_q4k_rown_smem<NT>((const uint8_t *)v41_q4k_st + (uint64_t)rloc * nblk * V41_Q4K_BYTES, nblk, ksplit, kpart, x, x_stride, acc);
    v41_q4k_finishn<NT>(acc, out, out_stride, r, out_dim, ksplit, rloc, kpart, round_out, red);
}
template <uint32_t NT>
__global__ static void __launch_bounds__(256, 4) v41_q4k_gemv1_pipe_kernel(float *out, const uint8_t *w, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t ksplit, uint64_t w_gstride, uint32_t x_gstride, uint32_t out_gstride, int round_out,
        uint32_t x_stride, uint32_t out_stride) {
    extern __shared__ uint4 v41_q4k_st[];
    __shared__ float red[V41_GEMV_WARPS * NT];
    const uint32_t g = blockIdx.y;
    x += (uint64_t)g * x_gstride; out += (uint64_t)g * out_gstride; w += (uint64_t)g * w_gstride;
    const uint32_t warp = threadIdx.x >> 5, rpb = V41_GEMV_WARPS / ksplit, rloc = warp / ksplit, kpart = warp % ksplit;
    const uint32_t nblk = in_dim / V41_Q4K_BLK, ngrp = (out_dim + rpb - 1u) / rpb, grp16 = rpb * nblk * (V41_Q4K_BYTES / 16u);
    uint32_t rg = blockIdx.x, buf = 0;
    #define V41_Q4K_ISSUE(RG, BUF) do {                                                                           \
        const uint32_t r0_ = (RG) * rpb, nr_ = out_dim - r0_ < rpb ? out_dim - r0_ : rpb;                          \
        const uint32_t n16_ = nr_ * nblk * (V41_Q4K_BYTES / 16u);                                                  \
        const uint4 *src_ = (const uint4 *)(w + (uint64_t)r0_ * nblk * V41_Q4K_BYTES);                            \
        uint4 *dst_ = v41_q4k_st + (uint64_t)(BUF) * grp16;                                                        \
        for (uint32_t i_ = threadIdx.x; i_ < n16_; i_ += blockDim.x) __pipeline_memcpy_async(dst_ + i_, src_ + i_, 16); \
    } while (0)
    if (rg < ngrp) V41_Q4K_ISSUE(rg, 0u);
    __pipeline_commit();
    v41_pdl_wait();   /* PDL: first group already in flight, then wait for the upstream activations */
    for (; rg < ngrp; rg += gridDim.x, buf ^= 1u) {
        if (rg + gridDim.x < ngrp) V41_Q4K_ISSUE(rg + gridDim.x, buf ^ 1u);
        __pipeline_commit();          /* an empty commit counts too: wait_prior(1) counts commits */
        __pipeline_wait_prior(1);     /* this group's half arrived (the next is still in flight) */
        __syncthreads();
        const uint32_t r = rg * rpb + rloc;
        float acc[NT];
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) acc[t] = 0.f;
        if (r < out_dim) v41_q4k_rown_smem<NT>((const uint8_t *)(v41_q4k_st + (uint64_t)buf * grp16) + (uint64_t)rloc * nblk * V41_Q4K_BYTES,
                                           nblk, ksplit, kpart, x, x_stride, acc);
        v41_q4k_finishn<NT>(acc, out, out_stride, r, out_dim, ksplit, rloc, kpart, round_out, red);
        __syncthreads();              /* this half is overwritten next round: everyone done reading */
    }
    #undef V41_Q4K_ISSUE
}

/* The q4_K GEMV launcher (cuda_v41_q4k.inc.cu:232-277): alignment hard-fails
 * (a uint32 qs read needs 4-byte alignment; block length 144 is a multiple of
 * 16 and GGUF data starts 32 B aligned, so it holds — refuse rather than run
 * a silent slow path), ksplit target 8192, stage/pipe by group bytes. */
static int v41_q4k_gemv(const void *model_map, uint64_t model_size, uint64_t off, uint64_t in_dim, uint64_t out_dim,
                        const float *x, uint32_t x_stride, float *out, uint32_t out_stride, uint32_t n_tok,
                        uint32_t n_groups, uint32_t x_gstride, uint32_t out_gstride, int round_out, const char *what) {
    if ((in_dim % V41_Q4K_BLK) != 0u || n_tok == 0 || n_tok > V41_GEMV_MAX_TOK || n_groups == 0) return 0;
    const uint64_t wg = out_dim * (in_dim / V41_Q4K_BLK) * V41_Q4K_BYTES, wbytes = wg * n_groups;
    if (off > model_size || wbytes > model_size - off) return 0;
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, off, wbytes, what);
    if (!w) return 0;
    if (((uintptr_t)w & 15u) != 0u) { fprintf(stderr, "ds4: %s q4_K tensor base is not 16-byte aligned\n", what); return 0; }
    uint32_t ksplit = 1;
    while (ksplit < V41_GEMV_WARPS && out_dim * ksplit * n_groups < 8192u) ksplit <<= 1;
    const uint32_t nb = (uint32_t)(in_dim / V41_Q4K_BLK);
    while (ksplit > 1u && ksplit > nb) ksplit >>= 1;
    const uint32_t rpb1 = V41_GEMV_WARPS / ksplit, ngrp = (uint32_t)((out_dim + rpb1 - 1u) / rpb1);
    const uint32_t grp = rpb1 * nb * V41_Q4K_BYTES;
    uint32_t gx = 0;
    if (grp > V41_Q4K_PIPE_MIN && grp <= V41_Q4K_PIPE_MAX) {
        static int s_sm = 0;
        if (!s_sm && cudaDeviceGetAttribute(&s_sm, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) return cuda_ok(cudaGetLastError(), what);
        gx = (uint32_t)s_sm * 4u / n_groups;   /* 4 = the CTA/SM count __launch_bounds__ pins */
        if (gx == 0u) gx = 1u;
        if (gx > ngrp) gx = ngrp;
    }
    /* Largest groups are the head / wo_b kind: 8 rows x 20 blocks or 4 x 32 = 23 KB, inside the 48 KB dynamic shared default. */
    #define V41_Q4K_LAUNCH(NT) do {                                                                                   \
        v41_pdl_register((const void *)v41_q4k_gemv1_stage_kernel<NT>);   /* both kernels v41_pdl_wait before x/out */ \
        v41_pdl_register((const void *)v41_q4k_gemv1_pipe_kernel<NT>);                                              \
        if (gx) v41_q4k_gemv1_pipe_kernel<NT><<<dim3(gx, n_groups), 256, 2u * grp, ds4_current_stream()>>>(          \
                    out, w, x, (uint32_t)in_dim, (uint32_t)out_dim, ksplit, wg, x_gstride, out_gstride, round_out, x_stride, out_stride); \
        else v41_q4k_gemv1_stage_kernel<NT><<<dim3(ngrp, n_groups), 256, grp, ds4_current_stream()>>>(               \
                    out, w, x, (uint32_t)in_dim, (uint32_t)out_dim, ksplit, wg, x_gstride, out_gstride, round_out, x_stride, out_stride); \
    } while (0)
    switch (n_tok) {
        case 1: V41_Q4K_LAUNCH(1u); break;  case 2: V41_Q4K_LAUNCH(2u); break;
        case 3: V41_Q4K_LAUNCH(3u); break;  case 4: V41_Q4K_LAUNCH(4u); break;
        case 5: V41_Q4K_LAUNCH(5u); break;  case 6: V41_Q4K_LAUNCH(6u); break;
        case 7: V41_Q4K_LAUNCH(7u); break;  default: V41_Q4K_LAUNCH(8u); break;
    }
    #undef V41_Q4K_LAUNCH
    return cuda_ok(cudaGetLastError(), what);
}

/* Embedding row fetch: token id -> that row as f32 (one warp per block). */
__global__ static void v41_q4k_embed_kernel(float *out, const int32_t *tok, const uint8_t *w,
                                            uint32_t n_vocab, uint32_t dim) {
    const uint32_t t = blockIdx.y, nblk = dim / V41_Q4K_BLK;
    const uint32_t b = blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= nblk) return;
    const int32_t id = tok[t];
    if (id < 0 || (uint32_t)id >= n_vocab) return;
    const uint8_t *blk = w + ((uint64_t)id * nblk + b) * V41_Q4K_BYTES;
    const uint32_t lane = threadIdx.x & 31u, gidx = lane >> 3, q0 = (lane & 7u) * 4u;
    uint16_t hd = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    uint16_t hm = (uint16_t)blk[2] | ((uint16_t)blk[3] << 8);
    const float d = __half2float(__ushort_as_half(hd)), dmin = __half2float(__ushort_as_half(hm));
    float s_lo, m_lo, s_hi, m_hi;
    v41_q4k_sm(blk + 4, (int)(gidx * 2u), &s_lo, &m_lo);
    v41_q4k_sm(blk + 4, (int)(gidx * 2u + 1u), &s_hi, &m_hi);
    const uint32_t qw = *(const uint32_t *)(blk + 16u + 4u * lane);
    float *o = out + (uint64_t)t * dim + b * V41_Q4K_BLK;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t byte = (qw >> (8 * i)) & 0xFFu;
        o[gidx * 64u + q0 + i]       = d * s_lo * (float)(byte & 0xFu) - dmin * m_lo;
        o[gidx * 64u + 32u + q0 + i] = d * s_hi * (float)(byte >> 4)  - dmin * m_hi;
    }
}

/* q4_K -> bf16 prefill decode (cuda_v41_q4k.inc.cu:183-205): one warp per
 * block, the lane layout identical to the GEMV's, so both paths decode the
 * same value bits. */
__global__ static void v41_q4k_to_bf16_kernel(__nv_bfloat16 *o, const uint8_t *w, uint64_t nblk) {
    const uint64_t b = (uint64_t)blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= nblk) return;
    const uint8_t *blk = w + b * V41_Q4K_BYTES;
    const uint32_t lane = threadIdx.x & 31u, gidx = lane >> 3, q0 = (lane & 7u) * 4u;
    uint16_t hd = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    uint16_t hm = (uint16_t)blk[2] | ((uint16_t)blk[3] << 8);
    const float d = __half2float(__ushort_as_half(hd)), dmin = __half2float(__ushort_as_half(hm));
    float s_lo, m_lo, s_hi, m_hi;
    v41_q4k_sm(blk + 4, (int)(gidx * 2u), &s_lo, &m_lo);
    v41_q4k_sm(blk + 4, (int)(gidx * 2u + 1u), &s_hi, &m_hi);
    const uint32_t qw = *(const uint32_t *)(blk + 16u + 4u * lane);
    __nv_bfloat16 *ob = o + b * V41_Q4K_BLK;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t byte = (qw >> (8 * i)) & 0xFFu;
        ob[gidx * 64u + q0 + i]       = __float2bfloat16(d * s_lo * (float)(byte & 0xFu) - dmin * m_lo);
        ob[gidx * 64u + 32u + q0 + i] = __float2bfloat16(d * s_hi * (float)(byte >> 4)  - dmin * m_hi);
    }
}

/* Prefill GEMM (cuda_v41_q4k.inc.cu:330-390): decode the tensor to bf16 in
 * row tiles bounded by V41_BF16_STAGE_ELEMS, then one cuBLAS bf16 GEMM per
 * tile.  Only the head (129280 rows) exceeds the cap; grouped wo_a (67 MB)
 * decodes whole once and issues one GEMM per block-diagonal group.
 * Deviation, named: the engine's v41_wc_* weight cache (a backward-pass
 * buffer, cuda_v41_q4k.inc.cu:350-353) has no port — `hit` is NULL on every
 * call here, so the cache branches are omitted.  The cache stores exactly the
 * bytes v41_q4k_to_bf16_kernel writes, so the arithmetic is identical. */
#define V41_BF16_STAGE_ELEMS (200ull * 1000000ull)   /* cuda_v41_1.inc.cu:27 */
static v41_scratch g_v41_q4k_wbf, g_v41_q4k_xbf;
static int v41_q4k_gemm(const void *model_map, uint64_t model_size, uint64_t off, uint64_t in_dim,
                        uint64_t out_dim, const float *x, float *out, uint32_t n_tok, uint32_t n_groups,
                        uint32_t x_stride, uint32_t out_stride, int round_out, const char *what) {
    (void)round_out;   /* rounding is the caller's (one whole-tensor pass when round_out) */
    const uint64_t bpr = in_dim / V41_Q4K_BLK, nblk_g = out_dim * bpr, nblk = nblk_g * n_groups;
    if (off > model_size || nblk * V41_Q4K_BYTES > model_size - off) return 0;
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, off, nblk * V41_Q4K_BYTES, what);
    if (!w) return 0;
    if (((uintptr_t)w & 3u) != 0u) { fprintf(stderr, "ds4: %s q4_K tensor start is not 4-byte aligned\n", what); return 0; }
    const uint64_t rows_cap = V41_BF16_STAGE_ELEMS / in_dim;
    const uint64_t tile = n_groups > 1u ? out_dim : (rows_cap < 256u ? 256u : (rows_cap >= out_dim ? out_dim : (rows_cap & ~255ull)));
    const uint64_t stage = n_groups > 1u ? nblk * V41_Q4K_BLK : tile * in_dim;
    __nv_bfloat16 *wb = (__nv_bfloat16 *)v41_grow(&g_v41_q4k_wbf, stage * sizeof(__nv_bfloat16), "v41 q4k w bf16");
    if (!wb) return 0;
    const uint64_t xn = (uint64_t)n_tok * x_stride;
    __nv_bfloat16 *xb = (__nv_bfloat16 *)v41_grow(&g_v41_q4k_xbf, xn * sizeof(__nv_bfloat16), "v41 q4k x bf16");
    if (!xb) return 0;
    v41_x_to_bf16_kernel<<<(unsigned)((xn + 255) / 256), 256, 0, ds4_current_stream()>>>(xb, x, xn);
    if (!cuda_ok(cudaGetLastError(), "v41 q4k x->bf16")) return 0;
    const float alpha = 1.0f, beta = 0.0f;
    const dim3 dblk(32, 8);
    (void)cublasSetStream(g_cublas, ds4_current_stream());   /* the engine's call (cuda_v41_q4k.inc.cu:359): cuBLAS keys its algorithm on the bound workspace, and this call resets it to the default pool (docs 2.4.7) -- the state the engine's GEMMs actually run in */
    cuda_cublas_ws_prep(ds4_current_stream());
    if (n_groups > 1u) {
        v41_q4k_to_bf16_kernel<<<(unsigned)((nblk + 7) / 8), dblk, 0, ds4_current_stream()>>>(wb, w, nblk);
        if (!cuda_ok(cudaGetLastError(), "v41 q4k->bf16")) return 0;
        for (uint32_t g = 0; g < n_groups; g++) {
            cublasStatus_t st = cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)out_dim, (int)n_tok, (int)in_dim, &alpha,
                                             wb + (uint64_t)g * nblk_g * V41_Q4K_BLK, CUDA_R_16BF, (int)in_dim,
                                             xb + (uint64_t)g * in_dim, CUDA_R_16BF, (int)x_stride, &beta,
                                             out + (uint64_t)g * out_dim, CUDA_R_32F, (int)out_stride,
                                             CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
            if (!cublas_ok(st, what)) return 0;
        }
        return 1;
    }
    for (uint64_t r0 = 0; r0 < out_dim; r0 += tile) {
        const uint64_t rows = out_dim - r0 < tile ? out_dim - r0 : tile, tb = rows * bpr;
        v41_q4k_to_bf16_kernel<<<(unsigned)((tb + 7) / 8), dblk, 0, ds4_current_stream()>>>(wb, w + r0 * bpr * V41_Q4K_BYTES, tb);
        if (!cuda_ok(cudaGetLastError(), "v41 q4k->bf16")) return 0;
        cublasStatus_t st = cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)rows, (int)n_tok, (int)in_dim, &alpha,
                                         wb, CUDA_R_16BF, (int)in_dim, xb, CUDA_R_16BF, (int)x_stride, &beta,
                                         out + r0, CUDA_R_32F, (int)out_stride, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, what)) return 0;
    }
    return 1;
}

/* Prefill projections on FP16 tensor cores (glmvq_q4p, cuda/glm_vq_q4p.cuh,
 * the GLM VQ family's kernel).  Why: the cuBLAS route above dequantizes the
 * whole weight to a BF16 copy on every call, converts the input and rounds
 * the output in separate passes, and its GEMMs run at 52-58% of the BF16 peak
 * (nsys, 64K hVQ prefill: 15 ms per layer for these seven projections).  The
 * q4p producer warps dequantize each Q4_K row once into MMA fragments; the
 * epilogue writes the BF16-rounded F32 value the round pass wrote.
 *
 * Numerics, a named deviation from the engine (bf16(w) x bf16(x) on cuBLAS):
 * weights round to F16 twice, fp16(fp16(q S) - M), half the error of one BF16
 * rounding against the exact Q4_K value; BF16-valued inputs are exact in
 * F16's normal range.  Rounded projections (round_out) take it, wo_a and
 * wo_b through their own entry; the head (unrounded logits) stays on
 * cuBLAS.  DS4_DS41_Q4P=0 keeps every projection on cuBLAS. */
struct ds41_bf16f { float v; };   /* an F32 slot holding the BF16-rounded value */
template<> __device__ __forceinline__ ds41_bf16f glmvq_cast<ds41_bf16f>(float value) { return {v41_bf16r(value)}; }

static int v41_q4p_on(void) {
    static int on = -1;
    if (on >= 0) { return on; }
    const char *env = getenv("DS4_DS41_Q4P");
    on = !(env && !strcmp(env, "0"));
    if (!on) { return 0; }
    if (glmvq_q4p_init() != cudaSuccess) { on = 0; }   /* the BF16-output instances (wo_a's planes) */
    const void *kernels[] = {(const void *)glmvq_q4p<true, ds41_bf16f>, (const void *)glmvq_q4p<false, ds41_bf16f>};
    for (const void *kernel : kernels) {
        if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GLMVQ_Q4P_SMEM) != cudaSuccess) { on = 0; }
    }
    (void)cudaGetLastError();
    fprintf(stderr, "ds4: [ds41] prefill q4_K projections %s\n", on ? "on FP16 tensor cores (glmvq_q4p)" : "*stay on cuBLAS: shared memory cannot be raised*");
    return on;
}

/* Whether a tensor's Q4_K scales fit q4p's folded F16 range
 * (glmvq_q4p_check): checked once per tensor, keyed by its device address.
 * The check reads back one flag: prefill blocks only, never under capture. */
#define V41_Q4P_SLOTS 1024u
static struct { const uint8_t *w; int ok; } g_v41_q4p_seen[V41_Q4P_SLOTS];
static v41_scratch g_v41_q4p_flag, g_v41_q4p_x;
static int v41_q4p_scales_fit(const uint8_t *w, uint64_t blocks) {
    uint32_t h = (uint32_t)(((uintptr_t)w >> 4) % V41_Q4P_SLOTS);
    for (uint32_t i = 0; i < V41_Q4P_SLOTS && g_v41_q4p_seen[h].w; i++, h = (h + 1u) % V41_Q4P_SLOTS) {
        if (g_v41_q4p_seen[h].w == w) { return g_v41_q4p_seen[h].ok; }
    }
    if (g_v41_q4p_seen[h].w) { return 0; }   /* table full: cuBLAS */

    uint32_t *flag = (uint32_t *)v41_grow(&g_v41_q4p_flag, sizeof(uint32_t), "v41 q4p check");
    if (!flag) { return 0; }
    const cudaStream_t stream = ds4_current_stream();
    uint32_t bad = 1;
    (void)cudaMemsetAsync(flag, 0, sizeof(uint32_t), stream);
    glmvq_q4p_check<<<(unsigned)((blocks + 255u) / 256u), 256, 0, stream>>>(w, blocks, flag);
    (void)cudaMemcpyAsync(&bad, flag, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    const int ok = cudaStreamSynchronize(stream) == cudaSuccess && !bad;
    (void)cudaGetLastError();
    if (!ok) { fprintf(stderr, "ds4: [ds41] a q4_K tensor's scales leave q4p's F16 range; it stays on cuBLAS\n"); }
    g_v41_q4p_seen[h].w = w;
    g_v41_q4p_seen[h].ok = ok;
    return ok;
}

/* A q4_K matrix q4p can take (`groups` of rows x cols, contiguous): its
 * device address in *w.  1 taken, 0 failed, -1 not taken (cuBLAS runs). */
static int v41_q4p_weight(const void *model_map, uint64_t model_size, uint64_t off, uint64_t cols, uint64_t rows,
                          uint32_t groups, const uint8_t **w) {
    if (!v41_q4p_on() || cols > UINT32_MAX || rows > UINT32_MAX || !glmvq_q4p_fits((uint32_t)rows, (uint32_t)cols)) { return -1; }
    const uint64_t blocks = groups * rows * (cols / V41_Q4K_BLK);
    if (off > model_size || blocks * V41_Q4K_BYTES > model_size - off) { return 0; }
    *w = (const uint8_t *)cuda_model_range_ptr(model_map, off, blocks * V41_Q4K_BYTES, "v41 q4p");
    if (!*w) { return 0; }
    /* The producers and the check load 16-byte pieces. */
    return ((uintptr_t)*w & 15u) || !v41_q4p_scales_fit(*w, blocks) ? -1 : 1;
}

/* x [n][cols] (BF16-valued F32) as q4p's FP16 operand in `scratch`. */
static int v41_q4p_input(v41_scratch *scratch, const float *x, uint32_t n_tok, uint32_t cols, glmvq_q4p_input *in) {
    void *p = v41_grow(scratch, glmvq_q4p_bytes(n_tok, cols), "v41 q4p x");
    if (!p) { return 0; }
    *in = glmvq_q4p_layout(p, n_tok, cols);
    return cuda_ok(glmvq_q4p_prepare(*in, x, n_tok, cols, ds4_current_stream()), "v41 q4p prep");
}

static int v41_q4p_run(const uint8_t *w, const glmvq_q4p_input &in, float *out, uint64_t rows, uint64_t cols, uint32_t n_tok) {
    return cuda_ok(glmvq_q4p_gemm(w, in, (ds41_bf16f *)out, (uint32_t)rows, (uint32_t)cols, n_tok, ds4_current_stream()), "v41 q4p");
}

/* One projection on q4p: 1 done, 0 failed, -1 not taken (cuBLAS runs). */
static int v41_q4p_gemm(const void *model_map, uint64_t model_size, uint64_t off, uint64_t in_dim, uint64_t out_dim,
                        const float *x, float *out, uint32_t n_tok) {
    const uint8_t *w = NULL;
    const int take = v41_q4p_weight(model_map, model_size, off, in_dim, out_dim, 1u, &w);
    if (take <= 0) { return take; }
    glmvq_q4p_input in;
    return v41_q4p_input(&g_v41_q4p_x, x, n_tok, (uint32_t)in_dim, &in) && v41_q4p_run(w, in, out, out_dim, in_dim, n_tok);
}

/* Two projections of one input (kv and q_a, the shared gate and up): one
 * prepared operand for both.  1 done, 0 failed, -1 not taken. */
extern "C" int ds4_gpu_v41_matmul_q4k_pair_tensor(ds4_gpu_tensor *out_a, uint64_t off_a, uint64_t dim_a,
                                                  ds4_gpu_tensor *out_b, uint64_t off_b, uint64_t dim_b,
                                                  const void *model_map, uint64_t model_size, uint64_t in_dim,
                                                  const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (n_tok <= V41_GEMV_MAX_TOK || !out_a || !out_b || !x || x->bytes < (uint64_t)n_tok * in_dim * 4 ||
        out_a->bytes < (uint64_t)n_tok * dim_a * 4 || out_b->bytes < (uint64_t)n_tok * dim_b * 4) { return -1; }
    const uint8_t *wa = NULL, *wb = NULL;
    const int ta = v41_q4p_weight(model_map, model_size, off_a, in_dim, dim_a, 1u, &wa);
    const int tb = ta > 0 ? v41_q4p_weight(model_map, model_size, off_b, in_dim, dim_b, 1u, &wb) : ta;
    if (ta <= 0 || tb <= 0) { return ta < 0 || tb < 0 ? -1 : 0; }
    glmvq_q4p_input in;
    return v41_q4p_input(&g_v41_q4p_x, (const float *)x->ptr, n_tok, (uint32_t)in_dim, &in) &&
           v41_q4p_run(wa, in, (float *)out_a->ptr, dim_a, in_dim, n_tok) &&
           v41_q4p_run(wb, in, (float *)out_b->ptr, dim_b, in_dim, n_tok);
}

/* q4p's prep (glmvq_q4p_prep) over pieces: operand row (token t = x, group
 * g = y) is `segs` pieces of `seg` values, piece s at
 * in + g in_group + t in_token + s in_seg; it lands at
 * xh + g out_group + t segs seg with its scale at xscale[g n + t]. */
template <typename T>
__global__ static void __launch_bounds__(GLMVQ_Q4P_PREP) v41_q4p_prep_seg(const T *in, uint64_t in_token, uint64_t in_group,
        uint64_t in_seg, uint32_t segs, uint32_t seg, __half *xh, float *xscale, uint64_t out_group, uint32_t n_tok) {
    __shared__ float peaks[GLMVQ_Q4P_PREP / 32];
    const uint32_t t = blockIdx.x, g = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const T *x = in + g * in_group + t * in_token;
    __half *out = xh + g * out_group + (uint64_t)t * segs * seg;
    float peak = 0.0f;
    for (uint32_t s = 0; s < segs; s++) { peak = fmaxf(peak, glmvq_q4p_row(x + s * in_seg, out + s * seg, seg / 4, 1.0f)); }
    for (uint32_t k = 16; k; k >>= 1) { peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, k)); }
    if (!lane) { peaks[warp] = peak; }
    __syncthreads();
    peak = fmaxf(fmaxf(peaks[0], peaks[1]), fmaxf(peaks[2], peaks[3]));
    int e = 0;
    if (peak > GLMVQ_Q4P_RANGE) {
        frexpf(peak / GLMVQ_Q4P_RANGE, &e);
        for (uint32_t s = 0; s < segs; s++) { glmvq_q4p_row(x + s * in_seg, out + s * seg, seg / 4, ldexpf(1.0f, -e)); }
    }
    if (!threadIdx.x) { xscale[(uint64_t)g * n_tok + t] = ldexpf(1.0f, e - GLMVQ_Q4P_SHIFT); }
}

/* The attention output pair on q4p (n > 8): wo_a, block-diagonal (`groups`
 * of group_dim -> rank), then wo_b (groups rank -> out_dim).  The heads are
 * prepared as one operand plane per group, each group's BF16 output lands
 * in its own plane, and wo_b's prep reads the planes back as rows.  Same
 * values as the two projections one by one: wo_a's output is the BF16 the
 * round pass wrote.  1 done, 0 failed, -1 not taken. */
static v41_scratch g_v41_q4p_planes, g_v41_q4p_x2;
extern "C" int ds4_gpu_v41_out_proj_q4k_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                               uint64_t off_a, uint64_t off_b, uint32_t groups, uint64_t group_dim,
                                               uint64_t rank, uint64_t out_dim, const ds4_gpu_tensor *heads, uint32_t n_tok) {
    const uint64_t in_all = groups * group_dim, low = groups * rank;
    if (n_tok <= V41_GEMV_MAX_TOK || !out || !heads || heads->bytes < (uint64_t)n_tok * in_all * 4 ||
        out->bytes < (uint64_t)n_tok * out_dim * 4 || (rank % 4u)) { return -1; }
    const uint8_t *wa = NULL, *wb = NULL;
    const int ta = v41_q4p_weight(model_map, model_size, off_a, group_dim, rank, groups, &wa);
    const int tb = ta > 0 ? v41_q4p_weight(model_map, model_size, off_b, low, out_dim, 1u, &wb) : ta;
    if (ta <= 0 || tb <= 0) { return ta < 0 || tb < 0 ? -1 : 0; }

    const cudaStream_t stream = ds4_current_stream();
    void *xa = v41_grow(&g_v41_q4p_x, glmvq_q4p_bytes(n_tok * groups, (uint32_t)group_dim), "v41 q4p x");
    __nv_bfloat16 *planes = (__nv_bfloat16 *)v41_grow(&g_v41_q4p_planes, (uint64_t)n_tok * low * 2u, "v41 q4p wo_a planes");
    void *xb = v41_grow(&g_v41_q4p_x2, glmvq_q4p_bytes(n_tok, (uint32_t)low), "v41 q4p wo_b x");
    if (!xa || !planes || !xb) { return 0; }

    /* Group g's operand: plane g of [groups][n][group_dim]. */
    const glmvq_q4p_input ia = glmvq_q4p_layout(xa, n_tok * groups, (uint32_t)group_dim);
    v41_q4p_prep_seg<<<dim3(n_tok, groups), GLMVQ_Q4P_PREP, 0, stream>>>((const float *)heads->ptr, in_all, group_dim, 0u, 1u,
        (uint32_t)group_dim, ia.xh, ia.xscale, (uint64_t)n_tok * group_dim, n_tok);
    if (!cuda_ok(cudaGetLastError(), "v41 q4p wo_a prep")) { return 0; }
    const uint64_t wg = rank * (group_dim / V41_Q4K_BLK) * V41_Q4K_BYTES;
    for (uint32_t g = 0; g < groups; g++) {
        const glmvq_q4p_input in = {ia.xh + (uint64_t)g * n_tok * group_dim, ia.xscale + (uint64_t)g * n_tok};
        if (!cuda_ok(glmvq_q4p_gemm(wa + g * wg, in, planes + (uint64_t)g * n_tok * rank, (uint32_t)rank,
                                    (uint32_t)group_dim, n_tok, stream), "v41 q4p wo_a")) { return 0; }
    }

    /* wo_b's operand row t: the groups' planes at row t, in group order. */
    const glmvq_q4p_input ib = glmvq_q4p_layout(xb, n_tok, (uint32_t)low);
    v41_q4p_prep_seg<<<dim3(n_tok, 1), GLMVQ_Q4P_PREP, 0, stream>>>((const __nv_bfloat16 *)planes, rank, 0u, (uint64_t)n_tok * rank,
        groups, (uint32_t)rank, ib.xh, ib.xscale, 0u, n_tok);
    return cuda_ok(cudaGetLastError(), "v41 q4p wo_b prep") && v41_q4p_run(wb, ib, (float *)out->ptr, out_dim, low, n_tok);
}

/* The q4_K entries (cuda_v41_q4k.inc.cu:392-433).  n <= 8 takes the GEMV,
 * n > 8 the prefill GEMM above. */
extern "C" int ds4_gpu_v41_matmul_q4k_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                             uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                             const ds4_gpu_tensor *x, uint32_t n_tok, int round_out) {
    if (!out || !x || !g_cublas_ready || n_tok == 0) return 0;
    if (x->bytes < (uint64_t)n_tok * in_dim * 4 || out->bytes < (uint64_t)n_tok * out_dim * 4) return 0;
    if (n_tok <= V41_GEMV_MAX_TOK) {
        /* The GEMV finish folds round_out in (engine cuda_v41_q4k.inc.cu:402-403
         * returns straight from this arm) -- no second round launch.  This arm
         * used to fall through to the trailing round below, rounding an
         * already-bf16 buffer with one extra kernel per site (328 per decode
         * step; the node diff against the engine, docs/ds41-perf-2026-10-11.md). */
        return v41_q4k_gemv(model_map, model_size, weight_offset, in_dim, out_dim, (const float *)x->ptr, (uint32_t)in_dim,
                            (float *)out->ptr, (uint32_t)out_dim, n_tok, 1u, 0u, 0u, round_out, "v41 q4k gemv");
    }
    if (round_out) {
        const int q4p = v41_q4p_gemm(model_map, model_size, weight_offset, in_dim, out_dim, (const float *)x->ptr,
                                     (float *)out->ptr, n_tok);
        if (q4p >= 0) { return q4p; }
    }
    if (!v41_q4k_gemm(model_map, model_size, weight_offset, in_dim, out_dim, (const float *)x->ptr,
                      (float *)out->ptr, n_tok, 1u, (uint32_t)in_dim, (uint32_t)out_dim, round_out, "v41 q4k gemm")) {
        return 0;
    }
    return round_out ? ds4_gpu_v41_round_bf16_tensor(out, (uint64_t)n_tok * out_dim) : 1;   /* GEMM arm only (engine :410-412) */
}
extern "C" int ds4_gpu_v41_grouped_matmul_q4k_tensor(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
                                                     uint64_t weight_offset, uint32_t n_groups, uint64_t group_dim,
                                                     uint64_t rank, const ds4_gpu_tensor *heads, uint32_t n_tok, int round_out) {
    if (!low || !heads || !g_cublas_ready || n_tok == 0) return 0;
    const uint64_t in_all = (uint64_t)n_groups * group_dim, out_all = (uint64_t)n_groups * rank;
    if (heads->bytes < (uint64_t)n_tok * in_all * 4 || low->bytes < (uint64_t)n_tok * out_all * 4) return 0;
    if (n_tok <= V41_GEMV_MAX_TOK) {
        if (group_dim % V41_Q4K_BLK) return 0;
        /* Same early return as the single-group entry (engine :423-425). */
        return v41_q4k_gemv(model_map, model_size, weight_offset, group_dim, rank, (const float *)heads->ptr, (uint32_t)in_all,
                            (float *)low->ptr, (uint32_t)out_all, n_tok, n_groups, (uint32_t)group_dim, (uint32_t)rank, round_out, "v41 wo_a q4k gemv");
    }
    if (!v41_q4k_gemm(model_map, model_size, weight_offset, group_dim, rank, (const float *)heads->ptr,
                      (float *)low->ptr, n_tok, n_groups, (uint32_t)in_all, (uint32_t)out_all, round_out, "v41 wo_a q4k gemm")) {
        return 0;
    }
    return round_out ? ds4_gpu_v41_round_bf16_tensor(low, (uint64_t)n_tok * out_all) : 1;   /* GEMM arm only */
}
extern "C" int ds4_gpu_v41_embed_q4k_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens, const void *model_map,
                                            uint64_t model_size, uint64_t weight_offset, uint64_t n_vocab,
                                            uint32_t n_tok, uint64_t dim) {
    if (!out || !tokens || n_tok == 0 || (dim % V41_Q4K_BLK)) return 0;
    const uint64_t nblk = n_vocab * (dim / V41_Q4K_BLK);
    if (weight_offset > model_size || nblk * V41_Q4K_BYTES > model_size - weight_offset) return 0;
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset, nblk * V41_Q4K_BYTES, "v41 q4k embed");
    if (!w || ((uintptr_t)w & 3u) != 0u) return 0;
    const uint32_t nb = (uint32_t)(dim / V41_Q4K_BLK);
    const dim3 dblk(32, 8), grid((nb + 7u) / 8u, n_tok);
    v41_q4k_embed_kernel<<<grid, dblk, 0, ds4_current_stream()>>>((float *)out->ptr, (const int32_t *)tokens->ptr, w,
                                                                  (uint32_t)n_vocab, (uint32_t)dim);
    return cuda_ok(cudaGetLastError(), "v41 q4k embed");
}
