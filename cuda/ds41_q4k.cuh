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
