/* ds41_vq_decode.cuh — VQ expert "decode is the multiply" kernels, ported from
 * the C engine at /data/YoungAi (commit 3946dbc), file
 * `src/cuda/cuda_vq_decode.inc.cu` (whole file).
 *
 * One subject: how compressed expert weights take part in the multiply
 * directly. On disk (DQVL v2 blob): 16 B header + 384*3 slot table; each slot
 * is a DQVQ payload = 16 B header + codebook (nc*8 f16; nc4096 = 64 KB) + row
 * gains f16 + bitstream (one codeword per 8 elements). The index width comes
 * from nc: nc4096 = 12 bit, nc2048 = 11 bit. The v41_vq_row_dot family is
 * NBIT-templated and geometrically general: a block is always 8 rounds =
 * 8*NBIT words, 3 registers per lane, row starts always whole bytes.
 *
 * Measured history kept from the engine: activation stored as bf16 (the
 * activation was two thirds of the load traffic; 14.23 -> 13.28 ms/step,
 * bit-identical because values sit on the bf16 grid); whole-block bitstream
 * reads (the three earlier read schemes all requested only 48 B per round —
 * DRAM fetches 64 B, half wasted). The 20 B-stride bank-conflict experiment
 * and the "dedupe experts per token" batch experiment were both measured
 * negative and are archived in the engine, not resurrected here.
 *
 * Must follow ds41_vq_row.cuh (uses its v41_vq_mat / row_dot) and
 * ds41_vq_probe.cuh (the prof watchdogs); ds41_vq_group.cuh and
 * ds41_vq_persist.cuh follow this file and are instantiated from here.
 */
#pragma once
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "ds41_primitives.cuh"
#include "ds41_vq_row.cuh"

/* Copy the codebook into shared with 8 loads in flight (each one previously
 * waited a full L2 round trip; the copy is a full-block sync point with nothing
 * to hide it). The codebook is only 8 B aligned, so uint2. bytes % 8 == 0. */
__device__ __forceinline__ static void v41_vq_cb_to_shared(uint8_t *dst, const uint8_t *src, uint32_t bytes) {
    const uint2 *s = (const uint2 *)src; uint2 *d = (uint2 *)dst;
    const uint32_t n = bytes / 8u;
    uint32_t i = threadIdx.x;
    for (; i + 7u * blockDim.x < n; i += 8u * blockDim.x) {
        uint2 t[8];
        #pragma unroll
        for (int q = 0; q < 8; q++) t[q] = s[i + (uint32_t)q * blockDim.x];
        #pragma unroll
        for (int q = 0; q < 8; q++) d[i + (uint32_t)q * blockDim.x] = t[q];
    }
    for (; i < n; i += blockDim.x) d[i] = s[i];
}

/* SwiGLU exit, same form as the official Expert: clamp -> silu(g)*u -> bf16.
 * g and u are already on the bf16 grid. */
__device__ __forceinline__ static uint16_t v41_vq_swiglu(float gi, float ui, float clamp) {
    if (clamp > 0.f) { if (gi > clamp) gi = clamp; if (ui > clamp) ui = clamp; if (ui < -clamp) ui = -clamp; }
    const float sg = gi / (1.0f + expf(-gi));
    return (uint16_t)(__float_as_uint(v41_bf16r(sg * ui)) >> 16);
}

/* Rows per block. One block = 32 warps x ITERS rows so the codebook is copied
 * once per block; with ITERS=8 the grid degenerates to ~1 wave per SM and the
 * tail dominates, with ITERS=2 (64 rows/block, 4.5 waves) the scheduler keeps
 * feeding finished SMs — measured 58.7 vs 63.0 ms. gateup and down were
 * re-scanned separately after the block-pipeline change (gateup 2 rows 6.72 <
 * 4 rows 7.19 < 8 rows 8.77; down 4 rows 3.76 < 2 rows 4.51). */
#define V41_VQ_ITERS_GU 2u
#define V41_VQ_ITERS_DN 4u
#define V41_VQ_WARPS    32u
#define V41_VQ_GU_ROWS  (V41_VQ_WARPS * V41_VQ_ITERS_GU)
#define V41_VQ_DN_ROWS  (V41_VQ_WARPS * V41_VQ_ITERS_DN)

/* gate/up in one kernel (official Expert: w1/w3 -> bf16 -> f32 -> silu(g)*u ->
 * bf16); cb_bytes > 0: codebook into shared (grid.x over 32-row units), else
 * global gather (grid.x over 8-row units). x is already on the bf16 grid.
 * Do not touch this kernel lightly: 1024 threads x 64 registers x 64 KB
 * codebook exactly fills one SM (1 block/SM, 67% occupancy); one extra runtime
 * possibility cost 21% even on the n=1 path with byte-identical output. */
template <int NBIT, int V3, int EXT, int SORTED>
__global__ static void v41_vq_gateup_kernel(uint16_t *h, const uint8_t *blob, const int32_t *sel, const uint32_t *x,
                                            uint32_t IN, uint32_t MID, uint32_t K, float clamp, uint32_t cb_bytes, const int32_t *order,
                                            uint32_t np, uint32_t uniq_only) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const uint32_t pair = SORTED ? (uint32_t)order[blockIdx.y] : blockIdx.y, t = pair / K, rows = cb_bytes ? V41_VQ_GU_ROWS : 8u;
    const int32_t e = sel[pair];
    if (e < 0) return;
    if (SORTED && uniq_only &&
        ((blockIdx.y > 0u && sel[order[blockIdx.y - 1u]] == e) || (blockIdx.y + 1u < np && sel[order[blockIdx.y + 1u]] == e))) return;
    const v41_vq_mat mg = v41_vq_open<V3>(blob, e, 0, MID, IN, NULL), mu = v41_vq_open<V3>(blob, e, 1, MID, IN, NULL);
    if (!mg.ok || !mu.ok) return;
    const uint32_t r0 = blockIdx.x * rows + (threadIdx.x >> 5), nit = cb_bytes ? V41_VQ_ITERS_GU : 1u;
    const uint32_t *xs = x + (uint64_t)t * (IN / 2u);   /* two bf16 per word */
    uint16_t *hp = h + (uint64_t)pair * MID;
    const bool lead = (threadIdx.x & 31u) == 0u;
    if (cb_bytes) {
        /* One shared buffer serves twice: gate and up each carry a codebook;
         * both at once exceeds the per-block shared limit, which would push the
         * whole kernel back to global gather. Load gate's, compute this block's
         * rows, sync, overwrite with up's, compute u. No return inside the loop
         * (a missing warp would deadlock the __syncthreads); out-of-range rows
         * just skip the compute. The gate result parks as bf16 in h's own row
         * slot — it is already on the bf16 grid, so the round trip is lossless
         * and the same lane writes and reads it back. */
        constexpr uint32_t CW = V3 ? 8u : 16u;   /* bytes per codeword: v3 = 8 E4M3, v2 = 8 f16 */
        if (mg.nc * CW != cb_bytes || mu.nc * CW != cb_bytes) return;
        v41_vq_blk carry; carry.w0 = 0u; carry.w1 = 0u; carry.w2 = 0u; carry.ex = 0u;
        if (r0 < MID) carry = v41_vq_row_first_blk<NBIT, V3, EXT>(mg, r0);   /* first block issues behind the codebook copy */
        v41_vq_cb_to_shared(vqsh, mg.cb, cb_bytes);
        __syncthreads();
        for (uint32_t i = 0; i < nit; i++) {
            const uint32_t r = r0 + i * V41_VQ_WARPS;
            if (r >= MID) break;
            const uint32_t rn = r + V41_VQ_WARPS;   /* on gate's last row, preload up's first (computed after the swap) */
            const int own = (i + 1u < nit && rn < MID);
            const uint32_t *next = own ? v41_vq_row_ptr<NBIT, V3>(mg, rn) : v41_vq_row_ptr<NBIT, V3>(mu, r0);
            const uint32_t *nextex = EXT ? (own ? v41_vq_ext_ptr(mg, rn) : v41_vq_ext_ptr(mu, r0)) : NULL;
            const float gv = v41_bf16r(v41_vq_row_dot<NBIT, V3, EXT>(mg, r, xs, vqsh, 1, &carry, next, nextex));
            if (lead) hp[r] = (uint16_t)(__float_as_uint(gv) >> 16);
        }
        __syncthreads();                            /* every warp done with gate's codebook before overwriting it */
        v41_vq_cb_to_shared(vqsh, mu.cb, cb_bytes);
        __syncthreads();
        for (uint32_t i = 0; i < nit; i++) {
            const uint32_t r = r0 + i * V41_VQ_WARPS;
            if (r >= MID) break;
            const uint32_t rn = r + V41_VQ_WARPS;
            const int own = (i + 1u < nit && rn < MID);
            const uint32_t *next = own ? v41_vq_row_ptr<NBIT, V3>(mu, rn) : NULL;
            const float ui = v41_bf16r(v41_vq_row_dot<NBIT, V3, EXT>(mu, r, xs, vqsh, 1, &carry, next, EXT && own ? v41_vq_ext_ptr(mu, rn) : NULL));
            if (lead) hp[r] = v41_vq_swiglu(__uint_as_float((uint32_t)hp[r] << 16), ui, clamp);
        }
    } else if (r0 < MID) {
        v41_vq_blk carry = v41_vq_row_first_blk<NBIT, V3, EXT>(mg, r0);
        const float gv = v41_bf16r(v41_vq_row_dot<NBIT, V3, EXT>(mg, r0, xs, mg.cb, 0, &carry,
                                   v41_vq_row_ptr<NBIT, V3>(mu, r0), EXT ? v41_vq_ext_ptr(mu, r0) : NULL));
        const float ui = v41_bf16r(v41_vq_row_dot<NBIT, V3, EXT>(mu, r0, xs, mu.cb, 0, &carry, NULL, NULL));
        if (lead) hp[r0] = v41_vq_swiglu(gv, ui, clamp);
    }
}

/* down: partial[pair][OUT] = bf16(W2 * h). */
template <int NBIT, int V3, int EXT, int SORTED>
__global__ static void v41_vq_down_kernel(float *partial, const uint8_t *blob, const int32_t *sel, const uint32_t *h,
                                          uint32_t MID, uint32_t OUT, uint32_t K, uint32_t cb_bytes, const float *gr, const int32_t *order,
                                          uint32_t np, uint32_t uniq_only) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const uint32_t pair = SORTED ? (uint32_t)order[blockIdx.y] : blockIdx.y, rows = cb_bytes ? V41_VQ_DN_ROWS : 8u;
    const int32_t e = sel[pair];
    if (e < 0) return;
    if (SORTED && uniq_only &&
        ((blockIdx.y > 0u && sel[order[blockIdx.y - 1u]] == e) || (blockIdx.y + 1u < np && sel[order[blockIdx.y + 1u]] == e))) return;
    const v41_vq_mat md = v41_vq_open<V3>(blob, e, 2, OUT, MID, gr ? gr + (size_t)e * OUT : NULL);
    const uint32_t r0 = blockIdx.x * rows + (threadIdx.x >> 5), nit = cb_bytes ? V41_VQ_ITERS_DN : 1u;
    if (!md.ok) {   /* bad payload: this block's rows all write 0 (not just one — the reduce would read stale values) */
        for (uint32_t i = 0; i < nit; i++) { const uint32_t r = r0 + i * V41_VQ_WARPS; if (r < OUT && (threadIdx.x & 31u) == 0) partial[(uint64_t)pair * OUT + r] = 0.f; }
        return;
    }
    const uint8_t *cbd = md.cb;
    v41_vq_blk carry; carry.w0 = 0u; carry.w1 = 0u; carry.w2 = 0u; carry.ex = 0u;
    if (r0 < OUT) carry = v41_vq_row_first_blk<NBIT, V3, EXT>(md, r0);
    if (cb_bytes) {
        if (md.nc * (V3 ? 8u : 16u) != cb_bytes) return;
        v41_vq_cb_to_shared(vqsh, md.cb, cb_bytes);
        __syncthreads();
        cbd = vqsh;
    }
    const uint32_t *hs = h + (uint64_t)pair * (MID / 2u);
    for (uint32_t i = 0; i < nit; i++) {
        const uint32_t r = r0 + i * V41_VQ_WARPS;
        if (r >= OUT) break;
        const uint32_t rn = r + V41_VQ_WARPS;
        const int own = (i + 1u < nit && rn < OUT);
        const uint32_t *next = own ? v41_vq_row_ptr<NBIT, V3>(md, rn) : NULL;
        const float y = v41_bf16r(v41_vq_row_dot<NBIT, V3, EXT>(md, r, hs, cbd, cb_bytes != 0, &carry, next,
                                  EXT && own ? v41_vq_ext_ptr(md, rn) : NULL));
        if ((threadIdx.x & 31u) == 0) partial[(uint64_t)pair * OUT + r] = y;
    }
    (void)K;
}

/* out[t][o] = sum_k w[t][k] * partial[t*K+k][o] (f32; the official y += weights * expert_out). */
__global__ static void v41_vq_reduce_kernel(float *out, const float *partial, const float *w, uint32_t K, uint32_t OUT) {
    const uint32_t t = blockIdx.y, o = blockIdx.x * 256u + threadIdx.x;
    if (o >= OUT) return;
    float a = 0.f;
    for (uint32_t k = 0; k < K; k++) a += w[(uint64_t)t * K + k] * partial[((uint64_t)t * K + k) * OUT + o];
    out[(uint64_t)t * OUT + o] = a;
}

/* Verify-batch block order: order[q] = the pair computed by block q, sorted by
 * (expert, pair); one block of np (<=48) threads counts how many precede it. */
__global__ static void v41_vq_order_kernel(int32_t *order, const int32_t *sel, uint32_t np) {
    const uint32_t p = threadIdx.x;
    if (p >= np) return;
    const int32_t e = sel[p];
    uint32_t rank = 0;
    for (uint32_t q = 0; q < np; q++) { const int32_t eq = sel[q]; if (eq < e || (eq == e && q < p)) rank++; }
    order[rank] = (int32_t)p;
}

static v41_scratch g_v41_vq_h, g_v41_vq_part, g_v41_vq_xb, g_v41_vq_ogc, g_v41_vq_ord;

/* Forward declarations of the family members that follow this file in the
 * aggregate include order (group after decode, persist after decode, launch
 * last). The group flag only affects n >= 2. */
template <int NBIT, int V3, int EXT>
static int v41_vq_grp_launch(int stage, uint32_t n_tok, uint16_t *h, float *part, const uint8_t *blob, const int32_t *sel, const uint32_t *xb,
                             uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t K, float clamp, uint32_t cbb, const float *gr,
                             const int32_t *ord, uint32_t np);
extern int g_ds4_v41_vq_group;
template <int NBIT, int EXT>
static int v41_vq_persist_launch(int stage, uint16_t *h, float *part, const uint8_t *blob, const int32_t *sel, const uint32_t *xb,
                                 uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t np, float clamp, uint32_t cbb, const float *gr);
template <int NBIT, int EXT>
static int v41_vq_persist_n_launch(int stage, uint32_t n_tok, uint16_t *h, float *part, const uint8_t *blob, const int32_t *sel,
                                   const int32_t *ord, const uint32_t *xb, uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t K,
                                   uint32_t np, float clamp, uint32_t cbb, const float *gr);

/* The n=1..8 worker. gateup and down are gated separately: one codebook (64 KB)
 * fits a block's shared, two (128 KB) do not, and a single ok-flag once pushed
 * both kernels to global gather when only gateup's request failed. */
template <int NBIT, int V3, int EXT>
static int v41_vq_fused_moe_n(float *out, const uint8_t *blob, uint32_t IN, uint32_t MID, uint32_t OUT,
                              const int32_t *sel, const float *w, uint32_t K, float clamp, const float *x, uint32_t n_tok, uint32_t nc,
                              const float *gr) {
    if (n_tok == 0 || n_tok > V41_GEMV_MAX_TOK || (IN % 8u) || (MID % 8u)) return 0;
    const uint64_t np = (uint64_t)n_tok * K;
    /* h holds bf16 (2 B/element): only the down kernel reads it, in packed form. */
    uint16_t *h = (uint16_t *)v41_grow(&g_v41_vq_h, np * MID * 2, "v41 vq h");
    float *part = (float *)v41_grow(&g_v41_vq_part, np * OUT * 4, "v41 vq partial");
    uint16_t *xb = (uint16_t *)v41_grow(&g_v41_vq_xb, (uint64_t)n_tok * IN * 2, "v41 vq x(bf16)");
    if (!h || !part || !xb) return 0;
    {   /* pack the activation to bf16: once per layer, negligible next to the expert kernels */
        const uint64_t nx = (uint64_t)n_tok * IN;
        v41_vq_xpack_kernel<<<(unsigned)((nx + 255) / 256), 256, 0, ds4_current_stream()>>>(xb, x, nx);
        if (!cuda_ok(cudaGetLastError(), "v41 vq xpack")) return 0;
        extern int g_ds4_v41_prof;
        if (g_ds4_v41_prof) {
            uint32_t *c = (uint32_t *)v41_grow(&g_v41_vq_ogc, 4, "v41 vq offgrid");
            uint32_t hc = 0;
            if (c && cudaMemsetAsync(c, 0, 4, ds4_current_stream()) == cudaSuccess) {
                v41_vq_offgrid_kernel<<<(unsigned)((nx + 255) / 256), 256, 0, ds4_current_stream()>>>(c, x, nx);
                if (cudaStreamSynchronize(ds4_current_stream()) == cudaSuccess &&
                    cudaMemcpy(&hc, c, 4, cudaMemcpyDeviceToHost) == cudaSuccess) {
                    static uint64_t tot = 0, bad = 0;
                    tot += nx; bad += hc;
                    if ((tot / nx) % 40u == 0u)
                        fprintf(stderr, "[vq-grid] activations off the bf16 grid: %llu / %llu cumulative\n",
                                (unsigned long long)bad, (unsigned long long)tot);
                }
            }
            v41_f16range_probe(x, NULL, nx, 0, "x");
        }
    }
    const uint32_t cbb = nc * (V3 ? 8u : 16u);   /* v3 codebook is 8 E4M3 per word => nc8192 is still 64 KB */
    /* The opt-in must be recorded per (kernel instance, this layer's codebook):
     * cudaFuncSetAttribute grants a max dynamic shared size to ONE function and
     * the grant then caps every later launch. Both codebook sizes coexist
     * (shallow 13-bit layers nc8192 = 64 KB, deep ones nc4096 = 32 KB), so a
     * file-level "done once" flag granted the first size seen and later failed
     * the other with "invalid argument". static locals in a template are per
     * instantiation, which is exactly per (NBIT,V3,EXT). */
    static uint32_t s_gu_optin = 0u, s_dn_optin = 0u;
    if (s_gu_optin < cbb || s_dn_optin < cbb) {
        /* Clear any error a previous call left latched BEFORE the opt-in:
         * the runtime reports a latched error to a later call, and a failed
         * cudaFuncSetAttribute silently drops this layer set to the other
         * arm (global gather — a different accumulation order).  The arm
         * must depend on the device alone, never on prior error state. */
        (void)cudaGetLastError();
        int cap = 0; (void)cudaDeviceGetAttribute(&cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        int sh_sm = 0, thr_sm = 0, nsm = 0, regs_sm = 0;
        (void)cudaDeviceGetAttribute(&sh_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
        (void)cudaDeviceGetAttribute(&thr_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
        (void)cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
        (void)cudaDeviceGetAttribute(&regs_sm, cudaDevAttrMaxRegistersPerMultiprocessor, 0);
        const bool og = cudaFuncSetAttribute(v41_vq_gateup_kernel<NBIT, V3, EXT, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess &&
                        cudaFuncSetAttribute(v41_vq_gateup_kernel<NBIT, V3, EXT, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess;
        const bool od = cudaFuncSetAttribute(v41_vq_down_kernel<NBIT, V3, EXT, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess &&
                        cudaFuncSetAttribute(v41_vq_down_kernel<NBIT, V3, EXT, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess;
        /* Occupancy must be asked AFTER the opt-in: before it the API uses the
         * default 48 KB limit and reports "no block fits" (the engine printed
         * 0% once and nearly drew the wrong conclusion). */
        cudaFuncAttributes fa; memset(&fa, 0, sizeof fa);
        (void)cudaFuncGetAttributes(&fa, v41_vq_gateup_kernel<NBIT, V3, EXT, 0>);
        int blocks_sm = 0;
        const int thr_blk = (int)(V41_VQ_WARPS * 32u);
        (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_sm, v41_vq_gateup_kernel<NBIT, V3, EXT, 0>, thr_blk, (size_t)cbb);
        fprintf(stderr, "ds4: [ds41] device: %d SM / %d KB shared per SM / %d threads per SM / %d regs per SM\n"
                        "ds4: [ds41] VQ gateup kernel: %d threads/block, %d regs/thread, dynamic shared %u KB"
                        " => %d blocks/SM = %d threads, occupancy %.0f%%\n"
                        "ds4: [ds41]   bound by: shared %d / registers %d / thread slots %d (min)\n",
                nsm, sh_sm >> 10, thr_sm, regs_sm, thr_blk, fa.numRegs, cbb >> 10,
                blocks_sm, blocks_sm * thr_blk, thr_sm ? 100.0 * blocks_sm * thr_blk / thr_sm : 0.0,
                cbb ? sh_sm / (int)cbb : 0,
                fa.numRegs ? regs_sm / (fa.numRegs * thr_blk) : 0, thr_sm / thr_blk);
        (void)cudaGetLastError();
        if (og) s_gu_optin = cbb;   /* a failed grant leaves the layer on global gather */
        if (od) s_dn_optin = cbb;
        fprintf(stderr, "ds4: [ds41] VQ codebook %u words (%u KB each, NBIT %d); per-block dynamic shared cap %d KB => gate+up %s / down %s\n",
                nc, cbb >> 10, NBIT, cap >> 10, og ? "in shared" : "global gather", od ? "in shared" : "global gather");
    }
    const bool shg = cbb <= s_gu_optin, shd = cbb <= s_dn_optin;
    /* Threads: always 32 warps (shared path, one warp loops ITERS rows) or 8
     * warps (global gather, one warp one row) — never rows*32, which would ask
     * for 8192 threads and exceed the 1024 limit. */
    const uint32_t rg = shg ? V41_VQ_GU_ROWS : 8u, rd = shd ? V41_VQ_DN_ROWS : 8u;
    const uint32_t tg = shg ? V41_VQ_WARPS * 32u : 8u * 32u, td = shd ? V41_VQ_WARPS * 32u : 8u * 32u;
    int32_t *ord = NULL;
    if (n_tok >= 2u) {
        ord = (int32_t *)v41_grow(&g_v41_vq_ord, np * 4, "v41 vq order");
        if (!ord) return 0;
        v41_vq_order_kernel<<<1, (unsigned)np, 0, ds4_current_stream()>>>(ord, sel, (uint32_t)np);
        if (!cuda_ok(cudaGetLastError(), "v41 vq order")) return 0;
    }
    /* Drain upstream errors before launching: CUDA errors are sticky, so
     * without this an unchecked launch elsewhere is reported as "vq gateup
     * failed" and the real culprit is invisible (this cost the engine a full
     * debugging round). */
    {
        const cudaError_t pre = cudaGetLastError();
        if (pre != cudaSuccess)
            fprintf(stderr, "ds4: *[ds41] an unchecked CUDA error predates vq gateup: %s*\n", cudaGetErrorString(pre));
    }
    const bool pern = ord && V3 && shg && shd && g_ds4_v41_vq_group &&
        v41_vq_persist_n_launch<NBIT, EXT>(0, n_tok, h, part, blob, sel, ord, (const uint32_t *)xb, IN, MID, OUT, K, (uint32_t)np, clamp, cbb, gr);
    int grp = 0;
    if (!pern && ord && shg && shd && g_ds4_v41_vq_group) {
        grp = v41_vq_grp_launch<NBIT, V3, EXT>(0, n_tok, h, part, blob, sel, (const uint32_t *)xb, IN, MID, OUT, K, clamp, cbb, gr, ord, (uint32_t)np);
        if (grp < 0) return 0;
    }
    /* Diagnostic switch, kept (P4-2): DS41_VQ_NO_PERSIST=1 forces the
     * v2-shaped plain kernels so the persist path can be cmp'd against them —
     * the engine's own equivalence gate ("gate = cmp against the v2-shaped
     * kernels"). Verified byte-identical (mid + partial dumps) on the real
     * layer-0 payload (sm_121, P4-2 evidence). Not usable on sm_89: the plain
     * gateup kernel compiles to 92 registers there and cannot launch at 1024
     * threads ("too many resources"); the persist kernels are pinned by
     * __launch_bounds__(1024, 1). */
    const bool per = n_tok == 1u && V3 && shg && shd && getenv("DS41_VQ_NO_PERSIST") == NULL;
    if (pern) { /* already launched */ }
    else if (per) { if (!v41_vq_persist_launch<NBIT, EXT>(0, h, part, blob, sel, (const uint32_t *)xb, IN, MID, OUT, (uint32_t)np, clamp, cbb, gr)) return 0; }
    else if (ord) v41_vq_gateup_kernel<NBIT, V3, EXT, 1><<<dim3((MID + rg - 1u) / rg, (unsigned)np), tg, shg ? cbb : 0u, ds4_current_stream()>>>(
        h, blob, sel, (const uint32_t *)xb, IN, MID, K, clamp, shg ? cbb : 0u, ord, (uint32_t)np, grp > 0 ? 1u : 0u);
    else v41_vq_gateup_kernel<NBIT, V3, EXT, 0><<<dim3((MID + rg - 1u) / rg, (unsigned)np), tg, shg ? cbb : 0u, ds4_current_stream()>>>(
        h, blob, sel, (const uint32_t *)xb, IN, MID, K, clamp, shg ? cbb : 0u, NULL, 0u, 0u);
    if (!per && !pern && !cuda_ok(cudaGetLastError(), "v41 vq gateup")) {
        /* Log the launch parameters with the failure: "invalid argument" alone
         * does not say which one, and the shared-memory grant is per instance
         * while cbb varies per layer. */
        fprintf(stderr, "ds4: [ds41] gateup launch: grid(%u,%u) block %u dynamic shared %u B; "
                        "NBIT %d V3 %d EXT %d SORTED %d; n_tok %u K %u np %llu IN %u MID %u nc %u cbb %u B granted %u B\n",
                (MID + rg - 1u) / rg, (unsigned)np, tg, shg ? cbb : 0u,
                NBIT, V3, EXT, ord ? 1 : 0, n_tok, K, (unsigned long long)np, IN, MID, nc, cbb, s_gu_optin);
        return 0;
    }
    {
        extern int g_ds4_v41_prof;
        if (g_ds4_v41_prof) v41_f16range_probe(NULL, h, np * MID, 1, "h");
    }
    if (grp > 0 && v41_vq_grp_launch<NBIT, V3, EXT>(1, n_tok, h, part, blob, sel, (const uint32_t *)xb, IN, MID, OUT, K, clamp, cbb, gr, ord, (uint32_t)np) < 0) return 0;
    if (pern) { if (!v41_vq_persist_n_launch<NBIT, EXT>(1, n_tok, h, part, blob, sel, ord, (const uint32_t *)xb, IN, MID, OUT, K, (uint32_t)np, clamp, cbb, gr)) return 0; }
    else if (per) { if (!v41_vq_persist_launch<NBIT, EXT>(1, h, part, blob, sel, (const uint32_t *)xb, IN, MID, OUT, (uint32_t)np, clamp, cbb, gr)) return 0; }
    else if (ord) v41_vq_down_kernel<NBIT, V3, EXT, 1><<<dim3((OUT + rd - 1u) / rd, (unsigned)np), td, shd ? cbb : 0u, ds4_current_stream()>>>(
        part, blob, sel, (const uint32_t *)h, MID, OUT, K, shd ? cbb : 0u, gr, ord, (uint32_t)np, grp > 0 ? 1u : 0u);
    else v41_vq_down_kernel<NBIT, V3, EXT, 0><<<dim3((OUT + rd - 1u) / rd, (unsigned)np), td, shd ? cbb : 0u, ds4_current_stream()>>>(
        part, blob, sel, (const uint32_t *)h, MID, OUT, K, shd ? cbb : 0u, gr, NULL, 0u, 0u);
    if (!per && !pern && !cuda_ok(cudaGetLastError(), "v41 vq down")) return 0;
    if (!out) return 1;   /* the caller folds the routed sum and the shared expert later via ds4_gpu_v41_moe_tail_tensor */
    v41_vq_reduce_kernel<<<dim3((OUT + 255u) / 256u, n_tok), 256, 0, ds4_current_stream()>>>(out, part, w, K, OUT);
    return cuda_ok(cudaGetLastError(), "v41 vq reduce");
}

/* MoE tail, four launches folded into one: y = bf16(sum_k w*partial + so). The
 * arithmetic is untouched (same k order, then + so, then bf16 round), so the
 * output is bit-identical to the reduce -> copy -> add -> round chain it
 * replaced. partial is what the down kernel left in the scratch. */
__global__ static void v41_vq_tail_kernel(float *y, const float *partial, const float *w, const float *so, uint32_t K, uint32_t OUT) {
    v41_pdl_wait();
    const uint32_t t = blockIdx.y, o = blockIdx.x * 256u + threadIdx.x;
    if (o >= OUT) return;
    float a = 0.f;
    for (uint32_t k = 0; k < K; k++) a += w[(uint64_t)t * K + k] * partial[((uint64_t)t * K + k) * OUT + o];
    y[(uint64_t)t * OUT + o] = v41_bf16r(a + so[(uint64_t)t * OUT + o]);
}

extern "C" int ds4_gpu_v41_moe_tail_tensor(ds4_gpu_tensor *y, const ds4_gpu_tensor *so, const ds4_gpu_tensor *weights,
                                           uint32_t n_tok, uint32_t n_used, uint32_t out_dim) {
    if (!y || !so || !weights || !g_v41_vq_part.p || g_v41_vq_part.cap < (uint64_t)n_tok * n_used * out_dim * 4) return 0;
    v41_vq_tail_kernel<<<dim3((out_dim + 255u) / 256u, n_tok), 256, 0, ds4_current_stream()>>>(
        (float *)y->ptr, (const float *)g_v41_vq_part.p, (const float *)weights->ptr, (const float *)so->ptr, n_used, out_dim);
    return cuda_ok(cudaGetLastError(), "v41 moe tail");
}
