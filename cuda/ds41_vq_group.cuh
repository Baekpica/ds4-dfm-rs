/* ds41_vq_group.cuh — the multi-token grouped VQ "decode is the multiply"
 * kernels (verify batches and draft-tower small batches), ported from the C
 * engine at /data/YoungAi (commit 3946dbc), file
 * `src/cuda/cuda_vq_group.inc.cu`.
 *
 * The disease this treats: a verify batch of n rows through the per-pair
 * kernels costs n=1 0.356 ms/layer and n=4 1.283 ms/layer = 3.6x — the weight
 * bytes do not amortize at all, so "verifying k+1 positions costs about one"
 * does not hold in that shape. ncu named the cost unit: L1TEX wavefronts — per
 * round (32 indices) the codebook lookups are ~11.7 + activations 4 + bitstream
 * ~1.5, and the per-pair path pays the codebook lookups m times when m tokens
 * pick the same expert. Here one block owns one expert x the m tokens of this
 * batch that chose it (m <= M = batch size): the bitstream is read once, each
 * index's codeword decoded once, and each token does one 8-element multiply-add
 * — 11.7 + 4m wavefronts per round versus 15.7m for per-pair.
 *
 * The "same track" property (speculative == pure decode, byte for byte) holds
 * because each token's row goes through the same function the single-token
 * kernel uses (v41_vq_dot8_cw): per-lane accumulation order, the cross-lane
 * reduction tree and the gain multiply are untouched; only which block computes
 * which pair when changes. Gate = cmp.
 * A grouping table error (order sorted by expert) writes one token's output
 * into another pair's slot — no error, only the temperature-0 byte gate catches
 * it.
 *
 * Must follow ds41_vq_decode.cuh (its v41_vq_cb_to_shared / v41_vq_swiglu and
 * the row family) and precede ds41_vq_launch.cuh (which instantiates
 * v41_vq_fused_moe_n and needs v41_vq_grp_launch visible).
 */
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ds41_vq_decode.cuh"

/* One block = 16 warps = 512 threads => 128 registers/thread, enough for M
 * accumulators + M activation loads (the per-pair kernel allows only 64 at 1024
 * threads and sits at 56-64). At <= 64 registers two such blocks fit per SM
 * (2x64 KB shared, 1024 threads — within the device budget), same occupancy as
 * the per-pair kernel. Rows per warp: gateup 2 (32 rows/block), down 4 (64) —
 * half the block size of the per-pair kernel; the load balance comes from more
 * waves (measured: this kernel lives on multiple waves). */
#define V41_VQG_WARPS    16u
#define V41_VQG_THREADS  (V41_VQG_WARPS * 32u)
#define V41_VQG_ITERS_GU 2u
#define V41_VQG_ITERS_DN 4u

/* Up to 8 rounds per block, M-token form: index and codeword decoded once, one
 * v41_vq_dot8_cw per token. xs[j] = token j's activation row (packed bf16),
 * boff = this block's word offset within the row (32 indices x 4 words per
 * round); m is block-uniform, so the branch does not diverge. */
template <int NBIT, int V3, int EXT, int M>
__device__ __forceinline__ static void v41_vqg_blk_rounds(const v41_vq_blk &cur, uint32_t rounds, uint32_t a, uint32_t sh, uint32_t imsk,
                                                          const uint32_t *const *xs, uint32_t boff, uint32_t m, const uint8_t *cbs, float *acc) {
    const uint32_t lane = threadIdx.x & 31u;
    constexpr uint32_t MB = v41_vq_mbit<NBIT, V3>();
    #pragma unroll
    for (uint32_t k = 0; k < 8u; k++) {
        if (k >= rounds) break;
        const uint32_t f = k * MB;
        const uint32_t lo_r = ((f & 31u) > 32u - MB) ? ((f + 31u - lane) >> 5) : (f >> 5);
        const uint32_t hi_r = (((f + 1u) & 31u) > 32u - MB) ? ((f + 32u - lane) >> 5) : ((f + 1u) >> 5);
        const uint32_t lo = __shfl_sync(0xffffffffu, v41_vq_sel3(lo_r, cur.w0, cur.w1, cur.w2), (int)(f + a));
        const uint32_t hi = __shfl_sync(0xffffffffu, v41_vq_sel3(hi_r, cur.w0, cur.w1, cur.w2), (int)(f + a + 1u));
        uint32_t v = __funnelshift_r(lo, hi, sh) & (EXT ? 0xFFFu : imsk);
        if (EXT) {
            const uint32_t exw = __shfl_sync(0xffffffffu, cur.ex, (int)k);
            v |= ((exw >> lane) & 1u) << 12;
        }
        float c[8];
        v41_vq_cw<V3>(v, cbs, 1, c);                    /* codeword decoded once (codebook always in shared here) */
        const uint32_t xo = boff + (k * 32u + lane) * 4u;
        #pragma unroll
        for (int j = 0; j < M; j++) {
            if ((uint32_t)j < m) {
                const uint4 xa = *(const uint4 *)(xs[j] + xo);   /* 8 bf16 = 16 B, L1 hit */
                acc[j] += v41_vq_dot8_cw(c, xa);              /* the same multiply-add expression as the single-token kernel */
            }
        }
    }
}

/* One warp dots one row against m tokens (same cross-block pipeline as
 * v41_vq_row_dot: carry in = this row's first block, out = next row's first).
 * out[j] = gain * sum (warp-reduced), the multiply order identical to the
 * single-token kernel (acc * row gain * gov, no pre-multiply). */
template <int NBIT, int V3, int EXT, int M>
__device__ __forceinline__ static void v41_vqg_row_dot(const v41_vq_mat &mt, uint32_t r, const uint32_t *const *xs, uint32_t m, const uint8_t *cbs,
                                                       v41_vq_blk *carry, const uint32_t *next, const uint32_t *nextex, float *out) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t R = mt.nidx_row >> 5;
    constexpr uint32_t MB = v41_vq_mbit<NBIT, V3>();
    const uint32_t a = (lane * MB) >> 5, sh = (lane * MB) & 31u;
    const uint32_t *row = v41_vq_row_ptr<NBIT, V3>(mt, r);
    const uint32_t *rex = EXT ? v41_vq_ext_ptr(mt, r) : NULL;
    v41_vq_blk cur = *carry;
    float acc[M];
    #pragma unroll
    for (int j = 0; j < M; j++) acc[j] = 0.f;
    for (uint32_t b = 0; b < R; b += 8u) {
        v41_vq_blk nxt;
        if (b + 8u < R) {
            const uint32_t rem = R - b - 8u, nr = (rem < 8u ? rem : 8u);
            nxt = v41_vq_blk_load<EXT>(row + (size_t)(b + 8u) * MB, nr * MB, EXT ? rex + (b + 8u) : NULL, nr);   /* plane advances per group (fixed 09-22) */
        }
        else if (next) { const uint32_t nr = (R < 8u ? R : 8u); nxt = v41_vq_blk_load<EXT>(next, nr * MB, nextex, nr); }
        else { nxt.w0 = 0u; nxt.w1 = 0u; nxt.w2 = 0u; nxt.ex = 0u; }
        v41_vqg_blk_rounds<NBIT, V3, EXT, M>(cur, (R - b < 8u) ? R - b : 8u, a, sh, mt.imsk, xs, b * 32u * 4u, m, cbs, acc);
        cur = nxt;
    }
    *carry = cur;
    __half gh; memcpy(&gh, mt.gr + (size_t)r * 2u, 2);
    #pragma unroll
    for (int j = 0; j < M; j++) {
        if ((uint32_t)j < m) {
            float v = acc[j];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            out[j] = v * __half2float(gh) * (mt.gov ? mt.gov[r] : 1.0f);
        }
    }
}

/* Which expert and which tokens this block owns: order[] is sorted by (expert,
 * pair) and same-expert pairs are adjacent; only the head block of a run works
 * (the rest return immediately, block-uniform so no __syncthreads is harmed),
 * run length m (a token picks an expert at most once).
 * Mixed dispatch (2026-09-22): letting ALL groups go through the grouped kernel
 * measured 94.0 ms versus 87.5 for per-pair on a k=3 verify batch — most groups
 * hold one token, and they are slower in the 16-warp/high-register shape. So
 * m=1 groups stay on the per-pair kernel (its SORTED instance skips m>=2 pairs)
 * and the grouped kernel takes m in [lo, M]; M=2/4/6 instances (registers grow
 * with M; large groups use the large instance).
 * Empty slots j >= m point at the head pair: valid pointer, never read or
 * written. Returns m; 0 = not a run head or the run length is outside this
 * instance's band. */
__device__ __forceinline__ static uint32_t v41_vqg_group(const int32_t *sel, const int32_t *order, uint32_t np, uint32_t q, int M, uint32_t lo,
                                                         int32_t *e_out, uint32_t *pairs) {
    const uint32_t p0 = (uint32_t)order[q];
    const int32_t e = sel[p0];
    *e_out = e;
    if (e < 0) return 0u;
    if (q > 0u && sel[order[q - 1u]] == e) return 0u;
    uint32_t m = 1u;
    while (q + m < np && sel[order[q + m]] == e) m++;
    if (m < lo || m > (uint32_t)M) return 0u;
    for (int j = 0; j < M; j++) pairs[j] = ((uint32_t)j < m) ? (uint32_t)order[q + (uint32_t)j] : p0;
    return m;
}

/* gate/up in one kernel (same form as v41_vq_gateup_kernel: w1/w3 -> bf16 ->
 * silu(g)*u -> bf16), one block per expert x m tokens. One shared buffer serves
 * twice (gate's codebook then up's); the gate result parks as bf16 in each
 * token's own h row. */
template <int NBIT, int V3, int EXT, int M>
__global__ static void __launch_bounds__(V41_VQG_THREADS)
v41_vqg_gateup_kernel(uint16_t *h, const uint8_t *blob, const int32_t *sel, const uint32_t *x,
                      uint32_t IN, uint32_t MID, uint32_t K, float clamp, uint32_t cb_bytes, const int32_t *order, uint32_t np, uint32_t mlo) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    constexpr uint32_t W = V41_VQG_WARPS, NIT = V41_VQG_ITERS_GU;
    int32_t e; uint32_t pairs[M];
    const uint32_t m = v41_vqg_group(sel, order, np, blockIdx.y, M, mlo, &e, pairs);
    if (!m) return;
    const uint32_t *xs[M]; uint16_t *hp[M];
    #pragma unroll
    for (int j = 0; j < M; j++) { xs[j] = x + (uint64_t)(pairs[j] / K) * (IN / 2u); hp[j] = h + (uint64_t)pairs[j] * MID; }
    const v41_vq_mat mg = v41_vq_open<V3>(blob, e, 0, MID, IN, NULL), mu = v41_vq_open<V3>(blob, e, 1, MID, IN, NULL);
    if (!mg.ok || !mu.ok) return;
    constexpr uint32_t CW = V3 ? 8u : 16u;
    if (mg.nc * CW != cb_bytes || mu.nc * CW != cb_bytes) return;
    const uint32_t r0 = blockIdx.x * (W * NIT) + (threadIdx.x >> 5);
    const bool lead = (threadIdx.x & 31u) == 0u;
    v41_vq_blk carry; carry.w0 = 0u; carry.w1 = 0u; carry.w2 = 0u; carry.ex = 0u;
    if (r0 < MID) carry = v41_vq_row_first_blk<NBIT, V3, EXT>(mg, r0);
    v41_vq_cb_to_shared(vqsh, mg.cb, cb_bytes);
    __syncthreads();
    for (uint32_t i = 0; i < NIT; i++) {
        const uint32_t r = r0 + i * W;
        if (r >= MID) break;
        const uint32_t rn = r + W;
        const int own = (i + 1u < NIT && rn < MID);
        const uint32_t *next = own ? v41_vq_row_ptr<NBIT, V3>(mg, rn) : v41_vq_row_ptr<NBIT, V3>(mu, r0);
        const uint32_t *nextex = EXT ? (own ? v41_vq_ext_ptr(mg, rn) : v41_vq_ext_ptr(mu, r0)) : NULL;
        float g[M];
        v41_vqg_row_dot<NBIT, V3, EXT, M>(mg, r, xs, m, vqsh, &carry, next, nextex, g);
        if (lead) {
            #pragma unroll
            for (int j = 0; j < M; j++) if ((uint32_t)j < m) hp[j][r] = (uint16_t)(__float_as_uint(v41_bf16r(g[j])) >> 16);
        }
    }
    __syncthreads();                            /* all warps done with gate's codebook before overwriting it */
    v41_vq_cb_to_shared(vqsh, mu.cb, cb_bytes);
    __syncthreads();
    for (uint32_t i = 0; i < NIT; i++) {
        const uint32_t r = r0 + i * W;
        if (r >= MID) break;
        const uint32_t rn = r + W;
        const int own = (i + 1u < NIT && rn < MID);
        const uint32_t *next = own ? v41_vq_row_ptr<NBIT, V3>(mu, rn) : NULL;
        float u[M];
        v41_vqg_row_dot<NBIT, V3, EXT, M>(mu, r, xs, m, vqsh, &carry, next, EXT && own ? v41_vq_ext_ptr(mu, rn) : NULL, u);
        if (lead) {
            #pragma unroll
            for (int j = 0; j < M; j++)
                if ((uint32_t)j < m) hp[j][r] = v41_vq_swiglu(__uint_as_float((uint32_t)hp[j][r] << 16), v41_bf16r(u[j]), clamp);
        }
    }
}

/* down: partial[pair][OUT] = bf16(W2 * h_pair), one block per expert x m tokens. */
template <int NBIT, int V3, int EXT, int M>
__global__ static void __launch_bounds__(V41_VQG_THREADS)
v41_vqg_down_kernel(float *partial, const uint8_t *blob, const int32_t *sel, const uint32_t *hh,
                    uint32_t MID, uint32_t OUT, uint32_t K, uint32_t cb_bytes, const float *gr, const int32_t *order, uint32_t np, uint32_t mlo) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    constexpr uint32_t W = V41_VQG_WARPS, NIT = V41_VQG_ITERS_DN;
    int32_t e; uint32_t pairs[M];
    const uint32_t m = v41_vqg_group(sel, order, np, blockIdx.y, M, mlo, &e, pairs);
    if (!m) return;
    const uint32_t *hs[M];
    #pragma unroll
    for (int j = 0; j < M; j++) hs[j] = hh + (uint64_t)pairs[j] * (MID / 2u);
    const v41_vq_mat md = v41_vq_open<V3>(blob, e, 2, OUT, MID, gr ? gr + (size_t)e * OUT : NULL);
    const uint32_t r0 = blockIdx.x * (W * NIT) + (threadIdx.x >> 5);
    const bool lead = (threadIdx.x & 31u) == 0u;
    if (!md.ok) {   /* bad payload: this block's rows all write 0 (the reduce reads them) */
        for (uint32_t i = 0; i < NIT; i++) {
            const uint32_t r = r0 + i * W;
            if (r < OUT && lead) for (int j = 0; j < M; j++) if ((uint32_t)j < m) partial[(uint64_t)pairs[j] * OUT + r] = 0.f;
        }
        return;
    }
    if (md.nc * (V3 ? 8u : 16u) != cb_bytes) return;
    v41_vq_blk carry; carry.w0 = 0u; carry.w1 = 0u; carry.w2 = 0u; carry.ex = 0u;
    if (r0 < OUT) carry = v41_vq_row_first_blk<NBIT, V3, EXT>(md, r0);
    v41_vq_cb_to_shared(vqsh, md.cb, cb_bytes);
    __syncthreads();
    for (uint32_t i = 0; i < NIT; i++) {
        const uint32_t r = r0 + i * W;
        if (r >= OUT) break;
        const uint32_t rn = r + W;
        const int own = (i + 1u < NIT && rn < OUT);
        const uint32_t *next = own ? v41_vq_row_ptr<NBIT, V3>(md, rn) : NULL;
        float y[M];
        v41_vqg_row_dot<NBIT, V3, EXT, M>(md, r, hs, m, vqsh, &carry, next, EXT && own ? v41_vq_ext_ptr(md, rn) : NULL, y);
        if (lead) {
            #pragma unroll
            for (int j = 0; j < M; j++) if ((uint32_t)j < m) partial[(uint64_t)pairs[j] * OUT + r] = v41_bf16r(y[j]);
        }
    }
    (void)K;
}

/* Per-M-instance opt-in (template statics are per instantiation — the same rule
 * as the per-pair kernel's 09-22 lesson). 1 = cbb granted; 0 = cannot (the whole
 * grouped path falls back to per-pair). */
template <int NBIT, int V3, int EXT, int M>
static int v41_vqg_optin(uint32_t cbb) {
    static uint32_t s_optin = 0u;
    if (s_optin >= cbb) return 1;
    (void)cudaGetLastError();   /* the opt-in must not see a previously latched error (see ds41_vq_decode.cuh) */
    const bool ok = cudaFuncSetAttribute(v41_vqg_gateup_kernel<NBIT, V3, EXT, M>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess &&
                    cudaFuncSetAttribute(v41_vqg_down_kernel<NBIT, V3, EXT, M>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess;
    (void)cudaGetLastError();
    if (!ok) return 0;
    s_optin = cbb;
    cudaFuncAttributes fg, fd; memset(&fg, 0, sizeof fg); memset(&fd, 0, sizeof fd);
    (void)cudaFuncGetAttributes(&fg, v41_vqg_gateup_kernel<NBIT, V3, EXT, M>);
    (void)cudaFuncGetAttributes(&fd, v41_vqg_down_kernel<NBIT, V3, EXT, M>);
    int bg = 0, bd = 0;
    (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bg, v41_vqg_gateup_kernel<NBIT, V3, EXT, M>, (int)V41_VQG_THREADS, (size_t)cbb);
    (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bd, v41_vqg_down_kernel<NBIT, V3, EXT, M>, (int)V41_VQG_THREADS, (size_t)cbb);
    (void)cudaGetLastError();
    fprintf(stderr, "ds4: [ds41] VQ grouped kernel M=%d (NBIT %d): %u threads/block, dynamic shared %u KB; gateup %d regs => %d blocks/SM, down %d regs => %d blocks/SM\n",
            M, NBIT, V41_VQG_THREADS, cbb >> 10, fg.numRegs, bg, fd.numRegs, bd);
    return 1;
}

/* Launch one M-band's gateup or down (takes runs with m in [mlo, M]); grid.y =
 * np, blocks outside the band return immediately. */
template <int NBIT, int V3, int EXT, int M>
static int v41_vqg_launch_gu(uint16_t *h, const uint8_t *blob, const int32_t *sel, const uint32_t *xb, uint32_t IN, uint32_t MID, uint32_t K,
                             float clamp, uint32_t cbb, const int32_t *ord, uint32_t np, uint32_t mlo) {
    const uint32_t rg = V41_VQG_WARPS * V41_VQG_ITERS_GU;
    v41_vqg_gateup_kernel<NBIT, V3, EXT, M><<<dim3((MID + rg - 1u) / rg, np), V41_VQG_THREADS, cbb, ds4_current_stream()>>>(
        h, blob, sel, xb, IN, MID, K, clamp, cbb, ord, np, mlo);
    return cuda_ok(cudaGetLastError(), "v41 vq gateup (grouped)");
}

template <int NBIT, int V3, int EXT, int M>
static int v41_vqg_launch_dn(float *part, const uint8_t *blob, const int32_t *sel, const uint16_t *h, uint32_t MID, uint32_t OUT, uint32_t K,
                             uint32_t cbb, const float *gr, const int32_t *ord, uint32_t np, uint32_t mlo) {
    const uint32_t rd = V41_VQG_WARPS * V41_VQG_ITERS_DN;
    v41_vqg_down_kernel<NBIT, V3, EXT, M><<<dim3((OUT + rd - 1u) / rd, np), V41_VQG_THREADS, cbb, ds4_current_stream()>>>(
        part, blob, sel, (const uint32_t *)h, MID, OUT, K, cbb, gr, ord, np, mlo);
    return cuda_ok(cudaGetLastError(), "v41 vq down (grouped)");
}

/* The two stages of the grouped path (gateup then down), called from
 * v41_vq_fused_moe_n: the per-pair kernel's SORTED instance computes only m=1
 * pairs while the grouped kernel takes m>=2 runs by M band (2 / 4 / 6), each
 * writing disjoint pair slots. Batch size picks the bands (m <= n_tok): n <= 2
 * only M=2; 3..4 adds M=4 (takes 3..4); 5..6 adds M=6 (takes 5..6).
 * 1 = launched; 0 = some band could not opt in (the caller falls back to all
 * per-pair); -1 = launch failure (hard error). */
template <int NBIT, int V3, int EXT>
static int v41_vq_grp_launch(int stage, uint32_t n_tok, uint16_t *h, float *part, const uint8_t *blob, const int32_t *sel, const uint32_t *xb,
                             uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t K, float clamp, uint32_t cbb, const float *gr,
                             const int32_t *ord, uint32_t np) {
    if (n_tok < 2u || n_tok > 6u) return 0;
    if (!v41_vqg_optin<NBIT, V3, EXT, 2>(cbb)) return 0;
    if (n_tok >= 3u && !v41_vqg_optin<NBIT, V3, EXT, 4>(cbb)) return 0;
    if (n_tok >= 5u && !v41_vqg_optin<NBIT, V3, EXT, 6>(cbb)) return 0;
    if (stage == 0) {
        if (!v41_vqg_launch_gu<NBIT, V3, EXT, 2>(h, blob, sel, xb, IN, MID, K, clamp, cbb, ord, np, 2u)) return -1;
        if (n_tok >= 3u && !v41_vqg_launch_gu<NBIT, V3, EXT, 4>(h, blob, sel, xb, IN, MID, K, clamp, cbb, ord, np, 3u)) return -1;
        if (n_tok >= 5u && !v41_vqg_launch_gu<NBIT, V3, EXT, 6>(h, blob, sel, xb, IN, MID, K, clamp, cbb, ord, np, 5u)) return -1;
        return 1;
    }
    if (!v41_vqg_launch_dn<NBIT, V3, EXT, 2>(part, blob, sel, h, MID, OUT, K, cbb, gr, ord, np, 2u)) return -1;
    if (n_tok >= 3u && !v41_vqg_launch_dn<NBIT, V3, EXT, 4>(part, blob, sel, h, MID, OUT, K, cbb, gr, ord, np, 3u)) return -1;
    if (n_tok >= 5u && !v41_vqg_launch_dn<NBIT, V3, EXT, 6>(part, blob, sel, h, MID, OUT, K, cbb, gr, ord, np, 5u)) return -1;
    return 1;
}
