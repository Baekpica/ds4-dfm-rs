/* ds41_vq_persist.cuh — the v3 persistent decode kernels, ported from the C
 * engine at /data/YoungAi (commit 3946dbc), file `src/cuda/cuda_vq_persist.inc.cu`.
 *
 * Why: a v3 codebook is ONE PER LAYER, but the v2-shaped gateup/down kernels
 * copy it into shared per block (and gateup twice, once per matrix) with three
 * full-block barriers each time; at 1 block/SM the whole SM waits during every
 * copy. Grid = SM count, one persistent block per SM: the codebook is copied
 * once per layer, then each warp walks a contiguous span of (expert, row) work
 * items with no further block-wide barriers. gateup computes gate and up for
 * the same row in one warp and SwiGLUs in registers (no h round trip).
 *
 * Byte-identical: every row still goes through the same v41_vq_row_dot (same
 * lane<->index map, same accumulation order, same reduction tree); the gate
 * value used to park as bf16 in h and read back, and it is on the bf16 grid by
 * construction (v41_bf16r exit), so the round trip was lossless and SwiGLU's
 * input is bit-identical. Gate = cmp against the v2-shaped kernels.
 * A payload whose codebook does not point at the layer's one codebook is a
 * format violation => __trap(), never a silently different codebook.
 *
 * Must follow ds41_vq_decode.cuh (uses its v41_vq_cb_to_shared / v41_vq_swiglu
 * / V41_VQ_WARPS) and precede ds41_vq_launch.cuh.
 */
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ds41_vq_decode.cuh"

/* The first valid pair's codebook for a matrix = the layer codebook (same value
 * across the whole block, so the early return is block-uniform — no warp can
 * miss a barrier). */
__device__ __forceinline__ static const uint8_t *v41_vq_layer_cb(const uint8_t *blob, const int32_t *sel, uint32_t np, int which,
                                                                 uint32_t rows, uint32_t cols) {
    for (uint32_t p = 0; p < np; p++) {
        const int32_t e = sel[p];
        if (e < 0) continue;
        const v41_vq_mat m = v41_vq_open<1>(blob, e, which, rows, cols, NULL);
        if (m.ok) return m.cb;
    }
    return NULL;
}

/* A run of consecutive rows read as ONE bitstream. The disease (ncu on the
 * first persistent version): long_scoreboard ~14, and moving activations to
 * shared changed nothing — the wait was the bitstream. v41_vq_row_dot's
 * pipeline is "read the next block while computing this one" (8 rounds), but a
 * gateup row is 20 rounds = 8+8+4 and a down row 9 = 8+1: the last block's
 * 4 (1) rounds cannot hide the next row's first-block DRAM latency (~1.2-1.4
 * us loaded) => every row stalls once at its head, plus the row gain gr[r]
 * (and the sidecar gov[r]) is a global read issued at the row's end and needed
 * immediately.
 * Fix: consecutive rows of the same matrix are contiguous in the bitstream (a
 * row is R rounds x 12 words, the plane R words per row), so a warp's n
 * consecutive rows are one continuous stream — cut into 8-round blocks by
 * "global round number within the span", never restarting at row boundaries.
 * A row that runs out of rounds reduces, applies the gain, and moves on; row
 * gains are prefetched at the span head (lane i takes row i).
 * Byte-identical: each row still starts at 0 and accumulates the same
 * v41_vq_dot8 sequence, the reduction tree and gain multiply match
 * v41_vq_row_dot; the lane<->word map within a round only depends on the
 * round's position k in the block (exactly 12 whole words per round). */
/* Archived negatives (engine, do not resurrect): f32 activations in shared
 * (6.46 -> 6.67 ms, not instruction-count-bound); bitstream through a shared
 * ring with cp.async (noise); M as a warp-count knob (fewer warps in flight
 * loses more than the dedupe saves). */
/* M = how many activations one span multiplies (2026-09-24): pure decode M=1
 * (xs[0]); in a verify batch, when one expert is chosen by nt <= M tokens, the
 * bitstream is read once, the codeword decoded once (v41_vq_cw), and each
 * token does its own v41_vq_dot8_cw — same expression tree as M=1's
 * v41_vq_dot8, so each token's accumulation order / reduction / gain multiply
 * is unchanged => verify row t == the pure-decode row. M=1 keeps the original
 * single v41_vq_dot8 call, so pure decode's codegen is untouched. */
template <int NBIT, int EXT, int M>
__device__ __forceinline__ static void v41_vq_stream(const v41_vq_mat &m, uint32_t r0, uint32_t n, const uint32_t *const *xs, uint32_t nt,
                                                     const uint8_t *cbs, v41_vq_blk *carry, const uint32_t *next, const uint32_t *nextex,
                                                     float *res) {
    const uint32_t lane = threadIdx.x & 31u;
    constexpr uint32_t MB = 12u;                                         /* v3 main stream is always 12 bits */
    const uint32_t R = m.nidx_row >> 5, G = n * R;                       /* rounds per row / rounds in the span */
    const uint32_t a = (lane * MB) >> 5, sh = (lane * MB) & 31u;
    const uint32_t *base = v41_vq_row_ptr<NBIT, 1>(m, r0);
    const uint32_t *exb = EXT ? v41_vq_ext_ptr(m, r0) : NULL;
    float g1 = 0.f, g2 = 1.f;                                            /* lane i: row r0+i's payload gain / sidecar gain */
    if (lane < n) { __half gh; memcpy(&gh, m.gr + (size_t)(r0 + lane) * 2u, 2); g1 = __half2float(gh); if (m.gov) g2 = m.gov[r0 + lane]; }
    v41_vq_blk cur = *carry;
    float acc[M];
    #pragma unroll
    for (int j = 0; j < M; j++) { acc[j] = 0.f; res[j] = 0.f; }
    uint32_t kin = 0, row = 0;
    for (uint32_t g0 = 0; g0 < G; g0 += 8u) {
        v41_vq_blk nxt;
        if (g0 + 8u < G) { const uint32_t rem = G - g0 - 8u, nr = rem < 8u ? rem : 8u;
                           nxt = v41_vq_blk_load<EXT>(base + (size_t)(g0 + 8u) * MB, nr * MB, EXT ? exb + (g0 + 8u) : NULL, nr); }
        else if (next) nxt = v41_vq_blk_load<EXT>(next, 8u * MB, nextex, 8u);   /* caller guarantees that span has >= 8 rounds */
        else { nxt.w0 = 0u; nxt.w1 = 0u; nxt.w2 = 0u; nxt.ex = 0u; }
        const uint32_t rounds = G - g0 < 8u ? G - g0 : 8u;
        #pragma unroll
        for (uint32_t k = 0; k < 8u; k++) {
            if (k >= rounds) break;
            uint4 xa0;   /* M=1: read as the round opens; M>1: read per use, saving 4(M-1) registers (64 per thread at 1024 threads) */
            if (M == 1) xa0 = *(const uint4 *)(xs[0] + (size_t)(kin * 32u + lane) * 4u);
            const uint32_t f = k * MB;                                   /* the four lines below are verbatim v41_vq_blk_rounds */
            const uint32_t lo_r = ((f & 31u) > 32u - MB) ? ((f + 31u - lane) >> 5) : (f >> 5);
            const uint32_t hi_r = (((f + 1u) & 31u) > 32u - MB) ? ((f + 32u - lane) >> 5) : ((f + 1u) >> 5);
            const uint32_t lo = __shfl_sync(0xffffffffu, v41_vq_sel3(lo_r, cur.w0, cur.w1, cur.w2), (int)(f + a));
            const uint32_t hi = __shfl_sync(0xffffffffu, v41_vq_sel3(hi_r, cur.w0, cur.w1, cur.w2), (int)(f + a + 1u));
            uint32_t v = __funnelshift_r(lo, hi, sh) & 0xFFFu;
            if (EXT) { const uint32_t exw = __shfl_sync(0xffffffffu, cur.ex, (int)k); v |= ((exw >> lane) & 1u) << 12; }
            if (M == 1) acc[0] += v41_vq_dot8<1>(v, xa0, cbs, 1);
            else {
                float c[8];
                v41_vq_cw<1>(v, cbs, 1, c);
                #pragma unroll
                for (int j = 0; j < M; j++)
                    if ((uint32_t)j < nt) acc[j] += v41_vq_dot8_cw(c, *(const uint4 *)(xs[j] + (size_t)(kin * 32u + lane) * 4u));
            }
            if (++kin == R) {                                            /* a row ran out of rounds: same finish as v41_vq_row_dot */
                #pragma unroll
                for (int j = 0; j < M; j++)
                    if ((uint32_t)j < nt) for (int o = 16; o > 0; o >>= 1) acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], o);
                const float a1 = __shfl_sync(0xffffffffu, g1, (int)row), a2 = __shfl_sync(0xffffffffu, g2, (int)row);
                #pragma unroll
                for (int j = 0; j < M; j++) {
                    const float val = acc[j] * a1 * (m.gov ? a2 : 1.0f);
                    if (lane == row && (uint32_t)j < nt) res[j] = val;
                    acc[j] = 0.f;
                }
                kin = 0u; row++;
            }
        }
        cur = nxt;
    }
    *carry = cur;
}

/* __launch_bounds__(1024, 1): 1024 threads/block caps 64 registers/thread; the
 * 13-bit instance (plane) naturally exceeds it, so it must be pinned; a spill
 * (cuobjdump LOCAL) disqualifies the build. Work split: all np*rows
 * "(expert, row)" items cut into contiguous per-warp spans, a span at most 32
 * rows and never crossing experts (lane i receives row i's result). */
#define V41_VQ_SEG(U, UEND, ROWS, P, R0, N) \
    const uint32_t P = (U) / (ROWS), R0 = (U) % (ROWS); \
    uint32_t N = (UEND) - (U); if (N > (ROWS) - R0) N = (ROWS) - R0; if (N > 32u) N = 32u

template <int NBIT, int EXT>
__global__ static void __launch_bounds__(1024, 1) v41_vq_gu_persist_kernel(uint16_t *h, const uint8_t *blob, const int32_t *sel, const uint32_t *x,
                                                uint32_t IN, uint32_t MID, uint32_t np, float clamp, uint32_t cbb) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const uint32_t gw = blockIdx.x * V41_VQ_WARPS + (threadIdx.x >> 5), nw = gridDim.x * V41_VQ_WARPS;
    const uint32_t total = np * MID, per = (total + nw - 1u) / nw, lane = threadIdx.x & 31u;
    uint32_t u = gw * per;
    const uint32_t uend = (u + per < total) ? u + per : total;
    /* PDL: the codebook is a constant, located via expert 0's payload (v3
     * shares one per layer) and copied first; sel/x are upstream output and
     * only touched after v41_pdl_wait. */
    const v41_vq_mat m0 = v41_vq_open<1>(blob, 0, 0, MID, IN, NULL);
    const uint8_t *cb = m0.ok ? m0.cb : NULL;
    if (cb) v41_vq_cb_to_shared(vqsh, cb, cbb);
    v41_pdl_wait();
    if (!cb) { cb = v41_vq_layer_cb(blob, sel, np, 0, MID, IN); if (!cb) return; v41_vq_cb_to_shared(vqsh, cb, cbb); }
    v41_vq_blk carry; carry.w0 = 0u; carry.w1 = 0u; carry.w2 = 0u; carry.ex = 0u;
    if (u < uend) {   /* the first span's first block issues behind the activation copy */
        const int32_t e = sel[u / MID];
        if (e >= 0) { const v41_vq_mat mg = v41_vq_open<1>(blob, e, 0, MID, IN, NULL); if (mg.ok) carry = v41_vq_row_first_blk<NBIT, 1, EXT>(mg, u % MID); }
    }
    v41_vq_cb_to_shared(vqsh + cbb, (const uint8_t *)x, IN * 2u);   /* activations (bf16) into shared too */
    __syncthreads();   /* the whole kernel's only block-wide barrier */
    x = (const uint32_t *)(vqsh + cbb);
    bool first = true;
    while (u < uend) {
        V41_VQ_SEG(u, uend, MID, p, r0, n);
        u += n;
        const int32_t e = sel[p];
        if (e < 0) { first = false; continue; }   /* invalid expert: write nothing, as the v2 kernel does */
        const v41_vq_mat mg = v41_vq_open<1>(blob, e, 0, MID, IN, NULL), mu = v41_vq_open<1>(blob, e, 1, MID, IN, NULL);
        if (!mg.ok || !mu.ok) { first = false; continue; }
        if (mg.cb != cb || mu.cb != cb) __trap();   /* format violation: never compute with a foreign codebook */
        if (!first) carry = v41_vq_row_first_blk<NBIT, 1, EXT>(mg, r0);
        first = false;
        /* gate span -> up span (gate's last block preloads up's first); both spans >= 8 rounds (a row is 20) */
        float gs, us;
        v41_vq_stream<NBIT, EXT, 1>(mg, r0, n, &x, 1u, vqsh, &carry, v41_vq_row_ptr<NBIT, 1>(mu, r0), EXT ? v41_vq_ext_ptr(mu, r0) : NULL, &gs);
        v41_vq_stream<NBIT, EXT, 1>(mu, r0, n, &x, 1u, vqsh, &carry, NULL, NULL, &us);
        if (lane < n) h[(uint64_t)p * MID + r0 + lane] = v41_vq_swiglu(v41_bf16r(gs), v41_bf16r(us), clamp);
    }
}

template <int NBIT, int EXT>
__global__ static void __launch_bounds__(1024, 1) v41_vq_dn_persist_kernel(float *partial, const uint8_t *blob, const int32_t *sel, const uint32_t *h,
                                                uint32_t MID, uint32_t OUT, uint32_t np, uint32_t cbb, const float *gr) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const uint32_t gw = blockIdx.x * V41_VQ_WARPS + (threadIdx.x >> 5), nw = gridDim.x * V41_VQ_WARPS;
    const uint32_t total = np * OUT, per = (total + nw - 1u) / nw, lane = threadIdx.x & 31u;
    uint32_t u = gw * per;
    const uint32_t uend = (u + per < total) ? u + per : total;
    const v41_vq_mat m0 = v41_vq_open<1>(blob, 0, 2, OUT, MID, NULL);   /* PDL: codebook first, same as gateup */
    const uint8_t *cb = m0.ok ? m0.cb : NULL;
    if (cb) v41_vq_cb_to_shared(vqsh, cb, cbb);
    v41_pdl_wait();
    if (!cb) { cb = v41_vq_layer_cb(blob, sel, np, 2, OUT, MID); if (!cb) return; v41_vq_cb_to_shared(vqsh, cb, cbb); }
    v41_vq_blk carry; carry.w0 = 0u; carry.w1 = 0u; carry.w2 = 0u; carry.ex = 0u;
    if (u < uend) {
        const int32_t e = sel[u / OUT];
        if (e >= 0) { const v41_vq_mat md = v41_vq_open<1>(blob, e, 2, OUT, MID, NULL); if (md.ok) carry = v41_vq_row_first_blk<NBIT, 1, EXT>(md, u % OUT); }
    }
    v41_vq_cb_to_shared(vqsh + cbb, (const uint8_t *)h, np * MID * 2u);   /* every pair's h (bf16): 6 x 2304 x 2 B = 27 KB */
    __syncthreads();
    h = (const uint32_t *)(vqsh + cbb);
    bool first = true;
    while (u < uend) {
        V41_VQ_SEG(u, uend, OUT, p, r0, n);
        u += n;
        const int32_t e = sel[p];
        if (e < 0) { first = false; continue; }
        const v41_vq_mat md = v41_vq_open<1>(blob, e, 2, OUT, MID, gr ? gr + (size_t)e * OUT : NULL);
        if (!md.ok) { if (lane < n) partial[(uint64_t)p * OUT + r0 + lane] = 0.f; first = false; continue; }   /* bad payload: zeros, as the v2 kernel */
        if (md.cb != cb) __trap();
        if (!first) carry = v41_vq_row_first_blk<NBIT, 1, EXT>(md, r0);
        first = false;
        const uint32_t *hp = h + (uint64_t)p * (MID / 2u);
        float ys;
        v41_vq_stream<NBIT, EXT, 1>(md, r0, n, &hp, 1u, vqsh, &carry, NULL, NULL, &ys);
        if (lane < n) partial[(uint64_t)p * OUT + r0 + lane] = v41_bf16r(ys);
    }
}

/* ==== verify-batch (n_tok >= 2) persistent kernels (2026-09-24) ====
 * Work items change from "(pair, row)" to "(unique expert, row)": the block head
 * groups same-expert pairs from `order` (pairs sorted by expert), the bitstream
 * is read once, codewords decoded once, and the nt tokens of the group each
 * multiply (v41_vq_stream<M>). Codebook: one per layer, one persistent block
 * per SM, copied once — same as the n=1 kernels. Activations/h stay in global
 * (L1): a 13-bit codebook (64 KB) plus 4 rows of activations (40 KB) exceeds
 * the ~99 KB opt-in cap.
 * Registers: the n=1 kernel already sits at 64 (1024 threads); multi-token
 * adds M-1 accumulators + results and reads activations on use, so M=2 fits in
 * 32 warps.
 * Byte-identical: see v41_vq_stream; the gate is the engine's "same track"
 * test (speculative == pure decode, byte for byte). */
#define V41_VQPN_MAXP 64u   /* pair cap: V41_GEMV_MAX_TOK(8) * top-k(6) = 48 */
/* One group holds at most M=2 tokens; larger groups split (measured: fewer
 * warps in flight loses more than the byte savings — 32 warps, M=2, always). */
#define V41_VQPN_M 2u

/* Block head: group same-expert pairs from `order` (q = group start in order,
 * m = pairs in the group, <= V41_VQPN_M); thread 0 does it, the rest wait at
 * the barrier. (Moving this table into v41_vq_order_kernel was measured
 * noise and reverted in the engine.) */
__device__ __forceinline__ static void v41_vqpn_groups(const int32_t *sel, const int32_t *order, uint32_t np,
                                                       uint32_t *gq, uint32_t *gm, uint32_t *ng) {
    if (threadIdx.x == 0) {
        uint32_t g = 0, q = 0;
        while (q < np) {
            const int32_t e = sel[order[q]];
            uint32_t m = 1u;
            while (q + m < np && sel[order[q + m]] == e && m < V41_VQPN_M) m++;
            gq[g] = q; gm[g] = m; g++; q += m;
        }
        *ng = g;
    }
}

template <int NBIT, int EXT, int M, int NW>
__global__ static void __launch_bounds__(NW * 32, 1) v41_vq_gu_persist_n_kernel(uint16_t *h, const uint8_t *blob, const int32_t *sel,
        const int32_t *order, const uint32_t *x, uint32_t IN, uint32_t MID, uint32_t K, uint32_t np, float clamp, uint32_t cbb) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    __shared__ uint32_t gq[V41_VQPN_MAXP], gm[V41_VQPN_MAXP], ng;
    const uint32_t lane = threadIdx.x & 31u;
    const v41_vq_mat m0 = v41_vq_open<1>(blob, 0, 0, MID, IN, NULL);   /* PDL: codebook first; sel/order/x after the wait */
    const uint8_t *cb = m0.ok ? m0.cb : NULL;
    if (cb) v41_vq_cb_to_shared(vqsh, cb, cbb);
    v41_pdl_wait();
    if (!cb) { cb = v41_vq_layer_cb(blob, sel, np, 0, MID, IN); if (!cb) return; v41_vq_cb_to_shared(vqsh, cb, cbb); }
    v41_vqpn_groups(sel, order, np, gq, gm, &ng);
    __syncthreads();   /* the kernel's only block-wide barrier (codebook + group table) */
    const uint32_t gw = blockIdx.x * NW + (threadIdx.x >> 5), nw = gridDim.x * NW;
    const uint32_t total = ng * MID, per = (total + nw - 1u) / nw;
    uint32_t u = gw * per;
    const uint32_t uend = (u + per < total) ? u + per : total;
    while (u < uend) {
        V41_VQ_SEG(u, uend, MID, g, r0, n);
        u += n;
        const uint32_t q = gq[g], nt = gm[g];
        const int32_t e = sel[order[q]];
        if (e < 0) continue;
        const v41_vq_mat mg = v41_vq_open<1>(blob, e, 0, MID, IN, NULL), mu = v41_vq_open<1>(blob, e, 1, MID, IN, NULL);
        if (!mg.ok || !mu.ok) continue;
        if (mg.cb != cb || mu.cb != cb) __trap();
        const uint32_t *xs[M]; uint32_t pr[M];
        #pragma unroll
        for (int j = 0; j < M; j++) {
            pr[j] = (uint32_t)order[q + ((uint32_t)j < nt ? (uint32_t)j : 0u)];   /* empty slots point at the group head: valid pointer, never read/written */
            xs[j] = x + (uint64_t)(pr[j] / K) * (IN / 2u);
        }
        v41_vq_blk carry = v41_vq_row_first_blk<NBIT, 1, EXT>(mg, r0);
        float gs[M], us[M];
        v41_vq_stream<NBIT, EXT, M>(mg, r0, n, xs, nt, vqsh, &carry, v41_vq_row_ptr<NBIT, 1>(mu, r0), EXT ? v41_vq_ext_ptr(mu, r0) : NULL, gs);
        v41_vq_stream<NBIT, EXT, M>(mu, r0, n, xs, nt, vqsh, &carry, NULL, NULL, us);
        if (lane < n) {
            #pragma unroll
            for (int j = 0; j < M; j++)
                if ((uint32_t)j < nt) h[(uint64_t)pr[j] * MID + r0 + lane] = v41_vq_swiglu(v41_bf16r(gs[j]), v41_bf16r(us[j]), clamp);
        }
    }
}

template <int NBIT, int EXT, int M, int NW>
__global__ static void __launch_bounds__(NW * 32, 1) v41_vq_dn_persist_n_kernel(float *partial, const uint8_t *blob, const int32_t *sel,
        const int32_t *order, const uint32_t *h, uint32_t MID, uint32_t OUT, uint32_t np, uint32_t cbb, const float *gr) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    __shared__ uint32_t gq[V41_VQPN_MAXP], gm[V41_VQPN_MAXP], ng;
    const uint32_t lane = threadIdx.x & 31u;
    const v41_vq_mat m0 = v41_vq_open<1>(blob, 0, 2, OUT, MID, NULL);
    const uint8_t *cb = m0.ok ? m0.cb : NULL;
    if (cb) v41_vq_cb_to_shared(vqsh, cb, cbb);
    v41_pdl_wait();
    if (!cb) { cb = v41_vq_layer_cb(blob, sel, np, 2, OUT, MID); if (!cb) return; v41_vq_cb_to_shared(vqsh, cb, cbb); }
    v41_vqpn_groups(sel, order, np, gq, gm, &ng);
    __syncthreads();
    const uint32_t gw = blockIdx.x * NW + (threadIdx.x >> 5), nw = gridDim.x * NW;
    const uint32_t total = ng * OUT, per = (total + nw - 1u) / nw;
    uint32_t u = gw * per;
    const uint32_t uend = (u + per < total) ? u + per : total;
    while (u < uend) {
        V41_VQ_SEG(u, uend, OUT, g, r0, n);
        u += n;
        const uint32_t q = gq[g], nt = gm[g];
        const int32_t e = sel[order[q]];
        if (e < 0) continue;
        uint32_t pr[M]; const uint32_t *hs[M];
        #pragma unroll
        for (int j = 0; j < M; j++) {
            pr[j] = (uint32_t)order[q + ((uint32_t)j < nt ? (uint32_t)j : 0u)];
            hs[j] = h + (uint64_t)pr[j] * (MID / 2u);
        }
        const v41_vq_mat md = v41_vq_open<1>(blob, e, 2, OUT, MID, gr ? gr + (size_t)e * OUT : NULL);
        if (!md.ok) {                                              /* bad payload: zeros (the tail reads them) */
            if (lane < n) for (uint32_t j = 0; j < nt; j++) partial[(uint64_t)pr[j] * OUT + r0 + lane] = 0.f;
            continue;
        }
        if (md.cb != cb) __trap();
        v41_vq_blk carry = v41_vq_row_first_blk<NBIT, 1, EXT>(md, r0);
        float ys[M];
        v41_vq_stream<NBIT, EXT, M>(md, r0, n, hs, nt, vqsh, &carry, NULL, NULL, ys);
        if (lane < n) {
            #pragma unroll
            for (int j = 0; j < M; j++) if ((uint32_t)j < nt) partial[(uint64_t)pr[j] * OUT + r0 + lane] = v41_bf16r(ys[j]);
        }
    }
}

/* One (M, NW) instance's opt-in + launch. The opt-in is recorded per instance
 * (template statics are per instantiation; both codebook sizes coexist). */
template <int NBIT, int EXT, int M, int NW>
static int v41_vq_persist_n_go(int stage, uint16_t *h, float *part, const uint8_t *blob, const int32_t *sel, const int32_t *ord,
                               const uint32_t *xb, uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t K, uint32_t np, float clamp,
                               uint32_t cbb, const float *gr, int nsm) {
    static uint32_t s_optin[2] = { 0u, 0u };
    if (s_optin[stage] < cbb) {
        (void)cudaGetLastError();   /* the opt-in must not see a previously latched error (see ds41_vq_decode.cuh) */
        const cudaError_t e = stage == 0
            ? cudaFuncSetAttribute(v41_vq_gu_persist_n_kernel<NBIT, EXT, M, NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb)
            : cudaFuncSetAttribute(v41_vq_dn_persist_n_kernel<NBIT, EXT, M, NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb);
        if (e != cudaSuccess) {
            fprintf(stderr, "ds4: [ds41] VQ verify-batch persistent kernel (M=%d) cannot opt in %u B dynamic shared: %s\n", M, cbb, cudaGetErrorString(e));
            (void)cudaGetLastError();
            return 0;
        }
        s_optin[stage] = cbb;
    }
    if (stage == 0) {
        v41_pdl_register((const void *)v41_vq_gu_persist_n_kernel<NBIT, EXT, M, NW>);   /* v41_pdl_wait before touching sel/order/x */
        v41_vq_gu_persist_n_kernel<NBIT, EXT, M, NW><<<(unsigned)nsm, NW * 32, cbb, ds4_current_stream()>>>(h, blob, sel, ord, xb, IN, MID, K, np, clamp, cbb);
    } else {
        v41_pdl_register((const void *)v41_vq_dn_persist_n_kernel<NBIT, EXT, M, NW>);
        v41_vq_dn_persist_n_kernel<NBIT, EXT, M, NW><<<(unsigned)nsm, NW * 32, cbb, ds4_current_stream()>>>(part, blob, sel, ord, (const uint32_t *)h, MID, OUT, np, cbb, gr);
    }
    return cuda_ok(cudaGetLastError(), stage == 0 ? "v41 vq gateup persist (verify batch)" : "v41 vq down persist (verify batch)");
}

/* 1 = launched; 0 = not applicable or the shared opt-in failed (caller falls
 * back to pair + group kernels). Group size is always <= V41_VQPN_M, independent
 * of batch size => one instance. */
template <int NBIT, int EXT>
static int v41_vq_persist_n_launch(int stage, uint32_t n_tok, uint16_t *h, float *part, const uint8_t *blob, const int32_t *sel,
                                   const int32_t *ord, const uint32_t *xb, uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t K,
                                   uint32_t np, float clamp, uint32_t cbb, const float *gr) {
    static int s_nsm = 0;
    if (np > V41_VQPN_MAXP || n_tok < 2u) return 0;
    if (!s_nsm && cudaDeviceGetAttribute(&s_nsm, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    return v41_vq_persist_n_go<NBIT, EXT, (int)V41_VQPN_M, 32>(stage, h, part, blob, sel, ord, xb, IN, MID, OUT, K, np, clamp, cbb, gr, s_nsm);
}
#undef V41_VQ_SEG

/* Launch: stage 0 = gateup, 1 = down. 1 = launched; 0 = launch failure (the
 * caller treats it as an error, never falls back to the old kernels). The
 * opt-in is per instance ("granted so far"), the same trap as
 * v41_vq_fused_moe_n's. */
template <int NBIT, int EXT>
static int v41_vq_persist_launch(int stage, uint16_t *h, float *part, const uint8_t *blob, const int32_t *sel, const uint32_t *xb,
                                 uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t np, float clamp, uint32_t cbb, const float *gr) {
    static uint32_t s_optin[2] = { 0u, 0u };   /* [gateup, down] granted so far */
    static int s_nsm = 0;
    const uint32_t need = cbb + (stage == 0 ? IN * 2u : np * MID * 2u);   /* codebook + activations (bf16) */
    if (!s_nsm && cudaDeviceGetAttribute(&s_nsm, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) {
        fprintf(stderr, "ds4: [ds41] VQ persistent kernel cannot read SM count: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 0;
    }
    if (s_optin[stage] < need) {
        (void)cudaGetLastError();   /* the opt-in must not see a previously latched error (see ds41_vq_decode.cuh) */
        const cudaError_t e = stage == 0
            ? cudaFuncSetAttribute(v41_vq_gu_persist_kernel<NBIT, EXT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)need)
            : cudaFuncSetAttribute(v41_vq_dn_persist_kernel<NBIT, EXT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)need);
        if (e != cudaSuccess) {
            fprintf(stderr, "ds4: [ds41] VQ persistent kernel cannot opt in %u B dynamic shared (codebook %u + activations): %s\n", need, cbb, cudaGetErrorString(e));
            (void)cudaGetLastError();
            return 0;
        }
        s_optin[stage] = need;
    }
    v41_pdl_register((const void *)v41_vq_gu_persist_kernel<NBIT, EXT>);   /* both kernels v41_pdl_wait before sel/x/h */
    v41_pdl_register((const void *)v41_vq_dn_persist_kernel<NBIT, EXT>);
    if (stage == 0) v41_vq_gu_persist_kernel<NBIT, EXT><<<(unsigned)s_nsm, V41_VQ_WARPS * 32u, need, ds4_current_stream()>>>(h, blob, sel, xb, IN, MID, np, clamp, cbb);
    else v41_vq_dn_persist_kernel<NBIT, EXT><<<(unsigned)s_nsm, V41_VQ_WARPS * 32u, need, ds4_current_stream()>>>(part, blob, sel, (const uint32_t *)h, MID, OUT, np, cbb, gr);
    return cuda_ok(cudaGetLastError(), stage == 0 ? "v41 vq gateup persist" : "v41 vq down persist");
}
