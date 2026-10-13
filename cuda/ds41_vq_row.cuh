/* ds41_vq_row.cuh — V4.1 VQ payload geometry and the row dot product, ported
 * from the C engine at /data/YoungAi (commit 3946dbc), file
 * `src/cuda/cuda_vq_row.inc.cu` (whole file; split out of cuda_vq_decode in the
 * engine, 2026-09-21).
 *
 * This family (v41_vq_open / v41_vq_dot8 / v41_vq_row_dot) is the ONLY consumer
 * of the on-disk payload layout; the v3 layout changes touched nothing else.
 *
 * Two on-disk versions selected by the V3 template parameter, never a runtime
 * branch (measured in the engine: one runtime `order ? :` cost 21% at n=1):
 *   V3=0  DQVL v2: payload = [16 B header][codebook nc*8 f16][row gains f16]
 *         [NBIT packed bitstream][8 B pad]; the codebook travels per payload.
 *   V3=1  DQVL v3: payload = [32 B header with flags/main-bit-width/cb_off]
 *         [row gains][fixed 12-bit main stream][for 13-bit layers the 1-bit
 *         plane][8 B pad]; the codebook is ONE PER LAYER behind the blob header
 *         (cb_off), E4M3 only (1 B/element).
 *   For V3=1, NBIT is the effective width (12 or 13); the main stream is
 *   always 12 bits, so the block geometry (3 whole 128 B lines = 96 words =
 *   256 indices = 8 rounds) is untouched. The 13th bit travels as a bit plane:
 *   one block = 256 indices = 32 B = 8 words, lane j<8 reads one word; round k
 *   needs "bit 13 of index 32k+lane" = bit `lane` of the block's word k => one
 *   shfl + shift/and. One extra register per lane only (the kernel sits on the
 *   64-register wall at 1024 threads/block; anything more is negative).
 *
 * Requires: ds41_primitives.cuh (v41_bf16r, v41_pdl_wait), ds41_vq_fmt.h (the
 * payload magics), and it must precede ds41_vq_decode.cuh (which uses this
 * file's v41_vq_mat / row_dot).
 */
#pragma once
#include <stdint.h>
#include <string.h>

#include "ds41_primitives.cuh"
#include "../ds41_vq_fmt.h"

/* gov: per-row gain override [rows] f32 (the sidecar zchain gain); NULL =
 * payload as-is. ex: the v3 13-bit plane (NULL for 12-bit layers and all v2
 * payloads). */
typedef struct { const uint8_t *cb, *gr, *ix, *ex; const float *gov; uint32_t nidx_row, nbit, imsk, nc; int ok; } v41_vq_mat;

/* Parse one expert matrix from the blob (`cuda_vq_row.inc.cu:19-49`). A version
 * mismatch (magic) yields ok=0 and the caller writes zeros or skips — never a
 * hard decode under the wrong layout. */
template <int V3>
__device__ __forceinline__ static v41_vq_mat v41_vq_open(const uint8_t *blob, int e, int which, uint32_t rows, uint32_t cols,
                                                         const float *gov) {
    v41_vq_mat m; m.ok = 0; m.cb = m.gr = m.ix = m.ex = NULL; m.gov = gov; m.nidx_row = m.nbit = m.imsk = m.nc = 0;
    uint64_t off; memcpy(&off, blob + 16 + ((size_t)e * 3 + which) * 8, 8);
    if (!off) return m;
    const uint8_t *pay = blob + off;
    uint32_t mg; memcpy(&mg, pay, 4);
    if (mg != (V3 ? DS4VQ_MAT3_MAGIC : DS4VQ_MAT_MAGIC)) return m;
    uint16_t d16, n16; memcpy(&d16, pay + 4, 2); memcpy(&n16, pay + 6, 2);
    uint32_t r32, c32; memcpy(&r32, pay + 8, 4); memcpy(&c32, pay + 12, 4);
    if (r32 != rows || c32 != cols || d16 != 8u) return m;   /* this kernel only writes dim 8 (the vq8 recipe) */
    uint32_t nbit = 0; while ((1u << nbit) < (uint32_t)n16) nbit++; if (nbit < 1u) nbit = 1u;
    m.nidx_row = cols / 8u; m.nbit = nbit; m.imsk = (1u << nbit) - 1u; m.nc = n16;
    if (V3) {
        uint32_t flags, mnb; memcpy(&flags, pay + 16, 4); memcpy(&mnb, pay + 20, 4);
        uint64_t cb_off; memcpy(&cb_off, pay + 24, 8);
        if (mnb != 12u || !(flags & 1u)) return m;            /* main stream must be 12-bit, codebook E4M3 */
        m.cb = blob + cb_off;                                 /* layer codebook: shared by the layer's three matrices */
        m.gr = pay + 32;
        m.ix = m.gr + (size_t)rows * 2u;
        const uint32_t mrow = (m.nidx_row * 12u + 7u) / 8u;
        m.ex = (flags & 2u) ? m.ix + (size_t)rows * mrow : NULL;
        if (((flags & 2u) != 0u) != (nbit > 12u)) return m;   /* plane presence must match the width from nc */
    } else {
        m.cb = pay + 16; m.gr = m.cb + (size_t)n16 * 8u * 2u; m.ix = m.gr + (size_t)rows * 2u;
    }
    m.ok = 1;
    return m;
}

/* E4M3 -> half2 via the hardware instruction (`cuda_vq_row.inc.cu:75-79`; the
 * scalar table was numerically right but cost 26% per step and 20% prefill).
 * E4M3 subset of f16, so this is an exact conversion — instruction count, not
 * semantics. sm_89+ (GB10 = sm_121, the local 4070 = sm_89). */
__device__ __forceinline__ static __half2 v41_e4m3x2_to_half2(uint32_t two) {
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((unsigned short)(two & 0xffffu)));
    __half2 out; memcpy(&out, &h, 4);
    return out;
}

/* Codeword v -> 8 f32 (`cuda_vq_row.inc.cu:84-99`). Split out of dot8 so the
 * multi-token group kernel decodes a codeword once and dots it per token; the
 * single-token kernel keeps dot8 = cw + dot8_cw, the same expression tree. */
template <int FP8>
__device__ __forceinline__ static void v41_vq_cw(uint32_t v, const uint8_t *cbs, int cb_shared, float *c) {
    if (FP8) {
        const uint2 w = *(const uint2 *)(cbs + (size_t)v * 8u);
        const __half2 p0 = v41_e4m3x2_to_half2(w.x), p1 = v41_e4m3x2_to_half2(w.x >> 16),
                      p2 = v41_e4m3x2_to_half2(w.y), p3 = v41_e4m3x2_to_half2(w.y >> 16);
        const float2 e0 = __half22float2(p0), e1 = __half22float2(p1), e2 = __half22float2(p2), e3 = __half22float2(p3);
        c[0] = e0.x; c[1] = e0.y; c[2] = e1.x; c[3] = e1.y; c[4] = e2.x; c[5] = e2.y; c[6] = e3.x; c[7] = e3.y;
    } else {
        uint2 cw0, cw1;
        if (cb_shared) { const uint4 q = *(const uint4 *)(cbs + (size_t)v * 16u); cw0.x = q.x; cw0.y = q.y; cw1.x = q.z; cw1.y = q.w; }
        else { cw0 = *(const uint2 *)(cbs + (size_t)v * 16u); cw1 = *(const uint2 *)(cbs + (size_t)v * 16u + 8u); }
        __half2 h0, h1, h2, h3; memcpy(&h0, &cw0.x, 4); memcpy(&h1, &cw0.y, 4); memcpy(&h2, &cw1.x, 4); memcpy(&h3, &cw1.y, 4);
        const float2 f0 = __half22float2(h0), f1 = __half22float2(h1), f2 = __half22float2(h2), f3 = __half22float2(h3);
        c[0] = f0.x; c[1] = f0.y; c[2] = f1.x; c[3] = f1.y; c[4] = f2.x; c[5] = f2.y; c[6] = f3.x; c[7] = f3.y;
    }
}

/* 8-element multiply-add (`cuda_vq_row.inc.cu:102-106`). THIS ONE EXPRESSION
 * decides bit-identity with the reference: every kernel goes through it and
 * nobody may keep a second copy (under fast-math a differently written sum
 * changes the FMA merge and the accumulation order). A uint32 holds two bf16:
 * low half element 2k, high half 2k+1; the shifts restore f32 by zero-fill. */
__device__ __forceinline__ static float v41_vq_dot8_cw(const float *c, uint4 xw) {
    return c[0] * __uint_as_float(xw.x << 16) + c[1] * __uint_as_float(xw.x & 0xffff0000u)
         + c[2] * __uint_as_float(xw.y << 16) + c[3] * __uint_as_float(xw.y & 0xffff0000u)
         + c[4] * __uint_as_float(xw.z << 16) + c[5] * __uint_as_float(xw.z & 0xffff0000u)
         + c[6] * __uint_as_float(xw.w << 16) + c[7] * __uint_as_float(xw.w & 0xffff0000u);
}

template <int FP8>
__device__ __forceinline__ static float v41_vq_dot8(uint32_t v, uint4 xw, const uint8_t *cbs, int cb_shared) {
    float c[8];
    v41_vq_cw<FP8>(v, cbs, cb_shared, c);
    return v41_vq_dot8_cw(c, xw);
}

/* f32 (already on the bf16 grid) -> packed bf16. One thread per element, once
 * per layer (`cuda_vq_row.inc.cu:116-120`). */
__global__ static void v41_vq_xpack_kernel(uint16_t *dst, const float *src, uint64_t n) {
    v41_pdl_wait();   /* PDL: wait for upstream first (no-op when not PDL-launched) */
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (uint16_t)(__float_as_uint(v41_bf16r(src[i])) >> 16);
}

/* The bitstream is read as whole 384 B blocks (3 x 128 B lines) and distributed
 * by shfl (`cuda_vq_row.inc.cu:122-137`). Measured basis: a 48 B-per-warp
 * stream runs at 135-142 GB/s no matter the depth; a full 128 B line per
 * instruction reaches 219 (wall 237-241) — the last wall is the request
 * granularity, DRAM fetches 64 B. A block = 3 lines = 96 words = 256 indices =
 * 8 rounds; lane j takes words j / j+32 / j+64 with three LDG.32, one line per
 * instruction. Round k's index j+32k lives in word 12k+floor(3j/8), shift
 * (12j)%32; the two lanes per 8 that cross (j%8 in {2,5}) need the second shfl.
 * Lane<->index map, per-lane accumulation order and the cross-lane reduction
 * tree are unchanged => byte-identical output, gate = cmp. */
__device__ __forceinline__ static uint32_t v41_vq_sel3(uint32_t r, uint32_t w0, uint32_t w1, uint32_t w2) {
    return r == 0u ? w0 : (r == 1u ? w1 : w2);
}

/* A bitstream block in registers: lane j holds words j / j+32 / j+64; ex = the
 * plane block's word for the lane (valid for lane<8 only). */
typedef struct { uint32_t w0, w1, w2, ex; } v41_vq_blk;

/* Main-stream width: v3 always 12 (the 13th travels the plane); v2 is NBIT. */
template <int NBIT, int V3> __device__ __forceinline__ static constexpr uint32_t v41_vq_mbit() { return V3 ? 12u : (uint32_t)NBIT; }

template <int NBIT, int V3>
__device__ __forceinline__ static const uint32_t *v41_vq_row_ptr(const v41_vq_mat &m, uint32_t r) {
    return (const uint32_t *)(m.ix + (((uint64_t)r * m.nidx_row * v41_vq_mbit<NBIT, V3>()) >> 3));
}

/* Plane row start (v3 13-bit layers only): (nidx_row+7)/8 bytes per row. */
__device__ __forceinline__ static const uint32_t *v41_vq_ext_ptr(const v41_vq_mat &m, uint32_t r) {
    return (const uint32_t *)(m.ex + (size_t)r * ((m.nidx_row + 7u) >> 3));
}

/* Load one block: only the words that exist (48 in the tail block); lanes past
 * the end issue no read — the payload's 8 B zero pad is not relied on.
 * EXT=1 adds one LDG.32: a 256-index block's plane is 32 B = 8 words, lane j<8
 * takes one (nwx = words that exist in this block). */
template <int EXT>
__device__ __forceinline__ static v41_vq_blk v41_vq_blk_load(const uint32_t *blk, uint32_t nw, const uint32_t *bex, uint32_t nwx) {
    const uint32_t lane = threadIdx.x & 31u;
    v41_vq_blk b; b.w0 = 0u; b.w1 = 0u; b.w2 = 0u; b.ex = 0u;
    if (lane < nw) b.w0 = blk[lane];
    if (lane + 32u < nw) b.w1 = blk[lane + 32u];
    if (lane + 64u < nw) b.w2 = blk[lane + 64u];
    if (EXT && bex && lane < nwx) b.ex = bex[lane];
    return b;
}

/* Cross-block software pipeline, one block ahead (`cuda_vq_row.inc.cu:163-175`):
 * after whole-block reads landed, ncu still showed long_scoreboard 17-18 — the
 * 32 warps move in lockstep through three __syncthreads, all waiting on DRAM
 * together; issuing the next block's 3 words before computing this one hides
 * that. Two further escalations were measured negative and are archived in the
 * engine (activation preload; a two-block queue) — this kernel is pinned at the
 * 64-register wall and any extra register is negative. */
template <int NBIT, int V3, int EXT>
__device__ __forceinline__ static v41_vq_blk v41_vq_row_first_blk(const v41_vq_mat &m, uint32_t r) {
    const uint32_t R = m.nidx_row >> 5, nr = (R < 8u ? R : 8u);
    return v41_vq_blk_load<EXT>(v41_vq_row_ptr<NBIT, V3>(m, r), nr * v41_vq_mbit<NBIT, V3>(),
                                EXT ? v41_vq_ext_ptr(m, r) : NULL, nr);
}

/* Up to 8 rounds of one block: round k, lane j takes index 32(b+k)+j; two shfls
 * fetch the words, funnelshift extracts 12 bits, table lookup and dot. The
 * multiply-add order is unchanged from the reference kernel.
 * EXT=1: round k's 13th bit = bit `lane` of the block's word k (held by lane k)
 * => one shfl + shift/and, zero extra memory traffic. */
template <int NBIT, int V3, int EXT>
__device__ __forceinline__ static void v41_vq_blk_rounds(const v41_vq_blk &cur, uint32_t rounds, uint32_t a, uint32_t sh, uint32_t imsk,
                                                         const uint32_t *xb, const uint8_t *cbs, int cb_shared, float &acc) {
    const uint32_t lane = threadIdx.x & 31u;
    constexpr uint32_t MB = v41_vq_mbit<NBIT, V3>();
    #pragma unroll
    for (uint32_t k = 0; k < 8u; k++) {
        if (k >= rounds) break;
        const uint4 xa = *(const uint4 *)(xb + (size_t)(k * 32u + lane) * 4u);   /* 8 bf16 = 16 B, L1 hit */
        const uint32_t f = k * MB;                                       /* first word of this round (compile-time constant) */
        const uint32_t lo_r = ((f & 31u) > 32u - MB) ? ((f + 31u - lane) >> 5) : (f >> 5);
        const uint32_t hi_r = (((f + 1u) & 31u) > 32u - MB) ? ((f + 32u - lane) >> 5) : ((f + 1u) >> 5);
        const uint32_t lo = __shfl_sync(0xffffffffu, v41_vq_sel3(lo_r, cur.w0, cur.w1, cur.w2), (int)(f + a));
        const uint32_t hi = __shfl_sync(0xffffffffu, v41_vq_sel3(hi_r, cur.w0, cur.w1, cur.w2), (int)(f + a + 1u));
        uint32_t v = __funnelshift_r(lo, hi, sh) & (EXT ? 0xFFFu : imsk);
        if (EXT) {
            const uint32_t exw = __shfl_sync(0xffffffffu, cur.ex, (int)k);
            v |= ((exw >> lane) & 1u) << 12;
        }
        acc += v41_vq_dot8<V3>(v, xa, cbs, cb_shared);
    }
}

/* One warp dots one row with the activation x: index j belongs to lane j%32,
 * one round covers 32 indices. Returns gain * sum (warp-reduced). NBIT =
 * codebook width (4096 words = 12; v3 13-bit layers = 13, but the main stream
 * stays 12 plus the plane).
 * carry in = this row's first block, out = next row's first block (NULL = empty
 * block); each block's words issue before the previous block computes.
 * nextex = the next row's plane start (EXT=1 MUST pair it with next — otherwise
 * every next row's 13th bit reads as 0, which is silently wrong weights). */
template <int NBIT, int V3, int EXT>
__device__ __forceinline__ static float v41_vq_row_dot(const v41_vq_mat &m, uint32_t r, const uint32_t *xs, const uint8_t *cbs, int cb_shared,
                                                       v41_vq_blk *carry, const uint32_t *next, const uint32_t *nextex) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t R = m.nidx_row >> 5;                                  /* rounds: gateup 20, down 9 */
    constexpr uint32_t MB = v41_vq_mbit<NBIT, V3>();
    const uint32_t a = (lane * MB) >> 5, sh = (lane * MB) & 31u;         /* this lane's word index and shift in a round */
    const uint32_t *row = v41_vq_row_ptr<NBIT, V3>(m, r);
    const uint32_t *rex = EXT ? v41_vq_ext_ptr(m, r) : NULL;
    v41_vq_blk cur = *carry;
    float acc = 0.f;
    for (uint32_t b = 0; b < R; b += 8u) {
        v41_vq_blk nxt;
        if (b + 8u < R) {
            const uint32_t rem = R - b - 8u, nr = (rem < 8u ? rem : 8u);
            /* The plane advances ONE WORD PER GROUP, not per block (fixed
             * 2026-09-22; this was the v3 13-bit decode path running 42% PPL
             * high): b is the group number (32 indices), a block = 8 groups =
             * 256 indices = 8 plane words, so the next block's words start at
             * rex + (b+8). The original wrote rex + (b+8)/8 (block number), so
             * only each row's first block had correct 13th bits. Wrong index
             * +-4096 = a different codeword = a whole wrong weight slice.
             * Only 13-bit v3 layers (EXT=1) show it; 12-bit and all v2 are
             * clean, so the old models never exposed it. */
            nxt = v41_vq_blk_load<EXT>(row + (size_t)(b + 8u) * MB, nr * MB, EXT ? rex + (b + 8u) : NULL, nr);
        }
        else if (next) { const uint32_t nr = (R < 8u ? R : 8u); nxt = v41_vq_blk_load<EXT>(next, nr * MB, nextex, nr); }
        else { nxt.w0 = 0u; nxt.w1 = 0u; nxt.w2 = 0u; nxt.ex = 0u; }
        v41_vq_blk_rounds<NBIT, V3, EXT>(cur, (R - b < 8u) ? R - b : 8u, a, sh, m.imsk, xs + (size_t)b * 32u * 4u, cbs, cb_shared, acc);
        cur = nxt;
    }
    *carry = cur;
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    /* The row gain is read at the row's end (moving it to the head costs a
     * register and 6% on gateup in the engine's measurement). */
    __half gh; memcpy(&gh, m.gr + (size_t)r * 2u, 2);
    return acc * __half2float(gh) * (m.gov ? m.gov[r] : 1.0f);
}
