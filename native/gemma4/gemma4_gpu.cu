/* gemma4_gpu.cu - device residency and the fused VQ row dot.
 *
 * The arithmetic is the VQ tools' reference device kernel, ported for this
 * artifact's payload and checked against the CPU forward on the same input.
 * Nothing here decodes a weight matrix: the 9.0 GB artifact stays in its
 * containers, and a projection reads indices, codebook words and gains.
 *
 * Sources read first-hand before this file was written:
 * - /data/VQ_tools/c/gpu/decode-gpu.cu: the DQV3 block kernel (gemv_v3blk,
 *   :285-333 with v3_rounds at :246-283), the dim-4 fast path (gemv_warp4,
 *   :199-237), the generic warp kernel (:619-656), the index reader
 *   (rd_idx_fmt, :110-133) and the selection rule (v3_blk_ok, :66-68).
 * - /data/VQ_tools/crates/vq-format/src/lib.rs:314-361, :426-507: the tools'
 *   payload header and the min(nbit, 12)-bit row stream with its 13th-bit
 *   plane, which is the layout the artifact carries.
 * - crates/ds4-core/src/vq.rs (this worktree) and gemma4_a4b/decode.rs: the
 *   same field rules on the Rust side, and from_payload's refusal of the
 *   engine's DQV3 (a non-zero word at payload offset 20).
 *
 * The one deliberate difference from the reference: the reference takes one
 * payload per buffer and expects 8 bytes of zeroed slack past it. Here a
 * whole container is uploaded with every payload start on a 256-byte
 * boundary and its own zeroed 64-byte tail, because the container is what
 * the release hands us. The kernels never see that framing.
 *
 * The routed gather (gm4_gpu_rows) is the same kernels with a row list: a
 * warp's row is rl[lo] instead of row0 + lo, so a mixture layer reads its
 * eight chosen expert slices and nothing else.
 */
#include "gemma4_gpu.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr int CONTAINER_HEADER = 32; /* RQVQ, vq-format/src/lib.rs:314-361 */
constexpr int PAYLOAD_HEADER = 24;   /* the tools' DQV3 */
constexpr int SLACK = 64;            /* zeroed tail behind every payload */
constexpr size_t PAYLOAD_AT = 256;   /* device alignment of each payload */

/* Payload fields as the kernels need them; the same names and meaning as the
 * reference's `layout`. Offsets are relative to the payload start. */
struct Layout {
    int dim, nc, rows, cols, nbit, v3;
    int row0, nrows;
    /* A gather whose input differs per row block: row `lo` reads the
     * activation at (lo / xrows) * xcols rather than at the one x pointer.
     * Zero means one input for every row, which is every case but the routed
     * mixture's down projection, where each expert has its own activation. */
    int xrows, xcols;
    /* The first stage of a tensor assigns its row instead of accumulating into
     * it, so the caller does not have to clear the output first. Parsed
     * layouts leave it 0 (accumulate). */
    int first;
    unsigned mask;
    size_t cb_off, gain_off, stream_off, nidx, nidx_row;
    size_t mrow, prow;
};

struct Payload {
    const uint8_t *dev;
    Layout l;
    int variant;
    size_t bytes;
};

struct Tensor {
    Payload *pays;
    int npay;
};

int fail(char *err, size_t len, const char *what, cudaError_t e) {
    snprintf(err, len, "%s: %s", what, cudaGetErrorString(e));
    return -1;
}

uint16_t u16le(const uint8_t *b, size_t at) {
    return (uint16_t) (b[at] | ((uint16_t) b[at + 1] << 8));
}

uint32_t u32le(const uint8_t *b, size_t at) {
    return (uint32_t) b[at] | ((uint32_t) b[at + 1] << 8)
         | ((uint32_t) b[at + 2] << 16) | ((uint32_t) b[at + 3] << 24);
}

uint64_t u64le(const uint8_t *b, size_t at) {
    return (uint64_t) u32le(b, at) | ((uint64_t) u32le(b, at + 4) << 32);
}

size_t align_to(size_t value, size_t a) { return (value + a - 1) / a * a; }

/* ---- the reference kernels, ported ------------------------------------
 *
 * `layout`/`rd_idx`/`rd_idx_fmt`/`v3blk`/`v3_sel3`/`v3_blk_load`/`v3_rounds`
 * are the VQ tools' code (/data/VQ_tools/c/gpu/decode-gpu.cu:86-333). The
 * row walk, the funnel shifts and the gain-after-the-sum are unchanged; the
 * only edits are the struct name and a comment pointing at the source.
 */
__device__ __forceinline__ uint32_t rd_idx(const uint8_t *s, size_t i, int nbit, uint32_t mask) {
    if (nbit == 8) return s[i];
    const size_t bit = i * (size_t) nbit;
    const uint32_t w = (uint32_t) s[bit >> 3]
                     | ((uint32_t) s[(bit >> 3) + 1] << 8)
                     | ((uint32_t) s[(bit >> 3) + 2] << 16);
    return (w >> (bit & 7)) & mask;
}

__device__ __forceinline__ uint32_t rd_idx_fmt(const uint8_t *st, const Layout &L, size_t r, size_t i) {
    if (!L.v3) return rd_idx(st, r * (size_t) (L.cols / L.dim) + i, L.nbit, L.mask);
    const uint32_t mb = L.nbit > 12 ? 12u : (uint32_t) L.nbit;
    const uint8_t *base = st + r * L.mrow;
    const size_t bit = i * mb;
    const uint8_t *q = base + (bit >> 3);
    const uint32_t w = (uint32_t) q[0] | ((uint32_t) q[1] << 8) | ((uint32_t) q[2] << 16) | ((uint32_t) q[3] << 24);
    uint32_t v = (w >> (bit & 7)) & ((1u << mb) - 1u);
    if (L.prow) {
        const uint8_t *pl = st + L.mrow * (size_t) L.rows + r * L.prow;
        v |= (uint32_t) ((pl[i >> 3] >> (i & 7)) & 1u) << 12;
    }
    return v;
}

struct v3blk { uint32_t w0, w1, w2, ex; };

__device__ __forceinline__ uint32_t v3_sel3(uint32_t r, uint32_t w0, uint32_t w1, uint32_t w2) {
    return r == 0u ? w0 : (r == 1u ? w1 : w2);
}

__device__ __forceinline__ v3blk v3_blk_load(const uint32_t *blk, uint32_t nw,
                                             const uint32_t *bex, uint32_t nwx) {
    const uint32_t lane = threadIdx.x & 31u;
    v3blk b;
    b.w0 = b.w1 = b.w2 = b.ex = 0u;
    if (lane < nw) b.w0 = blk[lane];
    if (lane + 32u < nw) b.w1 = blk[lane + 32u];
    if (lane + 64u < nw) b.w2 = blk[lane + 64u];
    if (bex && lane < nwx) b.ex = bex[lane];
    return b;
}

template <int DIM, int EXT>
__device__ __forceinline__ void v3_rounds(const v3blk &cur, uint32_t rounds, uint32_t a, uint32_t sh,
                                          const float *xb, const __half *cb, int dim, float &acc) {
    const uint32_t lane = threadIdx.x & 31u;
    #pragma unroll
    for (uint32_t k = 0; k < 8u; k++) {
        if (k >= rounds) break;
        const uint32_t f = k * 12u;
        const uint32_t lo_r = ((f & 31u) > 32u - 12u) ? ((f + 31u - lane) >> 5) : (f >> 5);
        const uint32_t hi_r = (((f + 1u) & 31u) > 32u - 12u) ? ((f + 32u - lane) >> 5) : ((f + 1u) >> 5);
        const uint32_t lo = __shfl_sync(0xffffffffu, v3_sel3(lo_r, cur.w0, cur.w1, cur.w2), (int) (f + a));
        const uint32_t hi = __shfl_sync(0xffffffffu, v3_sel3(hi_r, cur.w0, cur.w1, cur.w2), (int) (f + a + 1u));
        uint32_t v = __funnelshift_r(lo, hi, sh) & 0xFFFu;
        if (EXT) {
            const uint32_t exw = __shfl_sync(0xffffffffu, cur.ex, (int) k);
            v |= ((exw >> lane) & 1u) << 12;
        }
        const size_t g = (size_t) (k * 32u + lane);
        if (DIM == 4) {
            const float4 xv = *(const float4 *) (xb + g * 4u);
            const uint2 c = __ldg((const uint2 *) cb + v);
            const float2 f0 = __half22float2(*reinterpret_cast<const __half2 *> (&c.x));
            const float2 f1 = __half22float2(*reinterpret_cast<const __half2 *> (&c.y));
            acc += f0.x * xv.x + f0.y * xv.y + f1.x * xv.z + f1.y * xv.w;
        } else if (DIM == 8) {
            const float4 x0 = *(const float4 *) (xb + g * 8u);
            const float4 x1 = *(const float4 *) (xb + g * 8u + 4u);
            const uint2 c0 = __ldg((const uint2 *) cb + (size_t) v * 2u);
            const uint2 c1 = __ldg((const uint2 *) cb + (size_t) v * 2u + 1u);
            const float2 f0 = __half22float2(*reinterpret_cast<const __half2 *> (&c0.x));
            const float2 f1 = __half22float2(*reinterpret_cast<const __half2 *> (&c0.y));
            const float2 f2 = __half22float2(*reinterpret_cast<const __half2 *> (&c1.x));
            const float2 f3 = __half22float2(*reinterpret_cast<const __half2 *> (&c1.y));
            acc += f0.x * x0.x + f0.y * x0.y + f1.x * x0.z + f1.y * x0.w
                 + f2.x * x1.x + f2.y * x1.y + f3.x * x1.z + f3.y * x1.w;
        } else {
            const __half *cw = cb + (size_t) v * (size_t) dim;
            const float *xg = xb + g * (size_t) dim;
            for (int d = 0; d < dim; d++) acc += __half2float(cw[d]) * xg[d];
        }
    }
}

/* Which row of the payload a warp walks: the next contiguous row, or the
 * row the caller's gather list names. The list is uniform across a launch,
 * so the branch costs one predicate per row, not per index. */
__device__ __forceinline__ size_t row_of(const Layout &L, uint32_t lo, const uint32_t *rl) {
    return rl ? (size_t) rl[lo] : (size_t) L.row0 + (size_t) lo;
}

/* The activation a row reads. Every matvec and every single-input gather
 * reads the one x pointer; the routed mixture's down projection is the case
 * where each expert block of rows has an activation of its own, which the
 * caller declares with xrows (rows per block) and xcols (elements to skip
 * between blocks). */
__device__ __forceinline__ const float *row_x(const float *x, const Layout &L, uint32_t lo) {
    if (L.xrows <= 0) return x;
    return x + (size_t) (lo / (uint32_t) L.xrows) * (size_t) L.xcols;
}

/* One row of one payload: the block kernel's arithmetic exactly, the eight
 * 32-index rounds per block, the cross-block prefetch, the lane<->index map
 * of `v3_rounds` and the gain after the sum. Extracted so the block kernel
 * and the persistent expert walk run one code path instead of two that could
 * drift; the gate's layer bands are the check that they did not. */
template <int DIM, int EXT>
__device__ __forceinline__ float v3blk_row_dot(const uint8_t *p, const Layout &L, size_t r,
                                               const float *xr) {
    const uint32_t lane = threadIdx.x & 31u;
    const __half *cb = (const __half *) (p + L.cb_off);
    const __half *gr = (const __half *) (p + L.gain_off);
    const uint32_t *row = (const uint32_t *) (p + L.stream_off + (size_t) r * L.mrow);
    const uint32_t *rex = EXT
        ? (const uint32_t *) (p + L.stream_off + L.mrow * (size_t) L.rows + (size_t) r * L.prow)
        : NULL;
    const uint32_t R = (uint32_t) (L.nidx_row >> 5);
    const uint32_t nblk = (R + 7u) >> 3;
    const uint32_t a = (lane * 12u) >> 5, sh = (lane * 12u) & 31u;

    v3blk cur = v3_blk_load(row, R >= 8u ? 96u : R * 12u, rex, R >= 8u ? 8u : R);
    float acc = 0.0f;
    for (uint32_t b = 0; b < nblk; b++) {
        v3blk nxt;
        if (b + 1u < nblk) {
            const uint32_t nrem = R - (b + 1u) * 8u, nr = nrem < 8u ? nrem : 8u;
            nxt = v3_blk_load(row + (size_t) (b + 1u) * 96u, nr * 12u,
                              EXT ? rex + (b + 1u) * 8u : NULL, nr);
        } else {
            nxt.w0 = nxt.w1 = nxt.w2 = nxt.ex = 0u;
        }
        const uint32_t rem = R - b * 8u;
        v3_rounds<DIM, EXT>(cur, rem < 8u ? rem : 8u, a, sh,
                            xr + (size_t) b * 256u * (DIM > 0 ? DIM : L.dim), cb,
                            L.dim, acc);
        cur = nxt;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    return __shfl_sync(0xffffffffu, acc, 0) * __half2float(gr[r]);
}

/* One warp per row, the three 128-byte lines of the next block issued before
 * this one computes (the reference's cross-block pipeline). */
template <int DIM, int EXT>
__global__ void gemv_v3blk_kernel(const uint8_t *p, const Layout L, const uint32_t *rl,
                                  const float *x, float *y) {
    const uint32_t lo = (uint32_t) ((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const uint32_t lane = threadIdx.x & 31u;
    if (lo >= (uint32_t) L.nrows) return;
    const float v = v3blk_row_dot<DIM, EXT>(p, L, row_of(L, lo, rl), row_x(x, L, lo));
    if (lane == 0) {
        if (L.first) y[lo] = v;
        else y[lo] += v;
    }
}

/* One round of v3_rounds at a slot the caller picks, with the activation
 * already advanced to that round's first index. The body is v3_rounds' (:158-
 * :194) verbatim - same funnel shifts, same lane->index map (a, sh), same
 * codeword loads, same accumulation order - with `g` becoming `lane` because
 * xg is offset by the caller. The stream kernel below is its only user; the
 * block kernel keeps v3_rounds so its codegen does not move. */
template <int DIM, int EXT>
__device__ __forceinline__ void v3_round_one(const v3blk &cur, uint32_t k, uint32_t a, uint32_t sh,
                                             const float *xg, const __half *cb, int dim, float &acc) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t f = k * 12u;
    const uint32_t lo_r = ((f & 31u) > 32u - 12u) ? ((f + 31u - lane) >> 5) : (f >> 5);
    const uint32_t hi_r = (((f + 1u) & 31u) > 32u - 12u) ? ((f + 32u - lane) >> 5) : ((f + 1u) >> 5);
    const uint32_t lo = __shfl_sync(0xffffffffu, v3_sel3(lo_r, cur.w0, cur.w1, cur.w2), (int) (f + a));
    const uint32_t hi = __shfl_sync(0xffffffffu, v3_sel3(hi_r, cur.w0, cur.w1, cur.w2), (int) (f + a + 1u));
    uint32_t v = __funnelshift_r(lo, hi, sh) & 0xFFFu;
    if (EXT) {
        const uint32_t exw = __shfl_sync(0xffffffffu, cur.ex, (int) k);
        v |= ((exw >> lane) & 1u) << 12;
    }
    if (DIM == 4) {
        const float4 xv = *(const float4 *) (xg + lane * 4u);
        const uint2 c = __ldg((const uint2 *) cb + v);
        const float2 f0 = __half22float2(*reinterpret_cast<const __half2 *> (&c.x));
        const float2 f1 = __half22float2(*reinterpret_cast<const __half2 *> (&c.y));
        acc += f0.x * xv.x + f0.y * xv.y + f1.x * xv.z + f1.y * xv.w;
    } else if (DIM == 8) {
        const float4 x0 = *(const float4 *) (xg + lane * 8u);
        const float4 x1 = *(const float4 *) (xg + lane * 8u + 4u);
        const uint2 c0 = __ldg((const uint2 *) cb + (size_t) v * 2u);
        const uint2 c1 = __ldg((const uint2 *) cb + (size_t) v * 2u + 1u);
        const float2 f0 = __half22float2(*reinterpret_cast<const __half2 *> (&c0.x));
        const float2 f1 = __half22float2(*reinterpret_cast<const __half2 *> (&c0.y));
        const float2 f2 = __half22float2(*reinterpret_cast<const __half2 *> (&c1.x));
        const float2 f3 = __half22float2(*reinterpret_cast<const __half2 *> (&c1.y));
        acc += f0.x * x0.x + f0.y * x0.y + f1.x * x0.z + f1.y * x0.w
             + f2.x * x1.x + f2.y * x1.y + f3.x * x1.z + f3.y * x1.w;
    } else {
        const __half *cw = cb + (size_t) v * (size_t) dim;
        for (int d = 0; d < dim; d++) acc += __half2float(cw[d]) * xg[lane * (size_t) dim + d];
    }
}

/* The reference's cross-row stream, on this container's v3 payload. Source
 * read first-hand: /data/ds4-dfm-rs/cuda/ds41_vq_persist.cuh:46-56 (why the
 * disease is the row head: a row's last block cannot hide the next row's
 * first-block latency, so every row stalls once) and :86-116 (v41_vq_stream:
 * one warp walks a span of consecutive rows as ONE bitstream, its 8-round
 * blocks cut by the span-global round, a row that runs out of rounds reducing
 * and finishing inside the block while the remaining rounds go to the next
 * row). A span here is a single stream because rows of a v3 payload are
 * contiguous - stream_off + r*mrow, mrow = R*48 bytes - the same property the
 * reference relies on.
 *
 * Arithmetic: a row still starts at its own round 0 with acc = 0, accumulates
 * the same rounds in the same order with the same lane map, reduces with the
 * same tree and applies the same gain, so the value is bit-identical to
 * v3blk_row_dot and the gate's bands cannot move. Only the pipeline changes.
 *
 * One warp per span; rl and a per-row-block activation are not covered and
 * the caller keeps the block kernel for those. */
template <int DIM>
__global__ void gemv_v3stream_kernel(const uint8_t *p, const Layout L, const float *x, float *y,
                                     uint32_t span) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t r0 = warp * span;
    if (r0 >= (uint32_t) L.nrows) return;
    const uint32_t rows = (uint32_t) L.nrows;
    const uint32_t n = rows - r0 < span ? rows - r0 : span;
    const uint32_t R = (uint32_t) (L.nidx_row >> 5);
    const uint32_t G = n * R;
    const uint32_t a = (lane * 12u) >> 5, sh = (lane * 12u) & 31u;
    const __half *cb = (const __half *) (p + L.cb_off);
    const __half *gr = (const __half *) (p + L.gain_off);
    const uint32_t *base = (const uint32_t *) (p + L.stream_off + (size_t) r0 * L.mrow);
    float acc = 0.0f;
    uint32_t kin = 0, row = 0;
    v3blk cur = v3_blk_load(base, (G < 8u ? G : 8u) * 12u, NULL, 0);
    for (uint32_t g0 = 0; g0 < G; g0 += 8u) {
        v3blk nxt;
        if (g0 + 8u < G) {
            const uint32_t left = G - g0 - 8u;
            nxt = v3_blk_load(base + (size_t) (g0 + 8u) * 12u, (left < 8u ? left : 8u) * 12u, NULL, 0);
        } else {
            nxt.w0 = nxt.w1 = nxt.w2 = nxt.ex = 0u;
        }
        const uint32_t rem = G - g0;
        const uint32_t rounds = rem < 8u ? rem : 8u;
        #pragma unroll
        for (uint32_t k = 0; k < 8u; k++) {
            if (k >= rounds) break;
            v3_round_one<DIM, 0>(cur, k, a, sh,
                                 x + (size_t) kin * 32u * (size_t) (DIM > 0 ? DIM : L.dim),
                                 cb, L.dim, acc);
            if (++kin == R) {
                float v = acc;
                for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
                v = __shfl_sync(0xffffffffu, v, 0) * __half2float(gr[r0 + row]);
                if (lane == 0) {
                    if (L.first) y[r0 + row] = v;
                    else y[r0 + row] += v;
                }
                acc = 0.0f;
                kin = 0;
                row++;
            }
        }
        cur = nxt;
    }
}

/* dim-4 fast path: one 8-byte codebook load and one half2 pair of conversions
 * per index, unrolled to hide L1 gather latency. This is the kernel the tools
 * select for nc below 4096 (v3_blk_ok requires nbit >= 12), which is the
 * router geometry. */
#define GEMV_UNROLL 4
__global__ void gemv_warp4_kernel(const uint8_t *p, const Layout L, const uint32_t *rl,
                                  const float *x, float *y) {
    const int lo = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (lo >= L.nrows) return;
    const size_t r = row_of(L, (uint32_t) lo, rl);
    const int nidx_row = L.cols / 4;
    const uint2 *cb = (const uint2 *) (p + L.cb_off);
    const __half *gr = (const __half *) (p + L.gain_off);
    const uint8_t *st = p + L.stream_off;
    const float *xr = row_x(x, L, (uint32_t) lo);
    const float4 *x4 = (const float4 *) xr;

    float acc = 0.0f;
    int i = lane;
    for (; i + 32 * (GEMV_UNROLL - 1) < nidx_row; i += 32 * GEMV_UNROLL) {
        for (int k = 0; k < GEMV_UNROLL; k++) {
            const int ii = i + k * 32;
            const uint32_t v = rd_idx_fmt(st, L, r, ii);
            const uint2 c = __ldg(cb + v);
            const float4 xv = x4[ii];
            const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&c.x));
            const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&c.y));
            acc += a.x * xv.x + a.y * xv.y + b.x * xv.z + b.y * xv.w;
        }
    }
    for (; i < nidx_row; i += 32) {
        const uint32_t v = rd_idx_fmt(st, L, r, i);
        const uint2 c = __ldg(cb + v);
        const float4 xv = x4[i];
        const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&c.x));
        const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&c.y));
        acc += a.x * xv.x + a.y * xv.y + b.x * xv.z + b.y * xv.w;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) {
        const float v = acc * __half2float(gr[r]);
        if (L.first) y[lo] = v;
        else y[lo] += v;
    }
}

/* A span of consecutive rows of one matrix, walked by one warp. This is the
 * shape for a row too short to fill the block kernel's 8-round group: the
 * expert down projection is 704 columns at dim 8, so a row is 88 indices -
 * 2.75 rounds - and v3_blk_ok refuses it, which leaves the generic warp
 * kernel at 73 GB/s against the block kernel's 128.
 *
 * The reference engine solved the same stall differently and says why: its
 * persistent kernel reads a span of consecutive rows as ONE continuous
 * bitstream so a row never restarts the stream at its head
 * (/data/ds4-dfm-rs/cuda/ds41_vq_persist.cuh:46-56, :88-135), and that
 * requires every row to be at least 8 rounds
 * (docs/deepseek41-port-plan.md:708-711). Our row is 2.75 rounds and 88 % 32
 * is 24, so a 32-index group would straddle two rows and the span cannot be
 * read as one stream. What transfers is the span itself: one warp owns
 * `span` consecutive rows, so their index reads and codebook gathers are
 * independent and the scheduler can hide one row's head with the others'
 * work, instead of each row paying that head alone.
 *
 * Per row this is the generic kernel's arithmetic exactly - lane l takes
 * indices l, l+32, l+64 (skipping i >= nidx_row), the same shfl_down tree,
 * the gain after the sum - so a gather through it is bit-identical to the
 * generic path and the layer's bands cannot move. The index read drops from
 * four bytes to three: a min(nbit,12)-bit field at byte offset b needs at
 * most 19 bits, which is b[0..2]. A payload with a 13th-bit plane is left on
 * the generic path, since the plane is a second bit array this kernel does
 * not read. */
/* One row of one payload by the three-byte index read: the span kernel's
 * arithmetic, extracted for the same reason as v3blk_row_dot. A payload with
 * a 13th-bit plane is not read here (the caller checks L.prow == 0). */
template <int DIM>
__device__ __forceinline__ float span_row_dot(const uint8_t *p, const Layout &L, size_t r,
                                              const float *xr) {
    const int lane = threadIdx.x & 31;
    const int nidx_row = L.cols / L.dim;
    const int groups = (nidx_row + 31) / 32;
    const __half *cb = (const __half *) (p + L.cb_off);
    const __half *gr = (const __half *) (p + L.gain_off);
    const uint8_t *st = p + L.stream_off;
    const uint32_t mb = L.nbit > 12 ? 12u : (uint32_t) L.nbit;
    const uint32_t imsk = (1u << mb) - 1u;
    const uint8_t *row = st + r * (size_t) L.mrow;
    float acc = 0.0f;
    for (int g = 0; g < groups; g++) {
        const int i = g * 32 + lane;
        if (i >= nidx_row) break;
        const uint32_t bit = (uint32_t) i * mb;
        const uint8_t *q = row + (bit >> 3);
        const uint32_t w = (uint32_t) q[0] | ((uint32_t) q[1] << 8) | ((uint32_t) q[2] << 16);
        const uint32_t v = (w >> (bit & 7)) & imsk;
        if (DIM == 4) {
            const float4 xv = *(const float4 *) (xr + (size_t) i * 4u);
            const uint2 c = __ldg((const uint2 *) cb + v);
            const float2 a = __half22float2(*reinterpret_cast<const __half2 *> (&c.x));
            const float2 b = __half22float2(*reinterpret_cast<const __half2 *> (&c.y));
            acc += a.x * xv.x + a.y * xv.y + b.x * xv.z + b.y * xv.w;
        } else {
            const float4 x0 = *(const float4 *) (xr + (size_t) i * (size_t) DIM);
            const float4 x1 = *(const float4 *) (xr + (size_t) i * (size_t) DIM + 4u);
            const uint2 c0 = __ldg((const uint2 *) cb + (size_t) v * 2u);
            const uint2 c1 = __ldg((const uint2 *) cb + (size_t) v * 2u + 1u);
            const float2 a = __half22float2(*reinterpret_cast<const __half2 *> (&c0.x));
            const float2 b = __half22float2(*reinterpret_cast<const __half2 *> (&c0.y));
            const float2 c = __half22float2(*reinterpret_cast<const __half2 *> (&c1.x));
            const float2 d = __half22float2(*reinterpret_cast<const __half2 *> (&c1.y));
            acc += a.x * x0.x + a.y * x0.y + b.x * x0.z + b.y * x0.w
                 + c.x * x1.x + c.y * x1.y + d.x * x1.z + d.y * x1.w;
        }
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    return __shfl_sync(0xffffffffu, acc, 0) * __half2float(gr[r]);
}

template <int DIM>
__global__ void gemv_rowspan_kernel(const uint8_t *p, const Layout L, const uint32_t *rl,
                                    const float *x, float *y, int span) {
    const int lo0 = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * span;
    const int lane = threadIdx.x & 31;
    if (lo0 >= L.nrows) return;
    #pragma unroll 4
    for (int s = 0; s < span; s++) {
        const int lo = lo0 + s;
        if (lo >= L.nrows) break;
        const float v = span_row_dot<DIM>(p, L, row_of(L, (uint32_t) lo, rl), row_x(x, L, (uint32_t) lo));
        if (lane == 0) {
            if (L.first) y[lo] = v;
            else y[lo] += v;
        }
    }
}

/* ---- the expert block, in the reference engine's persistent shape --------
 *
 * Source read first-hand: /data/ds4-dfm-rs/cuda/ds41_vq_persist.cuh:157-213
 * (v41_vq_gu_persist_kernel) and :215-238 (v41_vq_dn_persist_kernel), with
 * the launcher at :399-411. What transfers unchanged:
 *
 *  - grid sized to the SMs, one persistent block each, so the block ramp and
 *    every per-block setup are paid once per stack, not once per row or per
 *    block of rows (the engine's :8-13 says why: with its shared-memory
 *    codebook, a per-block copy stalls the whole SM at every block);
 *  - warps cut the (expert, row) items into contiguous spans of at most 32
 *    rows, a span never crossing an expert (V41_VQ_SEG, :144-146);
 *  - gate and up for the same row are one warp's work, and the SwiGLU is
 *    applied in registers, so the gate value never round-trips through h
 *    (:11-13);
 *  - the down stack writes its f32 partials and the weighted reduce is one
 *    separate launch (:361-363).
 *
 * The two deviations forced by this container, both named rather than
 * silent. (1) A row carries two payloads - the tools' two-stage residual VQ
 * - so the row dot runs twice per row, stage 0 then stage 1, each with its
 * own gain, and the two values are added in that order. That is exactly the
 * pair of launches this replaces (stage 0 assigns, stage 1 accumulates), so
 * the value is bit-identical. (2) A down row here is 88 indices, 2.75
 * rounds, so the engine's cross-row bitstream stream (:46-56) does not
 * apply; each row still starts its own stream, as the span kernel already
 * does. The gate/up row is 352 indices, 11 rounds, and keeps the block
 * walk.
 *
 * The grid is nsm * blocks_per_sm because this kernel has no shared-memory
 * request to cap occupancy (the engine's is 1 block/SM at 1024 threads);
 * blocks_per_sm is the SM's thread capacity divided by the block. */

static constexpr int GM4_EXPERT_THREADS = 512;

template <int DIM>
__global__ void expert_gu_persist_kernel(const Payload a, const Payload b, const int *picked,
                                         const float *x, float *act, int k, int ff, int stride) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps = (uint32_t) (blockDim.x >> 5);
    const uint32_t gw = (uint32_t) blockIdx.x * warps + (threadIdx.x >> 5);
    const uint32_t nw = (uint32_t) gridDim.x * warps;
    const uint32_t total = (uint32_t) k * (uint32_t) ff;
    const uint32_t per = (total + nw - 1u) / nw;
    uint32_t u = gw * per;
    const uint32_t uend = (u + per < total) ? u + per : total;
    while (u < uend) {
        const uint32_t p = u / (uint32_t) ff;
        const uint32_t r0 = u % (uint32_t) ff;
        uint32_t n = uend - u;
        if (n > (uint32_t) ff - r0) n = (uint32_t) ff - r0;
        if (n > 32u) n = 32u;
        u += n;
        const int e = picked[p];
        if (e < 0) continue;
        const size_t base = (size_t) e * (size_t) stride + (size_t) r0;
        for (uint32_t j = 0; j < n; j++) {
            const size_t r = base + j;
            /* A row of this container carries the two VQ stages, not gate and
             * up: both payloads hold the tensor's 2*ff rows, with the gate at
             * row j and the up half at row ff+j inside the expert's block. So
             * one output row reads four rows of two payloads - gate stage 0
             * plus gate stage 1, up stage 0 plus up stage 1 - and the order of
             * the two terms is the legacy path's (stage 0 assigns, stage 1
             * accumulates). Verified against the container bytes and the
             * activation on the host: row 143616 of layer 0's gate_up is
             * -0.4833564 (stage 0) plus -0.0456458 (stage 1), -0.5290022, the
             * eight-launch value. */
            const float g = v3blk_row_dot<DIM, 0>(a.dev, a.l, r, x)
                          + v3blk_row_dot<DIM, 0>(b.dev, b.l, r, x);
            const float v = v3blk_row_dot<DIM, 0>(a.dev, a.l, r + (size_t) ff, x)
                          + v3blk_row_dot<DIM, 0>(b.dev, b.l, r + (size_t) ff, x);
            if (lane == 0) {
                const float kk = 0.7978846f; /* sqrt(2/pi), gelu_mul_blocks_kernel */
                act[(size_t) p * (size_t) ff + r0 + j] =
                    0.5f * g * (1.0f + tanhf(kk * (g + 0.044715f * g * g * g))) * v;
            }
        }
    }
}

template <int DIM>
__global__ void expert_dn_persist_kernel(const Payload a, const Payload b, const int *picked,
                                         const float *act, float *d, int k, int hidden, int ff) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps = (uint32_t) (blockDim.x >> 5);
    const uint32_t gw = (uint32_t) blockIdx.x * warps + (threadIdx.x >> 5);
    const uint32_t nw = (uint32_t) gridDim.x * warps;
    const uint32_t total = (uint32_t) k * (uint32_t) hidden;
    const uint32_t per = (total + nw - 1u) / nw;
    uint32_t u = gw * per;
    const uint32_t uend = (u + per < total) ? u + per : total;
    while (u < uend) {
        const uint32_t p = u / (uint32_t) hidden;
        const uint32_t r0 = u % (uint32_t) hidden;
        uint32_t n = uend - u;
        if (n > (uint32_t) hidden - r0) n = (uint32_t) hidden - r0;
        if (n > 32u) n = 32u;
        u += n;
        const int e = picked[p];
        if (e < 0) continue;
        const size_t base = (size_t) e * (size_t) hidden + (size_t) r0;
        const float *xr = act + (size_t) p * (size_t) ff;
        for (uint32_t j = 0; j < n; j++) {
            const size_t r = base + j;
            const float v = span_row_dot<DIM>(a.dev, a.l, r, xr)
                          + span_row_dot<DIM>(b.dev, b.l, r, xr);
            if (lane == 0) d[(size_t) p * (size_t) hidden + r0 + j] = v;
        }
    }
}

/* One warp per row, no shared codebook: the 64 KB table stays in L1. The
 * fallback for any dim and any index width. */
template <int DIM>
__global__ void gemv_warp_kernel(const uint8_t *p, const Layout L, const uint32_t *rl,
                                 const float *xin, float *y) {
    const int lo = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (lo >= L.nrows) return;
    const size_t r = row_of(L, (uint32_t) lo, rl);
    const int nidx_row = L.cols / L.dim;
    const __half *cb = (const __half *) (p + L.cb_off);
    const __half *gr = (const __half *) (p + L.gain_off);
    const uint8_t *st = p + L.stream_off;
    const float *x = row_x(xin, L, (uint32_t) lo);

    float acc = 0.0f;
    for (int i = lane; i < nidx_row; i += 32) {
        const uint32_t v = rd_idx_fmt(st, L, r, i);
        if (DIM == 4) {
            const float4 xv = *(const float4 *) (x + (size_t) i * 4u);
            const uint2 c = __ldg((const uint2 *) cb + v);
            const float2 a = __half22float2(*reinterpret_cast<const __half2 *> (&c.x));
            const float2 b = __half22float2(*reinterpret_cast<const __half2 *> (&c.y));
            acc += a.x * xv.x + a.y * xv.y + b.x * xv.z + b.y * xv.w;
        } else if (DIM == 8) {
            const float4 x0 = *(const float4 *) (x + (size_t) i * 8u);
            const float4 x1 = *(const float4 *) (x + (size_t) i * 8u + 4u);
            const uint2 c0 = __ldg((const uint2 *) cb + (size_t) v * 2u);
            const uint2 c1 = __ldg((const uint2 *) cb + (size_t) v * 2u + 1u);
            const float2 a = __half22float2(*reinterpret_cast<const __half2 *> (&c0.x));
            const float2 b = __half22float2(*reinterpret_cast<const __half2 *> (&c0.y));
            const float2 c = __half22float2(*reinterpret_cast<const __half2 *> (&c1.x));
            const float2 d = __half22float2(*reinterpret_cast<const __half2 *> (&c1.y));
            acc += a.x * x0.x + a.y * x0.y + b.x * x0.z + b.y * x0.w
                 + c.x * x1.x + c.y * x1.y + d.x * x1.z + d.y * x1.w;
        } else {
            const __half *cw = cb + (size_t) v * L.dim;
            const float *xg = x + (size_t) i * L.dim;
            for (int d = 0; d < L.dim; d++) acc += __half2float(cw[d]) * xg[d];
        }
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) {
        const float v = acc * __half2float(gr[r]);
        if (L.first) y[lo] = v;
        else y[lo] += v;
    }
}

/* One row of a table, reconstructed: the token embedding, which is the one
 * place the forward reads a row instead of dotting one (`forward.rs:316`).
 * The CPU's form is `add_row_values` (crates/ds4-core/src/vq.rs:714-726):
 * `out[i*dim + d] += f16(codebook[idx][d]) * gain`, stage by stage in
 * ascending order, and then the whole row is scaled by embed_scale
 * (`forward.rs:319-322`). The stage sum is therefore formed raw and the scale
 * is applied once, on the last stage, so this is the CPU's rounding order and
 * not a per-stage rescale. A stage contributes a whole word, so the innermost
 * loop is the DIM code values, never a partial word. */
template <int DIM>
__global__ void embed_row_kernel(const uint8_t *p, const Layout L, const int32_t *tokd, float *out,
                                 int first, int last, float scale) {
    const int nidx_row = L.cols / L.dim;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nidx_row) return;
    const int t = blockIdx.y; /* the block's token: a prefill reads n rows */
    const uint32_t row = (uint32_t) tokd[t];
    const __half *cb = (const __half *) (p + L.cb_off);
    const uint32_t idx = rd_idx_fmt(p + L.stream_off, L, row, i);
    const float gain = __half2float(((const __half *) (p + L.gain_off))[row]);
    float *o = out + ((size_t) t * (size_t) nidx_row + (size_t) i) * (size_t) DIM;
    float w[8];
    if (DIM == 4) {
        const uint2 c = __ldg((const uint2 *) cb + idx);
        const float2 a = __half22float2(*reinterpret_cast<const __half2 *> (&c.x));
        const float2 b = __half22float2(*reinterpret_cast<const __half2 *> (&c.y));
        w[0] = a.x * gain;
        w[1] = a.y * gain;
        w[2] = b.x * gain;
        w[3] = b.y * gain;
    } else {
        const uint2 c0 = __ldg((const uint2 *) cb + (size_t) idx * 2u);
        const uint2 c1 = __ldg((const uint2 *) cb + (size_t) idx * 2u + 1u);
        const float2 a = __half22float2(*reinterpret_cast<const __half2 *> (&c0.x));
        const float2 b = __half22float2(*reinterpret_cast<const __half2 *> (&c0.y));
        const float2 c = __half22float2(*reinterpret_cast<const __half2 *> (&c1.x));
        const float2 d = __half22float2(*reinterpret_cast<const __half2 *> (&c1.y));
        w[0] = a.x * gain;
        w[1] = a.y * gain;
        w[2] = b.x * gain;
        w[3] = b.y * gain;
        w[4] = c.x * gain;
        w[5] = c.y * gain;
        w[6] = d.x * gain;
        w[7] = d.y * gain;
    }
    for (int k = 0; k < DIM; k++) {
        float v = first ? w[k] : o[k] + w[k];
        if (last) v *= scale;
        o[k] = v;
    }
}

/* ==== Tiled prefill: f16 tensor cores =====================================
 *
 * Ported from /data/VQ_tools/c/gpu/decode-gpu.cu:320-520 (the tiled kernel),
 * :588-612 (the legacy one-thread-per-(row, token) kernel), :531-560 (the
 * shared-memory account and the geometry gate) and :1010-1160 (the entry
 * points). The structure is the reference's, unchanged:
 *
 * One block computes 128 weight rows x BN tokens over all K, stepping BK 64
 * columns per round: 16 warps = 8 row tiles of 16 rows x 2 token halves; a
 * warp owns 16 rows and BN/16 n8 slices, and a round is 4 k16 mma groups.
 * The tile shape, the 16-warp split, the ldmatrix addressing and the
 * 16-byte-chunk XOR swizzle are the tiled prefill's layout.
 *
 * The products are f16, not bf16: our codebook is f16 (10 mantissa bits), and
 * bf16 would round every learned codeword by 2^-8 - a quality change, not a
 * layout choice. So the tile is f16: a code word is a straight 8- or 16-byte
 * copy out of the codebook with no conversion at all, the activations are
 * rounded f32 -> f16, and the mma is f32.f16.f16.f32. An f16 x f16 product is
 * exact in f32.
 *
 * The mma starts every k16 from zero and adds back with a plain FADD (ds4's
 * 2026-09-24 finding, decode-gpu.cu:337-341: letting K accumulate inside the
 * tensor core's own f32 adder truncates low bits on alignment instead of
 * rounding).
 */
#define PRE_BM 128
#define PRE_THREADS 512
#define PRE_LEGACY_THREADS 256

/* A tile row is BK f16 = BK/8 chunks of 16 B; chunk ^ (row & 7) spreads the 8
 * rows ldmatrix reads at once over 8 bank groups. `stride` is the row's byte
 * stride, BK*2. */
__device__ __forceinline__ uint32_t pre_swz(uint32_t row, uint32_t chunk, uint32_t stride) {
    return row * stride + ((chunk ^ (row & 7u)) << 4);
}

__device__ __forceinline__ void pre_ldsm4(uint32_t *r, const uint8_t *p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void pre_mma(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pre_h2u(__half2 h) { return *reinterpret_cast<uint32_t *>(&h); }

/* CB 0 keeps the codebook in global (it is 64 KB here: it stays in L1 under
 * three resident blocks), CB 1 stages it in shared behind the two tiles. */
template <int BN, int BK, int CB, int DIM>
__global__ __launch_bounds__(PRE_THREADS, 1) void gemm_tiled_kernel(
        const uint8_t *p, const Layout L, const float *xT, int ntok, float *y) {
    constexpr uint32_t CW = BK / (uint32_t) DIM;     /* code words per row-round: 16 (dim 4), 8 (dim 8) */
    constexpr uint32_t TPR = 4u, CPT = CW / TPR;     /* 4 threads per row, 4 / 2 code words each */
    constexpr uint32_t NFW = BN / 16u;               /* n8 slices per warp */
    constexpr uint32_t CPL = BK / 8u;                /* 16 B chunks per tile row: 8 (BK 64), 16 (BK 128) */
    constexpr uint32_t NCH = BN * CPL / PRE_THREADS; /* B-tile chunks per thread */
    constexpr uint32_t STR = BK * 2u;                /* tile row stride in bytes */
    extern __shared__ __align__(16) uint8_t sh[];
    uint8_t *As = sh, *Bs = sh + PRE_BM * STR;
    uint8_t *cbs = Bs + BN * STR;
    const __half *cb = (const __half *) (p + L.cb_off);
    const __half *gr = (const __half *) (p + L.gain_off);
    const uint8_t *st = p + L.stream_off;
    const uint8_t *pl = L.prow ? st + L.mrow * (size_t) L.rows : NULL;

    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t wr = warp & 7u, th = warp >> 3u;
    if (CB) { /* the codebook bytes are already the tile format: copy, no convert */
        const uint2 *src = (const uint2 *) cb;
        uint2 *dst = (uint2 *) cbs;
        for (uint32_t i = tid; i < (uint32_t) L.nc * (uint32_t) L.dim * 2u / 8u; i += PRE_THREADS)
            dst[i] = __ldg(src + i);
    }

    const uint32_t ar = tid / TPR, sub = tid % TPR; /* A-tile row this thread fills */
    const int r0 = L.row0 + (int) blockIdx.x * PRE_BM, t0 = (int) blockIdx.y * BN;
    const int nt = min((int) BN, ntok - t0);
    const int nrnd = (int) (L.cols / BK);
    const bool rv = r0 + (int) ar < L.row0 + L.nrows;
    const uint32_t rsrc = (uint32_t) (rv ? r0 + (int) ar : 0);
    const uint8_t *rowp = st + (size_t) rsrc * L.mrow;
    const uint8_t *rowq = pl ? pl + (size_t) rsrc * L.prow : NULL;

    float acc[NFW][4];
    #pragma unroll
    for (uint32_t f = 0; f < NFW; f++) acc[f][0] = acc[f][1] = acc[f][2] = acc[f][3] = 0.f;
    __syncthreads();

    for (int i = 0; i < nrnd; i++) {
        /* This thread's code words of round i sit at bit 12*(i*CW + sub*CPT).
         * Two aligned words are read and shifted; the offset inside them is
         * 0 or 16 for dim 4 and in {0,8,16,24} for dim 8, and CPT*12 <= 48, so
         * every index of the round stays inside the 64-bit window. The window
         * can reach up to 5 bytes past the row's stream (decode-gpu.cu:96-101);
         * the container's zeroed tail past the payload covers it. */
        const uint32_t bit = (uint32_t) i * 12u * CW + sub * 12u * CPT;
        const uint32_t aoff = (bit >> 5) << 2, sh64 = bit & 31u;
        uint32_t w0 = 0, w1 = 0, eb = 0;
        if (rv) {
            w0 = __ldg((const uint32_t *) (rowp + aoff));
            w1 = __ldg((const uint32_t *) (rowp + aoff + 4u));
            if (rowq) eb = (uint32_t) __ldg(rowq + ((uint32_t) i * CW + sub * CPT) / 8u);
        }
        /* The next round's activations, converted here so the global latency
         * overlaps the previous round's mma. */
        uint4 bx[NCH];
        #pragma unroll
        for (uint32_t c = 0; c < NCH; c++) {
            const uint32_t ch = tid + c * PRE_THREADS, t = ch / CPL, cc = ch % CPL;
            bx[c] = make_uint4(0, 0, 0, 0);
            if ((int) t < nt) {
                const float *s = xT + (size_t) ((uint32_t) i * BK + cc * 8u) * (size_t) ntok
                               + (size_t) (t0 + (int) t);
                const size_t S = (size_t) ntok;
                bx[c].x = pre_h2u(__floats2half2_rn(s[0], s[S]));
                bx[c].y = pre_h2u(__floats2half2_rn(s[2 * S], s[3 * S]));
                bx[c].z = pre_h2u(__floats2half2_rn(s[4 * S], s[5 * S]));
                bx[c].w = pre_h2u(__floats2half2_rn(s[6 * S], s[7 * S]));
            }
        }
        __syncthreads(); /* the previous round's tiles are done being read */

        const uint64_t u = ((uint64_t) w1 << 32) | (uint64_t) w0;
        #pragma unroll
        for (uint32_t q = 0; q < CPT; q++) {
            const uint32_t j = sub * CPT + q; /* code word of this round */
            uint32_t v = (uint32_t) (u >> (sh64 + 12u * q)) & 0xFFFu;
            if (rowq) v |= ((eb >> (j & 7u)) & 1u) << 12;
            if (DIM == 4) { /* 8-byte code word: half of a 16 B chunk */
                const uint8_t *src = CB ? cbs + (size_t) v * 8u : (const uint8_t *) cb + (size_t) v * 8u;
                const uint2 cw = CB ? *(const uint2 *) src : __ldg((const uint2 *) src);
                *(uint2 *) (As + pre_swz(ar, j >> 1, STR) + ((j & 1u) << 3)) = cw;
            } else { /* dim 8: one code word is one 16 B chunk, two aligned 8 B halves */
                const uint8_t *src = CB ? cbs + (size_t) v * 16u : (const uint8_t *) cb + (size_t) v * 16u;
                const uint2 lo = CB ? *(const uint2 *) src : __ldg((const uint2 *) src);
                const uint2 hi = CB ? *(const uint2 *) (src + 8u) : __ldg((const uint2 *) (src + 8u));
                uint8_t *dst = As + pre_swz(ar, j, STR);
                *(uint2 *) dst = lo;
                *(uint2 *) (dst + 8u) = hi;
            }
        }
        #pragma unroll
        for (uint32_t c = 0; c < NCH; c++) {
            const uint32_t ch = tid + c * PRE_THREADS;
            *(uint4 *) (Bs + pre_swz(ch / CPL, ch % CPL, STR)) = bx[c];
        }
        __syncthreads();

        /* The warp's 16 rows over this half's n8 slices. */
        #pragma unroll
        for (uint32_t kk = 0; kk < BK / 16u; kk++) {
            const uint32_t mt = lane >> 3;
            uint32_t a[4];
            pre_ldsm4(a, As + pre_swz(wr * 16u + (lane & 7u) + (mt & 1u) * 8u, kk * 2u + (mt >> 1), STR));
            #pragma unroll
            for (uint32_t np = 0; np < NFW / 2u; np++) {
                const uint32_t f0 = th * NFW + 2u * np;
                if ((int) (f0 * 8u) >= nt) break;
                uint32_t b[4];
                pre_ldsm4(b, Bs + pre_swz((f0 + (mt >> 1)) * 8u + (lane & 7u), kk * 2u + (mt & 1u), STR));
                float t0v[4] = {0.f, 0.f, 0.f, 0.f};
                pre_mma(t0v, a, b[0], b[1]);
                #pragma unroll
                for (uint32_t e = 0; e < 4u; e++) acc[2 * np][e] += t0v[e];
                if ((int) ((f0 + 1u) * 8u) < nt) {
                    float t1v[4] = {0.f, 0.f, 0.f, 0.f};
                    pre_mma(t1v, a, b[2], b[3]);
                    #pragma unroll
                    for (uint32_t e = 0; e < 4u; e++) acc[2 * np + 1][e] += t1v[e];
                }
            }
        }
    }

    /* Exit: c0,c1 = row lane/4, token (lane&3)*2+{0,1}; c2,c3 = row + 8. */
    #pragma unroll
    for (uint32_t h = 0; h < 2u; h++) {
        const uint32_t r = (uint32_t) r0 + wr * 16u + (lane >> 2) + h * 8u;
        if (r >= (uint32_t) (L.row0 + L.nrows)) continue;
        const float g = __half2float(__ldg(gr + r));
        #pragma unroll
        for (uint32_t f = 0; f < NFW; f++) {
            const uint32_t gf = th * NFW + f;
            if ((int) (gf * 8u) >= nt) break;
            #pragma unroll
            for (uint32_t e = 0; e < 2u; e++) {
                const uint32_t t = gf * 8u + (lane & 3u) * 2u + e;
                if ((int) t >= nt) continue;
                y[(size_t) (t0 + (int) t) * L.rows + r] += acc[f][h * 2u + e] * g;
            }
        }
    }
}

/* Prefill: one thread per (row, token). The row's indices are staged in shared
 * memory once by the block, then every thread walks them, reading the codebook
 * (the same address across the block, so L1 broadcasts it) and its own token's
 * activations. xT is [col][token], so the block's reads at a fixed (i, d) are
 * consecutive addresses: one coalesced transaction per warp. There is no
 * cross-thread reduction, because each thread owns one output element.
 *
 * The reference keeps this as the fallback for any geometry the tiled kernel
 * cannot take (dim-4 8-bit codebooks among them), and as the comparison that
 * showed why the tiled shape exists. */
__global__ void gemm_prefill_kernel(const uint8_t *p, const Layout L, const float *xT,
                                    int ntok, float *y) {
    extern __shared__ uint16_t sidx[];
    const int r = L.row0 + blockIdx.x;
    const int nidx_row = L.cols / L.dim;
    const uint8_t *st = p + L.stream_off;

    for (int i = threadIdx.x; i < nidx_row; i += blockDim.x)
        sidx[i] = (uint16_t) rd_idx_fmt(st, L, r, i);
    __syncthreads();

    const int t = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= ntok) return;

    const __half *cb = (const __half *) (p + L.cb_off);
    const float *xt = xT + t;
    float acc = 0.0f;
    for (int i = 0; i < nidx_row; i++) {
        const __half *c = cb + (size_t) sidx[i] * L.dim;
        const float *xg = xt + (size_t) i * L.dim * ntok;
        for (int d = 0; d < L.dim; d++) acc += __half2float(c[d]) * xg[(size_t) d * ntok];
    }
    const __half gh = ((const __half *) (p + L.gain_off))[r];
    y[(size_t) t * L.rows + r] += acc * __half2float(gh);
}

/* ---- the layer's elementwise arithmetic --------------------------------
 *
 * The reference here is not a device kernel: it is this repository's CPU
 * forward (crates/ds4-core/src/gemma4_a4b/forward.rs), whose rms_norm, gelu
 * and bf16_round are the family's arithmetic. These kernels are that
 * arithmetic on the device, gated against it, so the chain can use one
 * definition of a norm and one of the activation.
 */

/* `Gemma4RMSNorm.forward`: y = x * pow(mean(x^2) + eps, -0.5) * w. One block
 * per row. The block sums the squares tree-wise, which orders the f32 adds
 * differently from the CPU's sequential sum and nothing more. */
__global__ void rmsnorm_kernel(const float *x, const float *w, float *y, int width, float eps) {
    extern __shared__ float red[];
    const long long r = blockIdx.x;
    const float *row = x + (size_t) r * (size_t) width;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) sum += row[i] * row[i];
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float) width + eps);
    float *out = y + (size_t) r * (size_t) width;
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        out[i] = w ? row[i] * inv * w[i] : row[i] * inv;
}

/* `ACT2FN["gelu_pytorch_tanh"]` and the `gelu(gate) * up` the forward writes:
 * the constants are forward.rs:42-46 verbatim. */
__global__ void gelu_mul_kernel(const float *gate, const float *up, float *act, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = gate[i];
    const float k = 0.7978846f; /* sqrt(2/pi) */
    act[i] = 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x))) * up[i];
}

/* gelu(gate) * up for every expert block of a routed layer in one launch: the
 * stack holds each expert's gate rows then its up rows, `ff` apart, so block
 * b's activation is gelu(gu[b*2ff + j]) * gu[b*2ff + ff + j] for j < ff. The
 * same arithmetic as gelu_mul_kernel, strided by the expert block. */
__global__ void gelu_mul_blocks_kernel(const float *gu, float *act, int ff) {
    const int b = blockIdx.y;
    const size_t base = (size_t) b * 2 * (size_t) ff;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < ff; j += gridDim.x * blockDim.x) {
        const float x = gu[base + j];
        const float k = 0.7978846f; /* sqrt(2/pi) */
        act[(size_t) b * ff + j] = 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)))
                                   * gu[base + ff + j];
    }
}

/* The routed mixture's accumulation: the k experts' down projections sit in
 * one buffer, block i at i*n, and the layer's feed-forward output is their
 * selection-weighted sum. The k adds are the order the per-expert axpys made
 * them in, so this is the same sum with the buffer read once per thread
 * instead of k times. */
__global__ void moe_reduce_kernel(const float *d, const float *w, float *out, int k, int n) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < k; i++) acc += w[i] * d[(size_t) i * (size_t) n + j];
        out[j] = acc;
    }
}

/* out = (x + add) * scale, with add optional: the residual add and the layer
 * scalar the forward applies as two separate loops. */
__global__ void residual_scale_kernel(const float *x, const float *add, float *out, size_t n,
                                      float scale) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (x[i] + (add ? add[i] : 0.0f)) * scale;
}

/* The logit softcap the forward applies last: cap * tanh(x / cap). */
__global__ void softcap_kernel(float *y, size_t n, float cap) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = cap * tanhf(y[i] / cap);
}

/* ---- attention ---------------------------------------------------------
 *
 * The reference is the official implementation, read first-hand this session
 * at /data/refs/transformers-5.14.1:
 *   - modeling_gemma4.py:786-814  rotate_half and apply_rotary_pos_emb
 *     (x * cos + rotate_half(x) * sin, with rotate_half pairing d and
 *     d + width/2 and the table folded by cat(freqs, freqs));
 *   - modeling_rope_utils.py:227-244  proportional RoPE: rope_angles =
 *     int(partial * head_dim // 2), inv = 1/base^(2i/head_dim) over the full
 *     head width, zeros for the unrotated pairs;
 *   - modeling_gemma4.py:1183-1296  the attention block: q_norm, k_norm,
 *     weightless v_norm, k_eq_v taking the value from the raw k projection,
 *     scaling 1.0;
 *   - modeling_gemma4.py:823-855  eager attention: scores, additive mask,
 *     softmax upcast to fp32, then probs . v;
 *   - masking_utils.py:92-101  the sliding predicate kv > q - window with the
 *     causal kv <= q, i.e. exactly `window` keys including the query.
 */

/* The rotary application: one thread per pair, so the two halves of a pair
 * are written by the same thread and no barrier is needed. A zero frequency
 * is the identity pair and is skipped. The position comes from the device
 * (`posd[0]`), not from a launch parameter: a captured step is replayed at
 * many positions and a baked-in one would be the first position forever. The
 * reference engine's convention, /data/ds4-dfm-rs/ds4_ds41_graph.inc:9-12 and
 * cuda/ds41_attn.cuh:125-127. */
__global__ void rope_kernel(float *x, int heads, int head_dim, const int32_t *posd,
                            const float *inv) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    const int half = head_dim / 2;
    if (d >= half) return;
    const float w = inv[d];
    if (w == 0.0f) return;
    float *base = x + (size_t) h * (size_t) head_dim;
    const float angle = (float) posd[0] * w;
    float s, c;
    sincosf(angle, &s, &c);
    const float a = base[d], b = base[d + half];
    base[d] = a * c - b * s;
    base[d + half] = b * c + a * s;
}

__device__ __forceinline__ float block_max(float v, float *red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x + 31) / 32 ? red[threadIdx.x] : -INFINITY;
    if (warp == 0) {
        for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    v = red[0];
    __syncthreads();
    return v;
}

__device__ __forceinline__ float block_sum(float v, float *red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x + 31) / 32 ? red[threadIdx.x] : 0.0f;
    if (warp == 0) {
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    v = red[0];
    __syncthreads();
    return v;
}

/* Which slot holds position `p`. The CPU session's rule, verbatim
 * (crates/ds4-core/src/gemma4_a4b/forward.rs:248-254): a sliding layer wraps
 * at its capacity, a full layer appends, so the caller passing cap ==
 * positions and ring == 0 gets the identity mapping the host-history path
 * always used. */
__device__ __forceinline__ size_t kv_slot(int p, int cap, int ring) {
    return ring ? (size_t) (p % cap) : (size_t) p;
}

/* Split-position decode attention, in the flash-decoding shape: one block per
 * (head, run of the window), then one small combine.
 *
 * The single-block-per-head kernel this replaces left the card mostly idle -
 * 16 heads is 16 blocks of 56 SMs - and its value pass was one thread per
 * dimension walking the window in one dependent ~1000-iteration chain, which
 * measured 17 GB/s against the 130 the card does. The split gives every SM a
 * run of the window instead.
 *
 * The split is arithmetic-neutral by construction: a run is a contiguous
 * segment of the one ascending position walk, so each block sums its segment
 * in the same order the unsplit kernel would have, and the combine is the
 * standard max-rescale
 *
 *   out = sum_s exp(m_s - M) o_s  /  sum_s exp(m_s - M) l_s,   M = max_s m_s
 *
 * over the runs' (max, sum of exponentials, weighted value sum). The schedule
 * depends only on the head count and the window, so the dense walk and the
 * ring walk of the same history pick the same runs; the ring gate's check
 * that the two are bit-identical therefore still holds exactly. */
__global__ void attn_split_kernel(const float *q, const __half *k, const __half *v, float *scores,
                                  float *part, float *pm, float *pl, int heads, int kv_heads,
                                  int head_dim, const int32_t *posd, int window, int splits,
                                  int chunk, int kv_cap, int kv_ring) {
    extern __shared__ float red[];
    const int block = blockIdx.x;
    const int h = block / splits;
    const int s = block - h * splits;
    const int group = heads / kv_heads;
    const int kv_head = h / group;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const size_t kv_stride = (size_t) kv_heads * (size_t) head_dim;
    const float *qh = q + (size_t) h * (size_t) head_dim;
    float *sc = scores + (size_t) block * (size_t) chunk;
    float *outp = part + (size_t) block * (size_t) head_dim;
    /* The window of this step, from the device position: `window` is the most
     * the caller's cache can hold (the ring's capacity, the capture's bucket
     * bound, or 0 for everything), so a ring layer keeps its `cap` most recent
     * positions and an appending one keeps 0..pos. `chunk` is the bucket's
     * bound (ds4_ds41_graph.inc:205-209), so a run past the true window is
     * simply empty and the combine gives it weight 0. */
    const int positions = posd[0] + 1;
    const int nwin = window > 0 && positions > window ? window : positions;
    const int first = positions - nwin;
    const int lo = s * chunk;
    const int hi = lo + chunk < nwin ? lo + chunk : nwin;
    const int n = hi > lo ? hi - lo : 0;

    /* The score pass: a warp takes every nwarps-th position of the run, two
     * dimensions per lane as one half2, one warp reduction per position. */
    for (int p = lo + warp; p < hi; p += nwarps) {
        const size_t ks = kv_slot(first + p, kv_cap, kv_ring);
        const __half *kp = k + ks * kv_stride + (size_t) kv_head * head_dim;
        float acc = 0.0f;
        for (int d = lane * 2; d < head_dim; d += 64) {
            const float2 qq = *(const float2 *) (qh + d);
            const float2 kk = __half22float2(*(const __half2 *) (kp + d));
            acc += qq.x * kk.x + qq.y * kk.y;
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) sc[p - lo] = acc; /* scaling is 1.0 */
    }
    __syncthreads();

    /* The run's softmax in fp32, as the reference upcasts it (`:846`). The
     * exponential stays unnormalised: the combine divides once. */
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, sc[i]);
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float e = expf(sc[i] - mx);
        sc[i] = e;
        sum += e;
    }
    sum = block_sum(sum, red);

    /* The value pass: two dimensions per thread as one half2, over the run in
     * position order. The dimensions are disjoint, so there is no reduction. */
    for (int d = threadIdx.x * 2; d < head_dim; d += blockDim.x * 2) {
        float2 acc = make_float2(0.0f, 0.0f);
        for (int i = 0; i < n; i++) {
            const size_t ks = kv_slot(first + lo + i, kv_cap, kv_ring);
            const float pr = sc[i];
            const float2 vv = __half22float2(
                *(const __half2 *) (v + ks * kv_stride + (size_t) kv_head * head_dim + d));
            acc.x = fmaf(pr, vv.x, acc.x);
            acc.y = fmaf(pr, vv.y, acc.y);
        }
        outp[d] = acc.x;
        outp[d + 1] = acc.y;
    }
    if (threadIdx.x == 0) {
        pm[block] = n ? mx : -INFINITY;
        pl[block] = n ? sum : 0.0f;
    }
}

/* The combine: heads blocks, one per query head, over its runs. A block's
 * weight on its own value sum is exp(m_s - M) / sum_t exp(m_t - M) l_t, which
 * thread 0 computes once into shared for the block's 256 threads to multiply
 * with. Runs that hold no position carry m = -INFINITY and l = 0. */
__global__ void attn_combine_kernel(const float *part, const float *pm, const float *pl, float *out,
                                    int heads, int head_dim, int splits) {
    __shared__ float w[66];
    const int h = blockIdx.x;
    if (threadIdx.x == 0) {
        float M = -INFINITY;
        for (int s = 0; s < splits; s++) M = fmaxf(M, pm[h * splits + s]);
        float denom = 0.0f;
        for (int s = 0; s < splits; s++) {
            const float e = expf(pm[h * splits + s] - M);
            w[s] = e;
            denom += e * pl[h * splits + s];
        }
        for (int s = 0; s < splits; s++) w[s] = denom > 0.0f ? w[s] / denom : 0.0f;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < splits; s++)
            acc += w[s] * part[(size_t) (h * splits + s) * head_dim + d];
        out[(size_t) h * head_dim + d] = acc;
    }
}

/* Append one position to a cache: the position is read from the device and
 * lands at slot(pos), the same rule the host used to compute before a captured
 * step had to stop baking it in. One thread per value of the position. */
__global__ void kv_put_kernel(__half *k, __half *v, const float *ksrc, const float *vsrc, int per,
                              const int32_t *posd, int cap, int ring) {
    const int i = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;
    if (i >= per) return;
    const int slot = ring ? (posd[0] % cap) : posd[0];
    const size_t at = (size_t) slot * (size_t) per + (size_t) i;
    k[at] = __float2half_rn(ksrc[i]);
    v[at] = __float2half_rn(vsrc[i]);
}

/* ---- the block of tokens: the prefill's kernels -------------------------
 *
 * A decode step carries one token, so every launch above is shaped for one
 * row of activations. A prefill carries a block of them at consecutive
 * positions, and the same work is then two different problems:
 *
 *  - the dense projections are a matrix times a matrix. The container still
 *    never becomes a weight matrix, so the shape that pays is the tiled
 *    tensor-core kernel already in this file (`gemm_tiled_kernel`), whose
 *    token tile is the reason the tools' prefill exists at all. The block
 *    path feeds it a transposed activation block, one launch per projection.
 *  - the routed experts are not: a 128-token block picks 8 of 128 experts per
 *    token, so each expert owns ~8 tokens and its 2*ff rows are read once for
 *    all of them. That is the reference engine's work item - `(expert, token
 *    segment, count)` at /data/ds4-dfm-rs/cuda/ds41_vq_prefill_fused.cuh:38-
 *    54, :86-116 - and the reason the fused shape exists at all (one expert
 *    is 35.4M weights there; dequantised for 8 tokens it is arithmetic
 *    intensity 8). The row walk below is this file's v3 walk, so the token
 *    segment only changes how many activations one decoded row is dotted
 *    against.
 *
 * The elementwise arithmetic needs no new kernel: rmsnorm, gelu, the residual
 * and the router's scale already take rows and widths. Attention, rotary, the
 * cache write, the router's selection and the reduce do need one launch for
 * the block instead of one per token, which is what the kernels in this
 * section are.
 */

/* Rows one warp walks per block in the expert item passes. Two: enough to
 * keep the row's bitstream in L1 between the two, few enough that the block
 * count still fills the card (a 128-token block's gate/up pass is ~1400 rows
 * over ~150 items). Diagnostic, like the other shape knobs: it changes how
 * much work one block holds, never a row's arithmetic. */
#define GM4_MOE_ROWS_PER_WARP 2

/* x is [rows][cols], xT is [cols][rows]: the layout the tiled prefill reads
 * (its B-tile is one 64-column window per thread, which is why the tools
 * transposed once at the boundary). A 32x32 tile through shared, so both the
 * read and the write are coalesced. */
__global__ void transpose_kernel(const float *x, float *xT, int rows, int cols) {
    __shared__ float tile[32][33];
    const int c = blockIdx.x * 32 + threadIdx.x;
    const int r = blockIdx.y * 32 + threadIdx.y;
    if (c < cols && r < rows) tile[threadIdx.y][threadIdx.x] = x[(size_t) r * cols + c];
    __syncthreads();
    const int ct = blockIdx.y * 32 + threadIdx.x; /* transposed col = row */
    const int rt = blockIdx.x * 32 + threadIdx.y; /* transposed row = col */
    if (ct < rows && rt < cols) xT[(size_t) rt * rows + ct] = tile[threadIdx.x][threadIdx.y];
}

/* Rotary for every token of the block, each at its own position: the same
 * table and the same pair rule as rope_kernel, with the angle from pos0 + t
 * instead of a device slot (a block's positions are the host's true ones; the
 * reference's prefill also runs direct dispatch for that reason,
 * ds4_ds41_forward.inc:1237-1240). */
__global__ void rope_rows_kernel(float *x, int heads, int head_dim, int pos0, const float *inv) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int d = threadIdx.x;
    const int half = head_dim / 2;
    if (d >= half) return;
    const float w = inv[d];
    if (w == 0.0f) return;
    float *base = x + ((size_t) t * (size_t) heads + (size_t) h) * (size_t) head_dim;
    const float angle = (float) (pos0 + t) * w;
    float s, c;
    sincosf(angle, &s, &c);
    const float a = base[d], b = base[d + half];
    base[d] = a * c - b * s;
    base[d + half] = b * c + a * s;
}

/* The block's cache writes: one row per position, the same kv_slot rule and
 * the same f16 rounding as kv_put_kernel. */
__global__ void kv_put_rows_kernel(__half *k, __half *v, const float *ksrc, const float *vsrc,
                                   int per, int pos0, int cap, int ring) {
    const int t = blockIdx.y;
    const int i = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;
    if (i >= per) return;
    const int pos = pos0 + t;
    const int slot = ring ? (pos % cap) : pos;
    const size_t at = (size_t) slot * (size_t) per + (size_t) i;
    k[at] = __float2half_rn(ksrc[(size_t) t * (size_t) per + (size_t) i]);
    v[at] = __float2half_rn(vsrc[(size_t) t * (size_t) per + (size_t) i]);
}

/* One block per (query head, query token), over the window that token may
 * see. The arithmetic is attn_split_kernel's with one run - the fp32 score
 * pass in ascending position order, the max, the sum of the exponentials,
 * then the value pass in the same order - so the prefill and a decode over
 * the same history differ only by the split rescale the decode's runs add.
 * The scores of one query fit in the launch's dynamic shared memory: the
 * widest window here is the context bound. */
__global__ void attn_rows_kernel(const float *q, const __half *k, const __half *v, float *out,
                                 int heads, int kv_heads, int head_dim, int pos0, int window,
                                 int kv_cap, int kv_ring) {
    extern __shared__ float red[];
    float *sc = red + 32;
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int group = heads / kv_heads;
    const int kv_head = h / group;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const size_t kv_stride = (size_t) kv_heads * (size_t) head_dim;
    const float *qh = q + ((size_t) t * (size_t) heads + (size_t) h) * (size_t) head_dim;
    const int positions = pos0 + t + 1;
    const int nwin = window > 0 && positions > window ? window : positions;
    const int first = positions - nwin;

    for (int p = warp; p < nwin; p += nwarps) {
        const size_t ks = kv_slot(first + p, kv_cap, kv_ring);
        const __half *kp = k + ks * kv_stride + (size_t) kv_head * head_dim;
        float acc = 0.0f;
        for (int d = lane * 2; d < head_dim; d += 64) {
            const float2 qq = *(const float2 *) (qh + d);
            const float2 kk = __half22float2(*(const __half2 *) (kp + d));
            acc += qq.x * kk.x + qq.y * kk.y;
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) sc[p] = acc; /* scaling is 1.0 */
    }
    __syncthreads();

    float mx = -INFINITY;
    for (int i = threadIdx.x; i < nwin; i += blockDim.x) mx = fmaxf(mx, sc[i]);
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int i = threadIdx.x; i < nwin; i += blockDim.x) {
        const float e = expf(sc[i] - mx);
        sc[i] = e;
        sum += e;
    }
    sum = block_sum(sum, red);

    float *o = out + ((size_t) t * (size_t) heads + (size_t) h) * (size_t) head_dim;
    for (int d = threadIdx.x * 2; d < head_dim; d += blockDim.x * 2) {
        float2 acc = make_float2(0.0f, 0.0f);
        for (int i = 0; i < nwin; i++) {
            const size_t ks = kv_slot(first + i, kv_cap, kv_ring);
            const float pr = sc[i];
            const float2 vv = __half22float2(
                *(const __half2 *) (v + ks * kv_stride + (size_t) kv_head * head_dim + d));
            acc.x = fmaf(pr, vv.x, acc.x);
            acc.y = fmaf(pr, vv.y, acc.y);
        }
        o[d] = acc.x / sum;
        o[d + 1] = acc.y / sum;
    }
}

/* The routed mixture's reduce for a block: row t sums its own k experts, the
 * same add order moe_reduce_kernel makes for one token. */
__global__ void moe_reduce_tok_kernel(const float *d, const float *w, float *out, int k, int n) {
    const int t = blockIdx.y;
    const float *dt = d + (size_t) t * (size_t) k * (size_t) n;
    const float *wt = w + (size_t) t * (size_t) k;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < k; i++) acc += wt[i] * dt[(size_t) i * n + j];
        out[(size_t) t * n + j] = acc;
    }
}

/* The expert pass's activations, gathered into the work items' order: pair p
 * (a (token, slot) routed to this expert) reads the token's activation row.
 * The reference's vqp_gather32_kernel, ds41_vq_prefill_fused.cuh:34-40. */
__global__ void moe_gather_kernel(float *xs, const float *x, const int32_t *perm, int cols) {
    const int p = blockIdx.y;
    const float *src = x + (size_t) perm[p] * (size_t) cols;
    float *dst = xs + (size_t) p * (size_t) cols;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cols; i += gridDim.x * blockDim.x)
        dst[i] = src[i];
}

/* One row decoded once and dotted against the item's activations: the block
 * walk of v3blk_row_dot, round for round and with the same funnel shifts and
 * lane map, with the codeword loaded once per round and the accumulation
 * repeated per activation. `zero` clears the accumulators; the value stays
 * WARP-LOCAL.
 *
 * The reduce and the gain are the caller's, one per payload, and that is the
 * decode path's own order: stage 0 is one launch that assigns, stage 1 another
 * that accumulates (the pair `rows_dev` makes on L->egu), and the reference
 * reduces at the end of v41_vq_row_dotN, once per matrix
 * (/data/ds4-dfm-rs/cuda/ds41_vq_prefill_fused.cuh:76-80). Reducing here and
 * then calling again for the second stage is wrong by construction: the second
 * call would start its shuffle tree from the first stage's already-reduced
 * value, so every lane carries it and the tree adds it 32 times (measured: the
 * block's gu was 33*a + b against the step's a + b). */
template <int DIM, int NT>
__device__ __forceinline__ void v3blk_row_dotN(const uint8_t *p, const Layout &L, size_t r,
                                               const float *xs, int xstride, int ntok, float *acc,
                                               int zero) {
    const uint32_t lane = threadIdx.x & 31u;
    const __half *cb = (const __half *) (p + L.cb_off);
    const uint32_t *row = (const uint32_t *) (p + L.stream_off + (size_t) r * L.mrow);
    const uint32_t R = (uint32_t) (L.nidx_row >> 5);
    const uint32_t nblk = (R + 7u) >> 3;
    const uint32_t a = (lane * 12u) >> 5, sh = (lane * 12u) & 31u;
    if (zero)
        #pragma unroll
        for (int t = 0; t < NT; t++) acc[t] = 0.0f;

    v3blk cur = v3_blk_load(row, R >= 8u ? 96u : R * 12u, NULL, 0);
    for (uint32_t b = 0; b < nblk; b++) {
        v3blk nxt;
        if (b + 1u < nblk) {
            const uint32_t nrem = R - (b + 1u) * 8u, nr = nrem < 8u ? nrem : 8u;
            nxt = v3_blk_load(row + (size_t) (b + 1u) * 96u, nr * 12u, NULL, 0);
        } else {
            nxt.w0 = nxt.w1 = nxt.w2 = nxt.ex = 0u;
        }
        const uint32_t rem = R - b * 8u;
        const uint32_t rounds = rem < 8u ? rem : 8u;
        /* The activation a block reads is the one its own 256 indices name:
         * v3_rounds advances the pointer by b*256*dim for exactly this reason
         * (:189-191), and the index map restarts at zero every block. */
        const float *xb0 = xs + (size_t) b * 256u * (size_t) DIM;
        #pragma unroll
        for (uint32_t k = 0; k < 8u; k++) {
            if (k >= rounds) break;
            const uint32_t f = k * 12u;
            const uint32_t lo_r = ((f & 31u) > 32u - 12u) ? ((f + 31u - lane) >> 5) : (f >> 5);
            const uint32_t hi_r = (((f + 1u) & 31u) > 32u - 12u) ? ((f + 32u - lane) >> 5) : ((f + 1u) >> 5);
            const uint32_t lo = __shfl_sync(0xffffffffu, v3_sel3(lo_r, cur.w0, cur.w1, cur.w2), (int) (f + a));
            const uint32_t hi = __shfl_sync(0xffffffffu, v3_sel3(hi_r, cur.w0, cur.w1, cur.w2), (int) (f + a + 1u));
            const uint32_t v = __funnelshift_r(lo, hi, sh) & 0xFFFu;
            const size_t g = (size_t) (k * 32u + lane);
            if (DIM == 4) {
                const uint2 c = __ldg((const uint2 *) cb + v);
                const float2 f0 = __half22float2(*reinterpret_cast<const __half2 *> (&c.x));
                const float2 f1 = __half22float2(*reinterpret_cast<const __half2 *> (&c.y));
                for (int t = 0; t < ntok; t++) {
                    const float4 xv = *(const float4 *) (xb0 + (size_t) t * xstride + g * 4u);
                    acc[t] += f0.x * xv.x + f0.y * xv.y + f1.x * xv.z + f1.y * xv.w;
                }
            } else {
                const uint2 c0 = __ldg((const uint2 *) cb + (size_t) v * 2u);
                const uint2 c1 = __ldg((const uint2 *) cb + (size_t) v * 2u + 1u);
                const float2 f0 = __half22float2(*reinterpret_cast<const __half2 *> (&c0.x));
                const float2 f1 = __half22float2(*reinterpret_cast<const __half2 *> (&c0.y));
                const float2 f2 = __half22float2(*reinterpret_cast<const __half2 *> (&c1.x));
                const float2 f3 = __half22float2(*reinterpret_cast<const __half2 *> (&c1.y));
                for (int t = 0; t < ntok; t++) {
                    const float *xb = xb0 + (size_t) t * xstride + g * (size_t) DIM;
                    const float4 x0 = *(const float4 *) xb;
                    const float4 x1 = *(const float4 *) (xb + 4u);
                    acc[t] += f0.x * x0.x + f0.y * x0.y + f1.x * x0.z + f1.y * x0.w
                            + f2.x * x1.x + f2.y * x1.y + f3.x * x1.z + f3.y * x1.w;
                }
            }
        }
        cur = nxt;
    }
}

/* One row's warp-local partials into lane 0's value, scaled by that payload's
 * row gain: the reduce v3blk_row_dot ends with (:250-254), lifted out so the
 * caller can apply it once per stage. */
__device__ __forceinline__ float moe_row_reduce(float partial, const __half *gr, size_t r) {
    float v = partial;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return __shfl_sync(0xffffffffu, v, 0) * __half2float(gr[r]);
}

/* The same for the short row: span_row_dot's 32-index groups and three-byte
 * index read, repeated per activation of the item. A down row is 88 indices,
 * which the block walk cannot take (88 % 32 != 0), so this is the walk the
 * expert down stack uses on the decode path too. Like its block twin the value
 * stays warp-local; the caller reduces and scales it once per payload. */
template <int DIM, int NT>
__device__ __forceinline__ void span_row_dotN(const uint8_t *p, const Layout &L, size_t r,
                                              const float *xs, int xstride, int ntok, float *acc,
                                              int zero) {
    const int lane = threadIdx.x & 31;
    const int nidx_row = L.cols / L.dim;
    const int groups = (nidx_row + 31) / 32;
    const __half *cb = (const __half *) (p + L.cb_off);
    const uint32_t mb = L.nbit > 12 ? 12u : (uint32_t) L.nbit;
    const uint32_t imsk = (1u << mb) - 1u;
    const uint8_t *row = p + L.stream_off + r * (size_t) L.mrow;
    if (zero)
        #pragma unroll
        for (int t = 0; t < NT; t++) acc[t] = 0.0f;
    for (int g = 0; g < groups; g++) {
        const int i = g * 32 + lane;
        if (i >= nidx_row) break;
        const uint32_t bit = (uint32_t) i * mb;
        const uint8_t *q = row + (bit >> 3);
        const uint32_t w = (uint32_t) q[0] | ((uint32_t) q[1] << 8) | ((uint32_t) q[2] << 16);
        const uint32_t v = (w >> (bit & 7)) & imsk;
        if (DIM == 4) {
            const uint2 c = __ldg((const uint2 *) cb + v);
            const float2 f0 = __half22float2(*reinterpret_cast<const __half2 *> (&c.x));
            const float2 f1 = __half22float2(*reinterpret_cast<const __half2 *> (&c.y));
            for (int t = 0; t < ntok; t++) {
                const float4 xv = *(const float4 *) (xs + (size_t) t * xstride + (size_t) i * 4u);
                acc[t] += f0.x * xv.x + f0.y * xv.y + f1.x * xv.z + f1.y * xv.w;
            }
        } else {
            const uint2 c0 = __ldg((const uint2 *) cb + (size_t) v * 2u);
            const uint2 c1 = __ldg((const uint2 *) cb + (size_t) v * 2u + 1u);
            const float2 f0 = __half22float2(*reinterpret_cast<const __half2 *> (&c0.x));
            const float2 f1 = __half22float2(*reinterpret_cast<const __half2 *> (&c0.y));
            const float2 f2 = __half22float2(*reinterpret_cast<const __half2 *> (&c1.x));
            const float2 f3 = __half22float2(*reinterpret_cast<const __half2 *> (&c1.y));
            for (int t = 0; t < ntok; t++) {
                const float *xb = xs + (size_t) t * xstride + (size_t) i * (size_t) DIM;
                const float4 x0 = *(const float4 *) xb;
                const float4 x1 = *(const float4 *) (xb + 4u);
                acc[t] += f0.x * x0.x + f0.y * x0.y + f1.x * x0.z + f1.y * x0.w
                        + f2.x * x1.x + f2.y * x1.y + f3.x * x1.z + f3.y * x1.w;
            }
        }
    }
}

/* The work items of one block, the reference's shape: one block per (row
 * span, item), a warp per row, the item's token segment dotted against the
 * row it decoded. `items` is a device array of (expert, first pair, count)
 * triples built on the host from the router's picks, the same host-side
 * grouping the reference makes into `vqp_item` (ds41_vq_prefill_fused.cuh:214-
 * 225); the pairs themselves are the permuted (token, slot) list, so the
 * activation of pair p is xs + p * xstride.
 *
 * gate and up are one pass because this container puts them inside the
 * expert's 2*ff rows: row j is the gate and row ff + j the up, so a single
 * launch covers both and the SwiGLU is a separate elementwise pass over the
 * compact block (gelu_mul_blocks_kernel, already row-agnostic). The reference
 * splits the two into two launches for register pressure
 * (ds41_vq_prefill_fused.cuh:88-93); here one row dot's accumulators are NT
 * floats either way, so one launch is enough. */
template <int DIM, int NT>
__global__ void moe_gu_items_kernel(const Payload a, const Payload b, const int32_t *items,
                                    const float *xs, int xstride, float *gu, int ff, int stride) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps = (uint32_t) (blockDim.x >> 5);
    const int it = blockIdx.y;
    const int32_t e = items[it * 3 + 0];
    const int32_t p0 = items[it * 3 + 1];
    const int32_t nt = items[it * 3 + 2];
    const int rows = 2 * ff;
    const uint32_t per = warps * GM4_MOE_ROWS_PER_WARP;
    /* One row per warp per iteration, striding by the warp count - the
     * reference's `r = r0 + i * 32` (ds41_vq_prefill_fused.cuh:110-112).
     * Striding by the block's whole row span instead leaves the block's second
     * half unassigned (measured: rows 4..7 of every 8 came out zero). */
    for (uint32_t i = 0; i < GM4_MOE_ROWS_PER_WARP; i++) {
        const uint32_t j = (uint32_t) (blockIdx.x * (int) per) + (threadIdx.x >> 5) + i * warps;
        if (j >= (uint32_t) rows) break;
        const size_t r = (size_t) e * (size_t) stride + (size_t) j;
        float pa[NT], pb[NT];
        v3blk_row_dotN<DIM, NT>(a.dev, a.l, r, xs + (size_t) p0 * xstride, xstride, nt, pa, 1);
        v3blk_row_dotN<DIM, NT>(b.dev, b.l, r, xs + (size_t) p0 * xstride, xstride, nt, pb, 1);
        const __half *gra = (const __half *) (a.dev + a.l.gain_off);
        const __half *grb = (const __half *) (b.dev + b.l.gain_off);
        for (int t = 0; t < nt; t++) {
            const float v = moe_row_reduce(pa[t], gra, r) + moe_row_reduce(pb[t], grb, r);
            if (lane == 0) gu[(size_t) (p0 + t) * (size_t) (2 * ff) + j] = v;
        }
    }
}

/* The down stack: the same items, the short row walk, and the output written
 * at the pair's own slot so the reduce can be the per-token k-sum. */
template <int DIM, int NT>
__global__ void moe_dn_items_kernel(const Payload a, const Payload b, const int32_t *items,
                                    const int32_t *dst, const float *act, float *d, int hidden,
                                    int ff) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps = (uint32_t) (blockDim.x >> 5);
    const int it = blockIdx.y;
    const int32_t e = items[it * 3 + 0];
    const int32_t p0 = items[it * 3 + 1];
    const int32_t nt = items[it * 3 + 2];
    const uint32_t per = warps * GM4_MOE_ROWS_PER_WARP;
    for (uint32_t i = 0; i < GM4_MOE_ROWS_PER_WARP; i++) {
        const uint32_t j = (uint32_t) (blockIdx.x * (int) per) + (threadIdx.x >> 5) + i * warps;
        if (j >= (uint32_t) hidden) break;
        const size_t r = (size_t) e * (size_t) hidden + (size_t) j;
        float pa[NT], pb[NT];
        span_row_dotN<DIM, NT>(a.dev, a.l, r, act + (size_t) p0 * (size_t) ff, ff, nt, pa, 1);
        span_row_dotN<DIM, NT>(b.dev, b.l, r, act + (size_t) p0 * (size_t) ff, ff, nt, pb, 1);
        const __half *gra = (const __half *) (a.dev + a.l.gain_off);
        const __half *grb = (const __half *) (b.dev + b.l.gain_off);
        for (int t = 0; t < nt; t++) {
            const float v = moe_row_reduce(pa[t], gra, r) + moe_row_reduce(pb[t], grb, r);
            if (lane == 0) d[(size_t) dst[p0 + t] * (size_t) hidden + j] = v;
        }
    }
}

/* The layer chain's row lists, built on the device from the router's picks so
 * the selection never travels to the host. Entry (e, r) is
 * picked[e] * stride + r: the CPU's own formula for an expert's slice
 * (forward.rs:377-383, `expert * 2 * moe_ff + r` and
 * `expert * hidden + r`). */
__global__ void rowlist_kernel(uint32_t *rows, const int *picked, int k, int count, int stride) {
    const size_t i = (size_t) blockIdx.x * (size_t) blockDim.x + (size_t) threadIdx.x;
    const size_t n = (size_t) k * (size_t) count;
    if (i >= n) return;
    const int e = (int) (i / (size_t) count);
    const int r = (int) (i % (size_t) count);
    rows[i] = (uint32_t) (picked[e] * stride + r);
}

/* y += scale * x, with the scale read from device memory: the router's
 * renormalised weight for the expert whose down projection x holds. */
__global__ void axpy_kernel(float *y, const float *x, const float *scale, size_t n) {
    const size_t i = (size_t) blockIdx.x * (size_t) blockDim.x + (size_t) threadIdx.x;
    if (i >= n) return;
    y[i] += scale[0] * x[i];
}

/* ---- the router ---------------------------------------------------------
 *
 * The reference's router (`modeling_gemma4.py:1357-1372` as the CPU forward
 * cites it): a weightless norm, times `router.scale / sqrt(hidden)`, the
 * 128-way projection, an fp32 softmax, the top-k, a renormalise and the
 * per-expert scale. The projection is the existing dim-4 row kernel; these
 * two pieces are the rest.
 */

/* out = x * w, the router's per-dimension scale. */
__global__ void mul_kernel(const float *x, const float *w, float *out, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = x[i] * w[i];
}

/* out = x * w, the router's per-dimension scale. The same product for a block
 * of rows, where `w` stays one row wide: the scale is per dimension, not per
 * token (mul_kernel's `w[i]` would read past a single row on the second
 * token). */
__global__ void mul_rows_kernel(const float *x, const float *w, float *out, int width, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = x[i] * w[i % (size_t) width];
}

/* The selection: the k highest probabilities, highest first, ties to the
 * lower index (the order `torch.topk` returns for an exact tie, which the CPU
 * forward reproduces at forward.rs:80-88), renormalised over the picked set
 * and scaled per expert. One block; the selection itself runs on one thread,
 * which is 128 comparisons per slot and keeps the tie rule explicit. */
__device__ void router_select(const float *scores, const float *per_expert_scale, int experts,
                              int k, float *probs, int *indices, float *weights, float *red) {
    const int t = threadIdx.x;
    float mx = -INFINITY;
    for (int i = t; i < experts; i += blockDim.x) mx = fmaxf(mx, scores[i]);
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int i = t; i < experts; i += blockDim.x) {
        const float e = expf(scores[i] - mx);
        probs[i] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    for (int i = t; i < experts; i += blockDim.x) probs[i] /= sum;
    __syncthreads();

    /* The selection is k rounds of a warp-wide argmax, not a serial scan on
     * one thread: the rule is the CPU's `top_k`
     * (crates/ds4-core/src/gemma4_a4b/forward.rs:80-88) - the highest
     * probability first, an exact tie to the lower index - and the compare
     * below is that order, candidate by candidate. Every lane scans its own
     * strided slice, the warp reduces the pair, and the winner is broadcast
     * so all lanes agree on the picked list. */
    if (threadIdx.x >= 32) return;
    int *pick = (int *) red;
    const int lane = threadIdx.x;
    const int per_lane = (experts + 31) / 32;
    for (int j = 0; j < k; j++) {
        float bp = -INFINITY;
        int bi = -1;
        for (int s = 0; s < per_lane; s++) {
            const int i = lane + s * 32;
            if (i >= experts) break;
            bool taken = false;
            for (int q = 0; q < j; q++) taken = taken || pick[q] == i;
            if (taken) continue;
            const float p = probs[i];
            if (p > bp) { /* ascending i keeps the lower index on a tie */
                bp = p;
                bi = i;
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float op = __shfl_down_sync(0xffffffffu, bp, off);
            const int oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (op > bp || (op == bp && oi >= 0 && (bi < 0 || oi < bi))) {
                bp = op;
                bi = oi;
            }
        }
        bi = __shfl_sync(0xffffffffu, bi, 0);
        pick[j] = bi;
        if (lane == 0) indices[j] = bi;
    }

    float picked = 0.0f;
    for (int j = 0; j < k; j++) picked += probs[pick[j]];
    if (lane != 0) return;
    for (int j = 0; j < k; j++) {
        const int expert = pick[j];
        weights[j] = probs[expert] / picked
                   * (per_expert_scale ? per_expert_scale[expert] : 1.0f);
    }
}

/* One token's selection: the step path and the host entry point. */
__global__ void router_topk_kernel(const float *scores, const float *per_expert_scale, int experts,
                                   int k, float *probs, int *indices, float *weights) {
    extern __shared__ float red[];
    router_select(scores, per_expert_scale, experts, k, probs, indices, weights, red);
}

/* The block's: one warp per token, the same body with this row's pointers, so
 * a token's picks cannot differ between a prefill and a decode. The shared
 * scratch is the warp's own 32 floats plus room for the k picks. */
__global__ void router_topk_rows_kernel(const float *scores, const float *per_expert_scale,
                                        int experts, int k, float *probs, int *indices,
                                        float *weights) {
    extern __shared__ float red[];
    const int t = blockIdx.x;
    router_select(scores + (size_t) t * experts, per_expert_scale, experts, k,
                  probs + (size_t) t * experts, indices + (size_t) t * k,
                  weights + (size_t) t * k, red);
}

/* Largest dynamic shared-memory window the device accepts; the shared-codebook
 * variant needs nc*dim f16 in one block, which a wide codebook cannot fit. */
size_t smem_max(void) {
    static size_t v = 0;
    if (!v) {
        int x = 0;
        v = (cudaDeviceGetAttribute(&x, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0) == cudaSuccess && x > 0)
            ? (size_t) x : (size_t) (48 * 1024);
    }
    return v;
}

/* Shared bytes a tiled launch needs: the two tiles, plus the codebook when it
 * is staged (cb 1). */
size_t pre_smem(int bn, int bk, const Layout &L, int cb) {
    return (size_t) (PRE_BM + bn) * bk * 2u + (cb ? (size_t) L.nc * L.dim * 2u : 0u);
}

/* Geometry the tiled kernel can take: a DQV3 payload, dim 4 or 8 (the
 * 4-thread-per-row split assumes 2 or 4 code words each), a 12-bit main
 * stream (nc >= 4096), whole bk-column rounds, and a row stream whose 4-byte
 * loads are aligned. BK is 64 only: the A-tile fill extracts CPT = BK/(4*DIM)
 * code words from one 64-bit window, which is exactly 4 code words (48 bits)
 * at BK 64. BK 128 needs 8 (96 bits) and would silently truncate, so this
 * build does not offer it (decode-gpu.cu:543-556). */
bool pre_ok(const Layout &L, int bk) {
    return L.v3 && (L.dim == 4 || L.dim == 8) && L.rows > 0 && L.cols > 0 && L.nbit >= 12
        && L.cols % bk == 0 && L.mrow % 4 == 0 && L.stream_off % 4 == 0;
}

/* The shared-memory opt-in is monotone per instantiation: it only has to
 * grow, and the largest matrix's window covers every smaller one. */
template <int BN, int BK, int CB, int DIM>
int pre_opt_in(size_t smem) {
    static size_t opted = 0;
    if (smem > opted) {
        if (cudaFuncSetAttribute(gemm_tiled_kernel<BN, BK, CB, DIM>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem) != cudaSuccess)
            return -1;
        opted = smem;
    }
    return 0;
}

template <int BN, int BK, int CB, int DIM>
int pre_launch_impl(const uint8_t *p, const Layout &L, const float *xT, int ntok, float *y) {
    const size_t smem = pre_smem(BN, BK, L, CB);
    if (pre_opt_in<BN, BK, CB, DIM>(smem) != 0) return -1;
    const dim3 grid((unsigned) ((L.nrows + PRE_BM - 1) / PRE_BM),
                    (unsigned) ((ntok + BN - 1) / BN));
    gemm_tiled_kernel<BN, BK, CB, DIM><<<grid, PRE_THREADS, smem>>>(p, L, xT, ntok, y);
    return 0;
}

/* The tile shape is a runtime choice, so the dispatch is a table over the
 * instantiations. A shape not in the table is an error, never a silent
 * fallback to the default. */
#define PRE_CASE(BN_, CB_)                                                        \
    if (bn == (BN_) && cb == (CB_))                                               \
        return L.dim == 4 ? pre_launch_impl<BN_, 64, CB_, 4>(p, L, xT, ntok, y)   \
                          : pre_launch_impl<BN_, 64, CB_, 8>(p, L, xT, ntok, y);

int pre_launch_cfg(const uint8_t *p, const Layout &L, const float *xT, int ntok, float *y,
                   int bn, int cb) {
    PRE_CASE(64, 0)
    PRE_CASE(128, 0)
    PRE_CASE(256, 0)
    PRE_CASE(64, 1)
    PRE_CASE(128, 1)
    PRE_CASE(256, 1)
    return -1;
}

int pre_legacy(const uint8_t *p, const Layout &L, const float *xT, int ntok, float *y) {
    const size_t smem = (size_t) (L.cols / L.dim) * 2;
    if (smem > 48 * 1024
        && cudaFuncSetAttribute(gemm_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int) smem) != cudaSuccess)
        return -1;
    const dim3 grid((unsigned) L.nrows, (unsigned) ((ntok + PRE_LEGACY_THREADS - 1) / PRE_LEGACY_THREADS));
    gemm_prefill_kernel<<<grid, PRE_LEGACY_THREADS, smem>>>(p, L, xT, ntok, y);
    return 0;
}

/* ---- host side -------------------------------------------------------- */

int index_bits(int nc) {
    int b = 1;
    while ((1 << b) < nc) b++;
    return b;
}

/* Preconditions of the DQV3 block kernel, verbatim from the reference:
 * a 12-bit main stream, no partial 32-index group, a 4-byte-aligned row
 * stream (crates/ds4-core/src/vq.rs mirrors the same rule). */
bool v3_blk_ok(const Layout &L) {
    return L.v3 && L.nbit >= 12 && L.nidx_row % 32 == 0 && L.mrow % 4 == 0 && L.stream_off % 4 == 0;
}

int pick_variant(const Layout &L) {
    if (v3_blk_ok(L)) return 3;
    /* A v3 row too short to be cut into whole 32-index groups: the tools'
     * block kernel cannot describe it and its plane variant is a second bit
     * array, so it takes the span kernel (see gemv_rowspan_kernel). */
    if (L.v3 && L.dim == 8 && L.prow == 0 && L.nidx_row % 32 != 0) return 4;
    if (L.dim == 4) return 2;
    return 1;
}

/* Parse one payload and fill its layout. `where` names it in an error. */
int parse_payload(const uint8_t *pay, size_t per, int dim, int nc, int rows, int cols, int payloads,
                  char *err, size_t len, Layout *out, int *variant) {
    if (per < (size_t) PAYLOAD_HEADER || memcmp(pay, "DQV3", 4) != 0) {
        snprintf(err, len, "payload is not DQV3");
        return -1;
    }
    const int p_dim = u16le(pay, 4);
    const int p_nc = u16le(pay, 6);
    const int p_rows = (int) u32le(pay, 8);
    const int p_cols = (int) u32le(pay, 12);
    const uint32_t flags = u32le(pay, 16);
    if (p_dim != dim || p_nc != nc || p_rows != rows || p_cols != cols) {
        snprintf(err, len, "payload %dx%d/%dx%d disagrees with the container %dx%d/%dx%d",
                 p_dim, p_nc, p_rows, p_cols, dim, nc, rows, cols);
        return -1;
    }
    /* Bytes 20..24 are our zero pad; the engine's layout puts `mnb` there and
     * is a different format (crates/ds4-core/src/vq.rs, from_payload). */
    if (u32le(pay, 20) != 0 || (flags & ~1u) != 0) {
        snprintf(err, len, "payload flags %#x / word20 %#x are not the tools' DQV3",
                 flags, u32le(pay, 20));
        return -1;
    }
    const int nbit = index_bits(nc);
    if ((flags & 1u) != (nbit > 12 ? 1u : 0u)) {
        snprintf(err, len, "payload plane flag %#x does not match nbit %d", flags, nbit);
        return -1;
    }
    Layout L;
    memset(&L, 0, sizeof L);
    L.dim = dim;
    L.nc = nc;
    L.rows = rows;
    L.cols = cols;
    L.nbit = nbit;
    L.v3 = 1;
    L.row0 = 0;
    L.nrows = rows;
    L.mask = (1u << nbit) - 1u;
    L.nidx_row = (size_t) cols / (size_t) dim;
    L.cb_off = PAYLOAD_HEADER;
    L.gain_off = L.cb_off + (size_t) nc * (size_t) dim * 2;
    L.stream_off = L.gain_off + (size_t) rows * 2;
    const int mb = nbit > 12 ? 12 : nbit;
    L.mrow = (L.nidx_row * (size_t) mb + 7) / 8;
    L.prow = nbit > 12 ? (L.nidx_row + 7) / 8 : 0;
    L.nidx = (size_t) rows * L.nidx_row;
    const size_t need = L.stream_off + (size_t) rows * (L.mrow + L.prow);
    if (need > per) {
        snprintf(err, len, "payload needs %zu bytes, the container gives %zu", need, per);
        return -1;
    }
    /* uint2 codebook loads and float4 activation loads. */
    if (L.cb_off % 8 != 0 || (L.prow != 0 && L.stream_off % 4 != 0) || (L.prow != 0 && L.mrow % 4 != 0)) {
        snprintf(err, len, "payload offsets are not aligned for the kernels");
        return -1;
    }
    *out = L;
    *variant = pick_variant(L);
    (void) payloads;
    return 0;
}

/* Map, check and upload one container. Returns 0 or -1 with `err`. */
int upload(gm4_gpu *g, Tensor *t, const gm4_spec *spec, gm4_account *acct, char *err, size_t len) {
    int fd = open(spec->path, O_RDONLY);
    if (fd < 0) {
        snprintf(err, len, "open %s failed", spec->path);
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < CONTAINER_HEADER + PAYLOAD_HEADER) {
        snprintf(err, len, "%s is too small to be an RQVQ container", spec->path);
        close(fd);
        return -1;
    }
    const size_t size = (size_t) st.st_size;
    const uint8_t *file = (const uint8_t *) mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file == MAP_FAILED) {
        snprintf(err, len, "mmap %s failed", spec->path);
        close(fd);
        return -1;
    }
    int rc = -1;
    uint8_t *dev = NULL;
    do {
        if (memcmp(file, "RQVQ", 4) != 0) {
            snprintf(err, len, "%s: not an RQVQ container", spec->path);
            break;
        }
        const int dim = u16le(file, 4);
        const int nc = u16le(file, 6);
        const int stages = u16le(file, 8);
        const int groups = u16le(file, 10) == 0 ? 1 : u16le(file, 10);
        const uint32_t rows = u32le(file, 12);
        const uint32_t cols = u32le(file, 16);
        const uint32_t format = u32le(file, 20);
        const uint64_t payload = u64le(file, 24);
        if (rows != spec->rows || cols != spec->cols) {
            snprintf(err, len, "%s: header is %ux%u, the release says %ux%u",
                     spec->path, rows, cols, spec->rows, spec->cols);
            break;
        }
        if (format != 0) {
            snprintf(err, len, "%s: codebook format %#x is not f16; the f16 kernels cannot read it",
                     spec->path, format);
            break;
        }
        if (dim <= 0 || nc <= 0 || stages <= 0 || groups <= 0 || rows % (uint32_t) groups != 0) {
            snprintf(err, len, "%s: header states an inconsistent geometry", spec->path);
            break;
        }
        const int npay = stages * groups;
        if (payload % (uint64_t) npay != 0 || CONTAINER_HEADER + payload != size) {
            snprintf(err, len, "%s: payload count does not describe the file", spec->path);
            break;
        }
        const size_t per = (size_t) (payload / (uint64_t) npay);
        const int group_rows = (int) (rows / (uint32_t) groups);

        /* One allocation per container, every payload on a 256-byte boundary
         * with a zeroed tail so the four-byte index window past a row's last
         * index is always readable. */
        size_t need = 0;
        for (int p = 0; p < npay; p++) need += align_to(per, PAYLOAD_AT) + SLACK;
        cudaError_t e = cudaMalloc((void **) &dev, need);
        if (e != cudaSuccess) {
            snprintf(err, len, "%s: cudaMalloc of %.1f MiB failed: %s", spec->path,
                     (double) need / (1024.0 * 1024.0), cudaGetErrorString(e));
            break;
        }
        cudaMemset(dev, 0, need);
        t->pays = (Payload *) calloc((size_t) npay, sizeof(Payload));
        if (!t->pays) {
            snprintf(err, len, "%s: out of host memory for payload descriptors", spec->path);
            break;
        }
        t->npay = npay;
        size_t at = 0;
        int bad = 0;
        for (int p = 0; p < npay && !bad; p++) {
            const size_t span = align_to(per, PAYLOAD_AT);
            Layout L;
            int variant = 0;
            char why[256];
            if (parse_payload(file + CONTAINER_HEADER + (size_t) p * per, per, dim, nc,
                              group_rows, (int) cols, npay, why, sizeof why, &L, &variant) != 0) {
                snprintf(err, len, "%s payload %d: %s", spec->path, p, why);
                bad = 1;
                break;
            }
            uint8_t *dst = dev + at;
            e = cudaMemcpy(dst, file + CONTAINER_HEADER + (size_t) p * per, per, cudaMemcpyHostToDevice);
            if (e != cudaSuccess) {
                snprintf(err, len, "%s payload %d: upload failed: %s", spec->path, p, cudaGetErrorString(e));
                bad = 1;
                break;
            }
            t->pays[p].dev = dst;
            t->pays[p].l = L;
            t->pays[p].variant = variant;
            t->pays[p].bytes = per;
            at += span;
            acct->payloads++;
            acct->payload_bytes += per;
        }
        if (bad) break;
        acct->device_bytes += need;
        acct->files_bytes += size;
        acct->tensors++;
        rc = 0;
    } while (0);
    if (dev) {
        if (rc != 0) cudaFree(dev);
    }
    munmap((void *) file, size);
    close(fd);
    return rc;
}

/* How many consecutive rows one warp walks in the span kernel. Four is the
 * default because it gives four independent index streams and four
 * accumulator chains per warp; the gate sweeps it. A diagnostic knob, in the
 * same spirit as the engine's DS41_VQ_NO_PERSIST. */
int g_span = 4;

/* The v3 stream kernel's knobs: 0 leaves every v3 payload on the block
 * kernel, N walks N consecutive rows per warp. Diagnostic, the same spirit as
 * g_span; the gate turns it on and the bands are the check that the two
 * walks agree to the bit. */
int g_stream = 0;
int g_stream_span = 8;

int launch_payload(const Payload &p, const uint32_t *rl, const float *x, float *y, int threads) {
    const int blocks = (p.l.nrows * 32 + threads - 1) / threads;
    if (p.variant == 4) {
        /* One warp per span of consecutive rows. The span is the ILP knob: a
         * row of 88 indices is 2.75 rounds, so on its own it cannot hide its
         * own head, and `span` rows of independent work can. */
        const int span = g_span > 0 ? g_span : 4;
        const int warps = (p.l.nrows + span - 1) / span;
        const int sb = (warps * 32 + threads - 1) / threads;
        if (p.l.dim == 8) gemv_rowspan_kernel<8><<<sb, threads>>>(p.dev, p.l, rl, x, y, span);
        else gemv_rowspan_kernel<4><<<sb, threads>>>(p.dev, p.l, rl, x, y, span);
    } else if (p.variant == 3 && g_stream > 0 && rl == NULL && p.l.xrows <= 0 && p.l.prow == 0
               && (p.l.dim == 8 || p.l.dim == 4)) {
        /* Consecutive rows only: a gather list or a per-block activation
         * breaks the span's single bitstream, and a 13th-bit plane is a
         * second array this kernel does not read. */
        const int span = g_stream_span > 0 ? g_stream_span : 8;
        const int warps = (p.l.nrows + span - 1) / span;
        const int sb = (warps * 32 + threads - 1) / threads;
        if (p.l.dim == 8) gemv_v3stream_kernel<8><<<sb, threads>>>(p.dev, p.l, x, y, span);
        else gemv_v3stream_kernel<4><<<sb, threads>>>(p.dev, p.l, x, y, span);
    } else if (p.variant == 3) {
        if (p.l.dim == 8) gemv_v3blk_kernel<8, 0><<<blocks, threads>>>(p.dev, p.l, rl, x, y);
        else if (p.l.dim == 4) gemv_v3blk_kernel<4, 0><<<blocks, threads>>>(p.dev, p.l, rl, x, y);
        else gemv_v3blk_kernel<0, 0><<<blocks, threads>>>(p.dev, p.l, rl, x, y);
    } else if (p.variant == 2 && p.l.dim == 4) {
        gemv_warp4_kernel<<<blocks, threads>>>(p.dev, p.l, rl, x, y);
    } else if (p.l.dim == 4) {
        gemv_warp_kernel<4><<<blocks, threads>>>(p.dev, p.l, rl, x, y);
    } else if (p.l.dim == 8) {
        gemv_warp_kernel<8><<<blocks, threads>>>(p.dev, p.l, rl, x, y);
    } else {
        gemv_warp_kernel<0><<<blocks, threads>>>(p.dev, p.l, rl, x, y);
    }
    return 0;
}

} /* namespace */

/* The handle's definition stays at file scope: `gm4_gpu` is the opaque name
 * the C ABI declares, and a second definition inside the unnamed namespace
 * would make every exported prototype ambiguous. */
struct gm4_gpu {
    Tensor *ts;
    int nt;
    float *xbuf, *ybuf;
    size_t xcap, ycap;
    uint32_t *rlbuf;   /* the gather's row list, grown on demand */
    size_t rlcap;
    float *pxt, *py;   /* the prefill's [cols][ntok] and [ntok][rows] */
    size_t pxtcap, pycap;
    float *ew[6];      /* the elementwise and router operands, grown on demand */
    size_t ewcap[6];
    float *aq, *ax, *ai, *ascores, *aout; /* attention's and RoPE's operands */
    size_t aqcap, axcap, aicap, ascorescap, aoutcap;
    float *apart, *apm, *apl; /* the split attention's per-run partials */
    size_t apartcap, apmcap, aplcap;
    __half *akv;
    size_t akvcap;
    float *kvu; /* one position's k and v, on their way into a ring slot */
    size_t kvucap;
    int32_t *posone; /* the host entry points' staged position (gm4_gpu_rope, gm4_gpu_attention) */
    int expert_legacy; /* 1 = the eight-launch expert path, for the A/B gate */
    char devname[256];
    gm4_account acct;
};

/* One layer's key/value cache on the device, the same structure the CPU
 * session keeps (crates/ds4-core/src/gemma4_a4b/forward.rs:239-269): `per`
 * values per position, `cap` positions, and `pos % cap` as the slot of a ring
 * layer. Keys and values are f16 at the boundary, which is the cache format
 * the attention gate already measured against the CPU's f32 history. The
 * handle owns both buffers; a session keeps one per layer. */
struct gm4_kv {
    __half *keys;
    __half *values;
    int kv_heads;
    int head_dim;
    int per;   /* kv_heads * head_dim */
    int cap;   /* positions held: the window for a ring, max_pos otherwise */
    int ring;  /* 1 when the slot wraps at cap (the sliding layers) */
};

/* One layer wired for the chain: the weights' tensor indices, the small
 * vectors uploaded once, and one arena holding every intermediate. The arena
 * is cut once at new and never grows, so a step allocates nothing. */
struct gm4_layer {
    int q, k, v, o, dg_t, du_t, dd_t, rt, egu, edn;
    int heads, kv_heads, head_dim, hidden, dense_ff, moe_ff, experts, top_k, threads;
    float eps, scalar;
    float *small; /* the norms, the router's folded scale, the tables */
    float *in_norm, *post_attn, *pre_ffn, *post_ffn, *pre_ffn2, *post_ffn1, *post_ffn2;
    float *q_norm, *k_norm, *router_scale, *pes, *rope_inv;
    float *arena;
    size_t arena_floats, used;
    size_t w[28]; /* each arena buffer's cut; 0-25 are the probe's numbering */
    float *xa, *xb, *hb, *qa, *qb, *ka, *kb, *va, *vb, *attn, *attn2;
    float *dg, *du, *dact, *dd, *dense, *r, *scores, *probs, *wsel;
    float *moe, *moe2, *comb, *comb2, *outd;
    float *gu, *act, *d;
    int *picked;
    uint32_t *rl1, *rl2;
    int32_t *posd; /* this chain's step position, read by rope, the cache write and attention */
    /* The VQ cache pair when this layer runs the KV unit's path; NULL is the
     * f16 cache the caller passes at every forward. */
    gm4_vqkv *kq, *vq;
    /* Opt-in section timing: nine boundaries of the chain, recorded on stream
     * 0 only when gm4_gpu_layer_timing has turned them on. A record costs a
     * few hundred nanoseconds and never stalls the stream, so the spans are
     * the chain's own timeline and not a sequence of forced waits. */
    cudaEvent_t ev[12];
    int timed;
};

/* The section boundaries, in order, matching the event indices the forward
 * records. */
#define GM4_SECTIONS 11
static const char *const gm4_section_names[GM4_SECTIONS] = {
    "input norm + qkv", "qkv norms + rope + cache", "attention",
    "o_proj + residual", "dense feed-forward", "router norm + scale",
    "router matvec", "router top-k", "routed experts", "close",
    "sync + download"};

static float *arena_cut(gm4_layer *L, int k, size_t n) {
    float *p = L->arena + L->used;
    L->used += n;
    L->w[k] = n;
    return p;
}

/* One section boundary. A record is a stream-ordered timestamp, not a wait, so
 * the recorded spans are the chain's own timeline; nothing here changes the
 * order or the arithmetic of a launch. */
static inline void mark(gm4_layer *L, int i) {
    if (L->timed) cudaEventRecord(L->ev[i], 0);
}

extern "C" {

gm4_gpu *gm4_gpu_open(const gm4_spec *specs, int n, char *err, size_t errlen) {
    /* n == 0 is a context with no artifact: the self-contained VQ gates (the
     * container reader, the encoder, the VQ attention) need the device's
     * scratch and nothing else. A negative count is still a caller bug. */
    if (n < 0 || (n > 0 && !specs)) {
        snprintf(err, errlen, "no containers to open");
        return NULL;
    }
    cudaDeviceProp prop;
    memset(&prop, 0, sizeof prop);
    cudaError_t e = cudaGetDeviceProperties(&prop, 0);
    if (e != cudaSuccess) {
        fail(err, errlen, "device properties", e);
        return NULL;
    }
    size_t free_before = 0, total = 0;
    cudaMemGetInfo(&free_before, &total);

    gm4_gpu *g = (gm4_gpu *) calloc(1, sizeof(gm4_gpu));
    if (!g) {
        snprintf(err, errlen, "out of host memory");
        return NULL;
    }
    g->ts = (Tensor *) calloc((size_t) n, sizeof(Tensor));
    if (!g->ts) {
        free(g);
        snprintf(err, errlen, "out of host memory");
        return NULL;
    }
    snprintf(g->devname, sizeof g->devname, "%s", prop.name);
    g->acct.device_free_before = free_before;
    /* The eight-launch expert path is the default until the persistent walk
     * passes the layer gate (it is one switch away, GM4_EXPERT_PERSIST=1). */
    g->expert_legacy = getenv("GM4_EXPERT_PERSIST") ? 0 : 1;

    for (int i = 0; i < n; i++) {
        if (upload(g, &g->ts[i], &specs[i], &g->acct, err, errlen) != 0) {
            gm4_gpu_close(g);
            return NULL;
        }
        g->nt++;
    }
    size_t free_after = 0;
    cudaMemGetInfo(&free_after, &total);
    g->acct.device_free_after = free_after;

    /* The workspace: the widest x and the tallest y any listed tensor needs. */
    size_t xcap = 0, ycap = 0;
    for (int i = 0; i < g->nt; i++) {
        const Layout &L = g->ts[i].pays[0].l;
        if ((size_t) L.cols > xcap) xcap = (size_t) L.cols;
        if ((size_t) L.rows > ycap) ycap = (size_t) L.rows;
    }
    g->xcap = xcap;
    g->ycap = ycap;
    e = xcap ? cudaMalloc((void **) &g->xbuf, xcap * sizeof(float)) : cudaSuccess;
    if (e != cudaSuccess) {
        snprintf(err, errlen, "workspace x cudaMalloc failed: %s", cudaGetErrorString(e));
        gm4_gpu_close(g);
        return NULL;
    }
    e = ycap ? cudaMalloc((void **) &g->ybuf, ycap * sizeof(float)) : cudaSuccess;
    if (e != cudaSuccess) {
        snprintf(err, errlen, "workspace y cudaMalloc failed: %s", cudaGetErrorString(e));
        gm4_gpu_close(g);
        return NULL;
    }
    return g;
}

void gm4_gpu_close(gm4_gpu *g) {
    if (!g) return;
    for (int i = 0; i < g->nt; i++) free(g->ts[i].pays);
    free(g->ts);
    if (g->xbuf) cudaFree(g->xbuf);
    if (g->ybuf) cudaFree(g->ybuf);
    if (g->rlbuf) cudaFree(g->rlbuf);
    if (g->pxt) cudaFree(g->pxt);
    if (g->py) cudaFree(g->py);
    for (int i = 0; i < 6; i++)
        if (g->ew[i]) cudaFree(g->ew[i]);
    if (g->aq) cudaFree(g->aq);
    if (g->ax) cudaFree(g->ax);
    if (g->ai) cudaFree(g->ai);
    if (g->ascores) cudaFree(g->ascores);
    if (g->apart) cudaFree(g->apart);
    if (g->apm) cudaFree(g->apm);
    if (g->apl) cudaFree(g->apl);
    if (g->aout) cudaFree(g->aout);
    if (g->akv) cudaFree(g->akv);
    if (g->kvu) cudaFree(g->kvu);
    if (g->posone) cudaFree(g->posone);
    free(g);
}

void gm4_gpu_account_of(const gm4_gpu *g, gm4_account *out) { *out = g->acct; }

/* Set the span kernel's rows per warp, returning the previous value. A
 * diagnostic knob: it changes how much independent work one warp holds, not
 * the arithmetic of any row, so the gate can sweep it against the same
 * reference. */
int gm4_gpu_span(int span) {
    const int prev = g_span;
    if (span > 0) g_span = span;
    return prev;
}

/* Turn the v3 cross-row stream on with `span` rows per warp, or off with 0,
 * returning the previous span (0 = off). A negative `span` only reads the
 * current value. Diagnostic, like gm4_gpu_span: it changes the pipeline over a
 * span of rows, never any row's arithmetic, so the gate can require the same
 * bands with it on. */
int gm4_gpu_stream(int span) {
    const int prev = g_stream ? g_stream_span : 0;
    if (span > 0) {
        g_stream = 1;
        g_stream_span = span;
    } else if (span == 0) {
        g_stream = 0;
    }
    return prev;
}

/* Force the eight-launch expert path (1) or the persistent walk (0),
 * returning the previous value. A diagnostic knob for the A/B gate, in the
 * spirit of the engine's DS41_VQ_NO_PERSIST; it changes the number and shape
 * of the launches, never any row's arithmetic, so the gate's bands are the
 * check and the interleaved sampler is the measurement. */
int gm4_gpu_expert_legacy(gm4_gpu *g, int legacy) {
    if (!g) return 0;
    const int prev = g->expert_legacy;
    g->expert_legacy = legacy ? 1 : 0;
    return prev;
}

const char *gm4_gpu_device_name(const gm4_gpu *g) { return g->devname; }

int gm4_gpu_variant(const gm4_gpu *g, int tensor) {
    if (!g || tensor < 0 || tensor >= g->nt) return 0;
    return g->ts[tensor].pays[0].variant;
}

int gm4_gpu_matvec(gm4_gpu *g, int tensor, const float *x, float *y, int threads,
                   char *err, size_t errlen) {
    if (!g || tensor < 0 || tensor >= g->nt) {
        snprintf(err, errlen, "tensor %d is not in the residency", tensor);
        return -1;
    }
    if (!x || !y) {
        snprintf(err, errlen, "matvec needs input and output");
        return -1;
    }
    if (threads < 32 || threads > 1024 || threads % 32) {
        snprintf(err, errlen, "launch width %d is not a warp multiple up to 1024", threads);
        return -1;
    }
    const Tensor *t = &g->ts[tensor];
    const Layout &L = t->pays[0].l;
    if (L.cols < 0 || (size_t) L.cols > g->xcap || (size_t) L.rows > g->ycap) {
        snprintf(err, errlen, "tensor geometry overflows the workspace");
        return -1;
    }
    cudaError_t e = cudaMemcpy(g->xbuf, x, (size_t) L.cols * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "x upload", e);
    /* The kernels accumulate (y +=), so every stage adds into a zeroed y. */
    e = cudaMemset(g->ybuf, 0, (size_t) L.rows * sizeof(float));
    if (e != cudaSuccess) return fail(err, errlen, "y clear", e);
    for (int p = 0; p < t->npay; p++) {
        if (t->pays[p].l.rows != L.rows || t->pays[p].l.cols != L.cols) {
            snprintf(err, errlen, "payload %d disagrees with payload 0", p);
            return -1;
        }
        launch_payload(t->pays[p], NULL, g->xbuf, g->ybuf, threads);
        e = cudaGetLastError();
        if (e != cudaSuccess) return fail(err, errlen, "matvec launch", e);
    }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "matvec sync", e);
    e = cudaMemcpy(y, g->ybuf, (size_t) L.rows * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "y download", e);
    return 0;
}

/* Grow the device row list without leaking the old one on failure. cudaFree
 * synchronises, which is harmless here: no kernel from this module is in
 * flight when a call reaches this point. */
static int ensure_rows(gm4_gpu *g, int nrows, char *err, size_t errlen) {
    if ((size_t) nrows <= g->rlcap) return 0;
    uint32_t *next = NULL;
    cudaError_t e = cudaMalloc((void **) &next, (size_t) nrows * sizeof(uint32_t));
    if (e != cudaSuccess) {
        snprintf(err, errlen, "row list cudaMalloc of %d entries failed: %s", nrows,
                 cudaGetErrorString(e));
        return -1;
    }
    if (g->rlbuf) cudaFree(g->rlbuf);
    g->rlbuf = next;
    g->rlcap = (size_t) nrows;
    return 0;
}

int gm4_gpu_rows(gm4_gpu *g, int tensor, const uint32_t *rows, int nrows, const float *x,
                 float *y, int threads, char *err, size_t errlen) {
    if (!g || tensor < 0 || tensor >= g->nt) {
        snprintf(err, errlen, "tensor %d is not in the residency", tensor);
        return -1;
    }
    if (!rows || !x || !y || nrows <= 0) {
        snprintf(err, errlen, "rows needs a row list, an input and an output");
        return -1;
    }
    if (threads < 32 || threads > 1024 || threads % 32) {
        snprintf(err, errlen, "launch width %d is not a warp multiple up to 1024", threads);
        return -1;
    }
    const Tensor *t = &g->ts[tensor];
    const Layout &L = t->pays[0].l;
    /* The kernels index the payload's own row stream, so a row past the
     * tensor's row count would read the next payload. Refuse it here, where
     * the message can name the offending entry. */
    for (int i = 0; i < nrows; i++) {
        if (rows[i] >= (uint32_t) L.rows) {
            snprintf(err, errlen, "row %u at entry %d is outside the tensor's %d rows",
                     rows[i], i, L.rows);
            return -1;
        }
    }
    if ((size_t) L.cols > g->xcap || (size_t) nrows > g->ycap) {
        snprintf(err, errlen, "gather geometry overflows the workspace");
        return -1;
    }
    if (ensure_rows(g, nrows, err, errlen) != 0) return -1;

    cudaError_t e = cudaMemcpy(g->xbuf, x, (size_t) L.cols * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "x upload", e);
    e = cudaMemcpy(g->rlbuf, rows, (size_t) nrows * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "row list upload", e);
    e = cudaMemset(g->ybuf, 0, (size_t) nrows * sizeof(float));
    if (e != cudaSuccess) return fail(err, errlen, "y clear", e);

    /* Each payload is the whole matrix for one stage, so the gather runs
     * every payload with the same list and the same output buffer; the
     * stages accumulate and each applies its own gain. */
    for (int p = 0; p < t->npay; p++) {
        if (t->pays[p].l.rows != L.rows || t->pays[p].l.cols != L.cols) {
            snprintf(err, errlen, "payload %d disagrees with payload 0", p);
            return -1;
        }
        Payload one = t->pays[p];
        one.l.row0 = 0;
        one.l.nrows = nrows;
        launch_payload(one, g->rlbuf, g->xbuf, g->ybuf, threads);
        e = cudaGetLastError();
        if (e != cudaSuccess) return fail(err, errlen, "row gather launch", e);
    }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "row gather sync", e);
    e = cudaMemcpy(y, g->ybuf, (size_t) nrows * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "row gather download", e);
    return 0;
}

/* Whether this tensor can run the tiled prefill with this tile and codebook
 * placement. Returns 1 when it can, 0 when it cannot with the reason in
 * `err`; a geometry the tiled kernel cannot take still has the legacy path,
 * which gm4_gpu_prefill asks for with bn 0. */
int gm4_gpu_prefill_ready(const gm4_gpu *g, int tensor, int bn, int cb, char *err, size_t errlen) {
    if (!g || tensor < 0 || tensor >= g->nt) {
        snprintf(err, errlen, "tensor %d is not in the residency", tensor);
        return 0;
    }
    const Layout &L = g->ts[tensor].pays[0].l;
    if (bn == 0) return 1;
    if (bn != 64 && bn != 128 && bn != 256) {
        snprintf(err, errlen, "token tile %d is not one of 64, 128, 256", bn);
        return 0;
    }
    if (cb != 0 && cb != 1) {
        snprintf(err, errlen, "codebook placement %d is not 0 (L1) or 1 (shared)", cb);
        return 0;
    }
    if (!pre_ok(L, 64)) {
        snprintf(err, errlen,
                 "dim %d nc %d cols %d nbit %d does not fit the tiled kernel: it needs dim 4 or 8, "
                 "nc >= 4096, cols a multiple of 64 and an aligned row stream",
                 L.dim, L.nc, L.cols, L.nbit);
        return 0;
    }
    if (cb && pre_smem(bn, 64, L, 1) > smem_max()) {
        snprintf(err, errlen, "a shared codebook with tile %d needs %zu bytes of shared memory, the device allows %zu",
                 bn, pre_smem(bn, 64, L, 1), smem_max());
        return 0;
    }
    return 1;
}

/* Grow a prefill buffer without leaking the old one on failure. */
static int ensure_pre(gm4_gpu *g, int which, size_t need, char *err, size_t errlen) {
    float **slot = which ? &g->py : &g->pxt;
    size_t *cap = which ? &g->pycap : &g->pxtcap;
    if (need <= *cap) return 0;
    float *next = NULL;
    cudaError_t e = cudaMalloc((void **) &next, need * sizeof(float));
    if (e != cudaSuccess) {
        snprintf(err, errlen, "prefill %s cudaMalloc of %.1f MiB failed: %s",
                 which ? "y" : "xT", (double) need * sizeof(float) / (1024.0 * 1024.0),
                 cudaGetErrorString(e));
        return -1;
    }
    if (*slot) cudaFree(*slot);
    *slot = next;
    *cap = need;
    return 0;
}

/* One prefill: y[t * rows + r] = the row dot for token t, every payload, gain
 * applied per stage. row0/nrows carve out a contiguous slice; the output keeps
 * the container's full-row layout so a slice writes exactly the addresses a
 * full launch writes for those rows. bn 0 takes the tiled kernel when the
 * geometry fits and the legacy kernel otherwise. */
static int prefill_core(gm4_gpu *g, int tensor, int row0, int nrows, const float *xT, int ntok,
                        float *y, int bn, int cb, char *err, size_t errlen) {
    if (!g || tensor < 0 || tensor >= g->nt) {
        snprintf(err, errlen, "tensor %d is not in the residency", tensor);
        return -1;
    }
    if (!xT || !y || ntok <= 0) {
        snprintf(err, errlen, "prefill needs transposed input, an output and at least one token");
        return -1;
    }
    const Tensor *t = &g->ts[tensor];
    const Layout &L = t->pays[0].l;
    if (row0 < 0 || nrows <= 0 || row0 + nrows > L.rows) {
        snprintf(err, errlen, "row slice %d+%d is outside the tensor's %d rows", row0, nrows, L.rows);
        return -1;
    }
    const int choice = bn == 0 ? (pre_ok(L, 64) ? 1 : 0) : 1;
    if (choice == 1) {
        char why[256];
        if (!gm4_gpu_prefill_ready(g, tensor, bn == 0 ? 128 : bn, cb, why, sizeof why)) {
            snprintf(err, errlen, "prefill: %s", why);
            return -1;
        }
    }
    const int tile = bn == 0 ? 128 : bn;
    if (ensure_pre(g, 0, (size_t) L.cols * (size_t) ntok, err, errlen) != 0) return -1;
    if (ensure_pre(g, 1, (size_t) L.rows * (size_t) ntok, err, errlen) != 0) return -1;

    cudaError_t e = cudaMemcpy(g->pxt, xT, (size_t) L.cols * (size_t) ntok * sizeof(float),
                               cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "prefill xT upload", e);
    e = cudaMemset(g->py, 0, (size_t) L.rows * (size_t) ntok * sizeof(float));
    if (e != cudaSuccess) return fail(err, errlen, "prefill y clear", e);

    for (int p = 0; p < t->npay; p++) {
        const Payload &one = t->pays[p];
        if (one.l.rows != L.rows || one.l.cols != L.cols) {
            snprintf(err, errlen, "payload %d disagrees with payload 0", p);
            return -1;
        }
        /* The slice belongs to the whole tensor's rows: the kernels index the
         * payload's stream by the container row, not by the slice. */
        Layout slice = one.l;
        slice.row0 = row0;
        slice.nrows = nrows;
        const int rc = choice == 1
            ? pre_launch_cfg(one.dev, slice, g->pxt, ntok, g->py, tile, cb)
            : pre_legacy(one.dev, slice, g->pxt, ntok, g->py);
        if (rc != 0) {
            snprintf(err, errlen, "prefill launch setup failed for payload %d (tile %d, placement %d)",
                     p, tile, cb);
            return -1;
        }
        e = cudaGetLastError();
        if (e != cudaSuccess) return fail(err, errlen, "prefill launch", e);
    }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "prefill sync", e);
    e = cudaMemcpy(y, g->py, (size_t) L.rows * (size_t) ntok * sizeof(float),
                   cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "prefill download", e);
    return 0;
}

int gm4_gpu_prefill(gm4_gpu *g, int tensor, const float *xT, int ntok, float *y, int bn, int cb,
                    char *err, size_t errlen) {
    if (!g || tensor < 0 || tensor >= g->nt) {
        snprintf(err, errlen, "tensor %d is not in the residency", tensor);
        return -1;
    }
    const Layout &L = g->ts[tensor].pays[0].l;
    return prefill_core(g, tensor, 0, L.rows, xT, ntok, y, bn, cb, err, errlen);
}

int gm4_gpu_prefill_rows(gm4_gpu *g, int tensor, int row0, int nrows, const float *xT, int ntok,
                         float *y, int bn, int cb, char *err, size_t errlen) {
    return prefill_core(g, tensor, row0, nrows, xT, ntok, y, bn, cb, err, errlen);
}

/* Grow one elementwise operand buffer without leaking the old one. */
static int ensure_ew(gm4_gpu *g, int slot, size_t need, char *err, size_t errlen) {
    if (need <= g->ewcap[slot]) return 0;
    float *next = NULL;
    cudaError_t e = cudaMalloc((void **) &next, need * sizeof(float));
    if (e != cudaSuccess) {
        snprintf(err, errlen, "elementwise buffer %d cudaMalloc of %.1f MiB failed: %s", slot,
                 (double) need * sizeof(float) / (1024.0 * 1024.0), cudaGetErrorString(e));
        return -1;
    }
    if (g->ew[slot]) cudaFree(g->ew[slot]);
    g->ew[slot] = next;
    g->ewcap[slot] = need;
    return 0;
}

/* Everything below is host in, host out, like the rest of this ABI. The
 * arithmetic never changes when step 9 gives the chain device pointers: only
 * the upload/download around these launches goes away. */
int gm4_gpu_rmsnorm(gm4_gpu *g, const float *x, int rows, int width, const float *weight, float *y,
                    float eps, char *err, size_t errlen) {
    if (!g || !x || !y || rows <= 0 || width <= 0) {
        snprintf(err, errlen, "rmsnorm needs an input, an output and a non-empty shape");
        return -1;
    }
    const size_t n = (size_t) rows * (size_t) width;
    if (ensure_ew(g, 0, n, err, errlen) != 0) return -1;
    if (ensure_ew(g, 1, n, err, errlen) != 0) return -1;
    if (weight && ensure_ew(g, 2, (size_t) width, err, errlen) != 0) return -1;
    cudaError_t e = cudaMemcpy(g->ew[0], x, n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "rmsnorm x upload", e);
    if (weight) {
        e = cudaMemcpy(g->ew[2], weight, (size_t) width * sizeof(float), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) return fail(err, errlen, "rmsnorm weight upload", e);
    }
    const int threads = 256;
    rmsnorm_kernel<<<(unsigned) rows, threads, threads * sizeof(float)>>>(
        g->ew[0], weight ? g->ew[2] : NULL, g->ew[1], width, eps);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "rmsnorm launch", e);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "rmsnorm sync", e);
    e = cudaMemcpy(y, g->ew[1], n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "rmsnorm download", e);
    return 0;
}

int gm4_gpu_gelu_mul(gm4_gpu *g, const float *gate, const float *up, float *act, size_t n,
                     char *err, size_t errlen) {
    if (!g || !gate || !up || !act || n == 0) {
        snprintf(err, errlen, "gelu_mul needs two inputs and an output");
        return -1;
    }
    for (int i = 0; i < 3; i++)
        if (ensure_ew(g, i, n, err, errlen) != 0) return -1;
    cudaError_t e = cudaMemcpy(g->ew[0], gate, n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "gelu_mul gate upload", e);
    e = cudaMemcpy(g->ew[1], up, n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "gelu_mul up upload", e);
    const unsigned threads = 256;
    gelu_mul_kernel<<<(unsigned) ((n + threads - 1) / threads), threads>>>(g->ew[0], g->ew[1],
                                                                         g->ew[2], n);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "gelu_mul launch", e);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "gelu_mul sync", e);
    e = cudaMemcpy(act, g->ew[2], n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "gelu_mul download", e);
    return 0;
}

int gm4_gpu_residual_scale(gm4_gpu *g, const float *x, const float *add, float *out, size_t n,
                           float scale, char *err, size_t errlen) {
    if (!g || !x || !out || n == 0) {
        snprintf(err, errlen, "residual_scale needs an input and an output");
        return -1;
    }
    for (int i = 0; i < (add ? 3 : 2); i++)
        if (ensure_ew(g, i, n, err, errlen) != 0) return -1;
    cudaError_t e = cudaMemcpy(g->ew[0], x, n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "residual x upload", e);
    if (add) {
        e = cudaMemcpy(g->ew[1], add, n * sizeof(float), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) return fail(err, errlen, "residual add upload", e);
    }
    const unsigned threads = 256;
    residual_scale_kernel<<<(unsigned) ((n + threads - 1) / threads), threads>>>(
        g->ew[0], add ? g->ew[1] : NULL, g->ew[2], n, scale);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "residual launch", e);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "residual sync", e);
    e = cudaMemcpy(out, g->ew[2], n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "residual download", e);
    return 0;
}

int gm4_gpu_softcap(gm4_gpu *g, float *y, size_t n, float cap, char *err, size_t errlen) {
    if (!g || !y || n == 0 || cap <= 0.0f) {
        snprintf(err, errlen, "softcap needs a vector and a positive cap");
        return -1;
    }
    if (ensure_ew(g, 0, n, err, errlen) != 0) return -1;
    cudaError_t e = cudaMemcpy(g->ew[0], y, n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "softcap upload", e);
    const unsigned threads = 256;
    softcap_kernel<<<(unsigned) ((n + threads - 1) / threads), threads>>>(g->ew[0], n, cap);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "softcap launch", e);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "softcap sync", e);
    e = cudaMemcpy(y, g->ew[0], n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "softcap download", e);
    return 0;
}

/* out = x * w over n values: the router's per-dimension scale, and any other
 * elementwise product the chain needs. */
int gm4_gpu_mul(gm4_gpu *g, const float *x, const float *w, float *out, size_t n, char *err,
                size_t errlen) {
    if (!g || !x || !w || !out || n == 0) {
        snprintf(err, errlen, "mul needs two inputs and an output");
        return -1;
    }
    for (int i = 0; i < 3; i++)
        if (ensure_ew(g, i, n, err, errlen) != 0) return -1;
    cudaError_t e = cudaMemcpy(g->ew[0], x, n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "mul x upload", e);
    e = cudaMemcpy(g->ew[1], w, n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "mul w upload", e);
    const unsigned threads = 256;
    mul_kernel<<<(unsigned) ((n + threads - 1) / threads), threads>>>(g->ew[0], g->ew[1], g->ew[2], n);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "mul launch", e);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "mul sync", e);
    e = cudaMemcpy(out, g->ew[2], n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "mul download", e);
    return 0;
}

/* The router's selection over `experts` scores: probs, the k indices highest
 * first, and their renormalised weights with the per-expert scale folded in.
 * `per_expert_scale` may be NULL, which leaves the weights renormalised. */
int gm4_gpu_router_topk(gm4_gpu *g, const float *scores, const float *per_expert_scale, int experts,
                        int k, float *probs, int *indices, float *weights, char *err,
                        size_t errlen) {
    if (!g || !scores || !probs || !indices || !weights || experts <= 0 || k <= 0 || k > experts) {
        snprintf(err, errlen, "router_topk needs scores, outputs and 1 <= k <= experts");
        return -1;
    }
    const size_t n = (size_t) experts;
    /* slots 0,1 hold the scores and the per-expert scale; a second pair holds
     * the probability vector and the selection's outputs as raw floats. */
    for (int i = 0; i < 5; i++)
        if (ensure_ew(g, i, n, err, errlen) != 0) return -1;
    cudaError_t e = cudaMemcpy(g->ew[0], scores, n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "router scores upload", e);
    if (per_expert_scale) {
        e = cudaMemcpy(g->ew[1], per_expert_scale, n * sizeof(float), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) return fail(err, errlen, "router scale upload", e);
    } else {
        e = cudaMemset(g->ew[1], 0, n * sizeof(float));
        if (e != cudaSuccess) return fail(err, errlen, "router scale clear", e);
    }
    unsigned threads = 32;
    while (threads < (unsigned) experts && threads < 1024) threads <<= 1;
    router_topk_kernel<<<1, threads, 32 * sizeof(float)>>>(
        g->ew[0], per_expert_scale ? g->ew[1] : NULL, experts, k, g->ew[2], (int *) g->ew[3],
        g->ew[4]);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "router_topk launch", e);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "router_topk sync", e);
    e = cudaMemcpy(probs, g->ew[2], n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e == cudaSuccess)
        e = cudaMemcpy(indices, g->ew[3], (size_t) k * sizeof(int), cudaMemcpyDeviceToHost);
    if (e == cudaSuccess)
        e = cudaMemcpy(weights, g->ew[4], (size_t) k * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "router_topk download", e);
    return 0;
}

/* Grow a plain device float buffer, keeping the old one on failure. */
static int ensure_float(float **slot, size_t *cap, size_t need, char *err, size_t errlen) {
    if (need <= *cap) return 0;
    float *next = NULL;
    cudaError_t e = cudaMalloc((void **) &next, need * sizeof(float));
    if (e != cudaSuccess) {
        snprintf(err, errlen, "cudaMalloc of %.1f MiB failed: %s",
                 (double) need * sizeof(float) / (1024.0 * 1024.0), cudaGetErrorString(e));
        return -1;
    }
    if (*slot) cudaFree(*slot);
    *slot = next;
    *cap = need;
    return 0;
}

/* ---- the layer chain's device-only launches ------------------------------
 *
 * Each one is the kernel its host entry point runs, without the upload and
 * the download around it, so a layer's intermediates stay on the device. The
 * host entry points remain the gated API; the chain is gated as a whole
 * against the CPU layer. Every buffer here must already be on the device.
 */

static int matvec_dev(gm4_gpu *g, int tensor, const float *xd, float *yd, int threads, char *err,
                      size_t errlen) {
    if (tensor < 0 || tensor >= g->nt) {
        snprintf(err, errlen, "tensor %d is not in the residency", tensor);
        return -1;
    }
    const Tensor *t = &g->ts[tensor];
    const Layout &L = t->pays[0].l;
    /* Each tensor here carries two stages. The first one assigns the row
     * instead of accumulating into it, so the output needs no clear: the chain
     * is the only writer of these buffers and it rewrites every row it reads. */
    for (int p = 0; p < t->npay; p++) {
        Payload one = t->pays[p];
        one.l.first = p == 0;
        one.l.nrows = (int) L.rows;
        launch_payload(one, NULL, xd, yd, threads);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return fail(err, errlen, "chain matvec launch", e);
    }
    return 0;
}

static int rows_dev(gm4_gpu *g, int tensor, const uint32_t *rld, int nrows, const float *xd,
                    int xrows, int xcols, float *yd, int threads, char *err, size_t errlen) {
    if (tensor < 0 || tensor >= g->nt) {
        snprintf(err, errlen, "tensor %d is not in the residency", tensor);
        return -1;
    }
    const Tensor *t = &g->ts[tensor];
    for (int p = 0; p < t->npay; p++) {
        Payload one = t->pays[p];
        one.l.row0 = 0;
        one.l.first = p == 0;
        one.l.nrows = nrows;
        one.l.xrows = xrows;
        one.l.xcols = xcols;
        launch_payload(one, rld, xd, yd, threads);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return fail(err, errlen, "chain gather launch", e);
    }
    return 0;
}

/* The whole routed expert block in two persistent launches plus the reduce,
 * the engine's shape (expert_gu_persist_kernel above). 1 = taken,
 * 0 = geometry not covered (the caller keeps the eight-launch path), -1 on a
 * CUDA error. The geometry is checked, never assumed. */
static int expert_block_dev(gm4_layer *L, gm4_gpu *g, char *err, size_t errlen) {
    /* Diagnostic switch, the engine's DS41_VQ_NO_PERSIST in spirit
     * (/data/ds4-dfm-rs/cuda/ds41_vq_decode.cuh:329-335): force the
     * eight-launch path so the persistent walk can be A/B'd against it. */
    if (g->expert_legacy) return 0;
    const Tensor *gu = &g->ts[L->egu];
    const Tensor *dn = &g->ts[L->edn];
    if (gu->npay != 2 || dn->npay != 2) return 0;
    if (gu->pays[0].variant != 3 || gu->pays[1].variant != 3) return 0;
    if (dn->pays[0].variant != 4 || dn->pays[1].variant != 4) return 0;
    if (gu->pays[0].l.dim != 8 || dn->pays[0].l.dim != 8) return 0;
    if (gu->pays[0].l.prow || dn->pays[0].l.prow) return 0;
    static int nsm = 0;
    if (!nsm && cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        return 0;
    }
    int per_sm = 1536 / GM4_EXPERT_THREADS; /* the SM's thread capacity, sm_89 */
    if (per_sm < 1) per_sm = 1;
    const int grid = nsm * per_sm;
    expert_gu_persist_kernel<8><<<grid, GM4_EXPERT_THREADS>>>(
        gu->pays[0], gu->pays[1], L->picked, L->hb, L->act, L->top_k, L->moe_ff, 2 * L->moe_ff);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "expert gate/up persistent launch", e);
    expert_dn_persist_kernel<8><<<grid, GM4_EXPERT_THREADS>>>(
        dn->pays[0], dn->pays[1], L->picked, L->act, L->d, L->top_k, L->hidden, L->moe_ff);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "expert down persistent launch", e);
    return 1;
}

/* The embedding row, reconstructed on the device: the one table read that is
 * not a dot. The token comes from the device (`tokd[0]`), so a captured step
 * can be replayed with a different token; the stages are summed raw and the
 * scale is applied once, which is `add_row_values` followed by the forward's
 * multiply (crates/ds4-core/src/vq.rs:714-726, forward.rs:319-322). */
static int embed_row_dev(gm4_gpu *g, int tensor, const int32_t *tokd, int ntok, float scale,
                         float *yd, char *err, size_t errlen) {
    if (tensor < 0 || tensor >= g->nt) {
        snprintf(err, errlen, "tensor %d is not in the residency", tensor);
        return -1;
    }
    const Tensor *t = &g->ts[tensor];
    const int cols = t->pays[0].l.cols;
    const int dim = t->pays[0].l.dim;
    const int nidx = cols / dim;
    const unsigned threads = 128;
    const dim3 grid((unsigned) ((nidx + (int) threads - 1) / (int) threads), (unsigned) ntok);
    for (int p = 0; p < t->npay; p++) {
        const Payload &one = t->pays[p];
        const int first = p == 0;
        const int last = p == t->npay - 1;
        if (dim == 8)
            embed_row_kernel<8><<<grid, threads>>>(one.dev, one.l, tokd, yd, first, last, scale);
        else if (dim == 4)
            embed_row_kernel<4><<<grid, threads>>>(one.dev, one.l, tokd, yd, first, last, scale);
        else {
            snprintf(err, errlen, "the table has dim %d, which the row kernel does not read", dim);
            return -1;
        }
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return fail(err, errlen, "embedding row launch", e);
    }
    return 0;
}

static int rmsnorm_dev(const float *xd, const float *wd, float *yd, int rows, int width, float eps,
                       char *err, size_t errlen) {
    const int threads = 256;
    rmsnorm_kernel<<<(unsigned) rows, threads, threads * sizeof(float)>>>(xd, wd, yd, width, eps);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "chain rmsnorm launch", e);
    return 0;
}

static int gelu_mul_dev(const float *gd, const float *ud, float *ad, size_t n, char *err,
                        size_t errlen) {
    const unsigned threads = 256;
    gelu_mul_kernel<<<(unsigned) ((n + threads - 1) / threads), threads>>>(gd, ud, ad, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "chain gelu_mul launch", e);
    return 0;
}

static int residual_dev(const float *xd, const float *addd, float *od, size_t n, float scale,
                        char *err, size_t errlen) {
    const unsigned threads = 256;
    residual_scale_kernel<<<(unsigned) ((n + threads - 1) / threads), threads>>>(xd, addd, od, n,
                                                                                scale);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "chain residual launch", e);
    return 0;
}

static int mul_dev(const float *xd, const float *wd, float *od, size_t n, char *err,
                   size_t errlen) {
    const unsigned threads = 256;
    mul_kernel<<<(unsigned) ((n + threads - 1) / threads), threads>>>(xd, wd, od, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "chain mul launch", e);
    return 0;
}

/* The one-row scale applied to a block: the router's per-dimension weight on
 * every token's vector. */
static int mul_rows_dev(const float *xd, const float *wd, float *od, int width, size_t n, char *err,
                        size_t errlen) {
    const unsigned threads = 256;
    mul_rows_kernel<<<(unsigned) ((n + threads - 1) / threads), threads>>>(xd, wd, od, width, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "block mul launch", e);
    return 0;
}

static int rope_dev_p(float *xd, int heads, int head_dim, const int32_t *posd, const float *invd,
                      char *err, size_t errlen) {
    rope_kernel<<<(unsigned) heads, (unsigned) (head_dim / 2)>>>(xd, heads, head_dim, posd, invd);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "chain rope launch", e);
    return 0;
}

static int router_topk_dev(const float *scoresd, const float *pesd, int experts, int k,
                           float *probsd, int *indicesd, float *weightsd, char *err,
                           size_t errlen) {
    if (k > experts) {
        snprintf(err, errlen, "top-k %d is past the %d experts", k, experts);
        return -1;
    }
    unsigned threads = 32;
    while (threads < (unsigned) experts && threads < 1024) threads <<= 1;
    router_topk_kernel<<<1, threads, 32 * sizeof(float)>>>(scoresd, pesd, experts, k, probsd,
                                                          indicesd, weightsd);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "chain router launch", e);
    return 0;
}

static int rowlist_dev(uint32_t *rows, const int *picked, int k, int count, int stride, char *err,
                       size_t errlen) {
    const size_t n = (size_t) k * (size_t) count;
    const unsigned threads = 256;
    rowlist_kernel<<<(unsigned) ((n + threads - 1) / threads), threads>>>(rows, picked, k, count,
                                                                         stride);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "chain rowlist launch", e);
    return 0;
}

static int axpy_dev(float *yd, const float *xd, const float *scaled, size_t n, char *err,
                    size_t errlen) {
    const unsigned threads = 256;
    axpy_kernel<<<(unsigned) ((n + threads - 1) / threads), threads>>>(yd, xd, scaled, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "chain axpy launch", e);
    return 0;
}

static int gelu_blocks_dev(const float *gud, float *actd, int ff, int blocks, char *err,
                           size_t errlen) {
    const unsigned threads = 256;
    if (ff <= 0 || blocks <= 0) {
        snprintf(err, errlen, "gelu blocks needs a positive width and block count");
        return -1;
    }
    const unsigned x = (unsigned) ((ff + threads - 1) / threads);
    const dim3 grid(x, (unsigned) blocks);
    gelu_mul_blocks_kernel<<<grid, threads>>>(gud, actd, ff);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "chain gelu blocks launch", e);
    return 0;
}

static int moe_reduce_dev(const float *dd, const float *wd, float *outd, int k, int n, char *err,
                          size_t errlen) {
    const unsigned threads = 256;
    if (k <= 0 || n <= 0) {
        snprintf(err, errlen, "moe reduce needs a positive k and width");
        return -1;
    }
    const unsigned blocks = (unsigned) ((n + threads - 1) / threads);
    moe_reduce_kernel<<<blocks, threads>>>(dd, wd, outd, k, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "chain moe reduce launch", e);
    return 0;
}

static int kv_put_dev(gm4_kv *kv, int pos, const int32_t *posd, const float *kd, const float *vd,
                      char *err, size_t errlen) {
    if (pos < 0 || (!kv->ring && pos >= kv->cap)) {
        snprintf(err, errlen, "chain kv position %d outside the %s cache's %d positions", pos,
                 kv->ring ? "ring" : "appending", kv->cap);
        return -1;
    }
    if (!posd) {
        snprintf(err, errlen, "the cache write needs the position on the device");
        return -1;
    }
    const int block = 128;
    kv_put_kernel<<<(kv->per + block - 1) / block, block>>>(kv->keys, kv->values, kd, vd, kv->per,
                                                           posd, kv->cap, kv->ring);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "chain kv put launch", e);
    return 0;
}

/* One query token's attention, in the split shape. `nwin` positions starting
 * at position `first` of the history `k`/`v`, every access through kv_slot so
 * a ring and a dense walk differ only in addresses.
 *
 * The number of runs per head is chosen for the card, not fixed: 64 blocks is
 * enough to fill 56 SMs, so a 16-head layer takes four runs each and a 2-head
 * full layer takes as many as the window allows. A run is at least 64
 * positions before the split doubles, because below that the block's fixed
 * costs (two block-wide reductions) outweigh the parallelism it buys. */
#define ATN_BLOCKS 64
#define ATN_MIN_RUN 64

/* The run count per head for a window of `nwin` positions. Factored out of the
 * launcher so a capture can size the scratch for its bucket's upper bound
 * before any kernel is recorded (the engine's rule,
 * /data/ds4-dfm-rs/ds4_ds41_graph.inc:205-209: nothing may allocate under
 * capture). */
static int attn_splits(int heads, int nwin) {
    int splits = ATN_BLOCKS / heads; /* runs per head to reach 64 blocks */
    if (splits < 1) splits = 1;
    while (splits > 1 && nwin / splits < ATN_MIN_RUN) splits >>= 1;
    if (splits > 64) splits = 64; /* the combine's shared weight table */
    return splits;
}

static int attention_dev(gm4_gpu *g, const float *qd, const __half *kd, const __half *vd,
                         int heads, int kv_heads, int head_dim, const int32_t *posd, int window,
                         int kv_cap, int kv_ring, float *outd, char *err, size_t errlen) {
    if (window <= 0 || head_dim <= 0 || head_dim % 2 || heads <= 0 || kv_heads <= 0
        || heads % kv_heads) {
        snprintf(err, errlen, "attention needs a positive window, an even head width and heads "
                 "divisible by kv heads");
        return -1;
    }
    if (!posd) {
        snprintf(err, errlen, "attention needs the position on the device");
        return -1;
    }
    const int splits = attn_splits(heads, window);
    const int blocks = heads * splits;
    const int chunk = (window + splits - 1) / splits;
    if (ensure_float(&g->ascores, &g->ascorescap, (size_t) blocks * (size_t) chunk, err, errlen)
        != 0)
        return -1;
    if (ensure_float(&g->apart, &g->apartcap, (size_t) blocks * (size_t) head_dim, err, errlen)
        != 0)
        return -1;
    if (ensure_float(&g->apm, &g->apmcap, (size_t) blocks, err, errlen) != 0) return -1;
    if (ensure_float(&g->apl, &g->aplcap, (size_t) blocks, err, errlen) != 0) return -1;
    attn_split_kernel<<<(unsigned) blocks, 256, 32 * sizeof(float)>>>(
        qd, kd, vd, g->ascores, g->apart, g->apm, g->apl, heads, kv_heads, head_dim, posd, window,
        splits, chunk, kv_cap, kv_ring);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "attention split launch", e);
    attn_combine_kernel<<<(unsigned) heads, 256>>>(g->apart, g->apm, g->apl, outd, heads,
                                                   head_dim, splits);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "attention combine launch", e);
    return 0;
}

static int attention_ring_dev(gm4_gpu *g, gm4_kv *kv, const float *qd, int heads, int pos,
                              const int32_t *posd, int reserve, float *outd, char *err,
                              size_t errlen) {
    if (heads % kv->kv_heads) {
        snprintf(err, errlen, "%d query heads do not divide into %d kv heads", heads,
                 kv->kv_heads);
        return -1;
    }
    if (!kv->ring && pos >= kv->cap) {
        snprintf(err, errlen, "chain position %d past the appending cache's %d", pos, kv->cap);
        return -1;
    }
    /* The grid is sized for the widest window this cache can hold - the window
     * for a ring, `reserve` for an appending cache, which is the capture's
     * bucket bound - while the kernel takes the true window from posd. A
     * direct call passes the current position, so its grid is exactly the
     * window it reads. */
    const int bound = kv->ring ? kv->cap : (reserve > pos + 1 ? reserve : pos + 1);
    return attention_dev(g, qd, kv->keys, kv->values, heads, kv->kv_heads, kv->head_dim, posd,
                         bound, kv->cap, kv->ring, outd, err, errlen);
}

/* One int on the device, for the host entry points whose signature carries a
 * position: the chain and the model read their step's position from their own
 * device slot, but gm4_gpu_rope and gm4_gpu_attention are single calls that
 * take it as an argument, so they stage it here. Blocking on purpose - these
 * calls are not on the replay path, and 4 bytes is a few microseconds. */
static int stage_pos(gm4_gpu *g, int pos, char *err, size_t errlen) {
    if (!g->posone) {
        cudaError_t e = cudaMalloc((void **) &g->posone, sizeof(int32_t));
        if (e != cudaSuccess) return fail(err, errlen, "position slot cudaMalloc", e);
    }
    cudaError_t e = cudaMemcpy(g->posone, &pos, sizeof(int32_t), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "position slot upload", e);
    return 0;
}

/* Apply one layer type's rotary frequencies to every head of `x` in place, at
 * `pos`. `inv` is the layer type's table from Config::rope_inv: head_dim/2
 * entries, zero past the rotated pairs. */
int gm4_gpu_rope(gm4_gpu *g, float *x, int heads, int head_dim, int pos, const float *inv,
                 char *err, size_t errlen) {
    if (!g || !x || !inv || heads <= 0 || head_dim <= 0 || head_dim % 2 || pos < 0) {
        snprintf(err, errlen, "rope needs an input, a table and a positive even head width");
        return -1;
    }
    if (stage_pos(g, pos, err, errlen) != 0) return -1;
    const size_t n = (size_t) heads * (size_t) head_dim;
    if (ensure_float(&g->ax, &g->axcap, n, err, errlen) != 0) return -1;
    if (ensure_float(&g->ai, &g->aicap, (size_t) (head_dim / 2), err, errlen) != 0) return -1;
    cudaError_t e = cudaMemcpy(g->ax, x, n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "rope x upload", e);
    e = cudaMemcpy(g->ai, inv, (size_t) (head_dim / 2) * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "rope table upload", e);
    rope_kernel<<<(unsigned) heads, (unsigned) (head_dim / 2)>>>(g->ax, heads, head_dim, g->posone,
                                                               g->ai);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "rope launch", e);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "rope sync", e);
    e = cudaMemcpy(x, g->ax, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "rope download", e);
    return 0;
}

/* One query token's attention against `positions` keys and values, the last
 * `window` of them when `window` is positive (the sliding layers) and all of
 * them when it is zero (the full layers). q is [heads][head_dim]; k and v are
 * [positions][kv_heads][head_dim], f32 here and rounded to f16 on upload,
 * which is the cache's format. */
int gm4_gpu_attention(gm4_gpu *g, const float *q, const float *k, const float *v, int heads,
                      int kv_heads, int head_dim, int positions, int window, float *out,
                      char *err, size_t errlen) {
    if (!g || !q || !k || !v || !out || heads <= 0 || kv_heads <= 0 || head_dim <= 0
        || head_dim % 2 || heads % kv_heads || positions <= 0 || window < 0) {
        snprintf(err, errlen,
                 "attention needs q, k, v and an output with heads a multiple of kv_heads and "
                 "an even head width");
        return -1;
    }
    const int nwin = window > 0 && positions > window ? window : positions;
    const size_t qn = (size_t) heads * (size_t) head_dim;
    const size_t kvn = (size_t) positions * (size_t) kv_heads * (size_t) head_dim;
    if (ensure_float(&g->aq, &g->aqcap, qn, err, errlen) != 0) return -1;
    if (ensure_float(&g->aout, &g->aoutcap, qn, err, errlen) != 0) return -1;
    if (kvn * 2 > g->akvcap) {
        __half *next = NULL;
        cudaError_t e = cudaMalloc((void **) &next, kvn * 2 * sizeof(__half));
        if (e != cudaSuccess) {
            snprintf(err, errlen, "attention kv cudaMalloc of %.1f MiB failed: %s",
                     (double) kvn * 2 * sizeof(__half) / (1024.0 * 1024.0), cudaGetErrorString(e));
            return -1;
        }
        if (g->akv) cudaFree(g->akv);
        g->akv = next;
        g->akvcap = kvn * 2;
    }
    /* The cache format at the boundary: round and upload. */
    __half *host_kv = (__half *) malloc(kvn * 2 * sizeof(__half));
    if (!host_kv) {
        snprintf(err, errlen, "out of host memory for the f16 cache upload");
        return -1;
    }
    for (size_t i = 0; i < kvn; i++) {
        host_kv[i] = __float2half_rn(k[i]);
        host_kv[kvn + i] = __float2half_rn(v[i]);
    }
    cudaError_t e = cudaMemcpy(g->aq, q, qn * sizeof(float), cudaMemcpyHostToDevice);
    if (e == cudaSuccess)
        e = cudaMemcpy(g->akv, host_kv, kvn * 2 * sizeof(__half), cudaMemcpyHostToDevice);
    free(host_kv);
    if (e != cudaSuccess) return fail(err, errlen, "attention upload", e);

    /* The dense walk is the ring's slot rule with cap == positions and
     * ring == 0, so both go through one launcher and the same schedule. The
     * position the kernels read is the last one this history holds. */
    if (stage_pos(g, positions - 1, err, errlen) != 0) return -1;
    if (attention_dev(g, g->aq, g->akv, g->akv + kvn, heads, kv_heads, head_dim, g->posone,
                      window > 0 ? window : nwin, positions, 0, g->aout, err, errlen) != 0)
        return -1;
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "attention sync", e);
    e = cudaMemcpy(out, g->aout, qn * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "attention download", e);
    return 0;
}

/* ---- the per-layer KV cache ---------------------------------------------
 *
 * The CPU session keeps one cache per layer (forward.rs:239-269): a sliding
 * layer holds exactly the window and overwrites its oldest slot, `slot(pos) =
 * pos % cap`; a full layer appends, `slot(pos) = pos`. The device cache is the
 * same object, preallocated: `window > 0` makes a ring of that many positions,
 * `window == 0` makes an appending cache of `max_pos`. The CPU grows a full
 * layer lazily because the checkpoint's 262,144 positions are 5 GB; here the
 * caller names the session's bound and that bound is what gets allocated.
 *
 * The write order mirrors the CPU's `kv.append` then the attention walk:
 * gm4_gpu_kv_put for position pos, then gm4_gpu_attention_ring at pos, so the
 * query sees its own key and value, exactly as `first..=pos` does.
 */

int gm4_gpu_kv_new(gm4_gpu *g, int kv_heads, int head_dim, int window, int max_pos, gm4_kv **out,
                   char *err, size_t errlen) {
    if (!g || !out) {
        snprintf(err, errlen, "kv_new needs a device handle and an out pointer");
        return -1;
    }
    *out = NULL;
    if (kv_heads <= 0 || head_dim <= 0 || window < 0 || max_pos <= 0) {
        snprintf(err, errlen,
                 "kv_new needs positive kv_heads and head_dim, a non-negative window and a "
                 "positive max_pos");
        return -1;
    }
    const int ring = window > 0;
    const int cap = ring ? window : max_pos;
    const int per = kv_heads * head_dim;
    const size_t bytes = (size_t) cap * (size_t) per * sizeof(__half);
    gm4_kv *kv = (gm4_kv *) calloc(1, sizeof(gm4_kv));
    if (!kv) {
        snprintf(err, errlen, "out of host memory for the kv handle");
        return -1;
    }
    cudaError_t e = cudaMalloc((void **) &kv->keys, bytes);
    if (e == cudaSuccess) e = cudaMalloc((void **) &kv->values, bytes);
    if (e != cudaSuccess) {
        if (kv->keys) cudaFree(kv->keys);
        free(kv);
        snprintf(err, errlen, "kv cudaMalloc of 2 x %.1f MiB failed: %s",
                 (double) bytes / (1024.0 * 1024.0), cudaGetErrorString(e));
        return -1;
    }
    /* Zeroed so a slot read before it is written is a repeatable zero rather
     * than whatever the allocator last held. The walk never reads past the
     * written prefix, so this is for the gate's benefit, not attention's. */
    e = cudaMemset(kv->keys, 0, bytes);
    if (e == cudaSuccess) e = cudaMemset(kv->values, 0, bytes);
    if (e != cudaSuccess) {
        cudaFree(kv->keys);
        cudaFree(kv->values);
        free(kv);
        return fail(err, errlen, "kv clear", e);
    }
    kv->kv_heads = kv_heads;
    kv->head_dim = head_dim;
    kv->per = per;
    kv->cap = cap;
    kv->ring = ring;
    *out = kv;
    return 0;
}

void gm4_gpu_kv_free(gm4_kv *kv) {
    if (!kv) return;
    if (kv->keys) cudaFree(kv->keys);
    if (kv->values) cudaFree(kv->values);
    free(kv);
}

size_t gm4_gpu_kv_bytes(const gm4_kv *kv) {
    if (!kv) return 0;
    return 2 * (size_t) kv->cap * (size_t) kv->per * sizeof(__half);
}

int gm4_gpu_kv_reset(gm4_gpu *g, gm4_kv *kv, char *err, size_t errlen) {
    if (!g || !kv) {
        snprintf(err, errlen, "kv_reset needs a cache");
        return -1;
    }
    const size_t bytes = (size_t) kv->cap * (size_t) kv->per * sizeof(__half);
    cudaError_t e = cudaMemset(kv->keys, 0, bytes);
    if (e == cudaSuccess) e = cudaMemset(kv->values, 0, bytes);
    if (e != cudaSuccess) return fail(err, errlen, "kv reset", e);
    return 0;
}

int gm4_gpu_kv_put(gm4_gpu *g, gm4_kv *kv, int pos, const float *k, const float *v, char *err,
                   size_t errlen) {
    if (!g || !kv || !k || !v) {
        snprintf(err, errlen, "kv_put needs a cache, a key and a value");
        return -1;
    }
    if (pos < 0 || (!kv->ring && pos >= kv->cap)) {
        snprintf(err, errlen, "kv_put position %d outside the %s cache's %d positions", pos,
                 kv->ring ? "ring" : "appending", kv->cap);
        return -1;
    }
    const size_t need = 2 * (size_t) kv->per;
    if (need > g->kvucap) {
        float *next = NULL;
        cudaError_t e = cudaMalloc((void **) &next, need * sizeof(float));
        if (e != cudaSuccess) {
            snprintf(err, errlen, "kv scratch cudaMalloc of %.1f KiB failed: %s",
                     (double) need * sizeof(float) / 1024.0, cudaGetErrorString(e));
            return -1;
        }
        if (g->kvu) cudaFree(g->kvu);
        g->kvu = next;
        g->kvucap = need;
    }
    cudaError_t e = cudaMemcpy(g->kvu, k, (size_t) kv->per * sizeof(float), cudaMemcpyHostToDevice);
    if (e == cudaSuccess)
        e = cudaMemcpy(g->kvu + kv->per, v, (size_t) kv->per * sizeof(float),
                       cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "kv upload", e);
    if (stage_pos(g, pos, err, errlen) != 0) return -1;
    const int block = 128;
    kv_put_kernel<<<(kv->per + block - 1) / block, block>>>(kv->keys, kv->values, g->kvu,
                                                            g->kvu + kv->per, kv->per, g->posone,
                                                            kv->cap, kv->ring);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "kv put launch", e);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "kv put sync", e);
    return 0;
}

/* Read one position's slot back as f32. Not on any inference path; it is how
 * the gate checks the slot walk itself, including after the ring wrapped. */
int gm4_gpu_kv_get(gm4_gpu *g, gm4_kv *kv, int pos, float *k, float *v, char *err, size_t errlen) {
    if (!g || !kv || !k || !v) {
        snprintf(err, errlen, "kv_get needs a cache and two outputs");
        return -1;
    }
    if (pos < 0 || (!kv->ring && pos >= kv->cap)) {
        snprintf(err, errlen, "kv_get position %d outside the %s cache's %d positions", pos,
                 kv->ring ? "ring" : "appending", kv->cap);
        return -1;
    }
    const int slot = kv->ring ? pos % kv->cap : pos;
    const size_t per = (size_t) kv->per;
    __half *host = (__half *) malloc(2 * per * sizeof(__half));
    if (!host) {
        snprintf(err, errlen, "out of host memory for the kv readback");
        return -1;
    }
    cudaError_t e = cudaMemcpy(host, kv->keys + (size_t) slot * per, per * sizeof(__half),
                               cudaMemcpyDeviceToHost);
    if (e == cudaSuccess)
        e = cudaMemcpy(host + per, kv->values + (size_t) slot * per, per * sizeof(__half),
                       cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        free(host);
        return fail(err, errlen, "kv readback", e);
    }
    for (size_t i = 0; i < per; i++) {
        k[i] = __half2float(host[i]);
        v[i] = __half2float(host[per + i]);
    }
    free(host);
    return 0;
}

/* A whole cache as f32, in the position-major layout a capture wants: the
 * first `positions` positions, `k` and `v` each [positions][kv_heads *
 * head_dim]. A ring is read through its slot mapping, so an unwrapped ring and
 * an appending cache are the same walk; `positions` past what the ring holds
 * is refused rather than aliased. This is the writer a calibration capture
 * needs (VQ_tools' vq-kv/1, crates/vq-kv/src/capture.rs:1-38) and it is not on
 * any inference path. */
int gm4_gpu_kv_dump(gm4_gpu *g, gm4_kv *kv, int positions, float *k, float *v, char *err,
                    size_t errlen) {
    if (!g || !kv || !k || !v) {
        snprintf(err, errlen, "kv_dump needs a cache and two outputs");
        return -1;
    }
    if (positions < 0 || positions > kv->cap) {
        snprintf(err, errlen, "kv_dump of %d positions outside the cache's %d", positions, kv->cap);
        return -1;
    }
    const size_t per = (size_t) kv->per;
    const size_t at = 2 * per * sizeof(__half);
    __half *host = (__half *) malloc(at);
    if (!host) {
        snprintf(err, errlen, "out of host memory for the kv dump");
        return -1;
    }
    for (int p = 0; p < positions; p++) {
        const size_t slot = kv->ring ? (size_t) (p % kv->cap) : (size_t) p;
        cudaError_t e = cudaMemcpy(host, kv->keys + slot * per, per * sizeof(__half),
                                   cudaMemcpyDeviceToHost);
        if (e == cudaSuccess)
            e = cudaMemcpy(host + per, kv->values + slot * per, per * sizeof(__half),
                           cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            free(host);
            return fail(err, errlen, "kv dump readback", e);
        }
        for (size_t i = 0; i < per; i++) {
            k[(size_t) p * per + i] = __half2float(host[i]);
            v[(size_t) p * per + i] = __half2float(host[per + i]);
        }
    }
    free(host);
    return 0;
}

/* ---- the toolkit's VQ KV container --------------------------------------
 *
 * The second payload format, the one vq-kv writes for a captured cache. An
 * RQVQ container (the 32-byte header the weights arrive in) wraps one DQVQ
 * payload per residual stage per codebook group:
 *
 *   [u32 'DQVQ'][u16 dim][u16 nc][u32 rows][u32 cols]
 *   [codebook nc*dim f16][g_r rows f16][index stream]
 *
 * Sources read first-hand: c/vendor/vq_qc.h:241-266 writes the payload and
 * :274-296 reads it back; c/vendor/vq_qc_bytes.h:14-22 is the byte account;
 * crates/vq-format/src/lib.rs:316-453 is the toolkit's own reader, whose
 * DQVQ branch this mirrors. Two differences from the weight payload's DQV3
 * are load-bearing: the header is 16 bytes, not 24, and the index stream is
 * one bit stream over the whole matrix (`bit = i*nbit`, row r =
 * (i*dim)/cols), not a row-major stream restarting at every row. Above 8
 * bits it is packed little-endian across a three-byte window with a guard
 * byte at the end; at exactly 8 bits it is one byte per index.
 *
 * This reader materialises f32 and exists to gate the parse against the
 * toolkit's own dump. A cache that lives in this form is read in the
 * attention, not through here (see the encoder and the VQ attention).
 */
constexpr int DQVQ_HEADER = 16;

struct KvLayout {
    int dim, nc, rows, cols, nbit;
    int row0; /* first row of this payload in the tensor, for a group */
    unsigned mask;
    size_t cb_off, gain_off, stream_off, nidx, nidx_row;
};

/* One DQVQ payload. `per` is the payload's own byte count, as the container
 * header splits the file into equal payloads. */
int kv_parse_payload(const uint8_t *pay, size_t per, char *err, size_t len, KvLayout *out) {
    if (per < (size_t) DQVQ_HEADER || memcmp(pay, "DQVQ", 4) != 0) {
        snprintf(err, len, "payload is not DQVQ");
        return -1;
    }
    const int dim = u16le(pay, 4);
    const int nc = u16le(pay, 6);
    const int rows = (int) u32le(pay, 8);
    const int cols = (int) u32le(pay, 12);
    if (dim <= 0 || nc < 2 || rows <= 0 || cols <= 0 || cols % dim != 0) {
        snprintf(err, len, "DQVQ header %dx%d/%dx%d is inconsistent", dim, nc, rows, cols);
        return -1;
    }
    KvLayout L;
    memset(&L, 0, sizeof L);
    L.dim = dim;
    L.nc = nc;
    L.rows = rows;
    L.cols = cols;
    L.nbit = index_bits(nc);
    L.mask = (1u << L.nbit) - 1u;
    L.nidx_row = (size_t) cols / (size_t) dim;
    L.nidx = (size_t) rows * L.nidx_row;
    L.cb_off = DQVQ_HEADER;
    L.gain_off = L.cb_off + (size_t) nc * (size_t) dim * 2;
    L.stream_off = L.gain_off + (size_t) rows * 2;
    /* vq_qc_bytes.h:14-19: bytes at 8 bits, otherwise a bit stream plus the
     * guard byte the three-byte window reads past its last index. */
    const size_t idx = L.nbit == 8 ? L.nidx : (L.nidx * (size_t) L.nbit + 7) / 8 + 1;
    if (L.stream_off + idx > per) {
        snprintf(err, len, "payload needs %zu bytes, the container gives %zu",
                 L.stream_off + idx, per);
        return -1;
    }
    *out = L;
    return 0;
}

/* An f16 from a byte address that need not be two-byte aligned: the toolkit's
 * DQVQ payloads are packed back to back, so the second stage of a container
 * starts at an odd offset (c/vendor/vq_qc.h:241-266 aligns nothing). The
 * reference reads them through memcpy; this is the same pair of byte loads,
 * and `__ushort_as_half` is exact for every bit pattern, NaN included. */
__device__ __forceinline__ float f16_at(const uint8_t *base, size_t i) {
    const uint8_t *q = base + i * 2;
    return __half2float(__ushort_as_half((unsigned short) ((unsigned short) q[0] |
                                                           (unsigned short) ((unsigned) q[1] << 8))));
}

/* The reference's unpack (c/vendor/vq_qc.h:274-296) as a kernel: one thread
 * per index, `C[v] * g[row]` written at that index's segment. `first` assigns
 * where the reference resolves the first stage through a zeroed tensor; the
 * arithmetic is the same f32 product and the same stage order, without the
 * `0 + x` that would turn a negative zero positive. */
__global__ void kv_decode_kernel(const uint8_t *p, KvLayout L, float *out, int first) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= L.nidx) return;
    const size_t r = i / L.nidx_row;
    const size_t j = i - r * L.nidx_row;
    const uint8_t *st = p + L.stream_off;
    uint32_t v;
    if (L.nbit == 8) {
        v = st[i];
    } else {
        const size_t bit = i * (size_t) L.nbit;
        const uint8_t *q = st + (bit >> 3);
        const uint32_t w = (uint32_t) q[0] | ((uint32_t) q[1] << 8) | ((uint32_t) q[2] << 16);
        v = (w >> (bit & 7)) & L.mask;
    }
    const uint8_t *cb = p + L.cb_off;
    const uint8_t *gr = p + L.gain_off;
    const float gain = f16_at(gr, r);
    const uint8_t *cw = cb + (size_t) v * (size_t) L.dim * 2;
    float *o = out + (size_t) (L.row0 + (int) r) * (size_t) L.cols + j * (size_t) L.dim;
    for (int d = 0; d < L.dim; d++) {
        const float x = f16_at(cw, (size_t) d) * gain;
        if (first) o[d] = x;
        else o[d] += x;
    }
}

/* An RQVQ container opened for reading: the mapped bytes and the header's
 * geometry, validated once. The decoder and the encoder both start here, so
 * the file checks cannot drift apart. */
struct VqFile {
    const uint8_t *file;
    size_t size;
    int dim, nc, stages, groups, rows, cols;
    size_t per; /* bytes per payload */
};

int vq_open_file(const char *path, char *err, size_t len, VqFile *out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        snprintf(err, len, "open %s failed", path);
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < CONTAINER_HEADER + DQVQ_HEADER) {
        snprintf(err, len, "%s is too small to be an RQVQ container", path);
        close(fd);
        return -1;
    }
    const size_t size = (size_t) st.st_size;
    const uint8_t *file = (const uint8_t *) mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); /* the mapping outlives the descriptor */
    if (file == MAP_FAILED) {
        snprintf(err, len, "mmap %s failed", path);
        return -1;
    }
    int bad = 0;
    if (memcmp(file, "RQVQ", 4) != 0) {
        snprintf(err, len, "%s: not an RQVQ container", path);
        bad = 1;
    }
    const int dim = u16le(file, 4);
    const int nc = u16le(file, 6);
    const int stages = u16le(file, 8);
    const int groups = u16le(file, 10) == 0 ? 1 : u16le(file, 10);
    const uint32_t rows = u32le(file, 12);
    const uint32_t cols = u32le(file, 16);
    const uint32_t format = u32le(file, 20);
    const uint64_t payload = u64le(file, 24);
    if (!bad && format != 0) {
        snprintf(err, len, "%s: codebook format %#x is not f16", path, format);
        bad = 1;
    }
    if (!bad && (dim <= 0 || nc <= 0 || stages <= 0 || groups <= 0 || rows == 0 || cols == 0 ||
                 rows % (uint32_t) groups != 0)) {
        snprintf(err, len, "%s: header states an inconsistent geometry", path);
        bad = 1;
    }
    const int npay = stages * groups;
    if (!bad && (payload % (uint64_t) npay != 0 || CONTAINER_HEADER + payload != size)) {
        snprintf(err, len, "%s: payload count does not describe the file", path);
        bad = 1;
    }
    if (bad) {
        munmap((void *) file, size);
        return -1;
    }
    out->file = file;
    out->size = size;
    out->dim = dim;
    out->nc = nc;
    out->stages = stages;
    out->groups = groups;
    out->rows = (int) rows;
    out->cols = (int) cols;
    out->per = (size_t) (payload / (uint64_t) npay);
    return 0;
}

void vq_close_file(VqFile *f) {
    if (!f) return;
    if (f->file) munmap((void *) f->file, f->size);
    f->file = NULL;
}

/* Decode a whole RQVQ container of DQVQ stages into row-major f32 on the
 * host, the same walk as vq_format's Container::decode: stage 0 assigns, each
 * later stage adds into the same rows. The f32 codebook format word 0 is the
 * only one the toolkit's KV encoder writes. */
int gm4_gpu_vq_decode(const char *path, float **out, int *rows, int *cols, char *err,
                      size_t errlen) {
    if (!path || !out || !rows || !cols) {
        snprintf(err, errlen, "vq_decode needs a path and three outputs");
        return -1;
    }
    VqFile vf;
    if (vq_open_file(path, err, errlen, &vf) != 0) return -1;
    const int group_rows = vf.rows / vf.groups;
    const size_t n = (size_t) vf.rows * (size_t) vf.cols;
    uint8_t *dev = NULL;
    float *host = NULL, *dout = NULL;
    int rc = -1;
    do {
        cudaError_t e = cudaMalloc((void **) &dev, vf.size);
        if (e == cudaSuccess) e = cudaMemcpy(dev, vf.file, vf.size, cudaMemcpyHostToDevice);
        if (e == cudaSuccess) e = cudaMalloc((void **) &dout, n * sizeof(float));
        if (e == cudaSuccess) e = cudaMemset(dout, 0, n * sizeof(float));
        if (e != cudaSuccess) {
            fail(err, errlen, "vq container upload", e);
            break;
        }
        int bad = 0;
        for (int s = 0; s < vf.stages && !bad; s++) {
            for (int g = 0; g < vf.groups && !bad; g++) {
                /* The header is parsed on the host mapping and the kernel
                 * reads the device copy at the same offset: a cudaMalloc
                 * pointer is not readable from the CPU. */
                const uint8_t *hpay = vf.file + CONTAINER_HEADER + (size_t) (s * vf.groups + g) * vf.per;
                const uint8_t *dpay = dev + CONTAINER_HEADER + (size_t) (s * vf.groups + g) * vf.per;
                KvLayout L;
                char why[192];
                if (kv_parse_payload(hpay, vf.per, why, sizeof why, &L) != 0) {
                    snprintf(err, errlen, "%s stage %d group %d: %s", path, s, g, why);
                    bad = 1;
                    break;
                }
                if (L.dim != vf.dim || L.nc != vf.nc || L.rows != group_rows || L.cols != vf.cols) {
                    snprintf(err, errlen, "%s stage %d group %d is %dx%d/%dx%d, the container "
                             "%dx%d/%dx%d", path, s, g, L.dim, L.nc, L.rows, L.cols, vf.dim,
                             vf.nc, group_rows, vf.cols);
                    bad = 1;
                    break;
                }
                L.row0 = g * group_rows;
                const unsigned threads = 256;
                const unsigned blocks = (unsigned) ((L.nidx + threads - 1) / threads);
                kv_decode_kernel<<<blocks, threads>>>(dpay, L, dout, s == 0 ? 1 : 0);
                e = cudaGetLastError();
                if (e != cudaSuccess) {
                    fail(err, errlen, "kv decode launch", e);
                    bad = 1;
                }
            }
        }
        if (bad) break;
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            fail(err, errlen, "kv decode sync", e);
            break;
        }
        host = (float *) malloc(n * sizeof(float));
        if (!host) {
            snprintf(err, errlen, "out of host memory for %zu floats", n);
            break;
        }
        e = cudaMemcpy(host, dout, n * sizeof(float), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            fail(err, errlen, "kv decode download", e);
            break;
        }
        *out = host;
        *rows = vf.rows;
        *cols = vf.cols;
        rc = 0;
    } while (0);
    if (rc != 0 && host) free(host);
    if (dout) cudaFree(dout);
    if (dev) cudaFree(dev);
    vq_close_file(&vf);
    return rc;
}

void gm4_gpu_vq_free(float *p) { free(p); }

/* ---- the VQ encoder: the toolkit's nearest-codeword assignment ----------
 *
 * vq_assign_q (c/vendor/vq_qc.h:70-96) scores every codeword z by
 * c2[z] - 2*(v . C[z]) and takes the first minimum, c2 = |C[z]|^2 and the dot
 * both accumulated in double and rounded to f32 once. The kernel here scores
 * in single precision: the score is the only deviation, and the gate counts
 * the indices it moves. The gain is the payload's per-row f16; the toolkit's
 * KV encoder leaves it at 1.0 (vq_qc.h:236 takes that branch when X is NULL,
 * which is the only supported KV path), so a container that carries anything
 * else is refused at open rather than encoded without its row scaling.
 *
 * The runtime encodes against a calibration codebook, never one learned per
 * block: the toolkit's k-means needs the whole capture (minutes per layer,
 * measured 2026-11-02), which a step cannot pay. That is the KV unit's named
 * deviation, and the end-to-end gate measures what it costs.
 */
__device__ __forceinline__ float f16_at(const uint8_t *base, size_t i);

#define VQ_WARPS 32
#define VQ_THREADS (VQ_WARPS * 32)

/* One warp per segment. The stage's codebook and its c2 sit in shared memory,
 * so a segment's 4096 codewords cost shared-memory reads, not L2 traffic, and
 * the 80 KiB load is amortised over a block's 32 segments. A decode step has
 * only `cols/dim` segments to fill, which is the whole grid it needs.
 * `idx` is [nrows][cols/dim], u16. */
__global__ void vq_assign_kernel(const uint8_t *cb, const float *c2, int nc, int dim,
                                 const float *x, int cols, size_t nseg, uint16_t *idx, float *rec) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int nbytes = nc * dim * 2;
    float *sc2 = (float *) (smem + nbytes);
    const uint4 *src = (const uint4 *) cb;
    uint4 *dst = (uint4 *) smem;
    const int nq = nbytes / 16;
    for (int i = threadIdx.x; i < nq; i += blockDim.x) dst[i] = src[i];
    for (int i = threadIdx.x; i < nc; i += blockDim.x) sc2[i] = c2[i];
    __syncthreads();

    const size_t seg = (size_t) blockIdx.x * VQ_WARPS + (threadIdx.x >> 5);
    if (seg >= nseg) return;
    const int lane = threadIdx.x & 31;
    const int nidx_row = cols / dim;
    const size_t row = seg / (size_t) nidx_row;
    const int s = (int) (seg % (size_t) nidx_row);
    const int lim = dim < 8 ? dim : 8;
    const float *xs = x + row * (size_t) cols + (size_t) s * dim;
    float xv[8];
    for (int d = 0; d < lim; d++) xv[d] = xs[d];
    float best = 3.4e38f;
    int bi = 0;
    if (lim == 8) {
        /* dim 8 is one 16-byte codeword: two half2 pairs per uint4. The
         * accumulate order is the scalar loop's (d = 0..7, one f32 fma each),
         * so the codebook read is the only thing that changed. */
        for (int c = lane; c < nc; c += 32) {
            const uint4 w = *(const uint4 *) (smem + (size_t) c * 16);
            const float2 f0 = __half22float2(*(const __half2 *) &w.x);
            const float2 f1 = __half22float2(*(const __half2 *) &w.y);
            const float2 f2 = __half22float2(*(const __half2 *) &w.z);
            const float2 f3 = __half22float2(*(const __half2 *) &w.w);
            float dot = 0.0f;
            dot = fmaf(xv[0], f0.x, dot);
            dot = fmaf(xv[1], f0.y, dot);
            dot = fmaf(xv[2], f1.x, dot);
            dot = fmaf(xv[3], f1.y, dot);
            dot = fmaf(xv[4], f2.x, dot);
            dot = fmaf(xv[5], f2.y, dot);
            dot = fmaf(xv[6], f3.x, dot);
            dot = fmaf(xv[7], f3.y, dot);
            const float v = sc2[c] - 2.0f * dot;
            if (v < best) {
                best = v;
                bi = c;
            }
        }
    } else {
        for (int c = lane; c < nc; c += 32) {
            const uint8_t *cw = smem + (size_t) c * dim * 2;
            float dot = 0.0f;
            for (int d = 0; d < lim; d++) dot = fmaf(xv[d], f16_at(cw, d), dot);
            const float v = sc2[c] - 2.0f * dot;
            if (v < best) {
                best = v;
                bi = c;
            }
        }
    }
    /* The first minimum of the reference's argmin: a merge that keeps the
     * lower codeword on an equal score reproduces it across the lanes. */
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, best, off);
        const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov < best || (ov == best && oi < bi)) {
            best = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        idx[seg] = (uint16_t) bi;
        if (rec) {
            const uint8_t *cw = smem + (size_t) bi * dim * 2;
            float *o = rec + row * (size_t) cols + (size_t) s * dim;
            for (int d = 0; d < dim; d++) o[d] = f16_at(cw, d);
        }
    }
}

struct CbStage {
    uint8_t *cb; /* device: nc*dim f16 */
    float *c2;   /* device: nc, the f32 of the reference's double sum */
};

struct gm4_cb {
    int dim, nc, stages, rows, cols;
    CbStage *st;
};

/* Open a calibration container and keep its codebooks resident. This is the
 * codebook set a runtime encode needs: one per stage, the same one the
 * capture was encoded with. */
int gm4_gpu_cb_open(const char *path, gm4_cb **out, char *err, size_t errlen) {
    if (!path || !out) {
        snprintf(err, errlen, "cb_open needs a container path");
        return -1;
    }
    VqFile vf;
    if (vq_open_file(path, err, errlen, &vf) != 0) return -1;
    char why[192] = {0};
    if (vf.groups != 1) {
        snprintf(err, errlen, "%s holds %d codebook groups; a VQ KV container holds one per stage",
                 path, vf.groups);
        vq_close_file(&vf);
        return -1;
    }
    if (vf.dim > 8 || (vf.nc * vf.dim * 2) % 16 != 0) {
        snprintf(err, errlen, "%s: dim %d is outside the encoder's 1..8 or the codebook is not "
                 "16-byte aligned", path, vf.dim);
        vq_close_file(&vf);
        return -1;
    }
    gm4_cb *cb = (gm4_cb *) calloc(1, sizeof *cb);
    if (!cb) {
        snprintf(err, errlen, "out of host memory for the codebook set");
        vq_close_file(&vf);
        return -1;
    }
    cb->dim = vf.dim;
    cb->nc = vf.nc;
    cb->stages = vf.stages;
    cb->rows = vf.rows;
    cb->cols = vf.cols;
    cb->st = (CbStage *) calloc((size_t) vf.stages, sizeof *cb->st);
    int rc = cb->st ? 0 : -1;
    if (rc != 0) snprintf(err, errlen, "out of host memory for %d stages", vf.stages);

    const size_t smem = (size_t) vf.nc * vf.dim * 2 + (size_t) vf.nc * sizeof(float);
    if (rc == 0) {
        int max_shared = 0;
        cudaError_t e = cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                               0);
        if (e == cudaSuccess) {
            e = cudaFuncSetAttribute((const void *) vq_assign_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        }
        if (e != cudaSuccess || (size_t) max_shared < smem) {
            snprintf(err, errlen, "%s: a %d x %d codebook needs %.1f KiB of shared memory, the "
                     "device offers %.1f KiB", path, vf.nc, vf.dim, smem / 1024.0,
                     max_shared / 1024.0);
            rc = -1;
        }
    }
    for (int s = 0; s < vf.stages && rc == 0; s++) {
        const uint8_t *pay = vf.file + CONTAINER_HEADER + (size_t) s * vf.per;
        KvLayout L;
        if (kv_parse_payload(pay, vf.per, why, sizeof why, &L) != 0) {
            snprintf(err, errlen, "%s stage %d: %s", path, s, why);
            rc = -1;
            break;
        }
        if (L.dim != vf.dim || L.nc != vf.nc || L.rows != vf.rows || L.cols != vf.cols) {
            snprintf(err, errlen, "%s stage %d disagrees with the container header", path, s);
            rc = -1;
            break;
        }
        /* The payload's gains are per row and the runtime has none for a new
         * row, so only a unit-gain container can be encoded against. */
        for (int r = 0; r < L.rows; r++) {
            const uint16_t h = (uint16_t) (pay[L.gain_off + (size_t) r * 2] |
                                           ((uint16_t) pay[L.gain_off + (size_t) r * 2 + 1] << 8));
            if (h != 0x3c00u) {
                snprintf(err, errlen, "%s stage %d row %d carries gain 0x%04x, not 1.0; the "
                         "runtime encoder has no gain to apply", path, s, r, h);
                rc = -1;
                break;
            }
        }
        if (rc != 0) break;
        const size_t nb = (size_t) vf.nc * vf.dim * 2;
        float *c2h = (float *) malloc((size_t) vf.nc * sizeof(float));
        cudaError_t e = cudaMalloc((void **) &cb->st[s].cb, nb);
        if (e == cudaSuccess) e = cudaMemcpy(cb->st[s].cb, pay + L.cb_off, nb, cudaMemcpyHostToDevice);
        if (e == cudaSuccess) e = cudaMalloc((void **) &cb->st[s].c2, (size_t) vf.nc * sizeof(float));
        if (e != cudaSuccess || !c2h) {
            free(c2h);
            if (e == cudaSuccess) snprintf(err, errlen, "out of host memory for c2");
            else fail(err, errlen, "codebook upload", e);
            rc = -1;
            break;
        }
        /* c2 exactly as vq_assign_q computes it: a double sum per codeword,
         * rounded once (c/vendor/vq_qc.h:75-77). */
        for (int c = 0; c < vf.nc; c++) {
            double acc = 0.0;
            for (int d = 0; d < vf.dim; d++) {
                const uint8_t *p = pay + L.cb_off + ((size_t) c * vf.dim + d) * 2;
                const float v = __half2float(__ushort_as_half((unsigned short) (p[0] |
                                                      ((unsigned) p[1] << 8))));
                acc += (double) v * (double) v;
            }
            c2h[c] = (float) acc;
        }
        e = cudaMemcpy(cb->st[s].c2, c2h, (size_t) vf.nc * sizeof(float), cudaMemcpyHostToDevice);
        free(c2h);
        if (e != cudaSuccess) {
            fail(err, errlen, "c2 upload", e);
            rc = -1;
        }
    }
    vq_close_file(&vf);
    if (rc != 0) {
        gm4_gpu_cb_free(cb);
        return -1;
    }
    *out = cb;
    return 0;
}

void gm4_gpu_cb_free(gm4_cb *cb) {
    if (!cb) return;
    for (int s = 0; s < cb->stages; s++) {
        if (cb->st) {
            if (cb->st[s].cb) cudaFree(cb->st[s].cb);
            if (cb->st[s].c2) cudaFree(cb->st[s].c2);
        }
    }
    free(cb->st);
    free(cb);
}

int gm4_gpu_cb_geometry(const gm4_cb *cb, int *nc, int *dim, int *stages, int *cols) {
    if (!cb) return -1;
    if (nc) *nc = cb->nc;
    if (dim) *dim = cb->dim;
    if (stages) *stages = cb->stages;
    if (cols) *cols = cb->cols;
    return 0;
}

/* y += x */
__global__ void vq_add_kernel(float *y, const float *x, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

/* resid = x - y */
__global__ void vq_sub_kernel(float *dst, const float *x, const float *y, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = x[i] - y[i];
}

/* Encode `nrows` rows of `cols` f32 against the container's codebooks, the
 * reference's residual chain: stage 0 assigns the row, each later stage
 * assigns what the earlier ones left over. `idx` is [stages][nrows][cols/dim]
 * u16; `rec` (optional) is the f32 reconstruction the stages add up to. */
int gm4_gpu_vq_encode(gm4_cb *cb, const float *x, int nrows, int cols, uint16_t *idx, float *rec,
                      char *err, size_t errlen) {
    if (!cb || !x || !idx || nrows <= 0 || cols != cb->cols) {
        snprintf(err, errlen, "vq_encode needs a codebook set, %d-column rows and an index output",
                 cb ? cb->cols : 0);
        return -1;
    }
    const size_t n = (size_t) nrows * (size_t) cols;
    const size_t nq = (size_t) cb->nc * (size_t) cb->dim;
    float *xd = NULL, *resid = NULL, *recd = NULL, *part = NULL;
    uint16_t *idxd = NULL;
    int rc = -1;
    do {
        cudaError_t e = cudaMalloc((void **) &xd, n * sizeof(float));
        if (e == cudaSuccess) e = cudaMemcpy(xd, x, n * sizeof(float), cudaMemcpyHostToDevice);
        if (e == cudaSuccess) e = cudaMalloc((void **) &resid, n * sizeof(float));
        if (e == cudaSuccess) e = cudaMalloc((void **) &recd, n * sizeof(float));
        if (e == cudaSuccess) e = cudaMemset(recd, 0, n * sizeof(float));
        if (e == cudaSuccess) e = cudaMalloc((void **) &part, n * sizeof(float));
        if (e == cudaSuccess)
            e = cudaMalloc((void **) &idxd, (size_t) cb->stages * n / (size_t) cb->dim * 2);
        if (e != cudaSuccess) {
            fail(err, errlen, "vq encode buffers", e);
            break;
        }
        const size_t smem = nq * 2 + (size_t) cb->nc * sizeof(float);
        const size_t nseg = (size_t) nrows * (size_t) (cols / cb->dim);
        const unsigned blocks = (unsigned) ((nseg + VQ_WARPS - 1) / VQ_WARPS);
        for (int s = 0; s < cb->stages; s++) {
            const float *in = s == 0 ? xd : resid;
            uint16_t *idxs = idxd + (size_t) s * n / (size_t) cb->dim;
            vq_assign_kernel<<<blocks, VQ_THREADS, smem>>>(cb->st[s].cb, cb->st[s].c2, cb->nc,
                                                           cb->dim, in, cols, nseg, idxs, part);
            e = cudaGetLastError();
            if (e == cudaSuccess) e = cudaDeviceSynchronize();
            if (e != cudaSuccess) {
                fail(err, errlen, "vq assign launch", e);
                break;
            }
            const unsigned t = 256;
            const unsigned b = (unsigned) ((n + t - 1) / t);
            vq_add_kernel<<<b, t>>>(recd, part, n); /* recd += part */
            if (s + 1 < cb->stages) {
                vq_sub_kernel<<<b, t>>>(resid, xd, recd, n);
                e = cudaGetLastError();
                if (e == cudaSuccess) e = cudaDeviceSynchronize();
                if (e != cudaSuccess) {
                    fail(err, errlen, "vq residual", e);
                    break;
                }
            }
        }
        if (e != cudaSuccess) break;
        e = cudaMemcpy(idx, idxd, (size_t) cb->stages * n / (size_t) cb->dim * 2,
                       cudaMemcpyDeviceToHost);
        if (e == cudaSuccess && rec) e = cudaMemcpy(rec, recd, n * sizeof(float), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            fail(err, errlen, "vq encode readback", e);
            break;
        }
        rc = 0;
    } while (0);
    if (xd) cudaFree(xd);
    if (resid) cudaFree(resid);
    if (recd) cudaFree(recd);
    if (part) cudaFree(part);
    if (idxd) cudaFree(idxd);
    return rc;
}

/* ---- attention over a VQ cache -----------------------------------------
 *
 * The cache is the container's own bytes: one codebook and one index stream
 * per residual stage, resident. Three facts of the payload decide the kernel,
 * all read from the toolkit first-hand (c/vendor/vq_qc.h:241-296,
 * crates/vq-format/src/lib.rs:316-453, c/kv_encode.c):
 *
 *  - the stream is one bit stream over the whole matrix, index i at bit
 *    i*nbit, packed little-endian across a three-byte window with one guard
 *    byte (kv_parse_payload above), so the row of slot r starts at bit
 *    r*nidx_row*nbit;
 *  - a row's value is its stages' codewords summed, C0[idx0] + C1[idx1], the
 *    sum kv_decode_kernel writes and gm4_gpu_vq_encode reproduces bit for
 *    bit;
 *  - the per-row gain is 1.0 on this path by construction: the toolkit's KV
 *    encoder takes the X == NULL branch (vq_qc.h:236) and the encoder here
 *    refuses any other container.
 *
 * So the score pass reads an index per 8-wide segment and dots that
 * codeword with q's segment; the value pass reads the same two codewords per
 * dimension pair. This is the decode fused into the attention: no row and no
 * matrix is materialised, which is the format's whole point.
 */
#define VQ_STAGES 8

struct VqKvDev {
    const uint8_t *cb[VQ_STAGES];
    const uint8_t *stream[VQ_STAGES];
    int stages, dim, nbit, nidx_row, rows;
    unsigned mask;
};

/* Index j of row r, in the reference's three-byte window. */
__device__ __forceinline__ unsigned vq_index(const uint8_t *st, int nbit, unsigned mask, size_t r,
                                             int j, int nidx_row) {
    const size_t i = r * (size_t) nidx_row + (size_t) j;
    if (nbit == 8) return st[i];
    const size_t bit = i * (size_t) nbit;
    const uint8_t *q = st + (bit >> 3);
    const unsigned w = (unsigned) q[0] | ((unsigned) q[1] << 8) | ((unsigned) q[2] << 16);
    return (w >> (bit & 7)) & mask;
}

/* attn_split_kernel's shape and softmax, over rows read from the index
 * streams. The run schedule, the max-rescale combine and the fp32 softmax are
 * the f16 kernel's, so the two paths differ only in where a key or value
 * comes from - which is what makes their difference the codec's. */
__global__ void attn_vq_split_kernel(const float *q, VqKvDev K, VqKvDev V, float *scores,
                                     float *part, float *pm, float *pl, int heads, int kv_heads,
                                     int head_dim, const int32_t *posd, int window, int splits,
                                     int chunk, int kv_cap, int kv_ring) {
    extern __shared__ float red[];
    const int block = blockIdx.x;
    const int h = block / splits;
    const int s = block - h * splits;
    const int group = heads / kv_heads;
    const int kv_head = h / group;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const float *qh = q + (size_t) h * (size_t) head_dim;
    float *sc = scores + (size_t) block * (size_t) chunk;
    float *outp = part + (size_t) block * (size_t) head_dim;
    const int positions = posd[0] + 1;
    const int nwin = window > 0 && positions > window ? window : positions;
    const int first = positions - nwin;
    const int lo = s * chunk;
    const int hi = lo + chunk < nwin ? lo + chunk : nwin;
    const int n = hi > lo ? hi - lo : 0;
    const int dim = K.dim;
    const int nseg = head_dim / dim;
    const int seg0 = kv_head * nseg;

    /* The score pass: a warp takes every nwarps-th position of the run, the
     * run's segments are spread over the lanes, and each lane dots one
     * segment's codewords per stage against q. dim 8 is one 16-byte
     * codeword, read as vq_assign_kernel reads it. */
    for (int p = lo + warp; p < hi; p += nwarps) {
        const size_t ks = kv_slot(first + p, kv_cap, kv_ring);
        float acc = 0.0f;
        for (int t = lane; t < nseg; t += 32) {
            const float *qs = qh + (size_t) t * (size_t) dim;
            for (int st = 0; st < K.stages; st++) {
                const unsigned v =
                    vq_index(K.stream[st], K.nbit, K.mask, ks, seg0 + t, K.nidx_row);
                const uint8_t *cw = K.cb[st] + (size_t) v * (size_t) dim * 2;
                if (dim == 8) {
                    const uint4 w = *(const uint4 *) cw;
                    const float2 f0 = __half22float2(*(const __half2 *) &w.x);
                    const float2 f1 = __half22float2(*(const __half2 *) &w.y);
                    const float2 f2 = __half22float2(*(const __half2 *) &w.z);
                    const float2 f3 = __half22float2(*(const __half2 *) &w.w);
                    acc = fmaf(qs[0], f0.x, acc);
                    acc = fmaf(qs[1], f0.y, acc);
                    acc = fmaf(qs[2], f1.x, acc);
                    acc = fmaf(qs[3], f1.y, acc);
                    acc = fmaf(qs[4], f2.x, acc);
                    acc = fmaf(qs[5], f2.y, acc);
                    acc = fmaf(qs[6], f3.x, acc);
                    acc = fmaf(qs[7], f3.y, acc);
                } else {
                    for (int d = 0; d < dim; d++) acc = fmaf(qs[d], f16_at(cw, (size_t) d), acc);
                }
            }
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) sc[p - lo] = acc; /* scaling is 1.0 (modeling_gemma4.py:846) */
    }
    __syncthreads();

    float mx = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, sc[i]);
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float e = expf(sc[i] - mx);
        sc[i] = e;
        sum += e;
    }
    sum = block_sum(sum, red);

    /* The value pass: a thread owns one dimension pair, which lies inside one
     * segment, and walks the run in position order. The pair's two codewords
     * are f16, so a dim-8 container reads one half2 per stage. */
    const int base = kv_head * head_dim;
    for (int d = threadIdx.x * 2; d < head_dim; d += blockDim.x * 2) {
        const int col = base + d;
        const int j = col / dim;
        const int off = col % dim;
        float2 acc = make_float2(0.0f, 0.0f);
        for (int i = 0; i < n; i++) {
            const size_t ks = kv_slot(first + lo + i, kv_cap, kv_ring);
            const float pr = sc[i];
            float x = 0.0f, y = 0.0f;
            for (int st = 0; st < V.stages; st++) {
                const unsigned v = vq_index(V.stream[st], V.nbit, V.mask, ks, j, V.nidx_row);
                const uint8_t *cw = V.cb[st] + (size_t) v * (size_t) dim * 2;
                if (dim == 8) {
                    const float2 f = __half22float2(*(const __half2 *) (cw + (size_t) off * 2));
                    x += f.x;
                    y += f.y;
                } else {
                    x += f16_at(cw, (size_t) off);
                    y += f16_at(cw, (size_t) (off + 1));
                }
            }
            acc.x = fmaf(pr, x, acc.x);
            acc.y = fmaf(pr, y, acc.y);
        }
        outp[d] = acc.x;
        outp[d + 1] = acc.y;
    }
    if (threadIdx.x == 0) {
        pm[block] = n ? mx : -INFINITY;
        pl[block] = n ? sum : 0.0f;
    }
}

struct gm4_vqkv {
    int dim, nc, stages, rows, cols, nbit, nidx_row, window;
    uint8_t *cb[VQ_STAGES];
    float *c2[VQ_STAGES];
    uint8_t *stream[VQ_STAGES];
    /* The encoder's scratch, one row's worth: a stage's reconstruction, the
     * running sum of the stages, the residual the next stage assigns, and the
     * indices before they are packed. */
    float *part, *recd, *resid;
    uint16_t *idx;
};

/* Pack one row's indices into the stream at `row`: the payload's layout, a
 * single bit stream little-endian across the same three-byte window the
 * reader uses. One thread, so the byte-level read-modify-write cannot race;
 * a row is nidx_row indices (256 at the KV geometry) and the per-call cost
 * is a few microseconds. */
__global__ void vq_pack_kernel(uint8_t *stream, const uint16_t *idx, int nbit, size_t row,
                               int nidx_row) {
    const size_t base = row * (size_t) nidx_row * (size_t) nbit;
    const unsigned mask = (1u << nbit) - 1u;
    for (int j = 0; j < nidx_row; j++) {
        const size_t bit = base + (size_t) j * (size_t) nbit;
        uint8_t *q = stream + (bit >> 3);
        const unsigned sh = (unsigned) (bit & 7);
        uint32_t w = (uint32_t) q[0] | ((uint32_t) q[1] << 8) | ((uint32_t) q[2] << 16);
        w &= ~(mask << sh);
        w |= ((uint32_t) idx[j] & mask) << sh;
        q[0] = (uint8_t) (w & 0xffu);
        q[1] = (uint8_t) ((w >> 8) & 0xffu);
        q[2] = (uint8_t) ((w >> 16) & 0xffu);
    }
}

/* Open a container as a cache. `cap` defaults to the container's rows and
 * `window` 0 means an appending cache; a positive window makes it a ring of
 * `cap` slots. The container's stream is copied into the first rows and the
 * rest stays zero until something writes them (the runtime encoder, KV-3b),
 * so a calibration capture and a live ring share one shape. */
int gm4_gpu_vqkv_open(gm4_gpu *g, const char *path, int cap, int window, gm4_vqkv **out,
                      char *err, size_t errlen) {
    (void) g;
    if (!path || !out) {
        snprintf(err, errlen, "vqkv_open needs a container path");
        return -1;
    }
    if (cap < 0 || window < 0) {
        snprintf(err, errlen, "vqkv_open needs a non-negative capacity and window");
        return -1;
    }
    VqFile vf;
    if (vq_open_file(path, err, errlen, &vf) != 0) return -1;
    char why[192] = {0};
    if (vf.groups != 1) {
        snprintf(err, errlen, "%s holds %d codebook groups; a KV container holds one per stage",
                 path, vf.groups);
        vq_close_file(&vf);
        return -1;
    }
    if (vf.dim < 1 || vf.dim > 8 || vf.stages > VQ_STAGES) {
        snprintf(err, errlen, "%s: dim %d or %d stages is outside the reader's 1..8 / %d", path,
                 vf.dim, vf.stages, VQ_STAGES);
        vq_close_file(&vf);
        return -1;
    }
    gm4_vqkv *kv = (gm4_vqkv *) calloc(1, sizeof *kv);
    int rc = kv ? 0 : -1;
    if (rc != 0) snprintf(err, errlen, "out of host memory for the VQ cache");
    if (rc == 0) {
        kv->dim = vf.dim;
        kv->nc = vf.nc;
        kv->stages = vf.stages;
        kv->cols = vf.cols;
        kv->rows = cap > 0 ? cap : vf.rows;
        kv->window = window;
        kv->nbit = index_bits(vf.nc);
        kv->nidx_row = vf.cols / vf.dim;
    }
    for (int s = 0; s < vf.stages && rc == 0; s++) {
        const uint8_t *pay = vf.file + CONTAINER_HEADER + (size_t) s * vf.per;
        KvLayout L;
        if (kv_parse_payload(pay, vf.per, why, sizeof why, &L) != 0) {
            snprintf(err, errlen, "%s stage %d: %s", path, s, why);
            rc = -1;
            break;
        }
        if (L.dim != vf.dim || L.nc != vf.nc || L.rows != vf.rows || L.cols != vf.cols) {
            snprintf(err, errlen, "%s stage %d disagrees with the container header", path, s);
            rc = -1;
            break;
        }
        for (int r = 0; r < L.rows; r++) {
            const uint16_t h = (uint16_t) (pay[L.gain_off + (size_t) r * 2] |
                                           ((uint16_t) pay[L.gain_off + (size_t) r * 2 + 1] << 8));
            if (h != 0x3c00u) {
                snprintf(err, errlen, "%s stage %d row %d carries gain 0x%04x, not 1.0; the "
                         "attention reads the stream without a gain", path, s, r, h);
                rc = -1;
                break;
            }
        }
        if (rc != 0) break;
        const size_t ncb = (size_t) vf.nc * (size_t) vf.dim * 2;
        /* The stream needs a guard: the three-byte window reads past the last
         * index, exactly as the payload carries one (vq_qc_bytes.h:14-19). */
        const size_t nst = (size_t) vf.rows * (size_t) kv->nidx_row * (size_t) kv->nbit / 8 + 3;
        cudaError_t e = cudaMalloc((void **) &kv->cb[s], ncb);
        if (e == cudaSuccess)
            e = cudaMemcpy(kv->cb[s], pay + L.cb_off, ncb, cudaMemcpyHostToDevice);
        if (e == cudaSuccess) e = cudaMalloc((void **) &kv->stream[s], nst);
        if (e == cudaSuccess) e = cudaMemset(kv->stream[s], 0, nst);
        if (e == cudaSuccess) {
            const size_t used = (size_t) vf.rows * (size_t) kv->nidx_row * (size_t) kv->nbit / 8;
            e = cudaMemcpy(kv->stream[s], pay + L.stream_off, used, cudaMemcpyHostToDevice);
        }
        if (e != cudaSuccess) {
            fail(err, errlen, "VQ cache upload", e);
            rc = -1;
            break;
        }
        /* c2 exactly as vq_assign_q computes it: a double sum per codeword,
         * rounded once (c/vendor/vq_qc.h:75-77). The encoder needs it; the
         * attention does not. */
        float *c2h = (float *) malloc((size_t) vf.nc * sizeof(float));
        e = c2h ? cudaMalloc((void **) &kv->c2[s], (size_t) vf.nc * sizeof(float)) : cudaErrorMemoryAllocation;
        if (e != cudaSuccess) {
            free(c2h);
            fail(err, errlen, "VQ cache c2 allocation", e);
            rc = -1;
            break;
        }
        for (int c = 0; c < vf.nc; c++) {
            double acc = 0.0;
            for (int d = 0; d < vf.dim; d++) {
                const uint8_t *p = pay + L.cb_off + ((size_t) c * vf.dim + d) * 2;
                const float x = __half2float(__ushort_as_half((unsigned short) (p[0] |
                                                  ((unsigned) p[1] << 8))));
                acc += (double) x * (double) x;
            }
            c2h[c] = (float) acc;
        }
        e = cudaMemcpy(kv->c2[s], c2h, (size_t) vf.nc * sizeof(float), cudaMemcpyHostToDevice);
        free(c2h);
        if (e != cudaSuccess) {
            fail(err, errlen, "VQ cache c2 upload", e);
            rc = -1;
        }
    }
    if (rc == 0) {
        const size_t smem = (size_t) vf.nc * vf.dim * 2 + (size_t) vf.nc * sizeof(float);
        int max_shared = 0;
        cudaError_t e =
            cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        if (e == cudaSuccess)
            e = cudaFuncSetAttribute((const void *) vq_assign_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        if (e != cudaSuccess || (size_t) max_shared < smem) {
            snprintf(err, errlen, "%s: the encoder needs %.1f KiB of shared memory, the device "
                     "offers %.1f KiB", path, smem / 1024.0, max_shared / 1024.0);
            rc = -1;
        }
    }
    if (rc == 0) {
        cudaError_t e = cudaMalloc((void **) &kv->part, (size_t) kv->cols * sizeof(float));
        if (e == cudaSuccess) e = cudaMalloc((void **) &kv->recd, (size_t) kv->cols * sizeof(float));
        if (e == cudaSuccess) e = cudaMalloc((void **) &kv->resid, (size_t) kv->cols * sizeof(float));
        if (e == cudaSuccess) e = cudaMalloc((void **) &kv->idx, (size_t) kv->nidx_row * 2);
        if (e != cudaSuccess) {
            fail(err, errlen, "VQ cache encoder scratch", e);
            rc = -1;
        }
    }
    vq_close_file(&vf);
    if (rc != 0) {
        gm4_gpu_vqkv_free(kv);
        return -1;
    }
    *out = kv;
    return 0;
}

void gm4_gpu_vqkv_free(gm4_vqkv *kv) {
    if (!kv) return;
    for (int s = 0; s < VQ_STAGES; s++) {
        if (kv->cb[s]) cudaFree(kv->cb[s]);
        if (kv->c2[s]) cudaFree(kv->c2[s]);
        if (kv->stream[s]) cudaFree(kv->stream[s]);
    }
    if (kv->part) cudaFree(kv->part);
    if (kv->recd) cudaFree(kv->recd);
    if (kv->resid) cudaFree(kv->resid);
    if (kv->idx) cudaFree(kv->idx);
    free(kv);
}

/* Encode one position's row against the calibration codebooks, on the device,
 * and pack it into the cache at its slot. This is the runtime half of the
 * codec: the reference's assignment (c/vendor/vq_qc.h:70-96, c2 - 2*(v.C),
 * first minimum) then the residual through the stages, the same chain
 * gm4_gpu_vq_encode runs. `xd` is [cols] f32 on the device - the layer's own
 * normed, roped key or value - so the step never round-trips to the host.
 *
 * The toolkit learns a codebook from the whole capture (minutes per layer,
 * measured 2026-11-02), which a generation cannot pay; this encodes against
 * the calibration container's codebook instead, and that is the KV unit's
 * named deviation. */
int gm4_gpu_vqkv_encode_row(gm4_gpu *g, gm4_vqkv *kv, int pos, const float *xd, char *err,
                            size_t errlen) {
    (void) g;
    if (!kv || !xd) {
        snprintf(err, errlen, "vqkv_encode_row needs a cache and a device row");
        return -1;
    }
    if (pos < 0 || (!kv->window && pos >= kv->rows)) {
        snprintf(err, errlen, "vqkv_encode_row position %d outside the %s cache's %d rows", pos,
                 kv->window ? "ring" : "appending", kv->rows);
        return -1;
    }
    const size_t slot = kv->window ? (size_t) (pos % kv->rows) : (size_t) pos;
    /* The warp-per-segment kernel, not a per-row variant: an A/B against a
     * block-per-segment shape on this card measured 17.44 and 17.58 ms per
     * token, the same within drift. The cost is the scan itself - 256
     * segments x 4096 codewords x 16 B is 16.8 MB of codeword reads per row
     * per stage - and the card has one layer's worth of rows, never thirty:
     * the layers are sequential, so no cross-layer batch exists to launch. */
    const size_t smem = (size_t) kv->nc * kv->dim * 2 + (size_t) kv->nc * sizeof(float);
    const unsigned blocks = (unsigned) ((kv->nidx_row + VQ_WARPS - 1) / VQ_WARPS);
    cudaError_t e = cudaMemset(kv->recd, 0, (size_t) kv->cols * sizeof(float));
    if (e != cudaSuccess) return fail(err, errlen, "vq row accumulator clear", e);
    for (int s = 0; s < kv->stages; s++) {
        const float *in = s == 0 ? xd : kv->resid;
        vq_assign_kernel<<<blocks, VQ_THREADS, smem>>>(kv->cb[s], kv->c2[s], kv->nc, kv->dim, in,
                                                       kv->cols, (size_t) kv->nidx_row, kv->idx,
                                                       kv->part);
        e = cudaGetLastError();
        if (e != cudaSuccess) return fail(err, errlen, "vq row assign launch", e);
        vq_pack_kernel<<<1, 1>>>(kv->stream[s], kv->idx, kv->nbit, slot, kv->nidx_row);
        e = cudaGetLastError();
        if (e != cudaSuccess) return fail(err, errlen, "vq row pack launch", e);
        if (s + 1 < kv->stages) {
            const unsigned t = 256;
            const unsigned b = (unsigned) ((kv->cols + t - 1) / t);
            vq_add_kernel<<<b, t>>>(kv->recd, kv->part, (size_t) kv->cols);
            vq_sub_kernel<<<b, t>>>(kv->resid, xd, kv->recd, (size_t) kv->cols);
            e = cudaGetLastError();
            if (e != cudaSuccess) return fail(err, errlen, "vq row residual launch", e);
        }
    }
    return 0;
}

int gm4_gpu_vqkv_encode_host(gm4_gpu *g, gm4_vqkv *kv, const float *x, int nrows, char *err,
                             size_t errlen) {
    if (!kv || !x || nrows <= 0 || nrows > kv->rows) {
        snprintf(err, errlen, "vqkv_encode_host needs %d rows and the cache holds %d", nrows,
                 kv ? kv->rows : 0);
        return -1;
    }
    const size_t n = (size_t) nrows * (size_t) kv->cols;
    float *xd = NULL;
    cudaError_t e = cudaMalloc((void **) &xd, n * sizeof(float));
    if (e == cudaSuccess) e = cudaMemcpy(xd, x, n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        if (xd) cudaFree(xd);
        return fail(err, errlen, "vq host rows upload", e);
    }
    int rc = 0;
    for (int r = 0; r < nrows && rc == 0; r++)
        rc = gm4_gpu_vqkv_encode_row(g, kv, r, xd + (size_t) r * (size_t) kv->cols, err, errlen);
    e = cudaDeviceSynchronize();
    if (rc == 0 && e != cudaSuccess) rc = fail(err, errlen, "vq host rows sync", e);
    cudaFree(xd);
    return rc;
}

int gm4_gpu_vqkv_geometry(const gm4_vqkv *kv, int *rows, int *cols, int *dim, int *nc,
                          int *stages) {
    if (!kv) return -1;
    if (rows) *rows = kv->rows;
    if (cols) *cols = kv->cols;
    if (dim) *dim = kv->dim;
    if (nc) *nc = kv->nc;
    if (stages) *stages = kv->stages;
    return 0;
}

/* The chain's shape: q, the position and the output are already on the
 * device, so nothing here waits. The public entry point below is this plus
 * the two copies a single call needs. */
static int attention_vq_dev(gm4_gpu *g, gm4_vqkv *k, gm4_vqkv *v, const float *qd, int heads,
                            int head_dim, const int32_t *posd, float *outd, char *err,
                            size_t errlen) {
    if (!g || !k || !v || !qd || !outd || heads <= 0 || head_dim <= 0) {
        snprintf(err, errlen, "vq attention needs two caches, a query and a query width");
        return -1;
    }
    if (k->rows != v->rows || k->cols != v->cols || k->dim != v->dim || k->stages != v->stages
        || k->nbit != v->nbit) {
        snprintf(err, errlen, "vq attention: K is %dx%d dim %d/%d stages and V is %dx%d dim %d/%d",
                 k->rows, k->cols, k->dim, k->stages, v->rows, v->cols, v->dim, v->stages);
        return -1;
    }
    if (k->cols % head_dim) {
        snprintf(err, errlen, "vq attention: %d columns do not divide into a %d-wide head",
                 k->cols, head_dim);
        return -1;
    }
    const int kv_heads = k->cols / head_dim;
    if (heads % kv_heads) {
        snprintf(err, errlen, "%d query heads do not divide into %d kv heads", heads, kv_heads);
        return -1;
    }
    if (head_dim % 2 || k->dim % 2) {
        snprintf(err, errlen, "vq attention needs an even head width and segment width");
        return -1;
    }

    /* The grid is sized for the widest window the cache can hold, as
     * attention_ring does, so a captured step's schedule does not depend on
     * the position it replays at. */
    const int bound = k->window > 0 ? k->window : k->rows;
    const int ring = k->window > 0 ? 1 : 0;
    const int splits = attn_splits(heads, bound);
    const int blocks = heads * splits;
    const int chunk = (bound + splits - 1) / splits;
    if (ensure_float(&g->ascores, &g->ascorescap, (size_t) blocks * (size_t) chunk, err, errlen)
        != 0)
        return -1;
    if (ensure_float(&g->apart, &g->apartcap, (size_t) blocks * (size_t) head_dim, err, errlen)
        != 0)
        return -1;
    if (ensure_float(&g->apm, &g->apmcap, (size_t) blocks, err, errlen) != 0) return -1;
    if (ensure_float(&g->apl, &g->aplcap, (size_t) blocks, err, errlen) != 0) return -1;

    VqKvDev K;
    memset(&K, 0, sizeof K);
    for (int s = 0; s < k->stages; s++) {
        K.cb[s] = k->cb[s];
        K.stream[s] = k->stream[s];
    }
    K.stages = k->stages;
    K.dim = k->dim;
    K.nbit = k->nbit;
    K.nidx_row = k->nidx_row;
    K.rows = k->rows;
    K.mask = (1u << k->nbit) - 1u;
    VqKvDev V = K;
    for (int s = 0; s < v->stages; s++) {
        V.cb[s] = v->cb[s];
        V.stream[s] = v->stream[s];
    }

    attn_vq_split_kernel<<<(unsigned) blocks, 256, 32 * sizeof(float)>>>(
        qd, K, V, g->ascores, g->apart, g->apm, g->apl, heads, kv_heads, head_dim, posd, bound,
        splits, chunk, k->rows, ring);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "vq attention split launch", e);
    attn_combine_kernel<<<(unsigned) heads, 256>>>(g->apart, g->apm, g->apl, outd, heads,
                                                   head_dim, splits);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "vq attention combine launch", e);
    return 0;
}

int gm4_gpu_attention_vq(gm4_gpu *g, gm4_vqkv *k, gm4_vqkv *v, const float *q, int heads,
                         int head_dim, int pos, float *out, char *err, size_t errlen) {
    if (!g || !q || !out || pos < 0) {
        snprintf(err, errlen, "vq attention needs a query and a non-negative position");
        return -1;
    }
    const size_t qn = (size_t) heads * (size_t) head_dim;
    if (ensure_float(&g->aq, &g->aqcap, qn, err, errlen) != 0) return -1;
    if (ensure_float(&g->aout, &g->aoutcap, qn, err, errlen) != 0) return -1;
    cudaError_t e = cudaMemcpy(g->aq, q, qn * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "vq attention upload", e);
    if (stage_pos(g, pos, err, errlen) != 0) return -1;
    if (attention_vq_dev(g, k, v, g->aq, heads, head_dim, g->posone, g->aout, err, errlen) != 0)
        return -1;
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "vq attention sync", e);
    e = cudaMemcpy(out, g->aout, qn * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "vq attention download", e);
    return 0;
}

/* One query token against a cache: the last `cap` positions of a ring, every
 * position of an appending cache up to `pos`. The window is the cache's own
 * capacity, which is what the CPU passes as config.sliding_window
 * (forward.rs:430-433), so the two cannot disagree. */
int gm4_gpu_attention_ring(gm4_gpu *g, gm4_kv *kv, const float *q, int heads, int pos, float *out,
                           char *err, size_t errlen) {
    if (!g || !kv || !q || !out || heads <= 0 || pos < 0) {
        snprintf(err, errlen, "ring attention needs a cache, a query and a non-negative position");
        return -1;
    }
    if (heads % kv->kv_heads) {
        snprintf(err, errlen, "%d query heads do not divide into %d kv heads", heads,
                 kv->kv_heads);
        return -1;
    }
    if (!kv->ring && pos >= kv->cap) {
        snprintf(err, errlen, "position %d past the appending cache's %d positions", pos, kv->cap);
        return -1;
    }
    const int positions = pos + 1;
    const int head_dim = kv->head_dim;
    const size_t qn = (size_t) heads * (size_t) head_dim;
    if (ensure_float(&g->aq, &g->aqcap, qn, err, errlen) != 0) return -1;
    if (ensure_float(&g->aout, &g->aoutcap, qn, err, errlen) != 0) return -1;
    cudaError_t e = cudaMemcpy(g->aq, q, qn * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "ring attention upload", e);
    (void) positions;
    if (stage_pos(g, pos, err, errlen) != 0) return -1;
    /* The window is the cache's capacity for a ring and every position for an
     * appending one; the kernel takes the true window from the position. */
    if (attention_dev(g, g->aq, kv->keys, kv->values, heads, kv->kv_heads, head_dim, g->posone,
                      kv->ring ? kv->cap : pos + 1, kv->cap, kv->ring, g->aout, err, errlen) != 0)
        return -1;
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "ring attention sync", e);
    e = cudaMemcpy(out, g->aout, qn * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "ring attention download", e);
    return 0;
}

/* ---- one layer, chained on the device ------------------------------------
 *
 * The CPU layer (forward.rs:335-398) in its own order: the input norm, q/k/v,
 * the per-head norms and rotary, the cache write and the attention, the
 * post-attention norm and residual, the dense feed-forward, the router and its
 * top-k, the routed experts, and the two norms and the scalar that close the
 * layer. Every step below is the launch the gated entry point above performs;
 * what changes is that the intermediates stay in one arena, the router's picks
 * stay in device memory, and the whole layer is one upload at the start and
 * one download at the end instead of fourteen of each.
 */

int gm4_gpu_layer_new(gm4_gpu *g, const gm4_layer_spec *spec, gm4_layer **out, char *err,
                      size_t errlen) {
    if (!g || !spec || !out) {
        snprintf(err, errlen, "layer_new needs a device handle, a spec and an out pointer");
        return -1;
    }
    *out = NULL;
    const int idx[10] = {spec->q,    spec->k,   spec->v,           spec->o,        spec->dense_gate,
                         spec->dense_up, spec->dense_down, spec->router, spec->expert_gate_up,
                         spec->expert_down};
    for (int i = 0; i < 10; i++) {
        /* v is the one tensor a layer may not have (`k_eq_v`), and then it is
         * -1. Everything else must be resident. */
        if (i == 2 && spec->v < 0) continue;
        if (idx[i] < 0 || idx[i] >= g->nt) {
            snprintf(err, errlen, "layer tensor %d (slot %d) is not in the residency", idx[i], i);
            return -1;
        }
    }
    if (spec->heads <= 0 || spec->kv_heads <= 0 || spec->head_dim <= 0 || spec->head_dim % 2
        || spec->heads % spec->kv_heads || spec->hidden <= 0 || spec->dense_ff <= 0
        || spec->moe_ff <= 0 || spec->experts <= 0 || spec->top_k <= 0
        || spec->top_k > spec->experts) {
        snprintf(err, errlen, "layer_new's geometry is not a legal layer");
        return -1;
    }
    if (!spec->input_norm || !spec->post_attn_norm || !spec->pre_ffn_norm || !spec->post_ffn_norm
        || !spec->pre_ffn_norm2 || !spec->post_ffn_norm1 || !spec->post_ffn_norm2 || !spec->q_norm
        || !spec->k_norm || !spec->router_scale || !spec->per_expert_scale || !spec->rope_inv) {
        snprintf(err, errlen, "layer_new is missing one of the layer's vectors");
        return -1;
    }
    const size_t nh = (size_t) spec->hidden;
    const size_t nq = (size_t) spec->heads * (size_t) spec->head_dim;
    const size_t nk = (size_t) spec->kv_heads * (size_t) spec->head_dim;
    const size_t nff = (size_t) spec->dense_ff;
    const size_t nmoe = (size_t) spec->moe_ff;
    const size_t nex = (size_t) spec->top_k;
    const size_t nsmall = 8 * nh + 2 * (size_t) spec->head_dim + (size_t) spec->experts
                          + (size_t) spec->head_dim / 2;
    const size_t narena = 11 * nh + 4 * nq + 4 * nk + 3 * nff + 3 * (size_t) spec->experts
                          + 3 * nex * nmoe + nex * nh;

    gm4_layer *L = (gm4_layer *) calloc(1, sizeof(gm4_layer));
    if (!L) {
        snprintf(err, errlen, "out of host memory for the layer handle");
        return -1;
    }
    L->q = spec->q;
    L->k = spec->k;
    L->v = spec->v;
    L->o = spec->o;
    L->dg_t = spec->dense_gate;
    L->du_t = spec->dense_up;
    L->dd_t = spec->dense_down;
    L->rt = spec->router;
    L->egu = spec->expert_gate_up;
    L->edn = spec->expert_down;
    L->heads = spec->heads;
    L->kv_heads = spec->kv_heads;
    L->head_dim = spec->head_dim;
    L->hidden = spec->hidden;
    L->dense_ff = spec->dense_ff;
    L->moe_ff = spec->moe_ff;
    L->experts = spec->experts;
    L->top_k = spec->top_k;
    L->threads = spec->threads > 0 ? spec->threads : 256;
    L->eps = spec->eps;
    L->scalar = spec->layer_scalar;
    L->kq = spec->kvq_k;
    L->vq = spec->kvq_v;
    if ((L->kq && !L->vq) || (!L->kq && L->vq)) {
        gm4_gpu_layer_free(L);
        snprintf(err, errlen, "layer_new needs both VQ caches or neither");
        return -1;
    }
    L->arena_floats = narena;

    cudaError_t e = cudaMalloc((void **) &L->small, nsmall * sizeof(float));
    if (e == cudaSuccess) e = cudaMalloc((void **) &L->arena, narena * sizeof(float));
    if (e == cudaSuccess) e = cudaMalloc((void **) &L->picked, nex * sizeof(int));
    if (e == cudaSuccess)
        e = cudaMalloc((void **) &L->rl1, nex * 2 * nmoe * sizeof(uint32_t));
    if (e == cudaSuccess) e = cudaMalloc((void **) &L->rl2, nex * nh * sizeof(uint32_t));
    if (e == cudaSuccess) e = cudaMalloc((void **) &L->posd, sizeof(int32_t));
    if (e != cudaSuccess) {
        if (L->small) cudaFree(L->small);
        if (L->arena) cudaFree(L->arena);
        if (L->picked) cudaFree(L->picked);
        if (L->rl1) cudaFree(L->rl1);
        if (L->rl2) cudaFree(L->rl2);
        if (L->posd) cudaFree(L->posd);
        free(L);
        snprintf(err, errlen, "layer cudaMalloc failed: %s", cudaGetErrorString(e));
        return -1;
    }

    /* The small vectors, one block, in this fixed order. The router's scale is
     * folded with 1/sqrt(hidden) here, exactly as the CPU folds it
     * (forward.rs:362-365: `*slot *= scale * root`), so the kernel does one
     * multiply per dimension. */
    float *dst = L->small;
    const float root = 1.0f / sqrtf((float) spec->hidden);
    const float *const norms[7] = {spec->input_norm,     spec->post_attn_norm, spec->pre_ffn_norm,
                                   spec->post_ffn_norm,  spec->pre_ffn_norm2,  spec->post_ffn_norm1,
                                   spec->post_ffn_norm2};
    float **const dst_slots[7] = {&L->in_norm,  &L->post_attn, &L->pre_ffn, &L->post_ffn,
                                  &L->pre_ffn2, &L->post_ffn1, &L->post_ffn2};
    for (int i = 0; i < 7; i++) {
        *dst_slots[i] = dst;
        e = cudaMemcpy(dst, norms[i], nh * sizeof(float), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) {
            gm4_gpu_layer_free(L);
            return fail(err, errlen, "layer norm upload", e);
        }
        dst += nh;
    }
    L->q_norm = dst;
    e = cudaMemcpy(dst, spec->q_norm, (size_t) spec->head_dim * sizeof(float),
                   cudaMemcpyHostToDevice);
    dst += (size_t) spec->head_dim;
    L->k_norm = dst;
    if (e == cudaSuccess)
        e = cudaMemcpy(dst, spec->k_norm, (size_t) spec->head_dim * sizeof(float),
                       cudaMemcpyHostToDevice);
    dst += (size_t) spec->head_dim;
    L->router_scale = dst;
    if (e == cudaSuccess) {
        float *scaled = (float *) malloc(nh * sizeof(float));
        if (!scaled) {
            gm4_gpu_layer_free(L);
            snprintf(err, errlen, "out of host memory for the router scale");
            return -1;
        }
        for (size_t i = 0; i < nh; i++) scaled[i] = spec->router_scale[i] * root;
        e = cudaMemcpy(dst, scaled, nh * sizeof(float), cudaMemcpyHostToDevice);
        free(scaled);
    }
    dst += nh;
    L->pes = dst;
    if (e == cudaSuccess)
        e = cudaMemcpy(dst, spec->per_expert_scale, (size_t) spec->experts * sizeof(float),
                       cudaMemcpyHostToDevice);
    dst += (size_t) spec->experts;
    L->rope_inv = dst;
    if (e == cudaSuccess)
        e = cudaMemcpy(dst, spec->rope_inv, (size_t) spec->head_dim / 2 * sizeof(float),
                       cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        gm4_gpu_layer_free(L);
        return fail(err, errlen, "layer table upload", e);
    }

    /* The arena, cut in a fixed order; the sum above is its exact size. */
    L->xa = arena_cut(L, 0, nh);
    L->hb = arena_cut(L, 1, nh);
    L->qa = arena_cut(L, 2, nq);
    L->qb = arena_cut(L, 3, nq);
    L->ka = arena_cut(L, 4, nk);
    L->kb = arena_cut(L, 5, nk);
    L->va = arena_cut(L, 6, nk);
    L->vb = arena_cut(L, 7, nk);
    L->attn = arena_cut(L, 8, nq);
    L->attn2 = arena_cut(L, 9, nq);
    L->xb = arena_cut(L, 10, nh);
    L->dg = arena_cut(L, 11, nff);
    L->du = arena_cut(L, 12, nff);
    L->dact = arena_cut(L, 13, nff);
    L->dd = arena_cut(L, 14, nh);
    L->dense = arena_cut(L, 15, nh);
    L->r = arena_cut(L, 16, nh);
    L->scores = arena_cut(L, 17, (size_t) spec->experts);
    L->moe = arena_cut(L, 18, nh);
    L->moe2 = arena_cut(L, 19, nh);
    L->comb = arena_cut(L, 20, nh);
    L->comb2 = arena_cut(L, 21, nh);
    L->outd = arena_cut(L, 22, nh);
    L->gu = arena_cut(L, 23, nex * 2 * nmoe);
    L->act = arena_cut(L, 24, nex * nmoe);
    L->d = arena_cut(L, 25, nex * nh);
    L->probs = arena_cut(L, 26, (size_t) spec->experts);
    L->wsel = arena_cut(L, 27, (size_t) spec->experts);
    if (L->used > narena) {
        gm4_gpu_layer_free(L);
        snprintf(err, errlen, "the layer arena is short by %zu floats", L->used - narena);
        return -1;
    }
    /* The section events exist from the start; only the records are opt-in. */
    for (int i = 0; i < 12; i++) {
        e = cudaEventCreate(&L->ev[i]);
        if (e != cudaSuccess) {
            gm4_gpu_layer_free(L);
            return fail(err, errlen, "layer event create", e);
        }
    }
    L->timed = 0;
    *out = L;
    return 0;
}

void gm4_gpu_layer_free(gm4_layer *L) {
    if (!L) return;
    for (int i = 0; i < 12; i++)
        if (L->ev[i]) cudaEventDestroy(L->ev[i]);
    if (L->small) cudaFree(L->small);
    if (L->arena) cudaFree(L->arena);
    if (L->picked) cudaFree(L->picked);
    if (L->rl1) cudaFree(L->rl1);
    if (L->rl2) cudaFree(L->rl2);
    if (L->posd) cudaFree(L->posd);
    free(L);
}

size_t gm4_gpu_layer_bytes(const gm4_layer *L) {
    if (!L) return 0;
    return L->arena_floats * sizeof(float) + 8 * (size_t) L->hidden * sizeof(float) + 2
           * (size_t) L->head_dim * sizeof(float) + (size_t) L->experts * sizeof(float)
           + (size_t) L->head_dim / 2 * sizeof(float) + (size_t) L->top_k * sizeof(int)
           + ((size_t) L->top_k * 2 * (size_t) L->moe_ff + (size_t) L->top_k * (size_t) L->hidden)
                 * sizeof(uint32_t);
}

static int layer_forward_dev(gm4_gpu *g, gm4_layer *L, gm4_kv *kv, const float *xd, int pos,
                             const int32_t *posd, int reserve, float *outd, char *err,
                             size_t errlen) {
    if (!g || !L || !kv || !xd || !outd) {
        snprintf(err, errlen, "layer_forward needs a layer, a cache, an input and an output");
        return -1;
    }
    if (kv->kv_heads != L->kv_heads || kv->head_dim != L->head_dim) {
        snprintf(err, errlen, "the cache is %d kv heads of %d, the layer is %d of %d",
                 kv->kv_heads, kv->head_dim, L->kv_heads, L->head_dim);
        return -1;
    }
    if (pos < 0 || (!kv->ring && pos >= kv->cap)) {
        snprintf(err, errlen, "layer position %d outside the cache's %d positions", pos, kv->cap);
        return -1;
    }
    const int nh = L->hidden;
    const int nk = L->kv_heads * L->head_dim;
    mark(L, 0);
    /* The residual lives at L->xa for the whole chain; a caller that already
     * has the activation on the device pays one 11 KiB copy, not a round trip
     * through the host. */
    cudaError_t e = cudaSuccess;
    if (xd != L->xa)
        e = cudaMemcpyAsync(L->xa, xd, (size_t) nh * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    if (e != cudaSuccess) return fail(err, errlen, "layer x copy", e);

    /* residual = x; h = input_norm(residual); q, k and v from h. */
    if (rmsnorm_dev(L->xa, L->in_norm, L->hb, 1, nh, L->eps, err, errlen) != 0) return -1;
    if (matvec_dev(g, L->q, L->hb, L->qa, L->threads, err, errlen) != 0) return -1;
    if (matvec_dev(g, L->k, L->hb, L->ka, L->threads, err, errlen) != 0) return -1;
    if (L->v >= 0) {
        if (matvec_dev(g, L->v, L->hb, L->va, L->threads, err, errlen) != 0) return -1;
    } else {
        /* `value_states = key_states` before either norm (`:1258-1263`). */
        e = cudaMemcpyAsync(L->va, L->ka, (size_t) nk * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        if (e != cudaSuccess) return fail(err, errlen, "layer k to v copy", e);
    }
    mark(L, 1);

    /* The per-head norms, then rotary on q and k; v_norm has no weight. The
     * angle comes from posd, so a replayed capture rotates by the step's own
     * position. */
    if (rmsnorm_dev(L->qa, L->q_norm, L->qb, L->heads, L->head_dim, L->eps, err, errlen) != 0)
        return -1;
    if (rope_dev_p(L->qb, L->heads, L->head_dim, posd, L->rope_inv, err, errlen) != 0) return -1;
    if (rmsnorm_dev(L->ka, L->k_norm, L->kb, L->kv_heads, L->head_dim, L->eps, err, errlen) != 0)
        return -1;
    if (rope_dev_p(L->kb, L->kv_heads, L->head_dim, posd, L->rope_inv, err, errlen) != 0) return -1;
    if (rmsnorm_dev(L->va, NULL, L->vb, L->kv_heads, L->head_dim, L->eps, err, errlen) != 0)
        return -1;

    /* The cache takes this position first: the query sees its own key. On the
     * VQ path the key and value are encoded against the layer's calibration
     * codebooks and the attention reads the containers; the f16 cache is not
     * touched. */
    if (L->kq) {
        if (gm4_gpu_vqkv_encode_row(g, L->kq, pos, L->kb, err, errlen) != 0) return -1;
        if (gm4_gpu_vqkv_encode_row(g, L->vq, pos, L->vb, err, errlen) != 0) return -1;
        mark(L, 2);
        if (attention_vq_dev(g, L->kq, L->vq, L->qb, L->heads, L->head_dim, posd, L->attn, err,
                             errlen) != 0)
            return -1;
    } else {
        if (kv_put_dev(kv, pos, posd, L->kb, L->vb, err, errlen) != 0) return -1;
        mark(L, 2);
        if (attention_ring_dev(g, kv, L->qb, L->heads, pos, posd, reserve, L->attn, err, errlen) != 0)
            return -1;
    }
    mark(L, 3);
    /* The output projection is the last thing the CPU's attention does
     * (`layer.o.matvec(&out)`, forward.rs:462), so it is part of this step. */
    if (matvec_dev(g, L->o, L->attn, L->attn2, L->threads, err, errlen) != 0) return -1;
    if (rmsnorm_dev(L->attn2, L->post_attn, L->attn, 1, nh, L->eps, err, errlen) != 0) return -1;
    if (residual_dev(L->xa, L->attn, L->xb, (size_t) nh, 1.0f, err, errlen) != 0) return -1;
    mark(L, 4);

    /* The dense feed-forward, which reads the post-attention residual. */
    if (rmsnorm_dev(L->xb, L->pre_ffn, L->hb, 1, nh, L->eps, err, errlen) != 0) return -1;
    if (matvec_dev(g, L->dg_t, L->hb, L->dg, L->threads, err, errlen) != 0) return -1;
    if (matvec_dev(g, L->du_t, L->hb, L->du, L->threads, err, errlen) != 0) return -1;
    if (gelu_mul_dev(L->dg, L->du, L->dact, (size_t) L->dense_ff, err, errlen) != 0) return -1;
    if (matvec_dev(g, L->dd_t, L->dact, L->dd, L->threads, err, errlen) != 0) return -1;
    if (rmsnorm_dev(L->dd, L->post_ffn1, L->dense, 1, nh, L->eps, err, errlen) != 0) return -1;
    mark(L, 5);

    /* The router: the weightless norm, the folded scale, the projection, the
     * selection. The picks and the weights stay on the device from here. */
    if (rmsnorm_dev(L->xb, NULL, L->r, 1, nh, L->eps, err, errlen) != 0) return -1;
    if (mul_dev(L->r, L->router_scale, L->moe2, (size_t) nh, err, errlen) != 0) return -1;
    mark(L, 6);
    if (matvec_dev(g, L->rt, L->moe2, L->scores, L->threads, err, errlen) != 0) return -1;
    mark(L, 7);
    if (router_topk_dev(L->scores, L->pes, L->experts, L->top_k, L->probs, L->picked, L->wsel,
                        err, errlen) != 0)
        return -1;
    mark(L, 8);

    /* The experts: the engine's persistent walk when the geometry is its own
     * (one gate/up launch with the SwiGLU in registers, one down launch, then
     * the reduce), otherwise the per-payload gathers and the block gelu. */
    const int gcount = 2 * L->moe_ff;
    const size_t gn = (size_t) L->top_k * (size_t) gcount;
    if (rmsnorm_dev(L->xb, L->pre_ffn2, L->hb, 1, nh, L->eps, err, errlen) != 0) return -1;
    const int fast = expert_block_dev(L, g, err, errlen);
    if (fast < 0) return -1;
    if (fast) {
        if (moe_reduce_dev(L->d, L->wsel, L->moe, L->top_k, nh, err, errlen) != 0) return -1;
    } else {
        if (rowlist_dev(L->rl1, L->picked, L->top_k, gcount, gcount, err, errlen) != 0) return -1;
        if (rows_dev(g, L->egu, L->rl1, (int) gn, L->hb, 0, 0, L->gu, L->threads, err, errlen) != 0)
            return -1;
        /* One gelu over all k blocks: the same arithmetic as k launches, with
         * the expert's stride instead of the caller's loop. */
        if (gelu_blocks_dev(L->gu, L->act, (int) L->moe_ff, L->top_k, err, errlen) != 0) return -1;
        if (rowlist_dev(L->rl2, L->picked, L->top_k, nh, nh, err, errlen) != 0) return -1;
        /* One gather over all k experts and one weighted reduce: the down rows
         * are disjoint, so the launch is the concatenation of the k blocks. */
        e = cudaMemsetAsync(L->d, 0, (size_t) L->top_k * (size_t) nh * sizeof(float), cudaStreamPerThread);
        if (e != cudaSuccess) return fail(err, errlen, "layer down clear", e);
        if (rows_dev(g, L->edn, L->rl2, L->top_k * nh, L->act, nh, (int) L->moe_ff, L->d, L->threads,
                     err, errlen) != 0)
            return -1;
        if (moe_reduce_dev(L->d, L->wsel, L->moe, L->top_k, nh, err, errlen) != 0) return -1;
    }
    mark(L, 9);

    /* The close: moe norm, dense + moe, the pair norm, the residual, scalar. */
    if (rmsnorm_dev(L->moe, L->post_ffn2, L->moe2, 1, nh, L->eps, err, errlen) != 0) return -1;
    if (residual_dev(L->dense, L->moe2, L->comb, (size_t) nh, 1.0f, err, errlen) != 0) return -1;
    if (rmsnorm_dev(L->comb, L->post_ffn, L->comb2, 1, nh, L->eps, err, errlen) != 0) return -1;
    if (residual_dev(L->xb, L->comb2, L->outd, (size_t) nh, L->scalar, err, errlen) != 0)
        return -1;
    mark(L, 10);

    /* The output stays where the chain left it unless the caller asked for a
     * copy: in a model step the next layer's activation is another device
     * buffer, and the host is not involved at all. */
    if (outd != L->outd) {
        e = cudaMemcpyAsync(outd, L->outd, (size_t) nh * sizeof(float), cudaMemcpyDeviceToDevice,
                            cudaStreamPerThread);
        if (e != cudaSuccess) return fail(err, errlen, "layer output copy", e);
    }
    mark(L, 11);
    return 0;
}

int gm4_gpu_layer_forward(gm4_gpu *g, gm4_layer *L, gm4_kv *kv, const float *x, int pos,
                          float *out, char *err, size_t errlen) {
    if (!g || !L || !kv || !x || !out) {
        snprintf(err, errlen, "layer_forward needs a layer, a cache, an input and an output");
        return -1;
    }
    const int nh = L->hidden;
    cudaError_t e = cudaMemcpy(L->xa, x, (size_t) nh * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "layer x upload", e);
    e = cudaMemcpy(L->posd, &pos, sizeof(int32_t), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "layer position upload", e);
    /* A single layer call reads exactly the window its position implies, so the
     * launch is sized for pos + 1 and the reservation is not needed. */
    if (layer_forward_dev(g, L, kv, L->xa, pos, L->posd, pos + 1, L->outd, err, errlen) != 0)
        return -1;
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "layer sync", e);
    e = cudaMemcpy(out, L->outd, (size_t) nh * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "layer download", e);
    return 0;
}

/* Diagnostic: copy one intermediate of the last forward back as f32. Not on
 * any inference path; it is how the layer gate locates a wiring error instead
 * of guessing at one. The arena reuses buffers, so what a number holds is the
 * state the forward leaves it in; `n` is the caller's own width for the value
 * and must fit the buffer. */
int gm4_gpu_layer_probe(gm4_layer *L, int which, float *out, int n, char *err, size_t errlen) {
    if (!L || !out || n <= 0) {
        snprintf(err, errlen, "layer_probe needs a layer and an output");
        return -1;
    }
    const float *src = NULL;
    /* which: 0 xa 1 hb 2 qa 3 qb 4 ka 5 kb 6 va 7 vb 8 attn 9 attn2 10 xb
     * 11 dg 12 du 13 dact 14 dd 15 dense 16 r 17 scores 18 moe 19 moe2
     * 20 comb 21 comb2 22 outd 23 gu 24 act 25 d */
    switch (which) {
    case 0: src = L->xa; break;
    case 1: src = L->hb; break;
    case 2: src = L->qa; break;
    case 3: src = L->qb; break;
    case 4: src = L->ka; break;
    case 5: src = L->kb; break;
    case 6: src = L->va; break;
    case 7: src = L->vb; break;
    case 8: src = L->attn; break;
    case 9: src = L->attn2; break;
    case 10: src = L->xb; break;
    case 11: src = L->dg; break;
    case 12: src = L->du; break;
    case 13: src = L->dact; break;
    case 14: src = L->dd; break;
    case 15: src = L->dense; break;
    case 16: src = L->r; break;
    case 17: src = L->scores; break;
    case 18: src = L->moe; break;
    case 19: src = L->moe2; break;
    case 20: src = L->comb; break;
    case 21: src = L->comb2; break;
    case 22: src = L->outd; break;
    case 23: src = L->gu; break;
    case 24: src = L->act; break;
    case 25: src = L->d; break;
    default:
        snprintf(err, errlen, "layer_probe has no intermediate %d", which);
        return -1;
    }
    if ((size_t) n > L->w[which]) {
        snprintf(err, errlen, "intermediate %d holds %zu values, the caller asked for %d", which,
                 L->w[which], n);
        return -1;
    }
    cudaError_t e = cudaMemcpy(out, src, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "layer probe", e);
    return 0;
}

/* Turn the section records on or off. Timing is diagnostic: a run with it on
 * costs nine stream-ordered event records and one sync the forward already
 * does, so the numbers are the chain's own timeline rather than a guess split
 * of the per-layer total. */
int gm4_gpu_layer_timing(gm4_layer *L, int on) {
    if (!L) return -1;
    L->timed = on != 0;
    return 0;
}

/* Copy the last forward's section spans in milliseconds and their names. The
 * caller passes room for GM4_SECTIONS (9) entries; a shorter `n` gets a
 * prefix. Returns the count written, or -1 when the layer has not been timed.
 * The forward's own sync and download are the last section, so a caller that
 * adds this call to a loop is not timing its own wait twice. */
int gm4_gpu_layer_sections(gm4_layer *L, double *ms, const char **names, int n) {
    if (!L || !ms || n <= 0) return -1;
    if (!L->timed) return -1;
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return -1;
    const int count = n < GM4_SECTIONS ? n : GM4_SECTIONS;
    for (int i = 0; i < count; i++) {
        float span = 0.0f;
        e = cudaEventElapsedTime(&span, L->ev[i], L->ev[i + 1]);
        if (e != cudaSuccess) return -1;
        ms[i] = (double) span;
        if (names) names[i] = gm4_section_names[i];
    }
    return count;
}

/* ---- one token, whole model ----------------------------------------------
 *
 * A layer chain is one layer; this is all of them plus the two ends the CPU's
 * step runs around the stack (`forward.rs:301-336`): the tied table read as
 * one row and scaled by embed_scale, the final norm, the tied head over the
 * whole vocabulary, and the softcap. The activation never leaves the device
 * between those ends: two buffers alternate, so each layer's output is the
 * next one's input, and a token synchronises once, at the logits download.
 *
 * The buffers, the norm table and the logits vector belong to the model
 * handle; the layers and their caches belong to the caller, exactly as
 * gm4_gpu_layer_forward's do. */
/* Position buckets and graph slots. The width is the reference engine's
 * (ds4_ds41_graph.inc:48, DGRAPH_BUCKET 1024): inside a bucket every grid the
 * step launches is fixed, so one capture replays every step in it, and
 * crossing the bucket re-captures. Eight slots hold eight buckets, which at
 * this context bound is more than a session can reach. GM4_BUCKET overrides
 * the width for a diagnosis (a width of 1 captures each position with its own
 * grids, which separates the replay's dispatch cost from the bucket's). */
#define GM4_BUCKET_DEFAULT 1024
static int gm4_bucket_width(void) {
    const char *v = getenv("GM4_BUCKET");
    const int w = v ? atoi(v) : 0;
    return w > 0 ? w : GM4_BUCKET_DEFAULT;
}
#define GM4_MAX_BUCKETS 8

struct gm4_model {
    gm4_layer **layers;
    gm4_kv **kv;
    int n;
    int embed, head, hidden, vocab, threads;
    float eps, embed_scale, softcap;
    float *act[2];
    float *fnorm;
    float *logits;
    /* The only host writes a step needs: the token and the position. The
     * kernels read them from the device, so a captured step can be replayed at
     * any token and any position inside its bucket (the "device position"
     * convention, ds4_ds41_graph.inc:9-12). `staging` is pinned, so the 8-byte
     * upload is a real asynchronous copy and not a staged one. */
    int32_t *slots;   /* device: [0] token, [1] position */
    int32_t *staging; /* pinned host mirror */
    cudaGraphExec_t graphs[GM4_MAX_BUCKETS];
    int32_t glo[GM4_MAX_BUCKETS], gcap[GM4_MAX_BUCKETS];
    int ngraphs;
    int ctx;        /* the appending caches' bound: the highest position a step may take */
    int warm;       /* one direct step has run, the engine's precondition for a capture */
    int graph_off;  /* a capture failed: walk direct from here on */
    int captures;
    /* The last step, split the way the reference engine accounts a decode
     * step (/data/ds4-dfm-rs/ds4_ds41_graph.inc:106-113: submit, launch,
     * wait): `submit` is the host time from entry to the last launch's
     * return, `wait` is everything after it, the sync and the readback. One
     * wall number hides which half the next change has to move. */
    double t_submit, t_wait;
    /* The block path's scratch, grown on first use to the requested block
     * length (pre_scratch). A step never touches it. */
    struct gm4_pre *pre;
};

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

static void pre_free(struct gm4_pre *P);

void gm4_gpu_model_free(gm4_model *m) {
    if (!m) return;
    for (int i = 0; i < m->ngraphs; i++)
        if (m->graphs[i]) cudaGraphExecDestroy(m->graphs[i]);
    if (m->slots) cudaFree(m->slots);
    if (m->staging) cudaFreeHost(m->staging);
    if (m->act[0]) cudaFree(m->act[0]);
    if (m->act[1]) cudaFree(m->act[1]);
    if (m->fnorm) cudaFree(m->fnorm);
    if (m->logits) cudaFree(m->logits);
    if (m->pre) pre_free(m->pre);
    free(m->layers);
    free(m->kv);
    free(m);
}

int gm4_gpu_model_new(gm4_gpu *g, const gm4_model_spec *spec, gm4_model **out, char *err,
                      size_t errlen) {
    if (!g || !spec || !out) {
        snprintf(err, errlen, "model_new needs a device handle, a spec and an out pointer");
        return -1;
    }
    *out = NULL;
    if (!spec->layers || !spec->kv || spec->n <= 0) {
        snprintf(err, errlen, "model_new needs the layers and their caches");
        return -1;
    }
    if (spec->hidden <= 0 || spec->vocab <= 0 || spec->softcap <= 0.0f || !spec->final_norm) {
        snprintf(err, errlen, "model_new needs a hidden width, a vocabulary, a positive softcap "
                              "and the final norm's weight");
        return -1;
    }
    for (int i = 0; i < spec->n; i++) {
        if (!spec->layers[i] || !spec->kv[i]) {
            snprintf(err, errlen, "model layer %d is missing its chain or its cache", i);
            return -1;
        }
        if (spec->layers[i]->hidden != spec->hidden) {
            snprintf(err, errlen, "model layer %d is %d wide, the model is %d", i,
                     spec->layers[i]->hidden, spec->hidden);
            return -1;
        }
        if (spec->kv[i]->kv_heads != spec->layers[i]->kv_heads
            || spec->kv[i]->head_dim != spec->layers[i]->head_dim) {
            snprintf(err, errlen,
                     "model layer %d's cache is %d kv heads of %d, the layer is %d of %d", i,
                     spec->kv[i]->kv_heads, spec->kv[i]->head_dim, spec->layers[i]->kv_heads,
                     spec->layers[i]->head_dim);
            return -1;
        }
    }
    if (spec->embed < 0 || spec->embed >= g->nt || spec->head < 0 || spec->head >= g->nt) {
        snprintf(err, errlen, "model_new's tables are not in the residency");
        return -1;
    }
    const Layout &E = g->ts[spec->embed].pays[0].l;
    const Layout &H = g->ts[spec->head].pays[0].l;
    if (E.rows != spec->vocab || E.cols != spec->hidden) {
        snprintf(err, errlen, "the embedding table is %dx%d, the model is %dx%d", E.rows, E.cols,
                 spec->vocab, spec->hidden);
        return -1;
    }
    if (H.rows != spec->vocab || H.cols != spec->hidden) {
        snprintf(err, errlen, "the head is %dx%d, the model is %dx%d", H.rows, H.cols, spec->vocab,
                 spec->hidden);
        return -1;
    }

    gm4_model *m = (gm4_model *) calloc(1, sizeof(gm4_model));
    if (!m) {
        snprintf(err, errlen, "out of host memory for the model handle");
        return -1;
    }
    m->n = spec->n;
    m->layers = (gm4_layer **) calloc((size_t) spec->n, sizeof(gm4_layer *));
    m->kv = (gm4_kv **) calloc((size_t) spec->n, sizeof(gm4_kv *));
    if (!m->layers || !m->kv) {
        gm4_gpu_model_free(m);
        snprintf(err, errlen, "out of host memory for the model's layer list");
        return -1;
    }
    for (int i = 0; i < spec->n; i++) {
        m->layers[i] = spec->layers[i];
        m->kv[i] = spec->kv[i];
    }
    m->embed = spec->embed;
    m->head = spec->head;
    m->hidden = spec->hidden;
    m->vocab = spec->vocab;
    m->threads = spec->threads > 0 ? spec->threads : 256;
    m->eps = spec->eps;
    m->embed_scale = spec->embed_scale;
    m->softcap = spec->softcap;

    cudaError_t e = cudaMalloc((void **) &m->act[0], (size_t) spec->hidden * sizeof(float));
    if (e == cudaSuccess)
        e = cudaMalloc((void **) &m->act[1], (size_t) spec->hidden * sizeof(float));
    if (e == cudaSuccess)
        e = cudaMalloc((void **) &m->fnorm, (size_t) spec->hidden * sizeof(float));
    if (e == cudaSuccess)
        e = cudaMalloc((void **) &m->logits, (size_t) spec->vocab * sizeof(float));
    if (e == cudaSuccess)
        e = cudaMalloc((void **) &m->slots, 2 * sizeof(int32_t));
    if (e == cudaSuccess)
        e = cudaMallocHost((void **) &m->staging, 2 * sizeof(int32_t));
    if (e == cudaSuccess)
        e = cudaMemcpy(m->fnorm, spec->final_norm, (size_t) spec->hidden * sizeof(float),
                       cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        gm4_gpu_model_free(m);
        return fail(err, errlen, "model allocation", e);
    }
    /* The bound a step may reach: the appending caches' capacity, since they
     * are the ones that run out (a ring layer wraps instead). */
    m->ctx = INT32_MAX;
    for (int i = 0; i < spec->n; i++)
        if (!spec->kv[i]->ring && spec->kv[i]->cap < m->ctx) m->ctx = spec->kv[i]->cap;
    if (m->ctx == INT32_MAX) m->ctx = gm4_bucket_width();
    /* The graph path is built and gated, but it is off unless GM4_GRAPH asks
     * for it: measured in one run on this card, a replayed step is 2 ms slower
     * than the direct walk of the same kernels (17.93 vs 15.92 ms at position
     * 26, host submit 0.02 ms against 4.74), because the 1350 launches submit
     * at 3.6 us each while the device takes 11 us per kernel, so submission is
     * already hidden under device work and the replay only adds its per-node
     * dispatch. The reference engine is in the other regime: its step is larger
     * than its submission cost, so the graph pays there
     * (/data/ds4-dfm-rs/ds4_ds41_graph.inc:106-113). */
    m->graph_off = getenv("GM4_GRAPH") ? 0 : 1;
    *out = m;
    return 0;
}

/* The model's whole device footprint: the chains, the caches and the model's
 * own buffers, so a caller's VRAM account does not have to add them up. */
size_t gm4_gpu_model_bytes(const gm4_model *m) {
    if (!m) return 0;
    size_t total = (2 * (size_t) m->hidden + (size_t) m->hidden + (size_t) m->vocab)
                   * sizeof(float);
    for (int i = 0; i < m->n; i++) {
        total += gm4_gpu_layer_bytes(m->layers[i]);
        total += gm4_gpu_kv_bytes(m->kv[i]);
    }
    return total;
}

/* One step's kernels, submitted in order and not synchronised: the two ends of
 * `forward.rs:316-334` - the scaled embedding row, the 30 layers, the final
 * norm, the tied head, the softcap. `cap` is the widest window this step's
 * launch grids must cover, which is the direct step's own position and the
 * capture's bucket bound. Everything it reads about the step itself (the
 * token, the position) is already in the device slots. */
static int model_body(gm4_gpu *g, gm4_model *m, int cap, char *err, size_t errlen) {
    if (embed_row_dev(g, m->embed, m->slots, 1, m->embed_scale, m->act[0], err, errlen) != 0)
        return -1;
    int cur = 0;
    for (int i = 0; i < m->n; i++) {
        if (layer_forward_dev(g, m->layers[i], m->kv[i], m->act[cur], (int) m->staging[1],
                              m->slots + 1, cap, m->act[cur ^ 1], err, errlen) != 0)
            return -1;
        cur ^= 1;
    }
    /* In place: rmsnorm_kernel reads its own position before writing it, and
     * no thread reads another thread's position. */
    if (rmsnorm_dev(m->act[cur], m->fnorm, m->act[cur], 1, m->hidden, m->eps, err, errlen) != 0)
        return -1;
    if (matvec_dev(g, m->head, m->act[cur], m->logits, m->threads, err, errlen) != 0) return -1;
    const unsigned threads = 256;
    softcap_kernel<<<(unsigned) ((m->vocab + (int) threads - 1) / (int) threads), threads>>>(
        m->logits, (size_t) m->vocab, m->softcap);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "model softcap launch", e);
    return 0;
}

/* Grow every buffer a step can touch for a window bound of `cap`, before any
 * capture: an allocation under capture is an unsafe call and voids it, which
 * is the engine's own finding (/data/ds4-dfm-rs/ds4_ds41_graph.inc:205-209:
 * "Grow every scratch the body can need BEFORE st->graph=1"). The attention
 * buffers are the only ones a step allocates; matvec, the gathers and the
 * normalisations all write into the caller's buffers. */
static int model_reserve(gm4_gpu *g, gm4_model *m, int cap, char *err, size_t errlen) {
    for (int i = 0; i < m->n; i++) {
        const gm4_layer *L = m->layers[i];
        const gm4_kv *kv = m->kv[i];
        const int window = kv->ring && kv->cap < cap ? kv->cap : cap;
        const int splits = attn_splits(L->heads, window);
        const int blocks = L->heads * splits;
        const int chunk = (window + splits - 1) / splits;
        if (ensure_float(&g->ascores, &g->ascorescap, (size_t) blocks * (size_t) chunk, err,
                         errlen) != 0)
            return -1;
        if (ensure_float(&g->apart, &g->apartcap, (size_t) blocks * (size_t) L->head_dim, err,
                         errlen) != 0)
            return -1;
        if (ensure_float(&g->apm, &g->apmcap, (size_t) blocks, err, errlen) != 0) return -1;
        if (ensure_float(&g->apl, &g->aplcap, (size_t) blocks, err, errlen) != 0) return -1;
    }
    return 0;
}

/* One step through this bucket's graph, capturing it on first use. Returns 1
 * when a graph ran, 0 when the caller must walk direct (no graph allowed, the
 * bucket table is full, or a capture failed), -1 on a real error. */
static int model_graph(gm4_gpu *g, gm4_model *m, int bucket, int cap, char *err, size_t errlen) {
    for (int i = 0; i < m->ngraphs; i++) {
        if (m->glo[i] != bucket) continue;
        cudaError_t e = cudaGraphLaunch(m->graphs[i], cudaStreamPerThread);
        if (e != cudaSuccess) return fail(err, errlen, "model graph launch", e);
        return 1;
    }
    if (m->ngraphs >= GM4_MAX_BUCKETS) return 0;
    if (model_reserve(g, m, cap, err, errlen) != 0) return -1;
    cudaError_t e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) {
        (void) cudaGetLastError();
        m->graph_off = 1;
        return 0;
    }
    const int built = model_body(g, m, cap, err, errlen) == 0;
    cudaGraph_t graph = NULL;
    e = cudaStreamEndCapture(cudaStreamPerThread, &graph);
    if (!built || e != cudaSuccess || !graph) {
        if (graph) cudaGraphDestroy(graph);
        (void) cudaGetLastError();
        m->graph_off = 1;   /* fail closed: this step and the later ones walk direct */
        return 0;
    }
    cudaGraphExec_t exec = NULL;
    e = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    if (e != cudaSuccess) {
        (void) cudaGetLastError();
        m->graph_off = 1;
        return 0;
    }
    const int slot = m->ngraphs++;
    m->graphs[slot] = exec;
    m->glo[slot] = bucket;
    m->gcap[slot] = cap;
    m->captures++;
    e = cudaGraphLaunch(exec, cudaStreamPerThread);
    if (e != cudaSuccess) return fail(err, errlen, "model graph first launch", e);
    return 1;
}

/* ---- the block of tokens, driven ----------------------------------------
 *
 * One layer over a block of n tokens at pos0..pos0+n-1. Everything below is
 * the single-token chain's order (layer_forward_dev) with the block's kernels
 * substituted: the dense projections become one tiled gemm each (a transposed
 * activation block in, the container's rows out), attention becomes one
 * launch over (head, token), and the routed experts become the work-item
 * passes above. Nothing leaves the device between the first layer and the
 * last but the router's picks, which are grouped on the host into the item
 * list - the reference's own split (it builds vqp_item on the host from the
 * permutation and the offsets, ds41_vq_prefill_fused.cuh:214-225).
 *
 * Buffer sizes are the maximum over the 30 layers, so one scratch serves the
 * whole chain. The scratch is a separate arena from the layer's, because a
 * prefill walks the layers one at a time and a per-layer copy would be 30
 * times the memory for the same bytes.
 */
struct gm4_pre {
    int cap;    /* tokens the scratch holds */
    float *arena;
    size_t floats;
    float *xs[2], *hb, *q, *k, *v, *att, *o, *nrm;
    float *dg, *du, *dact, *dd, *dense;
    float *r, *rs, *scores, *probs, *wsel;
    float *xs32, *gu, *act, *ddn;
    float *moe, *moe2, *comb, *comb2;
    float *xT;
    int *picked;
    int32_t *items, *perm, *dst;
    int32_t *tokd; /* the block's tokens, so the embedding launch reads them */
    size_t listcap; /* pairs the MoE lists hold */
};

static float *pre_cut(gm4_pre *P, size_t n) {
    float *p = P->arena + P->floats;
    P->floats += n;
    return p;
}

/* The widest geometry over the chain: q and its attention output are the full
 * layer's heads * head_dim, the cache rows the full layer's kv heads *
 * head_dim, and the feed-forward widths split by layer type. */
static void model_widths(const gm4_model *m, int *nh, int *nq, int *nk, int *ndff, int *moeff,
                         int *nexp, int *topk) {
    *nh = m->hidden;
    *nq = 0;
    *nk = 0;
    *ndff = 0;
    *moeff = 0;
    *nexp = 0;
    *topk = 0;
    for (int i = 0; i < m->n; i++) {
        const gm4_layer *L = m->layers[i];
        if (L->heads * L->head_dim > *nq) *nq = L->heads * L->head_dim;
        if (L->kv_heads * L->head_dim > *nk) *nk = L->kv_heads * L->head_dim;
        if (L->dense_ff > *ndff) *ndff = L->dense_ff;
        if (L->moe_ff > *moeff) *moeff = L->moe_ff;
        if (L->experts > *nexp) *nexp = L->experts;
        if (L->top_k > *topk) *topk = L->top_k;
    }
}

static void pre_free(gm4_pre *P) {
    if (!P) return;
    if (P->arena) cudaFree(P->arena);
    if (P->items) cudaFree(P->items);
    if (P->perm) cudaFree(P->perm);
    if (P->dst) cudaFree(P->dst);
    if (P->picked) cudaFree(P->picked);
    if (P->tokd) cudaFree(P->tokd);
    free(P);
}

/* Cut the scratch for n tokens. Allocated once per chunk length and reused by
 * every shorter block, so a session's VRAM does not grow with the prompt. */
static int pre_scratch(gm4_model *m, int n, char *err, size_t errlen) {
    gm4_pre *P = m->pre;
    if (P && P->cap >= n) return 0;
    int nh, nq, nk, ndff, moeff, nexp, topk;
    model_widths(m, &nh, &nq, &nk, &ndff, &moeff, &nexp, &topk);
    if (!nh || !nq || !nk || !ndff || !moeff || !nexp || !topk) {
        snprintf(err, errlen, "the chain has no geometry to size a prefill scratch from");
        return -1;
    }
    /* Per token: the two activation blocks, the normed copy, q and its
     * attention output, the cache rows, the output projection, the dense
     * feed-forward's three widths, its projection, the router's two vectors
     * and its scores, the mixture's two, and the close's four. The MoE
     * lists are per pair (token, slot), not per token. */
    const size_t per_tok = (size_t) 2 * nh + nh + (size_t) nq + 2 * (size_t) nk + (size_t) nq
                         + nh + nh + 3 * (size_t) ndff + (size_t) nh + nh + nh + nh
                         + 2 * (size_t) nexp + (size_t) topk + 4 * nh + 3 * nh;
    const size_t pairs = (size_t) n * (size_t) topk;
    const size_t per_pair = (size_t) nh + 2 * (size_t) moeff + (size_t) moeff + (size_t) nh;
    const size_t floats = (size_t) n * per_tok + pairs * per_pair
                        + (size_t) n * (size_t) (nq > nh ? (nq > ndff ? nq : ndff) : (nh > ndff ? nh : ndff));
    gm4_pre *next = (gm4_pre *) calloc(1, sizeof *next);
    if (!next) {
        snprintf(err, errlen, "prefill scratch handle allocation failed");
        return -1;
    }
    next->cap = n;
    cudaError_t e = cudaMalloc((void **) &next->arena, floats * sizeof(float));
    if (e != cudaSuccess) {
        snprintf(err, errlen, "prefill scratch cudaMalloc of %.1f MiB failed: %s",
                 (double) floats * sizeof(float) / (1024.0 * 1024.0), cudaGetErrorString(e));
        free(next);
        return -1;
    }
    next->floats = 0;
    next->xs[0] = pre_cut(next, (size_t) n * nh);
    next->xs[1] = pre_cut(next, (size_t) n * nh);
    next->hb = pre_cut(next, (size_t) n * nh);
    next->q = pre_cut(next, (size_t) n * nq);
    next->k = pre_cut(next, (size_t) n * nk);
    next->v = pre_cut(next, (size_t) n * nk);
    next->att = pre_cut(next, (size_t) n * nq);
    next->o = pre_cut(next, (size_t) n * nh);
    next->nrm = pre_cut(next, (size_t) n * nh);
    next->dg = pre_cut(next, (size_t) n * ndff);
    next->du = pre_cut(next, (size_t) n * ndff);
    next->dact = pre_cut(next, (size_t) n * ndff);
    next->dd = pre_cut(next, (size_t) n * nh);
    next->dense = pre_cut(next, (size_t) n * nh);
    next->r = pre_cut(next, (size_t) n * nh);
    next->rs = pre_cut(next, (size_t) n * nh);
    next->scores = pre_cut(next, (size_t) n * nexp);
    next->probs = pre_cut(next, (size_t) n * nexp);
    next->wsel = pre_cut(next, (size_t) n * topk);
    next->moe = pre_cut(next, (size_t) n * nh);
    next->moe2 = pre_cut(next, (size_t) n * nh);
    next->comb = pre_cut(next, (size_t) n * nh);
    next->comb2 = pre_cut(next, (size_t) n * nh);
    next->xs32 = pre_cut(next, pairs * (size_t) nh);
    next->gu = pre_cut(next, pairs * 2 * (size_t) moeff);
    next->act = pre_cut(next, pairs * (size_t) moeff);
    next->ddn = pre_cut(next, pairs * (size_t) nh);
    next->xT = pre_cut(next, (size_t) n * (size_t) (nq > nh ? (nq > ndff ? nq : ndff) : (nh > ndff ? nh : ndff)));
    if (next->floats > floats) {
        snprintf(err, errlen, "prefill scratch cuts %zu floats past its %zu", next->floats, floats);
        cudaFree(next->arena);
        free(next);
        return -1;
    }
    if (cudaMalloc((void **) &next->picked, pairs * sizeof(int)) != cudaSuccess
        || cudaMalloc((void **) &next->items, pairs * 3 * sizeof(int32_t)) != cudaSuccess
        || cudaMalloc((void **) &next->perm, pairs * sizeof(int32_t)) != cudaSuccess
        || cudaMalloc((void **) &next->dst, pairs * sizeof(int32_t)) != cudaSuccess
        || cudaMalloc((void **) &next->tokd, (size_t) n * sizeof(int32_t)) != cudaSuccess) {
        (void) cudaGetLastError();
        snprintf(err, errlen, "prefill list allocation of %zu pairs failed", pairs);
        pre_free(next);
        return -1;
    }
    next->listcap = pairs;
    /* A larger block replaces the old scratch. A growth happens between
     * prefills, never inside one, and the sync keeps a still-queued launch
     * from reading a freed arena. */
    if (P) {
        (void) cudaDeviceSynchronize();
        pre_free(P);
    }
    m->pre = next;
    return 0;
}

/* The tile and codebook placement the block path asks the tiled kernel for.
 * Diagnostic, like gm4_gpu_span: the gate sweeps the six the tools measured
 * (PREFILL_TILES) and the default is the one the gate found best there. Two
 * globals rather than one packed value because they are independently read. */
static int g_pf_bn = 128, g_pf_cb = 0;

/* 0 or 1 = the block path's token tile and codebook placement; returns the
 * previous tile. */
int gm4_gpu_prefill_cfg(int bn, int cb) {
    const int prev = g_pf_bn;
    if (bn > 0) g_pf_bn = bn;
    if (cb >= 0) g_pf_cb = cb;
    return prev;
}

/* x [rows][cols] -> xT [cols][rows], the layout the tiled kernel's B-tile
 * reads. */
static int transpose_dev(const float *xd, float *xtd, int rows, int cols, char *err,
                         size_t errlen) {
    const dim3 grid((unsigned) ((cols + 31) / 32), (unsigned) ((rows + 31) / 32));
    const dim3 block(32, 32);
    transpose_kernel<<<grid, block>>>(xd, xtd, rows, cols);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "block transpose launch", e);
    return 0;
}

/* One projection for a block of tokens, every payload over the same
 * transposed activation block. The output is the container's full row layout
 * y[token][row] and is cleared first: the tiled kernel accumulates, which is
 * how the container's two stages sum there, in the same order the decode
 * path's two launches make them. */
static int prefill_dev(gm4_gpu *g, int tensor, const float *xTd, int ntok, float *yd, char *err,
                       size_t errlen) {
    if (tensor < 0 || tensor >= g->nt) {
        snprintf(err, errlen, "tensor %d is not in the residency", tensor);
        return -1;
    }
    const Tensor *t = &g->ts[tensor];
    const Layout &L = t->pays[0].l;
    /* Diagnostic: force the f32-accumulating prefill (gemm_prefill_kernel, no
     * f16 activation tile) so a gate can separate the tile's precision from a
     * wiring error. The tile is the tools' kernel and rounds its activation
     * tile to f16 (:848-856), which the tools' own prefill gate accounts for by
     * comparing against a reference over f16-rounded activations. */
    static const int legacy = getenv("GM4_PF_LEGACY") ? 1 : 0;
    const int tiled = !legacy && pre_ok(L, 64) ? 1 : 0;
    if (tiled) {
        char why[256];
        if (!gm4_gpu_prefill_ready(g, tensor, g_pf_bn, g_pf_cb, why, sizeof why)) {
            snprintf(err, errlen, "block projection: %s", why);
            return -1;
        }
    }
    cudaError_t e = cudaMemsetAsync(yd, 0, (size_t) L.rows * (size_t) ntok * sizeof(float),
                                    cudaStreamPerThread);
    if (e != cudaSuccess) return fail(err, errlen, "block projection clear", e);
    for (int p = 0; p < t->npay; p++) {
        const Payload &one = t->pays[p];
        if (one.l.rows != L.rows || one.l.cols != L.cols) {
            snprintf(err, errlen, "payload %d disagrees with payload 0", p);
            return -1;
        }
        Layout slice = one.l;
        slice.row0 = 0;
        slice.nrows = L.rows;
        const int rc = tiled ? pre_launch_cfg(one.dev, slice, xTd, ntok, yd, g_pf_bn, g_pf_cb)
                             : pre_legacy(one.dev, slice, xTd, ntok, yd);
        if (rc != 0) {
            snprintf(err, errlen, "block projection launch failed for payload %d", p);
            return -1;
        }
        e = cudaGetLastError();
        if (e != cudaSuccess) return fail(err, errlen, "block projection launch", e);
    }
    return 0;
}

/* Rotary for the block, every token at its own position. */
static int rope_rows_dev(float *xd, int n, int heads, int head_dim, int pos0, const float *invd,
                         char *err, size_t errlen) {
    const dim3 grid((unsigned) heads, (unsigned) n);
    rope_rows_kernel<<<grid, (unsigned) (head_dim / 2)>>>(xd, heads, head_dim, pos0, invd);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "block rope launch", e);
    return 0;
}

static int kv_put_rows_dev(gm4_kv *kv, int n, int pos0, const float *kd, const float *vd, char *err,
                           size_t errlen) {
    if (pos0 < 0 || (!kv->ring && pos0 + n > kv->cap)) {
        snprintf(err, errlen, "block cache write %d..%d outside the %s cache's %d positions", pos0,
                 pos0 + n - 1, kv->ring ? "ring" : "appending", kv->cap);
        return -1;
    }
    const int block = 128;
    const dim3 grid((unsigned) ((kv->per + block - 1) / block), (unsigned) n);
    kv_put_rows_kernel<<<grid, block>>>(kv->keys, kv->values, kd, vd, kv->per, pos0, kv->cap,
                                        kv->ring);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "block cache write launch", e);
    return 0;
}

/* The block's attention: one block per (head, query), the window each query's
 * own position allows. The widest window over the block sizes the launch's
 * shared scores, the same array attn_split_kernel keeps per run. */
static int attn_rows_dev(gm4_gpu *g, gm4_kv *kv, const float *qd, int n, int pos0, int heads,
                         float *outd, char *err, size_t errlen) {
    if (heads % kv->kv_heads) {
        snprintf(err, errlen, "%d query heads do not divide into %d kv heads", heads,
                 kv->kv_heads);
        return -1;
    }
    const int window = kv->ring ? kv->cap : 0;
    int widest = pos0 + n;
    if (window > 0 && widest > window) widest = window;
    const size_t smem = (32 + (size_t) widest) * sizeof(float);
    if (smem > smem_max()) {
        snprintf(err, errlen, "block attention needs %zu bytes of shared scores, the device allows %zu",
                 smem, smem_max());
        return -1;
    }
    const dim3 grid((unsigned) heads, (unsigned) n);
    attn_rows_kernel<<<grid, 256, smem>>>(qd, kv->keys, kv->values, outd, heads, kv->kv_heads,
                                          kv->head_dim, pos0, window, kv->cap, kv->ring);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "block attention launch", e);
    return 0;
}

/* The block's router selection: one warp per token, the single-token body. */
static int router_rows_dev(gm4_gpu *g, const float *scoresd, const float *pesd, int experts,
                           int k, int n, float *probsd, int *pickedd, float *wseld, char *err,
                           size_t errlen) {
    if (k > experts || k > 32) {
        snprintf(err, errlen, "block selection needs k <= 32 and k <= %d experts, got %d", experts, k);
        return -1;
    }
    const size_t smem = (size_t) (k > 32 ? k : 32) * sizeof(float);
    router_topk_rows_kernel<<<(unsigned) n, 32, smem>>>(scoresd, pesd, experts, k, probsd, pickedd,
                                                       wseld);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "block router launch", e);
    return 0;
}

/* Whether the expert geometry is the one the item passes implement. Checked,
 * never assumed: the two walks below are the block and span walks at dim 8
 * and this container's two payloads, and anything else is a refusal with the
 * reason, not a quiet slow path. */
static int moe_items_ready(gm4_gpu *g, const gm4_layer *L, char *err, size_t errlen) {
    const Tensor *gu = &g->ts[L->egu];
    const Tensor *dn = &g->ts[L->edn];
    const char *why = NULL;
    if (gu->npay != 2 || dn->npay != 2) why = "the expert stacks are not two payloads";
    else if (gu->pays[0].variant != 3 || gu->pays[1].variant != 3) why = "gate/up is not the block geometry";
    else if (dn->pays[0].variant != 4 || dn->pays[1].variant != 4) why = "down is not the span geometry";
    else if (gu->pays[0].l.dim != 8 || dn->pays[0].l.dim != 8) why = "the expert stacks are not dim 8";
    else if (gu->pays[0].l.prow || dn->pays[0].l.prow) why = "an expert payload carries a 13th-bit plane";
    if (why) {
        snprintf(err, errlen, "the block path needs this artifact's expert geometry: %s", why);
        return 0;
    }
    return 1;
}

#define GM4_MOE_THREADS 128
#define GM4_MOE_NT 8

/* The gathered activation rows of the pairs, the gate/up item pass, the SwiGLU
 * over the compact block, the down item pass, and the per-token reduce. */
static int moe_block_dev(gm4_gpu *g, const gm4_layer *L, const gm4_pre *P, int npairs,
                         int nitems, int k, int n, char *err, size_t errlen) {
    const Tensor *gu = &g->ts[L->egu];
    const Tensor *dn = &g->ts[L->edn];
    const int ff = L->moe_ff;
    const int nh = L->hidden;
    const int cols = nh;

    const dim3 ggrid((unsigned) ((cols + 255) / 256), (unsigned) npairs);
    moe_gather_kernel<<<ggrid, 256>>>(P->xs32, P->hb, P->perm, cols);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "block expert gather launch", e);

    const unsigned warps = GM4_MOE_THREADS / 32;
    const unsigned span = warps * GM4_MOE_ROWS_PER_WARP;
    const dim3 gu_grid((unsigned) ((2 * ff + (int) span - 1) / (int) span), (unsigned) nitems);
    moe_gu_items_kernel<8, GM4_MOE_NT><<<gu_grid, GM4_MOE_THREADS>>>(
        gu->pays[0], gu->pays[1], P->items, P->xs32, cols, P->gu, ff, 2 * ff);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "block expert gate/up launch", e);

    if (gelu_blocks_dev(P->gu, P->act, ff, npairs, err, errlen) != 0) return -1;

    const dim3 dn_grid((unsigned) ((nh + (int) span - 1) / (int) span), (unsigned) nitems);
    moe_dn_items_kernel<8, GM4_MOE_NT><<<dn_grid, GM4_MOE_THREADS>>>(
        dn->pays[0], dn->pays[1], P->items, P->dst, P->act, P->ddn, nh, ff);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "block expert down launch", e);

    const dim3 rgrid((unsigned) ((nh + 255) / 256), (unsigned) n);
    moe_reduce_tok_kernel<<<rgrid, 256>>>(P->ddn, P->wsel, P->moe, k, nh);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "block expert reduce launch", e);
    return 0;
}

/* The router's picks grouped into work items on the host: counts per expert,
 * the prefix offsets, the pairs in expert order (and, per pair, the slot the
 * down projection writes), then one (expert, first pair, count) triple per
 * segment of GM4_MOE_NT tokens. The reference builds the same three arrays the
 * same way (ds41_vq_prefill_fused.cuh:214-225); the pairs are the only thing
 * that has to leave the device, which is why this is the block path's one
 * host round trip per layer. */
static int moe_items_build(gm4_pre *P, int n, int k, int experts, int *nitems, char *err,
                           size_t errlen) {
    const size_t npairs = (size_t) n * (size_t) k;
    if (npairs > P->listcap) {
        snprintf(err, errlen, "%zu pairs exceed the scratch's %zu", npairs, P->listcap);
        return -1;
    }
    int *picks = (int *) malloc(npairs * sizeof(int));
    int *cnt = (int *) calloc((size_t) experts + 1, sizeof(int));
    int *off = (int *) malloc(((size_t) experts + 1) * sizeof(int));
    int32_t *perm = (int32_t *) malloc(npairs * sizeof(int32_t));
    int32_t *dst = (int32_t *) malloc(npairs * sizeof(int32_t));
    int32_t *items = (int32_t *) malloc(npairs * 3 * sizeof(int32_t));
    if (!picks || !cnt || !off || !perm || !dst || !items) {
        snprintf(err, errlen, "block item grouping: host allocation failed");
        free(picks);
        free(cnt);
        free(off);
        free(perm);
        free(dst);
        free(items);
        return -1;
    }
    cudaError_t e = cudaMemcpy(picks, P->picked, npairs * sizeof(int), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        fail(err, errlen, "block item grouping: picks readback", e);
        free(picks);
        free(cnt);
        free(off);
        free(perm);
        free(dst);
        free(items);
        return -1;
    }
    for (size_t i = 0; i < npairs; i++)
        if (picks[i] >= 0 && picks[i] < experts) cnt[picks[i]]++;
    off[0] = 0;
    for (int x = 0; x < experts; x++) off[x + 1] = off[x] + cnt[x];
    int *cursor = (int *) malloc(((size_t) experts + 1) * sizeof(int));
    if (!cursor) {
        snprintf(err, errlen, "block item grouping: host allocation failed");
        free(picks);
        free(cnt);
        free(off);
        free(perm);
        free(dst);
        free(items);
        return -1;
    }
    memcpy(cursor, off, ((size_t) experts + 1) * sizeof(int));
    for (int t = 0; t < n; t++) {
        for (int j = 0; j < k; j++) {
            const int x = picks[(size_t) t * k + j];
            if (x < 0 || x >= experts) continue;
            const int at = cursor[x]++;
            perm[at] = t;
            dst[at] = t * k + j;
        }
    }
    int ni = 0;
    for (int x = 0; x < experts; x++) {
        for (int t0 = 0; t0 < cnt[x]; t0 += GM4_MOE_NT) {
            const int nt = cnt[x] - t0 < GM4_MOE_NT ? cnt[x] - t0 : GM4_MOE_NT;
            items[ni * 3 + 0] = x;
            items[ni * 3 + 1] = off[x] + t0;
            items[ni * 3 + 2] = nt;
            ni++;
        }
    }
    e = cudaMemcpyAsync(P->perm, perm, npairs * sizeof(int32_t), cudaMemcpyHostToDevice,
                        cudaStreamPerThread);
    if (e == cudaSuccess)
        e = cudaMemcpyAsync(P->dst, dst, npairs * sizeof(int32_t), cudaMemcpyHostToDevice,
                            cudaStreamPerThread);
    if (e == cudaSuccess)
        e = cudaMemcpyAsync(P->items, items, (size_t) ni * 3 * sizeof(int32_t),
                            cudaMemcpyHostToDevice, cudaStreamPerThread);
    free(picks);
    free(cnt);
    free(off);
    free(cursor);
    free(perm);
    free(dst);
    free(items);
    if (e != cudaSuccess) {
        fail(err, errlen, "block item upload", e);
        return -1;
    }
    *nitems = ni;
    return 0;
}

/* One layer over a block of n tokens. The order is layer_forward_dev's, from
 * the input norm to the output scalar; `xin` must be the scratch's first
 * activation block and the result lands in `outd`. */
static int layer_prefill_dev(gm4_gpu *g, gm4_layer *L, gm4_kv *kv, gm4_pre *P, int n, int pos0,
                             float *outd, char *err, size_t errlen) {
    const int nh = L->hidden;
    const int nhq = L->heads * L->head_dim;
    const int nk = L->kv_heads * L->head_dim;
    float *x0 = P->xs[0], *x1 = P->xs[1];

    if (rmsnorm_dev(x0, L->in_norm, P->hb, n, nh, L->eps, err, errlen) != 0) return -1;
    if (transpose_dev(P->hb, P->xT, n, nh, err, errlen) != 0) return -1;
    if (prefill_dev(g, L->q, P->xT, n, P->q, err, errlen) != 0) return -1;
    if (prefill_dev(g, L->k, P->xT, n, P->k, err, errlen) != 0) return -1;
    if (L->v >= 0) {
        if (prefill_dev(g, L->v, P->xT, n, P->v, err, errlen) != 0) return -1;
    } else {
        cudaError_t e = cudaMemcpyAsync(P->v, P->k, (size_t) n * (size_t) nk * sizeof(float),
                                        cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        if (e != cudaSuccess) return fail(err, errlen, "block k to v copy", e);
    }

    /* The per-head norms, then rotary; v_norm has no weight. In place: each
     * thread reads its own element and the row's inverse comes from a
     * reduction that has already completed. */
    if (rmsnorm_dev(P->q, L->q_norm, P->q, n * L->heads, L->head_dim, L->eps, err, errlen) != 0)
        return -1;
    if (rope_rows_dev(P->q, n, L->heads, L->head_dim, pos0, L->rope_inv, err, errlen) != 0)
        return -1;
    if (rmsnorm_dev(P->k, L->k_norm, P->k, n * L->kv_heads, L->head_dim, L->eps, err, errlen) != 0)
        return -1;
    if (rope_rows_dev(P->k, n, L->kv_heads, L->head_dim, pos0, L->rope_inv, err, errlen) != 0)
        return -1;
    if (rmsnorm_dev(P->v, NULL, P->v, n * L->kv_heads, L->head_dim, L->eps, err, errlen) != 0)
        return -1;
    if (kv_put_rows_dev(kv, n, pos0, P->k, P->v, err, errlen) != 0) return -1;
    if (attn_rows_dev(g, kv, P->q, n, pos0, L->heads, P->att, err, errlen) != 0) return -1;

    if (transpose_dev(P->att, P->xT, n, nhq, err, errlen) != 0) return -1;
    if (prefill_dev(g, L->o, P->xT, n, P->o, err, errlen) != 0) return -1;
    if (rmsnorm_dev(P->o, L->post_attn, P->nrm, n, nh, L->eps, err, errlen) != 0) return -1;
    if (residual_dev(x0, P->nrm, x1, (size_t) n * nh, 1.0f, err, errlen) != 0) return -1;

    if (rmsnorm_dev(x1, L->pre_ffn, P->hb, n, nh, L->eps, err, errlen) != 0) return -1;
    if (transpose_dev(P->hb, P->xT, n, nh, err, errlen) != 0) return -1;
    if (prefill_dev(g, L->dg_t, P->xT, n, P->dg, err, errlen) != 0) return -1;
    if (prefill_dev(g, L->du_t, P->xT, n, P->du, err, errlen) != 0) return -1;
    if (gelu_mul_dev(P->dg, P->du, P->dact, (size_t) n * L->dense_ff, err, errlen) != 0) return -1;
    if (transpose_dev(P->dact, P->xT, n, L->dense_ff, err, errlen) != 0) return -1;
    if (prefill_dev(g, L->dd_t, P->xT, n, P->dd, err, errlen) != 0) return -1;
    if (rmsnorm_dev(P->dd, L->post_ffn1, P->dense, n, nh, L->eps, err, errlen) != 0) return -1;

    if (rmsnorm_dev(x1, NULL, P->r, n, nh, L->eps, err, errlen) != 0) return -1;
    if (mul_rows_dev(P->r, L->router_scale, P->rs, nh, (size_t) n * nh, err, errlen) != 0)
        return -1;
    if (transpose_dev(P->rs, P->xT, n, nh, err, errlen) != 0) return -1;
    if (prefill_dev(g, L->rt, P->xT, n, P->scores, err, errlen) != 0) return -1;
    if (router_rows_dev(g, P->scores, L->pes, L->experts, L->top_k, n, P->probs, P->picked,
                        P->wsel, err, errlen) != 0)
        return -1;
    if (rmsnorm_dev(x1, L->pre_ffn2, P->hb, n, nh, L->eps, err, errlen) != 0) return -1;
    int nitems = 0;
    if (moe_items_build(P, n, L->top_k, L->experts, &nitems, err, errlen) != 0) return -1;
    if (moe_block_dev(g, L, P, n * L->top_k, nitems, L->top_k, n, err, errlen) != 0) return -1;

    if (rmsnorm_dev(P->moe, L->post_ffn2, P->moe2, n, nh, L->eps, err, errlen) != 0) return -1;
    if (residual_dev(P->dense, P->moe2, P->comb, (size_t) n * nh, 1.0f, err, errlen) != 0)
        return -1;
    if (rmsnorm_dev(P->comb, L->post_ffn, P->comb2, n, nh, L->eps, err, errlen) != 0) return -1;
    if (residual_dev(x1, P->comb2, outd, (size_t) n * nh, L->scalar, err, errlen) != 0) return -1;
    return 0;
}

int gm4_gpu_model_step(gm4_gpu *g, gm4_model *m, int token, int pos, float *logits, char *err,
                       size_t errlen) {
    if (!g || !m || !logits) {
        snprintf(err, errlen, "model_step needs a model and room for the logits");
        return -1;
    }
    if (token < 0 || token >= m->vocab) {
        snprintf(err, errlen, "token %d is outside the vocabulary of %d", token, m->vocab);
        return -1;
    }
    if (pos < 0 || pos >= m->ctx) {
        snprintf(err, errlen, "position %d is outside the model's %d", pos, m->ctx);
        return -1;
    }
    const double t0 = now_sec();
    /* The step's whole host input: two ints into the pinned mirror, then one
     * asynchronous copy. Every kernel reads them from the device. */
    m->staging[0] = token;
    m->staging[1] = pos;
    cudaError_t e = cudaMemcpyAsync(m->slots, m->staging, 2 * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, cudaStreamPerThread);
    if (e != cudaSuccess) return fail(err, errlen, "model slot upload", e);
    const int width = gm4_bucket_width();
    const int bucket = pos / width;
    int cap = (bucket + 1) * width - 1;
    if (cap > m->ctx - 1) cap = m->ctx - 1;
    int ran = 0;
    if (!m->graph_off && m->warm) {
        ran = model_graph(g, m, bucket, cap, err, errlen);
        if (ran < 0) return -1;
    }
    if (!ran) {
        if (model_body(g, m, pos + 1, err, errlen) != 0) return -1;
        m->warm = 1;
    }
    m->t_submit = now_sec() - t0;
    const double t1 = now_sec();
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "model sync", e);
    e = cudaMemcpy(logits, m->logits, (size_t) m->vocab * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "model logits download", e);
    m->t_wait = now_sec() - t1;
    return 0;
}

/* How many tokens one prefill call takes at most: the scratch is sized for the
 * block it is asked for, and this is the bound the block path refuses past. */
#define GM4_PREFILL_MAX 1024

/* A block of n tokens at pos0..pos0+n-1, in one submission per layer: the
 * prefill the engine's sync() walks token by token today. The result is the
 * logits at the block's last position - a prompt's next-token distribution is
 * the last token's, and a full-vocabulary head over every token of the block
 * would be 262144 rows of output nothing reads.
 *
 * The caches take every position, so a prompt longer than one block is simply
 * several blocks: the ring layers keep their window and the appending layers
 * their prefix, and the second block reads the first block's keys. That is why
 * the reference's window-preserving chunk slice
 * (ds4_ds41_forward.inc:1222-1231) is not needed here - it exists for its
 * bounded replay, where a block that does not recompute a window would leave
 * the next block's first queries short of one. */
int gm4_gpu_model_prefill(gm4_gpu *g, gm4_model *m, const int32_t *tokens, int n, int pos0,
                          float *logits, char *err, size_t errlen) {
    if (!g || !m || !tokens || !logits) {
        snprintf(err, errlen, "prefill needs a model, tokens and room for the logits");
        return -1;
    }
    if (n <= 0 || n > GM4_PREFILL_MAX) {
        snprintf(err, errlen, "prefill block %d is outside 1..%d", n, GM4_PREFILL_MAX);
        return -1;
    }
    if (pos0 < 0 || pos0 + n > m->ctx) {
        snprintf(err, errlen, "prefill positions %d..%d are outside the model's %d", pos0,
                 pos0 + n - 1, m->ctx);
        return -1;
    }
    for (int i = 0; i < n; i++)
        if (tokens[i] < 0 || tokens[i] >= m->vocab) {
            snprintf(err, errlen, "token %d at index %d is outside the vocabulary of %d",
                     (int) tokens[i], i, m->vocab);
            return -1;
        }
    /* Every layer's expert geometry has to be the one the item passes
     * implement, and it is checked before the first launch rather than after
     * the first layer has already written its caches. */
    for (int i = 0; i < m->n; i++)
        if (!moe_items_ready(g, m->layers[i], err, errlen)) return -1;

    const double t0 = now_sec();
    if (pre_scratch(m, n, err, errlen) != 0) return -1;
    gm4_pre *P = m->pre;
    cudaError_t e = cudaMemcpyAsync(P->tokd, tokens, (size_t) n * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, cudaStreamPerThread);
    if (e != cudaSuccess) return fail(err, errlen, "prefill token upload", e);
    if (embed_row_dev(g, m->embed, P->tokd, n, m->embed_scale, P->xs[0], err, errlen) != 0)
        return -1;
    for (int i = 0; i < m->n; i++) {
        if (layer_prefill_dev(g, m->layers[i], m->kv[i], P, n, pos0, P->xs[1], err, errlen) != 0)
            return -1;
        /* The next layer reads the same block: one device copy, no round
         * trip through an arena the layers would each have to own. */
        e = cudaMemcpyAsync(P->xs[0], P->xs[1], (size_t) n * (size_t) m->hidden * sizeof(float),
                            cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        if (e != cudaSuccess) return fail(err, errlen, "prefill activation swap", e);
    }
    if (rmsnorm_dev(P->xs[0], m->fnorm, P->xs[0], n, m->hidden, m->eps, err, errlen) != 0)
        return -1;
    /* The head over the block's last token only. */
    if (matvec_dev(g, m->head, P->xs[0] + (size_t) (n - 1) * (size_t) m->hidden, m->logits,
                   m->threads, err, errlen) != 0)
        return -1;
    const unsigned threads = 256;
    softcap_kernel<<<(unsigned) ((m->vocab + (int) threads - 1) / (int) threads), threads>>>(
        m->logits, (size_t) m->vocab, m->softcap);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "prefill softcap launch", e);

    m->t_submit = now_sec() - t0;
    const double t1 = now_sec();
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "prefill sync", e);
    e = cudaMemcpy(logits, m->logits, (size_t) m->vocab * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "prefill logits download", e);
    m->t_wait = now_sec() - t1;
    return 0;
}

/* One scratch buffer back on the host, for the bisect that names the first
 * wrong stage of the block path. Only the single-use buffers are exposed; the
 * others hold a later stage's value by the time the call returns. */
int gm4_gpu_prefill_probe(gm4_model *m, int which, float *out, size_t floats, char *err,
                          size_t errlen) {
    if (!m || !m->pre || !out) {
        snprintf(err, errlen, "prefill_probe needs a model with a block scratch");
        return -1;
    }
    gm4_pre *P = m->pre;
    const float *src = NULL;
    switch (which) {
        case 3: src = P->q; break;
        case 5: src = P->k; break;
        case 7: src = P->v; break;
        case 8: src = P->att; break;
        case 9: src = P->o; break;
        case 15: src = P->dense; break;
        case 16: src = P->r; break;
        case 17: src = P->scores; break;
        case 18: src = P->moe; break;
        case 19: src = P->moe2; break;
        case 20: src = P->comb; break;
        case 21: src = P->comb2; break;
        case 22: src = P->xs[1]; break;
        case 23: src = P->gu; break;
        case 24: src = P->act; break;
        case 25: src = P->ddn; break;
        case 26: src = P->hb; break;    /* the block's pre-FFN2 normed rows */
        case 27: src = P->xs32; break;  /* the expert pass's gathered activations */
        default:
            snprintf(err, errlen, "prefill_probe has no stage %d", which);
            return -1;
    }
    cudaError_t e = cudaMemcpy(out, src, floats * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "prefill probe download", e);
    return 0;
}

/* The block's expert lists, read back for a gate that needs the mapping (the
 * step orders its expert rows by slot, the block by expert). Same single-use
 * rule as the probe: the lists are rebuilt every layer, so this is only valid
 * between the layer's item build and its first expert launch. */
int gm4_gpu_prefill_list(gm4_model *m, int which, int32_t *out, int n, int *count, char *err,
                         size_t errlen) {
    if (!m || !m->pre || !out || n <= 0) {
        snprintf(err, errlen, "prefill_list needs a model with a block scratch and room");
        return -1;
    }
    gm4_pre *P = m->pre;
    const void *src = NULL;
    size_t bytes = 0;
    if (which == 0) {
        src = P->perm;
        bytes = (size_t) n * sizeof(int32_t);
    } else if (which == 1) {
        src = P->dst;
        bytes = (size_t) n * sizeof(int32_t);
    } else if (which == 2) {
        src = P->items;
        bytes = (size_t) n * 3 * sizeof(int32_t);
        if (count) *count = n;
    } else if (which == 3) {
        src = P->picked;   /* the block's router selection, slot order, last layer */
        bytes = (size_t) n * sizeof(int32_t);
    } else {
        snprintf(err, errlen, "prefill_list has no list %d", which);
        return -1;
    }
    if (bytes > (size_t) P->listcap * 3 * sizeof(int32_t)) {
        snprintf(err, errlen, "prefill_list %d of %zu bytes exceeds the scratch's lists", which,
                 bytes);
        return -1;
    }
    cudaError_t e = cudaMemcpy(out, src, bytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "prefill list download", e);
    return 0;
}

/* This layer's router output, read back for a gate that compares a block's
 * routing against the step's. Same single-use rule as the probe. */
int gm4_gpu_layer_list(gm4_layer *L, int which, void *out, int n, char *err, size_t errlen) {
    if (!L || !out || n <= 0 || n > L->top_k) {
        snprintf(err, errlen, "layer_list needs a layer, room and n <= top_k");
        return -1;
    }
    const void *src = NULL;
    size_t bytes = 0;
    if (which == 0) {
        src = L->picked;
        bytes = (size_t) n * sizeof(int);
    } else if (which == 1) {
        src = L->wsel;
        bytes = (size_t) n * sizeof(float);
    } else {
        snprintf(err, errlen, "layer_list has no list %d", which);
        return -1;
    }
    cudaError_t e = cudaMemcpy(out, src, bytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "layer list download", e);
    return 0;
}

/* The block path's device footprint: one scratch for the whole chain, grown to
 * the longest block a session asked for. */
size_t gm4_gpu_model_prefill_bytes(const gm4_model *m) {
    if (!m || !m->pre) return 0;
    return m->pre->floats * sizeof(float)
         + m->pre->listcap * 3 * sizeof(float) + m->pre->listcap * 3 * sizeof(int32_t)
         + (size_t) m->pre->cap * sizeof(int32_t);
}

/* How the steps ran: captures taken, and whether the graphs are off. */
int gm4_gpu_model_graphs(const gm4_model *m, int *captures, int *off) {
    if (!m) return -1;
    if (captures) *captures = m->captures;
    if (off) *off = m->graph_off;
    return 0;
}

/* Turn the graph path on or off. A gate compares the two on the same step: the
 * graph must be a pure replay of the kernels the direct walk submits, so the
 * two must agree bit for bit. */
int gm4_gpu_model_set_graph(gm4_model *m, int on) {
    if (!m) return -1;
    m->graph_off = on ? 0 : 1;
    return 0;
}

/* The last step's split, in milliseconds: host submit and everything after
 * the last launch (the sync and the 1 MiB logits readback). */
int gm4_gpu_model_split(const gm4_model *m, double *submit, double *wait) {
    if (!m) return -1;
    if (submit) *submit = m->t_submit * 1e3;
    if (wait) *wait = m->t_wait * 1e3;
    return 0;
}

} /* extern "C" */
