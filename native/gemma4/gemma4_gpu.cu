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

/* One warp per row, the three 128-byte lines of the next block issued before
 * this one computes (the reference's cross-block pipeline). */
template <int DIM, int EXT>
__global__ void gemv_v3blk_kernel(const uint8_t *p, const Layout L, const uint32_t *rl,
                                  const float *x, float *y) {
    const uint32_t lo = (uint32_t) ((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const uint32_t lane = threadIdx.x & 31u;
    if (lo >= (uint32_t) L.nrows) return;
    const size_t r = row_of(L, lo, rl);
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
                            row_x(x, L, lo) + (size_t) b * 256u * (DIM > 0 ? DIM : L.dim), cb,
                            L.dim, acc);
        cur = nxt;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) {
        const float v = acc * __half2float(gr[r]);
        if (L.first) y[lo] = v;
        else y[lo] += v;
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
 * is the identity pair and is skipped. */
__global__ void rope_kernel(float *x, int heads, int head_dim, int pos, const float *inv) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    const int half = head_dim / 2;
    if (d >= half) return;
    const float w = inv[d];
    if (w == 0.0f) return;
    float *base = x + (size_t) h * (size_t) head_dim;
    const float angle = (float) pos * w;
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
                                  int head_dim, int first, int nwin, int splits, int chunk,
                                  int kv_cap, int kv_ring) {
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

/* Append one position to a cache: position `pos` goes to slot(pos), which the
 * host computed with the same rule. One thread per value of the position. */
__global__ void kv_put_kernel(__half *k, __half *v, const float *ksrc, const float *vsrc, int per,
                              int slot) {
    const int i = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;
    if (i >= per) return;
    const size_t at = (size_t) slot * (size_t) per + (size_t) i;
    k[at] = __float2half_rn(ksrc[i]);
    v[at] = __float2half_rn(vsrc[i]);
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

/* The selection: the k highest probabilities, highest first, ties to the
 * lower index (the order `torch.topk` returns for an exact tie, which the CPU
 * forward reproduces at forward.rs:80-88), renormalised over the picked set
 * and scaled per expert. One block; the selection itself runs on one thread,
 * which is 128 comparisons per slot and keeps the tie rule explicit. */
__global__ void router_topk_kernel(const float *scores, const float *per_expert_scale, int experts,
                                   int k, float *probs, int *indices, float *weights) {
    extern __shared__ float red[];
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

int launch_payload(const Payload &p, const uint32_t *rl, const float *x, float *y, int threads) {
    const int blocks = (p.l.nrows * 32 + threads - 1) / threads;
    if (p.variant == 3) {
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
    if (!specs || n <= 0) {
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
    e = cudaMalloc((void **) &g->xbuf, xcap * sizeof(float));
    if (e != cudaSuccess) {
        snprintf(err, errlen, "workspace x cudaMalloc failed: %s", cudaGetErrorString(e));
        gm4_gpu_close(g);
        return NULL;
    }
    e = cudaMalloc((void **) &g->ybuf, ycap * sizeof(float));
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
    free(g);
}

void gm4_gpu_account_of(const gm4_gpu *g, gm4_account *out) { *out = g->acct; }

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

static int rope_dev(float *xd, int heads, int head_dim, int pos, const float *invd, char *err,
                    size_t errlen) {
    rope_kernel<<<(unsigned) heads, (unsigned) (head_dim / 2)>>>(xd, heads, head_dim, pos, invd);
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

static int kv_put_dev(gm4_kv *kv, int pos, const float *kd, const float *vd, char *err,
                      size_t errlen) {
    if (pos < 0 || (!kv->ring && pos >= kv->cap)) {
        snprintf(err, errlen, "chain kv position %d outside the %s cache's %d positions", pos,
                 kv->ring ? "ring" : "appending", kv->cap);
        return -1;
    }
    const int slot = kv->ring ? pos % kv->cap : pos;
    const int block = 128;
    kv_put_kernel<<<(kv->per + block - 1) / block, block>>>(kv->keys, kv->values, kd, vd, kv->per,
                                                           slot);
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
static int attention_dev(gm4_gpu *g, const float *qd, const __half *kd, const __half *vd,
                         int heads, int kv_heads, int head_dim, int first, int nwin,
                         int kv_cap, int kv_ring, float *outd, char *err, size_t errlen) {
    if (nwin <= 0 || head_dim <= 0 || head_dim % 2 || heads <= 0 || kv_heads <= 0
        || heads % kv_heads) {
        snprintf(err, errlen, "attention needs a positive window, an even head width and heads "
                 "divisible by kv heads");
        return -1;
    }
    int splits = ATN_BLOCKS / heads; /* runs per head to reach 64 blocks */
    if (splits < 1) splits = 1;
    while (splits > 1 && nwin / splits < ATN_MIN_RUN) splits >>= 1;
    if (splits > 64) splits = 64; /* the combine's shared weight table */
    const int blocks = heads * splits;
    const int chunk = (nwin + splits - 1) / splits;
    if (ensure_float(&g->ascores, &g->ascorescap, (size_t) blocks * (size_t) chunk, err, errlen)
        != 0)
        return -1;
    if (ensure_float(&g->apart, &g->apartcap, (size_t) blocks * (size_t) head_dim, err, errlen)
        != 0)
        return -1;
    if (ensure_float(&g->apm, &g->apmcap, (size_t) blocks, err, errlen) != 0) return -1;
    if (ensure_float(&g->apl, &g->aplcap, (size_t) blocks, err, errlen) != 0) return -1;
    attn_split_kernel<<<(unsigned) blocks, 256, 32 * sizeof(float)>>>(
        qd, kd, vd, g->ascores, g->apart, g->apm, g->apl, heads, kv_heads, head_dim, first, nwin,
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
                              float *outd, char *err, size_t errlen) {
    if (heads % kv->kv_heads) {
        snprintf(err, errlen, "%d query heads do not divide into %d kv heads", heads,
                 kv->kv_heads);
        return -1;
    }
    if (!kv->ring && pos >= kv->cap) {
        snprintf(err, errlen, "chain position %d past the appending cache's %d", pos, kv->cap);
        return -1;
    }
    const int positions = pos + 1;
    const int nwin = kv->ring && positions > kv->cap ? kv->cap : positions;
    const int first = positions - nwin;
    return attention_dev(g, qd, kv->keys, kv->values, heads, kv->kv_heads, kv->head_dim, first,
                         nwin, kv->cap, kv->ring, outd, err, errlen);
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
    const size_t n = (size_t) heads * (size_t) head_dim;
    if (ensure_float(&g->ax, &g->axcap, n, err, errlen) != 0) return -1;
    if (ensure_float(&g->ai, &g->aicap, (size_t) (head_dim / 2), err, errlen) != 0) return -1;
    cudaError_t e = cudaMemcpy(g->ax, x, n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "rope x upload", e);
    e = cudaMemcpy(g->ai, inv, (size_t) (head_dim / 2) * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "rope table upload", e);
    rope_kernel<<<(unsigned) heads, (unsigned) (head_dim / 2)>>>(g->ax, heads, head_dim, pos, g->ai);
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
    const int first = positions - nwin;
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
     * ring == 0, so both go through one launcher and the same schedule. */
    if (attention_dev(g, g->aq, g->akv, g->akv + kvn, heads, kv_heads, head_dim, first, nwin,
                      positions, 0, g->aout, err, errlen) != 0)
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
    const int slot = kv->ring ? pos % kv->cap : pos;
    const int block = 128;
    kv_put_kernel<<<(kv->per + block - 1) / block, block>>>(kv->keys, kv->values, g->kvu,
                                                            g->kvu + kv->per, kv->per, slot);
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
    const int nwin = kv->ring && positions > kv->cap ? kv->cap : positions;
    const int first = positions - nwin;
    const int head_dim = kv->head_dim;
    const size_t qn = (size_t) heads * (size_t) head_dim;
    if (ensure_float(&g->aq, &g->aqcap, qn, err, errlen) != 0) return -1;
    if (ensure_float(&g->aout, &g->aoutcap, qn, err, errlen) != 0) return -1;
    cudaError_t e = cudaMemcpy(g->aq, q, qn * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "ring attention upload", e);
    if (attention_dev(g, g->aq, kv->keys, kv->values, heads, kv->kv_heads, head_dim, first, nwin,
                      kv->cap, kv->ring, g->aout, err, errlen) != 0)
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
    L->arena_floats = narena;

    cudaError_t e = cudaMalloc((void **) &L->small, nsmall * sizeof(float));
    if (e == cudaSuccess) e = cudaMalloc((void **) &L->arena, narena * sizeof(float));
    if (e == cudaSuccess) e = cudaMalloc((void **) &L->picked, nex * sizeof(int));
    if (e == cudaSuccess)
        e = cudaMalloc((void **) &L->rl1, nex * 2 * nmoe * sizeof(uint32_t));
    if (e == cudaSuccess) e = cudaMalloc((void **) &L->rl2, nex * nh * sizeof(uint32_t));
    if (e != cudaSuccess) {
        if (L->small) cudaFree(L->small);
        if (L->arena) cudaFree(L->arena);
        if (L->picked) cudaFree(L->picked);
        if (L->rl1) cudaFree(L->rl1);
        if (L->rl2) cudaFree(L->rl2);
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

int gm4_gpu_layer_forward(gm4_gpu *g, gm4_layer *L, gm4_kv *kv, const float *x, int pos,
                          float *out, char *err, size_t errlen) {
    if (!g || !L || !kv || !x || !out) {
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
    cudaError_t e = cudaMemcpy(L->xa, x, (size_t) nh * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "layer x upload", e);

    /* residual = x; h = input_norm(residual); q, k and v from h. */
    if (rmsnorm_dev(L->xa, L->in_norm, L->hb, 1, nh, L->eps, err, errlen) != 0) return -1;
    if (matvec_dev(g, L->q, L->hb, L->qa, L->threads, err, errlen) != 0) return -1;
    if (matvec_dev(g, L->k, L->hb, L->ka, L->threads, err, errlen) != 0) return -1;
    if (L->v >= 0) {
        if (matvec_dev(g, L->v, L->hb, L->va, L->threads, err, errlen) != 0) return -1;
    } else {
        /* `value_states = key_states` before either norm (`:1258-1263`). */
        e = cudaMemcpyAsync(L->va, L->ka, (size_t) nk * sizeof(float), cudaMemcpyDeviceToDevice, 0);
        if (e != cudaSuccess) return fail(err, errlen, "layer k to v copy", e);
    }
    mark(L, 1);

    /* The per-head norms, then rotary on q and k; v_norm has no weight. */
    if (rmsnorm_dev(L->qa, L->q_norm, L->qb, L->heads, L->head_dim, L->eps, err, errlen) != 0)
        return -1;
    if (rope_dev(L->qb, L->heads, L->head_dim, pos, L->rope_inv, err, errlen) != 0) return -1;
    if (rmsnorm_dev(L->ka, L->k_norm, L->kb, L->kv_heads, L->head_dim, L->eps, err, errlen) != 0)
        return -1;
    if (rope_dev(L->kb, L->kv_heads, L->head_dim, pos, L->rope_inv, err, errlen) != 0) return -1;
    if (rmsnorm_dev(L->va, NULL, L->vb, L->kv_heads, L->head_dim, L->eps, err, errlen) != 0)
        return -1;

    /* The cache takes this position first: the query sees its own key. */
    if (kv_put_dev(kv, pos, L->kb, L->vb, err, errlen) != 0) return -1;
    mark(L, 2);
    if (attention_ring_dev(g, kv, L->qb, L->heads, pos, L->attn, err, errlen) != 0) return -1;
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

    /* The experts: one gate_up gather for all k, one gelu per expert, then a
     * down gather and a weighted accumulate per expert, as forward.rs:376-388
     * does it. */
    const int gcount = 2 * L->moe_ff;
    const size_t gn = (size_t) L->top_k * (size_t) gcount;
    if (rmsnorm_dev(L->xb, L->pre_ffn2, L->hb, 1, nh, L->eps, err, errlen) != 0) return -1;
    if (rowlist_dev(L->rl1, L->picked, L->top_k, gcount, gcount, err, errlen) != 0) return -1;
    if (rows_dev(g, L->egu, L->rl1, (int) gn, L->hb, 0, 0, L->gu, L->threads, err, errlen) != 0)
        return -1;
    /* One gelu over all k blocks: the same arithmetic as k launches, with the
     * expert's stride instead of the caller's loop. */
    if (gelu_blocks_dev(L->gu, L->act, (int) L->moe_ff, L->top_k, err, errlen) != 0) return -1;
    if (rowlist_dev(L->rl2, L->picked, L->top_k, nh, nh, err, errlen) != 0) return -1;
    /* One gather over all k experts and one weighted reduce: the down rows
     * are disjoint, so the launch is the concatenation of the k blocks. */
    e = cudaMemsetAsync(L->d, 0, (size_t) L->top_k * (size_t) nh * sizeof(float), 0);
    if (e != cudaSuccess) return fail(err, errlen, "layer down clear", e);
    if (rows_dev(g, L->edn, L->rl2, L->top_k * nh, L->act, nh, (int) L->moe_ff, L->d, L->threads,
                 err, errlen) != 0)
        return -1;
    if (moe_reduce_dev(L->d, L->wsel, L->moe, L->top_k, nh, err, errlen) != 0) return -1;
    mark(L, 9);

    /* The close: moe norm, dense + moe, the pair norm, the residual, scalar. */
    if (rmsnorm_dev(L->moe, L->post_ffn2, L->moe2, 1, nh, L->eps, err, errlen) != 0) return -1;
    if (residual_dev(L->dense, L->moe2, L->comb, (size_t) nh, 1.0f, err, errlen) != 0) return -1;
    if (rmsnorm_dev(L->comb, L->post_ffn, L->comb2, 1, nh, L->eps, err, errlen) != 0) return -1;
    if (residual_dev(L->xb, L->comb2, L->outd, (size_t) nh, L->scalar, err, errlen) != 0)
        return -1;
    mark(L, 10);

    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "layer sync", e);
    e = cudaMemcpy(out, L->outd, (size_t) nh * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "layer download", e);
    mark(L, 11);
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

} /* extern "C" */
