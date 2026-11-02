/* gemma4_gpu.cu - device residency and the fused VQ row dot.
 *
 * The arithmetic is the VQ tools' reference device kernel, ported for this
 * artifact's payload and checked against the CPU forward on the same input.
 * Nothing here decodes a weight matrix: the 9.0 GB artifact stays in its
 * containers, and a projection reads indices, codebook words and gains.
 *
 * Sources read first-hand before this file was written:
 * - /data/VQ_Tools/c/gpu/decode-gpu.cu: the DQV3 block kernel (gemv_v3blk,
 *   :285-333 with v3_rounds at :246-283), the dim-4 fast path (gemv_warp4,
 *   :199-237), the generic warp kernel (:619-656), the index reader
 *   (rd_idx_fmt, :110-133) and the selection rule (v3_blk_ok, :66-68).
 * - /data/VQ_Tools/crates/vq-format/src/lib.rs:314-361, :426-507: the tools'
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
 * are the VQ tools' code (/data/VQ_Tools/c/gpu/decode-gpu.cu:86-333). The
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
                            x + (size_t) b * 256u * (DIM > 0 ? DIM : L.dim), cb, L.dim, acc);
        cur = nxt;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[lo] += acc * __half2float(gr[r]);
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
    const float4 *x4 = (const float4 *) x;

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
    if (lane == 0) y[lo] += acc * __half2float(gr[r]);
}

/* One warp per row, no shared codebook: the 64 KB table stays in L1. The
 * fallback for any dim and any index width. */
template <int DIM>
__global__ void gemv_warp_kernel(const uint8_t *p, const Layout L, const uint32_t *rl,
                                 const float *x, float *y) {
    const int lo = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (lo >= L.nrows) return;
    const size_t r = row_of(L, (uint32_t) lo, rl);
    const int nidx_row = L.cols / L.dim;
    const __half *cb = (const __half *) (p + L.cb_off);
    const __half *gr = (const __half *) (p + L.gain_off);
    const uint8_t *st = p + L.stream_off;

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
    if (lane == 0) y[lo] += acc * __half2float(gr[r]);
}

/* ==== Tiled prefill: f16 tensor cores =====================================
 *
 * Ported from /data/VQ_Tools/c/gpu/decode-gpu.cu:320-520 (the tiled kernel),
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

/* One query token against a K/V history. Grid is one block per head; a warp
 * takes every nwarps-th position in the window for the score pass, the whole
 * block shares the softmax, and the value pass is one thread per dimension
 * with no reduction at all. K and V are f16, which is finer than the
 * reference's bf16 activations. The history is either dense (kv_ring == 0) or
 * a ring of `kv_cap` positions (kv_ring == 1), in which case every access goes
 * through kv_slot; the walk itself is always in position order, so the ring
 * changes addresses, never arithmetic. */
__global__ void attn_decode_kernel(const float *q, const __half *k, const __half *v,
                                   float *scores, float *out, int heads, int kv_heads,
                                   int head_dim, int positions, int first, int nwin,
                                   int kv_cap, int kv_ring) {
    extern __shared__ float red[];
    const int h = blockIdx.x;
    const int group = heads / kv_heads;
    const int kv_head = h / group;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const float *qh = q + (size_t) h * (size_t) head_dim;
    float *sc = scores + (size_t) h * (size_t) nwin;
    const size_t kv_stride = (size_t) kv_heads * (size_t) head_dim;

    for (int p = first + warp; p < positions; p += nwarps) {
        const size_t ks = kv_slot(p, kv_cap, kv_ring);
        const __half *kp = k + ks * kv_stride + (size_t) kv_head * head_dim;
        float s = 0.0f;
        for (int d = lane; d < head_dim; d += 32) s += qh[d] * __half2float(kp[d]);
        for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
        if (lane == 0) sc[p - first] = s; /* scaling is 1.0 */
    }
    __syncthreads();

    /* fp32 softmax, as the reference upcasts it (`:846`). */
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

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int p = 0; p < nwin; p++) {
            const size_t ks = kv_slot(first + p, kv_cap, kv_ring);
            acc += (sc[p] / sum)
                   * __half2float(v[ks * kv_stride + (size_t) kv_head * head_dim + d]);
        }
        out[(size_t) h * (size_t) head_dim + d] = acc;
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

    if (t != 0) return;
    float picked = 0.0f;
    for (int j = 0; j < k; j++) {
        int best = -1;
        float best_prob = -1.0f;
        for (int i = 0; i < experts; i++) {
            bool taken = false;
            for (int q = 0; q < j && !taken; q++) taken = indices[q] == i;
            if (taken) continue;
            if (probs[i] > best_prob) {
                best_prob = probs[i];
                best = i;
            }
        }
        indices[j] = best;
        picked += probs[best];
    }
    for (int j = 0; j < k; j++) {
        const int expert = indices[j];
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
    if (ensure_float(&g->ascores, &g->ascorescap, (size_t) heads * (size_t) nwin, err, errlen) != 0)
        return -1;
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

    attn_decode_kernel<<<(unsigned) heads, 256, 32 * sizeof(float)>>>(
        g->aq, g->akv, g->akv + kvn, g->ascores, g->aout, heads, kv_heads, head_dim, positions,
        first, nwin, positions, 0);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "attention launch", e);
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
    if (ensure_float(&g->ascores, &g->ascorescap, (size_t) heads * (size_t) nwin, err, errlen) != 0)
        return -1;
    cudaError_t e = cudaMemcpy(g->aq, q, qn * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(err, errlen, "ring attention upload", e);
    attn_decode_kernel<<<(unsigned) heads, 256, 32 * sizeof(float)>>>(
        g->aq, kv->keys, kv->values, g->ascores, g->aout, heads, kv->kv_heads, head_dim, positions,
        first, nwin, kv->cap, kv->ring);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(err, errlen, "ring attention launch", e);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail(err, errlen, "ring attention sync", e);
    e = cudaMemcpy(out, g->aout, qn * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return fail(err, errlen, "ring attention download", e);
    return 0;
}

} /* extern "C" */
