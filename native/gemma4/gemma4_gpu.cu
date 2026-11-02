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
    char devname[256];
    gm4_account acct;
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

} /* extern "C" */
