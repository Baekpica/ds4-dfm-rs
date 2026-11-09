/* Bonsai (qwen35) dense PQ2_0 matmuls, owned by the family.
 *
 * Structure copied from cuda/spark2_5_vq_gemm.cuh (the Spark-X2.6 fused VQ
 * GEMM): one fused decode-and-multiply kernel for batched rows, one decode
 * first GEMV for the one-row arm, and a launcher that picks by row count.  The
 * point of the copy is ownership, not arithmetic: Bonsai's dense path no
 * longer lives in the vendored cuda/mmq PQ2_0 hunks that every family shares.
 *
 * PQ2_0 format (cuda/mmq/ggml-common.h:187-197): 128 weights per block, one
 * fp16 scale then 32 code bytes; value j is code (j/4) at bits 2*(j%4), the
 * level is code - 1 in {-1,0,1,2}, and the value is level * d.  The reference
 * implementation is ds4.c pq2_0_row_f32.
 *
 * Numerics.  The levels stay int8 and the block scale stays OUTSIDE the dot:
 * every 128-value block contributes d_w * d_x * sum(level * q_x), so the
 * ternary weights remain exact and only the activation is quantized.  d_x is
 * one scale per 128-value block, the same width as the weight block, which is
 * sound here because the artifact ships its weights Hadamard-folded and the
 * engine rotates every matmul input first (ds4_hadamard_forward), so the
 * rotated rows carry no outliers.
 *
 * Why the activation is quantized once per call: each 128-row CTA stages its
 * own activation tile, and out_dim/128 CTAs stage the same rows, so
 * quantizing inside the CTA repeats the work once per output tile (34x on
 * ffn_down's 17408x5120).  pq2_quantize_x runs it once into a scratch buffer
 * and every CTA copies int8, which is what the vendored MMQ does.
 *
 * MEASURED, 2026-11-09, RTX 4070 SUPER sm_89, `make bonsai-dense-bench`
 * (tests/bonsai_dense_bench.cu, 20 iterations, random blocks), after the
 * rewrite: see docs/BONSAI.md.  The verified baseline this must not regress:
 * mul_mat_q<PQ2_0> runs 102-125 TFLOPS with IMMA.16832.S8 in its SASS and the
 * vendored MMVQ decode arm sustains 266-443 GB/s. */

#ifndef DS4_BONSAI_PQ2_GEMM_CUH
#define DS4_BONSAI_PQ2_GEMM_CUH

#include <cuda_fp16.h>
#include <mma.h>

namespace bonsai_pq2 {

enum {
    BLOCK_VALUES = 128,          /* weights per PQ2_0 block */
    BLOCK_BYTES  = 34,           /* 2-byte scale + 32 code bytes */
    TILE_OUT     = 128,          /* weight rows per block */
    TILE_ROWS    = 64,           /* activation rows (tokens) per block */
    TILE_K       = 128,          /* one weight block staged per step */
    KLD          = TILE_K + 16,  /* padded row stride: rows rotate banks */
    WARPS        = 8,
    THREADS      = WARPS * 32,
    GEMV_ROWS    = 8,            /* above this a row batch takes the GEMM */
    GEMV_MAX_ROWS = GEMV_ROWS,
};

/* Four 2-bit codes in one byte -> the four int8 levels (code - 1) packed in an
 * int32.  (c + 0x7f) ^ 0x80 is c - 1 modulo 256 for c < 4 and cannot carry
 * between bytes.  Must agree with ds4_pq2_0_expand_byte
 * (cuda/mmq/vecdotq.cuh:730), the scalar twin the MMQ/MMVQ path uses. */
__device__ __forceinline__ uint32_t expand_levels(uint32_t b) {
    const uint32_t v = (b & 0x03u) | ((b & 0x0Cu) << 6) |
                       ((b & 0x30u) << 12) | ((b & 0xC0u) << 18);
    return (v + 0x7F7F7F7Fu) ^ 0x80808080u;
}

__device__ __forceinline__ float block_scale(const uint8_t *p) {
    return __half2float(*reinterpret_cast<const __half *>(p));
}

/* One scale and one int8 value per element, per 128-value block: the shared
 * activation form both arms consume.  One warp per (row, block). */
__global__ void pq2_quantize_x(const float *__restrict__ x,
                               int8_t *__restrict__ xq,
                               float *__restrict__ xs,
                               uint32_t rows, uint32_t in_dim) {
    const uint32_t nblk = in_dim / BLOCK_VALUES;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t job  = blockIdx.x * (blockDim.x >> 5u) + warp;
    if (job >= rows * nblk) {
        return;
    }
    const uint32_t r = job / nblk;
    const uint32_t b = job % nblk;
    const float   *xr = x + (size_t) r * in_dim + (size_t) b * BLOCK_VALUES;

    float amax = 0.0f;
    for (uint32_t i = lane; i < BLOCK_VALUES; i += 32u) {
        amax = fmaxf(amax, fabsf(xr[i]));
    }
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    const float d = amax > 0.0f ? amax / 127.0f : 1.0f;
    if (lane == 0u) {
        xs[(size_t) r * nblk + b] = d;
    }
    int8_t *q = xq + (size_t) r * in_dim + (size_t) b * BLOCK_VALUES;
    for (uint32_t i = lane; i < BLOCK_VALUES; i += 32u) {
        const int v = __float2int_rn(xr[i] / d);
        q[i] = (int8_t) max(-127, min(127, v));
    }
}

/* out[rows][out_dim] = x[rows][in_dim] @ W[out_dim][in_dim]^T over int8
 * levels.  A block owns 128 weight rows and 128 activation rows; warp w owns
 * weight rows [16*w, +16) and every 16-row token group in turn, so one warp
 * MMA tile is 16 weight rows x 16 tokens.  Each 128-value block's int32 dot
 * is scaled into f32 before the next block starts: the block scale never
 * leaves the epilogue. */
__global__ void __launch_bounds__(THREADS)
pq2_gemm_kernel(const uint8_t *__restrict__ w, const int8_t *__restrict__ xq,
                const float *__restrict__ xs, float *__restrict__ out,
                uint32_t rows, uint32_t in_dim, uint32_t out_dim) {
    using namespace nvcuda::wmma;

    __shared__ __align__(16) int8_t sx[TILE_ROWS][KLD];
    __shared__ __align__(16) int8_t sw[TILE_OUT][KLD];
    __shared__ int   sacc[WARPS][16 * 16];
    __shared__ float sdx[TILE_ROWS];
    __shared__ float sdw[TILE_OUT];

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t row0 = blockIdx.y * TILE_ROWS;
    const uint32_t out0 = blockIdx.x * TILE_OUT;
    const uint32_t nblk = in_dim / BLOCK_VALUES;
    const size_t   wrow = (size_t) nblk * BLOCK_BYTES;

    /* This lane's share of each 16x16 output tile: 8 f32 per tile. */
    float acc[TILE_ROWS / 16][8];
    for (uint32_t t = 0; t < TILE_ROWS / 16u; t++) {
        for (uint32_t l = 0; l < 8u; l++) {
            acc[t][l] = 0.0f;
        }
    }

    for (uint32_t kb = 0; kb < nblk; kb++) {
        /* ---- activations: copy the quantized tile, one scale per row ---- */
        for (uint32_t i = tid; i < TILE_ROWS * (BLOCK_VALUES / 16u); i += THREADS) {
            const uint32_t r    = i / (BLOCK_VALUES / 16u);
            const uint32_t k    = (i % (BLOCK_VALUES / 16u)) * 16u;
            const bool     live = (row0 + r) < rows;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (live) {
                v = *reinterpret_cast<const uint4 *>(
                        xq + (size_t) (row0 + r) * in_dim + (size_t) kb * BLOCK_VALUES + k);
            }
            *reinterpret_cast<uint4 *>(&sx[r][k]) = v;
        }
        for (uint32_t i = tid; i < TILE_ROWS; i += THREADS) {
            sdx[i] = (row0 + i) < rows ? xs[(size_t) (row0 + i) * nblk + kb] : 0.0f;
        }

        /* ---- weights: 16 values per thread step, one uint4 store ---- */
        for (uint32_t i = tid; i < TILE_OUT * 8u; i += THREADS) {
            const uint32_t n  = i >> 3u;          /* weight row inside the tile */
            const uint32_t jw = i & 7u;           /* four code bytes = 16 values */
            const uint32_t o  = out0 + n;
            uint4 word = make_uint4(0u, 0u, 0u, 0u);
            if (o < out_dim) {
                const uint8_t *blk = w + (size_t) o * wrow + (size_t) kb * BLOCK_BYTES;
                const uint8_t *qs  = blk + 2u + jw * 4u;
                word = make_uint4(expand_levels(qs[0]), expand_levels(qs[1]),
                                  expand_levels(qs[2]), expand_levels(qs[3]));
                if (jw == 0u) {
                    sdw[n] = block_scale(blk);
                }
            } else if (jw == 0u) {
                sdw[n] = 0.0f;
            }
            *reinterpret_cast<uint4 *>(&sw[n][jw * 16u]) = word;
        }
        __syncthreads();

        /* ---- int8 dots over the staged block ----
         * The weight fragment belongs to the warp and the k step, not to the
         * token group, so it is loaded once per k step and reused by every
         * token group.  Reloading it per group put the kernel over the shared
         * memory bandwidth (measured: 2.3x behind MMQ at 512 rows). */
        fragment<accumulator, 16, 16, 16, int> tint[TILE_ROWS / 16u];
        for (uint32_t t = 0; t < TILE_ROWS / 16u; t++) {
            fill_fragment(tint[t], 0);
        }
        for (uint32_t kk = 0; kk < TILE_K; kk += 16u) {
            fragment<matrix_a, 16, 16, 16, signed char, row_major> a;
            load_matrix_sync(a, &sw[warp * 16u][kk], KLD);
            for (uint32_t t = 0; t < TILE_ROWS / 16u; t++) {
                fragment<matrix_b, 16, 16, 16, signed char, col_major> b;
                /* B is the activation tile: sx[n][k] read column-major is (k, n). */
                load_matrix_sync(b, &sx[t * 16u][kk], KLD);
                mma_sync(tint[t], a, b, tint[t]);
            }
        }

        /* Scale this block's int32 dots into the f32 accumulators. */
        for (uint32_t t = 0; t < TILE_ROWS / 16u; t++) {
            store_matrix_sync(sacc[warp], tint[t], 16, mem_row_major);
            __syncwarp();
            for (uint32_t l = 0; l < 8u; l++) {
                const uint32_t e = lane + l * 32u;
                acc[t][l] += (float) sacc[warp][e] *
                             sdw[warp * 16u + (e >> 4u)] *
                             sdx[t * 16u + (e & 15u)];
            }
            __syncwarp();
        }
        __syncthreads();
    }

    /* ---- write the warp's tiles: rows of `out` are tokens ---- */
    for (uint32_t t = 0; t < TILE_ROWS / 16u; t++) {
        for (uint32_t l = 0; l < 8u; l++) {
            const uint32_t e = lane + l * 32u;
            const uint32_t r = row0 + t * 16u + (e & 15u);   /* token */
            const uint32_t c = out0 + warp * 16u + (e >> 4u); /* output row */
            if (r < rows && c < out_dim) {
                out[(size_t) r * out_dim + c] = acc[t][l];
            }
        }
    }
}

/* One warp per output row, ROWS tokens per launch: a row's weights are read
 * once and dotted with every token.  Each lane walks the row's 16-value words
 * (one 4-byte code group each), so the weight stream stays contiguous, and
 * the dot is four dp4a per word -- the vendored MMVQ shape. */
template <uint32_t ROWS>
__global__ void __launch_bounds__(THREADS)
pq2_gemv_kernel(const uint8_t *__restrict__ w, const int8_t *__restrict__ xq,
                const float *__restrict__ xs, float *__restrict__ out,
                uint32_t in_dim, uint32_t out_dim) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t o    = blockIdx.x * WARPS + warp;
    if (o >= out_dim) {
        return;
    }
    const uint32_t nblk   = in_dim / BLOCK_VALUES;
    const uint32_t nwords = in_dim / 16u;
    const uint8_t *row    = w + (size_t) o * nblk * BLOCK_BYTES;
    float acc[ROWS];
    for (uint32_t r = 0; r < ROWS; r++) {
        acc[r] = 0.0f;
    }

    for (uint32_t wd = lane; wd < nwords; wd += 32u) {
        const uint32_t b  = wd >> 3u;             /* 8 words per 128-value block */
        const uint32_t jw = wd & 7u;
        const uint8_t *blk = row + (size_t) b * BLOCK_BYTES;
        const uint8_t *qs  = blk + 2u + jw * 4u;
        const int l0 = (int) expand_levels(qs[0]);
        const int l1 = (int) expand_levels(qs[1]);
        const int l2 = (int) expand_levels(qs[2]);
        const int l3 = (int) expand_levels(qs[3]);
        const float dw = block_scale(blk);
        for (uint32_t r = 0; r < ROWS; r++) {
            const int *a = reinterpret_cast<const int *>(
                    xq + (size_t) r * in_dim + (size_t) wd * 16u);
            int s = 0;
            s = __dp4a(l0, a[0], s);
            s = __dp4a(l1, a[1], s);
            s = __dp4a(l2, a[2], s);
            s = __dp4a(l3, a[3], s);
            /* One f32 add per word per token: the block scale and the
             * activation scale for this block, exactly. */
            acc[r] += (float) s * dw * xs[(size_t) r * nblk + b];
        }
    }

    for (uint32_t r = 0; r < ROWS; r++) {
        float sum = acc[r];
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, off);
        }
        if (lane == 0u) {
            out[(size_t) r * out_dim + o] = sum;
        }
    }
}

/* Token-embedding row lookup: dequantize the rows of a [n_rows, in_dim] PQ2_0
 * matrix named by row0 (dense rows) or by a device token array.  Element for
 * element the arithmetic of pq2_0_row_f32 (ds4.c): the fp16 block scale times
 * (code - 1), with element j in code byte j/4 at bits 2*(j%4). */
__global__ void pq2_rows_kernel(float *__restrict__ out,
                                const uint8_t *__restrict__ w,
                                const int32_t *__restrict__ tokens,
                                uint64_t row0, uint32_t n_rows, uint32_t in_dim) {
    const uint64_t total = (uint64_t) n_rows * in_dim;
    const uint64_t gid   = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) {
        return;
    }
    const uint32_t r   = (uint32_t) (gid / in_dim);
    const uint32_t i   = (uint32_t) (gid - (uint64_t) r * in_dim);
    const int32_t  src = tokens ? tokens[r] : (int32_t) (row0 + r);
    if (src < 0) {
        out[gid] = 0.0f;
        return;
    }
    const uint8_t *blk = w + (size_t) (uint32_t) src * ((size_t) in_dim / BLOCK_VALUES) * BLOCK_BYTES +
                         (size_t) (i / BLOCK_VALUES) * BLOCK_BYTES;
    const uint32_t j    = i % BLOCK_VALUES;
    const uint8_t  code = (uint8_t) ((blk[2u + (j >> 2)] >> (2u * (j & 3u))) & 0x03u);
    out[gid] = block_scale(blk) * (float) ((int) code - 1);
}

} // namespace bonsai_pq2

/* out[rows][out_dim] = x[rows][in_dim] @ W[out_dim][in_dim]^T, W PQ2_0.
 * Returns 0 when the shape is outside this path (the caller then falls back),
 * 1 on success.  xq/xs are the shared activation scratch, owned by the caller
 * so its lifetime outlives graph capture. */
extern "C" int ds4_gpu_bonsai_pq2_matmul(const void *w, const float *x,
                                         int8_t *xq, float *xs, float *out,
                                         uint32_t out_dim, uint32_t in_dim,
                                         uint32_t rows, cudaStream_t stream) {
    if (!w || !x || !xq || !xs || !out || rows == 0u || out_dim == 0u ||
        in_dim == 0u || in_dim % bonsai_pq2::BLOCK_VALUES != 0u) {
        return 0;
    }
    const uint8_t *wb = (const uint8_t *) w;
    const uint32_t threads = 256u;
    const uint32_t warps   = threads / 32u;
    const uint32_t jobs    = rows * (in_dim / bonsai_pq2::BLOCK_VALUES);
    const uint32_t qgrid   = (jobs + warps - 1u) / warps;
    bonsai_pq2::pq2_quantize_x<<<qgrid, threads, 0, stream>>>(
            x, xq, xs, rows, in_dim);
    if (cudaGetLastError() != cudaSuccess) {
        return 0;
    }

    if (rows <= bonsai_pq2::GEMV_MAX_ROWS) {
        const uint32_t blocks = (out_dim + bonsai_pq2::WARPS - 1u) / bonsai_pq2::WARPS;
        switch (rows) {
        case 1: bonsai_pq2::pq2_gemv_kernel<1><<<blocks, 256, 0, stream>>>(wb, xq, xs, out, in_dim, out_dim); break;
        case 2: bonsai_pq2::pq2_gemv_kernel<2><<<blocks, 256, 0, stream>>>(wb, xq, xs, out, in_dim, out_dim); break;
        case 3: bonsai_pq2::pq2_gemv_kernel<3><<<blocks, 256, 0, stream>>>(wb, xq, xs, out, in_dim, out_dim); break;
        case 4: bonsai_pq2::pq2_gemv_kernel<4><<<blocks, 256, 0, stream>>>(wb, xq, xs, out, in_dim, out_dim); break;
        case 5: bonsai_pq2::pq2_gemv_kernel<5><<<blocks, 256, 0, stream>>>(wb, xq, xs, out, in_dim, out_dim); break;
        case 6: bonsai_pq2::pq2_gemv_kernel<6><<<blocks, 256, 0, stream>>>(wb, xq, xs, out, in_dim, out_dim); break;
        case 7: bonsai_pq2::pq2_gemv_kernel<7><<<blocks, 256, 0, stream>>>(wb, xq, xs, out, in_dim, out_dim); break;
        default: bonsai_pq2::pq2_gemv_kernel<8><<<blocks, 256, 0, stream>>>(wb, xq, xs, out, in_dim, out_dim); break;
        }
    } else {
        const dim3 grid((out_dim + bonsai_pq2::TILE_OUT - 1u) / bonsai_pq2::TILE_OUT,
                        (rows + bonsai_pq2::TILE_ROWS - 1u) / bonsai_pq2::TILE_ROWS);
        bonsai_pq2::pq2_gemm_kernel<<<grid, bonsai_pq2::THREADS, 0, stream>>>(
                wb, xq, xs, out, rows, in_dim, out_dim);
    }
    return cudaGetLastError() == cudaSuccess ? 1 : 0;
}

extern "C" int ds4_gpu_bonsai_pq2_rows(float *out, const void *w,
                                       const int32_t *tokens, uint64_t row0,
                                       uint32_t n_rows, uint32_t in_dim,
                                       cudaStream_t stream) {
    if (!out || !w || n_rows == 0u || in_dim == 0u ||
        in_dim % bonsai_pq2::BLOCK_VALUES != 0u) {
        fprintf(stderr, "ds4_gpu_bonsai_pq2_rows: bad arguments (n_rows=%u in_dim=%u)\n",
                n_rows, in_dim);
        return -1;
    }
    const uint64_t total = (uint64_t) n_rows * in_dim;
    bonsai_pq2::pq2_rows_kernel<<<(unsigned) ((total + 255u) / 256u), 256, 0, stream>>>(
        out, (const uint8_t *) w, tokens, row0, n_rows, in_dim);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_gpu_bonsai_pq2_rows: launch failed: %s\n",
                cudaGetErrorString(err));
        return -2;
    }
    return 0;
}

#endif
