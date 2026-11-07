/* Spark-X2.6 fused VQ projection: decode the container's codewords and
 * multiply them in one kernel, so the engine needs no decoded weight image.
 *
 * The decode is the restore's, term for term (cuda/spark2_5.cuh:111-159
 * spark_restore_kernel): per 8-weight vector a 13-bit index (12 bits from 3
 * bytes per two vectors, the 13th from a bit plane), an 8-bit residual index
 * into a 256-word book, and on a version-3 release a second 13-bit stream
 * with its own 8192-word book. Per output row g1 = gain[row] multiplies
 * codeword + residual, g2 = gain2[row] multiplies the stage-2 codeword, and
 * the sum takes one BF16 rounding. Producing that same BF16 value in shared
 * memory keeps this path numerically equal to the image: the multiply is the
 * engine's BF16 x BF16 -> F32 GEMM (cuda/ds41_dense.cuh:228-253), so only the
 * accumulation order differs, and the decode itself is bit-identical.
 *
 * Why it exists: the artifact carries 0.531 bytes a weight (two 13-bit
 * streams plus an 8-bit residual per vector) against 2 for the BF16 image, so
 * a chunk reads 3.8x fewer weight bytes and 7.03 GiB of VRAM stay free. The
 * price is decode work per row tile: a 512-row chunk decodes each matrix four
 * times, which is why the tile is 64 rows and not 16. */

#ifndef DS4_SPARK2_5_VQ_GEMM_CUH
#define DS4_SPARK2_5_VQ_GEMM_CUH

#include <mma.h>

enum {
    SPARK_VQG_VECTOR = 8,      /* weights per codeword */
    SPARK_VQG_TILE_ROWS = 64,  /* output rows per block (4 x 16-row groups) */
    SPARK_VQG_TILE_OUT = 64,   /* output columns per block (2 x 32 per warp) */
    SPARK_VQG_TILE_K = 32,     /* input columns staged per step */
    SPARK_VQG_WARPS = 8,
};

struct spark_vq_stream {
    const uint8_t *gain, *main, *plane;
};

/* One 8-weight vector: the restore's arithmetic, no shortcuts. Host-callable
 * so a bind-time debug print and the kernel share one implementation. */
__host__ __device__ __forceinline__ void spark_vq_decode(
        const uint8_t *book, const uint8_t *book2, const uint8_t *res_book,
        const spark_vq_stream &s, const spark_vq_stream &s2,
        const uint8_t *res, uint32_t v, uint32_t o, float out[8]) {
    const uint64_t pair = (uint64_t)(v >> 1) * 3u;
    uint32_t index = (v & 1u)
        ? (uint32_t)(s.main[pair + 1] >> 4) | ((uint32_t)s.main[pair + 2] << 4)
        : (uint32_t)s.main[pair] | ((uint32_t)(s.main[pair + 1] & 15u) << 8);
    index |= ((uint32_t)(s.plane[v / 8u] >> (v % 8u)) & 1u) << 12;

    const float g1 = __half2float(((const __half *)s.gain)[o]);
    const uint32_t r8 = res ? res[v] : 0u;
    float g2 = 0.0f;
    uint32_t index2 = 0;
    if (s2.main) {
        index2 = (v & 1u)
            ? (uint32_t)(s2.main[pair + 1] >> 4) | ((uint32_t)s2.main[pair + 2] << 4)
            : (uint32_t)s2.main[pair] | ((uint32_t)(s2.main[pair + 1] & 15u) << 8);
        index2 |= ((uint32_t)(s2.plane[v / 8u] >> (v % 8u)) & 1u) << 12;
        g2 = __half2float(((const __half *)s2.gain)[o]);
    }
    for (uint32_t d = 0; d < 8u; d++) {
        __nv_fp8_e4m3 c1;
        c1.__x = book[index * 8u + d];
        float word = static_cast<float>(c1);
        if (res) {
            __nv_fp8_e4m3 r;
            r.__x = res_book[r8 * 8u + d];
            word += static_cast<float>(r);
        }
        float value = g1 * word;
        if (s2.main && book2) {
            __nv_fp8_e4m3 c2;
            c2.__x = book2[index2 * 8u + d];
            value += g2 * static_cast<float>(c2);
        }
        out[d] = value;
    }
}

/* out[rows][out_dim] = x[rows][in_dim] @ W[out_dim][in_dim]^T with W decoded
 * on the fly. Every thread block owns 64 output rows and 64 output columns,
 * and decodes its weight tile once per 64-row pass. */
__global__ void spark_vq_gemm_kernel(
        const uint8_t *book, const uint8_t *book2, const uint8_t *res_book,
        const spark_vq_stream main_stream, const spark_vq_stream stage2_stream,
        const uint8_t *res, uint32_t rows, uint32_t in_dim, uint32_t out_dim,
        const float *x, float *out) {
    using namespace nvcuda::wmma;
    const uint32_t in_vectors = in_dim / SPARK_VQG_VECTOR;
    const uint32_t out0 = blockIdx.x * SPARK_VQG_TILE_OUT;
    const uint32_t row0 = blockIdx.y * SPARK_VQG_TILE_ROWS;
    constexpr uint32_t KLD = SPARK_VQG_TILE_K + 8;  /* padded ld: a multiple
                                                     * of 8 for 16-bit wmma */

    __shared__ __nv_bfloat16 sw[SPARK_VQG_TILE_OUT][KLD];
    __shared__ __nv_bfloat16 sx[SPARK_VQG_TILE_ROWS][KLD];
    __shared__ float sc[SPARK_VQG_TILE_ROWS][SPARK_VQG_TILE_OUT];

    const uint32_t warp = threadIdx.x / 32u;
    const uint32_t row_group = warp / 2u;   /* 4 groups of 16 rows */
    const uint32_t col_group = warp % 2u;   /* 2 groups of 32 columns */

    fragment<accumulator, 16, 16, 16, float> acc[2];
    for (uint32_t i = 0; i < 2; i++) {
        fill_fragment(acc[i], 0.0f);
    }

    for (uint32_t k0 = 0; k0 < in_dim; k0 += SPARK_VQG_TILE_K) {
        /* Activation rows, zero past the row count. */
        for (uint32_t i = threadIdx.x; i < SPARK_VQG_TILE_ROWS * SPARK_VQG_TILE_K;
             i += blockDim.x) {
            const uint32_t r = i / SPARK_VQG_TILE_K;
            const uint32_t k = i % SPARK_VQG_TILE_K;
            const float v = (row0 + r) < rows
                ? x[(uint64_t)(row0 + r) * in_dim + k0 + k] : 0.0f;
            sx[r][k] = __float2bfloat16_rn(v);
        }
        /* One thread per 8-weight vector of the weight tile. */
        for (uint32_t i = threadIdx.x;
             i < SPARK_VQG_TILE_OUT * (SPARK_VQG_TILE_K / SPARK_VQG_VECTOR);
             i += blockDim.x) {
            const uint32_t n = i / (SPARK_VQG_TILE_K / SPARK_VQG_VECTOR);
            const uint32_t kv = i % (SPARK_VQG_TILE_K / SPARK_VQG_VECTOR);
            const uint32_t o = out0 + n;
            if (o >= out_dim) { continue; }
            const uint32_t v = o * in_vectors + k0 / SPARK_VQG_VECTOR + kv;
            float w[8];
            spark_vq_decode(book, book2, res_book, main_stream,
                            stage2_stream, res, v, o, w);
            for (uint32_t d = 0; d < 8u; d++) {
                sw[n][kv * 8u + d] = __float2bfloat16_rn(w[d]);
            }
        }
        __syncthreads();

        for (uint32_t kk = 0; kk < SPARK_VQG_TILE_K; kk += 16) {
            fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a;
            load_matrix_sync(a, &sx[row_group * 16u][kk], KLD);
            for (uint32_t i = 0; i < 2; i++) {
                /* B is W^T: sw[n][k] read column-major is (k, n). */
                fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b;
                load_matrix_sync(b, &sw[col_group * 32u + i * 16u][kk], KLD);
                mma_sync(acc[i], a, b, acc[i]);
            }
        }
        __syncthreads();
    }

    for (uint32_t i = 0; i < 2; i++) {
        store_matrix_sync(&sc[row_group * 16u][col_group * 32u + i * 16u],
                          acc[i], SPARK_VQG_TILE_OUT, mem_row_major);
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < SPARK_VQG_TILE_ROWS * SPARK_VQG_TILE_OUT;
         i += blockDim.x) {
        const uint32_t r = i / SPARK_VQG_TILE_OUT;
        const uint32_t c = i % SPARK_VQG_TILE_OUT;
        if ((row0 + r) < rows && (out0 + c) < out_dim) {
            out[(uint64_t)(row0 + r) * out_dim + out0 + c] = sc[r][c];
        }
    }
}

/* Fast decode GEMV (rows <= 4) over the fused VQ + FP8 E4M3 codebooks.
 * 8 warps per block; each warp computes 1 output column across `rows` tokens. */
__global__ void spark_vq_gemv_kernel(
        const uint8_t *book, const uint8_t *book2, const uint8_t *res_book,
        const spark_vq_stream main_stream, const spark_vq_stream stage2_stream,
        const uint8_t *res, uint32_t rows, uint32_t in_dim, uint32_t out_dim,
        const float *x, float *out) {
    const uint32_t warp = threadIdx.x / 32u;
    const uint32_t lane = threadIdx.x % 32u;
    const uint32_t o = blockIdx.x * 8u + warp;
    if (o >= out_dim) { return; }

    const uint32_t in_vectors = in_dim / SPARK_VQG_VECTOR;
    const uint32_t v0 = o * in_vectors;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint32_t kv = lane; kv < in_vectors; kv += 32u) {
        float w[8];
        spark_vq_decode(book, book2, res_book, main_stream, stage2_stream,
                        res, v0 + kv, o, w);
        const uint32_t k0 = kv * 8u;
        #pragma unroll
        for (uint32_t d = 0; d < 8u; d++) {
            w[d] = __bfloat162float(__float2bfloat16_rn(w[d]));
        }
        #pragma unroll
        for (uint32_t r = 0; r < 4u; r++) {
            if (r < rows) {
                const float4 x0 = ((const float4 *)(x + (uint64_t)r * in_dim + k0))[0];
                const float4 x1 = ((const float4 *)(x + (uint64_t)r * in_dim + k0))[1];
                acc[r] += w[0] * x0.x + w[1] * x0.y + w[2] * x0.z + w[3] * x0.w +
                          w[4] * x1.x + w[5] * x1.y + w[6] * x1.z + w[7] * x1.w;
            }
        }
    }

    #pragma unroll
    for (uint32_t r = 0; r < 4u; r++) {
        if (r < rows) {
            float sum = acc[r];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            }
            if (lane == 0u) {
                out[(uint64_t)r * out_dim + o] = sum;
            }
        }
    }
}

static int spark_vq_gemm_launch(
        const uint8_t *book, const uint8_t *book2, const uint8_t *res_book,
        const spark_vq_stream main_stream, const spark_vq_stream stage2_stream,
        const uint8_t *res, uint32_t rows, uint32_t in_dim, uint32_t out_dim,
        const float *x, float *out) {
    if (rows <= 4u) {
        const uint32_t blocks = (out_dim + 7u) / 8u;
        spark_vq_gemv_kernel<<<blocks, 256u, 0, ds4_current_stream()>>>(
            book, book2, res_book, main_stream, stage2_stream, res, rows,
            in_dim, out_dim, x, out);
        return cuda_ok(cudaGetLastError(), "Spark-X2.6 fused VQ GEMV");
    }
    const dim3 grid((out_dim + SPARK_VQG_TILE_OUT - 1u) / SPARK_VQG_TILE_OUT,
                    (rows + SPARK_VQG_TILE_ROWS - 1u) / SPARK_VQG_TILE_ROWS);
    spark_vq_gemm_kernel<<<grid, SPARK_VQG_WARPS * 32u, 0,
        ds4_current_stream()>>>(book, book2, res_book, main_stream,
        stage2_stream, res, rows, out_dim ? in_dim : in_dim, out_dim, x, out);
    return cuda_ok(cudaGetLastError(), "Spark-X2.6 fused VQ GEMM");
}

#endif
