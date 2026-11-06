#ifndef DS4_GEMMA_VQ_DECODE_CUH
#define DS4_GEMMA_VQ_DECODE_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "glm_vq.cuh"

/* Gemma-4 batch-1 GVQ1 GEMV (decode).
 *
 * Why this header exists: the shared decode loader in
 * cuda/glm_vq_decode.cuh:154 requires every row to be whole 256-input
 * k-steps and, with a residual stage, whole 2048-input blocks; its partial
 * row branch (line 157) requires a row under 2048 inputs to be a multiple of
 * 256 and to have no residual. Gemma's routed expert rows are
 *   gate/up  704 rows x 2816 cols  (G4_EMBED = 2816, ds4_gemma4_plan.h:22)
 *   down    2816 rows x  704 cols  (G4_FF = 704, ds4_gemma4_plan.h:40)
 * 704 = 2*256 + 192 is not a whole k-step, and 2816 = 2048 + 3*256 is not a
 * whole 2048 block, so the R13+8 residual stage cannot use the shared loader
 * either. The GLM headers are shared and stay untouched; this is the
 * Gemma-only path.
 *
 * The arithmetic is the runtime's existing restore, not a new design:
 *   - index packing, plane bit and gain lookup: cuda/glm_vq.cuh:120-129
 *     (glmvq_restore), the same bytes cuda/glm_vq.cuh:112-118 describes;
 *   - residual: one index byte per vector into a K=256 E4M3 book, added to
 *     the main codeword before the gain (cuda/glm_vq.cuh:112-118);
 *   - BF16 boundary: glmvq_restore writes bf16 weights and glmvq_gemm feeds
 *     bf16 activations to a CUBLAS_COMPUTE_32F GEMM
 *     (cuda/glm_vq_linear.cuh:97-110), so this kernel rounds the weight and
 *     the activation to bf16 and accumulates in f32.
 *
 * The kernel reads only the VQ bytes (about 1.6 bits per weight) instead of
 * restoring the matrix to bf16 and calling cuBLAS, which is the decode cost
 * the port measured at 10.7 tok/s against the sibling families' 15-25.
 *
 * Variants and launch widths are diagnostic switches
 * (DS4_GEMMA4_VQ_GEMV_VARIANT, DS4_GEMMA4_VQ_GEMV_THREADS); variant 0 is the
 * runtime's own heuristic. */

enum g4_vq_variant { G4_VQ_AUTO = 0, G4_VQ_WARP = 1, G4_VQ_BLOCK = 2 };
enum { G4_VQ_VECTOR = GLMVQ_VECTOR };

/* The restore's rounding, kept as one named operation so the kernel and the
 * reference cannot drift: w = bf16(gain * word), x = bf16(x). */
__device__ __forceinline__ float g4_vq_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

/* Index i of the GVQ1 v1 stream: bits 12*i..12*i+11, least significant bit
 * first, packed as two 12-bit indices into three bytes; the 13th bit (nc
 * 8192 only) lives in the plane at bit i%8 of byte i/8. This is
 * glmvq_restore's own expression (cuda/glm_vq.cuh:120-129), with `vector`
 * kept flat across the matrix exactly as the restore does. */
__device__ __forceinline__ uint32_t g4_vq_index(const glmvq_matrix &m, uint64_t vector) {
    const uint64_t pair = vector / 2 * 3;
    uint32_t index = (vector & 1u)
        ? (uint32_t)(m.main[pair + 1] >> 4) | ((uint32_t)m.main[pair + 2] << 4)
        : (uint32_t)m.main[pair] | ((uint32_t)(m.main[pair + 1] & 15u) << 8);
    if (m.plane) {
        index |= ((uint32_t)(m.plane[vector / 8] >> (vector % 8)) & 1u) << 12;
    }
    return index;
}

/* One 8-weight vector dotted with its 8 activations. `col` is the vector's
 * first column. The residual index and the row gain come from the same flat
 * `vector` index the restore uses. */
__device__ __forceinline__ float g4_vq_dot_vector(const glmvq_matrix &m, uint64_t vector,
        uint32_t index, const float *x, uint32_t col) {
    const uint32_t row = (uint32_t)(vector / (m.cols / G4_VQ_VECTOR));
    const float gain = __half2float(m.gain[row]);
    const uint32_t res = m.res ? m.res[vector] : 0u;
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t d = 0; d < G4_VQ_VECTOR; d++) {
        const uint32_t i = index * G4_VQ_VECTOR + d;
        float word = m.precision == GLMVQ_E4M3
            ? glmvq_e4m3(((const uint8_t *)m.book)[i])
            : __half2float(((const __half *)m.book)[i]);
        if (m.res) {
            word += glmvq_e4m3(m.res_book[res * G4_VQ_VECTOR + d]);
        }
        const float w = g4_vq_bf16(glmvq_mul(gain, word));
        const float xv = g4_vq_bf16(x[col + d]);
        acc = fmaf(w, xv, acc);
    }
    return acc;
}

/* Variant 1: one warp per row, WARPS rows per block. A row of 704 inputs is
 * 88 vectors, so one warp keeps every lane busy for three vectors; a row of
 * 2816 inputs is 352 vectors, eleven per lane. */
template <uint32_t WARPS>
__global__ void g4_vq_gemv_warp_kernel(glmvq_matrix m, const float *x, float *y) {
    const uint32_t warp = threadIdx.x / 32u, lane = threadIdx.x % 32u;
    const uint32_t row = blockIdx.x * WARPS + warp;
    if (row >= m.rows) { return; }
    const uint32_t nvec = m.cols / G4_VQ_VECTOR;
    const uint64_t base = (uint64_t)row * nvec;
    float acc = 0.0f;
    for (uint32_t j = lane; j < nvec; j += 32u) {
        const uint64_t vector = base + j;
        acc += g4_vq_dot_vector(m, vector, g4_vq_index(m, vector), x, j * G4_VQ_VECTOR);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) { y[row] = acc; }
}

/* Variant 2: one block per row. `THREADS` threads split the row's vectors and
 * reduce through shared memory. This is the shape a wide row wants when a
 * warp cannot cover its vectors in one pass. */
template <uint32_t THREADS>
__global__ void g4_vq_gemv_block_kernel(glmvq_matrix m, const float *x, float *y) {
    const uint32_t row = blockIdx.x;
    const uint32_t nvec = m.cols / G4_VQ_VECTOR;
    const uint64_t base = (uint64_t)row * nvec;
    float acc = 0.0f;
    for (uint32_t j = threadIdx.x; j < nvec; j += THREADS) {
        const uint64_t vector = base + j;
        acc += g4_vq_dot_vector(m, vector, g4_vq_index(m, vector), x, j * G4_VQ_VECTOR);
    }
    __shared__ float sums[THREADS / 32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    const uint32_t warp = threadIdx.x / 32u;
    if ((threadIdx.x % 32u) == 0) { sums[warp] = acc; }
    __syncthreads();
    if (warp == 0) {
        acc = (threadIdx.x < THREADS / 32) ? sums[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (threadIdx.x == 0) { y[row] = acc; }
    }
}

/* Whether a variant and launch width can run this geometry. The kernel reads
 * the same GVQ1 bytes the restore does, so the only constraints are the
 * layout's own (cols a multiple of 64, dim 8) and the launch's. */
static int g4_vq_gemv_ready(const glmvq_matrix &m, int variant, int threads) {
    if (!m.main || !m.book || !m.gain || !m.rows || !m.cols) { return 0; }
    if ((m.cols % 64u) || (m.cols % G4_VQ_VECTOR)) { return 0; }
    if (m.precision != GLMVQ_E4M3 && m.precision != GLMVQ_F16) { return 0; }
    if (variant != G4_VQ_AUTO && variant != G4_VQ_WARP && variant != G4_VQ_BLOCK) { return 0; }
    if (threads < 32 || threads > 512 || (threads % 32)) { return 0; }
    return 1;
}

/* Launch one batch-1 GEMV. `y` has m.rows floats; `x` has m.cols floats. */
static int g4_vq_gemv(const glmvq_matrix &m, const float *x, float *y, int variant,
        int threads, cudaStream_t stream) {
    if (!x || !y || !g4_vq_gemv_ready(m, variant, threads)) { return 0; }
    if (variant == G4_VQ_AUTO) {
        /* 704 columns is 88 vectors: one warp covers it in three passes and
         * keeps eight rows per block. 2816 columns is 352 vectors: a block
         * per row avoids 32-way serialization on the wider row. */
        variant = m.cols <= 1024u ? G4_VQ_WARP : G4_VQ_BLOCK;
    }
    if (variant == G4_VQ_WARP) {
        const uint32_t warps = (uint32_t)threads / 32u;
        const uint32_t blocks = (m.rows + warps - 1u) / warps;
        switch (threads) {
            case 128: g4_vq_gemv_warp_kernel<4><<<blocks, 128, 0, stream>>>(m, x, y); break;
            case 256: g4_vq_gemv_warp_kernel<8><<<blocks, 256, 0, stream>>>(m, x, y); break;
            case 512: g4_vq_gemv_warp_kernel<16><<<blocks, 512, 0, stream>>>(m, x, y); break;
            default: return 0;
        }
    } else {
        switch (threads) {
            case 128: g4_vq_gemv_block_kernel<128><<<m.rows, 128, 0, stream>>>(m, x, y); break;
            case 256: g4_vq_gemv_block_kernel<256><<<m.rows, 256, 0, stream>>>(m, x, y); break;
            case 512: g4_vq_gemv_block_kernel<512><<<m.rows, 512, 0, stream>>>(m, x, y); break;
            default: return 0;
        }
    }
    return cudaGetLastError() == cudaSuccess;
}

#endif
