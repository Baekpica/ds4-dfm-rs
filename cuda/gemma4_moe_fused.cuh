#ifndef DS4_GEMMA4_MOE_FUSED_CUH
#define DS4_GEMMA4_MOE_FUSED_CUH

/* Gemma-4 batch-1 fused expert gate/up: one GEMV reads both the gate and the
 * up matrix of an expert and writes gelu(gate) * up in the epilogue, instead
 * of two GEMVs into two buffers and a separate elementwise pass. The
 * technique is llama.cpp's fused gate/up MMQ write-back (commit c49edf096,
 * "CUDA: fuse FFN gate/up matmuls and GLU in MMQ with interleaved gate/up
 * rows"), adapted to the GVQ1 bytes this runtime reads.
 *
 * The arithmetic is not new: both dot products call g4_vq_dot_vector
 * (cuda/gemma_vq_decode.cuh:74-93), the same restore and accumulation the
 * separate path uses, and the epilogue calls g4_gelu and g4_round
 * (cuda/gemma4_primitives.cuh:19-32), the same two roundings as
 * g4_gelu_kernel. The fused and unfused paths are therefore expected to be
 * bit-identical; the option exists to measure the launch and traffic
 * savings, not to change the result. */

#include "gemma_vq_decode.cuh"
#include "gemma4_primitives.cuh"

/* Variant 1: one warp per row, both matrices' vectors walked in one pass. */
template <uint32_t WARPS>
__global__ void g4_vq_gemv_glu_warp_kernel(glmvq_matrix gate, glmvq_matrix up,
        const float *x, float *y) {
    const uint32_t warp = threadIdx.x / 32u, lane = threadIdx.x % 32u;
    const uint32_t row = blockIdx.x * WARPS + warp;
    if (row >= gate.rows) { return; }
    const uint32_t nvec = gate.cols / G4_VQ_VECTOR;
    const uint64_t base = (uint64_t)row * nvec;
    float g = 0.0f, u = 0.0f;
    for (uint32_t j = lane; j < nvec; j += 32u) {
        const uint64_t vector = base + j;
        const uint32_t col = j * G4_VQ_VECTOR;
        g += g4_vq_dot_vector(gate, vector, g4_vq_index(gate, vector), x, col);
        u += g4_vq_dot_vector(up, vector, g4_vq_index(up, vector), x, col);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        g += __shfl_down_sync(0xffffffffu, g, off);
        u += __shfl_down_sync(0xffffffffu, u, off);
    }
    if (lane == 0) { y[row] = g4_round(g4_gelu(g) * g4_round(u)); }
}

/* Variant 2: one block per row, both reductions through shared memory. */
template <uint32_t THREADS>
__global__ void g4_vq_gemv_glu_block_kernel(glmvq_matrix gate, glmvq_matrix up,
        const float *x, float *y) {
    const uint32_t row = blockIdx.x;
    const uint32_t nvec = gate.cols / G4_VQ_VECTOR;
    const uint64_t base = (uint64_t)row * nvec;
    float g = 0.0f, u = 0.0f;
    for (uint32_t j = threadIdx.x; j < nvec; j += THREADS) {
        const uint64_t vector = base + j;
        const uint32_t col = j * G4_VQ_VECTOR;
        g += g4_vq_dot_vector(gate, vector, g4_vq_index(gate, vector), x, col);
        u += g4_vq_dot_vector(up, vector, g4_vq_index(up, vector), x, col);
    }
    __shared__ float gs[THREADS / 32], us[THREADS / 32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        g += __shfl_down_sync(0xffffffffu, g, off);
        u += __shfl_down_sync(0xffffffffu, u, off);
    }
    const uint32_t warp = threadIdx.x / 32u;
    if ((threadIdx.x % 32u) == 0) { gs[warp] = g; us[warp] = u; }
    __syncthreads();
    if (warp == 0) {
        g = (threadIdx.x < THREADS / 32) ? gs[threadIdx.x] : 0.0f;
        u = (threadIdx.x < THREADS / 32) ? us[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            g += __shfl_down_sync(0xffffffffu, g, off);
            u += __shfl_down_sync(0xffffffffu, u, off);
        }
        if (threadIdx.x == 0) { y[row] = g4_round(g4_gelu(g) * g4_round(u)); }
    }
}

/* Both matrices must share a geometry and pass the single-GEMV checks. */
static int g4_vq_gemv_glu_ready(const glmvq_matrix &gate, const glmvq_matrix &up,
        int variant, int threads) {
    if (gate.rows != up.rows || gate.cols != up.cols) { return 0; }
    if (!g4_vq_gemv_ready(gate, variant, threads) ||
        !g4_vq_gemv_ready(up, variant, threads)) { return 0; }
    return 1;
}

/* Launch one fused batch-1 gate/up GEMV. `y` has gate.rows floats; `x` has
 * gate.cols floats. */
static int g4_vq_gemv_glu(const glmvq_matrix &gate, const glmvq_matrix &up,
        const float *x, float *y, int variant, int threads, cudaStream_t stream) {
    if (!x || !y || !g4_vq_gemv_glu_ready(gate, up, variant, threads)) { return 0; }
    if (variant == G4_VQ_AUTO) {
        variant = gate.cols <= 1024u ? G4_VQ_WARP : G4_VQ_BLOCK;
    }
    if (variant == G4_VQ_WARP) {
        const uint32_t warps = (uint32_t)threads / 32u;
        const uint32_t blocks = (gate.rows + warps - 1u) / warps;
        switch (threads) {
            case 128: g4_vq_gemv_glu_warp_kernel<4><<<blocks, 128, 0, stream>>>(gate, up, x, y); break;
            case 256: g4_vq_gemv_glu_warp_kernel<8><<<blocks, 256, 0, stream>>>(gate, up, x, y); break;
            case 512: g4_vq_gemv_glu_warp_kernel<16><<<blocks, 512, 0, stream>>>(gate, up, x, y); break;
            default: return 0;
        }
    } else {
        switch (threads) {
            case 128: g4_vq_gemv_glu_block_kernel<128><<<gate.rows, 128, 0, stream>>>(gate, up, x, y); break;
            case 256: g4_vq_gemv_glu_block_kernel<256><<<gate.rows, 256, 0, stream>>>(gate, up, x, y); break;
            case 512: g4_vq_gemv_glu_block_kernel<512><<<gate.rows, 512, 0, stream>>>(gate, up, x, y); break;
            default: return 0;
        }
    }
    return cudaGetLastError() == cudaSuccess;
}

#endif
