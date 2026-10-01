/* Prism Bonsai (qwen35): the folded-weight activation transform on CUDA.
 * Included by the CUDA backend through ds4_qwen35_gpu.cuh so model residency,
 * streams and temporary allocations have the same lifetime as the other
 * CUDA paths.
 *
 * A folded Prism export stores the matmul weights in the rotated basis, so
 * the runtime rotates the activation instead:
 *
 *   forward:  a' = H_bs(s * a)     every folded matmul input
 *   inverse:  x  = s * H_bs(z)     token-embedding row lookups
 *
 * H_bs is the normalized Sylvester Walsh-Hadamard transform over blocks of
 * block_size consecutive values and s is the sign vector of the weight's
 * input width.  H is symmetric and orthogonal with H*H = I, which is why the
 * inverse only swaps the two steps.  ds4_hadamard_* in ds4.c is the
 * reference; these kernels are element-for-element the same butterfly.
 *
 * The gated delta-net output projection is folded over the grouped head order
 * instead of the tiled one, so its input is reordered from [hd][nk][rep] to
 * [hd][rep][nk] before the rotation (ds4_hadamard_gdn_permute). */

#pragma once

#include <stdint.h>

namespace qwen35_cuda {

/* Undo of the gdn reorder: element i of the grouped order comes from element
 * h + hd*(k + nk*r) of the tiled order, with i = h + hd*(r + rep*k).  It is a
 * bijection but, with nk != rep, not its own inverse, so only this explicit
 * map defines it (same map as ds4_hadamard_gdn_permute). */
__device__ __forceinline__ uint32_t gdn_src(uint32_t i, uint32_t hd, uint32_t nk, uint32_t rep) {
    const uint32_t h = i % hd;
    const uint32_t rest = i / hd;        /* r + rep*k */
    const uint32_t r = rest % rep;
    const uint32_t k = rest / rep;
    return h + hd * (k + nk * r);
}

/* One block of one row per CUDA block; blockDim.x must equal bs.  The butterfly
 * runs in shared memory: at each stage a thread takes its partner's value
 * before any write, so the two barriers per stage are what makes the exchange
 * safe.  Values stay per-thread registers between stages; the 10 stages of a
 * 1024-point transform are cheap next to the matmuls that follow. */
__global__ void fold_rotate(const float * __restrict__ src, float * __restrict__ dst,
                            const float * __restrict__ signs, uint32_t n, uint32_t bs,
                            int inverse) {
    extern __shared__ float tile[];
    const uint32_t tid = threadIdx.x;
    const uint32_t base = blockIdx.x * bs;
    const uint32_t i = base + tid;
    const float *row = src + (uint64_t) blockIdx.y * n;
    float *out = dst + (uint64_t) blockIdx.y * n;

    float v = row[i];
    if (signs && !inverse) v *= signs[i];
    tile[tid] = v;
    __syncthreads();

    for (uint32_t len = 1; len < bs; len <<= 1) {
        const float a = tile[tid];
        const float b = tile[tid ^ len];
        __syncthreads();
        tile[tid] = (tid & len) ? b - a : a + b;
        __syncthreads();
    }

    float r = tile[tid] * (1.0f / sqrtf((float) bs));
    if (signs && inverse) r *= signs[i];
    out[i] = r;
}

__global__ void fold_gdn_permute(float * __restrict__ dst, const float * __restrict__ src,
                                 uint32_t n, uint32_t hd, uint32_t nk, uint32_t rep) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    /* blockIdx.y is the row: a prefill chunk permutes hundreds of rows, and
     * one launch for the whole chunk costs the same GPU work as one launch per
     * row while removing hundreds of host launches per fold. */
    const uint64_t off = (uint64_t) blockIdx.y * n;
    dst[off + i] = src[off + gdn_src(i, hd, nk, rep)];
}

} // namespace qwen35_cuda
