#pragma once
#include "vecdotq.cuh"

enum { GLM_SHARED_K = 4096, GLM_SHARED_M = 2048, GLM_SHARED_WARPS = 4 };
static constexpr float GLM_SHARED_CLAMP = 10.0f;

// Preserve the raw MMVQ's Q8_1 scales, per-thread dots and four-warp tree.
// Pairing only removes duplicate quantization and gate/up materialization.
static __global__ void glm53_shared_q8(const void *gate, const void *up,
        const block_q8_1 *x, float *mid) {
    constexpr int lanes = 32;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int parts = QI8_0 / vdr;
    constexpr int stride = GLM_SHARED_WARPS * lanes / parts;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * lanes + lane;
    const int row = blockIdx.x * (GLM_SHARED_K / QK8_0);
    const int qs = vdr * (tid % parts);
    float g = 0.0f, u = 0.0f;
    for (int b = tid / parts; b < GLM_SHARED_K / QK8_0; b += stride) {
        g += vec_dot_q8_0_q8_1(gate, x + b, row + b, qs);
        u += vec_dot_q8_0_q8_1(up, x + b, row + b, qs);
    }

    __shared__ float partial[2][GLM_SHARED_WARPS - 1][lanes];
    if (warp > 0) {
        partial[0][warp - 1][lane] = g;
        partial[1][warp - 1][lane] = u;
    }
    __syncthreads();
    if (warp > 0) { return; }

#pragma unroll
    for (int w = 0; w < GLM_SHARED_WARPS - 1; w++) {
        g += partial[0][w][lane];
        u += partial[1][w][lane];
    }
    g = warp_reduce_sum<lanes>(g);
    u = warp_reduce_sum<lanes>(u);
    if (lane != 0) { return; }

    // The old matmuls sanitize before GLM's asymmetric clamp and SwiGLU.
    if (!isfinite(g)) { g = 0.0f; }
    if (!isfinite(u)) { u = 0.0f; }
    g = fminf(g, GLM_SHARED_CLAMP);
    u = fminf(fmaxf(u, -GLM_SHARED_CLAMP), GLM_SHARED_CLAMP);
    mid[blockIdx.x] = (g / (1.0f + expf(-g))) * u;
}
