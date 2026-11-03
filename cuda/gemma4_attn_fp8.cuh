#ifndef DS4_GEMMA4_ATTN_FP8_CUH
#define DS4_GEMMA4_ATTN_FP8_CUH

/* Causal attention over the combined FP8 full-layer rows, the optional
 * counterpart of cuda/gemma4_attn.cuh. The geometry is the same (grouped KV,
 * an optional sliding window, float32 softmax); the row is the Solar FP8
 * layout the Beam and MiMo full layers already use:
 *
 *     E4M3 K [kv_heads][dim] | E4M3 V [kv_heads][dim] |
 *     f16 K scale [kv_heads] | f16 V scale [kv_heads]
 *
 * The K scale multiplies the dot after the warp sum and the V scale
 * multiplies the probability before the P.V accumulation, exactly as
 * ds4_beam_gpu.cuh:795-880 does. E4M3 decode is the hardware conversion, the
 * same one beam_e4m3x4 uses, so a code round-trips exactly. */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ static float g4_fp8_e4m3(uint8_t code) {
    const __half2 pair = __half2(__nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)code, __NV_E4M3));
    return __low2float(pair);
}

__device__ __forceinline__ static float g4_fp8_warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ static void g4_attn_fp8_kernel(float *out, const float *query,
        const uint8_t *kv, uint32_t rows, uint32_t first, uint32_t heads,
        uint32_t kv_heads, uint32_t dim, uint32_t window, uint32_t ring,
        uint32_t keys, uint64_t row_bytes) {
    const uint32_t warp = threadIdx.x / 32u;
    const uint32_t lane = threadIdx.x % 32u;
    const uint64_t job = (uint64_t)blockIdx.x * (blockDim.x / 32u) + warp;
    if (job >= (uint64_t)rows * heads) { return; }
    const uint32_t row = (uint32_t)(job / heads);
    const uint32_t head = (uint32_t)(job % heads);
    const uint32_t group = heads / kv_heads;
    const uint32_t kv_head = head / group;
    const uint32_t position = first + row;
    const float *q = query + job * dim;
    float *o = out + job * dim;
    const uint64_t kv_dim = (uint64_t)kv_heads * dim;

    const uint32_t per_lane = (dim + 31u) / 32u;
    float acc[16];
    float qv[16];
    for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
        const uint32_t d = lane + i * 32u;
        qv[i] = d < dim ? q[d] : 0.0f;
        acc[i] = 0.0f;
    }
    float maximum = -INFINITY, total = 0.0f;
    const uint32_t begin = window && position + 1 > window ? position + 1 - window : 0u;
    const uint32_t last = keys && keys <= position ? keys - 1u : position;
    for (uint32_t k = begin; k <= last; k++) {
        const uint64_t slot = ring ? (uint64_t)(k % ring) : k;
        const uint8_t *base = kv + slot * row_bytes;
        const uint8_t *krow = base + (uint64_t)kv_head * dim;
        const uint8_t *vrow = base + kv_dim + (uint64_t)kv_head * dim;
        const __half *scales = (const __half *)(base + 2u * kv_dim);
        const float kscale = __half2float(scales[kv_head]);
        const float vscale = __half2float(scales[kv_heads + kv_head]);
        float dot = 0.0f;
        for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
            const uint32_t d = lane + i * 32u;
            if (d < dim) { dot += qv[i] * g4_fp8_e4m3(krow[d]); }
        }
        dot = g4_fp8_warp_sum(dot);
        dot = __shfl_sync(0xffffffffu, dot, 0) * kscale;
        const float m_new = fmaxf(maximum, dot);
        const float alpha = expf(maximum - m_new);
        const float p = expf(dot - m_new);
        total = total * alpha + p;
        maximum = m_new;
        for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
            const uint32_t d = lane + i * 32u;
            if (d < dim) {
                acc[i] = acc[i] * alpha + p * vscale * g4_fp8_e4m3(vrow[d]);
            }
        }
    }
    const float inverse = 1.0f / total;
    for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
        const uint32_t d = lane + i * 32u;
        if (d < dim) { o[d] = acc[i] * inverse; }
    }
}

#endif
