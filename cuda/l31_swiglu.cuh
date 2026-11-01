#ifndef DS4_L31_SWIGLU_CUH
#define DS4_L31_SWIGLU_CUH

#include "glm_vq_q4p.cuh"

// Keep the SwiGLU BF16 boundary, then write q4p's permuted FP16 operand.
__device__ __forceinline__ float l31_swiglu_pack(const float *gate, const float *up,
        __half *out, uint32_t cols, float limit, float scale) {
    float peak = 0.0f;
    for (uint32_t i = threadIdx.x; i < cols / 4; i += GLMVQ_Q4P_PREP) {
        const float4 g = ((const float4 *)gate)[i], u = ((const float4 *)up)[i];
        const float v[4] = {glmvq_swiglu_value<GLMVQ_CLAMP_SILU>(g.x, u.x, limit),
            glmvq_swiglu_value<GLMVQ_CLAMP_SILU>(g.y, u.y, limit),
            glmvq_swiglu_value<GLMVQ_CLAMP_SILU>(g.z, u.z, limit),
            glmvq_swiglu_value<GLMVQ_CLAMP_SILU>(g.w, u.w, limit)};
        peak = fmaxf(peak, fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))));
        const __half h0 = __float2half_rn(v[0] * scale), h1 = __float2half_rn(v[1] * scale);
        const __half h2 = __float2half_rn(v[2] * scale), h3 = __float2half_rn(v[3] * scale);
        ((uint2 *)out)[i] = make_uint2(glmvq_q4p_h2(h0, h2), glmvq_q4p_h2(h1, h3));
    }
    return peak;
}

__global__ void __launch_bounds__(GLMVQ_Q4P_PREP) l31_swiglu_prepare(const float *gate, const float *up,
        __half *xh, float *xscale, uint32_t cols, float limit) {
    __shared__ float peaks[GLMVQ_Q4P_PREP / 32];
    const uint64_t base = (uint64_t)blockIdx.x * cols;
    float peak = l31_swiglu_pack(gate + base, up + base, xh + base, cols, limit, 1.0f);
    for (uint32_t s = 16; s; s >>= 1) { peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, s)); }
    if (!(threadIdx.x & 31)) { peaks[threadIdx.x / 32] = peak; }
    __syncthreads();
    peak = fmaxf(fmaxf(peaks[0], peaks[1]), fmaxf(peaks[2], peaks[3]));
    int e = 0;
    if (peak > GLMVQ_Q4P_RANGE) {
        frexpf(peak / GLMVQ_Q4P_RANGE, &e);
        l31_swiglu_pack(gate + base, up + base, xh + base, cols, limit, ldexpf(1.0f, -e));
    }
    if (!threadIdx.x) { xscale[blockIdx.x] = ldexpf(1.0f, e - GLMVQ_Q4P_SHIFT); }
}

#endif
