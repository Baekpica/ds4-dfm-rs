#ifndef DS4_GLM53_VISION_NORM_CUH
#define DS4_GLM53_VISION_NORM_CUH

__device__ __forceinline__ static float glm53_vision_bf16(
        const uint16_t *p) {
    return __uint_as_float((uint32_t)(*p) << 16);
}

__device__ __forceinline__ static float glm53_vision_erf_approx(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float a = fabsf(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float p = (((((1.061405429f * t - 1.453152027f) * t) +
                       1.421413741f) * t - 0.284496736f) * t +
                       0.254829592f) * t;
    return sign * (1.0f - p * expf(-a * a));
}

__global__ static void glm53_vision_layernorm_gelu_kernel(
        float          *out,
        const float    *x,
        const uint16_t *weight,
        const uint16_t *bias,
        uint32_t        width,
        float           eps) {
    __shared__ float partial[256];
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const float *xr = x + (uint64_t)row * width;
    float *yr = out + (uint64_t)row * width;
    float sum = 0.0f;
    for (uint32_t d = tid; d < width; d += blockDim.x) sum += xr[d];
    partial[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float mean = partial[0] / (float)width;
    /* Every warp must consume the mean before variance reuses partial[0]. */
    __syncthreads();
    float var = 0.0f;
    for (uint32_t d = tid; d < width; d += blockDim.x) {
        const float centered = xr[d] - mean;
        var = fmaf(centered, centered, var);
    }
    partial[tid] = var;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / (float)width + eps);
    const float inv_sqrt2 = 0.7071067811865475f;
    for (uint32_t d = tid; d < width; d += blockDim.x) {
        float value = (xr[d] - mean) * inv * glm53_vision_bf16(weight + d) +
                      glm53_vision_bf16(bias + d);
        yr[d] = 0.5f * value *
                (1.0f + glm53_vision_erf_approx(value * inv_sqrt2));
    }
}

#endif
