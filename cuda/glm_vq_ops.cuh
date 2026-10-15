#ifndef DS4_GLM_VQ_OPS_CUH
#define DS4_GLM_VQ_OPS_CUH

#include "glm_vq.cuh"

/* Source arithmetic: Transformers 5.19.0 modeling_glm5_next.py,
 * lines 66-105, 286-337, 375-394 and 1300-1352. F32 graph buffers hold
 * BF16-rounded activations, except the explicit mHC/router F32 paths. */
enum glmvq_norm { GLMVQ_RMS_WEIGHTED, GLMVQ_RMS_GATED, GLMVQ_RMS_PLAIN };
enum { GLMVQ_HC = 4, GLMVQ_HC_MIX = GLMVQ_HC * (GLMVQ_HC + 2) };

__device__ __forceinline__ float glmvq_add(float a, float b) {
    float value;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(value) : "f"(a), "f"(b));
    return value;
}

__device__ __forceinline__ float glmvq_round(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float glmvq_sigmoid(float value) {
    return __fdiv_rn(1.0f, glmvq_add(1.0f, expf(-value)));
}

__global__ void glmvq_round_rows(float *values, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { values[i] = glmvq_round(values[i]); }
}

__global__ void glmvq_rms(const float *input, const __nv_bfloat16 *weight,
        const float *gate, float *output, uint32_t width, float eps, glmvq_norm kind) {
    const uint64_t base = (uint64_t)blockIdx.x * width;
    const uint32_t lane = threadIdx.x;
    float sum = 0.0f;
    for (uint32_t col = lane; col < width; col += blockDim.x) {
        const float x = input[base + col];
        sum = glmvq_add(sum, glmvq_mul(x, x));
    }
    __shared__ float partial[GLMVQ_THREADS];
    partial[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (lane < stride) { partial[lane] = glmvq_add(partial[lane], partial[lane + stride]); }
        __syncthreads();
    }
    const float variance = glmvq_add(__fdiv_rn(partial[0], (float)width), eps);
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(variance));
    for (uint32_t col = lane; col < width; col += blockDim.x) {
        float value = glmvq_mul(input[base + col], inv);
        if (kind == GLMVQ_RMS_PLAIN) {
            output[base + col] = value;
            continue;
        }
        // Ordinary RMS rounds before the weight; gated RMS rounds only last.
        if (kind == GLMVQ_RMS_WEIGHTED) { value = glmvq_round(value); }
        value = glmvq_mul(value, __bfloat162float(weight[col]));
        if (kind == GLMVQ_RMS_GATED) {
            value = glmvq_mul(value, glmvq_sigmoid(gate[base + col]));
        }
        output[base + col] = glmvq_round(value);
    }
}

__global__ void glmvq_swiglu(const float *gate, const float *up, float *output,
        uint64_t count, float limit) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const float g = fminf(gate[i], limit);
    const float u = fminf(fmaxf(up[i], -limit), limit);
    const float silu = glmvq_round(__fdiv_rn(g, glmvq_add(1.0f, expf(-g))));
    output[i] = glmvq_round(glmvq_mul(silu, u));
}

__device__ void glmvq_hc_weights(float *split, const float *mix,
        const float *scale, const float *base, uint32_t iters, float eps) {
    for (uint32_t h = 0; h < GLMVQ_HC; h++) {
        split[h] = glmvq_add(glmvq_sigmoid(glmvq_add(glmvq_mul(mix[h], scale[0]), base[h])), eps);
        const uint32_t p = GLMVQ_HC + h;
        split[p] = glmvq_mul(2.0f, glmvq_sigmoid(glmvq_add(glmvq_mul(mix[p], scale[1]), base[p])));
    }
    float *comb = split + 2 * GLMVQ_HC;
    for (uint32_t row = 0; row < GLMVQ_HC; row++) {
        float max_value = -INFINITY, sum = 0.0f;
        for (uint32_t col = 0; col < GLMVQ_HC; col++) {
            const uint32_t i = row * GLMVQ_HC + col;
            comb[i] = glmvq_add(glmvq_mul(mix[2 * GLMVQ_HC + i], scale[2]), base[2 * GLMVQ_HC + i]);
            max_value = fmaxf(max_value, comb[i]);
        }
        for (uint32_t col = 0; col < GLMVQ_HC; col++) {
            const uint32_t i = row * GLMVQ_HC + col;
            comb[i] = expf(glmvq_add(comb[i], -max_value));
            sum = glmvq_add(sum, comb[i]);
        }
        for (uint32_t col = 0; col < GLMVQ_HC; col++) {
            const uint32_t i = row * GLMVQ_HC + col;
            comb[i] = glmvq_add(__fdiv_rn(comb[i], sum), eps);
        }
    }
    // Source does the first column normalization before the remaining pairs.
    for (uint32_t step = 0; step < iters; step++) {
        if (step) {
            for (uint32_t row = 0; row < GLMVQ_HC; row++) {
                float sum = 0.0f;
                for (uint32_t col = 0; col < GLMVQ_HC; col++) {
                    sum = glmvq_add(sum, comb[row * GLMVQ_HC + col]);
                }
                sum = glmvq_add(sum, eps);
                for (uint32_t col = 0; col < GLMVQ_HC; col++) {
                    comb[row * GLMVQ_HC + col] = __fdiv_rn(comb[row * GLMVQ_HC + col], sum);
                }
            }
        }
        for (uint32_t col = 0; col < GLMVQ_HC; col++) {
            float sum = 0.0f;
            for (uint32_t row = 0; row < GLMVQ_HC; row++) {
                sum = glmvq_add(sum, comb[row * GLMVQ_HC + col]);
            }
            sum = glmvq_add(sum, eps);
            for (uint32_t row = 0; row < GLMVQ_HC; row++) {
                comb[row * GLMVQ_HC + col] = __fdiv_rn(comb[row * GLMVQ_HC + col], sum);
            }
        }
    }
}

__global__ void glmvq_hc_split(const float *mix, const float *residual,
        const float *scale, const float *base, float *output, float *split,
        uint32_t width, uint32_t iters, float eps) {
    const uint32_t row = blockIdx.x;
    __shared__ float coeff[GLMVQ_HC_MIX];
    if (threadIdx.x == 0) {
        glmvq_hc_weights(coeff, mix + (uint64_t)row * GLMVQ_HC_MIX, scale, base, iters, eps);
    }
    __syncthreads();
    if (threadIdx.x < GLMVQ_HC_MIX) {
        split[(uint64_t)row * GLMVQ_HC_MIX + threadIdx.x] = coeff[threadIdx.x];
    }
    for (uint32_t col = threadIdx.x; col < width; col += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t h = 0; h < GLMVQ_HC; h++) {
            sum = glmvq_add(sum, glmvq_mul(coeff[h], residual[((uint64_t)row * GLMVQ_HC + h) * width + col]));
        }
        output[(uint64_t)row * width + col] = glmvq_round(sum);
    }
}

__global__ void glmvq_hc_expand(const float *branch, const float *residual,
        const float *split, float *output, uint32_t width, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const uint32_t col = i % width, dst = (i / width) % GLMVQ_HC;
    const uint64_t row = i / (width * GLMVQ_HC);
    const float *coeff = split + row * GLMVQ_HC_MIX;
    float sum = 0.0f;
    for (uint32_t src = 0; src < GLMVQ_HC; src++) {
        const float c = glmvq_round(coeff[2 * GLMVQ_HC + src * GLMVQ_HC + dst]);
        sum = glmvq_add(sum, glmvq_mul(c, residual[(row * GLMVQ_HC + src) * width + col]));
    }
    const float part = glmvq_round(glmvq_mul(glmvq_round(coeff[GLMVQ_HC + dst]), branch[row * width + col]));
    output[i] = glmvq_round(glmvq_add(part, glmvq_round(sum)));
}

#endif
