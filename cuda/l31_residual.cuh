#ifndef DS4_L31_RESIDUAL_CUH
#define DS4_L31_RESIDUAL_CUH

#include "glm_vq_ops.cuh"

/* Residual at the BF16 boundary and the next weighted RMS norm in one pass:
 * cur = bf16(cur + branch) (add_kernel, glmvq_round_rows), then glmvq_rms's
 * lanes, load batch and reduction tree over the rounded row held in
 * registers, so cur and norm equal the three separate kernels bit for bit. */
__global__ void l31_residual_rms(float *cur, const float *branch, const __nv_bfloat16 *weight,
        float *norm, uint32_t width, float eps) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint64_t base = (uint64_t)blockIdx.x * width;
    const uint32_t lane = threadIdx.x;
    float x[GLMVQ_RMS_BATCH], sum = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < GLMVQ_RMS_BATCH; j++) {
        const uint32_t col = lane + j * blockDim.x;
        x[j] = col < width ? glmvq_round(cur[base + col] + branch[base + col]) : 0.0f;
    }
    #pragma unroll
    for (uint32_t j = 0; j < GLMVQ_RMS_BATCH; j++) {
        const uint32_t col = lane + j * blockDim.x;
        if (col < width) {
            cur[base + col] = x[j];
            sum = glmvq_add(sum, glmvq_mul(x[j], x[j]));
        }
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
    #pragma unroll
    for (uint32_t j = 0; j < GLMVQ_RMS_BATCH; j++) {
        const uint32_t col = lane + j * blockDim.x;
        if (col < width) {
            norm[base + col] = glmvq_rms_value(x[j], inv, weight, nullptr, base + col, col, GLMVQ_RMS_WEIGHTED);
        }
    }
}

#endif
