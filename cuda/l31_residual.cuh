#ifndef DS4_L31_RESIDUAL_CUH
#define DS4_L31_RESIDUAL_CUH

#include "glm_vq_ops.cuh"

/* Residual at the BF16 boundary and the next weighted RMS norm in one pass:
 * cur = bf16(cur + branch) (add_kernel, glmvq_round_rows), then glmvq_rms's
 * lanes, load batch and reduction tree over the rounded row held in
 * registers, so cur and norm equal the three separate kernels bit for bit.
 * A decode row is latency-bound: the weights load before the PDL wait, and
 * the tree's last five levels run as warp-0 shuffles (same pairs and order)
 * instead of five more block barriers. N is F32 or BF16 norm rows (the
 * values are BF16 either way). */
template <typename N>
__global__ void l31_residual_rms(float *cur, const float *branch, const __nv_bfloat16 *weight,
        N *norm, uint32_t width, float eps) {
    glmvq_pdl_open();
    const uint64_t base = (uint64_t)blockIdx.x * width;
    const uint32_t lane = threadIdx.x;
    __nv_bfloat16 w[GLMVQ_RMS_BATCH];
    #pragma unroll
    for (uint32_t j = 0; j < GLMVQ_RMS_BATCH; j++) {
        const uint32_t col = lane + j * blockDim.x;
        w[j] = col < width ? weight[col] : __float2bfloat16_rn(0.0f);
    }
    glmvq_pdl_wait();
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
    for (uint32_t stride = blockDim.x / 2; stride >= 32; stride /= 2) {
        if (lane < stride) { partial[lane] = glmvq_add(partial[lane], partial[lane + stride]); }
        __syncthreads();
    }
    if (lane < 32) {
        float total = partial[lane];
        for (uint32_t stride = 16; stride; stride /= 2) {
            total = glmvq_add(total, __shfl_down_sync(0xffffffffu, total, stride));
        }
        if (!lane) { partial[0] = total; }
    }
    __syncthreads();
    const float variance = glmvq_add(__fdiv_rn(partial[0], (float)width), eps);
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(variance));
    #pragma unroll
    for (uint32_t j = 0; j < GLMVQ_RMS_BATCH; j++) {
        const uint32_t col = lane + j * blockDim.x;
        // glmvq_rms_value's weighted form: round, weight, round.
        if (col < width) {
            const float value = glmvq_mul(glmvq_round(glmvq_mul(x[j], inv)), __bfloat162float(w[j]));
            norm[base + col] = glmvq_cast<N>(glmvq_round(value));
        }
    }
}

#endif
