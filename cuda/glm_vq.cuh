#ifndef DS4_GLM_VQ_CUH
#define DS4_GLM_VQ_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

/* GVQ1 v1: FORMAT.md at HF fe2a446e774624ef8d397136d81524c61db77f65.
 * The Rust loader validates geometry, finite values and all spans first.
 * This bounded restore is the numerical fallback; callers supply scratch
 * for selected matrices only, never a resident BF16 copy of all experts. */
enum { GLMVQ_VECTOR = 8, GLMVQ_THREADS = 256, GLMVQ_SCALE_BLOCK = 128 };
enum glmvq_precision { GLMVQ_E4M3 = 1, GLMVQ_F16 = 2 };

struct glmvq_matrix {
    const uint8_t *main;
    const uint8_t *plane;
    const void *book;
    const __half *gain;
    uint32_t rows;
    uint32_t cols;
    glmvq_precision precision;
};

__device__ __forceinline__ float glmvq_e4m3(uint8_t bits) {
    __nv_fp8_e4m3 value;
    value.__x = bits;
    return static_cast<float>(value);
}

__device__ __forceinline__ float glmvq_mul(float a, float b) {
    // NVCC's global fast-math setting must not flush legal F32 scale values.
    float value;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(value) : "f"(a), "f"(b));
    return value;
}

template<typename T> __device__ __forceinline__ T glmvq_cast(float value);
template<> __device__ __forceinline__ float glmvq_cast<float>(float value) {
    return value;
}
template<> __device__ __forceinline__ __nv_bfloat16 glmvq_cast<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template<typename T>
__global__ void glmvq_restore(glmvq_matrix m, T *out) {
    const uint64_t vector = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t count = (uint64_t)m.rows * m.cols / GLMVQ_VECTOR;
    if (vector >= count) { return; }

    const uint64_t pair = vector / 2 * 3;
    uint32_t index;
    if ((vector & 1u) == 0) {
        index = m.main[pair] | ((uint32_t)(m.main[pair + 1] & 15u) << 8);
    } else {
        index = (m.main[pair + 1] >> 4) | ((uint32_t)m.main[pair + 2] << 4);
    }
    if (m.plane) {
        index |= ((m.plane[vector / 8] >> (vector % 8)) & 1u) << 12;
    }
    const float gain = __half2float(m.gain[vector / (m.cols / GLMVQ_VECTOR)]);
    for (uint32_t d = 0; d < GLMVQ_VECTOR; d++) {
        const uint32_t i = index * GLMVQ_VECTOR + d;
        const float word = m.precision == GLMVQ_E4M3
            ? glmvq_e4m3(((const uint8_t *)m.book)[i])
            : __half2float(((const __half *)m.book)[i]);
        /* The reference first multiplies in FP32, then rounds to BF16 at
         * the GEMM boundary. Moving the gain outside GEMM changes that. */
        out[vector * GLMVQ_VECTOR + d] = glmvq_cast<T>(glmvq_mul(gain, word));
    }
}

template<typename T>
__global__ void glmvq_fp8_restore(const uint8_t *weight, const float *scales,
        uint32_t rows, uint32_t cols, T *out) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * cols) { return; }
    const uint64_t row = i / cols, col = i % cols;
    const float scale = scales[(row / GLMVQ_SCALE_BLOCK) * (cols / GLMVQ_SCALE_BLOCK)
                              + col / GLMVQ_SCALE_BLOCK];
    // Original weight_scale_inv stores multiplicative F32 128x128 scales.
    out[i] = glmvq_cast<T>(glmvq_mul(glmvq_e4m3(weight[i]), scale));
}

#endif
