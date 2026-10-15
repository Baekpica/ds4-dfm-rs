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
enum glmvq_restore_mode { GLMVQ_RESTORE_SCALAR, GLMVQ_RESTORE_VECTOR };

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

__device__ __forceinline__ uint32_t glmvq_scaled_pair(uint32_t bits, float gain) {
    const __half2_raw half = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)bits, __NV_E4M3);
    const float2 words = __half22float2(half);
    const __nv_bfloat162_raw value = __floats2bfloat162_rn(
        glmvq_mul(gain, words.x), glmvq_mul(gain, words.y));
    return (uint32_t)value.x | ((uint32_t)value.y << 16);
}

/* Scalar word loads and strided 16-bit stores dominate the initial restore.
 * Keep the exact gain/rounding boundary, but move one complete vector at once. */
__global__ void glmvq_restore_vec(glmvq_matrix m, __nv_bfloat16 *out) {
    const uint64_t vector = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (vector >= (uint64_t)m.rows * m.cols / GLMVQ_VECTOR) { return; }
    const uint64_t pair = vector / 2 * 3;
    uint32_t index = (vector & 1u)
        ? (m.main[pair + 1] >> 4) | ((uint32_t)m.main[pair + 2] << 4)
        : m.main[pair] | ((uint32_t)(m.main[pair + 1] & 15u) << 8);
    if (m.plane) { index |= ((m.plane[vector / 8] >> (vector % 8)) & 1u) << 12; }
    const float gain = __half2float(m.gain[vector / (m.cols / GLMVQ_VECTOR)]);
    const uint2 words = ((const uint2 *)m.book)[index];
    ((uint4 *)out)[vector] = make_uint4(
        glmvq_scaled_pair(words.x, gain), glmvq_scaled_pair(words.x >> 16, gain),
        glmvq_scaled_pair(words.y, gain), glmvq_scaled_pair(words.y >> 16, gain));
}

static cudaError_t glmvq_restore_bf16(glmvq_matrix m, __nv_bfloat16 *out,
        cudaStream_t stream, glmvq_restore_mode mode) {
    enum { GLM_HIDDEN = 4096, GLM_FFN = 2048, PACK_ALIGN = 128 };
    const bool shape = (m.rows == GLM_FFN && m.cols == GLM_HIDDEN) ||
        (m.rows == GLM_HIDDEN && m.cols == GLM_FFN);
    const bool aligned = !((uintptr_t)m.main % PACK_ALIGN) &&
        (!m.plane || !((uintptr_t)m.plane % PACK_ALIGN)) &&
        !((uintptr_t)m.book % sizeof(uint2)) && !((uintptr_t)out % sizeof(uint4));
    const uint64_t vectors = (uint64_t)m.rows * m.cols / GLMVQ_VECTOR;
    const unsigned blocks = (unsigned)((vectors + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
    if (mode == GLMVQ_RESTORE_VECTOR && shape && aligned && m.precision == GLMVQ_E4M3) {
        glmvq_restore_vec<<<blocks, GLMVQ_THREADS, 0, stream>>>(m, out);
    } else {
        glmvq_restore<<<blocks, GLMVQ_THREADS, 0, stream>>>(m, out);
    }
    return cudaGetLastError();
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
