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
    // Optional 8-bit residual stage (GVR8, container v2): one index byte per
    // vector, row-major, into a K=256 E4M3 layer book. The row gain scales
    // both codewords: W[r,8j+d] = gain[r] * (C1[i1][d] + C2[i2][d]).
    const uint8_t *res;
    const uint8_t *res_book;
};

enum { GLMVQ_RES_WORDS = 256 };

// GVQ1 version 2 (A.X-K2) stores 9- or 10-bit indices byte-major: a block
// of 2,048 inputs is 256 low bytes (64 words, a and b per lane) and a plane
// of the high bits (9-bit: one per index, 8 words; 10-bit: two, 16 words).
// Version 1 packs 12 bits (96 words) with an optional 13th-bit plane (8
// words). BITS selects the version: 9 and 10 exist only in v2.
template <uint32_t BITS>
__host__ __device__ constexpr bool glmvq_v2() { return BITS == 9 || BITS == 10; }
template <uint32_t BITS>
__host__ __device__ constexpr bool glmvq_planed() { return BITS == 13 || glmvq_v2<BITS>(); }
// Main and plane words per 2048-input block.
template <uint32_t BITS>
__host__ __device__ constexpr uint32_t glmvq_block_main() { return glmvq_v2<BITS>() ? 64u : 96u; }
template <uint32_t BITS>
__host__ __device__ constexpr uint32_t glmvq_block_plane() { return BITS == 10 ? 16u : 8u; }

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

/* Programmatic dependent launch for decode chains of short kernels. A
 * kernel launched with GLMVQ_LAUNCH_PDL may start while its predecessor
 * drains: it calls glmvq_pdl_open first (its successor may launch early too)
 * and glmvq_pdl_wait before it reads a predecessor's output or writes global
 * memory; constant weights load before the wait. 238 chained one-token
 * Q4_K GEMVs ran 7% faster; GLM decode gained 1.8%. Both calls are no-ops
 * for ordinary launches. */
enum glmvq_launch { GLMVQ_LAUNCH_PLAIN, GLMVQ_LAUNCH_PDL };

__device__ __forceinline__ void glmvq_pdl_open() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

__device__ __forceinline__ void glmvq_pdl_wait() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

template <typename... P, typename... A>
static cudaError_t glmvq_launch_kernel(glmvq_launch mode, void (*kernel)(P...), dim3 grid, dim3 block,
        size_t shared, cudaStream_t stream, A... args) {
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = shared;
    config.stream = stream;
    cudaLaunchAttribute attribute = {};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = mode == GLMVQ_LAUNCH_PDL ? 1 : 0;
    return cudaLaunchKernelEx(&config, kernel, args...);
}

// BF16-rounded activations stored as F32 or BF16 (mHC rows, expert outputs).
__device__ __forceinline__ float glmvq_value(float v) { return v; }
__device__ __forceinline__ float glmvq_value(__nv_bfloat16 v) { return __bfloat162float(v); }

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
    const uint32_t res = m.res ? m.res[vector] : 0u;
    for (uint32_t d = 0; d < GLMVQ_VECTOR; d++) {
        const uint32_t i = index * GLMVQ_VECTOR + d;
        float word = m.precision == GLMVQ_E4M3
            ? glmvq_e4m3(((const uint8_t *)m.book)[i])
            : __half2float(((const __half *)m.book)[i]);
        // Two E4M3 codewords add exactly in FP32.
        if (m.res) { word += glmvq_e4m3(m.res_book[res * GLMVQ_VECTOR + d]); }
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
    // The vector restore has no residual stage; the scalar one keeps it exact.
    if (mode == GLMVQ_RESTORE_VECTOR && shape && aligned && m.precision == GLMVQ_E4M3 && !m.res) {
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
