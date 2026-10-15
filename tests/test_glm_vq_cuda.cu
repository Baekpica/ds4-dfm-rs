/* Test-only ABI: production callers use the kernels in the native TU. */
#include "cuda/glm_vq.cuh"

enum output_kind { OUTPUT_F32 = 0, OUTPUT_BF16 = 1 };

extern "C" int test_glmvq_restore(const uint8_t *main, const uint8_t *plane,
        const void *book, const __half *gain, uint32_t rows, uint32_t cols,
        uint32_t precision, uint32_t output, void *out, cudaStream_t stream) {
    if (!main || !book || !gain || !out || !rows || !cols || cols % 64 ||
        (precision != GLMVQ_E4M3 && precision != GLMVQ_F16)) {
        return (int)cudaErrorInvalidValue;
    }
    const glmvq_matrix matrix = { main, plane, book, gain, rows, cols,
        (glmvq_precision)precision };
    const uint64_t vectors = (uint64_t)rows * cols / GLMVQ_VECTOR;
    const unsigned blocks = (unsigned)((vectors + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
    if (output == OUTPUT_F32) {
        glmvq_restore<<<blocks, GLMVQ_THREADS, 0, stream>>>(matrix, (float *)out);
    } else if (output == OUTPUT_BF16) {
        glmvq_restore<<<blocks, GLMVQ_THREADS, 0, stream>>>(matrix, (__nv_bfloat16 *)out);
    } else { return (int)cudaErrorInvalidValue; }
    return (int)cudaGetLastError();
}

extern "C" int test_glmvq_fp8(const uint8_t *weight, const float *scale,
        uint32_t rows, uint32_t cols, uint32_t output, void *out, cudaStream_t stream) {
    if (!weight || !scale || !out || !rows || !cols || rows % GLMVQ_SCALE_BLOCK ||
        cols % GLMVQ_SCALE_BLOCK) { return (int)cudaErrorInvalidValue; }
    const uint64_t count = (uint64_t)rows * cols;
    const unsigned blocks = (unsigned)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
    if (output == OUTPUT_F32) {
        glmvq_fp8_restore<<<blocks, GLMVQ_THREADS, 0, stream>>>(weight, scale, rows, cols, (float *)out);
    } else if (output == OUTPUT_BF16) {
        glmvq_fp8_restore<<<blocks, GLMVQ_THREADS, 0, stream>>>(weight, scale, rows, cols, (__nv_bfloat16 *)out);
    } else { return (int)cudaErrorInvalidValue; }
    return (int)cudaGetLastError();
}
