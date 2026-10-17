#ifndef DS4_GLM_VQ_LINEAR_CUH
#define DS4_GLM_VQ_LINEAR_CUH

#include "glm_vq_ops.cuh"
#include "glm_vq_q4.cuh"
#include <cublas_v2.h>
#include <limits.h>

enum glmvq_storage { GLMVQ_WEIGHT_BF16, GLMVQ_WEIGHT_FP8, GLMVQ_WEIGHT_VQ, GLMVQ_WEIGHT_Q4 };
enum glmvq_compute { GLMVQ_COMPUTE_BF16, GLMVQ_COMPUTE_F32 };

struct glmvq_source {
    const void *data;
    const float *scale;
    glmvq_matrix vq;
    uint32_t rows, cols;
    glmvq_storage storage;
};

/* These sizes describe temporary restore/input/output buffers, not resident
 * derived weights. A caller allocates its largest requirement before capture. */
static uint64_t glmvq_linear_bytes(const glmvq_source &w, uint32_t tokens,
        glmvq_compute compute) {
    const uint64_t count = (uint64_t)w.rows * w.cols;
    if (compute == GLMVQ_COMPUTE_F32) { return count * sizeof(float); }
    const uint64_t restored = w.storage == GLMVQ_WEIGHT_BF16 ? 0 : count;
    return (restored + (uint64_t)tokens * (w.rows + w.cols)) * sizeof(__nv_bfloat16);
}

template<typename D, typename S>
__global__ void glmvq_convert(const S *input, D *output, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { output[i] = glmvq_cast<D>((float)input[i]); }
}

enum glmvq_output { GLMVQ_OUT_F32, GLMVQ_OUT_BF16 };

// GLMVQ_OUT_BF16 leaves the BF16 GEMM result in `output` (first half) for
// readers that take BF16, skipping the widening pass.
static cublasStatus_t glmvq_gemm(cublasHandle_t handle, const glmvq_source &w,
        const float *input, float *output, uint32_t tokens, glmvq_compute compute,
        void *scratch, uint64_t bytes, cudaStream_t stream,
        glmvq_restore_mode restore = GLMVQ_RESTORE_VECTOR, glmvq_output kind = GLMVQ_OUT_F32) {
    if (!handle || !input || !output || !scratch || !w.rows || !w.cols || !tokens ||
        w.rows > INT_MAX || w.cols > INT_MAX || tokens > INT_MAX ||
        bytes < glmvq_linear_bytes(w, tokens, compute)) { return CUBLAS_STATUS_INVALID_VALUE; }
    const uint64_t count = (uint64_t)w.rows * w.cols;
    const unsigned blocks = (unsigned)((count + GLMVQ_THREADS - 1) / GLMVQ_THREADS);
    const float alpha = 1.0f, beta = 0.0f;
    if (compute == GLMVQ_COMPUTE_F32) {
        if (w.storage != GLMVQ_WEIGHT_BF16 || !w.data) { return CUBLAS_STATUS_INVALID_VALUE; }
        glmvq_convert<<<blocks, GLMVQ_THREADS, 0, stream>>>(
            (const __nv_bfloat16 *)w.data, (float *)scratch, count);
        if (cudaGetLastError() != cudaSuccess) { return CUBLAS_STATUS_EXECUTION_FAILED; }
        // mHC and router explicitly use .float(); never truncate to TF32/BF16.
        return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, w.rows, tokens, w.cols,
            &alpha, scratch, CUDA_R_32F, w.cols, input, CUDA_R_32F, w.cols,
            &beta, output, CUDA_R_32F, w.rows, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
    }

    __nv_bfloat16 *cursor = (__nv_bfloat16 *)scratch;
    const __nv_bfloat16 *weight = (const __nv_bfloat16 *)w.data;
    if (w.storage == GLMVQ_WEIGHT_FP8) {
        if (!w.data || !w.scale || w.rows % GLMVQ_SCALE_BLOCK || w.cols % GLMVQ_SCALE_BLOCK) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        glmvq_fp8_restore<<<blocks, GLMVQ_THREADS, 0, stream>>>(
            (const uint8_t *)w.data, w.scale, w.rows, w.cols, cursor);
        weight = cursor;
        cursor += count;
    } else if (w.storage == GLMVQ_WEIGHT_Q4) {
        if (!w.data || w.cols % GLMVQ_Q4_BLOCK) { return CUBLAS_STATUS_INVALID_VALUE; }
        glmvq_q4_restore<<<blocks, GLMVQ_THREADS, 0, stream>>>((const uint8_t *)w.data, cursor, count);
        weight = cursor;
        cursor += count;
    } else if (w.storage == GLMVQ_WEIGHT_VQ) {
        if (w.rows != w.vq.rows || w.cols != w.vq.cols || !w.vq.main || !w.vq.book || !w.vq.gain) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        if (glmvq_restore_bf16(w.vq, cursor, stream, restore) != cudaSuccess) {
            return CUBLAS_STATUS_EXECUTION_FAILED;
        }
        weight = cursor;
        cursor += count;
    } else if (w.storage != GLMVQ_WEIGHT_BF16 || !w.data) { return CUBLAS_STATUS_INVALID_VALUE; }

    const uint64_t nx = (uint64_t)tokens * w.cols, ny = (uint64_t)tokens * w.rows;
    __nv_bfloat16 *x = cursor, *y = kind == GLMVQ_OUT_BF16 ? (__nv_bfloat16 *)output : cursor + nx;
    glmvq_convert<<<(nx + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, stream>>>(input, x, nx);
    if (cudaGetLastError() != cudaSuccess) { return CUBLAS_STATUS_EXECUTION_FAILED; }
    const cublasStatus_t status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        w.rows, tokens, w.cols, &alpha, weight, CUDA_R_16BF, w.cols,
        x, CUDA_R_16BF, w.cols, &beta, y, CUDA_R_16BF, w.rows,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS || kind == GLMVQ_OUT_BF16) { return status; }
    glmvq_convert<<<(ny + GLMVQ_THREADS - 1) / GLMVQ_THREADS, GLMVQ_THREADS, 0, stream>>>(y, output, ny);
    return cudaGetLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

#endif
