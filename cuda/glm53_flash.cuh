#ifndef DS4_GLM53_FLASH_CUH
#define DS4_GLM53_FLASH_CUH

#include "glm_vq_flash.cuh"

/* Tensor-core latent attention for the compact (mixed-artifact) cache.
 * The reference kernel runs one SIMT block per (row, head): each of the 64
 * heads re-reads the row's 2,051 selected FP16 latents, 41.7% of an 8K
 * prefill (15.8 of 37.9 s). GLM VQ's flash tile scores all 64 heads against
 * each 32-key tile once (online softmax, P·V on m16n8k16); here it runs on
 * F16 tiles because the compact cache stores F16 latents.
 *
 *   low_q (F32) --F16--> qlow16 [low_out bytes] --flash--> low16 [low_q bytes]
 *   low16 --F32--> low_out
 *
 * Arithmetic change: the absorbed query and P round to F16 for the MMAs and
 * the latent output rounds to F16; scores, softmax and O stay F32. */
enum { GLM53_FLASH_MIN_ROWS = 2 };

__global__ static void glm53_f32_f16_kernel(__half *out, const float *in, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = __float2half_rn(in[i]); }
}

__global__ static void glm53_f16_f32_kernel(float *out, const __half *in, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = __half2float(in[i]); }
}

__global__ static void glm53_positions_kernel(uint32_t *out, uint32_t pos0, uint32_t rows) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) { out[i] = pos0 + i; }
}

/* low_q / low_out are [rows][heads][latent] F32 views; both are consumed as
 * F16 scratch on the way, so neither keeps its input. Returns 0 when the
 * geometry is not covered and the caller keeps the reference kernel. */
extern "C" int ds4_gpu_glm53_flash_low(ds4_gpu_tensor *low_out, ds4_gpu_tensor *low_q,
        const ds4_gpu_tensor *cache, const ds4_gpu_tensor *selected, uint32_t sel_stride,
        uint32_t rows, uint32_t pos0, uint32_t cap, uint32_t heads, uint32_t latent_dim,
        uint32_t head_dim) {
    static int ready = -1;
    if (ready < 0) {
        const char *env = getenv("DS4_GLM53_FLASH");
        ready = (!env || strcmp(env, "0") != 0) &&
            cuda_ok(cudaFuncSetAttribute(glmvq_fa_kernel_f16,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GLMVQ_FA_SMEM), "GLM-5.3 flash smem");
    }
    const uint64_t count = (uint64_t)rows * heads * latent_dim;
    if (!ready || ds4_capture_active() || rows < GLM53_FLASH_MIN_ROWS ||
        heads != GLMVQ_FA_HEADS || latent_dim != GLMVQ_FA_LATENT || !head_dim ||
        !glm53_attn_frontier(rows, pos0, cap, sel_stride,
            selected ? GLM53_ATTN_SELECTED : GLM53_ATTN_ALL) ||
        !glm53_compact_has(low_out, count, sizeof(float)) ||
        !glm53_compact_has(low_q, count, sizeof(float)) ||
        !glm53_compact_has(cache, (uint64_t)cap * latent_dim, sizeof(__half)) ||
        (selected && !glm53_compact_has(selected, (uint64_t)rows * sel_stride, sizeof(uint32_t)))) {
        return 0;
    }
    const cudaStream_t stream = ds4_current_stream();
    __half *qlow = (__half *)low_out->ptr;
    uint32_t *positions = (uint32_t *)((char *)low_out->ptr + count * sizeof(__half));
    __half *low = (__half *)low_q->ptr;
    const unsigned blocks = (unsigned)((count + 255u) / 256u);
    glm53_f32_f16_kernel<<<blocks, 256u, 0, stream>>>(qlow, (const float *)low_q->ptr, count);
    glm53_positions_kernel<<<(rows + 255u) / 256u, 256u, 0, stream>>>(positions, pos0, rows);
    glmvq_fa_kernel_f16<<<dim3(rows, 1u), GLMVQ_FA_THREADS, GLMVQ_FA_SMEM, stream>>>(
        low, nullptr, nullptr, qlow, (const __half *)cache->ptr, positions,
        selected ? (const uint32_t *)selected->ptr : nullptr, selected ? sel_stride : 0u,
        UINT32_MAX, 1.0f / sqrtf((float)head_dim));
    glm53_f16_f32_kernel<<<blocks, 256u, 0, stream>>>((float *)low_out->ptr, low, count);
    return cuda_ok(cudaGetLastError(), "GLM-5.3 flash latent attention") ? 1 : -1;
}

#endif
