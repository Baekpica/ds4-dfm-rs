#ifndef DS4_GLM53_FLASH_CUH
#define DS4_GLM53_FLASH_CUH

#include "glm_vq_flash.cuh"
#include "glm_vq_index.cuh"
#include "glm_vq_gemv.cuh"
#include "glm_vq_kda.cuh"

/* Fast paths of the compact (mixed-artifact) GLM-5.3 graph.
 *
 * Latent attention for the compact cache.
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

static bool glm53_flash_ready(void) {
    static int ready = -1;
    if (ready < 0) {
        const char *env = getenv("DS4_GLM53_FLASH");
        ready = (!env || strcmp(env, "0") != 0) &&
            cuda_ok(cudaFuncSetAttribute(glmvq_fa_kernel_f16,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GLMVQ_FA_SMEM), "GLM-5.3 flash smem");
    }
    return ready > 0;
}

static bool glm53_flash_shape(const ds4_gpu_tensor *cache, const ds4_gpu_tensor *selected,
        uint32_t sel_stride, uint32_t rows, uint32_t pos0, uint32_t cap, uint32_t heads,
        uint32_t latent_dim, uint32_t head_dim) {
    return glm53_flash_ready() && !ds4_capture_active() && rows >= GLM53_FLASH_MIN_ROWS &&
        heads == GLMVQ_FA_HEADS && latent_dim == GLMVQ_FA_LATENT && head_dim &&
        glm53_attn_frontier(rows, pos0, cap, sel_stride,
            selected ? GLM53_ATTN_SELECTED : GLM53_ATTN_ALL) &&
        glm53_compact_has(cache, (uint64_t)cap * latent_dim, sizeof(__half)) &&
        (!selected || glm53_compact_has(selected, (uint64_t)rows * sel_stride, sizeof(uint32_t)));
}

// One CTA per row; `positions` is rows x u32 of scratch.
static void glm53_flash_launch(__half *low, const __half *qlow, uint32_t *positions,
        const ds4_gpu_tensor *cache, const ds4_gpu_tensor *selected, uint32_t sel_stride,
        uint32_t rows, uint32_t pos0, uint32_t head_dim, cudaStream_t stream) {
    glm53_positions_kernel<<<(rows + 255u) / 256u, 256u, 0, stream>>>(positions, pos0, rows);
    glmvq_fa_kernel_f16<<<dim3(rows, 1u), GLMVQ_FA_THREADS, GLMVQ_FA_SMEM, stream>>>(
        low, nullptr, nullptr, qlow, (const __half *)cache->ptr, positions,
        selected ? (const uint32_t *)selected->ptr : nullptr, selected ? sel_stride : 0u,
        UINT32_MAX, 1.0f / sqrtf((float)head_dim));
}

/* low_q / low_out are [rows][heads][latent] F32 views; both are consumed as
 * F16 scratch on the way, so neither keeps its input. Returns 0 when the
 * geometry is not covered and the caller keeps the reference kernel. */
extern "C" int ds4_gpu_glm53_flash_low(ds4_gpu_tensor *low_out, ds4_gpu_tensor *low_q,
        const ds4_gpu_tensor *cache, const ds4_gpu_tensor *selected, uint32_t sel_stride,
        uint32_t rows, uint32_t pos0, uint32_t cap, uint32_t heads, uint32_t latent_dim,
        uint32_t head_dim) {
    const uint64_t count = (uint64_t)rows * heads * latent_dim;
    if (!glm53_flash_shape(cache, selected, sel_stride, rows, pos0, cap, heads, latent_dim, head_dim) ||
        !glm53_compact_has(low_out, count, sizeof(float)) ||
        !glm53_compact_has(low_q, count, sizeof(float))) {
        return 0;
    }
    const cudaStream_t stream = ds4_current_stream();
    __half *qlow = (__half *)low_out->ptr;
    uint32_t *positions = (uint32_t *)((char *)low_out->ptr + count * sizeof(__half));
    __half *low = (__half *)low_q->ptr;
    const unsigned blocks = (unsigned)((count + 255u) / 256u);
    glm53_f32_f16_kernel<<<blocks, 256u, 0, stream>>>(qlow, (const float *)low_q->ptr, count);
    glm53_flash_launch(low, qlow, positions, cache, selected, sel_stride, rows, pos0, head_dim, stream);
    glm53_f16_f32_kernel<<<blocks, 256u, 0, stream>>>((float *)low_out->ptr, low, count);
    return cuda_ok(cudaGetLastError(), "GLM-5.3 flash latent attention") ? 1 : -1;
}

/* Prefill rows: the whole absorbed chain on tensor cores. The reference
 * absorption and value projection re-read Q8_0 K_b/V_b per (row, head) on
 * SIMT (7.3 of an 8K prefill's 22.6 s after flash). Here K_b/V_b widen to
 * F16 once per call (16 MiB each, into `wscratch`) and both projections
 * are 64-head strided-batched F16 GEMMs with F32 accumulation:
 *
 *   q (F32) -> q16 -> qlow16 = q16 K_b^T -> flash -> low16 -> heads = low16 V_b^T
 *
 * q16/qlow16 live in low_q and low16/positions in low_out (both F32 views
 * with room to spare). Weights round to F16 (Q8_0 d*q needs up to 18
 * mantissa bits); q, the absorbed query, P and low round to F16 as well.
 * Returns 0 when not covered (decode and verify widths keep Q8_0 SIMT). */
enum { GLM53_GEMM_MIN_ROWS = 16, GLM53_KB_ROW = 256 };
static const uint64_t GLM53_WEIGHT_HALVES = (uint64_t)GLMVQ_FA_HEADS * GLMVQ_FA_LATENT * GLM53_KB_ROW;

__global__ static void glm53_q8_f16_kernel(__half *out, const char *w, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const char *block = w + i / GLM53_COMPACT_Q8_BLOCK * GLM53_COMPACT_Q8_BYTES;
    out[i] = __float2half_rn(__half2float(*(const __half *)block) *
        (float)((const int8_t *)(block + sizeof(__half)))[i % GLM53_COMPACT_Q8_BLOCK]);
}

static bool glm53_gemm_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *env = getenv("DS4_GLM53_ABSORB_GEMM");
        on = !env || strcmp(env, "0") != 0;
    }
    return on > 0;
}

extern "C" int ds4_gpu_glm53_flash(ds4_gpu_tensor *heads_out, const ds4_gpu_tensor *q,
        ds4_gpu_tensor *low_q, ds4_gpu_tensor *low_out, ds4_gpu_tensor *wscratch,
        const ds4_gpu_tensor *cache, const ds4_gpu_tensor *selected, uint32_t sel_stride,
        const void *model_map, uint64_t model_size, uint64_t k_b_offset, uint64_t v_b_offset,
        uint32_t rows, uint32_t pos0, uint32_t cap, uint32_t heads, uint32_t latent_dim,
        uint32_t head_dim) {
    const uint64_t qn = (uint64_t)rows * heads * head_dim, ln = (uint64_t)rows * heads * latent_dim;
    if (!glm53_gemm_on() || !g_cublas_ready || rows < GLM53_GEMM_MIN_ROWS || head_dim != GLM53_KB_ROW ||
        !glm53_flash_shape(cache, selected, sel_stride, rows, pos0, cap, heads, latent_dim, head_dim) ||
        !glm53_compact_has(heads_out, qn, sizeof(float)) || !glm53_compact_has(q, qn, sizeof(float)) ||
        // low_q: q16 + qlow16; low_out: low16 + positions.
        !glm53_compact_has(low_q, (qn + ln) * sizeof(__half), 1u) ||
        !glm53_compact_has(low_out, ln * sizeof(__half) + rows * sizeof(uint32_t), 1u) ||
        !glm53_compact_has(wscratch, 2u * GLM53_WEIGHT_HALVES, sizeof(__half))) {
        return 0;
    }
    const uint64_t wbytes = GLM53_WEIGHT_HALVES / GLM53_COMPACT_Q8_BLOCK * GLM53_COMPACT_Q8_BYTES;
    const char *kb = glm53_compact_weight(model_map, model_size, k_b_offset, wbytes, heads_out,
        "GLM-5.3 compact K-b Q8_0");
    const char *vb = glm53_compact_weight(model_map, model_size, v_b_offset, wbytes, heads_out,
        "GLM-5.3 compact V-b Q8_0");
    if (!kb || !vb) { return 0; }
    const cudaStream_t stream = ds4_current_stream();
    __half *wk = (__half *)wscratch->ptr, *wv = wk + GLM53_WEIGHT_HALVES;
    __half *q16 = (__half *)low_q->ptr, *qlow = q16 + qn;
    __half *low = (__half *)low_out->ptr;
    uint32_t *positions = (uint32_t *)(low + ln);
    const unsigned wblocks = (unsigned)((GLM53_WEIGHT_HALVES + 255u) / 256u);
    glm53_q8_f16_kernel<<<wblocks, 256u, 0, stream>>>(wk, kb, GLM53_WEIGHT_HALVES);
    glm53_q8_f16_kernel<<<wblocks, 256u, 0, stream>>>(wv, vb, GLM53_WEIGHT_HALVES);
    glm53_f32_f16_kernel<<<(unsigned)((qn + 255u) / 256u), 256u, 0, stream>>>(q16, (const float *)q->ptr, qn);
    if (!cuda_ok(cudaGetLastError(), "GLM-5.3 absorbed GEMM inputs")) { return -1; }

    // Per head h (batch): qlow[r][h][:] = W_k[h] (512 x 256, row-major) q[r][h][:].
    cudaStream_t prior = 0;
    (void)cublasGetStream(g_cublas, &prior);
    (void)cublasSetStream(g_cublas, stream);
    const float one = 1.0f, zero = 0.0f;
    const long long wstride = (long long)latent_dim * head_dim;
    cuda_cublas_ws_prep(stream);
    bool ok = cublas_ok(cublasGemmStridedBatchedEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        (int)latent_dim, (int)rows, (int)head_dim, &one, wk, CUDA_R_16F, (int)head_dim, wstride,
        q16, CUDA_R_16F, (int)(heads * head_dim), head_dim, &zero,
        qlow, CUDA_R_16F, (int)(heads * latent_dim), latent_dim, (int)heads,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "GLM-5.3 Q absorption GEMM");
    if (ok) {
        glm53_flash_launch(low, qlow, positions, cache, selected, sel_stride, rows, pos0, head_dim, stream);
        ok = cuda_ok(cudaGetLastError(), "GLM-5.3 flash latent attention");
    }
    // heads[r][h][:] = W_v[h] (256 x 512, row-major) low[r][h][:], F32 out.
    cuda_cublas_ws_prep(stream);
    ok = ok && cublas_ok(cublasGemmStridedBatchedEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        (int)head_dim, (int)rows, (int)latent_dim, &one, wv, CUDA_R_16F, (int)latent_dim, wstride,
        low, CUDA_R_16F, (int)(heads * latent_dim), latent_dim, &zero,
        heads_out->ptr, CUDA_R_32F, (int)(heads * head_dim), head_dim, (int)heads,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "GLM-5.3 V projection GEMM");
    (void)cublasSetStream(g_cublas, prior);
    return ok ? 1 : -1;
}

/* Rank-128 KDA projections (f_b, g_b: Q8_0, K=128 -> 8192) at prefill
 * widths. The generic Q8_0 dispatch sends K < 256 to a warp-per-row SIMT
 * kernel: 7.8 ms per 2,048-row call, 2.1 of an 8K prefill's 15.4 s. The
 * product is bound by its 64 MB F32 output, so widen the 4 MiB weight to
 * F32 into `scratch` and run one cuBLAS SGEMM (TF32 tensor cores unless
 * quality mode); activations stay F32. 1 done, 0 not covered, -1 failed. */
enum { GLM53_SMALL_K_MIN_ROWS = 16, GLM53_SMALL_K_MAX = 256 };

__global__ static void glm53_q8_f32_kernel(float *out, const char *w, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const char *block = w + i / GLM53_COMPACT_Q8_BLOCK * GLM53_COMPACT_Q8_BYTES;
    out[i] = __half2float(*(const __half *)block) *
        (float)((const int8_t *)(block + sizeof(__half)))[i % GLM53_COMPACT_Q8_BLOCK];
}

extern "C" int ds4_gpu_glm53_q8_small_k(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        ds4_gpu_tensor *scratch, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    static int on = -1;
    if (on < 0) {
        const char *env = getenv("DS4_GLM53_SMALL_K");
        on = !env || strcmp(env, "0") != 0;
    }
    const uint64_t count = (uint64_t)in_dim * out_dim;
    if (!on || !g_cublas_ready || ds4_capture_active() || rows < GLM53_SMALL_K_MIN_ROWS ||
        !in_dim || in_dim % GLM53_COMPACT_Q8_BLOCK || in_dim >= GLM53_SMALL_K_MAX || !out_dim ||
        out_dim > INT_MAX || rows > INT_MAX ||
        !glm53_compact_has(out, (uint64_t)rows * out_dim, sizeof(float)) ||
        !glm53_compact_has(x, (uint64_t)rows * in_dim, sizeof(float)) ||
        !glm53_compact_has(scratch, count, sizeof(float))) {
        return 0;
    }
    const char *w = glm53_compact_weight(model_map, model_size, weight_offset,
        count / GLM53_COMPACT_Q8_BLOCK * GLM53_COMPACT_Q8_BYTES, out, "GLM-5.3 small-K Q8_0");
    if (!w) { return 0; }
    const cudaStream_t stream = ds4_current_stream();
    float *wf = (float *)scratch->ptr;
    glm53_q8_f32_kernel<<<(unsigned)((count + 255u) / 256u), 256u, 0, stream>>>(wf, w, count);
    if (!cuda_ok(cudaGetLastError(), "GLM-5.3 small-K widen")) { return -1; }
    cudaStream_t prior = 0;
    (void)cublasGetStream(g_cublas, &prior);
    (void)cublasSetStream(g_cublas, stream);
    const float one = 1.0f, zero = 0.0f;
    // Row-major [out][in] weights are column-major (in x out): op = T.
    const bool ok = cublas_ok(cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)out_dim,
        (int)rows, (int)in_dim, &one, wf, (int)in_dim, (const float *)x->ptr, (int)in_dim,
        &zero, (float *)out->ptr, (int)out_dim), "GLM-5.3 small-K GEMM");
    (void)cublasSetStream(g_cublas, prior);
    return ok ? 1 : -1;
}

/* Pooled index scores at prefill widths. The reference runs one block per
 * (pool, row) and re-reads the row's 32 x 128 F32 query for every pool:
 * 18% of a 32K prefill (69 ms per 2,048-row call), growing with context.
 * GLM VQ's fused tile scores 4 rows x 128 pools with m16n8k16 MMAs and
 * folds the 32 heads in registers; here on the compact F16 pools. q rounds
 * to F16; the 1/sqrt(128 * 32) scale splits into per-dot and per-weight
 * factors and the head sum runs in another order. 1/0/-1 as above. */
extern "C" int ds4_gpu_glm53_index_mma(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *pool_cache, uint32_t n_pools,
        uint32_t rows, uint32_t pos0, uint32_t heads) {
    static int ready = -1;
    if (ready < 0) {
        const char *env = getenv("DS4_GLM53_INDEX_MMA");
        ready = (!env || strcmp(env, "0") != 0) &&
            cuda_ok(cudaFuncSetAttribute(glmvq_index_fused_f16,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GLMVQ_IDX_SMEM), "GLM-5.3 index smem");
    }
    if (!ready || ds4_capture_active() || rows < GLMVQ_IDX_ROWS || heads != GLMVQ_INDEX_HEADS ||
        !n_pools || pos0 > UINT32_MAX - rows || n_pools != (pos0 + rows) / DS4_GLM53_POOL_SIZE ||
        !glm53_compact_has(scores, (uint64_t)rows * n_pools, sizeof(float)) ||
        !glm53_compact_has(q, (uint64_t)rows * heads * DS4_GLM53_POOL_DIM, sizeof(float)) ||
        !glm53_compact_has(weights, (uint64_t)rows * heads, sizeof(float)) ||
        !glm53_compact_has(pool_cache, (uint64_t)n_pools * DS4_GLM53_POOL_DIM, sizeof(__half))) {
        return 0;
    }
    const dim3 grid((n_pools + GLMVQ_IDX_POOLS - 1) / GLMVQ_IDX_POOLS, (rows + GLMVQ_IDX_ROWS - 1) / GLMVQ_IDX_ROWS);
    glmvq_index_fused_f16<<<grid, GLMVQ_IDX_THREADS, GLMVQ_IDX_SMEM, ds4_current_stream()>>>(
        (float *)scores->ptr, (const float *)q->ptr, (const float *)weights->ptr,
        (const __half *)pool_cache->ptr, n_pools, rows, pos0);
    return cuda_ok(cudaGetLastError(), "GLM-5.3 fused index scores") ? 1 : -1;
}

/* mHC mix: RMS of the 4 x 4096 residual row and its 24 BF16 mapping dots.
 * The reference wrote the normalized 16K row (rms_norm_plain), widened it
 * to BF16 and ran a 24-output BF16 GEMM: four passes over each row and
 * 1.0 of an 8K prefill's 12.1 s (17.7 us + 6.6 us per decode call). GLM
 * VQ's kernels take F32 rows in one pass: prefill rows accumulate squares
 * and raw dots together (mix = inv * W x), decode rows recompute the RMS
 * per output block. Activations stay F32 instead of rounding to BF16.
 * 1 done, 0 not covered, -1 failed. */
extern "C" int ds4_gpu_glm53_hc_mix(ds4_gpu_tensor *mix, const ds4_gpu_tensor *residual,
        const void *model_map, uint64_t model_size, uint64_t fn_offset, uint32_t width,
        uint32_t outputs, uint32_t rows, float eps) {
    static int on = -1;
    if (on < 0) {
        const char *env = getenv("DS4_GLM53_HC_MIX");
        on = !env || strcmp(env, "0") != 0;
    }
    if (!on || !rows || outputs != GLMVQ_HC_MIX || !width ||
        width % (GLMVQ_THREADS * GLMVQ_DOT_VECTOR) || width % GLMVQ_MIX_TILE ||
        !glm53_compact_has(mix, (uint64_t)rows * outputs, sizeof(float)) ||
        !glm53_compact_has(residual, (uint64_t)rows * width, sizeof(float)) ||
        (uintptr_t)residual->ptr % 16) {
        return 0;
    }
    const __nv_bfloat16 *w = (const __nv_bfloat16 *)glm53_compact_weight(model_map, model_size,
        fn_offset, (uint64_t)outputs * width * sizeof(__nv_bfloat16), mix, "GLM-5.3 mHC fn BF16");
    if (!w || (uintptr_t)w % 16) { return 0; }
    const cudaStream_t stream = ds4_current_stream();
    if (rows > GLMVQ_ABS_ROWS) {
        glmvq_hc_mix_rows<<<(rows + GLMVQ_MIX_WARPS - 1) / GLMVQ_MIX_WARPS, GLMVQ_MIX_WARPS * 32, 0, stream>>>(
            (const float *)residual->ptr, w, (float *)mix->ptr, rows, width, eps);
    } else {
        glmvq_hc_mix<<<dim3(outputs, rows), GLMVQ_THREADS, 0, stream>>>(
            (const float *)residual->ptr, w, (float *)mix->ptr, width, eps);
    }
    return cuda_ok(cudaGetLastError(), "GLM-5.3 mHC mix") ? 1 : -1;
}

/* Chunked KDA prefill on tensor cores. The Solar F32 chunk kernels and the
 * separate gated head norm were 1.64 of an 8K prefill's 11.3 s (the scan
 * walks 32 chunks per head on SIMT). GLM VQ's local + scan kernels build
 * 64-token chunk records with F16 MMAs and keep each head's state in
 * registers; the scan also applies the gated RMS norm. State and conv tails
 * keep the Solar layout ([head][key][value], [channel][4]) and the same
 * decay/norm/beta algebra, so decode's recurrence continues from them.
 * Arithmetic: the VQ recurrence's, i.e. q/k/v after conv and SiLU, beta and
 * the output round to BF16 (the source model's boundaries); q and the
 * records are F16. `scratch` takes the records (90.6 KiB per row) and a BF16
 * copy of o_norm (exact: its values are BF16). 1/0/-1 as above. */
__global__ static void glm53_f32_bf16_kernel(__nv_bfloat16 *out, const float *in, uint32_t count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = __float2bfloat16_rn(in[i]); }
}

extern "C" int ds4_gpu_glm53_kda_mma(ds4_gpu_tensor *out, ds4_gpu_tensor *scratch,
        ds4_gpu_tensor *state, ds4_gpu_tensor *q_tail, ds4_gpu_tensor *k_tail, ds4_gpu_tensor *v_tail,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *forget, const ds4_gpu_tensor *beta,
        const ds4_gpu_tensor *q_weight, const ds4_gpu_tensor *k_weight, const ds4_gpu_tensor *v_weight,
        const ds4_gpu_tensor *a_log, const ds4_gpu_tensor *dt_bias, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *norm, uint32_t rows, uint32_t heads, float lower_bound, float eps) {
    static int ready = -1;
    if (ready < 0) {
        const char *env = getenv("DS4_GLM53_KDA_MMA");
        ready = (!env || strcmp(env, "0") != 0) &&
            cuda_ok(cudaFuncSetAttribute(glmvq_kda_local<float, float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GLMVQ_KLOCAL_SMEM), "GLM-5.3 KDA local smem") &&
            cuda_ok(cudaFuncSetAttribute(glmvq_kda_scan<float, float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GLMVQ_KSCAN_SMEM), "GLM-5.3 KDA scan smem");
    }
    const uint64_t width = (uint64_t)heads * GLMVQ_KDA_DIM, count = (uint64_t)rows * width;
    const uint64_t tail = width * GLMVQ_CONV;
    const uint64_t records = (uint64_t)((rows + GLMVQ_KC - 1) / GLMVQ_KC) * heads * GLMVQ_KR_BYTES;
    if (!ready || ds4_capture_active() || rows < GLMVQ_KC || !heads ||
        !glm53_compact_has(out, count, sizeof(float)) || !glm53_compact_has(gate, count, sizeof(float)) ||
        !glm53_compact_has(q, count, sizeof(float)) || !glm53_compact_has(k, count, sizeof(float)) ||
        !glm53_compact_has(v, count, sizeof(float)) || !glm53_compact_has(forget, count, sizeof(float)) ||
        !glm53_compact_has(beta, (uint64_t)rows * heads, sizeof(float)) ||
        !glm53_compact_has(state, width * GLMVQ_KDA_DIM, sizeof(float)) ||
        !glm53_compact_has(q_tail, tail, sizeof(float)) || !glm53_compact_has(k_tail, tail, sizeof(float)) ||
        !glm53_compact_has(v_tail, tail, sizeof(float)) || !glm53_compact_has(q_weight, tail, sizeof(float)) ||
        !glm53_compact_has(k_weight, tail, sizeof(float)) || !glm53_compact_has(v_weight, tail, sizeof(float)) ||
        !glm53_compact_has(a_log, heads, sizeof(float)) || !glm53_compact_has(dt_bias, width, sizeof(float)) ||
        !glm53_compact_has(norm, GLMVQ_KDA_DIM, sizeof(float)) ||
        !glm53_compact_has(scratch, records + GLMVQ_KDA_DIM * sizeof(__nv_bfloat16), 1u)) {
        return 0;
    }
    const cudaStream_t stream = ds4_current_stream();
    uint8_t *rec = (uint8_t *)scratch->ptr;
    __nv_bfloat16 *norm16 = (__nv_bfloat16 *)(rec + records);
    glm53_f32_bf16_kernel<<<1, GLMVQ_KDA_DIM, 0, stream>>>(norm16, (const float *)norm->ptr, GLMVQ_KDA_DIM);
    glmvq_kda_local<float, float><<<dim3((rows + GLMVQ_KC - 1) / GLMVQ_KC, heads), GLMVQ_KLOCAL,
        GLMVQ_KLOCAL_SMEM, stream>>>(rec, (const float *)q_tail->ptr, (const float *)k_tail->ptr,
        (const float *)v_tail->ptr, (const float *)q->ptr, (const float *)k->ptr, (const float *)v->ptr,
        (const float *)forget->ptr, (const float *)beta->ptr, (const float *)q_weight->ptr,
        (const float *)k_weight->ptr, (const float *)v_weight->ptr, (const float *)a_log->ptr,
        (const float *)dt_bias->ptr, rows, heads, lower_bound);
    // The first chunk read the previous tails; they advance only afterwards.
    glmvq_kda_tail<float><<<(unsigned)((width + 255u) / 256u), 256u, 0, stream>>>(
        (float *)q_tail->ptr, (float *)k_tail->ptr, (float *)v_tail->ptr,
        (const float *)q->ptr, (const float *)k->ptr, (const float *)v->ptr, rows, width);
    glmvq_kda_scan<float, float><<<heads, GLMVQ_KSCAN, GLMVQ_KSCAN_SMEM, stream>>>(
        (float *)out->ptr, (float *)state->ptr, rec, (const float *)gate->ptr, norm16, rows, heads, eps);
    return cuda_ok(cudaGetLastError(), "GLM-5.3 tensor-core KDA") ? 1 : -1;
}

#endif
