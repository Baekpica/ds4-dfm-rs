/* Included by the native CUDA backend; no host or Rust ABI state escapes. */
#include "cuda/ling3vl_primitives.cuh"

extern "C" int ds4_gpu_ling3vl_router(
        ds4_gpu_tensor *ids, ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits,
        const void *map, uint64_t size, uint64_t offset, uint32_t rows,
        float weight_scale) {
    const uint64_t bias_bytes = LING_EXPERTS * sizeof(float);
    if (!ids || !weights || !logits || !map || !rows || rows > INT_MAX ||
        offset > size || bias_bytes > size - offset ||
        !isfinite(weight_scale) || weight_scale <= 0.0f ||
        ids->bytes < (uint64_t)rows * LING_USED * sizeof(int) ||
        weights->bytes < (uint64_t)rows * LING_USED * sizeof(float) ||
        logits->bytes < (uint64_t)rows * LING_EXPERTS * sizeof(float)) { return 0; }
    const float *bias = (const float *)cuda_model_range_ptr(
        map, offset, bias_bytes, "ling3vl router bias");
    if (!bias) { return 0; }
    ling3vl_router<<<rows, 256, 0, ds4_current_stream()>>>(
        (int *)ids->ptr, (float *)weights->ptr, (const float *)logits->ptr,
        bias, weight_scale);
    return cuda_ok(cudaGetLastError(), "Ling-3.0 grouped router");
}

/* `positions` is the interleaved [t, h, w] triple per row. `head_stride` and
 * `offset` locate the rotary slice inside each head row, so the same kernel
 * rotates Q's 64 tail dims of 192 and the single shared K row's tail of 576. */
extern "C" int ds4_gpu_ling3vl_mrope(
        ds4_gpu_tensor *x, const ds4_gpu_tensor *positions,
        const ds4_gpu_tensor *inv_freq, uint32_t rows, uint32_t heads,
        uint32_t head_stride, uint32_t offset, uint32_t rotary,
        uint32_t section_t, uint32_t section_h, float attn_factor) {
    const uint32_t half = rotary / 2u;
    if (!x || !positions || !inv_freq || !rows || !heads || !rotary ||
        (rotary & 1u) || offset + rotary > head_stride ||
        section_t + section_h > half ||
        !isfinite(attn_factor) || attn_factor <= 0.0f ||
        (uint64_t)rows > UINT64_MAX / heads / head_stride) { return 0; }
    const uint64_t values = (uint64_t)rows * heads * head_stride;
    const uint64_t pairs = (uint64_t)rows * heads * half;
    if (!pairs || pairs > INT_MAX ||
        x->bytes < values * sizeof(float) ||
        inv_freq->bytes < (uint64_t)half * sizeof(float) ||
        positions->bytes < (uint64_t)rows * 3u * sizeof(int32_t)) { return 0; }
    ling3vl_mrope<<<(pairs + 255u) / 256u, 256, 0, ds4_current_stream()>>>(
        (float *)x->ptr, (const int32_t *)positions->ptr,
        (const float *)inv_freq->ptr, heads, head_stride, offset, half,
        section_t, section_t + section_h, pairs, attn_factor);
    return cuda_ok(cudaGetLastError(), "Ling-3.0 M-RoPE");
}

extern "C" int ds4_gpu_ling3vl_qk_absorb(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const void *map, uint64_t size, uint64_t k_b_offset,
        uint32_t rows, uint32_t heads, uint32_t key_dim,
        uint32_t qk_nope, uint32_t latent_dim) {
    const uint64_t weight_bytes =
        (uint64_t)heads * latent_dim * qk_nope * sizeof(__nv_bfloat16);
    const uint64_t out_values = (uint64_t)rows * heads * latent_dim;
    if (!out || !q || !map || !rows || !heads || qk_nope < 128u ||
        (qk_nope & 127u) || (latent_dim & 31u) || qk_nope > key_dim ||
        k_b_offset > size || weight_bytes > size - k_b_offset ||
        out_values > INT_MAX ||
        out->bytes < out_values * sizeof(float) ||
        q->bytes < (uint64_t)rows * heads * key_dim * sizeof(float)) { return 0; }
    const __nv_bfloat16 *k_b = (const __nv_bfloat16 *)cuda_model_range_ptr(
        map, k_b_offset, weight_bytes, "ling3vl k_b");
    if (!k_b) { return 0; }
    const dim3 grid(heads, rows);
    ling3vl_qk_absorb_bf16<<<grid, 256, qk_nope * sizeof(float),
                             ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)q->ptr, k_b, rows, heads, key_dim,
        qk_nope, latent_dim);
    return cuda_ok(cudaGetLastError(), "Ling-3.0 MLA absorb");
}

extern "C" int ds4_gpu_ling3vl_value_project(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *latent,
        const void *map, uint64_t size, uint64_t v_b_offset,
        uint32_t rows, uint32_t heads, uint32_t latent_dim,
        uint32_t value_dim) {
    const uint64_t weight_bytes =
        (uint64_t)heads * value_dim * latent_dim * sizeof(__nv_bfloat16);
    const uint64_t out_values = (uint64_t)rows * heads * value_dim;
    if (!out || !latent || !map || !rows || !heads || latent_dim < 128u ||
        (latent_dim & 127u) || (value_dim & 31u) ||
        v_b_offset > size || weight_bytes > size - v_b_offset ||
        out_values > INT_MAX ||
        out->bytes < out_values * sizeof(float) ||
        latent->bytes < (uint64_t)rows * heads * latent_dim * sizeof(float)) {
        return 0;
    }
    const __nv_bfloat16 *v_b = (const __nv_bfloat16 *)cuda_model_range_ptr(
        map, v_b_offset, weight_bytes, "ling3vl v_b");
    if (!v_b) { return 0; }
    const dim3 grid(heads, rows);
    ling3vl_value_project_bf16<<<grid, 256, latent_dim * sizeof(float),
                                 ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)latent->ptr, v_b, rows, heads,
        latent_dim, value_dim);
    return cuda_ok(cudaGetLastError(), "Ling-3.0 MLA value projection");
}

/* cuBLAS picks TMA kernels for the expansion GEMMs, and TMA faults on the
 * raw ATS host pointer that `mapped` base residency serves, so the expanded
 * path is admitted only when both weights resolve to device or managed
 * memory.  One probe per (map, offset); seven MLA layers, two weights each. */
extern "C" int ds4_gpu_ling3vl_expand_ready(
        const void *map, uint64_t size, uint64_t k_b_offset,
        uint64_t v_b_offset, uint32_t heads, uint32_t latent_dim,
        uint32_t qk_nope, uint32_t value_dim) {
    enum { PROBES = 2 * LING_MLA_LAYERS };
    struct probe { const void *map; uint64_t offset; int ok; };
    static struct probe cache[PROBES];
    static unsigned cached = 0;
    const uint64_t bytes[2] = {
        (uint64_t)heads * latent_dim * qk_nope * sizeof(__nv_bfloat16),
        (uint64_t)heads * value_dim * latent_dim * sizeof(__nv_bfloat16),
    };
    const uint64_t offsets[2] = {k_b_offset, v_b_offset};
    for (int w = 0; w < 2; w++) {
        int ok = -1;
        for (unsigned i = 0; i < cached; i++) {
            if (cache[i].map == map && cache[i].offset == offsets[w]) {
                ok = cache[i].ok;
                break;
            }
        }
        if (ok < 0) {
            if (offsets[w] > size || bytes[w] > size - offsets[w]) { return 0; }
            const void *ptr = cuda_model_range_ptr(map, offsets[w], bytes[w],
                                                   "ling3vl expand weight");
            cudaPointerAttributes attr;
            ok = ptr && cudaPointerGetAttributes(&attr, ptr) == cudaSuccess &&
                (attr.type == cudaMemoryTypeDevice ||
                 attr.type == cudaMemoryTypeManaged);
            (void)cudaGetLastError();
            if (!ok) {
                fprintf(stderr, "ds4: Ling expanded MLA prefill off: weight at "
                        "%llu is not device-resident; absorbed path\n",
                        (unsigned long long)offsets[w]);
            }
            if (cached < PROBES) {
                cache[cached++] = (struct probe){map, offsets[w], ok};
            }
        }
        if (!ok) { return 0; }
    }
    return 1;
}

/* Expanded-MLA prefill operand: one latent-cache segment [slot0, slot0+rows)
 * becomes per-head K = [latent . k_b[h] | k_pe] (qk_nope + qk_rope wide)
 * and V = latent . v_b[h]^T (value_dim wide), FP32, [rows][heads][dim].
 * Two BF16 strided-batched GEMMs (one batch per head) read the cache rows
 * in place, so nothing is converted or copied except the shared k_pe tail.
 *
 * Row-major C_h = latent . W_h is cuBLAS's column-major C_h^T = W_h^T .
 * latent^T: k_b[h] is stored [latent][qk_nope], already that W_h^T; v_b[h]
 * is [value][latent] and goes through op T.  The per-batch C offset is
 * one head's width inside the [rows][heads][dim] row. */
extern "C" int ds4_gpu_ling3vl_expand_kv(
        ds4_gpu_tensor *k_full, ds4_gpu_tensor *value,
        const ds4_gpu_tensor *latent_cache, const ds4_gpu_tensor *k_pe_cache,
        const void *map, uint64_t size, uint64_t k_b_offset,
        uint64_t v_b_offset, uint32_t slot0, uint32_t rows, uint32_t heads,
        uint32_t latent_dim, uint32_t qk_nope, uint32_t qk_rope,
        uint32_t value_dim, int kv_bf16) {
    const uint32_t key_dim = qk_nope + qk_rope;
    const uint64_t elem = kv_bf16 ? sizeof(__nv_bfloat16) : sizeof(float);
    const cudaDataType_t ctype = kv_bf16 ? CUDA_R_16BF : CUDA_R_32F;
    const uint64_t k_b_bytes =
        (uint64_t)heads * latent_dim * qk_nope * sizeof(__nv_bfloat16);
    const uint64_t v_b_bytes =
        (uint64_t)heads * value_dim * latent_dim * sizeof(__nv_bfloat16);
    const uint64_t end = (uint64_t)slot0 + rows;
    if (!k_full || !value || !latent_cache || !k_pe_cache || !map || !rows ||
        !heads || !g_cublas_ready || (qk_rope & 3u) || rows > INT_MAX ||
        heads * key_dim > INT_MAX ||
        k_b_offset > size || k_b_bytes > size - k_b_offset ||
        v_b_offset > size || v_b_bytes > size - v_b_offset ||
        k_full->bytes < (uint64_t)rows * heads * key_dim * elem ||
        value->bytes < (uint64_t)rows * heads * value_dim * elem ||
        latent_cache->bytes < end * latent_dim * sizeof(__nv_bfloat16) ||
        k_pe_cache->bytes < end * qk_rope * sizeof(__nv_bfloat16)) { return 0; }
    const __nv_bfloat16 *k_b = (const __nv_bfloat16 *)cuda_model_range_ptr(
        map, k_b_offset, k_b_bytes, "ling3vl k_b");
    const __nv_bfloat16 *v_b = (const __nv_bfloat16 *)cuda_model_range_ptr(
        map, v_b_offset, v_b_bytes, "ling3vl v_b");
    if (!k_b || !v_b) { return 0; }
    const __nv_bfloat16 *latent =
        (const __nv_bfloat16 *)latent_cache->ptr + (uint64_t)slot0 * latent_dim;
    const float alpha = 1.0f, beta = 0.0f;

    cublasStatus_t st = cublasGemmStridedBatchedEx(
        g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)qk_nope, (int)rows,
        (int)latent_dim, &alpha,
        k_b, CUDA_R_16BF, (int)qk_nope, (long long)latent_dim * qk_nope,
        latent, CUDA_R_16BF, (int)latent_dim, 0ll,
        &beta, k_full->ptr, ctype, (int)(heads * key_dim),
        (long long)key_dim, (int)heads, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (!cublas_ok(st, "Ling-3.0 MLA expand K")) { return 0; }

    st = cublasGemmStridedBatchedEx(
        g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)value_dim, (int)rows,
        (int)latent_dim, &alpha,
        v_b, CUDA_R_16BF, (int)latent_dim, (long long)value_dim * latent_dim,
        latent, CUDA_R_16BF, (int)latent_dim, 0ll,
        &beta, value->ptr, ctype, (int)(heads * value_dim),
        (long long)value_dim, (int)heads, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (!cublas_ok(st, "Ling-3.0 MLA expand V")) { return 0; }

    const uint64_t quads = (uint64_t)rows * heads * (qk_rope / 4u);
    const unsigned blocks = (unsigned)((quads + 255u) / 256u);
    const __nv_bfloat16 *k_pe = (const __nv_bfloat16 *)k_pe_cache->ptr;
    if (kv_bf16) {
        ling3vl_expand_k_pe<__nv_bfloat16><<<blocks, 256, 0, ds4_current_stream()>>>(
            (__nv_bfloat16 *)k_full->ptr, k_pe, slot0, rows, heads, key_dim,
            qk_nope, qk_rope);
    } else {
        ling3vl_expand_k_pe<float><<<blocks, 256, 0, ds4_current_stream()>>>(
            (float *)k_full->ptr, k_pe, slot0, rows, heads, key_dim, qk_nope,
            qk_rope);
    }
    return cuda_ok(cudaGetLastError(), "Ling-3.0 MLA expand k_pe");
}

/* Range attention over one expanded segment.  BF16 K/V take the Ling
 * kernel (64-key tiles; DS4_LING3VL_MLA_TK=32 restores the Motif tile);
 * FP32 K/V take Motif's range kernel unchanged. */
extern "C" int ds4_gpu_ling3vl_expanded_attn(
        ds4_gpu_tensor *out, ds4_gpu_tensor *lse, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k_full, const ds4_gpu_tensor *value,
        uint32_t n_query, uint32_t query_pos0, uint32_t n_kv,
        uint32_t kv_pos0, uint32_t heads, uint32_t key_dim,
        uint32_t value_dim, float scale, int kv_bf16) {
    if (!kv_bf16) {
        return ds4_gpu_motif3_expanded_attention_range_tensor(
            out, lse, q, k_full, value, n_query, query_pos0, n_kv, kv_pos0,
            heads, heads, key_dim, value_dim, scale, 0u);
    }
    static int tk = -1;
    if (tk < 0) {
        const char *env = getenv("DS4_LING3VL_MLA_TK");
        tk = env && env[0] == '3' ? 32 : 64;
    }
    const uint64_t elem = sizeof(__nv_bfloat16);
    if (!out || !lse || !q || !k_full || !value || !n_query || !n_kv ||
        !heads || kv_pos0 > query_pos0 ||
        out->bytes < (uint64_t)n_query * heads * value_dim * sizeof(float) ||
        lse->bytes < (uint64_t)n_query * heads * sizeof(float) ||
        q->bytes < (uint64_t)n_query * heads * key_dim * sizeof(float) ||
        k_full->bytes < (uint64_t)n_kv * heads * key_dim * elem ||
        value->bytes < (uint64_t)n_kv * heads * value_dim * elem) { return 0; }
    const int rc = ds4_mmq_ling3vl_prefill_attn_hmma(
        (float *)out->ptr, (float *)lse->ptr, (const float *)q->ptr,
        k_full->ptr, value->ptr, (int)n_query, (int)query_pos0, (int)n_kv,
        (int)kv_pos0, (int)heads, (int)key_dim, (int)value_dim, scale, tk,
        ds4_current_stream());
    /* Launcher already consumed CUDA status: -1 rejected shape, -2 launch
     * fail. Neither leaves an error for cudaGetLastError() to re-read. */
    if (rc != 0) {
        fprintf(stderr, "ds4: Ling-3.0 MLA expanded attention %s\n",
                rc == -1 ? "rejected the shape" : "launch failed");
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_ling3vl_rms_norm(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *in,
        const void *map, uint64_t size, uint64_t offset,
        uint32_t dim, uint32_t in_stride, uint32_t rows, float eps) {
    const uint64_t weight_bytes = (uint64_t)dim * sizeof(float);
    if (!out || !in || !map || !dim || !rows || in_stride < dim ||
        offset > size || weight_bytes > size - offset ||
        out->bytes < (uint64_t)rows * dim * sizeof(float) ||
        in->bytes < (uint64_t)rows * in_stride * sizeof(float)) { return 0; }
    const float *w = (const float *)cuda_model_range_ptr(
        map, offset, weight_bytes, "ling3vl kv_a_norm");
    if (!w) { return 0; }
    ling3vl_rms_norm<<<rows, 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)in->ptr, w, dim, in_stride, rows, eps);
    return cuda_ok(cudaGetLastError(), "Ling-3.0 latent RMSNorm");
}

extern "C" int ds4_gpu_ling3vl_store_latent(
        ds4_gpu_tensor *latent_cache, ds4_gpu_tensor *k_pe_cache,
        const ds4_gpu_tensor *kv_norm, const ds4_gpu_tensor *kv_raw,
        uint32_t rows, uint32_t pos0, uint32_t cache_cap,
        uint32_t kv_raw_dim, uint32_t latent_dim, uint32_t rope_dim) {
    const uint64_t width = (uint64_t)latent_dim + rope_dim;
    const uint64_t count = (uint64_t)rows * width;
    if (!latent_cache || !k_pe_cache || !kv_norm || !kv_raw || !rows ||
        width != kv_raw_dim || pos0 > cache_cap || rows > cache_cap - pos0 ||
        count > INT_MAX ||
        latent_cache->bytes <
            (uint64_t)cache_cap * latent_dim * sizeof(__nv_bfloat16) ||
        k_pe_cache->bytes <
            (uint64_t)cache_cap * rope_dim * sizeof(__nv_bfloat16) ||
        kv_norm->bytes < (uint64_t)rows * latent_dim * sizeof(float) ||
        kv_raw->bytes < (uint64_t)rows * kv_raw_dim * sizeof(float)) { return 0; }
    ling3vl_store_latent<<<(count + 255u) / 256u, 256, 0, ds4_current_stream()>>>(
        (__nv_bfloat16 *)latent_cache->ptr, (__nv_bfloat16 *)k_pe_cache->ptr,
        (const float *)kv_norm->ptr, (const float *)kv_raw->ptr,
        rows, pos0, cache_cap, kv_raw_dim, latent_dim, rope_dim);
    return cuda_ok(cudaGetLastError(), "Ling-3.0 latent KV store");
}

extern "C" int ds4_gpu_ling3vl_vision_patch_position(
        ds4_gpu_tensor *hidden, const void *map, uint64_t size,
        uint64_t bias_offset, uint64_t position_offset,
        const ds4_gpu_tensor *indices, const ds4_gpu_tensor *weights,
        uint32_t rows, uint32_t dim, uint32_t positions) {
    const uint64_t values = (uint64_t)rows * dim;
    const uint64_t bias_bytes = (uint64_t)dim * sizeof(float);
    const uint64_t table_bytes = (uint64_t)positions * dim * sizeof(float);
    if (!hidden || !indices || !weights || !map || !rows || !dim || !positions ||
        values > INT_MAX || bias_offset > size || bias_bytes > size - bias_offset ||
        position_offset > size || table_bytes > size - position_offset ||
        hidden->bytes < values * sizeof(float) ||
        indices->bytes < (uint64_t)rows * 4u * sizeof(int32_t) ||
        weights->bytes < (uint64_t)rows * 4u * sizeof(float)) { return 0; }
    const float *bias = (const float *)cuda_model_range_ptr(
        map, bias_offset, bias_bytes, "ling3vl patch bias");
    const float *table = (const float *)cuda_model_range_ptr(
        map, position_offset, table_bytes, "ling3vl position table");
    if (!bias || !table) { return 0; }
    ling3vl_vision_patch_position<<<(values + 255u) / 256u, 256, 0,
                                    ds4_current_stream()>>>(
        (float *)hidden->ptr, bias, table, (const int32_t *)indices->ptr,
        (const float *)weights->ptr, rows, dim);
    return cuda_ok(cudaGetLastError(), "Ling-3.0 vision patch/position");
}

enum {
    L3V_GEMV_WARPS = 8u,
    L3V_GEMV_THREADS = L3V_GEMV_WARPS * 32u,
    /* in_dim multiple of 256 => vecs multiple of 32, at most 16 chunks
     * (in_dim <= 4096).  Covers hidden 2560, attn 4096, expert 768. */
    L3V_GEMV_CHUNK = 256u,
    L3V_GEMV_XREG_MAX = 16u
};

/* Lane-strided uint4 dots, XOR-tree: same order as stable-rows warp GEMV. */
__device__ __forceinline__ static float ling3vl_gemv_reduce(float sum) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    return sum;
}

/* Cache x in registers; fully unrolled K-loop. */
template<uint32_t CHUNKS>
__global__ static void ling3vl_gemv_xreg_kernel(
        float *out, const __nv_bfloat16 *w, const __nv_bfloat16 *x,
        uint32_t in_dim, uint32_t out_dim) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row =
        (uint32_t)blockIdx.x * L3V_GEMV_WARPS + (threadIdx.x >> 5u);
    if (row >= out_dim) {
        return;
    }
    const uint4 *xr = reinterpret_cast<const uint4 *>(x);
    const uint4 *wr = reinterpret_cast<const uint4 *>(
        w + (uint64_t)row * in_dim);
    uint4 xv[CHUNKS];
#pragma unroll
    for (uint32_t k = 0; k < CHUNKS; k++) {
        xv[k] = xr[lane + k * 32u];
    }
    float sum = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < CHUNKS; k++) {
        sum += bf16x8_dot(__ldg(wr + lane + k * 32u), xv[k]);
    }
    sum = ling3vl_gemv_reduce(sum);
    if (lane == 0) {
        out[row] = sum;
    }
}

template<uint32_t CHUNKS>
__global__ static void ling3vl_gemv_pair_xreg_kernel(
        float *out0, float *out1, const __nv_bfloat16 *w0,
        const __nv_bfloat16 *w1, const __nv_bfloat16 *x,
        uint32_t in_dim, uint32_t out0_dim, uint32_t out1_dim) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t pair_dim = out0_dim + out1_dim;
    const uint32_t pair_row =
        (uint32_t)blockIdx.x * L3V_GEMV_WARPS + (threadIdx.x >> 5u);
    if (pair_row >= pair_dim) {
        return;
    }
    const bool second = pair_row >= out0_dim;
    const uint32_t row = second ? pair_row - out0_dim : pair_row;
    const __nv_bfloat16 *w = second ? w1 : w0;
    const uint4 *xr = reinterpret_cast<const uint4 *>(x);
    const uint4 *wr = reinterpret_cast<const uint4 *>(
        w + (uint64_t)row * in_dim);
    uint4 xv[CHUNKS];
#pragma unroll
    for (uint32_t k = 0; k < CHUNKS; k++) {
        xv[k] = xr[lane + k * 32u];
    }
    float sum = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < CHUNKS; k++) {
        sum += bf16x8_dot(__ldg(wr + lane + k * 32u), xv[k]);
    }
    sum = ling3vl_gemv_reduce(sum);
    if (lane == 0) {
        (second ? out1 : out0)[row] = sum;
    }
}

#define L3V_GEMV_LAUNCH(chunks, kernel, grid, ...)                             \
    do {                                                                       \
        switch (chunks) {                                                      \
        case 3u:                                                               \
            kernel<3u><<<grid, L3V_GEMV_THREADS, 0, ds4_current_stream()>>>(    \
                __VA_ARGS__);                                                  \
            break;                                                             \
        case 10u:                                                              \
            kernel<10u><<<grid, L3V_GEMV_THREADS, 0, ds4_current_stream()>>>(   \
                __VA_ARGS__);                                                  \
            break;                                                             \
        case 16u:                                                              \
            kernel<16u><<<grid, L3V_GEMV_THREADS, 0, ds4_current_stream()>>>(   \
                __VA_ARGS__);                                                  \
            break;                                                             \
        default:                                                               \
            return 0;                                                          \
        }                                                                      \
    } while (0)

static bool ling3vl_gemv_xreg_off(void) {
    const char *env = getenv("DS4_LING3VL_NO_GEMV_XREG");
    return env && env[0] == '1';
}

static uint32_t ling3vl_gemv_chunks(uint64_t in_dim) {
    if ((in_dim % L3V_GEMV_CHUNK) != 0u) {
        return 0;
    }
    const uint32_t chunks = (uint32_t)(in_dim / L3V_GEMV_CHUNK);
    return (chunks == 3u || chunks == 10u || chunks == 16u) ? chunks : 0u;
}

static __nv_bfloat16 *ling3vl_gemv_convert(
        const ds4_gpu_tensor *x, uint64_t in_dim, int tier) {
    if (((uintptr_t)x->ptr & 15u)) {
        return NULL;
    }
    __nv_bfloat16 *xb = (__nv_bfloat16 *)cuda_tmp_alloc_on(
        tier, in_dim * sizeof(__nv_bfloat16), "ling3vl gemv x");
    if (!xb || ((uintptr_t)xb & 15u)) {
        return NULL;
    }
    f32_to_bf16_kernel<<<(in_dim + 255u) / 256u, 256, 0, ds4_current_stream()>>>(
        xb, (const float *)x->ptr, in_dim);
    if (!cuda_ok(cudaGetLastError(), "ling3vl gemv convert")) {
        return NULL;
    }
    return xb;
}

static int ling3vl_gemv_xreg(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x) {
    const uint32_t chunks = ling3vl_gemv_chunks(in_dim);
    if (!chunks || !out || !x || !model_map || !out_dim ||
        weight_offset > model_size || out_dim > UINT64_MAX / in_dim) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out->bytes < out_dim * sizeof(float)) {
        return 0;
    }
    const int tier = ds4_tensor_device_idx(out);
    const char *wptr = cuda_resolve_weight_ptr(
        model_map, weight_offset, weight_bytes, tier, "ling3vl gemv");
    if (!wptr || ((uintptr_t)wptr & 15u)) {
        return 0;
    }
    __nv_bfloat16 *xb = ling3vl_gemv_convert(x, in_dim, tier);
    if (!xb) {
        return 0;
    }
    const uint32_t blocks =
        (uint32_t)((out_dim + L3V_GEMV_WARPS - 1u) / L3V_GEMV_WARPS);
    L3V_GEMV_LAUNCH(chunks, ling3vl_gemv_xreg_kernel, blocks,
                    (float *)out->ptr, (const __nv_bfloat16 *)wptr, xb,
                    (uint32_t)in_dim, (uint32_t)out_dim);
    return cuda_ok(cudaGetLastError(), "ling3vl gemv xreg");
}

extern "C" int ds4_gpu_ling3vl_gemv_pair(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
        const void *model_map, uint64_t model_size,
        uint64_t off0, uint64_t off1, uint64_t in_dim,
        uint64_t out0_dim, uint64_t out1_dim, const ds4_gpu_tensor *x) {
    const uint32_t chunks = ling3vl_gemv_chunks(in_dim);
    if (ling3vl_gemv_xreg_off() || !chunks) {
        return ds4_gpu_matmul_bf16_stable_rows_pair_tensor(
            out0, out1, model_map, model_size, off0, off1, in_dim, out0_dim,
            out1_dim, x, 1u);
    }
    if (!out0 || !out1 || !x || !model_map || !out0_dim || !out1_dim ||
        off0 > model_size || off1 > model_size ||
        out0_dim > UINT64_MAX / in_dim || out1_dim > UINT64_MAX / in_dim ||
        ds4_tensor_device_idx(out0) != ds4_tensor_device_idx(out1)) {
        return 0;
    }
    const uint64_t bytes0 = out0_dim * in_dim * sizeof(uint16_t);
    const uint64_t bytes1 = out1_dim * in_dim * sizeof(uint16_t);
    if (bytes0 > model_size - off0 || bytes1 > model_size - off1 ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out0_dim * sizeof(float) ||
        out1->bytes < out1_dim * sizeof(float)) {
        return 0;
    }
    const int tier = ds4_tensor_device_idx(out0);
    const __nv_bfloat16 *w0 = (const __nv_bfloat16 *)cuda_resolve_weight_ptr(
        model_map, off0, bytes0, tier, "ling3vl gemv pair0");
    const __nv_bfloat16 *w1 = (const __nv_bfloat16 *)cuda_resolve_weight_ptr(
        model_map, off1, bytes1, tier, "ling3vl gemv pair1");
    __nv_bfloat16 *xb = ling3vl_gemv_convert(x, in_dim, tier);
    if (!w0 || !w1 || !xb || ((uintptr_t)w0 & 15u) || ((uintptr_t)w1 & 15u)) {
        return 0;
    }
    const uint32_t pair_dim = (uint32_t)(out0_dim + out1_dim);
    const uint32_t blocks = (pair_dim + L3V_GEMV_WARPS - 1u) / L3V_GEMV_WARPS;
    L3V_GEMV_LAUNCH(chunks, ling3vl_gemv_pair_xreg_kernel, blocks,
                    (float *)out0->ptr, (float *)out1->ptr, w0, w1, xb,
                    (uint32_t)in_dim, (uint32_t)out0_dim,
                    (uint32_t)out1_dim);
    return cuda_ok(cudaGetLastError(), "ling3vl gemv pair xreg");
}

extern "C" int ds4_gpu_ling3vl_matmul_bf16(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    const char *vec_env = getenv("DS4_LING3VL_NO_BF16_VEC");
    const bool no_vec = vec_env && vec_env[0] == '1';
    if (n_tok == 1u && !no_vec) {
        if (!ling3vl_gemv_xreg_off() &&
            ling3vl_gemv_xreg(out, model_map, model_size, weight_offset, in_dim,
                              out_dim, x)) {
            return 1;
        }
        return ds4_gpu_matmul_bf16_stable_rows_tensor(
            out, model_map, model_size, weight_offset, in_dim, out_dim, x,
            n_tok);
    }
    return ds4_gpu_matmul_bf16_tensor(
        out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok);
}

/* n=1 F32: one warp per row instead of a 256-thread block per row. */
__global__ static void ling3vl_gemv_f32_kernel(
        float *out, const float *w, const float *x,
        uint32_t in_dim, uint32_t out_dim) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row =
        (uint32_t)blockIdx.x * L3V_GEMV_WARPS + (threadIdx.x >> 5u);
    if (row >= out_dim) {
        return;
    }
    const float *wr = w + (uint64_t)row * in_dim;
    float sum = 0.0f;
    if ((in_dim & 3u) == 0u) {
        const float4 *wr4 = reinterpret_cast<const float4 *>(wr);
        const float4 *x4 = reinterpret_cast<const float4 *>(x);
        const uint32_t vecs = in_dim >> 2u;
        for (uint32_t i = lane; i < vecs; i += 32u) {
            const float4 a = wr4[i];
            const float4 b = x4[i];
            sum = fmaf(a.x, b.x, sum);
            sum = fmaf(a.y, b.y, sum);
            sum = fmaf(a.z, b.z, sum);
            sum = fmaf(a.w, b.w, sum);
        }
    } else {
        for (uint32_t i = lane; i < in_dim; i += 32u) {
            sum = fmaf(wr[i], x[i], sum);
        }
    }
    sum = ling3vl_gemv_reduce(sum);
    if (lane == 0) {
        out[row] = sum;
    }
}

extern "C" int ds4_gpu_ling3vl_matmul_f32(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    const char *kill = getenv("DS4_LING3VL_NO_F32_VEC");
    if (n_tok != 1u || (kill && kill[0] == '1')) {
        return 0;
    }
    if (!out || !x || !model_map || !in_dim || !out_dim ||
        weight_offset > model_size || out_dim > UINT64_MAX / in_dim) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(float);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out->bytes < out_dim * sizeof(float) ||
        ((uintptr_t)x->ptr & 15u)) {
        return 0;
    }
    const int tier = ds4_tensor_device_idx(out);
    const char *wptr = cuda_resolve_weight_ptr(
        model_map, weight_offset, weight_bytes, tier, "ling3vl f32 gemv");
    if (!wptr || ((uintptr_t)wptr & 15u)) {
        return 0;
    }
    const uint32_t blocks =
        (uint32_t)((out_dim + L3V_GEMV_WARPS - 1u) / L3V_GEMV_WARPS);
    ling3vl_gemv_f32_kernel<<<blocks, L3V_GEMV_THREADS, 0,
                              ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)wptr, (const float *)x->ptr,
        (uint32_t)in_dim, (uint32_t)out_dim);
    return cuda_ok(cudaGetLastError(), "ling3vl gemv f32");
}

enum {
    L3V_MLA_H = 8u,
    L3V_MLA_KV = 16u,
    L3V_MLA_LAT = 512u,
    L3V_MLA_ROPE = 64u,
    L3V_MLA_HMMA_MIN_ROWS = 8u,
    L3V_MLA_HMMA_HEADS = 32u
};

__device__ __forceinline__ static void ling3vl_mla_issue_kv(
        __nv_bfloat16 lat[L3V_MLA_KV][L3V_MLA_LAT],
        __nv_bfloat16 kpe[L3V_MLA_KV][L3V_MLA_ROPE],
        const __nv_bfloat16 *latent_cache, const __nv_bfloat16 *k_pe_cache,
        uint32_t tile, uint32_t nrows, uint32_t tid, uint32_t cache_cap) {
    for (uint32_t i = tid; i < L3V_MLA_KV * 64u; i += 256u) {
        const uint32_t row = i / 64u;
        const uint32_t c8 = i % 64u;
        const uint32_t slot = tile + row;
        const bool pred = row < nrows && slot < cache_cap;
        const __nv_bfloat16 *src =
            latent_cache + (uint64_t)slot * L3V_MLA_LAT + c8 * 8u;
        tt_cp_async_16B(&lat[row][c8 * 8u], src, pred);
    }
    if (tid < 128u) {
        const uint32_t row = tid / 8u;
        const uint32_t c8 = tid % 8u;
        const uint32_t slot = tile + row;
        const bool pred = row < nrows && slot < cache_cap;
        const __nv_bfloat16 *src =
            k_pe_cache + (uint64_t)slot * L3V_MLA_ROPE + c8 * 8u;
        tt_cp_async_16B(&kpe[row][c8 * 8u], src, pred);
    }
    tt_cp_async_commit();
}

/* One head, eight queries.  They share each 16-row KV tile so historical
 * keys move through shared memory once.  Keys at or past a query's causal
 * end are skipped; softmax order for kept keys matches Motif's warp kernel. */
__global__ static void ling3vl_mla_prefill_kvtile_kernel(
        float *out, const float *q, const float *q_absorbed,
        const __nv_bfloat16 *latent_cache, const __nv_bfloat16 *k_pe_cache,
        uint32_t rows, uint32_t pos0, uint32_t cache_cap, uint32_t q_heads,
        uint32_t qk_nope, float scale) {
    __shared__ __nv_bfloat16 lat[2][L3V_MLA_KV][L3V_MLA_LAT];
    __shared__ __nv_bfloat16 kpe[2][L3V_MLA_KV][L3V_MLA_ROPE];

    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t head = blockIdx.x;
    const uint32_t q0 = blockIdx.y * L3V_MLA_H;
    const uint32_t token = q0 + warp;
    const bool live = head < q_heads && token < rows;
    const uint32_t end = live ? pos0 + token + 1u : 0u;
    const uint32_t group_hi = pos0 + min(q0 + L3V_MLA_H, rows);
    const uint32_t key_dim = qk_nope + L3V_MLA_ROPE;

    float4 low0 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 low1 = low0, low2 = low0, low3 = low0;
    float qrope[4] = {0.f, 0.f, 0.f, 0.f};
    if (live) {
        const float4 *low4 = (const float4 *)(q_absorbed +
            ((uint64_t)token * q_heads + head) * L3V_MLA_LAT);
        low0 = low4[lane];
        low1 = low4[lane + 32u];
        low2 = low4[lane + 64u];
        low3 = low4[lane + 96u];
        const float *qh =
            q + ((uint64_t)token * q_heads + head) * key_dim;
        if (lane < 16u) {
            qrope[0] = qh[qk_nope + lane * 4u + 0u];
            qrope[1] = qh[qk_nope + lane * 4u + 1u];
            qrope[2] = qh[qk_nope + lane * 4u + 2u];
            qrope[3] = qh[qk_nope + lane * 4u + 3u];
        }
    }

    float M = -FLT_MAX / 2.0f;
    float S = 0.0f;
    float4 o0 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 o1 = o0, o2 = o0, o3 = o0;
    const uint32_t i0 = lane * 4u;
    const uint32_t i1 = (lane + 32u) * 4u;
    const uint32_t i2 = (lane + 64u) * 4u;
    const uint32_t i3 = (lane + 96u) * 4u;

    uint32_t buf = 0u;
    if (group_hi > 0u) {
        ling3vl_mla_issue_kv(lat[0], kpe[0], latent_cache, k_pe_cache, 0u,
                             min((uint32_t)L3V_MLA_KV, group_hi), tid, cache_cap);
        tt_cp_async_wait_group<0>();
    }
    __syncthreads();

    for (uint32_t tile = 0u; tile < group_hi; tile += L3V_MLA_KV) {
        const uint32_t nrows = min((uint32_t)L3V_MLA_KV, group_hi - tile);
        const uint32_t next = tile + L3V_MLA_KV;
        const uint32_t nrows_n =
            next < group_hi ? min((uint32_t)L3V_MLA_KV, group_hi - next) : 0u;
        const uint32_t nbuf = buf ^ 1u;
        if (nrows_n) {
            ling3vl_mla_issue_kv(lat[nbuf], kpe[nbuf], latent_cache, k_pe_cache,
                                 next, nrows_n, tid, cache_cap);
        }

        if (live) {
            for (uint32_t row = 0u; row < nrows; row++) {
                if (tile + row >= end) { break; }
                const float4 k0 = motif3_load_bf16x4(&lat[buf][row][i0]);
                const float4 k1 = motif3_load_bf16x4(&lat[buf][row][i1]);
                const float4 k2 = motif3_load_bf16x4(&lat[buf][row][i2]);
                const float4 k3 = motif3_load_bf16x4(&lat[buf][row][i3]);
                float partial =
                    low0.x * k0.x + low0.y * k0.y + low0.z * k0.z + low0.w * k0.w +
                    low1.x * k1.x + low1.y * k1.y + low1.z * k1.z + low1.w * k1.w +
                    low2.x * k2.x + low2.y * k2.y + low2.z * k2.z + low2.w * k2.w +
                    low3.x * k3.x + low3.y * k3.y + low3.z * k3.z + low3.w * k3.w;
                if (lane < 16u) {
                    const float4 kp = motif3_load_bf16x4(&kpe[buf][row][lane * 4u]);
                    partial += qrope[0] * kp.x + qrope[1] * kp.y +
                               qrope[2] * kp.z + qrope[3] * kp.w;
                }
                for (uint32_t off = 16u; off > 0u; off >>= 1u) {
                    partial += __shfl_xor_sync(0xffffffffu, partial, off);
                }
                const float score = partial * scale;
                const float new_m = fmaxf(M, score);
                const float old_scale = expf(M - new_m);
                const float row_scale = expf(score - new_m);
#define L3V_ON4(o, k) do {                                                     \
                    (o).x = (o).x * old_scale + (k).x * row_scale;             \
                    (o).y = (o).y * old_scale + (k).y * row_scale;             \
                    (o).z = (o).z * old_scale + (k).z * row_scale;             \
                    (o).w = (o).w * old_scale + (k).w * row_scale;             \
                } while (0)
                L3V_ON4(o0, k0); L3V_ON4(o1, k1);
                L3V_ON4(o2, k2); L3V_ON4(o3, k3);
#undef L3V_ON4
                S = S * old_scale + row_scale;
                M = new_m;
            }
        }
        if (nrows_n) {
            tt_cp_async_wait_group<0>();
        }
        __syncthreads();
        buf = nbuf;
    }

    if (!live) { return; }
    const float inv_s = S > 0.0f ? 1.0f / S : 0.0f;
    float4 *dst = (float4 *)(out +
        ((uint64_t)token * q_heads + head) * L3V_MLA_LAT);
#define L3V_ST4(idx, o) do {                                                   \
        (o).x *= inv_s; (o).y *= inv_s; (o).z *= inv_s; (o).w *= inv_s;        \
        dst[(idx)] = (o);                                                      \
    } while (0)
    L3V_ST4(lane, o0); L3V_ST4(lane + 32u, o1);
    L3V_ST4(lane + 64u, o2); L3V_ST4(lane + 96u, o3);
#undef L3V_ST4
}

static int ling3vl_mla_args_ok(
        const ds4_gpu_tensor *latent_out, const ds4_gpu_tensor *q_full,
        const ds4_gpu_tensor *q_absorbed,
        const ds4_gpu_tensor *kv_latent_cache,
        const ds4_gpu_tensor *k_pe_cache,
        uint32_t rows, uint32_t pos0, uint32_t cache_cap,
        uint32_t q_heads, uint32_t kv_latent_dim,
        uint32_t qk_nope, uint32_t qk_rope) {
    if (!latent_out || !q_full || !q_absorbed || !kv_latent_cache ||
        !k_pe_cache || !rows || !cache_cap || pos0 > cache_cap ||
        rows > cache_cap - pos0) {
        return 0;
    }
    const uint64_t key_dim = (uint64_t)qk_nope + qk_rope;
    if (latent_out->bytes <
            (uint64_t)rows * q_heads * kv_latent_dim * sizeof(float) ||
        q_full->bytes < (uint64_t)rows * q_heads * key_dim * sizeof(float) ||
        q_absorbed->bytes <
            (uint64_t)rows * q_heads * kv_latent_dim * sizeof(float) ||
        kv_latent_cache->bytes <
            (uint64_t)cache_cap * kv_latent_dim * sizeof(__nv_bfloat16) ||
        k_pe_cache->bytes <
            (uint64_t)cache_cap * qk_rope * sizeof(__nv_bfloat16)) {
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_ling3vl_latent_attn(
        ds4_gpu_tensor *latent_out, const ds4_gpu_tensor *q_full,
        const ds4_gpu_tensor *q_absorbed,
        const ds4_gpu_tensor *kv_latent_cache,
        const ds4_gpu_tensor *k_pe_cache,
        uint32_t rows, uint32_t pos0, uint32_t cache_cap,
        uint32_t window, uint32_t q_heads, uint32_t kv_latent_dim,
        uint32_t qk_nope, uint32_t qk_rope, float scale) {
    const char *tile_env = getenv("DS4_LING3VL_MLA_TILE");
    const bool want_tile = tile_env && tile_env[0] == '1';
    const bool tile = want_tile && rows >= 16u && window == 0u &&
        kv_latent_dim == L3V_MLA_LAT && qk_rope == L3V_MLA_ROPE &&
        (q_heads % L3V_MLA_H) == 0u && qk_nope == 128u;
    if (tile) {
        if (!ling3vl_mla_args_ok(
                latent_out, q_full, q_absorbed, kv_latent_cache, k_pe_cache,
                rows, pos0, cache_cap, q_heads, kv_latent_dim, qk_nope,
                qk_rope)) {
            return 0;
        }
        dim3 grid(q_heads, (rows + L3V_MLA_H - 1u) / L3V_MLA_H, 1);
        ling3vl_mla_prefill_kvtile_kernel<<<grid, 256, 0, ds4_current_stream()>>>(
            (float *)latent_out->ptr, (const float *)q_full->ptr,
            (const float *)q_absorbed->ptr,
            (const __nv_bfloat16 *)kv_latent_cache->ptr,
            (const __nv_bfloat16 *)k_pe_cache->ptr,
            rows, pos0, cache_cap, q_heads, qk_nope, scale);
        return cuda_ok(cudaGetLastError(), "Ling-3.0 MLA prefill KV tile");
    }

    /* Prefill: same absorbed-MLA layout as dots3 (shared BF16 latent + k_pe,
     * per-head Q).  Tensor-core FATTN owns 32 heads against one token's keys.
     * Decode and the kill switch stay on Motif (HG split at long context). */
    const char *no_hmma_env = getenv("DS4_LING3VL_NO_MLA_HMMA");
    const bool no_hmma = no_hmma_env && no_hmma_env[0] == '1';
    if (!no_hmma && rows >= L3V_MLA_HMMA_MIN_ROWS && window == 0u &&
        kv_latent_dim == L3V_MLA_LAT && qk_rope == L3V_MLA_ROPE &&
        (q_heads % L3V_MLA_HMMA_HEADS) == 0u) {
        if (!ling3vl_mla_args_ok(
                latent_out, q_full, q_absorbed, kv_latent_cache, k_pe_cache,
                rows, pos0, cache_cap, q_heads, kv_latent_dim, qk_nope,
                qk_rope)) {
            return 0;
        }
        const int rc = ds4_mmq_dots3_prefill_attn_hmma(
            (float *)latent_out->ptr, (const float *)q_full->ptr,
            (const float *)q_absorbed->ptr, kv_latent_cache->ptr,
            k_pe_cache->ptr, NULL, 0, (int)rows, (int)pos0, (int)cache_cap,
            (int)window, (int)q_heads, (int)kv_latent_dim, (int)qk_nope,
            (int)qk_rope, scale, ds4_current_stream());
        if (rc == 0) {
            return 1;
        }
        if (rc != -1) {
            return cuda_ok(cudaGetLastError(), "Ling-3.0 MLA prefill HMMA");
        }
    }

    return ds4_gpu_motif3_latent_attention_bf16_tensor(
        latent_out, q_full, q_absorbed, kv_latent_cache, k_pe_cache,
        rows, pos0, cache_cap, window, q_heads, kv_latent_dim,
        qk_nope, qk_rope, scale);
}
