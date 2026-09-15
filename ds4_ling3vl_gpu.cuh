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
        uint32_t section_t, uint32_t section_h) {
    const uint32_t half = rotary / 2u;
    if (!x || !positions || !inv_freq || !rows || !heads || !rotary ||
        (rotary & 1u) || offset + rotary > head_stride ||
        section_t + section_h > half ||
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
        section_t, section_t + section_h, pairs);
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
