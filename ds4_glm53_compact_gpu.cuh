#ifndef DS4_GLM53_COMPACT_GPU_CUH
#define DS4_GLM53_COMPACT_GPU_CUH

#include "ds4_glm53_compact.h"
#include "ds4_glm53_attn.h"

/* Compact GLM-5.3 algebra and indexer pooling follow antirez/ds4 at
 * 0aaea5a238fb41a35106a551e73c8409dfb751ac, ds4_cuda.cu:
 * glm_qk_lowrank_q8_0_batch_kernel, glm53_indexer_pool_update_kernel,
 * glm_indexer_scores_f32_kernel and glm53_expand_pool_selection_kernel.
 * https://github.com/antirez/ds4/blob/0aaea5a238fb41a35106a551e73c8409dfb751ac/ds4_cuda.cu
 * GLM-5.3 has no DSA rotary tail: QK = (Q K-b^T) latent and
 * heads = (softmax(QK/sqrt(head_dim)) latent) V-b^T.
 * Only persistent latent/pool rows round to FP16; arithmetic stays F32. */
enum {
    GLM53_COMPACT_Q8_BLOCK = 32,
    GLM53_COMPACT_Q8_BYTES = 34,
    GLM53_COMPACT_THREADS = 256,
    GLM53_COMPACT_MAX_DIM = 1024,
    GLM53_COMPACT_MAX_HEADS = 256,
    GLM53_COMPACT_MAX_ROWS = 65535
};

static bool glm53_compact_has(const ds4_gpu_tensor *tensor,
                              uint64_t count, uint64_t elem) {
    return tensor && tensor->ptr && count <= UINT64_MAX / elem &&
           tensor->bytes >= count * elem;
}

static bool glm53_compact_shape(uint32_t rows, uint32_t heads,
                                uint32_t latent_dim, uint32_t head_dim) {
    return rows && rows <= GLM53_COMPACT_MAX_ROWS && heads &&
           heads <= GLM53_COMPACT_MAX_HEADS && latent_dim && head_dim &&
           latent_dim <= GLM53_COMPACT_MAX_DIM &&
           head_dim <= GLM53_COMPACT_MAX_DIM &&
           latent_dim % GLM53_COMPACT_Q8_BLOCK == 0u &&
           head_dim % GLM53_COMPACT_Q8_BLOCK == 0u;
}

static const char *glm53_compact_weight(const void *map, uint64_t size,
        uint64_t offset, uint64_t bytes, const ds4_gpu_tensor *out,
        const char *label) {
    if (!map || offset > size || bytes > size - offset) { return NULL; }
    return cuda_resolve_weight_ptr(map, offset, bytes,
                                   ds4_tensor_device_idx(out), label);
}

__device__ __forceinline__ static float glm53_compact_q8(
        const char *row, uint32_t col) {
    const char *block = row + (uint64_t)(col / GLM53_COMPACT_Q8_BLOCK) *
                              GLM53_COMPACT_Q8_BYTES;
    return __half2float(*(const __half *)block) *
           (float)((const int8_t *)(block + sizeof(__half)))[
                    col % GLM53_COMPACT_Q8_BLOCK];
}

__global__ static void glm53_low_store_kernel(__half *cache,
        const float *latent, uint32_t rows, uint32_t pos0, uint32_t dim) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * dim) { return; }
    cache[(uint64_t)pos0 * dim + i] = __float2half_rn(latent[i]);
}

extern "C" int ds4_gpu_glm53_store_low(ds4_gpu_tensor *cache,
        const ds4_gpu_tensor *latent, uint32_t rows, uint32_t pos0,
        uint32_t cap, uint32_t latent_dim) {
    /* Position arguments cannot be frozen into a decode graph. */
    if (ds4_capture_active() || !glm53_compact_shape(rows, 1u, latent_dim, 32u) ||
        pos0 > cap || rows > cap - pos0 ||
        !glm53_compact_has(cache, (uint64_t)cap * latent_dim, sizeof(__half)) ||
        !glm53_compact_has(latent, (uint64_t)rows * latent_dim, sizeof(float))) {
        return 0;
    }
    const uint64_t count = (uint64_t)rows * latent_dim;
    glm53_low_store_kernel<<<(unsigned)((count + 255u) / 256u),
        GLM53_COMPACT_THREADS, 0, ds4_current_stream()>>>(
            (__half *)cache->ptr, (const float *)latent->ptr,
            rows, pos0, latent_dim);
    return cuda_ok(cudaGetLastError(), "GLM-5.3 compact latent store");
}

__global__ static void glm53_absorb_q_kernel(float *out, const float *q,
        const char *weight, uint32_t heads, uint32_t latent_dim,
        uint32_t head_dim, uint64_t row_bytes) {
    const uint32_t h = blockIdx.x;
    const uint32_t token = blockIdx.y;
    const float *query = q + ((uint64_t)token * heads + h) * head_dim;
    float *dst = out + ((uint64_t)token * heads + h) * latent_dim;
    for (uint32_t j = threadIdx.x; j < latent_dim; j += blockDim.x) {
        const char *row = weight + ((uint64_t)h * latent_dim + j) * row_bytes;
        float value = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
            value = fmaf(query[d], glm53_compact_q8(row, d), value);
        }
        dst[j] = value;
    }
}

extern "C" int ds4_gpu_glm53_absorb_q(ds4_gpu_tensor *low_q,
        const ds4_gpu_tensor *q, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t rows, uint32_t heads,
        uint32_t latent_dim, uint32_t head_dim) {
    if (!glm53_compact_shape(rows, heads, latent_dim, head_dim) ||
        !glm53_compact_has(low_q, (uint64_t)rows * heads * latent_dim, sizeof(float)) ||
        !glm53_compact_has(q, (uint64_t)rows * heads * head_dim, sizeof(float))) {
        return 0;
    }
    const uint64_t row_bytes = (uint64_t)head_dim /
        GLM53_COMPACT_Q8_BLOCK * GLM53_COMPACT_Q8_BYTES;
    const char *weight = glm53_compact_weight(model_map, model_size,
        weight_offset, (uint64_t)heads * latent_dim * row_bytes, low_q,
        "GLM-5.3 compact K-b Q8_0");
    if (!weight) { return 0; }
    glm53_absorb_q_kernel<<<dim3(heads, rows), 128u, 0, ds4_current_stream()>>>(
        (float *)low_q->ptr, (const float *)q->ptr, weight,
        heads, latent_dim, head_dim, row_bytes);
    return cuda_ok(cudaGetLastError(), "GLM-5.3 compact Q absorption");
}

#include "cuda/glm53_dense_attn.cuh"
static int glm53_dense_engine(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *cache, uint32_t rows, uint32_t pos0,
        uint32_t heads, uint32_t latent, uint32_t head_dim) {
    void *scratch = cuda_tmp_alloc(glm53_dense_bytes(rows, pos0 + rows),
        "GLM dense attention");
    if (!scratch) { return 0; }
    return glm53_dense_run(g_cublas, ds4_current_stream(), (float *)out->ptr,
        (const float *)q->ptr, (const __half *)cache->ptr, scratch,
        rows, pos0, heads, latent, head_dim);
}
#define DS4_GLM53_DENSE_ENGINE
#include "cuda/glm53_low_attn.cuh"
#undef DS4_GLM53_DENSE_ENGINE

__global__ static void glm53_proj_v_kernel(float *out, const float *low,
        const char *weight, uint32_t heads, uint32_t latent_dim,
        uint32_t head_dim, uint64_t row_bytes) {
    const uint32_t h = blockIdx.x;
    const uint32_t token = blockIdx.y;
    const float *input = low + ((uint64_t)token * heads + h) * latent_dim;
    float *dst = out + ((uint64_t)token * heads + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const char *row = weight + ((uint64_t)h * head_dim + d) * row_bytes;
        float value = 0.0f;
        for (uint32_t j = 0; j < latent_dim; j++) { value = fmaf(input[j], glm53_compact_q8(row, j), value); }
        dst[d] = value;
    }
}

extern "C" int ds4_gpu_glm53_proj_v(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *low_out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t rows, uint32_t heads,
        uint32_t latent_dim, uint32_t head_dim) {
    if (!glm53_compact_shape(rows, heads, latent_dim, head_dim) ||
        !glm53_compact_has(out, (uint64_t)rows * heads * head_dim, sizeof(float)) ||
        !glm53_compact_has(low_out, (uint64_t)rows * heads * latent_dim, sizeof(float))) {
        return 0;
    }
    const uint64_t row_bytes = (uint64_t)latent_dim /
        GLM53_COMPACT_Q8_BLOCK * GLM53_COMPACT_Q8_BYTES;
    const char *weight = glm53_compact_weight(model_map, model_size,
        weight_offset, (uint64_t)heads * head_dim * row_bytes, out,
        "GLM-5.3 compact V-b Q8_0");
    if (!weight) { return 0; }
    glm53_proj_v_kernel<<<dim3(heads, rows), 128u, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)low_out->ptr, weight,
        heads, latent_dim, head_dim, row_bytes);
    return cuda_ok(cudaGetLastError(), "GLM-5.3 compact V projection");
}

__global__ static void glm53_pool_key_kernel(__half *pool_cache,
        float *tail_k, float *tail_gate, const float *raw_k, const float *gate,
        const float *norm, const float *bias, const uint16_t *ape,
        uint32_t pos0, uint32_t rows, float eps) {
    const uint32_t d = threadIdx.x;
    const uint32_t width = DS4_GLM53_POOL_DIM;
    const uint32_t ratio = DS4_GLM53_POOL_SIZE;
    const uint32_t pool = pos0 / ratio + blockIdx.x;
    const uint32_t first = pool * ratio;
    const uint32_t end = pos0 + rows;
    const bool complete = (uint64_t)first + ratio <= end;
    __shared__ float keys[DS4_GLM53_POOL_SIZE * DS4_GLM53_POOL_DIM];
    __shared__ float mean[DS4_GLM53_POOL_SIZE], inv[DS4_GLM53_POOL_SIZE];
    float logits[DS4_GLM53_POOL_SIZE];
    float maximum = -INFINITY;

    for (uint32_t r = 0; r < ratio; r++) {
        const uint32_t pos = first + r;
        float key = 0.0f, logit = 0.0f;
        if (pos >= pos0 && pos < end) {
            key = raw_k[(uint64_t)(pos - pos0) * width + d];
            logit = gate[(uint64_t)(pos - pos0) * width + d];
            if (!complete) {
                tail_k[(uint64_t)r * width + d] = key;
                tail_gate[(uint64_t)r * width + d] = logit;
            }
        } else if (pos < pos0) {
            key = tail_k[(uint64_t)r * width + d];
            logit = tail_gate[(uint64_t)r * width + d];
        }
        keys[r * width + d] = key;
        logits[r] = logit + __uint_as_float((uint32_t)ape[r * width + d] << 16);
        maximum = fmaxf(maximum, logits[r]);
    }
    __syncthreads();
    if (!complete) { return; }
    if (d < ratio) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < width; j++) { sum += keys[d * width + j]; }
        mean[d] = sum / width;
        float variance = 0.0f;
        for (uint32_t j = 0; j < width; j++) {
            const float delta = keys[d * width + j] - mean[d];
            variance = fmaf(delta, delta, variance);
        }
        inv[d] = rsqrtf(variance / width + eps);
    }
    __syncthreads();
    float denom = 0.0f;
    for (uint32_t r = 0; r < ratio; r++) { logits[r] = expf(logits[r] - maximum); denom += logits[r]; }
    float value = 0.0f;
    for (uint32_t r = 0; r < ratio; r++) {
        const float normalized = (keys[r * width + d] - mean[r]) * inv[r] * norm[d] + bias[d];
        value = fmaf(logits[r] / denom, normalized, value);
    }
    pool_cache[(uint64_t)pool * width + d] = __float2half_rn(value);
}

extern "C" int ds4_gpu_glm53_pool_key(ds4_gpu_tensor *pool_cache,
        ds4_gpu_tensor *tail_k, ds4_gpu_tensor *tail_gate,
        const ds4_gpu_tensor *raw_k, const ds4_gpu_tensor *gate,
        const void *model_map, uint64_t model_size, uint64_t norm_offset,
        uint64_t bias_offset, uint64_t ape_offset, uint32_t pos0,
        uint32_t rows, uint32_t cap, float eps) {
    const uint32_t width = DS4_GLM53_POOL_DIM;
    const uint32_t ratio = DS4_GLM53_POOL_SIZE;
    const uint64_t n_pools = ((uint64_t)cap + ratio - 1u) / ratio;
    if (ds4_capture_active() || !rows || rows > GLM53_COMPACT_MAX_ROWS ||
        pos0 > cap || rows > cap - pos0 || !isfinite(eps) || eps <= 0.0f ||
        !glm53_compact_has(pool_cache, n_pools * width, sizeof(__half)) ||
        !glm53_compact_has(tail_k, (uint64_t)ratio * width, sizeof(float)) ||
        !glm53_compact_has(tail_gate, (uint64_t)ratio * width, sizeof(float)) ||
        !glm53_compact_has(raw_k, (uint64_t)rows * width, sizeof(float)) ||
        !glm53_compact_has(gate, (uint64_t)rows * width, sizeof(float))) {
        return 0;
    }
    const float *norm = (const float *)glm53_compact_weight(model_map,
        model_size, norm_offset, width * sizeof(float), pool_cache, "GLM-5.3 pool norm");
    const float *bias = (const float *)glm53_compact_weight(model_map,
        model_size, bias_offset, width * sizeof(float), pool_cache, "GLM-5.3 pool bias");
    const uint16_t *ape = (const uint16_t *)glm53_compact_weight(model_map,
        model_size, ape_offset, (uint64_t)ratio * width * sizeof(uint16_t),
        pool_cache, "GLM-5.3 pool APE");
    if (!norm || !bias || !ape) { return 0; }

    /* Serialize the leading carry and trailing partial pool around parallel
     * complete groups, so the shared four-row tail has one writer. */
    uint32_t done = 0u;
    const uint32_t leading = pos0 % ratio;
    if (leading) {
        const uint32_t count = min(rows, ratio - leading);
        glm53_pool_key_kernel<<<1u, width, 0, ds4_current_stream()>>>(
            (__half *)pool_cache->ptr, (float *)tail_k->ptr,
            (float *)tail_gate->ptr, (const float *)raw_k->ptr,
            (const float *)gate->ptr, norm, bias, ape, pos0, count, eps);
        if (!cuda_ok(cudaGetLastError(), "GLM-5.3 leading key pool")) { return 0; }
        done = count;
    }
    const uint32_t full = (rows - done) / ratio * ratio;
    if (full) {
        glm53_pool_key_kernel<<<full / ratio, width, 0, ds4_current_stream()>>>(
            (__half *)pool_cache->ptr, (float *)tail_k->ptr,
            (float *)tail_gate->ptr, (const float *)raw_k->ptr + (uint64_t)done * width,
            (const float *)gate->ptr + (uint64_t)done * width,
            norm, bias, ape, pos0 + done, full, eps);
        if (!cuda_ok(cudaGetLastError(), "GLM-5.3 complete key pools")) { return 0; }
        done += full;
    }
    if (done < rows) {
        glm53_pool_key_kernel<<<1u, width, 0, ds4_current_stream()>>>(
            (__half *)pool_cache->ptr, (float *)tail_k->ptr,
            (float *)tail_gate->ptr, (const float *)raw_k->ptr + (uint64_t)done * width,
            (const float *)gate->ptr + (uint64_t)done * width,
            norm, bias, ape, pos0 + done, rows - done, eps);
        if (!cuda_ok(cudaGetLastError(), "GLM-5.3 trailing key pool")) { return 0; }
    }
    return 1;
}

#include "cuda/glm53_pool_score.cuh"

__global__ static void glm53_pool_expand_kernel(uint32_t *out,
        const uint32_t *selected, uint32_t rows, uint32_t pos0,
        uint32_t selected_pools, uint32_t index_topk, uint32_t out_width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * out_width) { return; }
    const uint32_t token = (uint32_t)(i / out_width);
    const uint32_t slot = (uint32_t)(i % out_width);
    const uint32_t visible = pos0 + token + 1u;
    const uint32_t ratio = DS4_GLM53_POOL_SIZE;
    uint32_t value = UINT32_MAX;
    if (slot < index_topk && slot / ratio < selected_pools) {
        const uint32_t pool = selected[(uint64_t)token * selected_pools + slot / ratio];
        if (pool < visible / ratio) { value = pool * ratio + slot % ratio; }
    } else if (slot >= index_topk && slot - index_topk < visible % ratio) {
        value = visible - visible % ratio + slot - index_topk;
    }
    out[i] = value;
}

extern "C" int ds4_gpu_glm53_pool_expand(ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *pool_selected, uint32_t rows, uint32_t pos0,
        uint32_t selected_pools, uint32_t index_topk, uint32_t out_width) {
    if (ds4_capture_active() || !rows || rows > GLM53_COMPACT_MAX_ROWS ||
        pos0 > UINT32_MAX - rows || !selected_pools || !index_topk ||
        index_topk > DS4_GLM53_INDEX_TOPK ||
        selected_pools > index_topk / DS4_GLM53_POOL_SIZE ||
        out_width < index_topk + DS4_GLM53_POOL_SIZE - 1u ||
        out_width > DS4_GLM53_MAX_SELECTED ||
        !glm53_compact_has(selected, (uint64_t)rows * out_width, sizeof(uint32_t)) ||
        !glm53_compact_has(pool_selected, (uint64_t)rows * selected_pools, sizeof(uint32_t))) {
        return 0;
    }
    const uint64_t count = (uint64_t)rows * out_width;
    glm53_pool_expand_kernel<<<(unsigned)((count + 255u) / 256u),
        GLM53_COMPACT_THREADS, 0, ds4_current_stream()>>>(
            (uint32_t *)selected->ptr, (const uint32_t *)pool_selected->ptr,
            rows, pos0, selected_pools, index_topk, out_width);
    return cuda_ok(cudaGetLastError(), "GLM-5.3 pooled selection expansion");
}

#endif
