#ifndef DS4_GLM53_LOW_ATTN_CUH
#define DS4_GLM53_LOW_ATTN_CUH

#include "../ds4_glm53_attn.h"

enum { GLM53_ATTN_TILE = 64, GLM53_ATTN_PREFILL_ROWS = 128,
       GLM53_ATTN_PREFILL_HEADS = 64 };
enum glm53_attn_path { GLM53_ATTN_REF, GLM53_ATTN_PAIR };

template<glm53_attn_path Path>
__global__ static void glm53_low_attn_kernel(float *out, const float *q,
        const __half *cache, const uint32_t *selected, uint32_t sel_stride,
        uint32_t pos0, uint32_t heads, uint32_t latent_dim, float scale) {
    const uint32_t h = blockIdx.x;
    const uint32_t token = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t visible = pos0 + token + 1u;
    const uint32_t count = selected ? sel_stride : visible;
    const float *query = q + ((uint64_t)token * heads + h) * latent_dim;
    const uint32_t *picked = selected ? selected + (uint64_t)token * sel_stride : NULL;
    __shared__ float score[DS4_GLM53_MAX_SELECTED];
    __shared__ float reduce[GLM53_COMPACT_THREADS];

    for (uint32_t s = warp; s < count; s += blockDim.x / 32u) {
        const uint32_t row = picked ? picked[s] : s;
        float dot = 0.0f;
        if (row < visible) {
            for (uint32_t j = lane; j < latent_dim; j += 32u) {
                dot = fmaf(query[j], __half2float(cache[(uint64_t)row * latent_dim + j]), dot);
            }
        }
        for (uint32_t step = 16u; step; step >>= 1u) {
            dot += __shfl_down_sync(0xffffffffu, dot, step);
        }
        if (lane == 0u) { score[s] = row < visible ? dot * scale : -INFINITY; }
    }
    __syncthreads();
    float maximum = -INFINITY;
    for (uint32_t s = tid; s < count; s += blockDim.x) { maximum = fmaxf(maximum, score[s]); }
    reduce[tid] = maximum;
    __syncthreads();
    for (uint32_t step = blockDim.x / 2u; step; step >>= 1u) {
        if (tid < step) { reduce[tid] = fmaxf(reduce[tid], reduce[tid + step]); }
        __syncthreads();
    }
    maximum = reduce[0];
    float denom = 0.0f;
    for (uint32_t s = tid; s < count; s += blockDim.x) {
        const float weight = isfinite(score[s]) ? expf(score[s] - maximum) : 0.0f;
        score[s] = weight;
        denom += weight;
    }
    __syncthreads();
    reduce[tid] = denom;
    __syncthreads();
    for (uint32_t step = blockDim.x / 2u; step; step >>= 1u) {
        if (tid < step) { reduce[tid] += reduce[tid + step]; }
        __syncthreads();
    }
    const float inv = reduce[0] > 0.0f ? 1.0f / reduce[0] : 0.0f;
    float *dst = out + ((uint64_t)token * heads + h) * latent_dim;
    if constexpr (Path == GLM53_ATTN_PAIR) {
        // For latent512, share row/weight loads while keeping each FMA sequence.
        float lo = 0.0f, hi = 0.0f;
        for (uint32_t s = 0; s < count; s++) {
            const uint32_t row = picked ? picked[s] : s;
            if (row >= visible) { continue; }
            const float weight = score[s] * inv;
            const uint64_t offset = (uint64_t)row * latent_dim + tid;
            lo = fmaf(weight, __half2float(cache[offset]), lo);
            hi = fmaf(weight, __half2float(cache[offset + GLM53_COMPACT_THREADS]), hi);
        }
        dst[tid] = lo;
        dst[tid + GLM53_COMPACT_THREADS] = hi;
    } else {
        for (uint32_t j = tid; j < latent_dim; j += blockDim.x) {
            float value = 0.0f;
            for (uint32_t s = 0; s < count; s++) {
                const uint32_t row = picked ? picked[s] : s;
                if (row >= visible) { continue; }
                value = fmaf(score[s] * inv,
                    __half2float(cache[(uint64_t)row * latent_dim + j]), value);
            }
            dst[j] = value;
        }
    }
}

static int glm53_low_pair(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *value = getenv("DS4_GLM53_LOW_ATTN");
        enabled = !value || atoi(value) > 0;
    }
    return enabled;
}

/* MTP has full causal attention. Online softmax keeps its workspace bounded
 * as predictor history grows beyond the trunk's selected-row capacity. */
__global__ static void glm53_low_all_kernel(float *out, const float *q,
        const __half *cache, uint32_t pos0, uint32_t heads,
        uint32_t latent_dim, float scale) {
    const uint32_t h = blockIdx.x, token = blockIdx.y;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t visible = pos0 + token + 1u;
    const float *query = q + ((uint64_t)token * heads + h) * latent_dim;
    __shared__ float scores[GLM53_ATTN_TILE];
    __shared__ float reduce[GLM53_COMPACT_THREADS];
    float accum[GLM53_COMPACT_MAX_DIM / GLM53_COMPACT_THREADS] = {0.0f};
    float maximum = -INFINITY, denominator = 0.0f;

    for (uint64_t first = 0u; first < visible; first += GLM53_ATTN_TILE) {
        for (uint32_t s = warp; s < GLM53_ATTN_TILE; s += blockDim.x / 32u) {
            const uint64_t row = (uint64_t)first + s;
            float dot = 0.0f;
            if (row < visible) {
                for (uint32_t j = lane; j < latent_dim; j += 32u) {
                    dot = fmaf(query[j], __half2float(cache[row * latent_dim + j]), dot);
                }
            }
            for (uint32_t step = 16u; step; step >>= 1u) {
                dot += __shfl_down_sync(0xffffffffu, dot, step);
            }
            if (!lane) { scores[s] = row < visible ? dot * scale : -INFINITY; }
        }
        __syncthreads();
        reduce[tid] = tid < GLM53_ATTN_TILE ? scores[tid] : -INFINITY;
        __syncthreads();
        for (uint32_t step = blockDim.x / 2u; step; step >>= 1u) {
            if (tid < step) { reduce[tid] = fmaxf(reduce[tid], reduce[tid + step]); }
            __syncthreads();
        }
        const float tile_max = reduce[0];
        __syncthreads();
        float weight = 0.0f;
        if (tid < GLM53_ATTN_TILE) {
            weight = isfinite(scores[tid]) ? expf(scores[tid] - tile_max) : 0.0f;
            scores[tid] = weight;
        }
        reduce[tid] = weight;
        __syncthreads();
        for (uint32_t step = blockDim.x / 2u; step; step >>= 1u) {
            if (tid < step) { reduce[tid] += reduce[tid + step]; }
            __syncthreads();
        }
        const float next_max = fmaxf(maximum, tile_max);
        const float old_scale = expf(maximum - next_max);
        const float tile_scale = expf(tile_max - next_max);
        denominator = denominator * old_scale + reduce[0] * tile_scale;
        maximum = next_max;
        for (uint32_t j = tid, a = 0u; j < latent_dim; j += blockDim.x, a++) {
            float value = 0.0f;
            for (uint32_t s = 0u; s < GLM53_ATTN_TILE && (uint64_t)first + s < visible; s++) {
                value = fmaf(scores[s], __half2float(cache[((uint64_t)first + s) * latent_dim + j]), value);
            }
            accum[a] = accum[a] * old_scale + value * tile_scale;
        }
        __syncthreads();
    }
    const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    float *dst = out + ((uint64_t)token * heads + h) * latent_dim;
    for (uint32_t j = tid, a = 0u; j < latent_dim; j += blockDim.x, a++) {
        dst[j] = accum[a] * inv;
    }
}

extern "C" int ds4_gpu_glm53_attn_low(ds4_gpu_tensor *low_out,
        const ds4_gpu_tensor *low_q, const ds4_gpu_tensor *cache,
        const ds4_gpu_tensor *selected, uint32_t sel_stride, uint32_t rows,
        uint32_t pos0, uint32_t cap, uint32_t heads, uint32_t latent_dim,
        uint32_t head_dim) {
    if (ds4_capture_active() || !glm53_compact_shape(rows, heads, latent_dim, head_dim) ||
        !glm53_attn_frontier(rows, pos0, cap, sel_stride,
            selected ? GLM53_ATTN_SELECTED : GLM53_ATTN_ALL) ||
        !glm53_compact_has(low_out, (uint64_t)rows * heads * latent_dim, sizeof(float)) ||
        !glm53_compact_has(low_q, (uint64_t)rows * heads * latent_dim, sizeof(float)) ||
        !glm53_compact_has(cache, (uint64_t)cap * latent_dim, sizeof(__half))) {
        return 0;
    }
    if (selected) {
        if (!sel_stride || sel_stride > DS4_GLM53_MAX_SELECTED ||
            !glm53_compact_has(selected, (uint64_t)rows * sel_stride, sizeof(uint32_t))) {
            return 0;
        }
    }
#ifdef DS4_GLM53_DENSE_ENGINE
    const char *dense = getenv("DS4_GLM53_DENSE_GEMM");
    if (dense && strcmp(dense, "0") != 0 && g_cublas_ready &&
        glm53_dense_shape(rows, pos0 + rows, heads, latent_dim,
            selected ? GLM53_ATTN_SELECTED : GLM53_ATTN_ALL)) {
        return glm53_dense_engine(low_out, low_q, cache, rows, pos0,
            heads, latent_dim, head_dim);
    }
#endif
    if (!selected && pos0 + rows > DS4_GLM53_MAX_SELECTED) {
        glm53_low_all_kernel<<<dim3(heads, rows), GLM53_COMPACT_THREADS,
            0, ds4_current_stream()>>>(
                (float *)low_out->ptr, (const float *)low_q->ptr,
                (const __half *)cache->ptr, pos0, heads, latent_dim,
                1.0f / sqrtf((float)head_dim));
        return cuda_ok(cudaGetLastError(), "GLM-5.3 full causal latent attention");
    }
    // Whole-model decode regresses with paired loads. Keep its reference path.
    if (selected && sel_stride == DS4_GLM53_MAX_SELECTED &&
        (rows == GLM53_ATTN_PREFILL_ROWS ||
         (rows > GLM53_ATTN_PREFILL_ROWS && rows <= GLM53_DENSE_ROWS &&
          (!getenv("DS4_GLM53_LOW_ATTN_WIDE") ||
           strcmp(getenv("DS4_GLM53_LOW_ATTN_WIDE"), "0") != 0))) &&
        heads == GLM53_ATTN_PREFILL_HEADS &&
        latent_dim == 2u * GLM53_COMPACT_THREADS && glm53_low_pair()) {
        glm53_low_attn_kernel<GLM53_ATTN_PAIR><<<dim3(heads, rows), GLM53_COMPACT_THREADS,
            0, ds4_current_stream()>>>(
                (float *)low_out->ptr, (const float *)low_q->ptr,
                (const __half *)cache->ptr,
                selected ? (const uint32_t *)selected->ptr : NULL,
                sel_stride, pos0, heads, latent_dim, 1.0f / sqrtf((float)head_dim));
        return cuda_ok(cudaGetLastError(), "GLM-5.3 paired compact attention");
    }
    glm53_low_attn_kernel<GLM53_ATTN_REF><<<dim3(heads, rows), GLM53_COMPACT_THREADS,
        0, ds4_current_stream()>>>(
            (float *)low_out->ptr, (const float *)low_q->ptr,
            (const __half *)cache->ptr,
            selected ? (const uint32_t *)selected->ptr : NULL,
            sel_stride, pos0, heads, latent_dim, 1.0f / sqrtf((float)head_dim));
    return cuda_ok(cudaGetLastError(), "GLM-5.3 compact attention");
}

#endif
