/* HYV4FlashHyperConnection, Engram and attention, Transformers c2ec8505.
 * All leaf launches run on the calling graph's stream. */
#include "ds4_hy4.h"
#include <cub/block/block_radix_sort.cuh>

namespace hy4 {
using bf16 = __nv_bfloat16;
__device__ inline float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
__device__ inline float e4m3(uint8_t x) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(x, __NV_E4M3)));
}
__device__ inline float warp_sum(float x) {
    for (int d = 16; d; d /= 2) {
        x += __shfl_xor_sync(0xffffffffu, x, d);
    }
    return x;
}
__device__ inline float sum(float x) {
    __shared__ float partial[32];
    x = warp_sum(x);
    if (!(threadIdx.x & 31)) {
        partial[threadIdx.x >> 5] = x;
    }
    __syncthreads();
    x = warp_sum((threadIdx.x & 31) < blockDim.x / 32 ? partial[threadIdx.x & 31] : 0.f);
    __syncthreads();
    return x;
}
__device__ inline float max_block(float x) {
    __shared__ float partial[32];
    for (int d = 16; d; d /= 2) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, d));
    }
    if (!(threadIdx.x & 31)) {
        partial[threadIdx.x >> 5] = x;
    }
    __syncthreads();
    x = (threadIdx.x & 31) < blockDim.x / 32 ? partial[threadIdx.x & 31] : -INFINITY;
    for (int d = 16; d; d /= 2) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, d));
    }
    __syncthreads();
    return x;
}

__global__ void embed(float *streams, const bf16 *input) {
    const uint32_t row = blockIdx.y, stream = blockIdx.x;
    for (unsigned d = threadIdx.x; d < 3072; d += blockDim.x) {
        streams[(row * 4 + stream) * 3072 + d] = __bfloat162float(input[row * 3072 + d]);
    }
}

__global__ void hc(float *out, float *post, const float *streams, const float *mix, const float *scale,
                   const float *base) {
    const uint32_t row = blockIdx.x;
    const float *x = streams + row * 12288;
    float sq = 0;
    for (unsigned d = threadIdx.x; d < 12288; d += blockDim.x) {
        sq += x[d] * x[d];
    }
    const float inv = rsqrtf(sum(sq) / 12288.f + 1e-5f);
    __shared__ float pre[4];
    if (threadIdx.x < 4) {
        const uint32_t c = threadIdx.x, i = row * (post ? 8 : 4) + c;
        pre[c] = sigmoid(mix[i] * inv * scale[0] + base[c]) + 1e-6f;
        if (post) {
            post[row * 4 + c] = 2.f * sigmoid(mix[i + 4] * inv * scale[1] + base[c + 4]) + 1e-6f;
        }
    }
    __syncthreads();
    for (unsigned d = threadIdx.x; d < 3072; d += blockDim.x) {
        float v = 0;
        for (unsigned c = 0; c < 4; c++) {
            v += x[c * 3072 + d] * pre[c];
        }
        out[row * 3072 + d] = bf(v);
    }
}
__global__ void residual(float *streams, const float *branch, const float *post) {
    const uint32_t row = blockIdx.y, c = blockIdx.x, i = row * 4 + c;
    for (unsigned d = threadIdx.x; d < 3072; d += blockDim.x) {
        streams[i * 3072 + d] = bf(streams[i * 3072 + d] + post[i] * branch[row * 3072 + d]);
    }
}

__global__ void unpack(float *out, const uint8_t *records, const float *norm) {
    const unsigned h = blockIdx.x, row = blockIdx.y, d = threadIdx.x;
    const uint8_t *p = records + (row * 16 + h) * 98;
    const float x = d < 96 ? bf(e4m3(p[d]) * __bfloat162float(*(const bf16 *)(p + 96))) : 0;
    // nn.RMSNorm(eps=None) uses the FP32 opmath epsilon for BF16 inputs.
    const float inv = rsqrtf(sum(x * x) / 96.f + 0x1p-23f);
    if (d < 96) {
        out[(row * 16 + h) * 96 + d] = bf(bf(x * inv) * norm[h * 96 + d]);
    }
}
__global__ void engram(float *streams, const float *kv, const int *ids, const float *bias, const float *wk,
                       const float *wq, const float *gamma, const float *scales) {
    const uint32_t row = blockIdx.y, c = blockIdx.x;
    if (ids[row] == 120002 || ids[row] == 120120 || ids[row] == 120124) {
        return;
    }
    float *x = streams + (row * 4 + c) * 3072;
    const float *k = kv + (row * 5 + c) * 3072, *v = kv + (row * 5 + 4) * 3072;
    float kk = 0, qq = 0, vv = 0;
    for (unsigned d = threadIdx.x; d < 3072; d += blockDim.x) {
        const float key = bf(k[d] + bias[c * 3072 + d]), value = bf(v[d] + bias[4 * 3072 + d]);
        kk += key * key;
        qq += x[d] * x[d];
        vv += value * value;
    }
    const float ik = rsqrtf(sum(kk) / 3072.f + 0x1p-23f);
    const float iq = rsqrtf(sum(qq) / 3072.f + 0x1p-23f);
    const float iv = rsqrtf(sum(vv) / 3072.f + 0x1p-23f);
    float dot = 0;
    for (unsigned d = threadIdx.x; d < 3072; d += blockDim.x) {
        float key = bf(bf(bf(k[d] + bias[c * 3072 + d]) * ik) * wk[c * 3072 + d]);
        float query = bf(bf(x[d] * iq) * wq[c * 3072 + d]);
        dot += bf(key * query);
    }
    dot = bf(bf(sum(dot)) * rsqrtf(3072.f));
    const float gate = dot == 0.f ? .5f : bf(sigmoid(copysignf(bf(sqrtf(fmaxf(fabsf(dot), 1e-6f))), dot)));
    for (unsigned d = threadIdx.x; d < 3072; d += blockDim.x) {
        const float signal = bf(gamma[d] * bf(bf(v[d] + bias[4 * 3072 + d]) * iv));
        x[d] = bf(x[d] + bf(bf(gate * signal) + bf(scales[c * 3072 + d] * signal)));
    }
}
__global__ void swiglu(float *out, const float *gate, const float *up, uint32_t n, float limit) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = fminf(bf(gate[i]), limit), u = fmaxf(-limit, fminf(bf(up[i]), limit));
        out[i] = bf(bf(g * sigmoid(g)) * u);
    }
}
__global__ void moe_sum(float *out, const float *down, const float *weights) {
    const uint32_t r = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= 1536) {
        return;
    }
    float v = 0;
    for (unsigned e = 0; e < 16; e++) {
        v += bf(down[(r * 16 + e) * 1536 + d]) * weights[r * 16 + e];
    }
    out[r * 1536 + d] = bf(v);
}

__device__ inline void rotate(float &a, float &b, unsigned f, const int *position, float theta) {
    const unsigned axis = f < 30 ? f % 3 : 0;
    const float angle = position[axis] * powf(theta, -float(f) / 32.f);
    float s, c;
    sincosf(angle, &s, &c);
    const float x = bf(a), y = bf(b);
    s = bf(s);
    c = bf(c);
    a = bf(x * c - y * s);
    b = bf(y * c + x * s);
}
__global__ void rope(float *x, const int *positions, unsigned heads, unsigned stride, unsigned tail,
                     float theta) {
    const uint32_t r = blockIdx.y, h = blockIdx.x, f = threadIdx.x;
    float *p = x + (r * heads + h) * stride + tail + 2 * f;
    float a = p[0], b = p[1];
    rotate(a, b, f, positions + 3 * r, theta);
    p[0] = a;
    p[1] = b;
}
__global__ void store(void *latent, bf16 *rope, const float *raw, const float *w, unsigned cap,
                      unsigned pos, bool fp8) {
    const unsigned r = blockIdx.x, slot = (pos + r) % cap, d = threadIdx.x;
    const float *p = raw + r * 576;
    const float a = bf(p[d]), b = bf(p[256 + d]);
    const float inv = rsqrtf(sum(a * a + b * b) / 512.f + 1e-5f);
    const float v0 = bf(w[d] * bf(a * inv)), v1 = bf(w[d + 256] * bf(b * inv));
    if (fp8) {
        const int e = l31_latent_exp(max_block(fmaxf(fabsf(v0), fabsf(v1))));
        const float scale = l31_pow2(-e);
        uint8_t *dst = (uint8_t *)latent + (uint64_t)slot * 512;
        dst[d] = __nv_cvt_float_to_fp8(v0 * scale, __NV_SATFINITE, __NV_E4M3);
        dst[d + 256] = __nv_cvt_float_to_fp8(v1 * scale, __NV_SATFINITE, __NV_E4M3);
        if (!d) {
            ((int8_t *)latent)[(uint64_t)cap * 512 + slot] = (int8_t)e;
        }
    } else {
        bf16 *dst = (bf16 *)latent + (uint64_t)slot * 512;
        dst[d] = __float2bfloat16(v0);
        dst[d + 256] = __float2bfloat16(v1);
    }
    if (d < 64) {
        rope[(uint64_t)slot * 64 + d] = __float2bfloat16(p[512 + d]);
    }
}
} // namespace hy4

extern "C" const float *ds4_gpu_hy4_weight(const void *map, uint64_t size, uint64_t off, uint64_t bytes) {
    return map && off <= size && bytes <= size - off
               ? (const float *)cuda_model_range_ptr(map, off, bytes, "Hy4 weight")
               : nullptr;
}
extern "C" int ds4_gpu_hy4_embed(ds4_gpu_tensor *s, const ds4_gpu_tensor *x, uint32_t n) {
    hy4::embed<<<dim3(4, n), 256, 0, ds4_current_stream()>>>((float *)s->ptr, (const hy4::bf16 *)x->ptr);
    return cuda_ok(cudaGetLastError(), "Hy4 embedding");
}
extern "C" int ds4_gpu_hy4_hc(ds4_gpu_tensor *out, ds4_gpu_tensor *post, const ds4_gpu_tensor *s,
                              const ds4_gpu_tensor *mix, const float *scale, const float *base,
                              uint32_t n) {
    if (!scale || !base) {
        return 0;
    }
    hy4::hc<<<n, 256, 0, ds4_current_stream()>>>((float *)out->ptr, post ? (float *)post->ptr : nullptr,
                                                 (const float *)s->ptr, (const float *)mix->ptr, scale,
                                                 base);
    return cuda_ok(cudaGetLastError(), "Hy4 iHC");
}
extern "C" int ds4_gpu_hy4_residual(ds4_gpu_tensor *s, const ds4_gpu_tensor *b, const ds4_gpu_tensor *p,
                                    uint32_t n) {
    hy4::residual<<<dim3(4, n), 256, 0, ds4_current_stream()>>>((float *)s->ptr, (const float *)b->ptr,
                                                                (const float *)p->ptr);
    return cuda_ok(cudaGetLastError(), "Hy4 residual");
}
extern "C" int ds4_gpu_hy4_unpack(ds4_gpu_tensor *out, const ds4_gpu_tensor *records, const float *w,
                                  uint32_t layer, uint32_t n) {
    if (!w) {
        return 0;
    }
    hy4::unpack<<<dim3(16, n), 128, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const uint8_t *)records->ptr + (uint64_t)layer * n * 16 * 98, w);
    return cuda_ok(cudaGetLastError(), "Hy4 Engram unpack");
}
extern "C" int ds4_gpu_hy4_engram(ds4_gpu_tensor *s, const ds4_gpu_tensor *kv, const ds4_gpu_tensor *ids,
                                  const float *bias, const float *wk, const float *wq, const float *gamma,
                                  const float *scales, uint32_t n) {
    if (!bias || !wk || !wq || !gamma || !scales) {
        return 0;
    }
    hy4::engram<<<dim3(4, n), 256, 0, ds4_current_stream()>>>(
        (float *)s->ptr, (const float *)kv->ptr, (const int *)ids->ptr, bias, wk, wq, gamma, scales);
    return cuda_ok(cudaGetLastError(), "Hy4 Engram residual");
}
extern "C" int ds4_gpu_hy4_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
                                  uint32_t n, float limit) {
    hy4::swiglu<<<(n + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, limit);
    return cuda_ok(cudaGetLastError(), "Hy4 SwiGLU");
}
extern "C" int ds4_gpu_hy4_moe_sum(ds4_gpu_tensor *out, const ds4_gpu_tensor *down,
                                   const ds4_gpu_tensor *weights, uint32_t n) {
    hy4::moe_sum<<<dim3(6, n), 256, 0, ds4_current_stream()>>>((float *)out->ptr, (const float *)down->ptr,
                                                               (const float *)weights->ptr);
    return cuda_ok(cudaGetLastError(), "Hy4 expert sum");
}
extern "C" int ds4_gpu_hy4_rope(ds4_gpu_tensor *x, const ds4_gpu_tensor *positions, uint32_t n, uint32_t h,
                                uint32_t stride, uint32_t tail, float theta) {
    hy4::rope<<<dim3(h, n), 32, 0, ds4_current_stream()>>>((float *)x->ptr, (const int *)positions->ptr, h,
                                                           stride, tail, theta);
    return cuda_ok(cudaGetLastError(), "Hy4 interleaved M-RoPE");
}
extern "C" int ds4_gpu_hy4_head_mm(ds4_gpu_tensor *out, ds4_gpu_tensor *pack, const ds4_gpu_tensor *x,
                                   const void *map, uint64_t size, uint64_t offset, uint32_t n, uint32_t h,
                                   uint32_t in, uint32_t stride, uint32_t width) {
    const uint64_t bytes = (uint64_t)h * in * width * 2;
    const void *weight = ds4_gpu_hy4_weight(map, size, offset, bytes);
    if (!weight || !g_cublas_ready || !ds4_gpu_f32_to_bf16(pack, x, (uint64_t)n * h * stride)) {
        return 0;
    }
    const float alpha = 1.f, beta = 0.f;
    if (cublasSetStream(g_cublas, ds4_current_stream()) != CUBLAS_STATUS_SUCCESS) {
        return 0;
    }
    cuda_cublas_ws_prep(ds4_current_stream());
    const cublasStatus_t status = cublasGemmStridedBatchedEx(
        g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, width, n, in, &alpha, weight, CUDA_R_16BF, in,
        (long long)in * width, pack->ptr, CUDA_R_16BF, h * stride, stride, &beta, out->ptr, CUDA_R_32F,
        h * width, width, h, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return status == CUBLAS_STATUS_SUCCESS;
}
extern "C" int ds4_gpu_hy4_store(ds4_hy4_cache *c, const ds4_gpu_tensor *raw, const float *norm,
                                 uint32_t pos, uint32_t n) {
    if (!norm) {
        return 0;
    }
    hy4::store<<<n, 256, 0, ds4_current_stream()>>>(c->latent->ptr, (hy4::bf16 *)c->rope->ptr,
                                                    (const float *)raw->ptr, norm, c->cap, pos, c->fp8);
    return cuda_ok(cudaGetLastError(), "Hy4 MLA cache");
}

namespace hy4 {
// Store FP32 pool operands and the three spatial positions of each slot.
// A partial four-token group may start in the preceding prefill chunk.
__global__ void stage_group(float *recent, const float *v, const float *g, const int *positions,
                            unsigned cap, unsigned pos) {
    const unsigned r = blockIdx.x, d = threadIdx.x;
    float *s = recent + (uint64_t)((pos + r) % cap) * 259;
    s[d] = v[r * 128 + d];
    s[128 + d] = g[r * 128 + d];
    if (d < 3) {
        ((int *)(s + 256))[d] = positions[r * 3 + d];
    }
}
__global__ void pool(bf16 *groups, const float *recent, const float *ape, const float *norm, unsigned cap,
                     unsigned first) {
    const unsigned group = first + blockIdx.x, d = threadIdx.x;
    float g[4], v[4], peak = -INFINITY;
    for (unsigned j = 0; j < 4; j++) {
        const float *s = recent + (uint64_t)((4 * group + j) % cap) * 259;
        g[j] = s[128 + d] + ape[j * 128 + d];
        v[j] = s[d];
        peak = fmaxf(peak, g[j]);
    }
    float z = 0, pooled = 0;
    for (unsigned j = 0; j < 4; j++) {
        const float w = expf(g[j] - peak);
        z += w;
        pooled += w * v[j];
    }
    pooled /= z;
    float inv = rsqrtf(sum(pooled * pooled) / 128.f + 1e-5f);
    __shared__ float key[128];
    key[d] = bf(pooled * inv * norm[d]);
    __syncthreads();
    if (d < 64) {
        groups[(uint64_t)group * 128 + d] = __float2bfloat16(key[d]);
    }
    if (d < 32) {
        const int *p = (const int *)(recent + (uint64_t)((4 * group) % cap) * 259 + 256);
        float a = key[64 + 2 * d], b = key[65 + 2 * d];
        rotate(a, b, d, p, 1e7f);
        groups[(uint64_t)group * 128 + 64 + 2 * d] = __float2bfloat16(a);
        groups[(uint64_t)group * 128 + 65 + 2 * d] = __float2bfloat16(b);
    }
}

__global__ void index_scores(float *tile, const float *q, const float *weights, const bf16 *groups,
                             unsigned pos, unsigned first) {
    const unsigned lane = threadIdx.x % 32, j = blockIdx.x * 8 + threadIdx.x / 32, row = blockIdx.y,
                   group = first + j;
    float score = -INFINITY;
    if (group < (pos + row) / 4) {
        float key[4];
        for (unsigned i = 0; i < 4; i++) {
            key[i] = __bfloat162float(groups[(uint64_t)group * 128 + i * 32 + lane]);
        }
        score = 0;
        for (unsigned h = 0; h < 16; h++) {
            const float *qh = q + (row * 16 + h) * 128;
            float dot = 0;
            for (unsigned i = 0; i < 4; i++) {
                dot += key[i] * bf(qh[i * 32 + lane]);
            }
            dot = warp_sum(dot);
            score += fmaxf(0.f, dot) * weights[row * 16 + h];
        }
    }
    if (!lane) {
        tile[row * 1024 + j] = score;
    }
}
__global__ void index_merge(int *ids, float *best, const float *tile, unsigned first) {
    using Sort = cub::BlockRadixSort<float, 128, 12, int>;
    __shared__ typename Sort::TempStorage scratch;
    const unsigned r = blockIdx.x;
    float scores[12];
    int values[12];
    for (unsigned i = 0; i < 12; i++) {
        const unsigned j = threadIdx.x * 12 + i;
        if (j < 512) {
            scores[i] = first && j < 511 ? best[r * 512 + j] : -INFINITY;
            values[i] = first && j < 511 ? ids[r * 512 + j] : -1;
        } else {
            scores[i] = tile[r * 1024 + j - 512];
            values[i] = (int)(first + j - 512);
        }
    }
    Sort(scratch).SortDescending(scores, values);
    for (unsigned i = 0; i < 12; i++) {
        const unsigned j = threadIdx.x * 12 + i;
        if (j < 511) {
            best[r * 512 + j] = scores[i];
            ids[r * 512 + j] = values[i];
        }
    }
}

// The selector keeps the current incomplete group after 511 completed groups.
template <unsigned HEADS> struct FlashKeys {
    static constexpr unsigned heads = HEADS, query_dim = 256;
    static constexpr bool contiguous = false, has_sinks = true;
    unsigned pos0, cap;
    const int *groups;
    const float *sinks;
    __device__ unsigned count(unsigned at) const {
        if constexpr (HEADS == 64) {
            return min(at + 1, 512u);
        }
        return at < 2048 ? at + 1 : 511 * 4 + at % 4 + 1;
    }
    __device__ unsigned at(unsigned row, unsigned position, unsigned slot) const {
        if constexpr (HEADS == 64) {
            return (position + 1 - count(position) + slot) % cap;
        }
        if (position < 2048) {
            return slot;
        }
        const unsigned group = slot < 511 * 4 ? groups[row * 512 + slot / 4] : position / 4;
        return group * 4 + slot % 4;
    }
};
template <unsigned HEADS, bool FP8>
__global__ void __launch_bounds__(HEADS * 4, 1)
    flash(float *out, const float *qa, const float *q, const void *latent, const int8_t *exps,
          const bf16 *rope, FlashKeys<HEADS> keys) {
    l31_fa_tile<FP8>(out, nullptr, nullptr, q, qa, latent, exps, rope, keys, UINT32_MAX, 0.0625f);
}
template <unsigned HEADS, bool FP8>
static bool launch_flash(ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *q,
                         const ds4_hy4_cache *c, const ds4_gpu_tensor *ids, const float *sinks,
                         unsigned pos, unsigned rows) {
    constexpr unsigned bytes = FP8 ? L31_FA8_SMEM : L31_FA_SMEM;
    static const cudaError_t setup =
        cudaFuncSetAttribute(flash<HEADS, FP8>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (setup != cudaSuccess) {
        return false;
    }
    const auto *exps = FP8 ? (const int8_t *)c->latent->ptr + (size_t)c->cap * 512 : nullptr;
    FlashKeys<HEADS> keys{pos, c->cap, ids ? (const int *)ids->ptr : nullptr, sinks};
    flash<HEADS, FP8><<<rows, HEADS * 4, bytes, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)a->ptr, (const float *)q->ptr, c->latent->ptr, exps,
        (const bf16 *)c->rope->ptr, keys);
    return true;
}

template <bool FP8, bool DRAFT>
__global__ void attend(float *out, const float *q_abs, const float *q, const void *latent, const bf16 *rope,
                       const int *groups, const float *sinks, unsigned cap, unsigned pos, unsigned heads) {
    const unsigned h = blockIdx.x, r = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned at = DRAFT ? pos - 1 : pos + r, group = at / 4;
    const bool window = heads == 64, indexed = !window && at >= 2048;
    const unsigned history = window ? min(at + 1, 512u) : indexed ? 511 * 4 + at % 4 + 1 : at + 1;
    const unsigned count = history + (DRAFT ? r + 1 : 0);
    const unsigned start = window ? at + 1 - history : 0;
    const float *qrow = q_abs + (r * heads + h) * 512;
    const float *qrot = q + (r * heads + h) * 256 + 192;
    float query[16], accum[16] = {};
    for (unsigned i = 0; i < 16; i++) {
        query[i] = qrow[i * 32 + lane];
    }
    float peak = sinks[h], z = warp ? 0.f : 1.f;
    for (unsigned j = warp; j < count; j += 8) {
        const unsigned token =
            DRAFT && j >= history ? pos + (j - history)
            : indexed
                ? ((j < 511 * 4 ? (unsigned)groups[(DRAFT ? 0 : r * 512) + j / 4] : group) * 4 + j % 4)
                : start + j;
        const unsigned slot = token % cap;
        const float scale = FP8 ? l31_pow2(((const int8_t *)latent)[(uint64_t)cap * 512 + slot]) : 1.f;
        float val[16], dot = 0;
        for (unsigned i = 0; i < 16; i++) {
            const uint64_t k = (uint64_t)slot * 512 + i * 32 + lane;
            val[i] = FP8 ? e4m3(((const uint8_t *)latent)[k]) * scale
                         : __bfloat162float(((const bf16 *)latent)[k]);
            dot += query[i] * val[i];
        }
        for (unsigned i = lane; i < 64; i += 32) {
            dot += qrot[i] * __bfloat162float(rope[(uint64_t)slot * 64 + i]);
        }
        dot = warp_sum(dot) * 0.0625f;
        const float next = fmaxf(peak, dot), prior = expf(peak - next), weight = expf(dot - next);
        z = z * prior + weight;
        for (unsigned i = 0; i < 16; i++) {
            accum[i] = accum[i] * prior + val[i] * weight;
        }
        peak = next;
    }
    __shared__ float peaks[8], totals[8], parts[8][512];
    if (!lane) {
        peaks[warp] = peak;
        totals[warp] = z;
    }
    for (unsigned i = 0; i < 16; i++) {
        parts[warp][i * 32 + lane] = accum[i];
    }
    __syncthreads();
    float maximum = peaks[0];
    for (unsigned w = 1; w < 8; w++) {
        maximum = fmaxf(maximum, peaks[w]);
    }
    float den = 0, scale[8];
    for (unsigned w = 0; w < 8; w++) {
        scale[w] = expf(peaks[w] - maximum);
        den += totals[w] * scale[w];
    }
    for (unsigned d = threadIdx.x; d < 512; d += blockDim.x) {
        float v = 0;
        for (unsigned w = 0; w < 8; w++) {
            v += parts[w][d] * scale[w];
        }
        out[(r * heads + h) * 512 + d] = v / den;
    }
}
} // namespace hy4

extern "C" int ds4_gpu_hy4_pool(ds4_hy4_cache *c, const ds4_gpu_tensor *v, const ds4_gpu_tensor *g,
                                const ds4_gpu_tensor *p, const float *ape, const float *norm, uint32_t pos,
                                uint32_t rows) {
    if (!ape || !norm || !c->recent || !c->groups) {
        return 0;
    }
    hy4::stage_group<<<rows, 128, 0, ds4_current_stream()>>>((float *)c->recent->ptr, (const float *)v->ptr,
                                                             (const float *)g->ptr, (const int *)p->ptr,
                                                             c->recent_cap, pos);
    const uint32_t first = pos / 4, end = (pos + rows) / 4;
    if (end > first) {
        hy4::pool<<<end - first, 128, 0, ds4_current_stream()>>>(
            (hy4::bf16 *)c->groups->ptr, (const float *)c->recent->ptr, ape, norm, c->recent_cap, first);
    }
    return cuda_ok(cudaGetLastError(), "Hy4 IndexPool");
}
extern "C" int ds4_gpu_hy4_index(ds4_gpu_tensor *ids, ds4_gpu_tensor *tile, ds4_gpu_tensor *best,
                                 const ds4_gpu_tensor *q, const ds4_gpu_tensor *w, const ds4_hy4_cache *c,
                                 uint32_t pos, uint32_t rows) {
    for (uint32_t first = 0; first < (pos + rows - 1) / 4; first += 1024) {
        hy4::index_scores<<<dim3(128, rows), 256, 0, ds4_current_stream()>>>(
            (float *)tile->ptr, (const float *)q->ptr, (const float *)w->ptr,
            (const hy4::bf16 *)c->groups->ptr, pos, first);
        hy4::index_merge<<<rows, 128, 0, ds4_current_stream()>>>((int *)ids->ptr, (float *)best->ptr,
                                                                 (const float *)tile->ptr, first);
    }
    return cuda_ok(cudaGetLastError(), "Hy4 top complete groups");
}
extern "C" int ds4_gpu_hy4_attention(ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *q,
                                     const ds4_hy4_cache *c, const ds4_gpu_tensor *ids, const float *sinks,
                                     uint32_t pos, uint32_t rows, uint32_t heads, int draft) {
    if (!sinks) {
        return 0;
    }
    const char *flash = getenv("DS4_HY4_FLASH");
    if (!draft && rows >= 16 && !(flash && strcmp(flash, "0") == 0)) {
        const bool launched = heads == 64
                                  ? hy4::launch_flash<64, false>(out, a, q, c, ids, sinks, pos, rows)
                              : c->fp8 ? hy4::launch_flash<32, true>(out, a, q, c, ids, sinks, pos, rows)
                                       : hy4::launch_flash<32, false>(out, a, q, c, ids, sinks, pos, rows);
        if (launched) {
            return cuda_ok(cudaGetLastError(), "Hy4 flash MLA");
        }
    }
    auto kernel = draft ? (c->fp8 ? hy4::attend<true, true> : hy4::attend<false, true>)
                        : (c->fp8 ? hy4::attend<true, false> : hy4::attend<false, false>);
    kernel<<<dim3(heads, rows), 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)a->ptr, (const float *)q->ptr, c->latent->ptr,
        (const hy4::bf16 *)c->rope->ptr, ids ? (const int *)ids->ptr : nullptr, sinks, c->cap, pos, heads);
    return cuda_ok(cudaGetLastError(), "Hy4 absorbed MLA");
}

// Vision uses a learned 128² grid with the source's fractional scale factor.
__global__ static void hy4_vision_pos(float *x, const __nv_bfloat16 *table, uint32_t gh, uint32_t gw) {
    const uint32_t row = blockIdx.x;
    const float y = fmaxf(0.f, (row / gw + 0.5f) * 128.f / (gh + 0.1f) - 0.5f);
    const float col = fmaxf(0.f, (row % gw + 0.5f) * 128.f / (gw + 0.1f) - 0.5f);
    const uint32_t y0 = (uint32_t)y, x0 = (uint32_t)col, y1 = min(y0 + 1, 127u), x1 = min(x0 + 1, 127u);
    for (uint32_t d = threadIdx.x; d < 1152; d += blockDim.x) {
        const float a = __bfloat162float(table[(y0 * 128 + x0) * 1152 + d]),
                    b = __bfloat162float(table[(y0 * 128 + x1) * 1152 + d]);
        const float c = __bfloat162float(table[(y1 * 128 + x0) * 1152 + d]),
                    e = __bfloat162float(table[(y1 * 128 + x1) * 1152 + d]);
        const float top = fmaf(col - x0, b - a, a), bottom = fmaf(col - x0, e - c, c);
        x[row * 1152 + d] = hy4::bf(x[row * 1152 + d] + hy4::bf(fmaf(y - y0, bottom - top, top)));
    }
}
extern "C" int ds4_gpu_hy4_vision_pos(ds4_gpu_tensor *x, const ds4_gpu_tensor *table, uint32_t gh,
                                      uint32_t gw) {
    if (!gh || !gw || (uint64_t)gh * gw > 16384 || !x || !table) {
        return 0;
    }
    hy4_vision_pos<<<gh * gw, 256, 0, ds4_current_stream()>>>((float *)x->ptr,
                                                              (const __nv_bfloat16 *)table->ptr, gh, gw);
    return cuda_ok(cudaGetLastError(), "Hy4 vision position embedding");
}

__global__ static void hy4_branch_add(float *out, const float *a, const float *b, const float *gate,
                                      uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = hy4::bf(b[i]);
        out[i] = hy4::bf(hy4::bf(a[i]) + (gate ? hy4::bf(v * gate[i % 3072]) : v));
    }
}
extern "C" int ds4_gpu_hy4_add(ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
                               const float *gate, uint32_t n) {
    hy4_branch_add<<<(n + 255) / 256, 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)a->ptr, (const float *)b->ptr, gate, n);
    return cuda_ok(cudaGetLastError(), "Hy4 residual add");
}
