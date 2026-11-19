/* HYV4FlashHyperConnection, Engram and attention, Transformers c2ec8505.
 * All leaf launches run on the calling graph's stream. */
#include "ds4_hy4.h"
#include "cuda/hy4_moe.cuh"
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
/* The preceding branch's residual (when `branch` is set), the iHC mix with
 * exact F32 dots (decode's GEMV arithmetic, not prefill's TF32 GEMM), its
 * combine and the next RMSNorm in one pass. Thread t owns stream values
 * t + 256k, which are all four streams of columns t + 256j, so the combine
 * and the norm stay in registers; their sums keep hc's and the RMSNorm's
 * orders. post is read for the residual before the new weights land. */
template <bool RESIDUAL>
__global__ void __launch_bounds__(256) hc_step(float *streams, float *post, const float *branch, const float *fn,
                                               const float *scale, const float *base, const float *norm_w,
                                               float *norm, bf16 *norm_bf16) {
    const uint32_t row = blockIdx.x, t = threadIdx.x;
    float *x = streams + (uint64_t)row * 12288;
    float v[48];
    #pragma unroll
    for (uint32_t k = 0; k < 48; k++) {
        float s = x[t + 256 * k];
        if constexpr (RESIDUAL) {
            s = bf(s + post[row * 4 + k / 12] * branch[(uint64_t)row * 3072 + t + 256 * (k % 12)]);
            x[t + 256 * k] = s;
        }
        v[k] = s;
    }
    float sq = 0, mix[8] = {};
    #pragma unroll
    for (uint32_t k = 0; k < 48; k++) {
        sq += v[k] * v[k];
        #pragma unroll
        for (uint32_t c = 0; c < 8; c++) {
            mix[c] += v[k] * fn[c * 12288 + t + 256 * k];
        }
    }
    const float inv = rsqrtf(sum(sq) / 12288.f + 1e-5f);
    #pragma unroll
    for (uint32_t c = 0; c < 8; c++) {
        mix[c] = sum(mix[c]);
    }
    __shared__ float pre[4];
    if (t < 4) {
        pre[t] = sigmoid(mix[t] * inv * scale[0] + base[t]) + 1e-6f;
        post[row * 4 + t] = 2.f * sigmoid(mix[t + 4] * inv * scale[1] + base[t + 4]) + 1e-6f;
    }
    __syncthreads();
    float o[12], ss = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 12; j++) {
        float a = 0;
        #pragma unroll
        for (uint32_t c = 0; c < 4; c++) {
            a += v[12 * c + j] * pre[c];
        }
        o[j] = bf(a);
        ss += o[j] * o[j];
    }
    __shared__ float partial[256];
    partial[t] = ss;
    __syncthreads();
    for (uint32_t stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) {
            partial[t] += partial[t + stride];
        }
        __syncthreads();
    }
    const float sc = rsqrtf(partial[0] / 3072.f + 1e-5f);
    #pragma unroll
    for (uint32_t j = 0; j < 12; j++) {
        const uint32_t d = t + 256 * j;
        const float y = bf(o[j] * sc * norm_w[d]);
        norm[(uint64_t)row * 3072 + d] = y;
        if (norm_bf16) {
            norm_bf16[(uint64_t)row * 3072 + d] = __float2bfloat16_rn(y);
        }
    }
}
/* q * sigmoid(gate) in place (solar_sigmoid_gate's expression) and the
 * attn_output GEMM's Q8_0 operand: 144-byte blocks of four F32 scales and
 * 128 int8, block (k / 128) * rows + row, with quantize_mmq_q8_1<D4>'s lane
 * mapping, shuffles and rounding. */
__global__ void gate_q8(float *q, const float *gate, uint8_t *out, unsigned width, unsigned rows) {
    const unsigned col = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (col >= width) {
        return;
    }
    const unsigned row = blockIdx.x;
    float4 *qp = (float4 *)(q + (uint64_t)row * width + col);
    const float4 x = *qp, g = *(const float4 *)(gate + (uint64_t)row * width + col);
    const float4 v = make_float4(x.x / (1.0f + expf(-g.x)), x.y / (1.0f + expf(-g.y)), x.z / (1.0f + expf(-g.z)),
                                 x.w / (1.0f + expf(-g.w)));
    *qp = v;
    float amax = fabsf(v.x);
    amax = fmaxf(amax, fabsf(v.y));
    amax = fmaxf(amax, fabsf(v.z));
    amax = fmaxf(amax, fabsf(v.w));
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset, 32));
    }
    const float d_inv = 127.0f / amax;
    char4 c;
    c.x = roundf(v.x * d_inv);
    c.y = roundf(v.y * d_inv);
    c.z = roundf(v.z * d_inv);
    c.w = roundf(v.w * d_inv);
    uint8_t *block = out + ((uint64_t)(col / 128) * rows + row) * 144;
    ((char4 *)(block + 16))[(col % 128) / 4] = c;
    if (col % 32 == 0) {
        ((float *)block)[(col % 128) / 32] = 1.0f / d_inv;
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
template <typename T> __global__ void moe_sum(float *out, const T *down, const float *weights) {
    const uint32_t r = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= 1536) {
        return;
    }
    // Non-finite expert rows read as zero, so the down MMQ skips its scrub.
    float v = 0;
    for (unsigned e = 0; e < 16; e++) {
        const float x = (float)down[(r * 16 + e) * 1536 + d];
        v += bf(isfinite(x) ? x : 0.f) * weights[r * 16 + e];
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
extern "C" int ds4_gpu_hy4_hc_step(ds4_gpu_tensor *streams, ds4_gpu_tensor *post, const ds4_gpu_tensor *branch,
                                   const float *fn, const float *scale, const float *base, const float *norm_w,
                                   ds4_gpu_tensor *norm, ds4_gpu_tensor *norm_bf16, uint32_t n) {
    if (!fn || !scale || !base || !norm_w) {
        return 0;
    }
    auto kernel = branch ? hy4::hc_step<true> : hy4::hc_step<false>;
    kernel<<<n, 256, 0, ds4_current_stream()>>>(
        (float *)streams->ptr, (float *)post->ptr, branch ? (const float *)branch->ptr : nullptr, fn, scale, base,
        norm_w, (float *)norm->ptr, norm_bf16 ? (hy4::bf16 *)norm_bf16->ptr : nullptr);
    return cuda_ok(cudaGetLastError(), "Hy4 residual and iHC");
}
/* Prefill gates the attention output and publishes its Q8 operand for the
 * attn_output GEMM, which then skips its own quantize pass; decode rows (and
 * DS4_CUDA_NO_NORM_Q8EMIT) take the plain gate. */
extern "C" int ds4_gpu_hy4_gate(ds4_gpu_tensor *q, const ds4_gpu_tensor *gate, uint32_t rows, uint32_t width) {
    size_t payload = 0;
    char *q8 = cuda_norm_q8_prepare(rows, width, &payload);
    if (!q8) {
        return ds4_gpu_solar_sigmoid_gate_tensor(q, q, gate, (uint64_t)rows * width);
    }
    hy4::gate_q8<<<dim3(rows, (width + 511) / 512), 128, 0, ds4_current_stream()>>>(
        (float *)q->ptr, (const float *)gate->ptr, (uint8_t *)q8, width, rows);
    if (!cuda_ok(cudaGetLastError(), "Hy4 gate and Q8")) {
        return 0;
    }
    cuda_norm_q8_publish(q->ptr, rows, width);
    return 1;
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
/* Prefill experts on FP16 tensor cores (cuda/hy4_moe.cuh): gate and up over
 * the token rows, the clamped SwiGLU as FP16 into `mid`, the down into `gate`
 * as BF16 rows (DS4_HY4_DOWN_BF16=0: F32), the type returned. A tensor whose
 * scales leave the folded FP16 range keeps MMQ. -1 refuses before any GPU
 * work (decode widths, DS4_HY4_FP16_MOE=0). */
extern "C" int ds4_gpu_hy4_experts(ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
                                   const ds4_gpu_tensor *x, const ds4_gpu_tensor *ids, const void *map,
                                   uint64_t size, const ds4_hy4_experts *w, uint32_t tokens) {
    enum { DIM = 1536, EXPERTS = 512, USED = 16, MIN_TOKENS = 64, Q4_K = 12, Q5_K = 13 };
    const char *env = getenv("DS4_HY4_FP16_MOE");
    if (tokens < MIN_TOKENS || (env && strcmp(env, "0") == 0) || w->type[0] != w->type[1]) {
        return -1;
    }
    const uint8_t *weight[3];
    static std::unordered_map<uint64_t, bool> fits;
    for (uint32_t i = 0; i < 3; i++) {
        const uint32_t bpb = w->type[i] == Q5_K ? HY4M_Q5 : w->type[i] == Q4_K ? HY4M_Q4 : 0;
        const uint64_t blocks = (uint64_t)EXPERTS * DIM * (DIM / 256);
        if (!bpb || w->bytes[i] < blocks * bpb) {
            return -1;
        }
        weight[i] = (const uint8_t *)cuda_model_range_ptr(map, w->offset[i], w->bytes[i], "Hy4 FP16 experts");
        if (!weight[i]) {
            return -1;
        }
        // One scale-range scan per tensor and process.
        const uint64_t key = (uint64_t)(uintptr_t)weight[i];
        auto known = fits.find(key);
        if (known == fits.end()) {
            uint32_t *bad = (uint32_t *)cuda_tmp_alloc(sizeof(uint32_t), "Hy4 expert scale check");
            uint32_t found = 1;
            if (!bad || cudaMemsetAsync(bad, 0, sizeof(uint32_t), ds4_current_stream()) != cudaSuccess) {
                return 0;
            }
            if (bpb == HY4M_Q5) {
                hy4m_check<HY4M_Q5><<<(blocks + 255) / 256, 256, 0, ds4_current_stream()>>>(weight[i], blocks, bad);
            } else {
                hy4m_check<HY4M_Q4><<<(blocks + 255) / 256, 256, 0, ds4_current_stream()>>>(weight[i], blocks, bad);
            }
            if (cudaMemcpyAsync(&found, bad, sizeof(found), cudaMemcpyDeviceToHost, ds4_current_stream()) != cudaSuccess ||
                cudaStreamSynchronize(ds4_current_stream()) != cudaSuccess) {
                return 0;
            }
            known = fits.emplace(key, found == 0).first;
        }
        if (!known->second) {
            return -1;
        }
    }
    using kernel = void (*)(const uint8_t *, const uint8_t *, const __half *, const uint32_t *, const float *, void *,
                            const uint32_t *, const uint4 *, const uint32_t *, uint32_t, uint32_t);
    static const kernel kernels[2][3] = {
        {hy4m_gemm<HY4M_Q4, HY4M_F32>, hy4m_gemm<HY4M_Q4, HY4M_BF16>, hy4m_gemm<HY4M_Q4, HY4M_GLU>},
        {hy4m_gemm<HY4M_Q5, HY4M_F32>, hy4m_gemm<HY4M_Q5, HY4M_BF16>, hy4m_gemm<HY4M_Q5, HY4M_GLU>}};
    static const bool ready = [] {
        for (const auto &type : kernels) {
            for (const kernel k : type) {
                if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, HY4M_SMEM) != cudaSuccess) {
                    return false;
                }
            }
        }
        return true;
    }();
    int sms = 0;
    if (!ready || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) {
        return 0;
    }
    // GLU items cover 64 outputs of gate and up each: twice the down's tiles.
    const char *glu_env = getenv("DS4_HY4_GLU_FUSED"), *bf16_env = getenv("DS4_HY4_DOWN_BF16");
    const hy4m_out mode = glu_env && strcmp(glu_env, "0") == 0 ? HY4M_F32 : HY4M_GLU;
    const hy4m_out down = bf16_env && strcmp(bf16_env, "0") == 0 ? HY4M_F32 : HY4M_BF16;
    const uint32_t rows = tokens * USED, cap = DIM / HY4M_BM * (rows / HY4M_BN + EXPERTS);
    const uint64_t xh_bytes = ((uint64_t)tokens * DIM * sizeof(__half) + 255) / 256 * 256;
    const uint64_t words = (uint64_t)tokens + 3u * EXPERTS + 3 + 5u * rows;
    uint8_t *base = (uint8_t *)cuda_tmp_alloc(xh_bytes + words * 4 + 3ull * cap * sizeof(uint4), "Hy4 FP16 experts");
    if (!base || mid->bytes < (uint64_t)rows * DIM * sizeof(__half) || gate->bytes < (uint64_t)rows * DIM * 4 ||
        up->bytes < (uint64_t)rows * DIM * 4) {
        return 0;
    }
    __half *xh = (__half *)base;
    uint4 *items = (uint4 *)(base + xh_bytes), *glu_items = items + cap;
    uint32_t *wp = (uint32_t *)(glu_items + 2 * cap);
    float *xscale = (float *)wp;
    uint32_t *count = wp + tokens, *bounds = count + EXPERTS, *cursor = bounds + EXPERTS + 1;
    uint32_t *n_items = cursor + EXPERTS, *n_glu = n_items + 1, *sorted = n_glu + 1, *src = sorted + rows;
    float *scale = (float *)(src + rows);
    uint32_t *mid_src = (uint32_t *)(scale + rows);
    float *mid_row_scale = (float *)(mid_src + rows);
    const cudaStream_t st = ds4_current_stream();
    const int32_t *route = (const int32_t *)ids->ptr;
    float *g = (float *)gate->ptr, *u = (float *)up->ptr;
    __half *mh = (__half *)mid->ptr;
    if (cudaMemsetAsync(count, 0, EXPERTS * 4, st) != cudaSuccess || cudaMemsetAsync(n_items, 0, 8, st) != cudaSuccess) {
        return 0;
    }
    hy4m_count<<<(rows + 255) / 256, 256, 0, st>>>(count, route, rows);
    hy4m_scan<<<1, EXPERTS, 0, st>>>(bounds, cursor, count, EXPERTS);
    hy4m_scatter<<<(rows + 255) / 256, 256, 0, st>>>(sorted, cursor, route, rows);
    hy4m_items<<<EXPERTS, 128, 0, st>>>(items, n_items, bounds, DIM / HY4M_BM);
    glmvq_q4p_prep<float><<<tokens, GLMVQ_Q4P_PREP, 0, st>>>((const float *)x->ptr, xh, xscale, DIM);
    hy4m_rows<<<(rows + 255) / 256, 256, 0, st>>>(src, scale, sorted, xscale, rows, USED);
    auto gemm = [&](uint32_t i, hy4m_out out_mode, const __half *in, const uint32_t *in_src, const float *in_scale,
                    void *out, const uint4 *list, const uint32_t *n) {
        kernels[w->type[i] == Q5_K][out_mode]<<<sms, HY4M_THREADS, HY4M_SMEM, st>>>(
            weight[i], weight[1], in, in_src, in_scale, out, sorted, list, n, DIM, DIM);
    };
    if (mode == HY4M_GLU) {
        hy4m_items<<<EXPERTS, 128, 0, st>>>(glu_items, n_glu, bounds, DIM / (HY4M_BM / 2));
        gemm(0, HY4M_GLU, xh, src, scale, mh, glu_items, n_glu);
        hy4m_rows<<<(rows + 255) / 256, 256, 0, st>>>(mid_src, mid_row_scale, sorted, nullptr, rows, 1);
    } else {
        gemm(0, HY4M_F32, xh, src, scale, g, items, n_items);
        gemm(1, HY4M_F32, xh, src, scale, u, items, n_items);
        hy4m_swiglu_prep<<<rows, GLMVQ_Q4P_PREP, 0, st>>>(g, u, mh, DIM);
        hy4m_rows<<<(rows + 255) / 256, 256, 0, st>>>(mid_src, mid_row_scale, sorted, nullptr, rows, 1);
    }
    gemm(2, down, mh, mid_src, mid_row_scale, g, items, n_items);
    if (!cuda_ok(cudaGetLastError(), "Hy4 FP16 experts")) {
        return 0;
    }
    return down == HY4M_BF16 ? DS4_HY4_ROWS_BF16 : DS4_HY4_ROWS_F32;
}
extern "C" int ds4_gpu_hy4_moe_sum(ds4_gpu_tensor *out, const ds4_gpu_tensor *down,
                                   const ds4_gpu_tensor *weights, uint32_t n, ds4_hy4_rows type) {
    if (type == DS4_HY4_ROWS_BF16) {
        hy4::moe_sum<<<dim3(6, n), 256, 0, ds4_current_stream()>>>(
            (float *)out->ptr, (const hy4::bf16 *)down->ptr, (const float *)weights->ptr);
    } else {
        hy4::moe_sum<<<dim3(6, n), 256, 0, ds4_current_stream()>>>((float *)out->ptr, (const float *)down->ptr,
                                                                   (const float *)weights->ptr);
    }
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
                                   uint32_t in, uint32_t stride, uint32_t width, ds4_hy4_head_input input,
                                   ds4_hy4_head_output output) {
    const uint64_t bytes = (uint64_t)h * in * width * 2;
    const void *weight = ds4_gpu_hy4_weight(map, size, offset, bytes);
    if (!weight || !g_cublas_ready ||
        (input == DS4_HY4_HEAD_F32 && !ds4_gpu_f32_to_bf16(pack, x, (uint64_t)n * h * stride))) {
        return 0;
    }
    const float alpha = 1.f, beta = 0.f;
    if (cublasSetStream(g_cublas, ds4_current_stream()) != CUBLAS_STATUS_SUCCESS) {
        return 0;
    }
    cuda_cublas_ws_prep(ds4_current_stream());
    const cublasStatus_t status = cublasGemmStridedBatchedEx(
        g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, width, n, in, &alpha, weight, CUDA_R_16BF, in,
        (long long)in * width, pack->ptr, CUDA_R_16BF, h * stride, stride, &beta, out->ptr,
        output == DS4_HY4_HEAD_OUT_BF16 ? CUDA_R_16BF : CUDA_R_32F, h * width, width, h, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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
__global__ void __launch_bounds__(GLMVQ_IDX_THREADS) index_mma(float *tile, const float *q,
        const float *w, const bf16 *groups, unsigned count, unsigned rows, unsigned pos, unsigned first) {
    glmvq_index_tile<bf16, true>(tile, q, w, groups + (uint64_t)first * 128, count, rows, pos, first);
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
    flash(bf16 *out, const bf16 *qa, const float *q, const void *latent, const int8_t *exps,
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
        (bf16 *)out->ptr, (const bf16 *)a->ptr, (const float *)q->ptr, c->latent->ptr, exps,
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
    constexpr unsigned smem = (GLMVQ_IDX_ROWS * 16 + GLMVQ_IDX_POOLS) * GLMVQ_IDX_PITCH *
                              sizeof(hy4::bf16) + GLMVQ_IDX_ROWS * 16 * sizeof(float);
    static const bool supported = cudaFuncSetAttribute(hy4::index_mma, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                       smem) == cudaSuccess;
    const char *env = getenv("DS4_HY4_INDEX_MMA");
    const bool mma = supported && rows >= 16 && !(env && strcmp(env, "0") == 0);
    const uint32_t groups = (pos + rows - 1) / 4;
    for (uint32_t first = 0; first < groups; first += 1024) {
        if (mma) {
            const dim3 grid(1024 / GLMVQ_IDX_POOLS, (rows + GLMVQ_IDX_ROWS - 1) / GLMVQ_IDX_ROWS);
            hy4::index_mma<<<grid, GLMVQ_IDX_THREADS, smem, ds4_current_stream()>>>(
                (float *)tile->ptr, (const float *)q->ptr, (const float *)w->ptr,
                (const hy4::bf16 *)c->groups->ptr, std::min(1024u, groups - first), rows, pos, first);
        } else {
            hy4::index_scores<<<dim3(128, rows), 256, 0, ds4_current_stream()>>>(
                (float *)tile->ptr, (const float *)q->ptr, (const float *)w->ptr,
                (const hy4::bf16 *)c->groups->ptr, pos, first);
        }
        hy4::index_merge<<<rows, 128, 0, ds4_current_stream()>>>((int *)ids->ptr, (float *)best->ptr,
                                                                 (const float *)tile->ptr, first);
    }
    return cuda_ok(cudaGetLastError(), "Hy4 top complete groups");
}
extern "C" int ds4_gpu_hy4_flash(uint32_t rows, int draft) {
    const char *flash = getenv("DS4_HY4_FLASH");
    return !draft && rows >= 16 && !(flash && strcmp(flash, "0") == 0);
}
/* Prefill rows (ds4_gpu_hy4_flash) go through the tiled kernel, reading BF16
 * q_abs, straight into `pack` as the v_b projection's BF16 operand
 * (DS4_HY4_HEAD_PACKED); decode and draft rows read F32 q_abs and write F32
 * `out` (DS4_HY4_HEAD_F32). Returns -1 on failure. */
extern "C" int ds4_gpu_hy4_attention(ds4_gpu_tensor *out, ds4_gpu_tensor *pack, const ds4_gpu_tensor *a,
                                     const ds4_gpu_tensor *q, const ds4_hy4_cache *c, const ds4_gpu_tensor *ids,
                                     const float *sinks, uint32_t pos, uint32_t rows, uint32_t heads, int draft) {
    if (!sinks) {
        return -1;
    }
    if (ds4_gpu_hy4_flash(rows, draft)) {
        const bool launched = heads == 64
                                  ? hy4::launch_flash<64, false>(pack, a, q, c, ids, sinks, pos, rows)
                              : c->fp8 ? hy4::launch_flash<32, true>(pack, a, q, c, ids, sinks, pos, rows)
                                       : hy4::launch_flash<32, false>(pack, a, q, c, ids, sinks, pos, rows);
        // q_abs is BF16 here, so the F32 kernels cannot stand in.
        return launched && cuda_ok(cudaGetLastError(), "Hy4 flash MLA") ? DS4_HY4_HEAD_PACKED : -1;
    }
    auto kernel = draft ? (c->fp8 ? hy4::attend<true, true> : hy4::attend<false, true>)
                        : (c->fp8 ? hy4::attend<true, false> : hy4::attend<false, false>);
    kernel<<<dim3(heads, rows), 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)a->ptr, (const float *)q->ptr, c->latent->ptr,
        (const hy4::bf16 *)c->rope->ptr, ids ? (const int *)ids->ptr : nullptr, sinks, c->cap, pos, heads);
    return cuda_ok(cudaGetLastError(), "Hy4 absorbed MLA") ? DS4_HY4_HEAD_F32 : -1;
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
