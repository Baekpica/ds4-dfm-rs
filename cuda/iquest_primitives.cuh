#ifndef DS4_IQUEST_PRIMITIVES_CUH
#define DS4_IQUEST_PRIMITIVES_CUH

/* Correctness-first primitives, not a serving-performance claim. All position
 * inputs are live device rows; capture must not bake a scalar frontier. */
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../ds4_iquest_ref.h"

// Torch CPU computes 1/(theta**(d/16)) in F32. Sharing these bits prevents
// device pow ULPs from becoming phase drift at the 524288-token frontier.
__device__ __constant__ static const uint32_t iq_rope_freq_bits[2][IQ_ROT/2] = {
    {1065353216u,1058010522u,1050798235u,1043732615u,1036831949u,1030116802u,1023510243u,1016180025u,
     1008981770u,1001931932u,995049016u,988353832u,981668463u,974350870u,967166816u,960132947u},
    {1065353216u,1054337179u,1043732615u,1033475085u,1023510243u,1012562930u,1001931932u,991652109u,
     981668463u,970790693u,960132947u,949830564u,939827890u,929020495u,918335681u,908010468u}
};

__device__ static float iq_reduce(float value, float *scratch) {
    const unsigned lane = threadIdx.x;
    scratch[lane] = value;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride) { scratch[lane] += scratch[lane + stride]; }
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

__device__ __forceinline__ static float iq_reduce128(float value, float *scratch) {
    constexpr unsigned warp = 32;
    static_assert(IQ_HEAD == 4 * warp, "attention reduction requires 128 threads");
    const unsigned lane = threadIdx.x;
    scratch[lane] = value;
    __syncthreads();
    if (lane < warp) {
        // Preserve the original +64 then +32 additions and their rounding.
        // Materialized leaves prevent product/add contraction under fast math.
        float sum = __fadd_rn(
            __fadd_rn(scratch[lane], scratch[lane + 2 * warp]),
            __fadd_rn(scratch[lane + warp], scratch[lane + 3 * warp]));
        #pragma unroll
        for (unsigned stride = warp / 2; stride; stride >>= 1) {
            const float other = __shfl_down_sync(UINT32_MAX, sum, stride);
            if (lane < stride) { sum = __fadd_rn(sum, other); }
        }
        if (!lane) { scratch[0] = sum; }
    }
    __syncthreads();
    const float result = scratch[0];
    // All four warps must consume the broadcast before the next reduction
    // reuses scratch[0]; attention invokes this once per key and for the sink.
    __syncthreads();
    return result;
}

__global__ static void iquest_rms_kernel(float *out, const float *x,
                                        const float *weight, unsigned width) {
    __shared__ float scratch[256];
    const unsigned row = blockIdx.x, lane = threadIdx.x;
    const uint64_t base = (uint64_t)row * width;
    float square = 0;
    for (unsigned d = lane; d < width; d += blockDim.x) {
        const float value = x[base + d];
        square += value * value;
    }
    const float inverse = rsqrtf(iq_reduce(square, scratch) / width + 1e-6f);
    for (unsigned d = lane; d < width; d += blockDim.x) {
        out[base + d] = x[base + d] * inverse * weight[d];
    }
}

__global__ static void iquest_rope_kernel(float *x, const unsigned *positions,
                                         unsigned heads, unsigned rows, float theta) {
    const uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (uint64_t)rows * heads * IQ_ROT / 2) { return; }
    const unsigned d = index % (IQ_ROT / 2);
    const uint64_t head = index / (IQ_ROT / 2);
    const unsigned row = head / heads;
    float *pair = x + head * IQ_HEAD + d;
    const unsigned variant = theta == 10000.0f ? 0 : 1;
    const float inv_freq = __uint_as_float(iq_rope_freq_bits[variant][d]);
    const float angle = positions[row] * inv_freq;
    // Fast-math sincosf loses phase at long positions. Keep the source F32
    // angle, then use full argument reduction before returning to F32.
    double sine, cosine;
    sincos((double)angle, &sine, &cosine);
    const float s = (float)sine, c = (float)cosine;
    const float first = pair[0], second = pair[IQ_ROT / 2];
    pair[0] = first * c - second * s;
    pair[IQ_ROT / 2] = second * c + first * s;
}

__global__ static void iquest_router_kernel(unsigned *ids, float *weights,
                                           const float *logits) {
    // Bounded reference dispatch. Replace only after top-k and weight parity.
    if (threadIdx.x) { return; }
    const unsigned row = blockIdx.x;
    unsigned *selected = ids + (uint64_t)row * IQ_USED;
    float *weight = weights + (uint64_t)row * IQ_USED;
    const float *scores = logits + (uint64_t)row * IQ_EXPERTS;
    for (unsigned k = 0; k < IQ_USED; k++) {
        unsigned best = IQ_EXPERTS;
        for (unsigned expert = 0; expert < IQ_EXPERTS; expert++) {
            bool used = false;
            for (unsigned j = 0; j < k; j++) { used |= selected[j] == expert; }
            if (used) { continue; }
            if (best == IQ_EXPERTS || scores[expert] > scores[best]) { best = expert; }
        }
        selected[k] = best;
    }
    float sum = 0;
    for (unsigned k = 0; k < IQ_USED; k++) {
        weight[k] = expf(scores[selected[k]] - scores[selected[0]]);
        sum += weight[k];
    }
    for (unsigned k = 0; k < IQ_USED; k++) { weight[k] /= sum; }
}

__global__ static void iquest_kv_kernel(iquest_q8 *cache, const float *key,
                                       const float *value, const unsigned *positions,
                                       unsigned rows, unsigned capacity) {
    __shared__ float scratch[32];
    const unsigned block = blockIdx.x, lane = threadIdx.x;
    const unsigned row = block / IQ_Q8_ROW_BLOCKS;
    if (row >= rows) { return; }
    const unsigned part = block % IQ_Q8_ROW_BLOCKS;
    const unsigned half = IQ_Q8_ROW_BLOCKS / 2;
    const float *src = part < half ? key : value;
    const float x = src[(uint64_t)row * IQ_KV_HEADS * IQ_HEAD + (part % half) * IQ_Q8_BLOCK + lane];
    scratch[lane] = fabsf(x);
    __syncthreads();
    for (unsigned stride = 16; stride; stride >>= 1) {
        if (lane < stride) { scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]); }
        __syncthreads();
    }
    const float d = scratch[0] / 127.0f;
    iquest_q8 *dst = cache + (uint64_t)(positions[row] % capacity) * IQ_Q8_ROW_BLOCKS + part;
    if (!lane) { dst->d = __half_as_ushort(__float2half_rn(d)); }
    dst->qs[lane] = (int8_t)roundf(d == 0 ? 0 : x * (1.0f / d));
}

__device__ static float iq_q8_read(const iquest_q8 *row, unsigned d) {
    const iquest_q8 *block = row + d / IQ_Q8_BLOCK;
    return __half2float(__ushort_as_half(block->d)) * block->qs[d % IQ_Q8_BLOCK];
}

enum class iquest_attn_reduce { Shared, Shuffle };

template<iquest_attn_reduce Reduce>
__device__ __forceinline__ static float iq_attn_reduce(float value, float *scratch) {
    if constexpr (Reduce == iquest_attn_reduce::Shuffle) { return iq_reduce128(value, scratch); }
    return iq_reduce(value, scratch);
}

template<iquest_attn_reduce Reduce>
__device__ __forceinline__ static void iq_attn_body(float *out, const float *query,
                                                  const iquest_q8 *cache, const float *sink,
                                                  const unsigned *positions, unsigned capacity,
                                                  unsigned window, float *scratch) {
    const unsigned row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const unsigned kh = head / (IQ_HEADS / IQ_KV_HEADS);
    const unsigned position = positions[row];
    const unsigned start = window && position + 1 > window ? position + 1 - window : 0;
    const float q = query[((uint64_t)row * IQ_HEADS + head) * IQ_HEAD + lane];
    const float scale = rsqrtf((float)IQ_HEAD);
    const float sink_score = iq_attn_reduce<Reduce>(q * sink[kh * IQ_HEAD + lane], scratch) * scale;
    float maximum = -INFINITY, total = 0, value = 0;
    // Sink is post-RoPE q dot learned key, with a zero value. It is not a
    // scalar logit, a persistent KV token, or a query-independent rescaling.
    for (unsigned token = start; token <= position; token++) {
        const iquest_q8 *kv = cache + (uint64_t)(token % capacity) * IQ_Q8_ROW_BLOCKS;
        const float score = iq_attn_reduce<Reduce>(q * iq_q8_read(kv, kh * IQ_HEAD + lane), scratch) * scale;
        const float next = fmaxf(maximum, score);
        const float prior = expf(maximum - next), weight = expf(score - next);
        value = prior * value + weight * iq_q8_read(kv + IQ_Q8_ROW_BLOCKS / 2, kh * IQ_HEAD + lane);
        total = prior * total + weight;
        maximum = next;
    }
    // The source stores the ordinary attention result as BF16 before
    // applying its FP32 learned-sink factor, then stores BF16 again.
    const float ordinary = __bfloat162float(__float2bfloat16_rn(value / total));
    const float lse = maximum + logf(total);
    const float factor = 1.0f / (1.0f + expf(sink_score - lse));
    out[((uint64_t)row * IQ_HEADS + head) * IQ_HEAD + lane] =
        __bfloat162float(__float2bfloat16_rn(ordinary * factor));
}

__global__ static void iquest_attn_kernel(float *out, const float *query,
                                         const iquest_q8 *cache, const float *sink,
                                         const unsigned *positions, unsigned capacity,
                                         unsigned window) {
    __shared__ float scratch[IQ_HEAD];
    iq_attn_body<iquest_attn_reduce::Shared>(out, query, cache, sink, positions, capacity, window, scratch);
}

__global__ static void iquest_attn_shuffle_kernel(float *out, const float *query,
                                                 const iquest_q8 *cache, const float *sink,
                                                 const unsigned *positions, unsigned capacity,
                                                 unsigned window) {
    __shared__ float scratch[IQ_HEAD];
    iq_attn_body<iquest_attn_reduce::Shuffle>(out, query, cache, sink, positions, capacity, window, scratch);
}

__global__ static void iquest_add_kernel(float *out, const float *residual,
                                        const float *branch, uint64_t count,
                                        float scale) {
    const uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) { out[index] = residual[index] + scale * branch[index]; }
}

__global__ static void iquest_sum_kernel(float *out, const float *experts,
                                        const float *weights, unsigned rows) {
    const uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (uint64_t)rows * IQ_EMBED) { return; }
    const unsigned row = index / IQ_EMBED, d = index % IQ_EMBED;
    float result = 0;
    for (unsigned k = 0; k < IQ_USED; k++) {
        const float down = __bfloat162float(__float2bfloat16_rn(
            experts[((uint64_t)row * IQ_USED + k) * IQ_EMBED + d]));
        result += down * weights[row * IQ_USED + k];
    }
    out[index] = __bfloat162float(__float2bfloat16_rn(result));
}

__global__ static void iquest_swiglu_kernel(float *out, const float *gate,
                                          const float *up, uint64_t count) {
    const uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) { return; }
    // CUDA SGLang kRoundActivation=false: BF16 GEMM inputs, FP32 SiLU and
    // multiply, then one BF16 activation cast before the down projection.
    const float g = __bfloat162float(__float2bfloat16_rn(gate[index]));
    const float u = __bfloat162float(__float2bfloat16_rn(up[index]));
    out[index] = __bfloat162float(__float2bfloat16_rn((g / (1.0f + expf(-g))) * u));
}

__global__ static void iquest_expert_kernel(float *out, const float *x,
        const unsigned *ids, const void *weight, unsigned type,
        unsigned in, unsigned width, unsigned rows, unsigned used, unsigned input_used) {
    __shared__ float scratch[256];
    const unsigned d = blockIdx.x, route = blockIdx.y, lane = threadIdx.x;
    const unsigned token = route / used, k = route % used;
    if (token >= rows) { return; }
    const unsigned expert = ids[token * used + k];
    const uint64_t base = ((uint64_t)expert * width + d) * in;
    const float *src = x + (uint64_t)(token * input_used + (input_used == used ? k : 0)) * in;
    float sum = 0;
    for (unsigned j = lane; j < in; j += blockDim.x) {
        float w;
        if (type == 0) { w = ((const float *)weight)[base + j]; }
        else if (type == 1) { w = __half2float(((const __half *)weight)[base + j]); }
        else { w = __uint_as_float((unsigned)((const uint16_t *)weight)[base + j] << 16); }
        sum += src[j] * w;
    }
    sum = iq_reduce(sum, scratch);
    if (!lane) { out[(uint64_t)route * width + d] = sum; }
}

#endif
