#pragma once
#include "naive_primitives.cuh"
static constexpr float N05_DF_QK_SCALE = 0.08838834764831845f; // 1/sqrt(128)

/* Capture post-layer residuals in source tap order, retaining only the tail
 * needed by the 1024-wide draft. Earlier prefill rows need no draft state. */
__global__ static void naive_df_tap(float *out, const float *hidden,
                                     unsigned first, unsigned rows, unsigned tap) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * N05_EMBED) { return; }
    out[(i / N05_EMBED) * N05_DF_SLOT + tap * N05_EMBED + i % N05_EMBED] = hidden[(uint64_t)first * N05_EMBED + i];
}

__global__ static void naive_df_mask(float *hidden, const float *mask, unsigned rows) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * N05_EMBED) { return; }
    hidden[i] = naive_bf16(i < N05_EMBED ? hidden[i] : mask[i % N05_EMBED]);
}

/* Q/K are head-normalized before full split-half RoPE. Source BF16 products
 * and sums remain distinct; compiler fusion must not cross these boundaries. */
__global__ static void naive_df_rope(float *x, const float2 *table, unsigned heads) {
    const unsigned d = threadIdx.x, head = blockIdx.x, row = blockIdx.y;
    const uint64_t base = ((uint64_t)row * heads + head) * N05_DF_DIM;
    __shared__ float input[N05_DF_DIM];
    input[d] = naive_bf16(x[base + d]);
    __syncthreads();
    const unsigned pair = d % (N05_DF_DIM / 2);
    const float2 cs = table[(uint64_t)row * (N05_DF_DIM / 2) + pair];
    const float a = input[pair], b = input[pair + N05_DF_DIM / 2];
    const float c = naive_bf16(cs.x), s = naive_bf16(cs.y);
    x[base + d] = d < N05_DF_DIM / 2
        ? naive_bf16(naive_bf16(a * c) - naive_bf16(b * s))
        : naive_bf16(naive_bf16(b * c) + naive_bf16(a * s));
}

__global__ static void naive_df_store(__nv_bfloat16 *cache, const float *k,
                                       const float *v, const unsigned *positions, unsigned rows) {
    const unsigned width = N05_DF_KV * N05_DF_DIM, stride = 2 * width;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * stride) { return; }
    const unsigned row = i / stride, col = i % stride;
    const float value = col < width ? k[(uint64_t)row * width + col] : v[(uint64_t)row * width + col - width];
    cache[(uint64_t)(positions[row] % N05_DF_CAP) * stride + col] = __float2bfloat16_rn(value);
}

/* Context is committed target-derived KV. Noise keys are the entire block,
 * including future positions. abs(query-key)<1024 matches the source mask.
 * The ring has six extra slots, so rejected verify rows cannot evict context
 * needed after an anchor-only commit. Frontier rollback is sufficient. */
__global__ static void naive_df_attn(float *out, const float *q,
                                      const __nv_bfloat16 *context, const float *k, const float *v,
                                      unsigned context_first, unsigned start, unsigned rows) {
    enum { WARP = 32, WARPS = 4, WIDTH = N05_DF_KV * N05_DF_DIM };
    const unsigned lane = threadIdx.x % WARP, head = blockIdx.x * WARPS + threadIdx.x / WARP;
    const unsigned row = blockIdx.y, pos = start + row;
    const unsigned first = max(context_first, pos >= N05_DF_WINDOW ? pos - N05_DF_WINDOW + 1 : 0);
    const unsigned count = start - first + rows, kh = head / (N05_DF_HEADS / N05_DF_KV);
    float query[N05_DF_DIM / WARP], acc[N05_DF_DIM / WARP] = {};
    for (unsigned d = 0; d < N05_DF_DIM / WARP; d++) {
        query[d] = q[((uint64_t)row * N05_DF_HEADS + head) * N05_DF_DIM + d * WARP + lane];
    }
    float maximum = -INFINITY, denominator = 0;
    for (unsigned pass = 0; pass < 2; pass++) {
        for (unsigned i = 0; i < count; i++) {
            const unsigned p = first + i;
            float dot = 0;
            for (unsigned d = 0; d < N05_DF_DIM / WARP; d++) {
                const unsigned col = kh * N05_DF_DIM + d * WARP + lane;
                const float key = p < start
                    ? __bfloat162float(context[(uint64_t)(p % N05_DF_CAP) * 2 * WIDTH + col])
                    : naive_bf16(k[(uint64_t)(p - start) * WIDTH + col]);
                dot = __fmaf_rn(query[d], key, dot);
            }
            for (unsigned step = WARP / 2; step; step /= 2) { dot += __shfl_xor_sync(0xffffffff, dot, step); }
            const float score = naive_bf16(naive_bf16(dot) * N05_DF_QK_SCALE);
            if (!pass) {
                const float next = fmaxf(maximum, score);
                denominator = denominator * expf(maximum - next) + expf(score - next);
                maximum = next;
                continue;
            }
            const float probability = naive_bf16(expf(score - maximum) / denominator);
            for (unsigned d = 0; d < N05_DF_DIM / WARP; d++) {
                const unsigned col = kh * N05_DF_DIM + d * WARP + lane;
                const float value = p < start
                    ? __bfloat162float(context[(uint64_t)(p % N05_DF_CAP) * 2 * WIDTH + WIDTH + col])
                    : naive_bf16(v[(uint64_t)(p - start) * WIDTH + col]);
                acc[d] = __fmaf_rn(probability, value, acc[d]);
            }
        }
    }
    for (unsigned d = 0; d < N05_DF_DIM / WARP; d++) {
        out[((uint64_t)row * N05_DF_HEADS + head) * N05_DF_DIM + d * WARP + lane] = naive_bf16(acc[d]);
    }
}

/* Confidence is the released raw linear scalar, without a sigmoid. */
__global__ static void naive_df_conf(float *out, const float *hidden, const float *markov,
                                      const float *weight, const float *bias) {
    const unsigned tid = threadIdx.x;
    __shared__ float work[256];
    float sum = 0;
    for (unsigned d = tid; d < N05_EMBED + N05_DF_RANK; d += 256) {
        const float x = d < N05_EMBED ? hidden[d] : naive_bf16(markov[d - N05_EMBED]);
        sum += x * weight[d];
    }
    work[tid] = sum;
    __syncthreads();
    for (unsigned step = 128; step; step /= 2) {
        if (tid < step) { work[tid] += work[tid + step]; }
        __syncthreads();
    }
    if (!tid) { out[0] = naive_bf16(work[0] + bias[0]); }
}

__device__ static void naive_df_insert(float value, unsigned id,
                                        float &a, unsigned &ia, float &b, unsigned &ib) {
    if (value > a || (value == a && id < ia)) { b = a; ib = ia; a = value; ia = id; return; }
    if (value > b || (value == b && id < ib)) { b = value; ib = id; }
}

/* A bounded head reduction supplies the shared recursive-draft margin
 * without copying an entire vocabulary to the host at every proposal. */
__global__ static void naive_df_top2(unsigned *out, const float *logits) {
    const unsigned tid = threadIdx.x;
    __shared__ float a[256], b[256];
    __shared__ unsigned ia[256], ib[256], bad[256];
    float va = -INFINITY, vb = -INFINITY;
    unsigned ida = UINT32_MAX, idb = UINT32_MAX, invalid = 0;
    for (unsigned i = tid; i < N05_VOCAB; i += 256) {
        const float value = logits[i];
        invalid |= !isfinite(value);
        naive_df_insert(value, i, va, ida, vb, idb);
    }
    a[tid] = va; b[tid] = vb; ia[tid] = ida; ib[tid] = idb; bad[tid] = invalid;
    __syncthreads();
    for (unsigned step = 128; step; step /= 2) {
        if (tid < step) {
            naive_df_insert(a[tid + step], ia[tid + step], a[tid], ia[tid], b[tid], ib[tid]);
            naive_df_insert(b[tid + step], ib[tid + step], a[tid], ia[tid], b[tid], ib[tid]);
            bad[tid] |= bad[tid + step];
        }
        __syncthreads();
    }
    if (tid) { return; }
    out[0] = ia[0]; out[1] = ib[0];
    float *values = (float *)(out + 2);
    values[0] = bad[0] ? NAN : a[0]; values[1] = b[0];
}
