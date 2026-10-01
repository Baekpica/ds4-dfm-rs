#pragma once
#include <stdint.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cub/block/block_radix_sort.cuh>
#include "../ds4_naive_plan.h"

static constexpr float N05_EPS = 1e-5f;
static constexpr float N05_V_SCALE = .707f;
static constexpr float N05_QK_SCALE = 0.07216878364870322f; // 1/sqrt(192)
static constexpr unsigned N05_INDEX_WARP = 32;
static constexpr unsigned N05_INDEX_PARTS = N05_INDEX_DIM / N05_INDEX_WARP;

enum class NaiveCache { Ring, Full };
enum class NaiveIndexLayout { Planar, Warp };
enum class NaiveSoftmax { Walk, Unit };

template<NaiveSoftmax MODE> __device__ static void naive_softmax_step(
        float score, float &maximum, float &denominator) {
    const float next = fmaxf(maximum, score);
    if constexpr (MODE == NaiveSoftmax::Unit) {
        // One exponent is exactly one. Keep FMA/add order and exceptional values.
        if (isfinite(maximum) && isfinite(score)) {
            denominator = score > maximum ? __fmaf_rn(denominator, expf(maximum - score), 1.0f)
                                          : denominator + expf(score - maximum);
            maximum = next;
            return;
        }
    }
    denominator = denominator * expf(maximum - next) + expf(score - next);
    maximum = next;
}

template<NaiveCache CACHE> __device__ static unsigned naive_cache_slot(unsigned key, unsigned capacity) {
    // DSA's bounded frontier keeps every causal ID inside its full history.
    if constexpr (CACHE == NaiveCache::Full) { return key; }
    return key % capacity;
}

__device__ static float naive_e4m3(uint8_t code) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw(code, __NV_E4M3);
    return __half2float(__half(h));
}

__device__ static float naive_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

/* Linear/MMQ accumulators are F32. Restore the source BF16 boundary before
 * partial RoPE, and round each source multiply/add as BF16 does. */
__global__ static void naive_rope(float *x, const float2 *table, unsigned heads, unsigned width) {
    const unsigned dim = threadIdx.x, head = blockIdx.x, row = blockIdx.y;
    const uint64_t base = ((uint64_t)row * heads + head) * width;
    __shared__ float input[N05_KEY];
    if (dim < width) { input[dim] = naive_bf16(x[base + dim]); }
    __syncthreads();
    if (dim >= width) { return; }
    float value = input[dim];
    if (dim < N05_ROT) {
        const unsigned pair = dim % (N05_ROT / 2);
        const float2 cs = table[(uint64_t)row * (N05_ROT / 2) + pair];
        const float c = naive_bf16(cs.x), s = naive_bf16(cs.y);
        const float a = input[pair], b = input[pair + N05_ROT / 2];
        value = dim < N05_ROT / 2
            ? naive_bf16(naive_bf16(a * c) - naive_bf16(b * s))
            : naive_bf16(naive_bf16(b * c) + naive_bf16(a * s));
    }
    x[base + dim] = value;
}

__global__ static void naive_round(float *x, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { x[i] = naive_bf16(x[i]); }
}

/* Key norm is affine LayerNorm, not RMSNorm. Store BF16 activations in an
 * F32 buffer so the existing direct-hidden matmul boundary remains narrow. */
__global__ static void naive_key_norm(float *x, const float *weight, const float *bias) {
    const unsigned dim = threadIdx.x, row = blockIdx.x;
    __shared__ float values[N05_INDEX_DIM], work[N05_INDEX_DIM];
    const float v = naive_bf16(x[(uint64_t)row * N05_INDEX_DIM + dim]);
    values[dim] = v; work[dim] = v;
    __syncthreads();
    for (unsigned step = N05_INDEX_DIM / 2; step; step /= 2) {
        if (dim < step) { work[dim] += work[dim + step]; }
        __syncthreads();
    }
    const float mean = work[0] / N05_INDEX_DIM;
    __syncthreads();
    work[dim] = (values[dim] - mean) * (values[dim] - mean);
    __syncthreads();
    for (unsigned step = N05_INDEX_DIM / 2; step; step /= 2) {
        if (dim < step) { work[dim] += work[dim + step]; }
        __syncthreads();
    }
    const float inv = rsqrtf(work[0] / N05_INDEX_DIM + N05_EPS);
    x[(uint64_t)row * N05_INDEX_DIM + dim] = naive_bf16((v - mean) * inv * weight[dim] + bias[dim]);
}

__global__ static void naive_rms(
        float *out, const float *x, const float *weight, unsigned width) {
    enum { THREADS = 256 };
    const unsigned tid = threadIdx.x, row = blockIdx.x;
    __shared__ float work[THREADS];
    float sum = 0;
    for (unsigned d = tid; d < width; d += THREADS) {
        const float value = x[(uint64_t)row * width + d];
        sum += value * value;
    }
    work[tid] = sum;
    __syncthreads();
    for (unsigned step = THREADS / 2; step; step /= 2) {
        if (tid < step) { work[tid] += work[tid + step]; }
        __syncthreads();
    }
    const float inv = rsqrtf(work[0] / width + N05_EPS);
    for (unsigned d = tid; d < width; d += THREADS) {
        const uint64_t at = (uint64_t)row * width + d;
        out[at] = naive_bf16(naive_bf16(x[at] * inv) * weight[d]);
    }
}

__global__ static void naive_kv_store(
        __nv_bfloat16 *cache, const float *k, const float *v,
        const unsigned *positions, unsigned heads, unsigned rows, unsigned capacity) {
    const unsigned kw = heads * N05_KEY, vw = heads * N05_VALUE, stride = kw + vw;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * stride) { return; }
    const unsigned row = i / stride, col = i % stride, slot = positions[row] % capacity;
    const float value = col < kw ? k[(uint64_t)row * kw + col] : v[(uint64_t)row * vw + col - kw];
    cache[(uint64_t)slot * stride + col] = __float2bfloat16_rn(value);
}

/* BF16 dot/logit and probability boundaries match the pinned reference.
 * One-row decode may retain the rounded scores on chip instead of repeating
 * QK; both paths retain the serial denominator and V accumulation order.
 * The two passes avoid a scores[heads,history] allocation. Sparse IDs are
 * ascending; SWA walks exactly the causal window and adds its zero-V sink. */
template<unsigned WARPS = 4, unsigned SCORE_CAP = 0, NaiveCache CACHE = NaiveCache::Ring,
         NaiveSoftmax SOFTMAX = NaiveSoftmax::Walk>
__global__ static void naive_attention(
        float *out, const float *q, const __nv_bfloat16 *cache, const float *sinks,
        const unsigned *positions, const unsigned *selected,
        unsigned kv_heads, unsigned capacity, unsigned window) {
    enum { WARP = 32 };
    const unsigned lane = threadIdx.x % WARP;
    const unsigned warp = threadIdx.x / WARP;
    const unsigned head = blockIdx.x * WARPS + warp, row = blockIdx.y;
    __shared__ __nv_bfloat16 scores[WARPS][SCORE_CAP ? SCORE_CAP : 1];
    const unsigned pos = positions[row];
    const unsigned first = window && pos + 1 > window ? pos + 1 - window : 0;
    const unsigned count = window ? pos - first + 1 : min(pos + 1, (unsigned)N05_TOP_K);
    const unsigned kv_head = head / (N05_HEADS / kv_heads), stride = kv_heads * (N05_KEY + N05_VALUE);
    float query[N05_KEY / WARP], acc[N05_VALUE / WARP] = {};
    for (unsigned d = 0; d < N05_KEY / WARP; d++) {
        query[d] = q[((uint64_t)row * N05_HEADS + head) * N05_KEY + lane + d * WARP];
    }
    float maximum = sinks ? sinks[head] : -INFINITY, denominator = sinks ? 1 : 0;
    for (unsigned i = 0; i < count; i++) {
        const unsigned key = window ? first + i : selected[(uint64_t)row * N05_TOP_K + i];
        if (key > pos) { continue; }
        const __nv_bfloat16 *slot = cache + (uint64_t)naive_cache_slot<CACHE>(key, capacity) * stride;
        float dot = 0;
        for (unsigned d = 0; d < N05_KEY / WARP; d++) {
            dot = __fmaf_rn(query[d], __bfloat162float(slot[kv_head * N05_KEY + lane + d * WARP]), dot);
        }
        for (unsigned step = WARP / 2; step; step /= 2) { dot += __shfl_xor_sync(0xffffffff, dot, step); }
        const float score = naive_bf16(naive_bf16(dot) * N05_QK_SCALE);
        if constexpr (SCORE_CAP) {
            if (!lane) { scores[warp][i] = __float2bfloat16_rn(score); }
        }
        naive_softmax_step<SOFTMAX>(score, maximum, denominator);
    }
    if constexpr (SCORE_CAP) { __syncwarp(); }
    for (unsigned i = 0; i < count; i++) {
        const unsigned key = window ? first + i : selected[(uint64_t)row * N05_TOP_K + i];
        if (key > pos) { continue; }
        const __nv_bfloat16 *slot = cache + (uint64_t)naive_cache_slot<CACHE>(key, capacity) * stride;
        float score;
        if constexpr (SCORE_CAP) {
            score = __bfloat162float(scores[warp][i]);
        } else {
            float dot = 0;
            for (unsigned d = 0; d < N05_KEY / WARP; d++) {
                dot = __fmaf_rn(query[d], __bfloat162float(slot[kv_head * N05_KEY + lane + d * WARP]), dot);
            }
            for (unsigned step = WARP / 2; step; step /= 2) { dot += __shfl_xor_sync(0xffffffff, dot, step); }
            score = naive_bf16(naive_bf16(dot) * N05_QK_SCALE);
        }
        const float probability = naive_bf16(__fdiv_rn(expf(score - maximum), denominator));
        for (unsigned d = 0; d < N05_VALUE / WARP; d++) {
            const float value = naive_bf16(__bfloat162float(slot[kv_heads * N05_KEY + kv_head * N05_VALUE + lane + d * WARP]) * N05_V_SCALE);
            acc[d] = __fmaf_rn(probability, value, acc[d]);
        }
    }
    for (unsigned d = 0; d < N05_VALUE / WARP; d++) {
        out[((uint64_t)row * N05_HEADS + head) * N05_VALUE + lane + d * WARP] = naive_bf16(acc[d]);
    }
}

__global__ static void naive_router(int *ids, float *weights, const float *logits, const float *bias) {
    __shared__ float prob[N05_EXPERTS], score[N05_EXPERTS];
    const unsigned e = threadIdx.x, row = blockIdx.x;
    prob[e] = 1.0f / (1.0f + expf(-logits[(uint64_t)row * N05_EXPERTS + e]));
    score[e] = prob[e] + bias[e];
    __syncthreads();
    if (e) { return; }
    ids += (uint64_t)row * N05_USED; weights += (uint64_t)row * N05_USED;
    float sum = 0;
    for (unsigned k = 0; k < N05_USED; k++) {
        unsigned best = 0;
        for (unsigned j = 1; j < N05_EXPERTS; j++) { if (score[j] > score[best]) { best = j; } }
        ids[k] = best; weights[k] = prob[best]; sum += prob[best]; score[best] = -INFINITY;
    }
    for (unsigned k = 0; k < N05_USED; k++) { weights[k] /= sum + 1e-20f; }
    // BF16 expert accumulation is in numeric expert order in the source.
    for (unsigned k = 1; k < N05_USED; k++) {
        const int id = ids[k]; const float weight = weights[k];
        unsigned j = k;
        while (j && ids[j - 1] > id) { ids[j] = ids[j - 1]; weights[j] = weights[j - 1]; j--; }
        ids[j] = id; weights[j] = weight;
    }
}

__global__ static void naive_swiglu(float *out, const float *gate, const float *up, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const float g = naive_bf16(gate[i]), u = naive_bf16(up[i]);
    out[i] = naive_bf16(naive_bf16(g / (1 + expf(-g))) * u);
}

__global__ static void naive_add(float *cur, const float *other, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { cur[i] = naive_bf16(cur[i] + naive_bf16(other[i])); }
}

__global__ static void naive_sum(float *out, const float *down, const float *weights, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const uint64_t row = i / N05_EMBED, col = i % N05_EMBED;
    float sum = 0;
    for (unsigned e = 0; e < N05_USED; e++) {
        const float term = naive_bf16(naive_bf16(down[(row * N05_USED + e) * N05_EMBED + col]) * weights[row * N05_USED + e]);
        sum = naive_bf16(sum + term);
    }
    out[i] = sum;
}

/* One head/token row. Keep the original F32 scale: rounding it to BF16
 * changes reconstructed keys and can change the selected history. */
__global__ static void naive_fp8_pack(
        uint8_t *codes, float *scales, const float *x, const unsigned *positions, unsigned rows) {
    const unsigned row = blockIdx.x, dim = threadIdx.x;
    if (row >= rows || dim >= N05_INDEX_DIM) { return; }
    __shared__ float magnitude[N05_INDEX_DIM];
    const float value = x[(uint64_t)row * N05_INDEX_DIM + dim];
    magnitude[dim] = fabsf(value);
    __syncthreads();
    for (unsigned step = N05_INDEX_DIM / 2; step; step /= 2) {
        if (dim < step) { magnitude[dim] = fmaxf(magnitude[dim], magnitude[dim + step]); }
        __syncthreads();
    }
    const float scale = __fdiv_rn(fmaxf(magnitude[0], 1e-4f), 448.0f);
    const unsigned position = positions[row];
    if (!dim) { scales[position] = scale; }
    codes[(uint64_t)position * N05_INDEX_DIM + dim] = __nv_cvt_float_to_fp8(
        fminf(448.0f, fmaxf(-448.0f, __fdiv_rn(value, scale))), __NV_SATFINITE, __NV_E4M3);
}

/* Query round-trip stays F32 for score GEMMs; history keeps codes/scales. */
template<NaiveIndexLayout LAYOUT = NaiveIndexLayout::Planar>
__global__ static void naive_fp8_query(float *x, unsigned rows) {
    const unsigned row = blockIdx.x, dim = threadIdx.x;
    if (row >= rows || dim >= N05_INDEX_DIM) { return; }
    __shared__ float magnitude[N05_INDEX_DIM];
    const uint64_t at = (uint64_t)row * N05_INDEX_DIM + dim;
    const float value = x[at];
    magnitude[dim] = fabsf(value);
    __syncthreads();
    for (unsigned step = N05_INDEX_DIM / 2; step; step /= 2) {
        if (dim < step) { magnitude[dim] = fmaxf(magnitude[dim], magnitude[dim + step]); }
        __syncthreads();
    }
    const float scale = __fdiv_rn(fmaxf(magnitude[0], 1e-4f), 448.0f);
    const uint8_t code = __nv_cvt_float_to_fp8(
        fminf(448.0f, fmaxf(-448.0f, __fdiv_rn(value, scale))), __NV_SATFINITE, __NV_E4M3);
    // Four values per lane permit one vector read with the same FMA order.
    const unsigned out = LAYOUT == NaiveIndexLayout::Warp
        ? (dim % N05_INDEX_WARP) * N05_INDEX_PARTS + dim / N05_INDEX_WARP : dim;
    x[(uint64_t)row * N05_INDEX_DIM + out] = __fmul_rn(naive_e4m3(code), scale);
}

/* Signed head projections weight ReLU(dot), not a head softmax. The scalar
 * baseline bounds scratch to queries*history and reconstructs each key once
 * across all head dots. Warp packing changes loads, not reconstructed
 * values, the four-FMA/XOR tree, or the serial signed head sum. */
template<NaiveIndexLayout LAYOUT = NaiveIndexLayout::Planar>
__global__ static void naive_index_scores(
        float *scores, const float *q, const uint8_t *codes, const float *scales,
        const float *weights, const unsigned *positions, unsigned history) {
    enum { WARP = 32, WARPS = 4 };
    const unsigned lane = threadIdx.x % WARP;
    const unsigned key = blockIdx.x * WARPS + threadIdx.x / WARP;
    const unsigned row = blockIdx.y;
    if (key >= history) { return; }
    if (key > positions[row]) {
        if (!lane) { scores[(uint64_t)row * history + key] = -INFINITY; }
        return;
    }
    float k[N05_INDEX_DIM / WARP];
    for (unsigned d = 0; d < N05_INDEX_DIM / WARP; d++) {
        k[d] = __fmul_rn(naive_e4m3(codes[(uint64_t)key * N05_INDEX_DIM + lane + d * WARP]), scales[key]);
    }
    float score = 0;
    for (unsigned h = 0; h < N05_INDEX_HEADS; h++) {
        float dot = 0;
        const float *head = q + ((uint64_t)row * N05_INDEX_HEADS + h) * N05_INDEX_DIM;
        if constexpr (LAYOUT == NaiveIndexLayout::Warp) {
            const float4 query = ((const float4 *)head)[lane];
            dot = __fmaf_rn(query.x, k[0], dot);
            dot = __fmaf_rn(query.y, k[1], dot);
            dot = __fmaf_rn(query.z, k[2], dot);
            dot = __fmaf_rn(query.w, k[3], dot);
        } else {
            for (unsigned d = 0; d < N05_INDEX_DIM / WARP; d++) {
                dot = __fmaf_rn(head[lane + d * WARP], k[d], dot);
            }
        }
        for (unsigned step = WARP / 2; step; step /= 2) { dot += __shfl_xor_sync(0xffffffff, dot, step); }
        score = __fadd_rn(score, __fmul_rn(fmaxf(0, dot), weights[(uint64_t)row * N05_INDEX_HEADS + h]));
    }
    if (!lane) { scores[(uint64_t)row * history + key] = score; }
}

/* Positive/negative zero tie. Absolute position is the secondary key, so
 * every leaf and merge has exactly the global stable descending order. */
__device__ static uint64_t naive_score_key(float score, unsigned position) {
    if (score == 0) { score = 0; }
    const uint32_t bits = __float_as_uint(score);
    const uint32_t ordered = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
    return ((uint64_t)ordered << 32) | (uint32_t)~position;
}

__global__ static void naive_topk_leaf(
        uint64_t *lists, const float *scores, const unsigned *positions,
        unsigned history, unsigned tiles) {
    enum { THREADS = 256, ITEMS = N05_HISTORY_TILE / THREADS };
    using Sort = cub::BlockRadixSort<uint64_t, THREADS, ITEMS>;
    __shared__ typename Sort::TempStorage temp;
    uint64_t keys[ITEMS];
    const unsigned row = blockIdx.y, tile = blockIdx.x;
    for (unsigned i = 0; i < ITEMS; i++) {
        const unsigned key = tile * N05_HISTORY_TILE + threadIdx.x * ITEMS + i;
        keys[i] = key < history && key <= positions[row]
            ? naive_score_key(scores[(uint64_t)row * history + key], key) : 0;
    }
    Sort(temp).SortDescending(keys);
    for (unsigned i = 0; i < ITEMS; i++) {
        const unsigned local = threadIdx.x * ITEMS + i;
        if (local < N05_TOP_K) { lists[((uint64_t)row * tiles + tile) * N05_TOP_K + local] = keys[i]; }
    }
}

/* Merge-path partitions the prefix at each output rank. Discarded leaf
 * candidates cannot enter a global top-k, so no full-history sort is needed. */
__global__ static void naive_topk_merge(
        uint64_t *out, const uint64_t *in, unsigned lists) {
    const unsigned row = blockIdx.y, pair = blockIdx.x;
    const unsigned out_lists = (lists + 1) / 2;
    const uint64_t *a = in + ((uint64_t)row * lists + 2 * pair) * N05_TOP_K;
    const uint64_t *b = 2 * pair + 1 < lists ? a + N05_TOP_K : nullptr;
    for (unsigned rank = threadIdx.x; rank < N05_TOP_K; rank += blockDim.x) {
        uint64_t key = a[rank];
        if (b) {
            unsigned lo = 0, hi = rank;
            while (lo < hi) {
                const unsigned i = (lo + hi) / 2, j = rank - i;
                if (j && a[i] > b[j - 1]) { lo = i + 1; }
                else { hi = i; }
            }
            const unsigned i = lo, j = rank - lo;
            key = a[i] > b[j] ? a[i] : b[j];
        }
        out[((uint64_t)row * out_lists + pair) * N05_TOP_K + rank] = key;
    }
}

/* Source selects a mask, then aggregates in original history order. Sort
 * chosen positions ascending; empty tail slots stay UINT32_MAX. */
__global__ static void naive_topk_ids(unsigned *ids, const uint64_t *keys) {
    enum { THREADS = 256, ITEMS = N05_TOP_K / THREADS };
    using Sort = cub::BlockRadixSort<unsigned, THREADS, ITEMS>;
    __shared__ typename Sort::TempStorage temp;
    unsigned selected[ITEMS];
    for (unsigned i = 0; i < ITEMS; i++) {
        const unsigned slot = threadIdx.x * ITEMS + i;
        const uint64_t key = keys[(uint64_t)blockIdx.x * N05_TOP_K + slot];
        selected[i] = key ? (unsigned)~key : UINT32_MAX;
    }
    Sort(temp).Sort(selected);
    for (unsigned i = 0; i < ITEMS; i++) {
        ids[(uint64_t)blockIdx.x * N05_TOP_K + threadIdx.x * ITEMS + i] = selected[i];
    }
}

__global__ static void naive_all_ids(unsigned *ids, const unsigned *positions) {
    for (unsigned i = threadIdx.x; i < N05_TOP_K; i += blockDim.x) {
        ids[(uint64_t)blockIdx.x * N05_TOP_K + i] = i <= positions[blockIdx.x] ? i : UINT32_MAX;
    }
}

static cudaError_t naive_topk_launch(
        unsigned *ids, const float *scores, uint64_t *a, uint64_t *b,
        const unsigned *positions, unsigned history, unsigned rows, cudaStream_t stream) {
    if (!ids || !positions || !history || history > N05_CONTEXT || !rows || rows > N05_QUERY_TILE) {
        return cudaErrorInvalidValue;
    }
    if (history <= N05_TOP_K) {
        naive_all_ids<<<rows, 256, 0, stream>>>(ids, positions);
        return cudaGetLastError();
    }
    if (!scores || !a || !b) { return cudaErrorInvalidValue; }
    unsigned lists = (history + N05_HISTORY_TILE - 1) / N05_HISTORY_TILE;
    naive_topk_leaf<<<dim3(lists, rows), 256, 0, stream>>>(a, scores, positions, history, lists);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { return error; }
    while (lists > 1) {
        naive_topk_merge<<<dim3((lists + 1) / 2, rows), 256, 0, stream>>>(b, a, lists);
        error = cudaGetLastError();
        if (error != cudaSuccess) { return error; }
        uint64_t *swap = a; a = b; b = swap;
        lists = (lists + 1) / 2;
    }
    naive_topk_ids<<<rows, 256, 0, stream>>>(ids, a);
    return cudaGetLastError();
}
