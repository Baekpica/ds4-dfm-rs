#pragma once
#include <stdint.h>
#include <cuda_fp16.h>

/* Fused projection rows are Q(64*192), K(kv*192), V(kv*128).
 * Only the first 64 Q/K dimensions rotate; MiMo has no Q/K norm.
 * V scaling belongs after the output projection, not in this unpacker. */
__global__ static void mimo2_split_rope(
        float *q, float *k, float *v, const float *qkv,
        const float2 *table, unsigned kv_heads, unsigned rows) {
    enum { Q_HEADS = 64, KEY = 192, VALUE = 128, ROTARY = 64 };
    const unsigned stride = Q_HEADS * KEY + kv_heads * (KEY + VALUE);
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * stride) { return; }
    const unsigned row = i / stride, col = i % stride;
    const unsigned q_end = Q_HEADS * KEY, k_end = q_end + kv_heads * KEY;
    if (col >= k_end) {
        v[(uint64_t)row * kv_heads * VALUE + col - k_end] = qkv[i];
        return;
    }
    const unsigned dim = col % KEY;
    float value = qkv[i];
    if (dim < ROTARY) {
        const unsigned pair = dim % (ROTARY / 2);
        const uint64_t first = i - dim + pair;
        const float2 cs = table[(uint64_t)row * (ROTARY / 2) + pair];
        const float a = qkv[first], b = qkv[first + ROTARY / 2];
        value = dim < ROTARY / 2 ? a * cs.x - b * cs.y : b * cs.x + a * cs.y;
    }
    if (col < q_end) {
        q[(uint64_t)row * q_end + col] = value;
    } else {
        k[(uint64_t)row * kv_heads * KEY + col - q_end] = value;
    }
}

/* One group, 256 experts, eight selected. Bias changes selection only.
 * Retain the pinned GGUF graph's 2^-14 denominator floor; route scale is 1. */
__global__ static void mimo2_router(
        int *ids, float *weights, const float *logits, const float *bias) {
    enum { EXPERTS = 256, USED = 8 };
    __shared__ float prob[EXPERTS], score[EXPERTS];
    const unsigned tid = threadIdx.x, row = blockIdx.x;
    for (unsigned e = tid; e < EXPERTS; e += blockDim.x) {
        prob[e] = 1.0f / (1.0f + expf(-logits[(uint64_t)row * EXPERTS + e]));
        score[e] = prob[e] + (bias ? bias[e] : 0.0f);
    }
    __syncthreads();
    if (tid) { return; }
    ids += (uint64_t)row * USED;
    weights += (uint64_t)row * USED;
    for (unsigned e = 0; e < EXPERTS; e++) {
        if (!isfinite(score[e])) {
            for (unsigned k = 0; k < USED; k++) { ids[k] = k; weights[k] = NAN; }
            return;
        }
    }
    float sum = 0;
    for (unsigned k = 0; k < USED; k++) {
        unsigned best = 0;
        for (unsigned e = 1; e < EXPERTS; e++) {
            if (score[e] > score[best]) { best = e; }
        }
        ids[k] = best;
        weights[k] = prob[best];
        sum += prob[best];
        score[best] = -INFINITY;
    }
    const float denominator = fmaxf(sum, 0x1p-14f);
    for (unsigned k = 0; k < USED; k++) { weights[k] /= denominator; }
}

/* Cache row: [all K heads (192 each) | all V heads (128 each)].
 * The caller retains window+batch-1 rows before attention, so a batched store
 * cannot overwrite keys needed by the earliest query in that batch. Positions
 * are consecutive live device values, so captured replay cannot bake pos0. */
__global__ static void mimo2_kv_store(
        __half *cache, const float *k, const float *v,
        unsigned kv_heads, unsigned rows, const unsigned *positions, unsigned capacity) {
    const unsigned kw = kv_heads * 192, vw = kv_heads * 128, stride = kw + vw;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * stride) { return; }
    const unsigned row = i / stride, col = i % stride;
    const uint64_t slot = (uint64_t)positions[row] % capacity;
    const float value = col < kw ? k[(uint64_t)row * kw + col]
                                : v[(uint64_t)row * vw + col - kw];
    cache[slot * stride + col] = __float2half_rn(value);
}

/* Correctness path for asymmetric GQA. One warp owns a query head and walks
 * its causal keys with online softmax. The learned sink has a zero value.
 * Positions remain device reads under capture. A tiled prefill path can be
 * checked against this kernel before replacing it for wide workloads. */
__global__ static void mimo2_attention(
        float *out, const float *q, const __half *cache, const float *sinks,
        const unsigned *positions, unsigned kv_heads, unsigned capacity,
        unsigned window) {
    enum { HEADS = 64, KEY = 192, VALUE = 128, WARP = 32 };
    const unsigned lane = threadIdx.x % WARP;
    const unsigned head = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const unsigned row = blockIdx.y;
    if (head >= HEADS) { return; }
    const unsigned pos = positions[row];
    const unsigned first = window && pos + 1 > window ? pos + 1 - window : 0;
    const unsigned kv_head = head / (HEADS / kv_heads), stride = kv_heads * (KEY + VALUE);
    float query[KEY / WARP], acc[VALUE / WARP] = {};
    for (unsigned d = 0; d < KEY / WARP; d++) {
        query[d] = q[((uint64_t)row * HEADS + head) * KEY + lane + d * WARP];
    }
    float maximum = sinks ? sinks[head] : -INFINITY;
    float denominator = sinks ? 1.0f : 0.0f;
    for (unsigned key = first; key <= pos; key++) {
        const __half *slot = cache + (uint64_t)(key % capacity) * stride;
        float dot = 0;
        for (unsigned d = 0; d < KEY / WARP; d++) {
            dot += query[d] * __half2float(slot[kv_head * KEY + lane + d * WARP]);
        }
        for (unsigned step = WARP / 2; step; step /= 2) {
            dot += __shfl_xor_sync(0xffffffff, dot, step);
        }
        const float score = dot * 0.07216878364870322f; // 1/sqrt(192)
        const float next = fmaxf(maximum, score);
        const float old_weight = expf(maximum - next), weight = expf(score - next);
        denominator = denominator * old_weight + weight;
        for (unsigned d = 0; d < VALUE / WARP; d++) {
            const float value = __half2float(slot[kv_heads * KEY + kv_head * VALUE + lane + d * WARP]);
            acc[d] = acc[d] * old_weight + weight * value;
        }
        maximum = next;
    }
    for (unsigned d = 0; d < VALUE / WARP; d++) {
        out[((uint64_t)row * HEADS + head) * VALUE + lane + d * WARP] = acc[d] / denominator;
    }
}

__device__ void m2_ld32(void *dst, const void *src);
static __device__ int m2_kernel_tag = 0;

/* One-row SWA. Eight KV heads, eight query warps each. The window is loaded
 * once per KV head; the walk reloads it for every query head.
 * DS4_MIMO2_SWA_DECODE=0 keeps that walk. */
__global__ static void mimo2_swa_decode(
        float *out, const float *q, const __half *cache, const float *sinks,
        const unsigned *positions, unsigned kv_heads, unsigned capacity,
        unsigned window, int vec) {
    enum { HEADS = 64, KEY = 192, VALUE = 128, WARP = 32, TILE = 32 };
    if (blockIdx.y == 0 && threadIdx.x == 0) { m2_kernel_tag = vec ? 2 : 1; }
    const unsigned lane = threadIdx.x % WARP;
    const unsigned warp = threadIdx.x / WARP;
    const unsigned group = HEADS / kv_heads;
    const unsigned kv_head = blockIdx.y;
    const unsigned head = kv_head * group + warp;
    const unsigned pos = positions[0];
    const unsigned first = pos + 1 > window ? pos + 1 - window : 0;
    const unsigned stride = kv_heads * (KEY + VALUE);
    float query[KEY / WARP], acc[VALUE / WARP] = {};
    for (unsigned d = 0; d < KEY / WARP; d++) {
        query[d] = q[head * KEY + lane + d * WARP];
    }
    float maximum = sinks ? sinks[head] : -INFINITY;
    float denominator = sinks ? 1.0f : 0.0f;
    __shared__ __align__(32) __half smk[TILE * KEY];
    __shared__ __align__(32) __half smv[TILE * VALUE];
    for (unsigned base = first; base <= pos; base += TILE) {
        const unsigned nkeys = pos - base + 1 < TILE ? pos - base + 1 : TILE;
        if (vec) {
            const unsigned k_groups = nkeys * (KEY / 16);
            const unsigned v_groups = nkeys * (VALUE / 16);
            for (unsigned i = threadIdx.x; i < k_groups; i += blockDim.x) {
                const unsigned local = i / (KEY / 16);
                const unsigned col = (i % (KEY / 16)) * 16;
                const unsigned slot = (base + local) % capacity;
                m2_ld32(smk + local * KEY + col,
                        cache + (uint64_t)slot * stride + kv_head * KEY + col);
            }
            for (unsigned i = threadIdx.x; i < v_groups; i += blockDim.x) {
                const unsigned local = i / (VALUE / 16);
                const unsigned col = (i % (VALUE / 16)) * 16;
                const unsigned slot = (base + local) % capacity;
                m2_ld32(smv + local * VALUE + col,
                        cache + (uint64_t)slot * stride + kv_heads * KEY + kv_head * VALUE + col);
            }
        } else {
            for (unsigned i = threadIdx.x; i < nkeys * KEY; i += blockDim.x) {
                const unsigned local = i / KEY, col = i % KEY;
                const unsigned slot = (base + local) % capacity;
                smk[i] = cache[(uint64_t)slot * stride + kv_head * KEY + col];
            }
            for (unsigned i = threadIdx.x; i < nkeys * VALUE; i += blockDim.x) {
                const unsigned local = i / VALUE, col = i % VALUE;
                const unsigned slot = (base + local) % capacity;
                smv[i] = cache[(uint64_t)slot * stride + kv_heads * KEY + kv_head * VALUE + col];
            }
        }
        __syncthreads();
        for (unsigned local = 0; local < nkeys; local++) {
            const __half *kslot = smk + local * KEY;
            const __half *vslot = smv + local * VALUE;
            float dot = 0.0f;
            for (unsigned d = 0; d < KEY / WARP; d++) {
                dot += query[d] * __half2float(kslot[lane + d * WARP]);
            }
            for (unsigned step = WARP / 2; step; step /= 2) {
                dot += __shfl_xor_sync(0xffffffff, dot, step);
            }
            const float score = dot * 0.07216878364870322f;
            const float next = fmaxf(maximum, score);
            const float old_weight = expf(maximum - next), weight = expf(score - next);
            denominator = denominator * old_weight + weight;
            for (unsigned d = 0; d < VALUE / WARP; d++) {
                acc[d] = acc[d] * old_weight + weight * __half2float(vslot[lane + d * WARP]);
            }
            maximum = next;
        }
        __syncthreads();
    }
    for (unsigned d = 0; d < VALUE / WARP; d++) {
        out[head * VALUE + lane + d * WARP] = acc[d] / denominator;
    }
}

/* Full attention only (4 KV heads, 16 query heads, no window).
 * One block owns one query row and one KV head. The 16 warps load that
 * head's K/V tile once and reuse it; the walking kernel reloads it per head.
 * Dot order matches the walking kernel, so the same halves stay bit-exact.
 * Positions stay device reads. Launch is host-side, so capture does not bake pos.
 *
 * The tile's sync is not paid back on a short KV. sm_121, 2048-row chunks:
 * slower at pos0<=8192, faster at pos0>=10240 (~0.79x the walk). A 4096-row
 * chunk, the prefill cap, is already faster at pos0>=8192. */
enum {
    M2_ATTN_TILE = 32,
    M2_FATTN_MIN_ROWS = 32,
    M2_FATTN_NARROW_POS = 10240,
    M2_FATTN_WIDE_ROWS = 4096,
    M2_FATTN_WIDE_POS = 8192
};

static int m2_use_tile(unsigned window, unsigned kv_heads, unsigned rows, unsigned pos0) {
    if (window != 0 || kv_heads != 4 || rows < (unsigned)M2_FATTN_MIN_ROWS) { return 0; }
    if (pos0 >= (unsigned)M2_FATTN_NARROW_POS) { return 1; }
    return rows >= (unsigned)M2_FATTN_WIDE_ROWS && pos0 >= (unsigned)M2_FATTN_WIDE_POS;
}

__global__ static void mimo2_attn_tile(
        float *out, const float *q, const __half *cache, const float *sinks,
        const unsigned *positions, unsigned kv_heads, unsigned capacity) {
    enum { HEADS = 64, KEY = 192, VALUE = 128, WARP = 32, GROUP = 16 };
    const unsigned lane = threadIdx.x % WARP;
    const unsigned warp = threadIdx.x / WARP;
    const unsigned row = blockIdx.x;
    const unsigned kv_head = blockIdx.y;
    const unsigned head = kv_head * GROUP + warp;
    const unsigned pos = positions[row];
    const unsigned stride = kv_heads * (KEY + VALUE);
    float query[KEY / WARP], acc[VALUE / WARP] = {};
    for (unsigned d = 0; d < KEY / WARP; d++) {
        query[d] = q[((uint64_t)row * HEADS + head) * KEY + lane + d * WARP];
    }
    float maximum = sinks ? sinks[head] : -INFINITY;
    float denominator = sinks ? 1.0f : 0.0f;
    __shared__ __half smk[M2_ATTN_TILE * KEY];
    __shared__ __half smv[M2_ATTN_TILE * VALUE];
    for (unsigned base = 0; base <= pos; base += M2_ATTN_TILE) {
        const unsigned nkeys = pos - base + 1 < M2_ATTN_TILE ? pos - base + 1 : M2_ATTN_TILE;
        const unsigned k_count = nkeys * KEY;
        const unsigned v_count = nkeys * VALUE;
        for (unsigned i = threadIdx.x; i < k_count; i += blockDim.x) {
            const unsigned local = i / KEY, col = i % KEY;
            const unsigned slot = (base + local) % capacity;
            smk[i] = cache[(uint64_t)slot * stride + kv_head * KEY + col];
        }
        for (unsigned i = threadIdx.x; i < v_count; i += blockDim.x) {
            const unsigned local = i / VALUE, col = i % VALUE;
            const unsigned slot = (base + local) % capacity;
            smv[i] = cache[(uint64_t)slot * stride + kv_heads * KEY + kv_head * VALUE + col];
        }
        __syncthreads();
        for (unsigned local = 0; local < nkeys; local++) {
            const __half *kslot = smk + local * KEY;
            const __half *vslot = smv + local * VALUE;
            float dot = 0;
            for (unsigned d = 0; d < KEY / WARP; d++) {
                dot += query[d] * __half2float(kslot[lane + d * WARP]);
            }
            for (unsigned step = WARP / 2; step; step /= 2) {
                dot += __shfl_xor_sync(0xffffffff, dot, step);
            }
            const float score = dot * 0.07216878364870322f; // 1/sqrt(192)
            const float next = fmaxf(maximum, score);
            const float old_weight = expf(maximum - next), weight = expf(score - next);
            denominator = denominator * old_weight + weight;
            for (unsigned d = 0; d < VALUE / WARP; d++) {
                const float value = __half2float(vslot[lane + d * WARP]);
                acc[d] = acc[d] * old_weight + weight * value;
            }
            maximum = next;
        }
        __syncthreads();
    }
    for (unsigned d = 0; d < VALUE / WARP; d++) {
        out[((uint64_t)row * HEADS + head) * VALUE + lane + d * WARP] = acc[d] / denominator;
    }
}

/* Same dot order as mimo2_attn_tile. sm_121 only honors an L2 eviction
 * hint on a 32-byte vector. KV is re-read by every query row, so those
 * loads stay resident; a narrower hint is rejected by ptxas. */
__device__ __forceinline__ void m2_ld32(void *dst, const void *src) {
    unsigned long long r0, r1, r2, r3;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile("ld.global.L2::evict_last.v4.b64 {%0, %1, %2, %3}, [%4];"
                 : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3) : "l"(src));
#else
    // Older virtual targets cannot encode a 256-bit load, even when their
    // PTX is later JIT-compiled on Blackwell. Keep the stored KV bits.
    asm volatile("ld.global.v2.b64 {%0, %1}, [%2];"
                 : "=l"(r0), "=l"(r1) : "l"(src));
    asm volatile("ld.global.v2.b64 {%0, %1}, [%2];"
                 : "=l"(r2), "=l"(r3) : "l"((const char *)src + 16));
#endif
    unsigned long long *out = (unsigned long long *)dst;
    out[0] = r0;
    out[1] = r1;
    out[2] = r2;
    out[3] = r3;
}

__global__ static void mimo2_attn_l2(
        float *out, const float *q, const __half *cache, const float *sinks,
        const unsigned *positions, unsigned kv_heads, unsigned capacity) {
    enum { HEADS = 64, KEY = 192, VALUE = 128, WARP = 32, GROUP = 16 };
    const unsigned lane = threadIdx.x % WARP;
    const unsigned warp = threadIdx.x / WARP;
    const unsigned row = blockIdx.x;
    const unsigned kv_head = blockIdx.y;
    const unsigned head = kv_head * GROUP + warp;
    const unsigned pos = positions[row];
    const unsigned stride = kv_heads * (KEY + VALUE);
    float query[KEY / WARP], acc[VALUE / WARP] = {};
    for (unsigned d = 0; d < KEY / WARP; d++) {
        query[d] = q[((uint64_t)row * HEADS + head) * KEY + lane + d * WARP];
    }
    float maximum = sinks ? sinks[head] : -INFINITY;
    float denominator = sinks ? 1.0f : 0.0f;
    __shared__ __align__(32) __half smk[M2_ATTN_TILE * KEY];
    __shared__ __align__(32) __half smv[M2_ATTN_TILE * VALUE];
    for (unsigned base = 0; base <= pos; base += M2_ATTN_TILE) {
        const unsigned nkeys = pos - base + 1 < M2_ATTN_TILE ? pos - base + 1 : M2_ATTN_TILE;
        const unsigned k_groups = nkeys * (KEY / 16);
        const unsigned v_groups = nkeys * (VALUE / 16);
        for (unsigned i = threadIdx.x; i < k_groups; i += blockDim.x) {
            const unsigned local = i / (KEY / 16);
            const unsigned col = (i % (KEY / 16)) * 16;
            const unsigned slot = (base + local) % capacity;
            m2_ld32(smk + local * KEY + col,
                    cache + (uint64_t)slot * stride + kv_head * KEY + col);
        }
        for (unsigned i = threadIdx.x; i < v_groups; i += blockDim.x) {
            const unsigned local = i / (VALUE / 16);
            const unsigned col = (i % (VALUE / 16)) * 16;
            const unsigned slot = (base + local) % capacity;
            m2_ld32(smv + local * VALUE + col,
                    cache + (uint64_t)slot * stride + kv_heads * KEY + kv_head * VALUE + col);
        }
        __syncthreads();
        for (unsigned local = 0; local < nkeys; local++) {
            const __half *kslot = smk + local * KEY;
            const __half *vslot = smv + local * VALUE;
            float dot = 0;
            for (unsigned d = 0; d < KEY / WARP; d++) {
                dot += query[d] * __half2float(kslot[lane + d * WARP]);
            }
            for (unsigned step = WARP / 2; step; step /= 2) {
                dot += __shfl_xor_sync(0xffffffff, dot, step);
            }
            const float score = dot * 0.07216878364870322f; // 1/sqrt(192)
            const float next = fmaxf(maximum, score);
            const float old_weight = expf(maximum - next), weight = expf(score - next);
            denominator = denominator * old_weight + weight;
            for (unsigned d = 0; d < VALUE / WARP; d++) {
                const float value = __half2float(vslot[lane + d * WARP]);
                acc[d] = acc[d] * old_weight + weight * value;
            }
            maximum = next;
        }
        __syncthreads();
    }
    for (unsigned d = 0; d < VALUE / WARP; d++) {
        out[((uint64_t)row * HEADS + head) * VALUE + lane + d * WARP] = acc[d] / denominator;
    }
}

/* Decode is one query row, so the walking kernel is 64 warps on the whole
 * GPU. 32 slices of one KV head fill the SMs. The sink stays on slice 0.
 * DS4_MIMO2_ATTN_SPLIT=0 keeps the walk. One row at 32K: 31 ms -> 1 ms. */
enum { M2_DECODE_SPLITS = 32 };

__global__ static void mimo2_attn_split(
        float *partial_max, float *partial_den, float *partial_acc,
        const float *q, const __half *cache, const float *sinks,
        const unsigned *positions, unsigned kv_heads, unsigned capacity,
        int nsplit, int vec) {
    enum { HEADS = 64, KEY = 192, VALUE = 128, WARP = 32, GROUP = 16, TILE = 32 };
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        m2_kernel_tag = vec ? 4 : 3;
    }
    const unsigned lane = threadIdx.x % WARP;
    const unsigned warp = threadIdx.x / WARP;
    const unsigned split = blockIdx.x;
    const unsigned kv_head = blockIdx.y;
    const unsigned head = kv_head * GROUP + warp;
    const unsigned pos = positions[0];
    const unsigned span = pos + 1;
    const unsigned chunk = (span + (unsigned)nsplit - 1) / (unsigned)nsplit;
    const unsigned begin = split * chunk;
    const unsigned stride = kv_heads * (KEY + VALUE);
    float query[KEY / WARP], acc[VALUE / WARP] = {};
    for (unsigned d = 0; d < KEY / WARP; d++) {
        query[d] = q[head * KEY + lane + d * WARP];
    }
    float maximum = -INFINITY;
    float denominator = 0.0f;
    if (split == 0 && sinks) {
        maximum = sinks[head];
        denominator = 1.0f;
    }
    if (begin <= pos) {
        const unsigned end = begin + chunk - 1 < pos ? begin + chunk - 1 : pos;
        __shared__ __align__(32) __half smk[TILE * KEY];
        __shared__ __align__(32) __half smv[TILE * VALUE];
        for (unsigned base = begin; base <= end; base += TILE) {
            const unsigned nkeys = end - base + 1 < TILE ? end - base + 1 : TILE;
            if (vec) {
                const unsigned k_groups = nkeys * (KEY / 16);
                const unsigned v_groups = nkeys * (VALUE / 16);
                for (unsigned i = threadIdx.x; i < k_groups; i += blockDim.x) {
                    const unsigned local = i / (KEY / 16);
                    const unsigned col = (i % (KEY / 16)) * 16;
                    const unsigned slot = (base + local) % capacity;
                    m2_ld32(smk + local * KEY + col,
                            cache + (uint64_t)slot * stride + kv_head * KEY + col);
                }
                for (unsigned i = threadIdx.x; i < v_groups; i += blockDim.x) {
                    const unsigned local = i / (VALUE / 16);
                    const unsigned col = (i % (VALUE / 16)) * 16;
                    const unsigned slot = (base + local) % capacity;
                    m2_ld32(smv + local * VALUE + col,
                            cache + (uint64_t)slot * stride + kv_heads * KEY + kv_head * VALUE + col);
                }
            } else {
                for (unsigned i = threadIdx.x; i < nkeys * KEY; i += blockDim.x) {
                    const unsigned local = i / KEY, col = i % KEY;
                    const unsigned slot = (base + local) % capacity;
                    smk[i] = cache[(uint64_t)slot * stride + kv_head * KEY + col];
                }
                for (unsigned i = threadIdx.x; i < nkeys * VALUE; i += blockDim.x) {
                    const unsigned local = i / VALUE, col = i % VALUE;
                    const unsigned slot = (base + local) % capacity;
                    smv[i] = cache[(uint64_t)slot * stride + kv_heads * KEY + kv_head * VALUE + col];
                }
            }
            __syncthreads();
            for (unsigned local = 0; local < nkeys; local++) {
                const __half *kslot = smk + local * KEY;
                const __half *vslot = smv + local * VALUE;
                float dot = 0.0f;
                for (unsigned d = 0; d < KEY / WARP; d++) {
                    dot += query[d] * __half2float(kslot[lane + d * WARP]);
                }
                for (unsigned step = WARP / 2; step; step /= 2) {
                    dot += __shfl_xor_sync(0xffffffff, dot, step);
                }
                const float score = dot * 0.07216878364870322f; // 1/sqrt(192)
                const float next = fmaxf(maximum, score);
                const float old_weight = expf(maximum - next), weight = expf(score - next);
                denominator = denominator * old_weight + weight;
                for (unsigned d = 0; d < VALUE / WARP; d++) {
                    const float value = __half2float(vslot[lane + d * WARP]);
                    acc[d] = acc[d] * old_weight + weight * value;
                }
                maximum = next;
            }
            __syncthreads();
        }
    }
    const unsigned slot = split * HEADS + head;
    if (lane == 0) {
        partial_max[slot] = maximum;
        partial_den[slot] = denominator;
    }
    for (unsigned d = 0; d < VALUE / WARP; d++) {
        partial_acc[slot * VALUE + lane + d * WARP] = acc[d];
    }
}

__global__ static void mimo2_attn_merge(
        float *out, const float *partial_max, const float *partial_den,
        const float *partial_acc, int nsplit) {
    enum { HEADS = 64, VALUE = 128, WARP = 32 };
    const unsigned head = blockIdx.x;
    const unsigned lane = threadIdx.x;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float acc[VALUE / WARP] = {};
    for (int split = 0; split < nsplit; split++) {
        const unsigned slot = (unsigned)split * HEADS + head;
        const float pden = partial_den[slot];
        if (!(pden > 0.0f)) { continue; }
        const float pmax = partial_max[slot];
        const float next = fmaxf(maximum, pmax);
        const float keep = expf(maximum - next);
        const float add = expf(pmax - next);
        denominator = denominator * keep + pden * add;
        for (unsigned d = 0; d < VALUE / WARP; d++) {
            acc[d] = acc[d] * keep + partial_acc[slot * VALUE + lane + d * WARP] * add;
        }
        maximum = next;
    }
    for (unsigned d = 0; d < VALUE / WARP; d++) {
        out[head * VALUE + lane + d * WARP] = acc[d] / denominator;
    }
}
