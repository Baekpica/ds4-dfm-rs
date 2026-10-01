#pragma once
#include "iquest_primitives.cuh"

namespace iq_decode {
enum { HEAD_BLOCKS = IQ_HEAD / IQ_Q8_BLOCK, HEAD_BYTES = HEAD_BLOCKS * sizeof(iquest_q8),
       WORD_BYTES = sizeof(uint64_t), HEAD_WORDS = HEAD_BYTES / WORD_BYTES,
       KEY_WORDS = 2 * HEAD_WORDS, THREADS = 128 };
static_assert(HEAD_BYTES % WORD_BYTES == 0, "Q8 head copy requires aligned whole words");

enum class Transfer { Synchronous, Asynchronous };

__device__ __forceinline__ static void copy_async(uint64_t *out, const uint64_t *in) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    const uint32_t shared = static_cast<uint32_t>(__cvta_generic_to_shared(out));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
                 :: "r"(shared), "l"(in) : "memory");
#else
    *out = *in;
#endif
}

// Cooperatively stage compressed heads, then walk keys in original order.
// Only warp0 computes; every warp participates in the tile lifetime barriers.
template<unsigned Tile, Transfer transfer = Transfer::Synchronous>
__global__ static void cached(float *out, const float *query, const iquest_q8 *cache,
        const float *sink, const unsigned *positions, unsigned capacity, unsigned window) {
    const unsigned lane = threadIdx.x % IQ_WARP_WIDTH;
    const unsigned head = blockIdx.y, kh = head / (IQ_HEADS / IQ_KV_HEADS);
    const uint64_t base = ((uint64_t)blockIdx.x * IQ_HEADS + head) * IQ_HEAD + lane;
    const unsigned d = kh * IQ_HEAD + lane;
    const unsigned position = positions[blockIdx.x];
    const unsigned start = window && position + 1 > window ? position + 1 - window : 0;
    const float scale = rsqrtf((float)IQ_HEAD);
    float q[4] = {}, value[4] = {}, sink_dot = 0;
    if (threadIdx.x < IQ_WARP_WIDTH) {
        #pragma unroll
        for (unsigned part = 0; part < 4; part++) { q[part] = query[base + part * IQ_WARP_WIDTH]; }
        sink_dot = iq_warp_sum(__fmul_rn(q[0], sink[d]), __fmul_rn(q[1], sink[d + IQ_WARP_WIDTH]),
            __fmul_rn(q[2], sink[d + 2 * IQ_WARP_WIDTH]), __fmul_rn(q[3], sink[d + 3 * IQ_WARP_WIDTH]));
    }
    float maximum = -INFINITY, total = 0;
    __shared__ __align__(16) uint64_t words[Tile * KEY_WORDS];
    for (unsigned first = start; first <= position; first += Tile) {
        const unsigned count = min(Tile, position + 1 - first);
        for (unsigned i = threadIdx.x; i < count * KEY_WORDS; i += blockDim.x) {
            const unsigned token = first + i / KEY_WORDS;
            const unsigned word = i % KEY_WORDS;
            const iquest_q8 *row = cache + (uint64_t)(token % capacity) * IQ_Q8_ROW_BLOCKS;
            const iquest_q8 *source = row + kh * HEAD_BLOCKS + (word / HEAD_WORDS) * IQ_Q8_ROW_BLOCKS / 2;
            const uint64_t *input = reinterpret_cast<const uint64_t *>(source) + word % HEAD_WORDS;
            if constexpr (transfer == Transfer::Asynchronous) {
                copy_async(words + i, input);
            } else {
                words[i] = *input;
            }
        }
        if constexpr (transfer == Transfer::Asynchronous) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
            // Finish each producer's copies before the CTA shares the tile.
            asm volatile("cp.async.wait_all;" ::: "memory");
#endif
        }
        __syncthreads();
        if (threadIdx.x < IQ_WARP_WIDTH) {
            for (unsigned at = 0; at < count; at++) {
                const iquest_q8 *key = reinterpret_cast<const iquest_q8 *>(words + at * KEY_WORDS);
                const float dot = iq_warp_sum(iq_q8_product(q[0], key, lane),
                    iq_q8_product(q[1], key, lane + IQ_WARP_WIDTH),
                    iq_q8_product(q[2], key, lane + 2 * IQ_WARP_WIDTH),
                    iq_q8_product(q[3], key, lane + 3 * IQ_WARP_WIDTH));
                const float score = __fmul_rn(dot, scale);
                const float next = fmaxf(maximum, score);
                const float prior = expf(__fadd_rn(maximum, -next));
                const float weight = expf(__fadd_rn(score, -next));
                #pragma unroll
                for (unsigned part = 0; part < 4; part++) {
                    value[part] = __fmaf_rn(prior, value[part],
                        iq_q8_product(weight, key + HEAD_BLOCKS, lane + part * IQ_WARP_WIDTH));
                }
                total = __fmaf_rn(prior, total, weight);
                maximum = next;
            }
        }
        // No loader can overwrite a row while warp0 still consumes it.
        __syncthreads();
    }
    if (threadIdx.x >= IQ_WARP_WIDTH) { return; }
    constexpr float ln2 = 0x1.62e430p-1f;
    const float lse = __fmaf_rn(__log2f(total), ln2, maximum);
    const float factor = 1.0f / (1.0f + expf(__fmaf_rn(sink_dot, scale, -lse)));
    const float inverse = 1.0f / total;
    #pragma unroll
    for (unsigned part = 0; part < 4; part++) {
        out[base + part * IQ_WARP_WIDTH] = iq_attn_output(value[part], inverse, factor);
    }
}
static bool async_supported() {
    cudaFuncAttributes attributes{};
    if (cudaFuncGetAttributes(&attributes, cached<128, Transfer::Asynchronous>) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return attributes.ptxVersion >= 80;
}
} // namespace iq_decode
