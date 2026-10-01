#pragma once
#include <cuda_runtime.h>
#include "iquest_primitives.cuh"

// Share Q8 decode across 64 queries. TF32 pairs retain Q8 operand bits;
// MMA and tiled softmax still change accumulation order versus the walk.
namespace iq_prefill {
enum { WARP = 32, WARPS = 4, THREADS = WARP * WARPS,
       TQ = 64, TK = 32, STEP = 16, ROW = IQ_HEAD + 8 };
enum class RowState { Inactive, Active };
enum class Output { Bf16, DiagnosticF32 };
struct A { float x[4]; };
struct B { float x[2]; };
struct C { float x[4] = {}; };

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__device__ __forceinline__ float tf32(float value) {
    uint32_t bits;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"(value));
    return __uint_as_float(bits);
}

__device__ __forceinline__ void pair(float value, float &hi, float &lo) {
    hi = tf32(value);
    lo = tf32(__fsub_rn(value, hi));
}

// PTX m16n8k8: A rows=(lane/4)+{0,8}, columns=(lane%4)+{0,4};
// B rows=(lane%4)+{0,4}, column=lane/4. C has two columns per row.
__device__ __forceinline__ void mma(C &out, const A &left, const B &right) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(out.x[0]), "+f"(out.x[1]), "+f"(out.x[2]), "+f"(out.x[3])
        : "r"(__float_as_uint(left.x[0])), "r"(__float_as_uint(left.x[1])),
          "r"(__float_as_uint(left.x[2])), "r"(__float_as_uint(left.x[3])),
          "r"(__float_as_uint(right.x[0])), "r"(__float_as_uint(right.x[1])));
}

__device__ __forceinline__ void probabilities(A &hi, A &lo,
                                              const C &scores, unsigned lane) {
    // C's adjacent columns become A's separated columns. Shuffle both
    // source elements uniformly before selecting the destination's parity.
    #pragma unroll
    for (unsigned i = 0; i < 4; i++) {
        const unsigned row = i % 2, col = lane % 4 + (i / 2) * 4;
        const unsigned source = (lane & ~3u) + col / 2;
        const float even = __shfl_sync(UINT32_MAX, scores.x[2 * row], source);
        const float odd = __shfl_sync(UINT32_MAX, scores.x[2 * row + 1], source);
        pair(col % 2 ? odd : even, hi.x[i], lo.x[i]);
    }
}

__device__ __forceinline__ void consume(C output[IQ_HEAD / 8],
        float maximum[2], float denominator[2], const A query[IQ_HEAD / 8],
        const float (*key)[ROW], const float (*value)[ROW],
        const unsigned pos[2], const RowState state[2], unsigned base,
        unsigned last, unsigned window, unsigned lane) {
    C scores[2];
    #pragma unroll
    for (unsigned nb = 0; nb < 2; nb++) {
        // Keep residual products out of the larger running high sum.
        C low;
        #pragma unroll
        for (unsigned kc = 0; kc < IQ_HEAD / 8; kc++) {
            B hi, lo;
            #pragma unroll
            for (unsigned i = 0; i < 2; i++) {
                pair(key[nb * 8 + lane / 4][kc * 8 + lane % 4 + i * 4],
                     hi.x[i], lo.x[i]);
            }
            mma(scores[nb], query[kc], hi);
            mma(low, query[kc], lo);
        }
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            scores[nb].x[i] = __fadd_rn(scores[nb].x[i], low.x[i]);
        }
    }

    const float scale = rsqrtf((float)IQ_HEAD);
    float tile_max[2] = {-INFINITY, -INFINITY};
    #pragma unroll
    for (unsigned nb = 0; nb < 2; nb++) {
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const unsigned r = i / 2;
            const unsigned token = base + nb * 8 + (lane % 4) * 2 + i % 2;
            const unsigned first = window && pos[r] + 1 > window ? pos[r] + 1 - window : 0;
            const float score = state[r] == RowState::Active && token <= last &&
                token <= pos[r] && token >= first ? __fmul_rn(scores[nb].x[i], scale) : -INFINITY;
            scores[nb].x[i] = score;
            tile_max[r] = fmaxf(tile_max[r], score);
        }
    }

    float rescale[2], tile_sum[2] = {};
    #pragma unroll
    for (unsigned r = 0; r < 2; r++) {
        tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(UINT32_MAX, tile_max[r], 1));
        tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(UINT32_MAX, tile_max[r], 2));
        const float next = fmaxf(maximum[r], tile_max[r]);
        rescale[r] = maximum[r] == -INFINITY ? 0.0f : expf(__fsub_rn(maximum[r], next));
        maximum[r] = next;
    }
    #pragma unroll
    for (unsigned nb = 0; nb < 2; nb++) {
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const unsigned r = i / 2;
            const float weight = scores[nb].x[i] == -INFINITY ? 0.0f
                : expf(__fsub_rn(scores[nb].x[i], maximum[r]));
            scores[nb].x[i] = weight;
            tile_sum[r] = __fadd_rn(tile_sum[r], weight);
        }
    }
    #pragma unroll
    for (unsigned r = 0; r < 2; r++) {
        tile_sum[r] = __fadd_rn(tile_sum[r], __shfl_xor_sync(UINT32_MAX, tile_sum[r], 1));
        tile_sum[r] = __fadd_rn(tile_sum[r], __shfl_xor_sync(UINT32_MAX, tile_sum[r], 2));
        denominator[r] = __fmaf_rn(denominator[r], rescale[r], tile_sum[r]);
    }
    // Accumulate a bounded tile in Tensor Cores, then update the long
    // running numerator once in FP32. Keep small cross terms separate.
    A phi[2], plo[2];
    #pragma unroll
    for (unsigned nb = 0; nb < 2; nb++) {
        probabilities(phi[nb], plo[nb], scores[nb], lane);
    }
    #pragma unroll
    for (unsigned cb = 0; cb < IQ_HEAD / 8; cb++) {
        C high, low;
        #pragma unroll
        for (unsigned nb = 0; nb < 2; nb++) {
            B vhi, vlo;
            #pragma unroll
            for (unsigned i = 0; i < 2; i++) {
                pair(value[nb * 8 + lane % 4 + i * 4][cb * 8 + lane / 4],
                     vhi.x[i], vlo.x[i]);
            }
            mma(high, phi[nb], vhi);
            mma(low, phi[nb], vlo);
            mma(low, plo[nb], vhi);
            mma(low, plo[nb], vlo);
        }
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const float tile = __fadd_rn(high.x[i], low.x[i]);
            output[cb].x[i] = __fmaf_rn(output[cb].x[i], rescale[i / 2], tile);
        }
    }
}
#endif

template<Output format = Output::Bf16>
__global__ static void prefill(float *out, float *lse, const float *q,
        const iquest_q8 *cache, const unsigned *positions, unsigned rows,
        unsigned capacity, unsigned window) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    const unsigned lane = threadIdx.x % WARP, warp = threadIdx.x / WARP;
    const unsigned row0 = blockIdx.x * TQ, head = blockIdx.y;
    if (row0 >= rows || head >= IQ_HEADS) { return; }
    const unsigned kh = head / (IQ_HEADS / IQ_KV_HEADS);
    const unsigned row[2] = {row0 + warp * STEP + lane / 4,
                             row0 + warp * STEP + lane / 4 + 8};
    unsigned pos[2];
    RowState state[2];
    float maximum[2] = {-INFINITY, -INFINITY}, denominator[2] = {};
    unsigned first = UINT32_MAX, last = 0;
    #pragma unroll
    for (unsigned r = 0; r < 2; r++) {
        state[r] = row[r] < rows ? RowState::Active : RowState::Inactive;
        pos[r] = row[r] < rows ? positions[row[r]] : 0;
        if (state[r] == RowState::Active) {
            const unsigned begin = window && pos[r] + 1 > window ? pos[r] + 1 - window : 0;
            first = min(first, begin);
            last = max(last, pos[r]);
        }
    }

    // Bounds include every live query, including diagnostic permutations.
    // Never round the lower bound down into already overwritten ring rows.
    #pragma unroll
    for (unsigned step = WARP / 2; step; step /= 2) {
        first = min(first, __shfl_xor_sync(UINT32_MAX, first, step));
        last = max(last, __shfl_xor_sync(UINT32_MAX, last, step));
    }
    __shared__ unsigned first_warp[WARPS], last_warp[WARPS];
    if (!lane) { first_warp[warp] = first; last_warp[warp] = last; }
    __syncthreads();
    #pragma unroll
    for (unsigned w = 0; w < WARPS; w++) {
        first = min(first, first_warp[w]);
        last = max(last, last_warp[w]);
    }

    A query[IQ_HEAD / 8];
    #pragma unroll
    for (unsigned kc = 0; kc < IQ_HEAD / 8; kc++) {
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const unsigned qr = row[i % 2], col = kc * 8 + lane % 4 + (i / 2) * 4;
            // Production queries crossed the post-RoPE BF16 boundary;
            // their values are exactly representable as TF32 operands.
            query[kc].x[i] = qr < rows ? tf32(q[((uint64_t)qr * IQ_HEADS + head) * IQ_HEAD + col]) : 0;
        }
    }
    C output[IQ_HEAD / 8];
    __shared__ __align__(16) float key[TK][ROW], value[TK][ROW];
    for (unsigned base = first; base <= last; base += TK) {
        for (unsigned i = threadIdx.x; i < TK * IQ_HEAD; i += blockDim.x) {
            const unsigned token = base + i / IQ_HEAD, col = i % IQ_HEAD;
            float k = 0, v = 0;
            if (token <= last) {
                const iquest_q8 *kv = cache + (uint64_t)(token % capacity) * IQ_Q8_ROW_BLOCKS;
                k = iq_q8_read(kv, kh * IQ_HEAD + col);
                v = iq_q8_read(kv + IQ_Q8_ROW_BLOCKS / 2, kh * IQ_HEAD + col);
            }
            key[i / IQ_HEAD][col] = k;
            value[i / IQ_HEAD][col] = v;
        }
        __syncthreads();
        consume(output, maximum, denominator, query, key, value, pos, state,
                base, last, window, lane);
        consume(output, maximum, denominator, query, key + STEP, value + STEP, pos, state,
                base + STEP, last, window, lane);
        __syncthreads();
    }

    #pragma unroll
    for (unsigned cb = 0; cb < IQ_HEAD / 8; cb++) {
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const unsigned r = i / 2;
            if (state[r] == RowState::Inactive) { continue; }
            const unsigned col = cb * 8 + (lane % 4) * 2 + i % 2;
            const float ordinary = __fmul_rn(output[cb].x[i], 1.0f / denominator[r]);
            const uint64_t at = ((uint64_t)row[r] * IQ_HEADS + head) * IQ_HEAD + col;
            // Only the numerical regression fixture instantiates raw output.
            if constexpr (format == Output::DiagnosticF32) {
                out[at] = ordinary;
            } else {
                out[at] = __bfloat162float(__float2bfloat16_rn(ordinary));
            }
        }
    }
    if (lane % 4 == 0) {
        constexpr float ln2 = 0x1.62e430p-1f;
        #pragma unroll
        for (unsigned r = 0; r < 2; r++) {
            if (state[r] == RowState::Active) {
                lse[(uint64_t)row[r] * IQ_HEADS + head] =
                    __fmaf_rn(__log2f(denominator[r]), ln2, maximum[r]);
            }
        }
    }
#else
    asm volatile("trap;");
#endif
}

__global__ static void sink(float *out, const float *lse, const float *q,
                            const float *learned_sink, unsigned rows) {
    const unsigned row = blockIdx.x, lane = threadIdx.x % IQ_WARP_WIDTH;
    const unsigned head = blockIdx.y * IQ_ATTN_HEADS_PER_BLOCK + threadIdx.x / IQ_WARP_WIDTH;
    if (row >= rows || head >= IQ_HEADS) { return; }
    const unsigned kh = head / (IQ_HEADS / IQ_KV_HEADS);
    const uint64_t base = ((uint64_t)row * IQ_HEADS + head) * IQ_HEAD + lane;
    const unsigned d = kh * IQ_HEAD + lane;
    const float dot = iq_warp_sum(
        __fmul_rn(q[base], learned_sink[d]),
        __fmul_rn(q[base + WARP], learned_sink[d + WARP]),
        __fmul_rn(q[base + 2 * WARP], learned_sink[d + 2 * WARP]),
        __fmul_rn(q[base + 3 * WARP], learned_sink[d + 3 * WARP]));
    const float delta = __fmaf_rn(dot, rsqrtf((float)IQ_HEAD), -lse[(uint64_t)row * IQ_HEADS + head]);
    const float factor = 1.0f / (1.0f + expf(delta));
    // Ordinary output is already BF16. Keep the second boundary after the
    // F32 learned-key correction, independently of the tiled MMA layout.
    #pragma unroll
    for (unsigned part = 0; part < IQ_HEAD / WARP; part++) {
        const uint64_t at = base + part * WARP;
        out[at] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(out[at], factor)));
    }
}

static bool supported() {
    cudaFuncAttributes attributes{};
    if (cudaFuncGetAttributes(&attributes, prefill<>) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return attributes.ptxVersion >= 80;
}
} // namespace iq_prefill
