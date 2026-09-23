#pragma once
#include <cuda_fp16.h>
#include <cstdint>

/* MiMo's QK=192 / V=128 cannot use the symmetric GQA primitive. Each
 * warp owns 16 query rows. Tensor cores share a KV tile across 64 rows;
 * the scalar path revisits it once per row. Q and probabilities use two
 * FP16 components, retaining their residual instead of rounding to FP16.
 * KV remains the exact stored FP16. Tile softmax changes summation order. */
namespace mimo2_hmma {
enum { QK = 192, V = 128, HEADS = 64, WQ = 16, WARPS = 4,
       TQ = WQ * WARPS, TK = 32, KSTRIDE = QK + 8, VSTRIDE = V + 8 };
struct A { half2 x[4]; };
struct B { half2 x[2]; };
struct C { float x[4] = {}; };
enum Copy { Scalar, Async };

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
// KV rows and their shared strides are 16-byte aligned. Copying eight
// halves at once avoids per-half address/modulo work and preserves the bits.
__device__ __forceinline__ void cp16(void *dst, const void *src, unsigned bytes) {
    const unsigned address = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
                 "r"(address), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mma(C &c, const A &a, const B &b) {
    const unsigned *ar = reinterpret_cast<const unsigned *>(a.x);
    const unsigned *br = reinterpret_cast<const unsigned *>(b.x);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c.x[0]), "+f"(c.x[1]), "+f"(c.x[2]), "+f"(c.x[3])
                 : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]),
                   "r"(br[0]), "r"(br[1]));
}

__device__ __forceinline__ void pair(float x, float y, half2 &hi, half2 &lo) {
    hi = __floats2half2_rn(x, y);
    const float2 rounded = __half22float2(hi);
    lo = __floats2half2_rn(x - rounded.x, y - rounded.y);
}

template<unsigned Window>
__device__ __forceinline__ void consume(
        C output[V / 8], float maximum[2], float denominator[2],
        const A qhi[QK / 16], const A qlo[QK / 16],
        const __half (*k)[KSTRIDE], const __half (*v)[VSTRIDE],
        const unsigned pos[2], const bool alive[2], unsigned base, unsigned lane) {
    C scores[2];
#pragma unroll
    for (int nb = 0; nb < 2; nb++) {
#pragma unroll
        for (int kc = 0; kc < QK / 16; kc++) {
            B keys;
#pragma unroll
            for (int l = 0; l < 2; l++) {
                keys.x[l] = *reinterpret_cast<const half2 *>(
                    &k[nb * 8 + lane / 4][kc * 16 + 2 * (l * 4 + lane % 4)]);
            }
            mma(scores[nb], qhi[kc], keys);
            mma(scores[nb], qlo[kc], keys);
        }
    }
    float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nb = 0; nb < 2; nb++) {
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const int r = l / 2;
            const unsigned key = base + nb * 8 + (lane % 4) * 2 + l % 2;
            const float score = alive[r] && key <= pos[r] && (!Window || pos[r] - key < Window)
                ? scores[nb].x[l] * 0.07216878364870322f : -INFINITY;
            scores[nb].x[l] = score;
            tile_max[r] = fmaxf(tile_max[r], score);
        }
    }
    float rescale[2], tile_sum[2] = {};
#pragma unroll
    for (int r = 0; r < 2; r++) {
        tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffff, tile_max[r], 1));
        tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffff, tile_max[r], 2));
        const float next = fmaxf(maximum[r], tile_max[r]);
        rescale[r] = maximum[r] == -INFINITY ? 0.0f : expf(maximum[r] - next);
        maximum[r] = next;
    }
#pragma unroll
    for (int nb = 0; nb < 2; nb++) {
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const int r = l / 2;
            const float weight = scores[nb].x[l] == -INFINITY ? 0.0f
                : expf(scores[nb].x[l] - maximum[r]);
            scores[nb].x[l] = weight;
            tile_sum[r] += weight;
        }
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        tile_sum[r] += __shfl_xor_sync(0xffffffff, tile_sum[r], 1);
        tile_sum[r] += __shfl_xor_sync(0xffffffff, tile_sum[r], 2);
        denominator[r] = denominator[r] * rescale[r] + tile_sum[r];
    }
    A phi, plo;
#pragma unroll
    for (int l = 0; l < 4; l++) {
        pair(scores[l / 2].x[(l % 2) * 2], scores[l / 2].x[(l % 2) * 2 + 1],
             phi.x[l], plo.x[l]);
    }
#pragma unroll
    for (int cb = 0; cb < V / 8; cb++) {
#pragma unroll
        for (int l = 0; l < 4; l++) { output[cb].x[l] *= rescale[l / 2]; }
        B values;
#pragma unroll
        for (int l = 0; l < 2; l++) {
            const int key = 2 * (l * 4 + lane % 4);
            values.x[l] = __halves2half2(v[key][cb * 8 + lane / 4],
                                       v[key + 1][cb * 8 + lane / 4]);
        }
        mma(output[cb], phi, values);
        mma(output[cb], plo, values);
    }
}
#endif

template<Copy CopyKind = Scalar, unsigned Window = 0>
__global__ static void prefill(
        float *out, const float *q, const __half *cache, const float *sinks,
        const unsigned *positions, unsigned rows, unsigned kv_heads, unsigned capacity) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned row0 = blockIdx.x * TQ, head = blockIdx.y;
    const unsigned kv_head = head / (HEADS / kv_heads), stride = kv_heads * (QK + V);
    const unsigned row[2] = {row0 + warp * WQ + lane / 4,
                             row0 + warp * WQ + lane / 4 + 8};
    bool alive[2];
    unsigned pos[2];
    float maximum[2], denominator[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        alive[r] = row[r] < rows;
        pos[r] = alive[r] ? positions[row[r]] : 0;
        maximum[r] = sinks ? sinks[head] : -INFINITY;
        denominator[r] = sinks ? 1.0f : 0.0f;
    }
    A qhi[QK / 16], qlo[QK / 16];
#pragma unroll
    for (int kc = 0; kc < QK / 16; kc++) {
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const unsigned qr = row[l % 2];
            const unsigned col = kc * 16 + 2 * ((l / 2) * 4 + lane % 4);
            const float2 value = qr < rows ? *reinterpret_cast<const float2 *>(
                q + ((uint64_t)qr * HEADS + head) * QK + col) : make_float2(0, 0);
            pair(value.x, value.y, qhi[kc].x[l], qlo[kc].x[l]);
        }
    }
    C output[V / 8];
    __shared__ __align__(16) __half sk[TK][KSTRIDE];
    __shared__ __align__(16) __half sv[TK][VSTRIDE];
    // Rows are contiguous; bounds and query positions remain device reads.
    const unsigned last_row = min(row0 + TQ, rows) - 1;
    const unsigned last = positions[last_row];
    // SWA begins at the first live query's retained window. Rounding down
    // keeps the KV tile aligned; consume masks each query's lower bound.
    const unsigned first = positions[row0];
    const unsigned begin = Window && first >= Window ? (first - Window + 1) / TK * TK : 0;
    for (unsigned base = begin; base <= last; base += TK) {
        if constexpr (CopyKind == Async) {
            for (unsigned i = threadIdx.x; i < TK * (QK / 8); i += blockDim.x) {
                const unsigned key = base + i / (QK / 8), col = (i % (QK / 8)) * 8;
                cp16(&sk[i / (QK / 8)][col],
                     cache + (uint64_t)(key % capacity) * stride + kv_head * QK + col,
                     key <= last ? 16 : 0);
            }
            for (unsigned i = threadIdx.x; i < TK * (V / 8); i += blockDim.x) {
                const unsigned key = base + i / (V / 8), col = (i % (V / 8)) * 8;
                cp16(&sv[i / (V / 8)][col],
                     cache + (uint64_t)(key % capacity) * stride + kv_heads * QK + kv_head * V + col,
                     key <= last ? 16 : 0);
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        } else {
            for (unsigned i = threadIdx.x; i < TK * QK; i += blockDim.x) {
                const unsigned key = base + i / QK, col = i % QK;
                sk[i / QK][col] = key <= last
                    ? cache[(uint64_t)(key % capacity) * stride + kv_head * QK + col]
                    : __float2half(0);
            }
            for (unsigned i = threadIdx.x; i < TK * V; i += blockDim.x) {
                const unsigned key = base + i / V, col = i % V;
                sv[i / V][col] = key <= last
                    ? cache[(uint64_t)(key % capacity) * stride + kv_heads * QK + kv_head * V + col]
                    : __float2half(0);
            }
        }
        __syncthreads();
        consume<Window>(output, maximum, denominator, qhi, qlo, sk, sv, pos, alive, base, lane);
        consume<Window>(output, maximum, denominator, qhi, qlo, sk + 16, sv + 16, pos, alive, base + 16, lane);
        __syncthreads();
    }
#pragma unroll
    for (int cb = 0; cb < V / 8; cb++) {
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const int r = l / 2;
            if (alive[r]) {
                out[((uint64_t)row[r] * HEADS + head) * V + cb * 8 + (lane % 4) * 2 + l % 2]
                    = output[cb].x[l] / denominator[r];
            }
        }
    }
#else
    // The host rejects this compiled target. A direct unsupported launch
    // must fail rather than return an untouched output buffer as success.
    asm volatile("trap;");
#endif
}

// Query during backend initialization, before any CUDA graph capture.
// Runtime device capability alone misses old PTX JIT-compiled on a new GPU.
static bool supported() {
    cudaFuncAttributes attributes{};
    if (cudaFuncGetAttributes(&attributes, prefill<Scalar>) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return attributes.ptxVersion >= 80;
}
} // namespace mimo2_hmma
