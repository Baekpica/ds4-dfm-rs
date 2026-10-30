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
// Fp8: the VQ release's full rows (cuda/mimo2_primitives.cuh), widened to
// f16 tiles with their head scales as they are staged.
enum Copy { Scalar, Async, Fp8 };

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
        float *out, const float *q, const void *cache_raw, const float *sinks,
        const unsigned *positions, unsigned rows, unsigned kv_heads, unsigned capacity) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    const __half *cache = static_cast<const __half *>(cache_raw);
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
        } else if constexpr (CopyKind == Fp8) {
            const uint8_t *rows8 = static_cast<const uint8_t *>(cache_raw);
            for (unsigned i = threadIdx.x; i < TK * QK; i += blockDim.x) {
                const unsigned key = base + i / QK, col = i % QK;
                const uint8_t *r = rows8 + (uint64_t)(key % capacity) * M2_FP8_ROW;
                sk[i / QK][col] = key <= last
                    ? __float2half_rn(m2_e4m3(r[kv_head * QK + col]) * m2_fp8_scale(r, 0, kv_head))
                    : __float2half(0);
            }
            for (unsigned i = threadIdx.x; i < TK * V; i += blockDim.x) {
                const unsigned key = base + i / V, col = i % V;
                const uint8_t *r = rows8 + (uint64_t)(key % capacity) * M2_FP8_ROW;
                sv[i / V][col] = key <= last
                    ? __float2half_rn(m2_e4m3(r[M2_FP8_KEYS + kv_head * V + col]) * m2_fp8_scale(r, 1, kv_head))
                    : __float2half(0);
            }
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

/* Full-attention FP8 rows (VQ release) at prefill widths. Eight warps hold
 * two query heads of one KV head, 64 rows each, so a 64-key tile is widened
 * once for 128 (row, head) pairs instead of 64. Each thread loads its share
 * of the next tile's codes into registers before this tile's MMAs and
 * widens them after, one barrier per tile. Q and P are single f16: the keys
 * carry four significant bits, so a second f16 component bought nothing.
 * (The f16 SWA rows keep mimo2_hmma::prefill: their 2,048-4,096-row calls
 * are bound by F32 Q and output traffic, and one f16 Q lost their accuracy.)
 * DS4_MIMO2VQ_GA8=0 restores mimo2_hmma::prefill<Fp8>. */
namespace mimo2_ga8 {
enum { QK = 192, V = 128, GROUP = 16, WARPS = 8, ROWS = 64, TK = 64,
       KS = QK + 8, VS = V + 8, THREADS = WARPS * 32,
       KPIECES = TK * QK / 16 / THREADS, VPIECES = TK * V / 16 / THREADS };
constexpr unsigned SMEM = 2u * TK * (KS + VS) * sizeof(__half);
// 192^-1/2 folded into exp2.
constexpr float SCALE = 0.07216878364870322f * 1.4426950408889634f;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
struct Raw { uint4 k[KPIECES], v[VPIECES]; __half ks[KPIECES], vs[VPIECES]; };

__device__ __forceinline__ void ldsm4(unsigned r[4], const void *p, bool trans) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    if (trans) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
    }
}

__device__ __forceinline__ void mma(float c[4], const unsigned a[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned pack(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const unsigned *>(&h);
}

// Full-layer rows never wrap; the modulo stays for a ring that would.
__device__ __forceinline__ const uint8_t *slot(const uint8_t *cache, unsigned key, unsigned capacity) {
    return cache + (uint64_t)(key < capacity ? key : key % capacity) * M2_FP8_ROW;
}

// Keys past `last` load as zero: their cache rows may hold NaN codes.
__device__ __forceinline__ void fetch(Raw &raw, const uint8_t *cache, unsigned base,
                                      unsigned last, unsigned capacity, unsigned kv) {
#pragma unroll
    for (int i = 0; i < KPIECES; i++) {
        const unsigned p = threadIdx.x + i * THREADS, key = base + p / (QK / 16);
        raw.k[i] = make_uint4(0, 0, 0, 0);
        raw.ks[i] = __ushort_as_half(0);
        if (key > last) { continue; }
        const uint8_t *row = slot(cache, key, capacity);
        raw.k[i] = __ldcg(reinterpret_cast<const uint4 *>(row + kv * QK) + p % (QK / 16));
        raw.ks[i] = reinterpret_cast<const __half *>(row + M2_FP8_SCALES)[kv];
    }
#pragma unroll
    for (int i = 0; i < VPIECES; i++) {
        const unsigned p = threadIdx.x + i * THREADS, key = base + p / (V / 16);
        raw.v[i] = make_uint4(0, 0, 0, 0);
        raw.vs[i] = __ushort_as_half(0);
        if (key > last) { continue; }
        const uint8_t *row = slot(cache, key, capacity);
        raw.v[i] = __ldcg(reinterpret_cast<const uint4 *>(row + M2_FP8_KEYS + kv * V) + p % (V / 16));
        raw.vs[i] = reinterpret_cast<const __half *>(row + M2_FP8_SCALES)[M2_FP8_HEADS + kv];
    }
}

// Sixteen codes times their head scale: one rounding of the exact product,
// the value f16(e4m3 * scale) the scalar staging stores.
__device__ __forceinline__ void widen16(__half *dst, uint4 codes, __half scale) {
    const __half2 s = __half2half2(scale);
    const unsigned w[4] = {codes.x, codes.y, codes.z, codes.w};
    unsigned h[8];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const __half2 lo = __hmul2(__half2(__nv_cvt_fp8x2_to_halfraw2(
            (__nv_fp8x2_storage_t)(w[i] & 0xffffu), __NV_E4M3)), s);
        const __half2 hi = __hmul2(__half2(__nv_cvt_fp8x2_to_halfraw2(
            (__nv_fp8x2_storage_t)(w[i] >> 16), __NV_E4M3)), s);
        h[2 * i] = *reinterpret_cast<const unsigned *>(&lo);
        h[2 * i + 1] = *reinterpret_cast<const unsigned *>(&hi);
    }
    reinterpret_cast<uint4 *>(dst)[0] = make_uint4(h[0], h[1], h[2], h[3]);
    reinterpret_cast<uint4 *>(dst)[1] = make_uint4(h[4], h[5], h[6], h[7]);
}

__device__ __forceinline__ void widen(__half (*k)[KS], __half (*v)[VS], const Raw &raw) {
#pragma unroll
    for (int i = 0; i < KPIECES; i++) {
        const unsigned p = threadIdx.x + i * THREADS;
        widen16(&k[p / (QK / 16)][p % (QK / 16) * 16], raw.k[i], raw.ks[i]);
    }
#pragma unroll
    for (int i = 0; i < VPIECES; i++) {
        const unsigned p = threadIdx.x + i * THREADS;
        widen16(&v[p / (V / 16)][p % (V / 16) * 16], raw.v[i], raw.vs[i]);
    }
}

// One 64-key tile for this warp's 16 rows. Scores and probabilities stay
// in registers; `mask` is set on tiles that reach past a row's position.
__device__ __forceinline__ void consume(
        float o[V / 8][4], float m[2], float l[2], const unsigned qa[QK / 16][4],
        const __half (*k)[KS], const __half (*v)[VS], unsigned base,
        const unsigned pos[2], bool mask, unsigned lane) {
    const unsigned mtx = lane / 8, r = lane % 8;
    float s[TK / 8][4] = {};
#pragma unroll
    for (int kc = 0; kc < QK / 16; kc++) {
#pragma unroll
        for (int nb = 0; nb < TK / 8; nb += 2) {
            unsigned b[4];
            ldsm4(b, &k[(nb + mtx / 2) * 8 + r][kc * 16 + (mtx % 2) * 8], false);
            mma(s[nb], qa[kc], b[0], b[1]);
            mma(s[nb + 1], qa[kc], b[2], b[3]);
        }
    }

    float top[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nb = 0; nb < TK / 8; nb++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const unsigned key = base + nb * 8 + (lane % 4) * 2 + e % 2;
            const float x = mask && key > pos[e / 2] ? -INFINITY : s[nb][e] * SCALE;
            s[nb][e] = x;
            top[e / 2] = fmaxf(top[e / 2], x);
        }
    }
    float alpha[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        top[i] = fmaxf(top[i], __shfl_xor_sync(0xffffffff, top[i], 1));
        top[i] = fmaxf(top[i], __shfl_xor_sync(0xffffffff, top[i], 2));
        const float next = fmaxf(m[i], top[i]);
        alpha[i] = next == -INFINITY ? 1.0f : exp2f(m[i] - next);
        m[i] = next;
        l[i] *= alpha[i];
    }
#pragma unroll
    for (int nb = 0; nb < TK / 8; nb++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const float p = m[e / 2] == -INFINITY ? 0.0f : exp2f(s[nb][e] - m[e / 2]);
            s[nb][e] = p;
            l[e / 2] += p;
        }
    }
#pragma unroll
    for (int n = 0; n < V / 8; n++) {
#pragma unroll
        for (int e = 0; e < 4; e++) { o[n][e] *= alpha[e / 2]; }
    }

#pragma unroll
    for (int j = 0; j < TK / 16; j++) {
        const unsigned pa[4] = {pack(s[2 * j][0], s[2 * j][1]), pack(s[2 * j][2], s[2 * j][3]),
                                pack(s[2 * j + 1][0], s[2 * j + 1][1]),
                                pack(s[2 * j + 1][2], s[2 * j + 1][3])};
#pragma unroll
        for (int n = 0; n < V / 8; n += 2) {
            unsigned b[4];
            ldsm4(b, &v[16 * j + (mtx % 2) * 8 + r][(n + mtx / 2) * 8], true);
            mma(o[n], pa, b[0], b[1]);
            mma(o[n + 1], pa, b[2], b[3]);
        }
    }
}
#endif

// Grid (row tiles, 32 head pairs); rows are contiguous positions.
__global__ __launch_bounds__(THREADS, 1) static void prefill(
        float *out, const float *q, const uint8_t *cache, const unsigned *positions,
        unsigned rows, unsigned capacity) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
    extern __shared__ __align__(16) unsigned char smem[];
    __half (*k)[KS] = reinterpret_cast<__half (*)[KS]>(smem);
    __half (*v)[VS] = reinterpret_cast<__half (*)[VS]>(smem + 2u * TK * KS * sizeof(__half));
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned row0 = blockIdx.x * ROWS, head = blockIdx.y * 2 + warp / 4, kv = head / GROUP;
    const unsigned first = row0 + (warp % 4) * 16 + lane / 4;
    const unsigned row[2] = {first, first + 8};
    bool alive[2];
    unsigned pos[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        alive[i] = row[i] < rows;
        pos[i] = alive[i] ? positions[row[i]] : 0;
    }
    // Tiles below every row of this warp skip the mask.
    const unsigned low = __reduce_min_sync(0xffffffff, min(pos[0], pos[1]));
    unsigned qa[QK / 16][4];
#pragma unroll
    for (int kc = 0; kc < QK / 16; kc++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const unsigned col = kc * 16 + (e / 2) * 8 + (lane % 4) * 2;
            const float2 x = alive[e % 2] ? *reinterpret_cast<const float2 *>(
                q + ((uint64_t)row[e % 2] * 64 + head) * QK + col) : make_float2(0, 0);
            qa[kc][e] = pack(x.x, x.y);
        }
    }

    float o[V / 8][4] = {}, m[2] = {-INFINITY, -INFINITY}, l[2] = {};
    const unsigned last = positions[min(row0 + ROWS, rows) - 1];
    Raw raw;
    fetch(raw, cache, 0, last, capacity, kv);
    widen(k, v, raw);
    __syncthreads();
    unsigned t = 0;
    for (unsigned base = 0; base <= last; base += TK, t ^= 1) {
        const bool more = base + TK <= last;
        if (more) { fetch(raw, cache, base + TK, last, capacity, kv); }
        consume(o, m, l, qa, k + t * TK, v + t * TK, base, pos, base + TK - 1 > low, lane);
        if (more) { widen(k + (t ^ 1) * TK, v + (t ^ 1) * TK, raw); }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 2; i++) {
        l[i] += __shfl_xor_sync(0xffffffff, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffff, l[i], 2);
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        if (!alive[i]) { continue; }
        float *dst = out + ((uint64_t)row[i] * 64 + head) * V + (lane % 4) * 2;
#pragma unroll
        for (int n = 0; n < V / 8; n++) {
            *reinterpret_cast<float2 *>(dst + n * 8) =
                make_float2(o[n][2 * i] / l[i], o[n][2 * i + 1] / l[i]);
        }
    }
#else
    asm volatile("trap;");
#endif
}

/* Decode and verify widths (1-8 rows). One warp per (key slice, KV head,
 * row) feeds its slice's codes into MMA fragments: the 16 query heads of the
 * KV head are the m16 rows, and the k order is chosen so each lane reads
 * contiguous codes (K: 48 bytes of one key for dims 48t..48t+47, V: 16
 * bytes of a key for output dims 16g..16g+15). The four KV heads of a slice
 * share a block, which copies whole 32-row steps into a two-stage ring with
 * contiguous cp.async (per-lane 16-byte segments capped a read-only probe at
 * 150 GB/s, whole rows 220). K scales multiply the scores; V codes widen to
 * f16(code * scale), the staging's rounding. A slice spans at least MIN_KEYS
 * keys of the live position (graph replays read it on the device); idle
 * slices write an empty partial; `merge` combines the partials.
 * DS4_MIMO2VQ_GA8=0 restores mimo2_attn_split<true>. */
enum { SLICES = 128, STEP = 32, MIN_KEYS = 128, RING = 2, DECODE_WARPS = 4,
       STAGE = STEP * M2_FP8_ROW, STAGE_PIECES = STAGE / 16 };
constexpr unsigned DECODE_SMEM = RING * STAGE;

// Two E4M3 codes (low byte first) as exact f16, or times their scales.
__device__ __forceinline__ unsigned codes2(unsigned codes) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)codes, __NV_E4M3);
    return (unsigned)h.x | (unsigned)h.y << 16;
}

__device__ __forceinline__ unsigned widen2(unsigned codes, __half2 scale) {
    const __half2 h = __hmul2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)codes, __NV_E4M3)), scale);
    return *reinterpret_cast<const unsigned *>(&h);
}

// The key of this lane's score column / value k index `i` (0..7) in a step:
// block i / 2 of 8 keys, column 2t + i % 2.
__device__ __forceinline__ unsigned column(unsigned key0, unsigned i, unsigned t) {
    return key0 + (i / 2) * 8 + 2 * t + i % 2;
}

/* Rows key0..key0 + STEP - 1 into a stage, the block's threads striding
 * 16-byte pieces; rows at or past `end` fill with zeros. Full-layer rows
 * never wrap, so a step's rows are contiguous. */
__device__ __forceinline__ void issue(unsigned char *stage, const uint8_t *cache, unsigned key0, unsigned end) {
    const unsigned valid = min(end - key0, (unsigned)STEP) * M2_FP8_ROW;
    const uint8_t *src = cache + (uint64_t)key0 * M2_FP8_ROW;
    for (unsigned p = threadIdx.x; p < STAGE_PIECES; p += 32 * DECODE_WARPS) {
        mimo2_hmma::cp16(stage + 16 * p, src + (16 * p < valid ? 16 * p : 0), 16 * p < valid ? 16 : 0);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__global__ __launch_bounds__(32 * DECODE_WARPS, 1) static void decode(
        float *pmax, float *pden, float *pacc, const float *q, const uint8_t *cache,
        const unsigned *positions, unsigned capacity, unsigned slices) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
    enum { NB = STEP / 8 };
    extern __shared__ __align__(16) unsigned char ring[];
    const unsigned lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const unsigned slice = blockIdx.x, kv = threadIdx.x / 32, row = blockIdx.y;
    // Rows are read unwrapped: a position past the cache reads no further.
    const unsigned span = min(positions[row] + 1, capacity);
    const unsigned live = min(slices, (span + MIN_KEYS - 1) / MIN_KEYS);
    const unsigned chunk = (span + live - 1) / live;
    const unsigned begin = min(slice * chunk, span), end = min(begin + chunk, span);
    const uint64_t at = ((uint64_t)row * slices + slice) * 64 + kv * GROUP;
    if (begin >= end) {
        if (t == 0) {
            pmax[at + g] = pmax[at + g + 8] = -INFINITY;
            pden[at + g] = pden[at + g + 8] = 0.0f;
        }
        return;
    }
    const unsigned steps = (end - begin + STEP - 1) / STEP;
    issue(ring, cache, begin, end);
    float o[V / 8][4] = {}, m[2] = {-INFINITY, -INFINITY}, l[2] = {};
    unsigned qa[QK / 16][4];
    const float *qh = q + ((uint64_t)row * 64 + kv * GROUP) * QK + 48 * t;
#pragma unroll
    for (int c = 0; c < QK / 16; c++) {
        const float4 x = *reinterpret_cast<const float4 *>(qh + g * QK + 4 * c);
        const float4 y = *reinterpret_cast<const float4 *>(qh + (g + 8) * QK + 4 * c);
        qa[c][0] = pack(x.x, x.y);
        qa[c][1] = pack(y.x, y.y);
        qa[c][2] = pack(x.z, x.w);
        qa[c][3] = pack(y.z, y.w);
    }
    for (unsigned i = 0; i < steps; i++) {
        // The other stage was released by last step's closing barrier.
        if (i + 1 < steps) {
            issue(ring + ((i + 1) % RING) * STAGE, cache, begin + (i + 1) * STEP, end);
            asm volatile("cp.async.wait_group 1;" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        __syncthreads();
        const unsigned key0 = begin + i * STEP;
        const unsigned char *stage = ring + (i % RING) * STAGE;
        // Scores: lane group g holds key key0 + 8 nb + g.
        float s[NB][4] = {};
#pragma unroll
        for (int p = 0; p < 3; p++) {
#pragma unroll
            for (int nb = 0; nb < NB; nb++) {
                const uint4 w = *reinterpret_cast<const uint4 *>(
                    stage + (nb * 8 + g) * M2_FP8_ROW + kv * QK + 48 * t + 16 * p);
                const unsigned ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    mma(s[nb], qa[4 * p + c], codes2(ws[c] & 0xffffu), codes2(ws[c] >> 16));
                }
            }
        }
        // Scales of this lane's eight keys (rows past the end are zero).
        __half ks[STEP / 4], vs[STEP / 4];
#pragma unroll
        for (int k = 0; k < STEP / 4; k++) {
            const __half *scales = reinterpret_cast<const __half *>(
                stage + (column(key0, k, t) - key0) * M2_FP8_ROW + M2_FP8_SCALES);
            ks[k] = scales[kv];
            vs[k] = scales[M2_FP8_HEADS + kv];
        }
        // Column scales and the slice end; this lane's columns are 2t, 2t + 1.
        float top[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int nb = 0; nb < NB; nb++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const bool in = column(key0, nb * 2 + e, t) < end;
                const float scale = __half2float(ks[nb * 2 + e]) * SCALE;
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    const float x = in ? s[nb][2 * r + e] * scale : -INFINITY;
                    s[nb][2 * r + e] = x;
                    top[r] = fmaxf(top[r], x);
                }
            }
        }
        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            top[r] = fmaxf(top[r], __shfl_xor_sync(0xffffffff, top[r], 1));
            top[r] = fmaxf(top[r], __shfl_xor_sync(0xffffffff, top[r], 2));
            const float next = fmaxf(m[r], top[r]);
            alpha[r] = next == -INFINITY ? 1.0f : exp2f(m[r] - next);
            m[r] = next;
            l[r] *= alpha[r];
        }
#pragma unroll
        for (int nb = 0; nb < NB; nb++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float p = m[e / 2] == -INFINITY ? 0.0f : exp2f(s[nb][e] - m[e / 2]);
                s[nb][e] = p;
                l[e / 2] += p;
            }
        }
#pragma unroll
        for (int n = 0; n < V / 8; n++) {
#pragma unroll
            for (int e = 0; e < 4; e++) { o[n][e] *= alpha[e / 2]; }
        }
        // Values, 16 keys per k chunk: keys of k 2t, 2t + 1 and 8 + 2t, 9 + 2t.
#pragma unroll
        for (int kc = 0; kc < STEP / 16; kc++) {
            const unsigned pa[4] = {pack(s[2 * kc][0], s[2 * kc][1]), pack(s[2 * kc][2], s[2 * kc][3]),
                                    pack(s[2 * kc + 1][0], s[2 * kc + 1][1]),
                                    pack(s[2 * kc + 1][2], s[2 * kc + 1][3])};
            const __half2 s01 = __halves2half2(vs[4 * kc], vs[4 * kc + 1]);
            const __half2 s23 = __halves2half2(vs[4 * kc + 2], vs[4 * kc + 3]);
            uint4 vv[4];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                vv[k] = *reinterpret_cast<const uint4 *>(stage + (column(key0, 4 * kc + k, t) - key0) * M2_FP8_ROW +
                    M2_FP8_KEYS + kv * V + 16 * g);
            }
            const uint4 va = vv[0], vb = vv[1], vc = vv[2], vd = vv[3];
            const unsigned a[4] = {va.x, va.y, va.z, va.w}, b[4] = {vb.x, vb.y, vb.z, vb.w};
            const unsigned c[4] = {vc.x, vc.y, vc.z, vc.w}, d[4] = {vd.x, vd.y, vd.z, vd.w};
#pragma unroll
            for (int k = 0; k < 4; k++) {
                // Bytes j of two keys side by side: (a_j, b_j, a_j+1, b_j+1).
                const unsigned ab[2] = {__byte_perm(a[k], b[k], 0x5140), __byte_perm(a[k], b[k], 0x7362)};
                const unsigned cd[2] = {__byte_perm(c[k], d[k], 0x5140), __byte_perm(c[k], d[k], 0x7362)};
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const unsigned x = j % 2 ? ab[j / 2] >> 16 : ab[j / 2] & 0xffffu;
                    const unsigned y = j % 2 ? cd[j / 2] >> 16 : cd[j / 2] & 0xffffu;
                    mma(o[4 * k + j], pa, widen2(x, s01), widen2(y, s23));
                }
            }
        }
        // Every warp has read this stage before it is refilled.
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 2);
    }
    // Output column n of block jb is dim 16 n + jb: this lane's 2t and 2t + 1.
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const unsigned head = g + 8 * r;
        if (t == 0) {
            pmax[at + head] = m[r] * 0.6931471805599453f;
            pden[at + head] = l[r];
        }
        float *dst = pacc + (at + head) * V + 32 * t;
#pragma unroll
        for (int n = 0; n < V / 8; n++) {
            dst[n] = o[n][2 * r];
            dst[16 + n] = o[n][2 * r + 1];
        }
    }
#else
    asm volatile("trap;");
#endif
}

// One block per (head, row), a thread per value dim: the slices' maximum
// first, then their weighted sums with independent loads.
__global__ __launch_bounds__(V) static void merge(
        float *out, const float *pmax, const float *pden, const float *pacc, unsigned slices) {
    const unsigned head = blockIdx.x, row = blockIdx.y, d = threadIdx.x;
    const uint64_t base = (uint64_t)row * slices * 64 + head;
    float top = -INFINITY;
    for (unsigned i = 0; i < slices; i++) {
        if (pden[base + i * 64] > 0.0f) { top = fmaxf(top, pmax[base + i * 64]); }
    }
    float den = 0.0f, acc = 0.0f;
    for (unsigned i = 0; i < slices; i++) {
        const float part = pden[base + i * 64];
        if (!(part > 0.0f)) { continue; }
        const float w = expf(pmax[base + i * 64] - top);
        den += part * w;
        acc += pacc[(base + i * 64) * V + d] * w;
    }
    out[((uint64_t)row * 64 + head) * V + d] = acc / den;
}

// Backend initialization: the 86 KB tile pair needs the opt-in carveout.
static bool init() {
    return cudaFuncSetAttribute(prefill, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess &&
        cudaFuncSetAttribute(decode, cudaFuncAttributeMaxDynamicSharedMemorySize, DECODE_SMEM) == cudaSuccess;
}
} // namespace mimo2_ga8
