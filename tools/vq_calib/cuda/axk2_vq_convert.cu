// A.X-K2 VQ conversion primitives: weighted Lloyd, weighted nearest
// assignment, FP32 row normalization, least-squares gain refit, GVQ v2
// (byte-major 9/10-bit) packing and restore.
//
// A fork of the Beam-1.0 converter (beam_vq_convert.cu), itself a fork of
// the Ling-3.1 tool whose assignment and FP64 Lloyd follow YoungAi 7fb26cc
// gguf-tools/quantize/v41_vq.cu:69-145. Determinism rules are kept:
// equidistant init (host side), FP64 atomic centroid sums, a centroid
// dimension whose weight sum is zero keeps its previous value, distance ties
// go to the lowest codeword index. With every weight equal to 1 the weighted
// paths compute exactly the unweighted arithmetic (w * delta == delta).
// Copyright (c) 2026 The ds4.c authors; MIT (see LICENSE-YoungAi).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

constexpr int DIM = 8;
constexpr int MAX_TILE = 1024;          // codewords per shared-memory tile (32 KiB)

// WMODE 0: unweighted. 1: per-vector weights w[n, 8]. 2: per-column weights
// w[nvec * 8] of a row-major matrix with nvec vectors per row (vector i uses
// columns 8 * (i % nvec) .. + 7). VPT vectors per thread share each codeword
// load from shared memory.
template <int WMODE, int VPT>
__global__ void nearest_kernel(const float* __restrict__ x, const float* __restrict__ w, int nvec,
                               const float* __restrict__ cb, int* __restrict__ ids,
                               float* __restrict__ dist, int64_t n, int k, int tile) {
    extern __shared__ float4 words4[];
    const int64_t first = int64_t(blockIdx.x) * blockDim.x * VPT + threadIdx.x;
    float v[VPT][DIM];
    float wt[VPT][DIM];
    float best[VPT];
    int index[VPT];
    #pragma unroll
    for (int u = 0; u < VPT; ++u) {
        const int64_t i = first + int64_t(u) * blockDim.x;
        const bool live = i < n;
        #pragma unroll
        for (int d = 0; d < DIM; ++d) {
            v[u][d] = live ? x[i * DIM + d] : 0.f;
            if (WMODE == 1) { wt[u][d] = live ? w[i * DIM + d] : 0.f; }
            if (WMODE == 2) { wt[u][d] = live ? w[(i % nvec) * DIM + d] : 0.f; }
        }
        best[u] = INFINITY;
        index[u] = 0;
    }
    for (int base = 0; base < k; base += tile) {
        __syncthreads();
        for (int t = threadIdx.x; t < tile * 2; t += blockDim.x) {
            words4[t] = reinterpret_cast<const float4*>(cb)[base * 2 + t];
        }
        __syncthreads();
        for (int j = 0; j < tile; ++j) {
            const float4 c0 = words4[2 * j], c1 = words4[2 * j + 1];
            const float c[DIM] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
            #pragma unroll
            for (int u = 0; u < VPT; ++u) {
                float distance = 0.f;
                #pragma unroll
                for (int d = 0; d < DIM; ++d) {
                    const float delta = v[u][d] - c[d];
                    if (WMODE == 0) {
                        distance = fmaf(delta, delta, distance);
                    } else {
                        distance = fmaf(wt[u][d] * delta, delta, distance);
                    }
                }
                // Ascending codeword order gives the lowest index on ties.
                if (distance < best[u]) { best[u] = distance; index[u] = base + j; }
            }
        }
    }
    #pragma unroll
    for (int u = 0; u < VPT; ++u) {
        const int64_t i = first + int64_t(u) * blockDim.x;
        if (i < n) {
            ids[i] = index[u];
            if (dist) { dist[i] = best[u]; }
        }
    }
}

// Weighted centroid sums: sum[c, d] += w * x, den[c, d] += w (FP64 atomics).
// w == nullptr means every weight is 1 (the unweighted Lloyd).
__global__ void accumulate_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                  const int* __restrict__ ids, double* sum, double* den, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    const int k = ids[i];
    #pragma unroll
    for (int d = 0; d < DIM; ++d) {
        const double om = w ? double(w[i * DIM + d]) : 1.0;
        if (om == 0.0) { continue; }
        atomicAdd(sum + k * DIM + d, om * double(x[i * DIM + d]));
        atomicAdd(den + k * DIM + d, om);
    }
}

__global__ void centroids_kernel(float* cb, const double* sum, const double* den, int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k * DIM) { return; }
    if (den[i] > 0.0) { cb[i] = float(sum[i] / den[i]); }
}

// One block per output row. FP32 input (FP8 x block scale dequantized in
// FP32), row RMS accumulated in FP64, rounded to F16; the stored F16 gain
// normalizes. Zero rows get gain +0 and normalize to exact zeros; a positive
// RMS that rounds to zero uses the smallest F16 subnormal; overflow or a
// non-finite input rejects the run.
__global__ void normalize_f32_kernel(const float* __restrict__ w, float* __restrict__ normalized,
                                     __half* __restrict__ gains, int cols, int* error) {
    const int64_t row = blockIdx.x;
    __shared__ double sums[256];
    double sum = 0.;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = w[row * cols + c];
        if (!isfinite(v)) { atomicExch(error, 1); }
        sum += double(v) * v;
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s; s /= 2) {
        if (threadIdx.x < s) { sums[threadIdx.x] += sums[threadIdx.x + s]; }
        __syncthreads();
    }
    const float rms = float(sqrt(sums[0] / cols));
    float gain = __half2float(__float2half_rn(rms));
    if (rms > 0.f && gain == 0.f) { gain = 0x1p-24f; }
    if (!isfinite(gain)) { atomicExch(error, 2); }
    if (threadIdx.x == 0) { gains[row] = __float2half_rn(gain); }
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = w[row * cols + c];
        normalized[row * cols + c] = gain > 0.f ? v / gain : 0.f;
    }
}

// Normalization by explicit F16 gains (after a gain refit).
__global__ void normalize_gain_kernel(const float* __restrict__ w, const __half* __restrict__ gains,
                                      float* __restrict__ normalized, int64_t elements, int cols) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) { return; }
    const float g = __half2float(gains[i / cols]);
    normalized[i] = g > 0.f ? w[i] / g : 0.f;
}

// Least-squares row gain for fixed indices: g = sum h w c / sum h c^2 (FP64),
// rounded to F16. Rows whose refit is not a positive finite F16 keep their
// previous gain (the format stores non-negative gains).
__global__ void refit_gain_kernel(const float* __restrict__ w, const int* __restrict__ ids,
                                  const float* __restrict__ cb, const float* __restrict__ colw,
                                  __half* gains, int cols) {
    const int64_t row = blockIdx.x;
    const int nvec = cols / DIM;
    __shared__ double nums[256], dens[256];
    double num = 0., den = 0.;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const int id = ids[row * nvec + c / DIM];
        const double cw = cb[id * DIM + (c % DIM)];
        const double h = colw ? double(colw[c]) : 1.0;
        num += h * double(w[row * cols + c]) * cw;
        den += h * cw * cw;
    }
    nums[threadIdx.x] = num;
    dens[threadIdx.x] = den;
    __syncthreads();
    for (int s = blockDim.x / 2; s; s /= 2) {
        if (threadIdx.x < s) {
            nums[threadIdx.x] += nums[threadIdx.x + s];
            dens[threadIdx.x] += dens[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && dens[0] > 0.) {
        const double g = nums[0] / dens[0];
        const __half h = __double2half(g);
        const float gf = __half2float(h);
        if (g > 0. && isfinite(gf) && gf > 0.f) { gains[row] = h; }
    }
}

// GVQ v2 restore: W[r, 8j + d] = f32(gain[r]) * f32(book[idx[r, j]][d]),
// idx = main[n] | plane bits << 8, n = r * (cols / 8) + j.
__global__ void restore_kernel(const uint8_t* __restrict__ low, const uint8_t* __restrict__ plane,
                               const __half* __restrict__ gain, const float* __restrict__ cb,
                               float* __restrict__ out, int64_t elements, int cols, int bits) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) { return; }
    const int64_t row = i / cols;
    const int col = int(i % cols);
    const int64_t nidx = row * (cols / DIM) + col / DIM;
    unsigned id = low[nidx];
    if (bits == 9) { id |= ((unsigned(plane[nidx >> 3]) >> (nidx & 7)) & 1u) << 8; }
    if (bits == 10) { id |= ((unsigned(plane[nidx >> 2]) >> (2 * (nidx & 3))) & 3u) << 8; }
    out[i] = cb[id * DIM + (col % DIM)] * __half2float(gain[row]);
}

// Decode straight from an int32 index array (no packing), for measurement.
__global__ void restore_ids_kernel(const int* __restrict__ ids, const __half* __restrict__ gain,
                                   const float* __restrict__ cb, float* __restrict__ out,
                                   int64_t elements, int cols) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) { return; }
    const int64_t row = i / cols;
    const int col = int(i % cols);
    out[i] = cb[ids[row * (cols / DIM) + col / DIM] * DIM + (col % DIM)] * __half2float(gain[row]);
}

// GVQ v2 packing: one thread per 8 consecutive indices (rows never split a
// group because cols % 64 == 0). main: low 8 bits, one byte per index.
// 9-bit plane: bit 8 of index n -> bit n % 8 of byte n / 8.
// 10-bit plane: bits 8-9 of index n -> 2-bit field at bit 2 (n % 4) of byte n / 4.
__global__ void pack_kernel(const int* __restrict__ ids, uint8_t* __restrict__ low,
                            uint8_t* __restrict__ plane, int64_t groups, int bits) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= groups) { return; }
    unsigned id[8];
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        id[u] = unsigned(ids[8 * g + u]);
        low[8 * g + u] = uint8_t(id[u] & 255u);
    }
    if (bits == 9) {
        unsigned byte = 0;
        #pragma unroll
        for (int u = 0; u < 8; ++u) { byte |= ((id[u] >> 8) & 1u) << u; }
        plane[g] = uint8_t(byte);
    } else if (bits == 10) {
        unsigned b0 = 0, b1 = 0;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            b0 |= ((id[u] >> 8) & 3u) << (2 * u);
            b1 |= ((id[u + 4] >> 8) & 3u) << (2 * u);
        }
        plane[2 * g] = uint8_t(b0);
        plane[2 * g + 1] = uint8_t(b1);
    }
}

static int tile_for(int k) { return k < MAX_TILE ? k : MAX_TILE; }

template <int WMODE, int VPT>
static int launch_nearest(const float* x, const float* w, int nvec, const float* cb, int* ids,
                          float* dist, int64_t n, int k, int threads, cudaStream_t stream) {
    const int tile = tile_for(k);
    const size_t shared = size_t(tile) * DIM * sizeof(float);
    const int64_t per_block = int64_t(threads) * VPT;
    const int64_t blocks = (n + per_block - 1) / per_block;
    if (shared > 48 * 1024) {
        cudaFuncSetAttribute(nearest_kernel<WMODE, VPT>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared));
    }
    nearest_kernel<WMODE, VPT><<<dim3(unsigned(blocks)), threads, shared, stream>>>(x, w, nvec, cb, ids, dist, n, k, tile);
    return int(cudaGetLastError());
}

static bool valid_k(int k) { return k >= 256 && k <= 8192 && (k & (k - 1)) == 0; }

// mode 0: unweighted, 1: per-vector weights [n, 8], 2: per-column weights [nvec * 8].
extern "C" int axvq_assign(const float* x, const float* w, int mode, int nvec, const float* cb,
                           int* ids, float* dist, int64_t n, int k, int vpt, cudaStream_t stream) {
    if (n <= 0 || !valid_k(k) || mode < 0 || mode > 2 || (mode && !w) || (mode == 2 && nvec <= 0)) {
        return int(cudaErrorInvalidValue);
    }
    const int threads = 256;
    if (vpt == 1) {
        if (mode == 0) { return launch_nearest<0, 1>(x, w, nvec, cb, ids, dist, n, k, threads, stream); }
        if (mode == 1) { return launch_nearest<1, 1>(x, w, nvec, cb, ids, dist, n, k, threads, stream); }
        return launch_nearest<2, 1>(x, w, nvec, cb, ids, dist, n, k, threads, stream);
    }
    if (vpt == 2) {
        if (mode == 0) { return launch_nearest<0, 2>(x, w, nvec, cb, ids, dist, n, k, threads, stream); }
        if (mode == 1) { return launch_nearest<1, 2>(x, w, nvec, cb, ids, dist, n, k, threads, stream); }
        return launch_nearest<2, 2>(x, w, nvec, cb, ids, dist, n, k, threads, stream);
    }
    if (vpt == 4) {
        if (mode == 0) { return launch_nearest<0, 4>(x, w, nvec, cb, ids, dist, n, k, threads, stream); }
        if (mode == 1) { return launch_nearest<1, 4>(x, w, nvec, cb, ids, dist, n, k, threads, stream); }
        return launch_nearest<2, 4>(x, w, nvec, cb, ids, dist, n, k, threads, stream);
    }
    return int(cudaErrorInvalidValue);
}

extern "C" int axvq_update(const float* x, const float* w, const int* ids, float* cb,
                           double* sum, double* den, int64_t n, int k, cudaStream_t stream) {
    if (n <= 0 || !valid_k(k)) { return int(cudaErrorInvalidValue); }
    cudaMemsetAsync(sum, 0, size_t(k) * DIM * sizeof(double), stream);
    cudaMemsetAsync(den, 0, size_t(k) * DIM * sizeof(double), stream);
    accumulate_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(x, w, ids, sum, den, n);
    centroids_kernel<<<(k * DIM + 255) / 256, 256, 0, stream>>>(cb, sum, den, k);
    return int(cudaGetLastError());
}

extern "C" int axvq_normalize_f32(const float* w, float* x, __half* gain, int rows, int cols,
                                  int* error, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || cols % 64) { return int(cudaErrorInvalidValue); }
    normalize_f32_kernel<<<rows, 256, 0, stream>>>(w, x, gain, cols, error);
    return int(cudaGetLastError());
}

extern "C" int axvq_normalize_gain(const float* w, const __half* gain, float* x, int64_t elements,
                                   int cols, cudaStream_t stream) {
    normalize_gain_kernel<<<unsigned((elements + 255) / 256), 256, 0, stream>>>(w, gain, x, elements, cols);
    return int(cudaGetLastError());
}

extern "C" int axvq_refit_gain(const float* w, const int* ids, const float* cb, const float* colw,
                               __half* gain, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols % 64) { return int(cudaErrorInvalidValue); }
    refit_gain_kernel<<<rows, 256, 0, stream>>>(w, ids, cb, colw, gain, cols);
    return int(cudaGetLastError());
}

extern "C" int axvq_restore(const uint8_t* low, const uint8_t* plane, const __half* gain,
                            const float* cb, float* out, int64_t elements, int cols, int bits,
                            cudaStream_t stream) {
    if ((bits != 8 && bits != 9 && bits != 10) || cols % 64) { return int(cudaErrorInvalidValue); }
    restore_kernel<<<unsigned((elements + 255) / 256), 256, 0, stream>>>(low, plane, gain, cb, out, elements, cols, bits);
    return int(cudaGetLastError());
}

extern "C" int axvq_restore_ids(const int* ids, const __half* gain, const float* cb, float* out,
                                int64_t elements, int cols, cudaStream_t stream) {
    if (cols % 64) { return int(cudaErrorInvalidValue); }
    restore_ids_kernel<<<unsigned((elements + 255) / 256), 256, 0, stream>>>(ids, gain, cb, out, elements, cols);
    return int(cudaGetLastError());
}

extern "C" int axvq_pack(const int* ids, uint8_t* low, uint8_t* plane, int64_t n, int bits,
                         cudaStream_t stream) {
    if (n <= 0 || n % 8 || (bits != 8 && bits != 9 && bits != 10)) { return int(cudaErrorInvalidValue); }
    const int64_t groups = n / 8;
    pack_kernel<<<unsigned((groups + 255) / 256), 256, 0, stream>>>(ids, low, plane, groups, bits);
    return int(cudaGetLastError());
}
