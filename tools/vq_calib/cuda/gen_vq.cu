// GVQ v2 (9/10-bit byte-major) kernels for the generation harness (generate_compare.py).
// Weights are formed as bf16(f32(gain[r]) * f32(book[idx][d])), the rounding the reference
// student (eval_e2e.py decode_gvq2 -> BF16) uses; products accumulate in F32.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

constexpr int ROWS_PER_WARP = 4;

__device__ __forceinline__ unsigned vq_index(const uint8_t* __restrict__ main, const uint8_t* __restrict__ plane,
                                             long n, int bits) {
    unsigned id = main[n];
    if (bits == 9) {
        id |= ((unsigned(plane[n >> 3]) >> (n & 7)) & 1u) << 8;
    } else if (bits == 10) {
        id |= ((unsigned(plane[n >> 2]) >> (2 * (n & 3))) & 3u) << 8;
    }
    return id;
}

// y[p, r] = sum_c w_e[r, c] * x[p, c] for every (expert slot u, pair p in [off[u], off[u+1])).
// grid = (ceil(R / 32), U); block = 256 threads (8 warps, 4 rows each); dynamic smem = K*8 floats.
__global__ void vq_gemv_pairs(const uint8_t* __restrict__ main, const uint8_t* __restrict__ plane,
                              const __half* __restrict__ gain, const float* __restrict__ book, int bits,
                              long main_stride, long plane_stride, int R, int C,
                              const int* __restrict__ slot_expert, const int* __restrict__ slot_off,
                              const __nv_bfloat16* __restrict__ x, float* __restrict__ y) {
    extern __shared__ float sbook[];
    const int K = 1 << bits;
    for (int i = threadIdx.x; i < K * 8; i += blockDim.x) { sbook[i] = book[i]; }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int u = blockIdx.y, e = slot_expert[u];
    const int p0 = slot_off[u], p1 = slot_off[u + 1];
    const int V = C / 8;
    const uint8_t* m = main + long(e) * main_stride;
    const uint8_t* pl = plane + long(e) * plane_stride;
    for (int k = 0; k < ROWS_PER_WARP; ++k) {
    const int r = (blockIdx.x * 8 + warp) * ROWS_PER_WARP + k;
    if (r >= R) { return; }
    const float g = __half2float(gain[long(e) * R + r]);
    for (int pb = p0; pb < p1; pb += 4) {
        const int nt = min(4, p1 - pb);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int j = lane; j < V; j += 32) {
            const unsigned id = vq_index(m, pl, long(r) * V + j, bits);
            float w[8];
            #pragma unroll
            for (int d = 0; d < 8; ++d) { w[d] = __bfloat162float(__float2bfloat16_rn(g * sbook[id * 8 + d])); }
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                if (t < nt) {
                    const __nv_bfloat16* xv = x + long(pb + t) * C + j * 8;
                    float s = 0.f;
                    #pragma unroll
                    for (int d = 0; d < 8; ++d) { s = fmaf(w[d], __bfloat162float(xv[d]), s); }
                    acc[t] += s;
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            float v = acc[t];
            #pragma unroll
            for (int o = 16; o; o >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, o); }
            if (lane == 0 && t < nt) { y[long(pb + t) * R + r] = v; }
        }
    }
    }
}

// Full BF16 restore of one expert matrix: out[r, 8j + d] = bf16(gain[r] * book[idx(r, j)][d]).
__global__ void vq_decode_bf16(const uint8_t* __restrict__ main, const uint8_t* __restrict__ plane,
                               const __half* __restrict__ gain, const float* __restrict__ book, int bits,
                               int R, int C, __nv_bfloat16* __restrict__ out) {
    const long n = long(blockIdx.x) * blockDim.x + threadIdx.x;
    const int V = C / 8;
    if (n >= long(R) * V) { return; }
    const int r = int(n / V);
    const unsigned id = vq_index(main, plane, n, bits);
    const float g = __half2float(gain[r]);
    #pragma unroll
    for (int d = 0; d < 8; ++d) { out[n * 8 + d] = __float2bfloat16_rn(g * book[id * 8 + d]); }
}

extern "C" int gen_vq_gemv_pairs(const uint8_t* main, const uint8_t* plane, const __half* gain, const float* book,
                                 int bits, long main_stride, long plane_stride, int R, int C,
                                 const int* slot_expert, const int* slot_off, int U,
                                 const __nv_bfloat16* x, float* y, cudaStream_t stream) {
    if (U <= 0) { return 0; }
    const int K = 1 << bits;
    const size_t smem = size_t(K) * 8 * sizeof(float);
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(vq_gemv_pairs, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    }
    dim3 grid((R + 8 * ROWS_PER_WARP - 1) / (8 * ROWS_PER_WARP), U);
    vq_gemv_pairs<<<grid, 256, smem, stream>>>(main, plane, gain, book, bits, main_stride, plane_stride, R, C,
                                               slot_expert, slot_off, x, y);
    return int(cudaGetLastError());
}

extern "C" int gen_vq_decode_bf16(const uint8_t* main, const uint8_t* plane, const __half* gain, const float* book,
                                  int bits, int R, int C, __nv_bfloat16* out, cudaStream_t stream) {
    const long n = long(R) * (C / 8);
    vq_decode_bf16<<<int((n + 255) / 256), 256, 0, stream>>>(main, plane, gain, book, bits, R, C, out);
    return int(cudaGetLastError());
}
