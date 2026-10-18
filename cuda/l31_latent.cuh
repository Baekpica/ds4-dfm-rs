#ifndef DS4_L31_LATENT_CUH
#define DS4_L31_LATENT_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "../ds4_ling31.h"

/* Ling-3.1 FP8 latent rows: the 512 normalized values of a row as E4M3
 * under one power-of-two scale that maps the row max into (224, 448]. E4M3
 * is floating point, so a shared exponent costs precision only 2^15 below
 * the max, and value x scale is an exact BF16: every reader converts to the
 * BF16 rows DS4_L31_LATENT_SIM stores, so the BF16 kernels run unchanged
 * and FP8 storage is bit-identical to the simulation. */
enum { L31_LATENT_THREADS = DS4_L31_LATENT / 4, L31_LATENT_EXP_MAX = 126 };

__device__ __forceinline__ int l31_latent_exp(float amax) {
    int k = 0;
    frexpf(amax, &k);  // amax < 2^k; 448 x 2^(k-9) = 0.875 x 2^k
    const int e = amax > ldexpf(448.0f, k - 9) ? k - 8 : k - 9;
    return amax > 0.0f ? max(-L31_LATENT_EXP_MAX, min(L31_LATENT_EXP_MAX, e)) : 0;
}

__device__ __forceinline__ float l31_pow2(int e) { return __int_as_float((e + 127) << 23); }

// Sixteen E4M3 values (one 16-byte chunk) as two 16-byte chunks of BF16.
__device__ __forceinline__ void l31_latent_bf16x16(uint4 raw, float scale, uint4 *lo, uint4 *hi) {
    const uint32_t in[4] = {raw.x, raw.y, raw.z, raw.w};
    uint32_t out[8];
    #pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        const __nv_fp8x2_storage_t pair = (__nv_fp8x2_storage_t)(in[i / 2] >> (16 * (i & 1u)));
        const float2 v = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3)));
        const __nv_bfloat162 b = __floats2bfloat162_rn(v.x * scale, v.y * scale);
        out[i] = *(const uint32_t *)&b;
    }
    *lo = make_uint4(out[0], out[1], out[2], out[3]);
    *hi = make_uint4(out[4], out[5], out[6], out[7]);
}

/* Rows pos0.. of the cache, one block per row, four values per thread. FP8
 * writes E4M3 bytes and the row exponent (DS4_L31_LATENT_FP8 layout); SIM
 * writes their BF16 values. k_pe stays BF16 either way. */
template <ds4_l31_latent MODE>
__global__ void l31_store_latent(void *latent, __nv_bfloat16 *k_pe, const float *kv_norm,
        const float *kv_raw, uint32_t pos0, uint32_t cap) {
    __shared__ float peak[L31_LATENT_THREADS / 32];
    const uint32_t row = blockIdx.x, t = threadIdx.x, slot = pos0 + row;
    const float4 x = *(const float4 *)(kv_norm + (uint64_t)row * DS4_L31_LATENT + 4 * t);
    float amax = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w)));
    for (uint32_t s = 16; s; s >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, s)); }
    if (!(t & 31u)) { peak[t >> 5] = amax; }
    __syncthreads();
    for (uint32_t w = 0; w < L31_LATENT_THREADS / 32; w++) { amax = fmaxf(amax, peak[w]); }
    const int e = l31_latent_exp(amax);
    const float inv = l31_pow2(-e);
    const float v[4] = {x.x, x.y, x.z, x.w};
    __nv_fp8_storage_t q[4];
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) { q[i] = __nv_cvt_float_to_fp8(v[i] * inv, __NV_SATFINITE, __NV_E4M3); }
    if (MODE == DS4_L31_LATENT_FP8) {
        uint8_t *bytes = (uint8_t *)latent;
        *(uint32_t *)(bytes + (uint64_t)slot * DS4_L31_LATENT + 4 * t) =
            (uint32_t)q[0] | (uint32_t)q[1] << 8 | (uint32_t)q[2] << 16 | (uint32_t)q[3] << 24;
        if (!t) { ((int8_t *)(bytes + (uint64_t)cap * DS4_L31_LATENT))[slot] = (int8_t)e; }
    } else {
        const float scale = l31_pow2(e);
        __nv_bfloat16 *dst = (__nv_bfloat16 *)latent + (uint64_t)slot * DS4_L31_LATENT + 4 * t;
        #pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
            dst[i] = __float2bfloat16(__half2float(__half(__nv_cvt_fp8_to_halfraw(q[i], __NV_E4M3))) * scale);
        }
    }
    if (t < DS4_L31_ROPE) {
        const float *pe = kv_raw + (uint64_t)row * (DS4_L31_LATENT + DS4_L31_ROPE) + DS4_L31_LATENT;
        k_pe[(uint64_t)slot * DS4_L31_ROPE + t] = __float2bfloat16(pe[t]);
    }
}

// BF16 rows [slot0, slot0 + gridDim.x) of an FP8 cache, one block per row.
__global__ void l31_latent_dequant(__nv_bfloat16 *out, const uint8_t *latent, uint32_t slot0, uint32_t cap) {
    const uint32_t row = blockIdx.x, c = threadIdx.x, slot = slot0 + row;
    const float scale = l31_pow2(((const int8_t *)(latent + (uint64_t)cap * DS4_L31_LATENT))[slot]);
    uint4 lo, hi;
    l31_latent_bf16x16(*(const uint4 *)(latent + (uint64_t)slot * DS4_L31_LATENT + 16 * c), scale, &lo, &hi);
    uint4 *dst = (uint4 *)(out + (uint64_t)row * DS4_L31_LATENT) + 2 * c;
    dst[0] = lo;
    dst[1] = hi;
}

#endif
