#ifndef DS4_SPARK2_5_CUH
#define DS4_SPARK2_5_CUH

/* Spark-X2.6's two graph-specific kernels. The rest of the graph reuses named
 * entries from ds4_spark2_5_graph.inc: the F32-weight RMS norm
 * (ds4_gpu_rms_norm_weight_rows_tensor), the VQ projections
 * (ds4_gpu_gvq_linear_many), the BF16 KV store and windowed GQA attention
 * (ds4_gpu_g4_kv_store / ds4_gpu_g4_attn), and the per-head sigmoid gate from
 * dots3, ds4_gpu_dots3_gate_mul_tensor (ds4_cuda.cu:42100), whose kernel keys
 * the gate as gate_logits[i / value_dim] and therefore takes Spark's 256-wide
 * head unchanged. step37_attn_gate is not that kernel: it hard-codes 128
 * (cuda/step37_primitives.cuh:20-25).
 *
 * Reference: modeling_spark.py in the Spark-X2.6-4B-Base checkpoint (<hf-dir>)
 *   compute_rope_cos_sin  :31-40
 *   apply_rotary_pos_emb  :43-56 (rotate_half :27-30)
 *   Spark2_5MLP.forward   :126-127 (ACT2FN["gelu"], the exact-erf GELU)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <mma.h>

/* Half-split rotary over the first 2*pairs dimensions of a head.
 *
 * The reference builds inv_freq over `rope_head_dim = int(head_dim *
 * partial_rotary_factor)` (modeling_spark.py:31-34), duplicates the angle
 * list and rotates with rotate_half, so dimension i of the head turns against
 * dimension i + rope_head_dim/2 (modeling_spark.py:43-56). `pairs` is
 * rope_head_dim/2 and the angle is position * inv[i], exponent 2i/rope_head_dim:
 * sliding layers pass pairs 128 of a 256-wide head, full layers pairs 32 of
 * the same head and leave dims 64..255 untouched. `dim` is the head width,
 * the input's row stride.
 *
 * One block per (row, head); every thread owns its own pair, so the in-place
 * rotation races nothing. */
__global__ static void spark_rope_kernel(float *x, const uint32_t *positions,
                                         const float *inv, uint32_t rows,
                                         uint32_t heads, uint32_t dim,
                                         uint32_t pairs) {
    const uint64_t job = (uint64_t)blockIdx.x;
    if (job >= (uint64_t)rows * heads || !pairs || 2u * pairs > dim) { return; }
    float *v = x + job * dim;
    const float pos = (float)positions[job / heads];
    for (uint32_t i = threadIdx.x; i < pairs; i += blockDim.x) {
        const float angle = pos * inv[i];
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float a = v[i];
        const float b = v[i + pairs];
        v[i] = a * c - b * s;
        v[i + pairs] = b * c + a * s;
    }
}

/* out = gelu(gate) * up, the exact erf GELU Transformers' ACT2FN["gelu"] is
 * (0.5 x (1 + erf(x/sqrt(2))); the tanh approximation belongs to other
 * families' kernels and is not this one). */
__global__ static void spark_gelu_mul_kernel(float *out, const float *gate,
                                             const float *up, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const float g = gate[i];
    out[i] = 0.5f * g * (1.0f + erff(g * 0.70710678118654752f)) * up[i];
}

/* The fused projection writes [rows][q | k | v] in that order
 * (modeling_spark.py:170-176 slices the one linear's output), while the
 * attention wants each of the three as its own [rows][heads*dim] buffer.
 * One thread per source element, so the split is a single pass. */
__global__ static void spark_qkv_split_kernel(float *q, float *k, float *v,
                                              const float *qkv, uint32_t rows,
                                              uint32_t q_dim, uint32_t kv_dim) {
    const uint64_t stride = (uint64_t)q_dim + 2u * kv_dim;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * stride) { return; }
    const uint32_t row = (uint32_t)(i / stride);
    const uint32_t col = (uint32_t)(i % stride);
    const float value = qkv[i];
    if (col < q_dim) {
        q[(uint64_t)row * q_dim + col] = value;
    } else if (col < q_dim + kv_dim) {
        k[(uint64_t)row * kv_dim + (col - q_dim)] = value;
    } else {
        v[(uint64_t)row * kv_dim + (col - q_dim - kv_dim)] = value;
    }
}

/* Restore one VQ matrix into BF16 rows.
 *
 * Container v2 (one stage): W[r,8j+d] = gain[r] * (C1[i1][d] + R8[j][d]).
 * Container v3 (artifact README "The format (version 3)") adds a second
 * 13-bit stream with its own book and row gains:
 *   W[r,d] = gain[r] * (C1[i1][d] + R8[j][d]) + gain2[r] * C2[i2][d].
 * The stage-2 term is added in FP32 before the single BF16 rounding, the
 * container's per-stage addition.
 *
 * Both index streams are the GVQ1 v1 shape (FORMAT.md): 13-bit indices packed
 * as pairs into 3 bytes plus a separate 13th-bit plane, one bit per vector.
 * The header's payload offsets are resolved on the host (ds4_gpu_spark_restore)
 * so this kernel only walks the streams. One thread per 8-weight vector. */
/* F32 multiply that the global fast-math setting cannot flush: the row gain
 * may be a legal denormal-adjacent value (cuda/glm_vq.cuh glmvq_mul). */
__device__ __forceinline__ float spark_mul(float a, float b) {
    float value;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(value) : "f"(a), "f"(b));
    return value;
}

__global__ static void spark_restore_kernel(
        const uint8_t *main, const uint8_t *plane, const __half *gain,
        const uint8_t *book, const uint8_t *res, const uint8_t *res_book,
        const uint8_t *main2, const uint8_t *plane2, const __half *gain2,
        const uint8_t *book2, uint32_t rows, uint32_t cols,
        __nv_bfloat16 *out) {
    const uint64_t vector = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t vectors = (uint64_t)rows * cols / 8u;
    if (vector >= vectors) { return; }

    const uint64_t pair = vector / 2u * 3u;
    uint32_t index = (vector & 1u)
        ? (main[pair + 1] >> 4) | ((uint32_t)main[pair + 2] << 4)
        : main[pair] | ((uint32_t)(main[pair + 1] & 15u) << 8);
    index |= ((plane[vector / 8u] >> (vector % 8u)) & 1u) << 12;

    uint32_t index2 = 0;
    if (main2) {
        index2 = (vector & 1u)
            ? (main2[pair + 1] >> 4) | ((uint32_t)main2[pair + 2] << 4)
            : main2[pair] | ((uint32_t)(main2[pair + 1] & 15u) << 8);
        index2 |= ((plane2[vector / 8u] >> (vector % 8u)) & 1u) << 12;
    }

    const uint64_t per_row = cols / 8u;
    const uint32_t row = (uint32_t)(vector / per_row);
    const float g1 = __half2float(gain[row]);
    const float g2 = main2 ? __half2float(gain2[row]) : 0.0f;
    const uint32_t r8 = res ? res[vector] : 0u;

    for (uint32_t d = 0; d < 8u; d++) {
        const uint32_t i = index * 8u + d;
        __nv_fp8_e4m3 c1;
        c1.__x = book[i];
        float word = static_cast<float>(c1);
        if (res) {
            __nv_fp8_e4m3 r;
            r.__x = res_book[r8 * 8u + d];
            word += static_cast<float>(r);
        }
        float value = spark_mul(g1, word);
        if (main2) {
            __nv_fp8_e4m3 c2;
            c2.__x = book2[index2 * 8u + d];
            value += spark_mul(g2, static_cast<float>(c2));
        }
        out[vector * 8u + d] = __float2bfloat16_rn(value);
    }
}

__device__ __forceinline__ static uint8_t spark_e4m3_encode(float x) {
    if (!isfinite(x)) { x = 0.0f; }
    return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ static float spark_e4m3_decode(uint8_t code) {
    const __half2 pair = __half2(__nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)code, __NV_E4M3));
    return __low2float(pair);
}

__device__ __forceinline__ static float spark_warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

/* Restore one VQ matrix into FP8 E4M3 rows + one f16 scale per 128 input cols:
 *   out_codes  [rows][cols]           (uint8_t E4M3)
 *   out_scales [rows][cols / 128]     (__half scale, immediately after codes)
 * Sixteen threads (one half-warp) decode 16 x 8 = 128 weights and reduce their
 * max abs across the half-warp before quantizing to E4M3. */
__global__ static void spark_restore_fp8_kernel(
        const uint8_t *main, const uint8_t *plane, const __half *gain,
        const uint8_t *book, const uint8_t *res, const uint8_t *res_book,
        const uint8_t *main2, const uint8_t *plane2, const __half *gain2,
        const uint8_t *book2, uint32_t rows, uint32_t cols,
        uint8_t *out_codes, __half *out_scales) {
    const uint64_t vector = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t vectors = (uint64_t)rows * cols / 8u;
    if (vector >= vectors) { return; }

    const uint64_t pair = vector / 2u * 3u;
    uint32_t index = (vector & 1u)
        ? (main[pair + 1] >> 4) | ((uint32_t)main[pair + 2] << 4)
        : main[pair] | ((uint32_t)(main[pair + 1] & 15u) << 8);
    index |= ((plane[vector / 8u] >> (vector % 8u)) & 1u) << 12;

    uint32_t index2 = 0;
    if (main2) {
        index2 = (vector & 1u)
            ? (main2[pair + 1] >> 4) | ((uint32_t)main2[pair + 2] << 4)
            : main2[pair] | ((uint32_t)(main2[pair + 1] & 15u) << 8);
        index2 |= ((plane2[vector / 8u] >> (vector % 8u)) & 1u) << 12;
    }

    const uint64_t per_row = cols / 8u;
    const uint32_t row = (uint32_t)(vector / per_row);
    const float g1 = __half2float(gain[row]);
    const float g2 = main2 ? __half2float(gain2[row]) : 0.0f;
    const uint32_t r8 = res ? res[vector] : 0u;

    float vals[8];
    float vmax = 0.0f;
    #pragma unroll
    for (uint32_t d = 0; d < 8u; d++) {
        const uint32_t i = index * 8u + d;
        __nv_fp8_e4m3 c1;
        c1.__x = book[i];
        float word = static_cast<float>(c1);
        if (res) {
            __nv_fp8_e4m3 r;
            r.__x = res_book[r8 * 8u + d];
            word += static_cast<float>(r);
        }
        float value = spark_mul(g1, word);
        if (main2) {
            __nv_fp8_e4m3 c2;
            c2.__x = book2[index2 * 8u + d];
            value += spark_mul(g2, static_cast<float>(c2));
        }
        vals[d] = value;
        vmax = fmaxf(vmax, fabsf(value));
    }

    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, offset, 16));
    }

    __half hscale = __float2half_rn(vmax > 0.0f ? (vmax / 448.0f) : 1.0f);
    float scale = __half2float(hscale);
    if (!(scale > 0.0f)) {
        hscale = __float2half_rn(1.0f);
        scale = 1.0f;
    }
    if ((threadIdx.x & 15u) == 0u) {
        out_scales[vector >> 4] = hscale;
    }
    const float inv = 1.0f / scale;
    #pragma unroll
    for (uint32_t d = 0; d < 8u; d++) {
        out_codes[vector * 8u + d] = spark_e4m3_encode(spark_mul(vals[d], inv));
    }
}

/* Quantize a BF16 matrix [rows][cols] into FP8 E4M3 rows + one f16 scale per
 * 128 input cols. 16 threads (one half-warp) process 16 x 8 = 128 elements. */
__global__ static void spark_bf16_to_fp8_kernel(
        const __nv_bfloat16 *src, uint32_t rows, uint32_t cols,
        uint8_t *out_codes, __half *out_scales) {
    const uint64_t vector = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t vectors = (uint64_t)rows * cols / 8u;
    if (vector >= vectors) { return; }

    const uint4 packed = ((const uint4 *)src)[vector];
    const __nv_bfloat16 *bf = reinterpret_cast<const __nv_bfloat16 *>(&packed);
    float vals[8];
    float vmax = 0.0f;
    #pragma unroll
    for (uint32_t d = 0; d < 8u; d++) {
        const float v = __bfloat162float(bf[d]);
        vals[d] = v;
        vmax = fmaxf(vmax, fabsf(v));
    }
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, offset, 16));
    }
    __half hscale = __float2half_rn(vmax > 0.0f ? (vmax / 448.0f) : 1.0f);
    float scale = __half2float(hscale);
    if (!(scale > 0.0f)) {
        hscale = __float2half_rn(1.0f);
        scale = 1.0f;
    }
    if ((threadIdx.x & 15u) == 0u) {
        out_scales[vector >> 4] = hscale;
    }
    const float inv = 1.0f / scale;
    #pragma unroll
    for (uint32_t d = 0; d < 8u; d++) {
        out_codes[vector * 8u + d] = spark_e4m3_encode(spark_mul(vals[d], inv));
    }
}

/* Gather and dequantize `n_tokens` embedding rows from an FP8 E4M3 + block-128
 * scale table [n_vocab][n_embd] into `out` [n_tokens][n_embd] F32. */
__global__ static void spark_embed_fp8_kernel(
        float *out, const int32_t *tokens,
        const uint8_t *codes, const __half *scales,
        uint32_t n_vocab, uint32_t n_embd) {
    const uint32_t r = blockIdx.x;
    int32_t tok = tokens[r];
    if (tok < 0 || (uint32_t)tok >= n_vocab) { tok = 0; }
    const uint32_t n_blocks = n_embd / 128u;
    const uint8_t *row_codes = codes + (uint64_t)tok * n_embd;
    const __half *row_scales = scales + (uint64_t)tok * n_blocks;
    float *row_out = out + (uint64_t)r * n_embd;

    for (uint32_t c4 = threadIdx.x; c4 < n_embd / 4u; c4 += blockDim.x) {
        const uint32_t c = c4 * 4u;
        const uint32_t packed = *(const uint32_t *)(row_codes + c);
        const float s = __half2float(row_scales[c / 128u]);
        const __half2 h01 = __half2(__nv_cvt_fp8x2_to_halfraw2(
            (__nv_fp8x2_storage_t)(packed & 0xffffu), __NV_E4M3));
        const __half2 h23 = __half2(__nv_cvt_fp8x2_to_halfraw2(
            (__nv_fp8x2_storage_t)(packed >> 16), __NV_E4M3));
        const float v0 = __bfloat162float(__float2bfloat16_rn(__low2float(h01) * s));
        const float v1 = __bfloat162float(__float2bfloat16_rn(__high2float(h01) * s));
        const float v2 = __bfloat162float(__float2bfloat16_rn(__low2float(h23) * s));
        const float v3 = __bfloat162float(__float2bfloat16_rn(__high2float(h23) * s));
        ((float4 *)row_out)[c4] = make_float4(v0, v1, v2, v3);
    }
}

/* Vectorized FP8 E4M3 + block-128 scale -> BF16 dequantizer for cuBLAS Tensor
 * Core prefill GEMMs (8 weights per thread: uint2 read, uint4 write). */
__global__ static void spark_fp8_dequant_bf16_kernel(
        __nv_bfloat16 *dst, const uint8_t *codes, const __half *scales,
        uint64_t vectors) {
    const uint64_t vec = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (vec >= vectors) { return; }
    const uint2 packed = ((const uint2 *)codes)[vec];
    const float s = __half2float(scales[vec >> 4]);
    const float2 f0 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(packed.x & 0xffffu), __NV_E4M3)));
    const float2 f1 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(packed.x >> 16u), __NV_E4M3)));
    const float2 f2 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(packed.y & 0xffffu), __NV_E4M3)));
    const float2 f3 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(packed.y >> 16u), __NV_E4M3)));
    const __nv_bfloat162 b0 = __floats2bfloat162_rn(f0.x * s, f0.y * s);
    const __nv_bfloat162 b1 = __floats2bfloat162_rn(f1.x * s, f1.y * s);
    const __nv_bfloat162 b2 = __floats2bfloat162_rn(f2.x * s, f2.y * s);
    const __nv_bfloat162 b3 = __floats2bfloat162_rn(f3.x * s, f3.y * s);
    ((uint4 *)dst)[vec] = make_uint4(
        *reinterpret_cast<const uint32_t *>(&b0),
        *reinterpret_cast<const uint32_t *>(&b1),
        *reinterpret_cast<const uint32_t *>(&b2),
        *reinterpret_cast<const uint32_t *>(&b3));
}

/* Shared-memory staged 8-warp single-row FP8 E4M3 GEMV (rows == 1).
 * Stages `x[in_dim]` into shared memory once per block, streams 128-byte
 * coalesced weight blocks with `__ldcs`, and broadcasts the block scale from
 * lane 0 via `__shfl_sync`. */
__global__ static void __launch_bounds__(256, 4) spark_fp8_gemv_1_kernel(
        const uint8_t *__restrict__ w_codes,
        const __half *__restrict__ w_scales,
        const float *__restrict__ x,
        float *__restrict__ out,
        uint32_t in_dim, uint32_t out_dim) {
    extern __shared__ __align__(16) float sx[];
    for (uint32_t c4 = threadIdx.x; c4 < in_dim / 4u; c4 += 256u) {
        ((float4 *)sx)[c4] = ((const float4 *)x)[c4];
    }
    __syncthreads();

    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t o = blockIdx.x * 8u + warp;
    if (o >= out_dim) { return; }

    const uint32_t n_blocks = in_dim / 128u;
    const uint8_t *w_row = w_codes + (uint64_t)o * in_dim;
    const __half *s_row = w_scales + (uint64_t)o * n_blocks;
    float acc = 0.0f;

    #pragma unroll 4
    for (uint32_t b = 0; b < n_blocks; b++) {
        const uint32_t packed = __ldcs(((const uint32_t *)(w_row + b * 128u)) + lane);
        const float s = __shfl_sync(0xffffffffu,
            lane == 0u ? __half2float(s_row[b]) : 0.0f, 0);
        const float4 xv = ((const float4 *)(sx + b * 128u))[lane];
        const __half2 h01 = __half2(__nv_cvt_fp8x2_to_halfraw2(
            (__nv_fp8x2_storage_t)(packed & 0xffffu), __NV_E4M3));
        const __half2 h23 = __half2(__nv_cvt_fp8x2_to_halfraw2(
            (__nv_fp8x2_storage_t)(packed >> 16), __NV_E4M3));
        const float dot = __low2float(h01) * xv.x + __high2float(h01) * xv.y +
                          __low2float(h23) * xv.z + __high2float(h23) * xv.w;
        acc = fmaf(dot, s, acc);
    }

    #pragma unroll
    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) {
        out[o] = acc;
    }
}

/* Fast decode GEMV (2 <= rows <= 4) over FP8 E4M3 weights + block-128 f16 scales.
 * 8 warps per block; each warp computes 1 output column across `rows` tokens. */
__global__ static void __launch_bounds__(256, 4) spark_fp8_gemv_kernel(
        const uint8_t *__restrict__ w_codes,
        const __half *__restrict__ w_scales,
        const float *__restrict__ x,
        float *__restrict__ out,
        uint32_t rows, uint32_t in_dim, uint32_t out_dim) {
    const uint32_t warp = threadIdx.x / 32u;
    const uint32_t lane = threadIdx.x % 32u;
    const uint32_t o = blockIdx.x * 8u + warp;
    if (o >= out_dim) { return; }

    const uint32_t n_blocks = in_dim / 128u;
    const uint8_t *w_row = w_codes + (uint64_t)o * in_dim;
    const __half *s_row = w_scales + (uint64_t)o * n_blocks;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint32_t b = 0; b < n_blocks; b++) {
        const uint32_t packed = __ldcs(((const uint32_t *)(w_row + b * 128u)) + lane);
        const __half2 h01 = __half2(__nv_cvt_fp8x2_to_halfraw2(
            (__nv_fp8x2_storage_t)(packed & 0xffffu), __NV_E4M3));
        const __half2 h23 = __half2(__nv_cvt_fp8x2_to_halfraw2(
            (__nv_fp8x2_storage_t)(packed >> 16), __NV_E4M3));
        const float s = __shfl_sync(0xffffffffu,
            lane == 0u ? __half2float(s_row[b]) : 0.0f, 0);
        const float w0 = __low2float(h01) * s;
        const float w1 = __high2float(h01) * s;
        const float w2 = __low2float(h23) * s;
        const float w3 = __high2float(h23) * s;

        #pragma unroll
        for (uint32_t r = 0; r < 4u; r++) {
            if (r < rows) {
                const float4 xv = __ldg(((const float4 *)(x + (uint64_t)r * in_dim + b * 128u)) + lane);
                acc[r] += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;
            }
        }
    }

    #pragma unroll
    for (uint32_t r = 0; r < 4u; r++) {
        if (r < rows) {
            const float sum = spark_warp_sum(acc[r]);
            if (lane == 0u) {
                out[(uint64_t)r * out_dim + o] = sum;
            }
        }
    }
}

/* Prefill GEMM (rows > 4) over FP8 E4M3 weights + block-128 f16 scales.
 * 64x64 output tile per block (8 warps), 32 K-columns per step. */
__global__ static void spark_fp8_gemm_kernel(
        const uint8_t *w_codes, const __half *w_scales,
        const float *x, float *out,
        uint32_t rows, uint32_t in_dim, uint32_t out_dim) {
    using namespace nvcuda::wmma;
    constexpr uint32_t TILE_ROWS = 64u;
    constexpr uint32_t TILE_OUT = 64u;
    constexpr uint32_t TILE_K = 32u;
    constexpr uint32_t KLD = TILE_K + 8u;

    const uint32_t out0 = blockIdx.x * TILE_OUT;
    const uint32_t row0 = blockIdx.y * TILE_ROWS;
    const uint32_t n_blocks = in_dim / 128u;

    __shared__ __nv_bfloat16 sw[TILE_OUT][KLD];
    __shared__ __nv_bfloat16 sx[TILE_ROWS][KLD];
    __shared__ float sc[TILE_ROWS][TILE_OUT];

    const uint32_t warp = threadIdx.x / 32u;
    const uint32_t row_group = warp / 2u;
    const uint32_t col_group = warp % 2u;

    fragment<accumulator, 16, 16, 16, float> acc[2];
    for (uint32_t i = 0; i < 2u; i++) {
        fill_fragment(acc[i], 0.0f);
    }

    for (uint32_t k0 = 0; k0 < in_dim; k0 += TILE_K) {
        for (uint32_t i = threadIdx.x; i < TILE_ROWS * TILE_K; i += blockDim.x) {
            const uint32_t r = i / TILE_K;
            const uint32_t k = i % TILE_K;
            const float v = (row0 + r) < rows
                ? x[(uint64_t)(row0 + r) * in_dim + k0 + k] : 0.0f;
            sx[r][k] = __float2bfloat16_rn(v);
        }
        const uint32_t b = k0 / 128u;
        for (uint32_t i = threadIdx.x; i < TILE_OUT * (TILE_K / 4u); i += blockDim.x) {
            const uint32_t n = i / (TILE_K / 4u);
            const uint32_t k4 = (i % (TILE_K / 4u)) * 4u;
            const uint32_t o = out0 + n;
            if (o < out_dim) {
                const uint32_t packed = *(const uint32_t *)(w_codes + (uint64_t)o * in_dim + k0 + k4);
                const __half2 h01 = __half2(__nv_cvt_fp8x2_to_halfraw2(
                    (__nv_fp8x2_storage_t)(packed & 0xffffu), __NV_E4M3));
                const __half2 h23 = __half2(__nv_cvt_fp8x2_to_halfraw2(
                    (__nv_fp8x2_storage_t)(packed >> 16), __NV_E4M3));
                const float s = __half2float(w_scales[(uint64_t)o * n_blocks + b]);
                sw[n][k4 + 0u] = __float2bfloat16_rn(__low2float(h01) * s);
                sw[n][k4 + 1u] = __float2bfloat16_rn(__high2float(h01) * s);
                sw[n][k4 + 2u] = __float2bfloat16_rn(__low2float(h23) * s);
                sw[n][k4 + 3u] = __float2bfloat16_rn(__high2float(h23) * s);
            } else {
                sw[n][k4 + 0u] = __float2bfloat16_rn(0.0f);
                sw[n][k4 + 1u] = __float2bfloat16_rn(0.0f);
                sw[n][k4 + 2u] = __float2bfloat16_rn(0.0f);
                sw[n][k4 + 3u] = __float2bfloat16_rn(0.0f);
            }
        }
        __syncthreads();

        #pragma unroll
        for (uint32_t kk = 0; kk < TILE_K; kk += 16u) {
            fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a;
            load_matrix_sync(a, &sx[row_group * 16u][kk], KLD);
            #pragma unroll
            for (uint32_t i = 0; i < 2u; i++) {
                fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag;
                load_matrix_sync(b_frag, &sw[col_group * 32u + i * 16u][kk], KLD);
                mma_sync(acc[i], a, b_frag, acc[i]);
            }
        }
        __syncthreads();
    }

    for (uint32_t i = 0; i < 2u; i++) {
        store_matrix_sync(&sc[row_group * 16u][col_group * 32u + i * 16u],
                          acc[i], TILE_OUT, mem_row_major);
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < TILE_ROWS * TILE_OUT; i += blockDim.x) {
        const uint32_t r = i / TILE_OUT;
        const uint32_t c = i % TILE_OUT;
        if ((row0 + r) < rows && (out0 + c) < out_dim) {
            out[(uint64_t)(row0 + r) * out_dim + out0 + c] = sc[r][c];
        }
    }
}

/* Spark's own FP8 KV store: E4M3 K [kv_heads][dim] | E4M3 V [kv_heads][dim] |
 * f16 K scale [kv_heads] | f16 V scale [kv_heads], for both sliding rings and
 * full-context layers. */
__global__ static void spark_kv_store_fp8_kernel(uint8_t *dst, const float *k,
        const float *v, const uint32_t *positions, uint32_t rows,
        uint32_t kv_heads, uint32_t dim, uint32_t ring) {
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (token >= rows || head >= kv_heads) { return; }

    const uint64_t kv_dim = (uint64_t)kv_heads * dim;
    const uint64_t src = ((uint64_t)token * kv_heads + head) * dim;
    const uint32_t slot = ring ? positions[token] % ring : positions[token];
    const uint64_t row_bytes = 2u * kv_dim + 2u * (uint64_t)kv_heads * sizeof(uint16_t);
    uint8_t *row = dst + (uint64_t)slot * row_bytes;

    __shared__ float k_abs[256];
    __shared__ float v_abs[256];
    float kmax = 0.0f, vmax = 0.0f;
    for (uint32_t d = tid; d < dim; d += blockDim.x) {
        const float kx = k[src + d];
        const float vx = v[src + d];
        if (isfinite(kx)) { kmax = fmaxf(kmax, fabsf(kx)); }
        if (isfinite(vx)) { vmax = fmaxf(vmax, fabsf(vx)); }
    }
    k_abs[tid] = kmax;
    v_abs[tid] = vmax;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            k_abs[tid] = fmaxf(k_abs[tid], k_abs[tid + stride]);
            v_abs[tid] = fmaxf(v_abs[tid], v_abs[tid + stride]);
        }
        __syncthreads();
    }
    __shared__ float kscale, vscale;
    __half *scales = (__half *)(row + 2u * kv_dim);
    if (tid == 0u) {
        __half hk = __float2half_rn(k_abs[0] > 0.0f ? (k_abs[0] / 448.0f) : 1.0f);
        __half hv = __float2half_rn(v_abs[0] > 0.0f ? (v_abs[0] / 448.0f) : 1.0f);
        float fk = __half2float(hk);
        float fv = __half2float(hv);
        if (!(fk > 0.0f)) { hk = __float2half_rn(1.0f); fk = 1.0f; }
        if (!(fv > 0.0f)) { hv = __float2half_rn(1.0f); fv = 1.0f; }
        scales[head] = hk;
        scales[kv_heads + head] = hv;
        kscale = fk;
        vscale = fv;
    }
    __syncthreads();
    const uint64_t head0 = (uint64_t)head * dim;
    for (uint32_t d = tid; d < dim; d += blockDim.x) {
        row[head0 + d] = spark_e4m3_encode(k[src + d] / kscale);
        row[kv_dim + head0 + d] = spark_e4m3_encode(v[src + d] / vscale);
    }
}

/* Spark's own FP8 causal GQA attention (both sliding window and full context).
 * Vectorized uint2 (8 E4M3 bytes per lane) when dim == 256. */
__global__ static void spark_attn_fp8_kernel(float *out, const float *query,
        const uint8_t *kv, uint32_t rows, uint32_t first, uint32_t heads,
        uint32_t kv_heads, uint32_t dim, uint32_t window, uint32_t ring,
        uint32_t keys, uint64_t row_bytes) {
    const uint32_t warp = threadIdx.x / 32u;
    const uint32_t lane = threadIdx.x % 32u;
    const uint64_t job = (uint64_t)blockIdx.x * (blockDim.x / 32u) + warp;
    if (job >= (uint64_t)rows * heads) { return; }

    const uint32_t row = (uint32_t)(job / heads);
    const uint32_t head = (uint32_t)(job % heads);
    const uint32_t group = heads / kv_heads;
    const uint32_t kv_head = head / group;
    const uint32_t position = first + row;
    const float *q = query + job * dim;
    float *o = out + job * dim;
    const uint64_t kv_dim = (uint64_t)kv_heads * dim;

    const uint32_t begin = window && position + 1u > window ? position + 1u - window : 0u;
    const uint32_t last = keys && keys <= position ? keys - 1u : position;

    if (dim == 256u) {
        const uint32_t d0 = lane * 8u;
        const float4 q0 = ((const float4 *)(q + d0))[0];
        const float4 q1 = ((const float4 *)(q + d0))[1];
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float maximum = -INFINITY, total = 0.0f;

        for (uint32_t k = begin; k <= last; k++) {
            const uint64_t slot = ring ? (uint64_t)(k % ring) : k;
            const uint8_t *base = kv + slot * row_bytes;
            const uint2 kp = ((const uint2 *)(base + (uint64_t)kv_head * 256u))[lane];
            const uint2 vp = ((const uint2 *)(base + kv_dim + (uint64_t)kv_head * 256u))[lane];
            const __half *scales = (const __half *)(base + 2u * kv_dim);
            const float kscale = __half2float(scales[kv_head]);
            const float vscale = __half2float(scales[kv_heads + kv_head]);

            const __half2 kh0 = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(kp.x & 0xffffu), __NV_E4M3));
            const __half2 kh1 = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(kp.x >> 16), __NV_E4M3));
            const __half2 kh2 = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(kp.y & 0xffffu), __NV_E4M3));
            const __half2 kh3 = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(kp.y >> 16), __NV_E4M3));

            float dot = q0.x * __low2float(kh0) + q0.y * __high2float(kh0) +
                        q0.z * __low2float(kh1) + q0.w * __high2float(kh1) +
                        q1.x * __low2float(kh2) + q1.y * __high2float(kh2) +
                        q1.z * __low2float(kh3) + q1.w * __high2float(kh3);
            dot = spark_warp_sum(dot);
            dot = __shfl_sync(0xffffffffu, dot, 0) * kscale;

            const float m_new = fmaxf(maximum, dot);
            const float alpha = expf(maximum - m_new);
            const float p = expf(dot - m_new);
            total = total * alpha + p;
            maximum = m_new;
            const float pv = p * vscale;

            const __half2 vh0 = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(vp.x & 0xffffu), __NV_E4M3));
            const __half2 vh1 = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(vp.x >> 16), __NV_E4M3));
            const __half2 vh2 = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(vp.y & 0xffffu), __NV_E4M3));
            const __half2 vh3 = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(vp.y >> 16), __NV_E4M3));

            acc[0] = acc[0] * alpha + pv * __low2float(vh0);
            acc[1] = acc[1] * alpha + pv * __high2float(vh0);
            acc[2] = acc[2] * alpha + pv * __low2float(vh1);
            acc[3] = acc[3] * alpha + pv * __high2float(vh1);
            acc[4] = acc[4] * alpha + pv * __low2float(vh2);
            acc[5] = acc[5] * alpha + pv * __high2float(vh2);
            acc[6] = acc[6] * alpha + pv * __low2float(vh3);
            acc[7] = acc[7] * alpha + pv * __high2float(vh3);
        }

        const float inverse = 1.0f / total;
        ((float4 *)(o + d0))[0] = make_float4(
            acc[0] * inverse, acc[1] * inverse, acc[2] * inverse, acc[3] * inverse);
        ((float4 *)(o + d0))[1] = make_float4(
            acc[4] * inverse, acc[5] * inverse, acc[6] * inverse, acc[7] * inverse);
        return;
    }

    const uint32_t per_lane = (dim + 31u) / 32u;
    float acc[16];
    float qv[16];
    for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
        const uint32_t d = lane + i * 32u;
        qv[i] = d < dim ? q[d] : 0.0f;
        acc[i] = 0.0f;
    }
    float maximum = -INFINITY, total = 0.0f;
    for (uint32_t k = begin; k <= last; k++) {
        const uint64_t slot = ring ? (uint64_t)(k % ring) : k;
        const uint8_t *base = kv + slot * row_bytes;
        const uint8_t *krow = base + (uint64_t)kv_head * dim;
        const uint8_t *vrow = base + kv_dim + (uint64_t)kv_head * dim;
        const __half *scales = (const __half *)(base + 2u * kv_dim);
        const float kscale = __half2float(scales[kv_head]);
        const float vscale = __half2float(scales[kv_heads + kv_head]);
        float dot = 0.0f;
        for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
            const uint32_t d = lane + i * 32u;
            if (d < dim) { dot += qv[i] * spark_e4m3_decode(krow[d]); }
        }
        dot = spark_warp_sum(dot);
        dot = __shfl_sync(0xffffffffu, dot, 0) * kscale;
        const float m_new = fmaxf(maximum, dot);
        const float alpha = expf(maximum - m_new);
        const float p = expf(dot - m_new);
        total = total * alpha + p;
        maximum = m_new;
        for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
            const uint32_t d = lane + i * 32u;
            if (d < dim) {
                acc[i] = acc[i] * alpha + p * vscale * spark_e4m3_decode(vrow[d]);
            }
        }
    }
    const float inverse = 1.0f / total;
    for (uint32_t i = 0; i < per_lane && i < 16u; i++) {
        const uint32_t d = lane + i * 32u;
        if (d < dim) { o[d] = acc[i] * inverse; }
    }
}

/* 8-warp parallel split-key FP8 decode attention (rows == 1, dim == 256).
 * One CTA of 8 warps (256 threads) per query head; each warp strides across
 * every 8th key in [begin, last] with vectorized uint2 E4M3 loads and merges
 * the 8 partial online-softmax states in shared memory. */
__global__ static void __launch_bounds__(256, 4) spark_attn_decode_fp8_kernel(
        float *out, const float *query, const uint8_t *kv,
        uint32_t first, uint32_t heads, uint32_t kv_heads,
        uint32_t window, uint32_t ring, uint32_t keys, uint64_t row_bytes) {
    constexpr uint32_t D = 256u, WARPS = 8u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t head = blockIdx.x;
    if (head >= heads) { return; }

    const uint32_t group = heads / kv_heads;
    const uint32_t kv_head = head / group;
    const uint32_t position = first;
    const float *q = query + (uint64_t)head * D;
    float *o = out + (uint64_t)head * D;
    const uint64_t kv_dim = (uint64_t)kv_heads * D;

    const uint32_t begin = window && position + 1u > window ? position + 1u - window : 0u;
    const uint32_t last = keys && keys <= position ? keys - 1u : position;

    const uint32_t d0 = lane * 8u;
    const float4 q0 = ((const float4 *)(q + d0))[0];
    const float4 q1 = ((const float4 *)(q + d0))[1];
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float maximum = -INFINITY, total = 0.0f;

    for (uint32_t k = begin + warp; k <= last; k += WARPS) {
        const uint64_t slot = ring ? (uint64_t)(k % ring) : k;
        const uint8_t *base = kv + slot * row_bytes;
        const uint2 kp = ((const uint2 *)(base + (uint64_t)kv_head * D))[lane];
        const uint2 vp = ((const uint2 *)(base + kv_dim + (uint64_t)kv_head * D))[lane];
        const __half *scales = (const __half *)(base + 2u * kv_dim);
        const float kscale = __half2float(scales[kv_head]);
        const float vscale = __half2float(scales[kv_heads + kv_head]);

        const __half2 kh0 = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(kp.x & 0xffffu), __NV_E4M3));
        const __half2 kh1 = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(kp.x >> 16), __NV_E4M3));
        const __half2 kh2 = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(kp.y & 0xffffu), __NV_E4M3));
        const __half2 kh3 = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(kp.y >> 16), __NV_E4M3));

        float dot = q0.x * __low2float(kh0) + q0.y * __high2float(kh0) +
                    q0.z * __low2float(kh1) + q0.w * __high2float(kh1) +
                    q1.x * __low2float(kh2) + q1.y * __high2float(kh2) +
                    q1.z * __low2float(kh3) + q1.w * __high2float(kh3);
        dot = spark_warp_sum(dot);
        dot = __shfl_sync(0xffffffffu, dot, 0) * kscale;

        const float m_new = fmaxf(maximum, dot);
        const float alpha = expf(maximum - m_new);
        const float p = expf(dot - m_new);
        total = total * alpha + p;
        maximum = m_new;
        const float pv = p * vscale;

        const __half2 vh0 = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(vp.x & 0xffffu), __NV_E4M3));
        const __half2 vh1 = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(vp.x >> 16), __NV_E4M3));
        const __half2 vh2 = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(vp.y & 0xffffu), __NV_E4M3));
        const __half2 vh3 = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(vp.y >> 16), __NV_E4M3));

        acc[0] = acc[0] * alpha + pv * __low2float(vh0);
        acc[1] = acc[1] * alpha + pv * __high2float(vh0);
        acc[2] = acc[2] * alpha + pv * __low2float(vh1);
        acc[3] = acc[3] * alpha + pv * __high2float(vh1);
        acc[4] = acc[4] * alpha + pv * __low2float(vh2);
        acc[5] = acc[5] * alpha + pv * __high2float(vh2);
        acc[6] = acc[6] * alpha + pv * __low2float(vh3);
        acc[7] = acc[7] * alpha + pv * __high2float(vh3);
    }

    __shared__ float s_max[WARPS];
    __shared__ float s_sum[WARPS];
    __shared__ float s_acc[WARPS][D];
    if (lane == 0u) {
        s_max[warp] = maximum;
        s_sum[warp] = total;
    }
    #pragma unroll
    for (uint32_t i = 0; i < 8u; i++) {
        s_acc[warp][d0 + i] = acc[i];
    }
    __syncthreads();

    float g_max = -INFINITY;
    #pragma unroll
    for (uint32_t w = 0; w < WARPS; w++) {
        g_max = fmaxf(g_max, s_max[w]);
    }
    float g_sum = 0.0f;
    float g_out = 0.0f;
    #pragma unroll
    for (uint32_t w = 0; w < WARPS; w++) {
        const float sw = s_sum[w] > 0.0f ? expf(s_max[w] - g_max) : 0.0f;
        g_sum += s_sum[w] * sw;
        g_out += s_acc[w][tid] * sw;
    }
    o[tid] = g_sum > 0.0f ? (g_out / g_sum) : 0.0f;
}

/* ---------------- Tensor-Core m16n8k16 MMA FP8 Prefill Attention ---------------- */

__device__ __forceinline__ static uint32_t spark_mma_swz(uint32_t key, uint32_t chunk, uint32_t dim) {
    return key * dim + ((chunk ^ (key & 7u)) << 3);
}

__device__ __forceinline__ static void spark_mma_ldsm4(uint32_t *r, const void *p, bool trans) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    if (trans) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    }
}

__device__ __forceinline__ static uint32_t spark_mma_pack_h(float lo, float hi) {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&v);
}

__device__ __forceinline__ static void spark_mma_mma_h(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ static uint4 spark_fp8_decode8(uint2 codes, __half scale) {
    const float s = __half2float(scale);
    const float2 f0 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(codes.x & 0xffffu), __NV_E4M3)));
    const float2 f1 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(codes.x >> 16u), __NV_E4M3)));
    const float2 f2 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(codes.y & 0xffffu), __NV_E4M3)));
    const float2 f3 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(codes.y >> 16u), __NV_E4M3)));
    const __half2 h0 = __floats2half2_rn(f0.x * s, f0.y * s);
    const __half2 h1 = __floats2half2_rn(f1.x * s, f1.y * s);
    const __half2 h2 = __floats2half2_rn(f2.x * s, f2.y * s);
    const __half2 h3 = __floats2half2_rn(f3.x * s, f3.y * s);
    return make_uint4(*reinterpret_cast<const uint32_t *>(&h0),
                      *reinterpret_cast<const uint32_t *>(&h1),
                      *reinterpret_cast<const uint32_t *>(&h2),
                      *reinterpret_cast<const uint32_t *>(&h3));
}

/* Spark Tensor-Core m16n8k16 MMA attention over combined FP8 KV rows:
 * D = 256, GROUP = 4 (16 query heads / 4 KV heads), 4 tokens x 4 query heads
 * per warp (16 M rows), 8 warps per CTA -> RTILE = 32 tokens per CTA.
 * Works for both the 27 sliding-window layers (window = 512) and the 9
 * full-attention layers (window = 0). */
__global__ static void __launch_bounds__(256, 1) spark_attn_mma_fp8_kernel(
        float *out, const float *query, const uint8_t *fp8_kv,
        uint32_t rows, uint32_t first, uint32_t heads, uint32_t kv_heads,
        uint32_t window, uint32_t ring, uint32_t keys, uint64_t row_bytes) {
    constexpr uint32_t D = 256u, GROUP = 4u, RTILE = 32u, TK = 32u;
    extern __shared__ __align__(16) uint8_t smem[];
    __half *ks = (__half *)smem;
    __half *vs = ks + TK * D;

    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t g = lane >> 2u, q = lane & 3u;
    const uint32_t kv_head = blockIdx.x;
    const uint32_t row0 = blockIdx.y * RTILE;
    const uint32_t row_end = row0 + RTILE < rows ? row0 + RTILE : rows;
    const uint32_t pos_first = first + row0;
    const uint32_t pos_last = first + row_end - 1u;
    const uint32_t begin_all = window && pos_first + 1u > window ? pos_first + 1u - window : 0u;
    const uint32_t last_all = keys && keys <= pos_last ? keys - 1u : pos_last;
    if (keys && begin_all >= keys) { return; }
    const uint32_t base_start = (begin_all / TK) * TK;
    const uint64_t kv_dim = (uint64_t)kv_heads * D;

    /* Q fragments: 16 M rows (4 tokens x 4 query heads) x 16 k16 chunks of D. */
    uint32_t qa[16][4];
    #pragma unroll
    for (uint32_t k = 0; k < 16u; k++) {
        const uint32_t col = k * 16u + 2u * q;
        #pragma unroll
        for (uint32_t m = 0; m < 2u; m++) {
            const uint32_t row_g = g + 8u * m;
            const uint32_t token = row0 + warp * 4u + (row_g >> 2u);
            const bool valid = token < rows;
            const uint32_t head = kv_head * GROUP + (row_g & 3u);
            const float *qr = query + ((uint64_t)token * heads + head) * D;
            const float lo = valid ? qr[col] : 0.0f;
            const float hi = valid ? qr[col + 1u] : 0.0f;
            const float lo8 = valid ? qr[col + 8u] : 0.0f;
            const float hi8 = valid ? qr[col + 9u] : 0.0f;
            if (m == 0u) {
                qa[k][0] = spark_mma_pack_h(lo, hi);
                qa[k][2] = spark_mma_pack_h(lo8, hi8);
            } else {
                qa[k][1] = spark_mma_pack_h(lo, hi);
                qa[k][3] = spark_mma_pack_h(lo8, hi8);
            }
        }
    }

    float o[32][4] = {};
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};

    for (uint32_t base = base_start; base <= last_all; base += TK) {
        for (uint32_t t = warp * 4u; t < warp * 4u + 4u; t++) {
            const uint32_t k_abs = base + t;
            const bool valid = k_abs >= begin_all && k_abs <= last_all;
            const uint32_t slot = valid ? (ring ? (k_abs % ring) : k_abs) : 0u;
            const uint8_t *rbase = fp8_kv + (uint64_t)slot * row_bytes;
            const uint8_t *ksrc = rbase + (uint64_t)kv_head * D;
            const uint8_t *vsrc = rbase + kv_dim + (uint64_t)kv_head * D;
            const __half *scales = (const __half *)(rbase + 2u * kv_dim);
            const __half ks_h = valid ? scales[kv_head] : __float2half_rn(0.0f);
            const __half vs_h = valid ? scales[kv_heads + kv_head] : __float2half_rn(0.0f);
            for (uint32_t c = lane; c < D / 8u; c += 32u) {
                const uint32_t swz = spark_mma_swz(t, c, D);
                *(uint4 *)(ks + swz) = valid
                    ? spark_fp8_decode8(*(const uint2 *)(ksrc + c * 8u), ks_h)
                    : make_uint4(0u, 0u, 0u, 0u);
                *(uint4 *)(vs + swz) = valid
                    ? spark_fp8_decode8(*(const uint2 *)(vsrc + c * 8u), vs_h)
                    : make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();

        /* S = Q K^T over the 32-key tile on Tensor Cores (m16n8k16). */
        float s[4][4] = {};
        #pragma unroll
        for (uint32_t k = 0; k < 16u; k++) {
            #pragma unroll
            for (uint32_t j = 0; j < 4u; j += 2u) {
                uint32_t b[4];
                const uint32_t mat = lane >> 3u;
                const uint32_t krow = (j + (mat >> 1u)) * 8u + (lane & 7u);
                const uint32_t chunk = k * 2u + (mat & 1u);
                spark_mma_ldsm4(b, ks + spark_mma_swz(krow, chunk, D), false);
                spark_mma_mma_h(s[j], qa[k], b[0], b[1]);
                spark_mma_mma_h(s[j + 1u], qa[k], b[2], b[3]);
            }
        }

        /* Mask and online softmax, per row h = 0 (g) and h = 1 (g + 8). */
        #pragma unroll
        for (uint32_t h = 0; h < 2u; h++) {
            const uint32_t mrow = g + 8u * h;
            const uint32_t token = row0 + warp * 4u + (mrow >> 2u);
            const bool rv = token < rows;
            const uint32_t pos = first + token;
            const uint32_t rb = window && pos + 1u > window ? pos + 1u - window : 0u;
            const uint32_t rl = keys && keys <= pos ? keys - 1u : pos;
            float peak = -INFINITY;
            #pragma unroll
            for (uint32_t nt = 0; nt < 4u; nt++) {
                #pragma unroll
                for (uint32_t e = 0; e < 2u; e++) {
                    const uint32_t kk = base + nt * 8u + 2u * q + e;
                    if (!rv || kk < rb || kk > rl) { s[nt][h * 2u + e] = -INFINITY; }
                    peak = fmaxf(peak, s[nt][h * 2u + e]);
                }
            }
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, 1));
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, 2));
            const bool tv = peak > -INFINITY;
            const float mn = tv ? fmaxf(m[h], peak) : m[h];
            const float alpha = tv ? (m[h] == -INFINITY ? 0.0f : expf(m[h] - mn)) : 1.0f;
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t nt = 0; nt < 4u; nt++) {
                #pragma unroll
                for (uint32_t e = 0; e < 2u; e++) {
                    const float p = tv ? expf(s[nt][h * 2u + e] - mn) : 0.0f;
                    s[nt][h * 2u + e] = p;
                    sum += p;
                }
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            l[h] = l[h] * alpha + sum;
            m[h] = mn;
            #pragma unroll
            for (uint32_t t = 0; t < 32u; t++) {
                o[t][h * 2u + 0u] *= alpha;
                o[t][h * 2u + 1u] *= alpha;
            }
        }

        /* O += P V on Tensor Cores (m16n8k16). */
        #pragma unroll
        for (uint32_t kk = 0; kk < 2u; kk++) {
            const uint32_t pa[4] = {
                spark_mma_pack_h(s[2u * kk][0], s[2u * kk][1]),
                spark_mma_pack_h(s[2u * kk][2], s[2u * kk][3]),
                spark_mma_pack_h(s[2u * kk + 1u][0], s[2u * kk + 1u][1]),
                spark_mma_pack_h(s[2u * kk + 1u][2], s[2u * kk + 1u][3])};
            #pragma unroll
            for (uint32_t t = 0; t < 32u; t += 2u) {
                uint32_t b[4];
                const uint32_t mat = lane >> 3u;
                const uint32_t krow = kk * 16u + (mat & 1u) * 8u + (lane & 7u);
                const uint32_t chunk = t + (mat >> 1u);
                spark_mma_ldsm4(b, vs + spark_mma_swz(krow, chunk, D), true);
                spark_mma_mma_h(o[t], pa, b[0], b[1]);
                spark_mma_mma_h(o[t + 1u], pa, b[2], b[3]);
            }
        }
        __syncthreads();
    }

    const float inv[2] = {l[0] > 0.0f ? 1.0f / l[0] : 0.0f,
                          l[1] > 0.0f ? 1.0f / l[1] : 0.0f};
    #pragma unroll
    for (uint32_t h = 0; h < 2u; h++) {
        const uint32_t mrow = g + 8u * h;
        const uint32_t token = row0 + warp * 4u + (mrow >> 2u);
        if (token >= rows) { continue; }
        const uint32_t head = kv_head * GROUP + (mrow & 3u);
        float *dst = out + ((uint64_t)token * heads + head) * D;
        #pragma unroll
        for (uint32_t t = 0; t < 32u; t++) {
            const uint32_t d = t * 8u + 2u * q;
            dst[d] = o[t][h * 2u + 0u] * inv[h];
            dst[d + 1u] = o[t][h * 2u + 1u] * inv[h];
        }
    }
}

#endif
