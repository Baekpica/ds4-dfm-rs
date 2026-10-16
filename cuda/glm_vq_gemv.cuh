#ifndef DS4_GLM_VQ_GEMV_CUH
#define DS4_GLM_VQ_GEMV_CUH

#include "glm_vq_ops.cuh"

/* N=1 decode projections over BF16 weights. The generic path converts the
 * input to a BF16 scratch row, runs a cuBLAS GEMV and converts the BF16
 * output back: three launches, and cuBLAS reached only 147-178 GB/s on
 * these shapes (177 GB/s for the 1.27 GB output head). One pass reads the
 * F32 row in place. Products are exact in FP32 either way (BF16 x BF16, or
 * BF16 x F32 for the F32 router/mHC math); only the FP32 accumulation order
 * differs from cuBLAS, and outputs keep their BF16 (or F32) boundary. */
enum glmvq_dot_math { GLMVQ_DOT_BF16, GLMVQ_DOT_F32 };
enum { GLMVQ_DOT_WARPS = 8, GLMVQ_DOT_VECTOR = 8, GLMVQ_DOT_SPLIT = 8,
       GLMVQ_DOT_WIDE_ROWS = 4096, GLMVQ_DOT_SPLIT_COLS = 2048, GLMVQ_DOT_TOKENS = 4 };

__device__ __forceinline__ float glmvq_bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float glmvq_bf16_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

template <glmvq_dot_math MATH>
__device__ __forceinline__ float glmvq_dot8(uint4 w, float4 a, float4 b, float sum) {
    if constexpr (MATH == GLMVQ_DOT_BF16) {
        a = make_float4(glmvq_round(a.x), glmvq_round(a.y), glmvq_round(a.z), glmvq_round(a.w));
        b = make_float4(glmvq_round(b.x), glmvq_round(b.y), glmvq_round(b.z), glmvq_round(b.w));
    }
    sum = fmaf(glmvq_bf16_lo(w.x), a.x, sum); sum = fmaf(glmvq_bf16_hi(w.x), a.y, sum);
    sum = fmaf(glmvq_bf16_lo(w.y), a.z, sum); sum = fmaf(glmvq_bf16_hi(w.y), a.w, sum);
    sum = fmaf(glmvq_bf16_lo(w.z), b.x, sum); sum = fmaf(glmvq_bf16_hi(w.z), b.y, sum);
    sum = fmaf(glmvq_bf16_lo(w.w), b.z, sum); sum = fmaf(glmvq_bf16_hi(w.w), b.w, sum);
    return sum;
}

// `split` warps share a row (few-row matrices); the partials add in warp order.
// MTP verify rows (NT up to 4) reuse each weight load; every token keeps the
// one-token FMA and reduction order.
template <glmvq_dot_math MATH, uint32_t NT>
__global__ void __launch_bounds__(GLMVQ_DOT_WARPS * 32) glmvq_bf16_dot(
        const __nv_bfloat16 *weight, const float *input, float *output,
        uint32_t rows, uint32_t cols, uint32_t split, uint32_t tokens) {
    __shared__ float partial[GLMVQ_DOT_WARPS][NT];
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const uint32_t local = warp / split, part = warp % split;
    const uint32_t row = blockIdx.x * (GLMVQ_DOT_WARPS / split) + local;
    float sum[NT] = {};
    if (row < rows) {
        const __nv_bfloat16 *w = weight + (uint64_t)row * cols;
        const uint32_t step = split * 32 * GLMVQ_DOT_VECTOR;
        #pragma unroll 4
        for (uint32_t col = (part * 32 + lane) * GLMVQ_DOT_VECTOR; col < cols; col += step) {
            const uint4 value = *(const uint4 *)(w + col);
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                if (t >= tokens) { continue; }
                const float *x = input + (uint64_t)t * cols + col;
                sum[t] = glmvq_dot8<MATH>(value, *(const float4 *)x, *(const float4 *)(x + 4), sum[t]);
            }
        }
    }
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        for (uint32_t stride = 16; stride; stride >>= 1) { sum[t] += __shfl_xor_sync(0xffffffffu, sum[t], stride); }
        if (!lane) { partial[warp][t] = sum[t]; }
    }
    __syncthreads();
    if (row >= rows || part || lane) { return; }
    for (uint32_t t = 0; t < tokens; t++) {
        float value = 0;
        for (uint32_t i = 0; i < split; i++) { value += partial[local * split + i][t]; }
        output[(uint64_t)t * rows + row] = MATH == GLMVQ_DOT_BF16 ? glmvq_round(value) : value;
    }
}

static bool glmvq_bf16_fits(const void *weight, const void *input, const void *output, uint32_t cols) {
    return weight && input && output && cols && cols % GLMVQ_DOT_VECTOR == 0 &&
        (uintptr_t)weight % 16 == 0 && (uintptr_t)input % 16 == 0 && (uintptr_t)output % 4 == 0;
}

template <glmvq_dot_math MATH, uint32_t NT>
static void glmvq_bf16_launch(const void *weight, const float *input, float *output,
        uint32_t rows, uint32_t cols, uint32_t split, uint32_t tokens, cudaStream_t stream) {
    const uint32_t grid = (rows + GLMVQ_DOT_WARPS / split - 1) / (GLMVQ_DOT_WARPS / split);
    glmvq_bf16_dot<MATH, NT><<<grid, GLMVQ_DOT_WARPS * 32, 0, stream>>>(
        (const __nv_bfloat16 *)weight, input, output, rows, cols, split, tokens);
}

static cudaError_t glmvq_bf16_decode(const void *weight, const float *input, float *output,
        uint32_t rows, uint32_t cols, glmvq_dot_math math, cudaStream_t stream, uint32_t tokens = 1) {
    if (!rows || !tokens || tokens > GLMVQ_DOT_TOKENS || !glmvq_bf16_fits(weight, input, output, cols)) {
        return cudaErrorInvalidValue;
    }
    // Few long rows split across a block's warps so every SM has work.
    const uint32_t split = rows < GLMVQ_DOT_WIDE_ROWS && cols >= GLMVQ_DOT_SPLIT_COLS ? GLMVQ_DOT_SPLIT : 1;
    if (math == GLMVQ_DOT_F32 && tokens == 1) {
        glmvq_bf16_launch<GLMVQ_DOT_F32, 1>(weight, input, output, rows, cols, split, tokens, stream);
    } else if (math == GLMVQ_DOT_F32) {
        glmvq_bf16_launch<GLMVQ_DOT_F32, GLMVQ_DOT_TOKENS>(weight, input, output, rows, cols, split, tokens, stream);
    } else if (tokens == 1) {
        glmvq_bf16_launch<GLMVQ_DOT_BF16, 1>(weight, input, output, rows, cols, split, tokens, stream);
    } else {
        glmvq_bf16_launch<GLMVQ_DOT_BF16, GLMVQ_DOT_TOKENS>(weight, input, output, rows, cols, split, tokens, stream);
    }
    return cudaGetLastError();
}

/* Decode mHC mix (source 286-329): F32 RMS of the 4-stream residual, then
 * the 24 F32 mapping dots. Block o owns output o and recomputes the RMS in
 * glmvq_rms's exact order (same normalized values), so the flat row is never
 * written or reread; four launches become one. */
__global__ void __launch_bounds__(GLMVQ_THREADS) glmvq_hc_mix(const float *residual,
        const __nv_bfloat16 *weight, float *mix, uint32_t width, float eps) {
    __shared__ float partial[GLMVQ_THREADS];
    __shared__ float total[GLMVQ_THREADS / 32];
    const uint32_t lane = threadIdx.x;
    // MTP verify rows: blockIdx.y selects the row; per-row math is unchanged.
    residual += (uint64_t)blockIdx.y * width;
    mix += (uint64_t)blockIdx.y * gridDim.x;
    float sum = 0.0f;
    for (uint32_t first = lane; first < width; first += GLMVQ_RMS_BATCH * blockDim.x) {
        float x[GLMVQ_RMS_BATCH];
        #pragma unroll
        for (uint32_t j = 0; j < GLMVQ_RMS_BATCH; j++) {
            const uint32_t col = first + j * blockDim.x;
            x[j] = col < width ? residual[col] : 0.0f;
        }
        #pragma unroll
        for (uint32_t j = 0; j < GLMVQ_RMS_BATCH; j++) {
            if (first + j * blockDim.x < width) { sum = glmvq_add(sum, glmvq_mul(x[j], x[j])); }
        }
    }
    partial[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (lane < stride) { partial[lane] = glmvq_add(partial[lane], partial[lane + stride]); }
        __syncthreads();
    }
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(glmvq_add(__fdiv_rn(partial[0], (float)width), eps)));
    const __nv_bfloat16 *w = weight + (uint64_t)blockIdx.x * width;
    float dot = 0.0f;
    #pragma unroll 4
    for (uint32_t col = lane * GLMVQ_DOT_VECTOR; col < width; col += blockDim.x * GLMVQ_DOT_VECTOR) {
        float4 a = *(const float4 *)(residual + col), b = *(const float4 *)(residual + col + 4);
        a = make_float4(glmvq_mul(a.x, inv), glmvq_mul(a.y, inv), glmvq_mul(a.z, inv), glmvq_mul(a.w, inv));
        b = make_float4(glmvq_mul(b.x, inv), glmvq_mul(b.y, inv), glmvq_mul(b.z, inv), glmvq_mul(b.w, inv));
        dot = glmvq_dot8<GLMVQ_DOT_F32>(*(const uint4 *)(w + col), a, b, dot);
    }
    for (uint32_t stride = 16; stride; stride >>= 1) { dot += __shfl_xor_sync(0xffffffffu, dot, stride); }
    if (lane % 32 == 0) { total[lane / 32] = dot; }
    __syncthreads();
    if (lane) { return; }
    dot = 0.0f;
    for (uint32_t i = 0; i < GLMVQ_THREADS / 32; i++) { dot += total[i]; }
    mix[blockIdx.x] = dot;
}

/* Prefill mHC mix: the 16K-wide RMS (written out) plus a cuBLAS F32 GEMM
 * with 24 outputs read every residual row twice and wrote it once, 0.9 s of
 * an 8K prefill. One pass now accumulates each row's square sum and its 24
 * raw dots, then scales by the RMS reciprocal: mix = inv * Σ W x instead of
 * Σ W (x * inv), an F32 reordering. Warp w of a block owns row w; each
 * 512-column W tile is staged once per block. */
enum { GLMVQ_MIX_TILE = 512, GLMVQ_MIX_WARPS = 8 };

__global__ void __launch_bounds__(GLMVQ_MIX_WARPS * 32) glmvq_hc_mix_rows(const float *residual,
        const __nv_bfloat16 *weight, float *mix, uint32_t rows, uint32_t width, float eps) {
    __shared__ __align__(16) __nv_bfloat16 tile[GLMVQ_HC_MIX * GLMVQ_MIX_TILE];
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const uint32_t row = blockIdx.x * GLMVQ_MIX_WARPS + warp;
    const float *x = residual + (uint64_t)min(row, rows - 1u) * width;
    float dot[GLMVQ_HC_MIX] = {}, squares = 0.0f;
    for (uint32_t base = 0; base < width; base += GLMVQ_MIX_TILE) {
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < GLMVQ_HC_MIX * GLMVQ_MIX_TILE / 8; i += blockDim.x) {
            const uint32_t o = i / (GLMVQ_MIX_TILE / 8), c = i % (GLMVQ_MIX_TILE / 8) * 8;
            *(uint4 *)(tile + o * GLMVQ_MIX_TILE + c) = *(const uint4 *)(weight + (uint64_t)o * width + base + c);
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t k = 0; k < GLMVQ_MIX_TILE / 128; k++) {
            const uint32_t c = k * 128 + lane * 4;
            const float4 v = *(const float4 *)(x + base + c);
            squares = fmaf(v.x, v.x, squares); squares = fmaf(v.y, v.y, squares);
            squares = fmaf(v.z, v.z, squares); squares = fmaf(v.w, v.w, squares);
            #pragma unroll
            for (uint32_t o = 0; o < GLMVQ_HC_MIX; o++) {
                const uint2 w = *(const uint2 *)(tile + o * GLMVQ_MIX_TILE + c);
                dot[o] = fmaf(glmvq_bf16_lo(w.x), v.x, dot[o]); dot[o] = fmaf(glmvq_bf16_hi(w.x), v.y, dot[o]);
                dot[o] = fmaf(glmvq_bf16_lo(w.y), v.z, dot[o]); dot[o] = fmaf(glmvq_bf16_hi(w.y), v.w, dot[o]);
            }
        }
    }
    for (uint32_t stride = 16; stride; stride >>= 1) { squares += __shfl_xor_sync(0xffffffffu, squares, stride); }
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(glmvq_add(__fdiv_rn(squares, (float)width), eps)));
    #pragma unroll
    for (uint32_t o = 0; o < GLMVQ_HC_MIX; o++) {
        for (uint32_t stride = 16; stride; stride >>= 1) { dot[o] += __shfl_xor_sync(0xffffffffu, dot[o], stride); }
    }
    if (lane || row >= rows) { return; }
    #pragma unroll
    for (uint32_t o = 0; o < GLMVQ_HC_MIX; o++) { mix[(uint64_t)row * GLMVQ_HC_MIX + o] = glmvq_mul(dot[o], inv); }
}

/* Prefill mHC expand plus the following mix in one pass. Each warp expands
 * its row with glmvq_hc_expand's arithmetic and folds the new four-stream
 * values into the RMS and 24 dots (as glmvq_hc_mix_rows, in another F32
 * order), so the residual it writes is not read back for the mix. A W tile
 * covers 128 columns of all four streams. */
enum { GLMVQ_XMIX_COLS = 128 };

__global__ void __launch_bounds__(GLMVQ_MIX_WARPS * 32) glmvq_hc_expand_mix(const float *branch,
        const float *residual, const float *split, float *output, const __nv_bfloat16 *weight,
        float *mix, uint32_t rows, uint32_t width, float eps) {
    __shared__ __align__(16) __nv_bfloat16 tile[GLMVQ_HC_MIX * GLMVQ_HC * GLMVQ_XMIX_COLS];
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const uint32_t row = blockIdx.x * GLMVQ_MIX_WARPS + warp, rr = min(row, rows - 1u);
    const uint32_t flat = GLMVQ_HC * width;
    const float *coeff = split + (uint64_t)rr * GLMVQ_HC_MIX;
    float post[GLMVQ_HC], comb[GLMVQ_HC][GLMVQ_HC];
    #pragma unroll
    for (uint32_t d = 0; d < GLMVQ_HC; d++) {
        post[d] = glmvq_round(coeff[GLMVQ_HC + d]);
        #pragma unroll
        for (uint32_t src = 0; src < GLMVQ_HC; src++) { comb[src][d] = glmvq_round(coeff[2 * GLMVQ_HC + src * GLMVQ_HC + d]); }
    }
    float dot[GLMVQ_HC_MIX] = {}, squares = 0.0f;
    for (uint32_t base = 0; base < width; base += GLMVQ_XMIX_COLS) {
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < GLMVQ_HC_MIX * GLMVQ_HC * GLMVQ_XMIX_COLS / 8; i += blockDim.x) {
            const uint32_t line = i / (GLMVQ_XMIX_COLS / 8), c = i % (GLMVQ_XMIX_COLS / 8) * 8;
            const uint32_t o = line / GLMVQ_HC, h = line % GLMVQ_HC;
            *(uint4 *)(tile + line * GLMVQ_XMIX_COLS + c) =
                *(const uint4 *)(weight + (uint64_t)o * flat + h * width + base + c);
        }
        __syncthreads();
        const uint32_t c = base + lane * 4;
        float4 res[GLMVQ_HC];
        #pragma unroll
        for (uint32_t src = 0; src < GLMVQ_HC; src++) {
            res[src] = *(const float4 *)(residual + ((uint64_t)rr * GLMVQ_HC + src) * width + c);
        }
        const float4 b = *(const float4 *)(branch + (uint64_t)rr * width + c);
        #pragma unroll
        for (uint32_t d = 0; d < GLMVQ_HC; d++) {
            float v[4];
            #pragma unroll
            for (uint32_t e = 0; e < 4; e++) {
                float sum = 0.0f;
                #pragma unroll
                for (uint32_t src = 0; src < GLMVQ_HC; src++) {
                    sum = glmvq_add(sum, glmvq_mul(comb[src][d], (&res[src].x)[e]));
                }
                const float part = glmvq_round(glmvq_mul(post[d], (&b.x)[e]));
                v[e] = glmvq_round(glmvq_add(part, glmvq_round(sum)));
                squares = fmaf(v[e], v[e], squares);
            }
            if (row < rows) {
                *(float4 *)(output + ((uint64_t)row * GLMVQ_HC + d) * width + c) = make_float4(v[0], v[1], v[2], v[3]);
            }
            #pragma unroll
            for (uint32_t o = 0; o < GLMVQ_HC_MIX; o++) {
                const uint2 w = *(const uint2 *)(tile + (o * GLMVQ_HC + d) * GLMVQ_XMIX_COLS + lane * 4);
                dot[o] = fmaf(glmvq_bf16_lo(w.x), v[0], dot[o]); dot[o] = fmaf(glmvq_bf16_hi(w.x), v[1], dot[o]);
                dot[o] = fmaf(glmvq_bf16_lo(w.y), v[2], dot[o]); dot[o] = fmaf(glmvq_bf16_hi(w.y), v[3], dot[o]);
            }
        }
    }
    for (uint32_t stride = 16; stride; stride >>= 1) { squares += __shfl_xor_sync(0xffffffffu, squares, stride); }
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(glmvq_add(__fdiv_rn(squares, (float)flat), eps)));
    #pragma unroll
    for (uint32_t o = 0; o < GLMVQ_HC_MIX; o++) {
        for (uint32_t stride = 16; stride; stride >>= 1) { dot[o] += __shfl_xor_sync(0xffffffffu, dot[o], stride); }
    }
    if (lane || row >= rows) { return; }
    #pragma unroll
    for (uint32_t o = 0; o < GLMVQ_HC_MIX; o++) { mix[(uint64_t)row * GLMVQ_HC_MIX + o] = glmvq_mul(dot[o], inv); }
}

#endif
