#ifndef DS4_GLM_VQ_OPS_CUH
#define DS4_GLM_VQ_OPS_CUH

#include "glm_vq.cuh"

/* Source arithmetic: Transformers 5.19.0 modeling_glm5_next.py,
 * lines 66-105, 286-337, 375-394 and 1300-1352. F32 graph buffers hold
 * BF16-rounded activations, except the explicit mHC/router F32 paths. */
enum glmvq_norm { GLMVQ_RMS_WEIGHTED, GLMVQ_RMS_GATED, GLMVQ_RMS_PLAIN };
enum { GLMVQ_HC = 4, GLMVQ_HC_MIX = GLMVQ_HC * (GLMVQ_HC + 2) };

__device__ __forceinline__ float glmvq_add(float a, float b) {
    float value;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(value) : "f"(a), "f"(b));
    return value;
}

__device__ __forceinline__ float glmvq_round(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float glmvq_sigmoid(float value) {
    return __fdiv_rn(1.0f, glmvq_add(1.0f, expf(-value)));
}

__global__ void glmvq_round_rows(float *values, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { values[i] = glmvq_round(values[i]); }
}

enum { GLMVQ_RMS_BATCH = 16, GLMVQ_RMS_NARROW = 512, GLMVQ_RMS_ROWS = 8 };

// Per-element tail shared by the block and warp RMS kernels.
__device__ __forceinline__ float glmvq_rms_value(float x, float inv, const __nv_bfloat16 *weight,
        const float *gate, uint64_t at, uint32_t col, glmvq_norm kind) {
    float value = glmvq_mul(x, inv);
    if (kind == GLMVQ_RMS_PLAIN) { return value; }
    // Ordinary RMS rounds before the weight; gated RMS rounds only last.
    if (kind == GLMVQ_RMS_WEIGHTED) { value = glmvq_round(value); }
    value = glmvq_mul(value, __bfloat162float(weight[col]));
    if (kind == GLMVQ_RMS_GATED) { value = glmvq_mul(value, glmvq_sigmoid(gate[at])); }
    return glmvq_round(value);
}

__global__ void glmvq_rms(const float *input, const __nv_bfloat16 *weight,
        const float *gate, float *output, uint32_t width, float eps, glmvq_norm kind) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint64_t base = (uint64_t)blockIdx.x * width;
    const uint32_t lane = threadIdx.x;
    float sum = 0.0f;
    // A decode row is one 256-thread block; one load per dependent add left
    // ~14 us of exposed latency on 16K mHC rows. Batch the loads, but keep
    // each thread's col, col+256, ... accumulation order bit-identical.
    for (uint32_t first = lane; first < width; first += GLMVQ_RMS_BATCH * blockDim.x) {
        float x[GLMVQ_RMS_BATCH];
        #pragma unroll
        for (uint32_t j = 0; j < GLMVQ_RMS_BATCH; j++) {
            const uint32_t col = first + j * blockDim.x;
            x[j] = col < width ? input[base + col] : 0.0f;
        }
        #pragma unroll
        for (uint32_t j = 0; j < GLMVQ_RMS_BATCH; j++) {
            if (first + j * blockDim.x < width) { sum = glmvq_add(sum, glmvq_mul(x[j], x[j])); }
        }
    }
    __shared__ float partial[GLMVQ_THREADS];
    partial[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (lane < stride) { partial[lane] = glmvq_add(partial[lane], partial[lane + stride]); }
        __syncthreads();
    }
    const float variance = glmvq_add(__fdiv_rn(partial[0], (float)width), eps);
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(variance));
    for (uint32_t first = lane; first < width; first += GLMVQ_RMS_BATCH * blockDim.x) {
        float x[GLMVQ_RMS_BATCH];
        #pragma unroll
        for (uint32_t j = 0; j < GLMVQ_RMS_BATCH; j++) {
            const uint32_t col = first + j * blockDim.x;
            x[j] = col < width ? input[base + col] : 0.0f;
        }
        #pragma unroll
        for (uint32_t j = 0; j < GLMVQ_RMS_BATCH; j++) {
            const uint32_t col = first + j * blockDim.x;
            if (col >= width) { break; }
            output[base + col] = glmvq_rms_value(x[j], inv, weight, gate, base + col, col, kind);
        }
    }
}

/* Narrow rows (KDA heads are 128 wide, rows x 64 per call): a warp per row.
 * The block kernel left half of its 256 lanes idle and paid two barriers
 * per tree level (0.2 s of an 8K prefill). FP32 add order differs. */
__global__ void __launch_bounds__(GLMVQ_RMS_ROWS * 32) glmvq_rms_warp(const float *input,
        const __nv_bfloat16 *weight, const float *gate, float *output, uint32_t rows,
        uint32_t width, float eps, glmvq_norm kind) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t lane = threadIdx.x & 31, row = blockIdx.x * GLMVQ_RMS_ROWS + threadIdx.x / 32;
    if (row >= rows) { return; }
    const uint64_t base = (uint64_t)row * width;
    // All of a lane's loads issue before the dependent adds (decode rows).
    float x[GLMVQ_RMS_NARROW / 32];
    #pragma unroll
    for (uint32_t j = 0; j < GLMVQ_RMS_NARROW / 32; j++) {
        const uint32_t col = lane + j * 32;
        x[j] = col < width ? input[base + col] : 0.0f;
    }
    float sum = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < GLMVQ_RMS_NARROW / 32; j++) { sum = glmvq_add(sum, glmvq_mul(x[j], x[j])); }
    for (uint32_t stride = 16; stride; stride >>= 1) { sum = glmvq_add(sum, __shfl_xor_sync(0xffffffffu, sum, stride)); }
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(glmvq_add(__fdiv_rn(sum, (float)width), eps)));
    #pragma unroll
    for (uint32_t j = 0; j < GLMVQ_RMS_NARROW / 32; j++) {
        const uint32_t col = lane + j * 32;
        if (col < width) { output[base + col] = glmvq_rms_value(x[j], inv, weight, gate, base + col, col, kind); }
    }
}

__device__ __forceinline__ float glmvq_swiglu_value(float gate, float up, float limit) {
    const float g = fminf(gate, limit);
    const float u = fminf(fmaxf(up, -limit), limit);
    const float silu = glmvq_round(__fdiv_rn(g, glmvq_add(1.0f, expf(-g))));
    return glmvq_round(glmvq_mul(silu, u));
}

__global__ void glmvq_swiglu(const float *gate, const float *up, float *output,
        uint64_t count, float limit) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { output[i] = glmvq_swiglu_value(gate[i], up[i], limit); }
}

__device__ void glmvq_hc_weights(float *split, const float *mix,
        const float *scale, const float *base, uint32_t iters, float eps) {
    for (uint32_t h = 0; h < GLMVQ_HC; h++) {
        split[h] = glmvq_add(glmvq_sigmoid(glmvq_add(glmvq_mul(mix[h], scale[0]), base[h])), eps);
        const uint32_t p = GLMVQ_HC + h;
        split[p] = glmvq_mul(2.0f, glmvq_sigmoid(glmvq_add(glmvq_mul(mix[p], scale[1]), base[p])));
    }
    float *comb = split + 2 * GLMVQ_HC;
    for (uint32_t row = 0; row < GLMVQ_HC; row++) {
        float max_value = -INFINITY, sum = 0.0f;
        for (uint32_t col = 0; col < GLMVQ_HC; col++) {
            const uint32_t i = row * GLMVQ_HC + col;
            comb[i] = glmvq_add(glmvq_mul(mix[2 * GLMVQ_HC + i], scale[2]), base[2 * GLMVQ_HC + i]);
            max_value = fmaxf(max_value, comb[i]);
        }
        for (uint32_t col = 0; col < GLMVQ_HC; col++) {
            const uint32_t i = row * GLMVQ_HC + col;
            comb[i] = expf(glmvq_add(comb[i], -max_value));
            sum = glmvq_add(sum, comb[i]);
        }
        for (uint32_t col = 0; col < GLMVQ_HC; col++) {
            const uint32_t i = row * GLMVQ_HC + col;
            comb[i] = glmvq_add(__fdiv_rn(comb[i], sum), eps);
        }
    }
    // Source does the first column normalization before the remaining pairs.
    for (uint32_t step = 0; step < iters; step++) {
        if (step) {
            for (uint32_t row = 0; row < GLMVQ_HC; row++) {
                float sum = 0.0f;
                for (uint32_t col = 0; col < GLMVQ_HC; col++) {
                    sum = glmvq_add(sum, comb[row * GLMVQ_HC + col]);
                }
                sum = glmvq_add(sum, eps);
                for (uint32_t col = 0; col < GLMVQ_HC; col++) {
                    comb[row * GLMVQ_HC + col] = __fdiv_rn(comb[row * GLMVQ_HC + col], sum);
                }
            }
        }
        for (uint32_t col = 0; col < GLMVQ_HC; col++) {
            float sum = 0.0f;
            for (uint32_t row = 0; row < GLMVQ_HC; row++) {
                sum = glmvq_add(sum, comb[row * GLMVQ_HC + col]);
            }
            sum = glmvq_add(sum, eps);
            for (uint32_t row = 0; row < GLMVQ_HC; row++) {
                comb[row * GLMVQ_HC + col] = __fdiv_rn(comb[row * GLMVQ_HC + col], sum);
            }
        }
    }
}

// Folds a 4-lane group in ascending lane order from `init`; every member of
// the group gets the serial loop's exact value.
template <typename F>
__device__ __forceinline__ float glmvq_hc_fold(float value, float init,
        uint32_t first, uint32_t step, F op) {
    float acc = init;
    #pragma unroll
    for (uint32_t i = 0; i < GLMVQ_HC; i++) {
        acc = op(acc, __shfl_sync(0xffffffffu, value, first + i * step));
    }
    return acc;
}

// glmvq_hc_weights with one lane per Sinkhorn element: lanes 0-15 own
// comb[row*4+col], lanes 16-23 own the pre/post gates. Each element sees the
// same operations in the same order, so the coefficients are bit-identical;
// one thread spent ~20 us per decode call on the serial divisions.
__device__ void glmvq_hc_weights_warp(float *coeff, const float *mix,
        const float *scale, const float *base, uint32_t iters, float eps) {
    const uint32_t lane = threadIdx.x & 31, i = lane & 15u;
    const uint32_t row = i / GLMVQ_HC, col = i % GLMVQ_HC;
    const auto add = [](float a, float b) { return glmvq_add(a, b); };
    const auto max = [](float a, float b) { return fmaxf(a, b); };
    float c = glmvq_add(glmvq_mul(mix[2 * GLMVQ_HC + i], scale[2]), base[2 * GLMVQ_HC + i]);
    const float peak = glmvq_hc_fold(c, -INFINITY, row * GLMVQ_HC, 1, max);
    c = expf(glmvq_add(c, -peak));
    c = glmvq_add(__fdiv_rn(c, glmvq_hc_fold(c, 0.0f, row * GLMVQ_HC, 1, add)), eps);
    // Source does the first column normalization before the remaining pairs.
    for (uint32_t step = 0; step < iters; step++) {
        if (step) {
            c = __fdiv_rn(c, glmvq_add(glmvq_hc_fold(c, 0.0f, row * GLMVQ_HC, 1, add), eps));
        }
        c = __fdiv_rn(c, glmvq_add(glmvq_hc_fold(c, 0.0f, col, GLMVQ_HC, add), eps));
    }
    if (lane < GLMVQ_HC * GLMVQ_HC) { coeff[2 * GLMVQ_HC + lane] = c; }
    const uint32_t h = lane - GLMVQ_HC * GLMVQ_HC;
    if (h < GLMVQ_HC) {
        coeff[h] = glmvq_add(glmvq_sigmoid(glmvq_add(glmvq_mul(mix[h], scale[0]), base[h])), eps);
    } else if (h < 2 * GLMVQ_HC) {
        coeff[h] = glmvq_mul(2.0f, glmvq_sigmoid(glmvq_add(glmvq_mul(mix[h], scale[1]), base[h])));
    }
}

enum { GLMVQ_HC_BATCH = 4 };

// With `norm`, the block also applies the following ordinary RMS norm to
// its own output row: each thread squares its columns in glmvq_rms's order
// and reduces with the same tree, so `norm` is bit-identical to a separate
// glmvq_rms launch without that launch or the extra row round trip.
template <typename R>
__global__ void glmvq_hc_split(const float *mix, const R *residual,
        const float *scale, const float *base, float *output, float *split,
        uint32_t width, uint32_t iters, float eps,
        const __nv_bfloat16 *norm_weight, float *norm, float norm_eps) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t row = blockIdx.x;
    __shared__ float coeff[GLMVQ_HC_MIX];
    __shared__ float partial[GLMVQ_THREADS];
    if (threadIdx.x < 32) {
        glmvq_hc_weights_warp(coeff, mix + (uint64_t)row * GLMVQ_HC_MIX, scale, base, iters, eps);
    }
    __syncthreads();
    if (threadIdx.x < GLMVQ_HC_MIX) {
        split[(uint64_t)row * GLMVQ_HC_MIX + threadIdx.x] = coeff[threadIdx.x];
    }
    const R *r = residual + (uint64_t)row * GLMVQ_HC * width;
    float *out = output + (uint64_t)row * width;
    float squares = 0.0f;
    // Batched loads expose several rows of latency at once; per-column order is unchanged.
    for (uint32_t first = threadIdx.x; first < width; first += GLMVQ_HC_BATCH * blockDim.x) {
        float x[GLMVQ_HC_BATCH][GLMVQ_HC];
        #pragma unroll
        for (uint32_t j = 0; j < GLMVQ_HC_BATCH; j++) {
            const uint32_t col = first + j * blockDim.x;
            #pragma unroll
            for (uint32_t h = 0; h < GLMVQ_HC; h++) {
                x[j][h] = col < width ? glmvq_value(r[(uint64_t)h * width + col]) : 0.0f;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < GLMVQ_HC_BATCH; j++) {
            const uint32_t col = first + j * blockDim.x;
            if (col >= width) { break; }
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t h = 0; h < GLMVQ_HC; h++) { sum = glmvq_add(sum, glmvq_mul(coeff[h], x[j][h])); }
            const float value = glmvq_round(sum);
            out[col] = value;
            squares = glmvq_add(squares, glmvq_mul(value, value));
        }
    }
    if (!norm) { return; }
    partial[threadIdx.x] = squares;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) { partial[threadIdx.x] = glmvq_add(partial[threadIdx.x], partial[threadIdx.x + stride]); }
        __syncthreads();
    }
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(glmvq_add(__fdiv_rn(partial[0], (float)width), norm_eps)));
    // Each thread rereads only the columns it wrote above, batched like glmvq_rms.
    for (uint32_t first = threadIdx.x; first < width; first += GLMVQ_RMS_BATCH * blockDim.x) {
        float x[GLMVQ_RMS_BATCH];
        #pragma unroll
        for (uint32_t j = 0; j < GLMVQ_RMS_BATCH; j++) {
            const uint32_t col = first + j * blockDim.x;
            x[j] = col < width ? out[col] : 0.0f;
        }
        #pragma unroll
        for (uint32_t j = 0; j < GLMVQ_RMS_BATCH; j++) {
            const uint32_t col = first + j * blockDim.x;
            if (col >= width) { break; }
            const float value = glmvq_round(glmvq_mul(x[j], inv));
            norm[(uint64_t)row * width + col] = glmvq_round(glmvq_mul(value, __bfloat162float(norm_weight[col])));
        }
    }
}

template <typename R>
__global__ void glmvq_hc_expand(const float *branch, const R *residual,
        const float *split, float *output, uint32_t width, uint64_t count) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const uint32_t col = i % width, dst = (i / width) % GLMVQ_HC;
    const uint64_t row = i / (width * GLMVQ_HC);
    const float *coeff = split + row * GLMVQ_HC_MIX;
    float sum = 0.0f;
    for (uint32_t src = 0; src < GLMVQ_HC; src++) {
        const float c = glmvq_round(coeff[2 * GLMVQ_HC + src * GLMVQ_HC + dst]);
        sum = glmvq_add(sum, glmvq_mul(c, glmvq_value(residual[(row * GLMVQ_HC + src) * width + col])));
    }
    const float part = glmvq_round(glmvq_mul(glmvq_round(coeff[GLMVQ_HC + dst]), branch[row * width + col]));
    output[i] = glmvq_round(glmvq_add(part, glmvq_round(sum)));
}

#endif
