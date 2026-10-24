#ifndef DS4_GLM_VQ_Q4_CUH
#define DS4_GLM_VQ_Q4_CUH

#include "q4_k.cuh"
#include <cuda_pipeline.h>

enum { GLMVQ_Q4_BLOCK = DS4_Q4_K_BLOCK, GLMVQ_Q4_BYTES = DS4_Q4_K_BYTES, GLMVQ_Q4_WARPS = 8 };
enum glmvq_q4_load { GLMVQ_Q4_SCALAR, GLMVQ_Q4_STAGE, GLMVQ_Q4_PIPE, GLMVQ_Q4_AUTO };
struct glmvq_q4_config {
    glmvq_q4_load mode;
    uint32_t blocks, shared, major;
    glmvq_launch launch;
};

// Q4_K layout/dot mapping: YoungAi 7fb26cc, cuda_v41_q4k.inc.cu:21-70.
// Unlike GVQ1 these weights use local affine groups, without a codebook.
template <typename T>
__global__ void glmvq_q4_restore(const uint8_t *weight, T *out, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        out[i] = glmvq_cast<T>(ds4_q4_k_value(weight + i / GLMVQ_Q4_BLOCK * GLMVQ_Q4_BYTES,
            i % GLMVQ_Q4_BLOCK));
    }
}

template <uint32_t NT>
__global__ void glmvq_q4_small(const uint8_t *weight, const float *input,
        float *out, uint32_t cols, uint32_t rows, uint32_t tokens, uint32_t split) {
    __shared__ float partial[GLMVQ_Q4_WARPS * NT];
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint32_t row = blockIdx.x * (GLMVQ_Q4_WARPS / split) + warp / split;
    const uint32_t part = warp % split, blocks = cols / GLMVQ_Q4_BLOCK;
    float sum[NT] = {};
    if (row < rows) {
        for (uint32_t b = part; b < blocks; b += split) {
            const uint8_t *packed = weight + ((uint64_t)row * blocks + b) * GLMVQ_Q4_BYTES;
            // Each warp covers one block; independent rows expose enough
            // parallelism for a single token without global BF16 restoration.
            #pragma unroll
            for (uint32_t j = 0; j < GLMVQ_Q4_BLOCK; j += 32) {
                const float w = ds4_q4_k_value(packed, j + lane);
                #pragma unroll
                for (uint32_t t = 0; t < NT; t++) {
                    if (t < tokens) {
                        sum[t] = fmaf(w, input[(uint64_t)t * cols + b * GLMVQ_Q4_BLOCK + j + lane], sum[t]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        for (int stride = 16; stride; stride >>= 1) {
            sum[t] += __shfl_xor_sync(0xffffffffu, sum[t], stride);
        }
        if (!lane) { partial[warp * NT + t] = sum[t]; }
    }
    __syncthreads();
    if (row >= rows || lane || part) { return; }
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        if (t >= tokens) { continue; }
        float value = 0;
        for (uint32_t p = 0; p < split; p++) { value += partial[(warp + p) * NT + t]; }
        out[(uint64_t)t * rows + row] = __bfloat162float(__float2bfloat16_rn(value));
    }
}

// YoungAi cuda_v41_q4k.inc.cu:29-70: register header and contiguous packed
// words. GLM keeps its FP32 FMA dot and BF16 output, with a new lane order.
__device__ __forceinline__ uint32_t glmvq_q4_byte(const uint4 &h, uint32_t index) {
    const uint32_t word = index < 4 ? h.y : index < 8 ? h.z : h.w;
    return word >> ((index & 3) * 8) & 255u;
}

__device__ __forceinline__ void glmvq_q4_scales(const uint4 &h, uint32_t group,
        float *scale, float *minimum) {
    *scale = group < 4 ? glmvq_q4_byte(h, group) & 63u :
        (glmvq_q4_byte(h, group + 4) & 15u) | ((glmvq_q4_byte(h, group - 4) >> 6) << 4);
    *minimum = group < 4 ? glmvq_q4_byte(h, group + 4) & 63u :
        (glmvq_q4_byte(h, group + 4) >> 4) | ((glmvq_q4_byte(h, group) >> 6) << 4);
}

template <uint32_t NT>
__device__ __forceinline__ void glmvq_q4_acc(const uint8_t *block, const float *input,
        uint32_t cols, uint32_t base, uint32_t tokens, float *sum) {
    const uint32_t lane = threadIdx.x & 31, group = lane >> 3, offset = (lane & 7) * 4;
    const uint4 h = *(const uint4 *)block;
    const uint32_t packed = *(const uint32_t *)(block + 16 + lane * 4);
    const float d = __half2float(__ushort_as_half(h.x & 65535u));
    const float dm = __half2float(__ushort_as_half(h.x >> 16));
    float sl, ml, sh, mh;
    glmvq_q4_scales(h, group * 2, &sl, &ml);
    glmvq_q4_scales(h, group * 2 + 1, &sh, &mh);
    sl *= d; sh *= d; ml *= dm; mh *= dm;
    float low[4], high[4];
    #pragma unroll
    for (uint32_t j = 0; j < 4; j++) {
        const uint32_t byte = packed >> (j * 8) & 255u;
        low[j] = sl * (byte & 15u) - ml;
        high[j] = sh * (byte >> 4) - mh;
    }
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        if (t >= tokens) { continue; }
        const float *x = input + (uint64_t)t * cols + base + group * 64 + offset;
        const float4 lo = *(const float4 *)x, hi = *(const float4 *)(x + 32);
        sum[t] = fmaf(low[0], lo.x, sum[t]); sum[t] = fmaf(low[1], lo.y, sum[t]);
        sum[t] = fmaf(low[2], lo.z, sum[t]); sum[t] = fmaf(low[3], lo.w, sum[t]);
        sum[t] = fmaf(high[0], hi.x, sum[t]); sum[t] = fmaf(high[1], hi.y, sum[t]);
        sum[t] = fmaf(high[2], hi.z, sum[t]); sum[t] = fmaf(high[3], hi.w, sum[t]);
    }
}

// YoungAi :112-179: CTA-wide aligned staging and persistent double buffer.
// GLM uses the current stream's ordinary dependency; no ds41 PDL state.
template <glmvq_q4_load LOAD>
__device__ __forceinline__ void glmvq_q4_copy(uint4 *dst, const uint4 *src, uint32_t count) {
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        if constexpr (LOAD == GLMVQ_Q4_PIPE) { __pipeline_memcpy_async(dst + i, src + i, sizeof(uint4)); }
        else { dst[i] = __ldcs(src + i); }
    }
}

template <uint32_t NT, glmvq_q4_load LOAD>
__global__ void __launch_bounds__(256, 4) glmvq_q4_stream(const uint8_t *weight,
        const float *input, float *out, uint32_t cols, uint32_t rows, uint32_t tokens, uint32_t split) {
    extern __shared__ uint4 staged[];
    __shared__ float partial[GLMVQ_Q4_WARPS * NT];
    glmvq_pdl_open();
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint32_t rpb = GLMVQ_Q4_WARPS / split, local = warp / split, part = warp % split;
    const uint32_t blocks = cols / GLMVQ_Q4_BLOCK, groups = 1 + (rows - 1) / rpb;
    const uint32_t row16 = blocks * (GLMVQ_Q4_BYTES / sizeof(uint4)), group16 = rpb * row16;
    uint32_t group = blockIdx.x, buffer = 0;
    const uint32_t first_rows = min(rpb, rows - group * rpb);
    glmvq_q4_copy<LOAD>(staged, (const uint4 *)weight + (uint64_t)group * group16, first_rows * row16);
    if constexpr (LOAD == GLMVQ_Q4_PIPE) { __pipeline_commit(); }
    // The first weights stream while the predecessor drains.
    glmvq_pdl_wait();
    for (; group < groups; group += LOAD == GLMVQ_Q4_PIPE ? gridDim.x : groups, buffer ^= 1) {
        if constexpr (LOAD == GLMVQ_Q4_PIPE) {
            const uint32_t next = group + gridDim.x;
            if (next < groups) {
                const uint32_t count = min(rpb, rows - next * rpb) * row16;
                glmvq_q4_copy<LOAD>(staged + (buffer ^ 1) * group16,
                    (const uint4 *)weight + (uint64_t)next * group16, count);
            }
            // Empty commits keep the last iteration's wait distance valid.
            __pipeline_commit();
            __pipeline_wait_prior(1);
        }
        __syncthreads();
        const uint32_t row = group * rpb + local;
        float sum[NT] = {};
        if (row < rows) {
            const uint8_t *w = (const uint8_t *)(staged + buffer * group16 + local * row16);
            for (uint32_t b = part; b < blocks; b += split) {
                glmvq_q4_acc<NT>(w + b * GLMVQ_Q4_BYTES, input, cols, b * GLMVQ_Q4_BLOCK, tokens, sum);
            }
        }
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            for (int stride = 16; stride; stride >>= 1) { sum[t] += __shfl_xor_sync(0xffffffffu, sum[t], stride); }
            if (!lane) { partial[warp * NT + t] = sum[t]; }
        }
        __syncthreads();
        if (row < rows && !lane && !part) {
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                if (t >= tokens) { continue; }
                float value = 0;
                for (uint32_t p = 0; p < split; p++) { value += partial[(warp + p) * NT + t]; }
                out[(uint64_t)t * rows + row] = __bfloat162float(__float2bfloat16_rn(value));
            }
        }
        // No warp may overwrite either staging or reduction data still read.
        if constexpr (LOAD == GLMVQ_Q4_PIPE) { __syncthreads(); }
    }
}

/* Up to four matrices sharing one input (Beam's q/k/v/gate): one STAGE
 * launch whose blocks take rpb-row groups matrix by matrix, so the 1,024-row
 * k/v projections stream beside the large ones instead of running the
 * latency-bound scalar kernel in their own launches. */
enum { GLMVQ_Q4_MANY = 4 };
struct glmvq_q4_many {
    const uint8_t *weight[GLMVQ_Q4_MANY];
    float *out[GLMVQ_Q4_MANY];
    uint32_t rows[GLMVQ_Q4_MANY], first[GLMVQ_Q4_MANY + 1];   // first[c]: matrix c's first group
};

template <uint32_t NT>
__global__ void __launch_bounds__(256, 4) glmvq_q4_stream_many(glmvq_q4_many many,
        const float *input, uint32_t cols, uint32_t tokens, uint32_t split) {
    extern __shared__ uint4 staged[];
    __shared__ float partial[GLMVQ_Q4_WARPS * NT];
    glmvq_pdl_open();
    // Selected by compares, not an index: a dynamic index would copy the
    // parameter block to local memory in every thread.
    const uint8_t *weight = many.weight[0];
    float *out = many.out[0];
    uint32_t rows = many.rows[0], first = 0;
    #pragma unroll
    for (uint32_t c = 1; c < GLMVQ_Q4_MANY; c++) {
        if (blockIdx.x >= many.first[c]) { weight = many.weight[c]; out = many.out[c]; rows = many.rows[c]; first = many.first[c]; }
    }
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint32_t rpb = GLMVQ_Q4_WARPS / split, local = warp / split, part = warp % split;
    const uint32_t blocks = cols / GLMVQ_Q4_BLOCK, group = blockIdx.x - first;
    const uint32_t row16 = blocks * (GLMVQ_Q4_BYTES / sizeof(uint4));
    glmvq_q4_copy<GLMVQ_Q4_STAGE>(staged, (const uint4 *)weight + (uint64_t)group * rpb * row16,
        min(rpb, rows - group * rpb) * row16);
    // The first weights stream while the predecessor drains.
    glmvq_pdl_wait();
    __syncthreads();
    const uint32_t row = group * rpb + local;
    float sum[NT] = {};
    if (row < rows) {
        const uint8_t *w = (const uint8_t *)(staged + local * row16);
        for (uint32_t b = part; b < blocks; b += split) {
            glmvq_q4_acc<NT>(w + b * GLMVQ_Q4_BYTES, input, cols, b * GLMVQ_Q4_BLOCK, tokens, sum);
        }
    }
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        for (int stride = 16; stride; stride >>= 1) { sum[t] += __shfl_xor_sync(0xffffffffu, sum[t], stride); }
        if (!lane) { partial[warp * NT + t] = sum[t]; }
    }
    __syncthreads();
    if (row < rows && !lane && !part) {
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            if (t >= tokens) { continue; }
            float value = 0;
            for (uint32_t p = 0; p < split; p++) { value += partial[(warp + p) * NT + t]; }
            out[(uint64_t)t * rows + row] = __bfloat162float(__float2bfloat16_rn(value));
        }
    }
}

static cudaError_t glmvq_q4_prepare(glmvq_q4_config *config, glmvq_q4_load mode) {
    int device, sms, shared, major;
    cudaError_t err = cudaGetDevice(&device);
    if (err == cudaSuccess) { err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device); }
    if (err == cudaSuccess) { err = cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlock, device); }
    if (err == cudaSuccess) { err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device); }
    if (err == cudaSuccess) { *config = {mode, (uint32_t)sms * 4u, (uint32_t)shared, (uint32_t)major, GLMVQ_LAUNCH_PLAIN}; }
    // The grouped GEMV reads its input row through L1. After a kernel that
    // needs a near-maximal shared carveout (Beam's 99 KiB decode attention)
    // the driver keeps that split for every launch it still fits, leaving
    // 28 KiB of L1: Beam's q/gate projections ran 85 -> 100 us. Half the
    // unified memory as shared still fits four 12 KiB stages per SM. The
    // single-matrix kernel keeps the default: the 46 KiB stages of wide rows
    // (dense down, the head) need the large split for two blocks per SM.
    if (err == cudaSuccess) {
        err = cudaFuncSetAttribute(glmvq_q4_stream_many<1>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
    }
    if (err == cudaSuccess) {
        err = cudaFuncSetAttribute(glmvq_q4_stream_many<4>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
    }
    if (err == cudaSuccess) {
        err = cudaFuncSetAttribute(glmvq_q4_stream_many<8>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
    }
    return err;
}

static glmvq_q4_load glmvq_q4_select(const glmvq_q4_config *config, const float *input,
        uint64_t group_bytes, uint64_t weight_bytes, uint32_t tokens) {
    if (!config || config->mode == GLMVQ_Q4_SCALAR || (uintptr_t)input % 16) { return GLMVQ_Q4_SCALAR; }
    // Actual GLM replays favor staging for large decode projections; small
    // one-token matrices keep their measured scalar baseline. Verify widths
    // stage too: the scalar kernel dequantized each weight and loaded each
    // token's input per element (2.7x one token's time at four tokens).
    constexpr uint64_t small_weight_bytes = 4ull << 20;
    if (config->mode == GLMVQ_Q4_AUTO && tokens == 1 && weight_bytes <= small_weight_bytes) {
        return GLMVQ_Q4_SCALAR;
    }
    const uint32_t partial = GLMVQ_Q4_WARPS * (tokens == 1 ? 1 : GLMVQ_Q4_WARPS) * sizeof(float);
    if (group_bytes + partial > config->shared) { return GLMVQ_Q4_SCALAR; }
    // Double buffering lost to one-stage loads on these GLM shapes; the
    // explicit mode remains available to the isolated diagnostic harness.
    if (config->mode == GLMVQ_Q4_PIPE && config->major >= 8 && config->blocks &&
        2 * group_bytes + partial <= config->shared) {
        return GLMVQ_Q4_PIPE;
    }
    return GLMVQ_Q4_STAGE;
}

template <uint32_t NT>
static void glmvq_q4_launch(const void *weight, const float *input, float *out,
        uint32_t rows, uint32_t cols, uint32_t tokens, uint32_t split, cudaStream_t stream,
        const glmvq_q4_config *config) {
    const uint32_t rpb = GLMVQ_Q4_WARPS / split, grid = 1 + (rows - 1) / rpb;
    const uint64_t bytes = (uint64_t)rpb * cols / GLMVQ_Q4_BLOCK * GLMVQ_Q4_BYTES;
    const uint64_t weight_bytes = (uint64_t)rows * cols / GLMVQ_Q4_BLOCK * GLMVQ_Q4_BYTES;
    const glmvq_launch launch = config ? config->launch : GLMVQ_LAUNCH_PLAIN;
    const auto *w = (const uint8_t *)weight;
    switch (glmvq_q4_select(config, input, bytes, weight_bytes, tokens)) {
        case GLMVQ_Q4_PIPE:
            glmvq_launch_kernel(launch, glmvq_q4_stream<NT, GLMVQ_Q4_PIPE>, min(grid, config->blocks),
                GLMVQ_Q4_WARPS * 32, 2 * bytes, stream, w, input, out, cols, rows, tokens, split);
            break;
        case GLMVQ_Q4_STAGE:
            glmvq_launch_kernel(launch, glmvq_q4_stream<NT, GLMVQ_Q4_STAGE>, grid,
                GLMVQ_Q4_WARPS * 32, bytes, stream, w, input, out, cols, rows, tokens, split);
            break;
        default:
            glmvq_launch_kernel(launch, glmvq_q4_small<NT>, grid, GLMVQ_Q4_WARPS * 32, 0, stream,
                w, input, out, cols, rows, tokens, split);
    }
}

// Up to eight tokens over `count` matrices of `cols` inputs
// (glmvq_q4_stream_many); split 2 (four rows per block) as the 6,144-row
// projections pick alone. A token's FMA order does not depend on the width:
// verify tokens equal one-token calls bit for bit.
static cudaError_t glmvq_q4_decode_many(const void *const *weight, float *const *out, const uint32_t *rows,
        uint32_t count, uint32_t cols, const float *input, uint32_t tokens, cudaStream_t stream,
        const glmvq_q4_config *config) {
    constexpr uint32_t split = 2, rpb = GLMVQ_Q4_WARPS / split;
    if (!config || !count || count > GLMVQ_Q4_MANY || !input || !cols || cols % GLMVQ_Q4_BLOCK ||
        !tokens || tokens > GLMVQ_Q4_WARPS || (uintptr_t)input % 16) { return cudaErrorInvalidValue; }
    glmvq_q4_many many = {};
    uint32_t groups = 0;
    for (uint32_t c = 0; c < GLMVQ_Q4_MANY; c++) {
        many.first[c] = groups;
        if (c >= count) { continue; }
        if (!weight[c] || !out[c] || !rows[c] || (uintptr_t)weight[c] % 16) { return cudaErrorInvalidValue; }
        many.weight[c] = (const uint8_t *)weight[c];
        many.out[c] = out[c];
        many.rows[c] = rows[c];
        groups += 1 + (rows[c] - 1) / rpb;
    }
    many.first[GLMVQ_Q4_MANY] = groups;
    const uint64_t bytes = (uint64_t)rpb * cols / GLMVQ_Q4_BLOCK * GLMVQ_Q4_BYTES;
    if (bytes + GLMVQ_Q4_WARPS * sizeof(float) > config->shared) { return cudaErrorInvalidValue; }
    if (tokens == 1) {
        glmvq_launch_kernel(config->launch, glmvq_q4_stream_many<1>, groups, GLMVQ_Q4_WARPS * 32, bytes, stream,
            many, input, cols, tokens, split);
    } else if (tokens <= 4) {
        glmvq_launch_kernel(config->launch, glmvq_q4_stream_many<4>, groups, GLMVQ_Q4_WARPS * 32, bytes, stream,
            many, input, cols, tokens, split);
    } else {
        glmvq_launch_kernel(config->launch, glmvq_q4_stream_many<8>, groups, GLMVQ_Q4_WARPS * 32, bytes, stream,
            many, input, cols, tokens, split);
    }
    return cudaGetLastError();
}

static cudaError_t glmvq_q4_decode(const void *weight, const float *input, float *out,
        uint32_t rows, uint32_t cols, uint32_t tokens, cudaStream_t stream,
        const glmvq_q4_config *config = nullptr) {
    if (!weight || !input || !out || !rows || !cols || cols % GLMVQ_Q4_BLOCK ||
        !tokens || tokens > GLMVQ_Q4_WARPS || (uintptr_t)weight % 16) { return cudaErrorInvalidValue; }
    uint32_t split = 1;
    constexpr uint32_t target_warps = 8192;
    while (split < GLMVQ_Q4_WARPS && split * rows < target_warps &&
        split * 2 <= cols / GLMVQ_Q4_BLOCK) { split *= 2; }
    if (tokens == 1) {
        glmvq_q4_launch<1>(weight, input, out, rows, cols, tokens, split, stream, config);
    } else if (tokens <= 4) {
        glmvq_q4_launch<4>(weight, input, out, rows, cols, tokens, split, stream, config);
    } else {
        glmvq_q4_launch<8>(weight, input, out, rows, cols, tokens, split, stream, config);
    }
    return cudaGetLastError();
}
#endif
