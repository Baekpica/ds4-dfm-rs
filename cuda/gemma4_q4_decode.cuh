#ifndef DS4_GEMMA4_Q4_DECODE_CUH
#define DS4_GEMMA4_Q4_DECODE_CUH

/* Gemma-4 batch-1 Q4_K decode.
 *
 * The shared dispatcher sends a one-token Q4_K matrix of at most 4 MiB to
 * glmvq_q4_small (cuda/glm_vq_q4.cuh:27-69), a per-element scalar decode:
 * Gemma's k_proj, v_proj, dense_gate and dense_up are all under that line
 * (3.09, 3.09, 3.19 and 3.19 MiB), so 115 of the 145 Q4_K matrices a token
 * touches take it. This header is the Gemma-owned replacement: every matrix
 * goes through the vectorized block dot with the weights staged in shared
 * memory, one warp per row, and matrices sharing one input (q/k/v, the
 * dense gate/up pair) run in one launch.
 *
 * The Q4_K block layout is the container's (cuda/q4_k.cuh: GGML Q4_K, 256
 * elements in 144 bytes: two f16 scales, eight packed six-bit scale/min
 * pairs, then low/high nibble groups). The block dot is Gemma's copy of
 * glmvq_q4_acc (cuda/glm_vq_q4.cuh:85-116): lane l covers group l/8 and the
 * four low plus four high nibbles at offset (l%8)*4, eight FMAs per block,
 * the FP32 sum reduced with the same XOR tree. It is the same arithmetic
 * with the same per-lane element sets; the row sum's FP32 order differs
 * from the scalar kernel's, which is the measured parity class.
 *
 * The GLU kernel computes both dense matrices from the same staged input and
 * applies the existing epilogue, round(gelu(gate) * round(up))
 * (cuda/gemma4_primitives.cuh:182-187), so it removes a launch and a pass
 * over the input without changing the arithmetic. */

#include "q4_k.cuh"
#include "gemma4_primitives.cuh"

enum { G4_Q4_BLOCK = DS4_Q4_K_BLOCK, G4_Q4_BYTES = DS4_Q4_K_BYTES, G4_Q4_WARPS = 8,
       G4_Q4_MANY = 5 };

struct g4_q4_many {
    const uint8_t *weight[G4_Q4_MANY];
    float *out[G4_Q4_MANY];
    uint32_t rows[G4_Q4_MANY], first[G4_Q4_MANY + 1];
};

/* One byte of the 12-byte scale header. */
__device__ __forceinline__ uint32_t g4_q4_byte(const uint4 &h, uint32_t index) {
    const uint32_t word = index < 4 ? h.y : index < 8 ? h.z : h.w;
    return word >> ((index & 3) * 8) & 255u;
}

/* The six-bit scale and minimum of group `group`, cuda/q4_k.cuh:12-24. */
__device__ __forceinline__ void g4_q4_scales(const uint4 &h, uint32_t group,
        float *scale, float *minimum) {
    *scale = group < 4 ? g4_q4_byte(h, group) & 63u :
        (g4_q4_byte(h, group + 4) & 15u) | ((g4_q4_byte(h, group - 4) >> 6) << 4);
    *minimum = group < 4 ? g4_q4_byte(h, group + 4) & 63u :
        (g4_q4_byte(h, group + 4) >> 4) | ((g4_q4_byte(h, group) >> 6) << 4);
}

/* One 256-element block against the input row: eight weights per lane, the
 * low nibbles of the even 32-element group and the high nibbles of the odd
 * one. `sum` carries the row's running FP32 sum. */
__device__ __forceinline__ void g4_q4_acc(const uint8_t *block, const float *input,
        uint32_t base, float *sum) {
    const uint32_t lane = threadIdx.x & 31, group = lane >> 3, offset = (lane & 7) * 4;
    const uint4 h = *(const uint4 *)block;
    const uint32_t packed = *(const uint32_t *)(block + 16 + lane * 4);
    const float d = __half2float(__ushort_as_half(h.x & 65535u));
    const float dm = __half2float(__ushort_as_half(h.x >> 16));
    float sl, ml, sh, mh;
    g4_q4_scales(h, group * 2, &sl, &ml);
    g4_q4_scales(h, group * 2 + 1, &sh, &mh);
    sl *= d;
    sh *= d;
    ml *= dm;
    mh *= dm;
    float low[4], high[4];
    #pragma unroll
    for (uint32_t j = 0; j < 4; j++) {
        const uint32_t byte = packed >> (j * 8) & 255u;
        low[j] = sl * (byte & 15u) - ml;
        high[j] = sh * (byte >> 4) - mh;
    }
    const float *x = input + base + group * 64 + offset;
    const float4 lo = *(const float4 *)x, hi = *(const float4 *)(x + 32);
    *sum = fmaf(low[0], lo.x, *sum);
    *sum = fmaf(low[1], lo.y, *sum);
    *sum = fmaf(low[2], lo.z, *sum);
    *sum = fmaf(low[3], lo.w, *sum);
    *sum = fmaf(high[0], hi.x, *sum);
    *sum = fmaf(high[1], hi.y, *sum);
    *sum = fmaf(high[2], hi.z, *sum);
    *sum = fmaf(high[3], hi.w, *sum);
}

/* Up to five Q4_K matrices sharing one input: one block per group of eight
 * rows, one warp per row, the group's weights staged in shared. The matrix
 * is selected by compares, not an index: a dynamic index would copy the
 * parameter block to local memory in every thread (cuda/glm_vq_q4.cuh:204). */
__global__ void __launch_bounds__(G4_Q4_WARPS * 32, 4) g4_q4_many_kernel(g4_q4_many many,
        const float *input, uint32_t cols) {
    extern __shared__ uint4 staged[];
    const uint8_t *weight = many.weight[0];
    float *out = many.out[0];
    uint32_t rows = many.rows[0], first = 0;
    #pragma unroll
    for (uint32_t c = 1; c < G4_Q4_MANY; c++) {
        if (blockIdx.x >= many.first[c]) {
            weight = many.weight[c];
            out = many.out[c];
            rows = many.rows[c];
            first = many.first[c];
        }
    }
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint32_t blocks = cols / G4_Q4_BLOCK, group = blockIdx.x - first;
    const uint32_t row16 = blocks * (G4_Q4_BYTES / sizeof(uint4));
    const uint32_t staged_rows = min(G4_Q4_WARPS, rows - group * G4_Q4_WARPS);
    const uint4 *src = (const uint4 *)weight + (uint64_t)group * G4_Q4_WARPS * row16;
    for (uint32_t i = threadIdx.x; i < staged_rows * row16; i += blockDim.x) {
        staged[i] = __ldcs(src + i);
    }
    __syncthreads();
    const uint32_t row = group * G4_Q4_WARPS + warp;
    float sum = 0.0f;
    if (row < rows) {
        const uint8_t *w = (const uint8_t *)(staged + warp * row16);
        for (uint32_t b = 0; b < blocks; b++) {
            g4_q4_acc(w + b * G4_Q4_BYTES, input, b * G4_Q4_BLOCK, &sum);
        }
    }
    #pragma unroll
    for (int stride = 16; stride; stride >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, stride);
    }
    if (row < rows && !lane) { out[row] = __bfloat162float(__float2bfloat16_rn(sum)); }
}

/* The dense gate/up pair: both matrices staged beside each other, both sums
 * per row, the existing GLU epilogue. */
__global__ void __launch_bounds__(G4_Q4_WARPS * 32, 4) g4_q4_glu_kernel(
        const uint8_t *gate, const uint8_t *up, float *out, const float *input,
        uint32_t cols, uint32_t rows) {
    extern __shared__ uint4 staged[];
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint32_t blocks = cols / G4_Q4_BLOCK, group = blockIdx.x;
    const uint32_t row16 = blocks * (G4_Q4_BYTES / sizeof(uint4));
    const uint32_t staged_rows = min(G4_Q4_WARPS, rows - group * G4_Q4_WARPS);
    uint4 *sg = staged, *su = staged + G4_Q4_WARPS * row16;
    const uint4 *g = (const uint4 *)gate + (uint64_t)group * G4_Q4_WARPS * row16;
    const uint4 *u = (const uint4 *)up + (uint64_t)group * G4_Q4_WARPS * row16;
    for (uint32_t i = threadIdx.x; i < staged_rows * row16; i += blockDim.x) {
        sg[i] = __ldcs(g + i);
        su[i] = __ldcs(u + i);
    }
    __syncthreads();
    const uint32_t row = group * G4_Q4_WARPS + warp;
    if (row >= rows) { return; }
    const uint8_t *wg = (const uint8_t *)(sg + warp * row16);
    const uint8_t *wu = (const uint8_t *)(su + warp * row16);
    float gs = 0.0f, us = 0.0f;
    for (uint32_t b = 0; b < blocks; b++) {
        g4_q4_acc(wg + b * G4_Q4_BYTES, input, b * G4_Q4_BLOCK, &gs);
        g4_q4_acc(wu + b * G4_Q4_BYTES, input, b * G4_Q4_BLOCK, &us);
    }
    #pragma unroll
    for (int stride = 16; stride; stride >>= 1) {
        gs += __shfl_xor_sync(0xffffffffu, gs, stride);
        us += __shfl_xor_sync(0xffffffffu, us, stride);
    }
    if (!lane) { out[row] = g4_round(g4_gelu(gs) * g4_round(us)); }
}

#endif
