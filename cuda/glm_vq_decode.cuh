#ifndef DS4_GLM_VQ_DECODE_CUH
#define DS4_GLM_VQ_DECODE_CUH

#include "glm_vq_ops.cuh"

enum { GLMVQ_DEC_HIDDEN = 4096, GLMVQ_DEC_FFN = 2048, GLMVQ_DEC_USED = 8 };
enum glmvq_decode_output { GLMVQ_DEC_NORMAL, GLMVQ_DEC_PROOF };
// Where the row gain enters an expert dot. WEIGHT rounds gain * codeword to
// BF16 per weight, as the restore path does; DOT multiplies the FP32 dot
// once, as the prefill tensor-core kernel does.
enum glmvq_decode_gain { GLMVQ_GAIN_WEIGHT, GLMVQ_GAIN_DOT };

struct glmvq_decode_item {
    glmvq_matrix gate, up, down;
    uint32_t pair;
};

struct glmvq_decode_block { uint32_t a, b, c, plane; };

// YoungAi 7fb26cc cuda_vq_row.inc.cu:146-280: a warp loads three complete
// 128-byte lines (256 indices), with a separate 32-byte thirteenth plane.
// A row narrower than 2048 inputs is one partial block of STEPS k-steps
// (Ling's 1280-wide down rows: five steps, 60 words, never padded).
template <uint32_t BITS, uint32_t STEPS = 8>
__device__ __forceinline__ glmvq_decode_block glmvq_decode_load(
        const uint32_t *main, const uint32_t *plane) {
    const uint32_t lane = threadIdx.x & 31;
    if constexpr (STEPS == 8) {
        return {main[lane], main[lane + 32], main[lane + 64],
            BITS == 13 && lane < 8 ? plane[lane] : 0u};
    } else {
        static_assert(STEPS >= 3 && STEPS < 8, "a partial block holds 3-7 k-steps");
        constexpr uint32_t words = 12 * STEPS;
        return {main[lane], lane + 32 < words ? main[lane + 32] : 0u,
            lane + 64 < words ? main[lane + 64] : 0u, BITS == 13 && lane < STEPS ? plane[lane] : 0u};
    }
}

// k-steps of a row's first block: full blocks have eight.
template <uint32_t COLS>
__host__ __device__ constexpr uint32_t glmvq_decode_steps() { return COLS >= 2048 ? 8u : COLS / 256u; }

__device__ __forceinline__ uint32_t glmvq_decode_word(
        const glmvq_decode_block &b, uint32_t which) {
    return which == 0 ? b.a : which == 1 ? b.b : b.c;
}

__device__ __forceinline__ float glmvq_decode_pair(uint32_t word, uint32_t x,
        float gain, float sum) {
    // GLM scales and rounds each weight before the dot. YoungAi applies
    // gain after reduction; that arithmetic is deliberately not imported.
    const uint32_t w = glmvq_scaled_pair(word, gain);
    sum = fmaf(__uint_as_float(w << 16), __uint_as_float(x << 16), sum);
    return fmaf(__uint_as_float(w & 0xffff0000u), __uint_as_float(x & 0xffff0000u), sum);
}

// Two E4M3 codeword values as exact floats.
__device__ __forceinline__ float2 glmvq_decode_raw(uint32_t bits) {
    return __half22float2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)bits, __NV_E4M3));
}

// sum += one 8-wide codeword . eight BF16 inputs.
template <glmvq_decode_gain GAIN>
__device__ __forceinline__ float glmvq_decode_word8(uint2 word, const uint4 &values, float gain, float sum) {
    if constexpr (GAIN == GLMVQ_GAIN_WEIGHT) {
        sum = glmvq_decode_pair(word.x, values.x, gain, sum);
        sum = glmvq_decode_pair(word.x >> 16, values.y, gain, sum);
        sum = glmvq_decode_pair(word.y, values.z, gain, sum);
        return glmvq_decode_pair(word.y >> 16, values.w, gain, sum);
    }
    const uint32_t ws[4] = {word.x, word.x >> 16, word.y, word.y >> 16}, xs[4] = {values.x, values.y, values.z, values.w};
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        const float2 w = glmvq_decode_raw(ws[i]);
        sum = fmaf(w.x, __uint_as_float(xs[i] << 16), sum);
        sum = fmaf(w.y, __uint_as_float(xs[i] & 0xffff0000u), sum);
    }
    return sum;
}

template <uint32_t COLS, uint32_t BITS, glmvq_decode_gain GAIN = GLMVQ_GAIN_WEIGHT>
__device__ __forceinline__ float glmvq_decode_dot(const glmvq_matrix &m,
        uint32_t row, const __nv_bfloat16 *x, const uint2 *book,
        glmvq_decode_block *carry, const uint32_t *next, const uint32_t *next_plane) {
    const uint32_t lane = threadIdx.x & 31, offset = lane * 12 / 32, shift = lane * 12 % 32;
    const float gain = __half2float(m.gain[row]);
    glmvq_decode_block cur = *carry;
    float sum = 0;
    if constexpr (COLS % 2048 != 0) {
        // One partial block; `next` (same width) is prefetched under it.
        static_assert(COLS < 2048 && COLS % 256 == 0, "partial rows are one block");
        constexpr uint32_t STEPS = COLS / 256;
        const glmvq_decode_block ahead = next ? glmvq_decode_load<BITS, STEPS>(next, next_plane)
                                              : glmvq_decode_block{};
        #pragma unroll
        for (uint32_t k = 0; k < STEPS; k++) {
            const uint32_t first = k * 12;
            const uint32_t lr = (first & 31u) > 20 ? (first + 31 - lane) / 32 : first / 32;
            const uint32_t hr = ((first + 1) & 31u) > 20 ? (first + 32 - lane) / 32 : (first + 1) / 32;
            const uint32_t lo = __shfl_sync(0xffffffffu, glmvq_decode_word(cur, lr), first + offset);
            const uint32_t hi = __shfl_sync(0xffffffffu, glmvq_decode_word(cur, hr), first + offset + 1);
            uint32_t index = __funnelshift_r(lo, hi, shift) & 4095u;
            if constexpr (BITS == 13) {
                index |= ((__shfl_sync(0xffffffffu, cur.plane, k) >> lane) & 1u) << 12;
            }
            const uint2 word = book[index];
            const uint4 values = *(const uint4 *)(x + k * 256 + lane * GLMVQ_VECTOR);
            sum = glmvq_decode_word8<GAIN>(word, values, gain, sum);
        }
        *carry = ahead;
        for (uint32_t stride = 16; stride; stride >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, stride); }
        return glmvq_round(GAIN == GLMVQ_GAIN_DOT ? glmvq_mul(sum, gain) : sum);
    }
    const auto *main = (const uint32_t *)m.main + (uint64_t)row * COLS * 3 / 64;
    const auto *plane = BITS == 13 ? (const uint32_t *)m.plane + (uint64_t)row * COLS / 256 : nullptr;
    for (uint32_t b = 0; b < COLS / 2048; b++) {
        // One block ahead in registers; GLM rows have no partial 384B block.
        glmvq_decode_block ahead = {};
        if (b + 1 < COLS / 2048) {
            ahead = glmvq_decode_load<BITS>(main + (b + 1) * 96,
                BITS == 13 ? plane + (b + 1) * 8 : nullptr);
        } else if (next) {
            ahead = glmvq_decode_load<BITS>(next, next_plane);
        }
        #pragma unroll
        for (uint32_t k = 0; k < 8; k++) {
            const uint32_t first = k * 12;
            const uint32_t lr = (first & 31u) > 20 ? (first + 31 - lane) / 32 : first / 32;
            const uint32_t hr = ((first + 1) & 31u) > 20 ? (first + 32 - lane) / 32 : (first + 1) / 32;
            const uint32_t lo = __shfl_sync(0xffffffffu, glmvq_decode_word(cur, lr), first + offset);
            const uint32_t hi = __shfl_sync(0xffffffffu, glmvq_decode_word(cur, hr), first + offset + 1);
            uint32_t index = __funnelshift_r(lo, hi, shift) & 4095u;
            if constexpr (BITS == 13) {
                index |= ((__shfl_sync(0xffffffffu, cur.plane, k) >> lane) & 1u) << 12;
            }
            const uint2 word = book[index];
            const uint4 values = *(const uint4 *)(x + b * 2048 + k * 256 + lane * GLMVQ_VECTOR);
            sum = glmvq_decode_word8<GAIN>(word, values, gain, sum);
        }
        cur = ahead;
    }
    *carry = cur;
    for (uint32_t stride = 16; stride; stride >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, stride); }
    return glmvq_round(GAIN == GLMVQ_GAIN_DOT ? glmvq_mul(sum, gain) : sum);
}

template <uint32_t BITS>
__device__ __forceinline__ void glmvq_decode_book(uint4 *dst, const uint4 *src) {
    for (uint32_t i = threadIdx.x; i < (1u << BITS) * GLMVQ_VECTOR / sizeof(uint4); i += blockDim.x) {
        dst[i] = src[i];
    }
}

// YoungAi cuda_vq_persist.inc.cu:1-210: each persistent CTA retains one
// layer book. E4M3 stays packed: even the 13-bit book needs only 64 KiB.
template <uint32_t BITS, uint32_t WARPS, glmvq_decode_output OUTPUT, glmvq_decode_gain GAIN = GLMVQ_GAIN_WEIGHT,
          uint32_t FFN = GLMVQ_DEC_FFN, glmvq_clamp CLAMP = GLMVQ_CLAMP_GATE>
__global__ void __launch_bounds__(WARPS * 32, 1) glmvq_decode_gu(
        const glmvq_decode_item *items, const uint4 *book, const float *input,
        __nv_bfloat16 *mid, float clamp, float *gate, float *up) {
    extern __shared__ __align__(16) uint8_t decode_stage[];
    constexpr uint32_t book_bytes = (1u << BITS) * GLMVQ_VECTOR;
    auto *x = (__nv_bfloat16 *)(decode_stage + book_bytes);
    glmvq_pdl_open();
    glmvq_decode_book<BITS>((uint4 *)decode_stage, book);
    glmvq_pdl_wait();
    for (uint32_t i = threadIdx.x; i < GLMVQ_DEC_HIDDEN; i += blockDim.x) {
        x[i] = __float2bfloat16_rn(input[i]);
    }
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31, warp = blockIdx.x * WARPS + threadIdx.x / 32;
    for (uint32_t task = warp; task < GLMVQ_DEC_USED * FFN; task += gridDim.x * WARPS) {
        const uint32_t e = task / FFN, row = task % FFN;
        const glmvq_decode_item item = items[e];
        const auto *gm = (const uint32_t *)item.gate.main + row * GLMVQ_DEC_HIDDEN * 3 / 64;
        const auto *gp = BITS == 13 ? (const uint32_t *)item.gate.plane + row * GLMVQ_DEC_HIDDEN / 256 : nullptr;
        const auto *um = (const uint32_t *)item.up.main + row * GLMVQ_DEC_HIDDEN * 3 / 64;
        const auto *up_plane = BITS == 13 ? (const uint32_t *)item.up.plane + row * GLMVQ_DEC_HIDDEN / 256 : nullptr;
        auto carry = glmvq_decode_load<BITS>(gm, gp);
        const float g = glmvq_decode_dot<GLMVQ_DEC_HIDDEN, BITS, GAIN>(item.gate, row, x,
            (const uint2 *)decode_stage, &carry, um, up_plane);
        const float u = glmvq_decode_dot<GLMVQ_DEC_HIDDEN, BITS, GAIN>(item.up, row, x,
            (const uint2 *)decode_stage, &carry, nullptr, nullptr);
        if (!lane) {
            mid[task] = __float2bfloat16_rn(glmvq_swiglu_value<CLAMP>(g, u, clamp));
            if constexpr (OUTPUT == GLMVQ_DEC_PROOF) { gate[task] = g; up[task] = u; }
        }
    }
}

template <uint32_t BITS, uint32_t WARPS, glmvq_decode_gain GAIN = GLMVQ_GAIN_WEIGHT, uint32_t FFN = GLMVQ_DEC_FFN>
__global__ void __launch_bounds__(WARPS * 32, 1) glmvq_decode_down(
        const glmvq_decode_item *items, const uint4 *book, const __nv_bfloat16 *mid, float *output) {
    extern __shared__ __align__(16) uint8_t decode_stage[];
    constexpr uint32_t book_bytes = (1u << BITS) * GLMVQ_VECTOR;
    auto *x = (__nv_bfloat16 *)(decode_stage + book_bytes);
    glmvq_pdl_open();
    glmvq_decode_book<BITS>((uint4 *)decode_stage, book);
    glmvq_pdl_wait();
    for (uint32_t i = threadIdx.x; i < GLMVQ_DEC_USED * FFN / 8; i += blockDim.x) {
        ((uint4 *)x)[i] = ((const uint4 *)mid)[i];
    }
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31, warp = blockIdx.x * WARPS + threadIdx.x / 32;
    for (uint32_t task = warp; task < GLMVQ_DEC_USED * GLMVQ_DEC_HIDDEN; task += gridDim.x * WARPS) {
        const uint32_t e = task / GLMVQ_DEC_HIDDEN, row = task % GLMVQ_DEC_HIDDEN;
        const glmvq_matrix m = items[e].down;
        const auto *main = (const uint32_t *)m.main + row * FFN * 3 / 64;
        const auto *plane = BITS == 13 ? (const uint32_t *)m.plane + row * FFN / 256 : nullptr;
        auto carry = glmvq_decode_load<BITS, glmvq_decode_steps<FFN>()>(main, plane);
        const float value = glmvq_decode_dot<FFN, BITS, GAIN>(m, row,
            x + e * FFN, (const uint2 *)decode_stage, &carry, nullptr, nullptr);
        if (!lane) { output[task] = value; }
    }
}

template <glmvq_reduce REDUCE = GLMVQ_REDUCE_BF16>
__global__ void glmvq_decode_reduce(float *out, const float *partial,
        const float *weights, const glmvq_decode_item *items, const float *shared = nullptr) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= GLMVQ_DEC_HIDDEN) { return; }
    float sum = 0;
    // Items are in ascending expert order, matching the existing scatter.
    for (uint32_t e = 0; e < GLMVQ_DEC_USED; e++) {
        sum = glmvq_reduce_add<REDUCE>(sum, partial[e * GLMVQ_DEC_HIDDEN + row], weights[items[e].pair]);
    }
    out[row] = glmvq_reduce_end<REDUCE>(sum, shared, row);
}

/* MTP verify widths (2-4 rows): one persistent pass over the union of the
 * rows' experts, so an expert two rows share is streamed once. Per row the
 * arithmetic is the one-row kernel's: same lanes, FMA order, rounding and
 * ascending-expert reduction, so each row's output is bit-identical. */
enum { GLMVQ_VER_ROWS = 4, GLMVQ_VER_PAIRS = GLMVQ_VER_ROWS * GLMVQ_DEC_USED, GLMVQ_VER_GROUP = 8 };

struct glmvq_verify_item {
    glmvq_matrix gate, up, down;
    uint32_t count, slot;                // pairs on this expert; first mid slot in its group
    uint32_t pair[GLMVQ_VER_ROWS];       // row * 8 + route slot, ascending row
};

// Down tasks run in groups of <= 8 pairs whose mid rows fit in shared memory.
struct glmvq_verify_plan {
    uint32_t items, groups;
    uint32_t group[GLMVQ_VER_PAIRS + 1];  // first item of each group
    uint32_t order[GLMVQ_VER_PAIRS];      // row * 8 + rank -> route slot, ascending expert
};

// sum[t] += w · x_t for one codeword's four scaled pairs (decode_pair order).
__device__ __forceinline__ float glmvq_verify_fma(const uint32_t *w, const uint4 &x, float sum) {
    const uint32_t xs[4] = {x.x, x.y, x.z, x.w};
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        sum = fmaf(__uint_as_float(w[i] << 16), __uint_as_float(xs[i] << 16), sum);
        sum = fmaf(__uint_as_float(w[i] & 0xffff0000u), __uint_as_float(xs[i] & 0xffff0000u), sum);
    }
    return sum;
}

// glmvq_decode_dot for up to four inputs sharing one decoded weight row.
template <uint32_t COLS, uint32_t BITS, glmvq_decode_gain GAIN = GLMVQ_GAIN_WEIGHT>
__device__ __forceinline__ void glmvq_verify_dot(const glmvq_matrix &m, uint32_t row,
        const __nv_bfloat16 *const *x, uint32_t count, const uint2 *book,
        glmvq_decode_block *carry, const uint32_t *next, const uint32_t *next_plane, float *out) {
    const uint32_t lane = threadIdx.x & 31, offset = lane * 12 / 32, shift = lane * 12 % 32;
    const auto *main = (const uint32_t *)m.main + (uint64_t)row * COLS * 3 / 64;
    const auto *plane = BITS == 13 ? (const uint32_t *)m.plane + (uint64_t)row * COLS / 256 : nullptr;
    const float gain = __half2float(m.gain[row]);
    glmvq_decode_block cur = *carry;
    float sum[GLMVQ_VER_ROWS] = {};
    // A partial row is one block of STEPS k-steps (see glmvq_decode_load).
    constexpr uint32_t BLOCKS = (COLS + 2047) / 2048, TAIL = COLS % 2048 / 256;
    for (uint32_t b = 0; b < BLOCKS; b++) {
        glmvq_decode_block ahead = {};
        if (b + 1 < BLOCKS) {
            ahead = glmvq_decode_load<BITS>(main + (b + 1) * 96,
                BITS == 13 ? plane + (b + 1) * 8 : nullptr);
        } else if (next) {
            ahead = glmvq_decode_load<BITS, glmvq_decode_steps<COLS>()>(next, next_plane);
        }
        #pragma unroll
        for (uint32_t k = 0; k < (TAIL ? TAIL : 8u); k++) {
            const uint32_t first = k * 12;
            const uint32_t lr = (first & 31u) > 20 ? (first + 31 - lane) / 32 : first / 32;
            const uint32_t hr = ((first + 1) & 31u) > 20 ? (first + 32 - lane) / 32 : (first + 1) / 32;
            const uint32_t lo = __shfl_sync(0xffffffffu, glmvq_decode_word(cur, lr), first + offset);
            const uint32_t hi = __shfl_sync(0xffffffffu, glmvq_decode_word(cur, hr), first + offset + 1);
            uint32_t index = __funnelshift_r(lo, hi, shift) & 4095u;
            if constexpr (BITS == 13) {
                index |= ((__shfl_sync(0xffffffffu, cur.plane, k) >> lane) & 1u) << 12;
            }
            const uint2 word = book[index];
            const uint32_t at = b * 2048 + k * 256 + lane * GLMVQ_VECTOR;
            if constexpr (GAIN == GLMVQ_GAIN_WEIGHT) {
                const uint32_t w[4] = {glmvq_scaled_pair(word.x, gain), glmvq_scaled_pair(word.x >> 16, gain),
                    glmvq_scaled_pair(word.y, gain), glmvq_scaled_pair(word.y >> 16, gain)};
                #pragma unroll
                for (uint32_t t = 0; t < GLMVQ_VER_ROWS; t++) {
                    if (t < count) { sum[t] = glmvq_verify_fma(w, *(const uint4 *)(x[t] + at), sum[t]); }
                }
            } else {
                // Convert the codeword once; every row reuses the floats.
                const float2 w[4] = {glmvq_decode_raw(word.x), glmvq_decode_raw(word.x >> 16),
                    glmvq_decode_raw(word.y), glmvq_decode_raw(word.y >> 16)};
                #pragma unroll
                for (uint32_t t = 0; t < GLMVQ_VER_ROWS; t++) {
                    if (t >= count) { continue; }
                    const uint4 v = *(const uint4 *)(x[t] + at);
                    const uint32_t xs[4] = {v.x, v.y, v.z, v.w};
                    #pragma unroll
                    for (uint32_t i = 0; i < 4; i++) {
                        sum[t] = fmaf(w[i].x, __uint_as_float(xs[i] << 16), sum[t]);
                        sum[t] = fmaf(w[i].y, __uint_as_float(xs[i] & 0xffff0000u), sum[t]);
                    }
                }
            }
        }
        cur = ahead;
    }
    *carry = cur;
    #pragma unroll
    for (uint32_t t = 0; t < GLMVQ_VER_ROWS; t++) {
        for (uint32_t stride = 16; stride; stride >>= 1) { sum[t] += __shfl_xor_sync(0xffffffffu, sum[t], stride); }
        out[t] = glmvq_round(GAIN == GLMVQ_GAIN_DOT ? glmvq_mul(sum[t], gain) : sum[t]);
    }
}

template <uint32_t BITS, uint32_t WARPS, glmvq_decode_gain GAIN = GLMVQ_GAIN_WEIGHT,
          uint32_t FFN = GLMVQ_DEC_FFN, glmvq_clamp CLAMP = GLMVQ_CLAMP_GATE>
__global__ void __launch_bounds__(WARPS * 32, 1) glmvq_verify_gu(
        const glmvq_verify_item *items, const glmvq_verify_plan *plan, const uint4 *book,
        const float *input, uint32_t rows, __nv_bfloat16 *mid, float clamp) {
    extern __shared__ __align__(16) uint8_t verify_stage[];
    constexpr uint32_t book_bytes = (1u << BITS) * GLMVQ_VECTOR;
    auto *x = (__nv_bfloat16 *)(verify_stage + book_bytes);
    glmvq_pdl_open();
    glmvq_decode_book<BITS>((uint4 *)verify_stage, book);
    glmvq_pdl_wait();
    for (uint32_t i = threadIdx.x; i < rows * GLMVQ_DEC_HIDDEN; i += blockDim.x) {
        x[i] = __float2bfloat16_rn(input[i]);
    }
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31, warp = blockIdx.x * WARPS + threadIdx.x / 32;
    const uint32_t tasks = plan->items * FFN;
    for (uint32_t task = warp; task < tasks; task += gridDim.x * WARPS) {
        const glmvq_verify_item &item = items[task / FFN];
        const uint32_t row = task % FFN, count = item.count;
        const __nv_bfloat16 *xs[GLMVQ_VER_ROWS];
        #pragma unroll
        for (uint32_t t = 0; t < GLMVQ_VER_ROWS; t++) {
            xs[t] = x + (t < count ? item.pair[t] / GLMVQ_DEC_USED : 0u) * GLMVQ_DEC_HIDDEN;
        }
        const auto *gm = (const uint32_t *)item.gate.main + row * GLMVQ_DEC_HIDDEN * 3 / 64;
        const auto *gp = BITS == 13 ? (const uint32_t *)item.gate.plane + row * GLMVQ_DEC_HIDDEN / 256 : nullptr;
        const auto *um = (const uint32_t *)item.up.main + row * GLMVQ_DEC_HIDDEN * 3 / 64;
        const auto *upl = BITS == 13 ? (const uint32_t *)item.up.plane + row * GLMVQ_DEC_HIDDEN / 256 : nullptr;
        float g[GLMVQ_VER_ROWS], u[GLMVQ_VER_ROWS];
        auto carry = glmvq_decode_load<BITS>(gm, gp);
        glmvq_verify_dot<GLMVQ_DEC_HIDDEN, BITS, GAIN>(item.gate, row, xs, count, (const uint2 *)verify_stage,
            &carry, um, upl, g);
        glmvq_verify_dot<GLMVQ_DEC_HIDDEN, BITS, GAIN>(item.up, row, xs, count, (const uint2 *)verify_stage,
            &carry, nullptr, nullptr, u);
        if (lane) { continue; }
        for (uint32_t t = 0; t < count; t++) {
            mid[(uint64_t)item.pair[t] * FFN + row] = __float2bfloat16_rn(glmvq_swiglu_value<CLAMP>(g[t], u[t], clamp));
        }
    }
}

template <uint32_t BITS, uint32_t WARPS, glmvq_decode_gain GAIN = GLMVQ_GAIN_WEIGHT, uint32_t FFN = GLMVQ_DEC_FFN>
__global__ void __launch_bounds__(WARPS * 32, 1) glmvq_verify_down(
        const glmvq_verify_item *items, const glmvq_verify_plan *plan, const uint4 *book,
        const __nv_bfloat16 *mid, float *partial) {
    extern __shared__ __align__(16) uint8_t verify_stage[];
    constexpr uint32_t book_bytes = (1u << BITS) * GLMVQ_VECTOR;
    auto *x = (__nv_bfloat16 *)(verify_stage + book_bytes);
    glmvq_pdl_open();
    glmvq_decode_book<BITS>((uint4 *)verify_stage, book);
    glmvq_pdl_wait();
    const uint32_t lane = threadIdx.x & 31, warp = blockIdx.x * WARPS + threadIdx.x / 32;
    constexpr uint32_t row16 = FFN * sizeof(__nv_bfloat16) / sizeof(uint4);
    for (uint32_t g = 0; g < plan->groups; g++) {
        const uint32_t first = plan->group[g], last = plan->group[g + 1];
        __syncthreads();
        for (uint32_t i = first; i < last; i++) {
            const glmvq_verify_item &item = items[i];
            for (uint32_t t = 0; t < item.count; t++) {
                const uint4 *src = (const uint4 *)(mid + (uint64_t)item.pair[t] * FFN);
                uint4 *dst = (uint4 *)(x + (uint64_t)(item.slot + t) * FFN);
                for (uint32_t j = threadIdx.x; j < row16; j += blockDim.x) { dst[j] = src[j]; }
            }
        }
        __syncthreads();
        const uint32_t tasks = (last - first) * GLMVQ_DEC_HIDDEN;
        for (uint32_t task = warp; task < tasks; task += gridDim.x * WARPS) {
            const glmvq_verify_item &item = items[first + task / GLMVQ_DEC_HIDDEN];
            const uint32_t row = task % GLMVQ_DEC_HIDDEN, count = item.count;
            const __nv_bfloat16 *xs[GLMVQ_VER_ROWS];
            #pragma unroll
            for (uint32_t t = 0; t < GLMVQ_VER_ROWS; t++) {
                xs[t] = x + (uint64_t)(item.slot + (t < count ? t : 0u)) * FFN;
            }
            const glmvq_matrix m = item.down;
            const auto *main = (const uint32_t *)m.main + row * FFN * 3 / 64;
            const auto *plane = BITS == 13 ? (const uint32_t *)m.plane + row * FFN / 256 : nullptr;
            float value[GLMVQ_VER_ROWS];
            auto carry = glmvq_decode_load<BITS, glmvq_decode_steps<FFN>()>(main, plane);
            glmvq_verify_dot<FFN, BITS, GAIN>(m, row, xs, count, (const uint2 *)verify_stage,
                &carry, nullptr, nullptr, value);
            if (lane) { continue; }
            for (uint32_t t = 0; t < count; t++) {
                partial[(uint64_t)item.pair[t] * GLMVQ_DEC_HIDDEN + row] = value[t];
            }
        }
    }
}

// glmvq_decode_reduce per row: ascending expert order, same roundings.
template <glmvq_reduce REDUCE = GLMVQ_REDUCE_BF16>
__global__ void glmvq_verify_reduce(float *out, const float *partial,
        const float *weights, const glmvq_verify_plan *plan, uint32_t rows, const float *shared = nullptr) {
    glmvq_pdl_open();
    glmvq_pdl_wait();
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * GLMVQ_DEC_HIDDEN) { return; }
    const uint32_t r = i / GLMVQ_DEC_HIDDEN, h = i % GLMVQ_DEC_HIDDEN;
    float sum = 0;
    for (uint32_t e = 0; e < GLMVQ_DEC_USED; e++) {
        const uint32_t pair = r * GLMVQ_DEC_USED + plan->order[r * GLMVQ_DEC_USED + e];
        sum = glmvq_reduce_add<REDUCE>(sum, partial[(uint64_t)pair * GLMVQ_DEC_HIDDEN + h], weights[pair]);
    }
    out[i] = glmvq_reduce_end<REDUCE>(sum, shared, i);
}

// The expert shape and arithmetic a family's routed MoE runs (GLM default).
template <uint32_t FFN_ = GLMVQ_DEC_FFN, glmvq_clamp CLAMP_ = GLMVQ_CLAMP_GATE,
          glmvq_reduce REDUCE_ = GLMVQ_REDUCE_BF16>
struct glmvq_moe_form {
    static constexpr uint32_t FFN = FFN_;
    static constexpr glmvq_clamp CLAMP = CLAMP_;
    static constexpr glmvq_reduce REDUCE = REDUCE_;
};
using glmvq_glm_form = glmvq_moe_form<>;

template <uint32_t BITS, uint32_t WARPS, glmvq_decode_output OUTPUT, glmvq_decode_gain GAIN = GLMVQ_GAIN_WEIGHT,
          typename FORM = glmvq_glm_form>
static cudaError_t glmvq_decode_prepare(uint32_t *blocks) {
    constexpr uint32_t FFN = FORM::FFN;
    int device, shared, sms;
    cudaError_t err = cudaGetDevice(&device);
    if (err == cudaSuccess) { err = cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device); }
    if (err == cudaSuccess) { err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device); }
    if (err != cudaSuccess) { return err; }
    constexpr uint32_t book = (1u << BITS) * GLMVQ_VECTOR;
    constexpr uint32_t gu = book + GLMVQ_DEC_HIDDEN * sizeof(__nv_bfloat16);
    constexpr uint32_t dn = book + GLMVQ_DEC_USED * FFN * sizeof(__nv_bfloat16);
    if (dn > (uint32_t)shared) { return cudaErrorNotSupported; }
    constexpr uint32_t vgu = book + GLMVQ_VER_ROWS * GLMVQ_DEC_HIDDEN * sizeof(__nv_bfloat16);
    constexpr uint32_t vdn = book + GLMVQ_VER_GROUP * FFN * sizeof(__nv_bfloat16);
    if (vgu > (uint32_t)shared || vdn > (uint32_t)shared) { return cudaErrorNotSupported; }
    err = cudaFuncSetAttribute(glmvq_decode_gu<BITS, WARPS, OUTPUT, GAIN, FFN, FORM::CLAMP>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, gu);
    if (err == cudaSuccess) {
        err = cudaFuncSetAttribute(glmvq_decode_down<BITS, WARPS, GAIN, FFN>, cudaFuncAttributeMaxDynamicSharedMemorySize, dn);
    }
    if (err == cudaSuccess) {
        err = cudaFuncSetAttribute(glmvq_verify_gu<BITS, WARPS, GAIN, FFN, FORM::CLAMP>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, vgu);
    }
    if (err == cudaSuccess) {
        err = cudaFuncSetAttribute(glmvq_verify_down<BITS, WARPS, GAIN, FFN>, cudaFuncAttributeMaxDynamicSharedMemorySize, vdn);
    }

    if (err == cudaSuccess) { *blocks = (uint32_t)sms; }
    return err;
}

template <uint32_t BITS, uint32_t WARPS, glmvq_decode_output OUTPUT, glmvq_decode_gain GAIN = GLMVQ_GAIN_WEIGHT,
          typename FORM = glmvq_glm_form>
static cudaError_t glmvq_decode_launch(const glmvq_decode_item *items, const void *book,
        const float *input, __nv_bfloat16 *mid, float *partial, float *out,
        const float *weights, float clamp, uint32_t blocks, cudaStream_t stream,
        float *gate = nullptr, float *up = nullptr, const float *shared = nullptr,
        glmvq_launch launch = GLMVQ_LAUNCH_PLAIN) {
    constexpr uint32_t cb = (1u << BITS) * GLMVQ_VECTOR, FFN = FORM::FFN;
    glmvq_launch_kernel(launch, glmvq_decode_gu<BITS, WARPS, OUTPUT, GAIN, FFN, FORM::CLAMP>, blocks, WARPS * 32,
        cb + GLMVQ_DEC_HIDDEN * sizeof(__nv_bfloat16), stream, items, (const uint4 *)book, input, mid, clamp, gate, up);
    glmvq_launch_kernel(launch, glmvq_decode_down<BITS, WARPS, GAIN, FFN>, blocks, WARPS * 32,
        cb + GLMVQ_DEC_USED * FFN * sizeof(__nv_bfloat16), stream, items, (const uint4 *)book,
        (const __nv_bfloat16 *)mid, partial);
    glmvq_launch_kernel(launch, glmvq_decode_reduce<FORM::REDUCE>, GLMVQ_DEC_HIDDEN / GLMVQ_THREADS, GLMVQ_THREADS,
        0, stream, out, (const float *)partial, weights, items, shared);
    return cudaGetLastError();
}

// Rows 2-4 after glmvq_verify_route filled items and plan on the device.
template <uint32_t BITS, uint32_t WARPS, glmvq_decode_gain GAIN = GLMVQ_GAIN_WEIGHT, typename FORM = glmvq_glm_form>
static cudaError_t glmvq_verify_launch(const glmvq_verify_item *items, const glmvq_verify_plan *plan,
        const void *book, const float *input, uint32_t rows, __nv_bfloat16 *mid, float *partial,
        float *out, const float *weights, float clamp, uint32_t blocks, cudaStream_t stream,
        const float *shared = nullptr, glmvq_launch launch = GLMVQ_LAUNCH_PLAIN) {
    constexpr uint32_t cb = (1u << BITS) * GLMVQ_VECTOR, FFN = FORM::FFN;
    glmvq_launch_kernel(launch, glmvq_verify_gu<BITS, WARPS, GAIN, FFN, FORM::CLAMP>, blocks, WARPS * 32,
        cb + rows * GLMVQ_DEC_HIDDEN * sizeof(__nv_bfloat16), stream, items, plan, (const uint4 *)book, input, rows, mid, clamp);
    glmvq_launch_kernel(launch, glmvq_verify_down<BITS, WARPS, GAIN, FFN>, blocks, WARPS * 32,
        cb + GLMVQ_VER_GROUP * FFN * sizeof(__nv_bfloat16), stream, items, plan, (const uint4 *)book,
        (const __nv_bfloat16 *)mid, partial);
    glmvq_launch_kernel(launch, glmvq_verify_reduce<FORM::REDUCE>, rows * GLMVQ_DEC_HIDDEN / GLMVQ_THREADS,
        GLMVQ_THREADS, 0, stream, out, (const float *)partial, weights, plan, rows, shared);
    return cudaGetLastError();
}

#endif
