#ifndef DS4_MINIMAX_DECODE_CUH
#define DS4_MINIMAX_DECODE_CUH

/* MiniMax-M3.1's routed-expert decode kernels (one token, top-4 experts of
 * 6144 -> 3072 -> 6144 on a 13-bit book, with or without the GVR8 residual
 * stage), shared by ds4_minimax_gpu.cuh and the model-free
 * tests/test_minimax_decode harness. */
#include "glm_vq_decode.cuh"
#include "../ds4_minimax.h"

enum { MM_USED = DS4_MINIMAX_USED, MM_HIDDEN = DS4_MINIMAX_HIDDEN, MM_FFN = DS4_MINIMAX_FFN, MM_WORDS = 8192,
       MM_WARPS = 32 };
static const float MM_LIMIT = 7.0f;

// One 3072-wide down row: a whole 2048-input block, then a 1024-input tail
// of four k-steps. R13+8 rows carry 64 residual words for the block and 32
// for the tail (one byte per vector, row-major), which GLM's whole-block
// helper does not cover. Same lanes, FMA order and gain step as
// glmvq_decode_dot.
template <bool RES>
__device__ __forceinline__ float mm_down_dot(const glmvq_matrix &m, uint32_t row, const __nv_bfloat16 *x,
        const uint2 *book, const uint2 *book2) {
    const uint32_t lane = threadIdx.x & 31, offset = lane * 12 / 32, shift = lane * 12 % 32;
    const float gain = __half2float(m.gain[row]);
    const auto *main = (const uint32_t *)m.main + (uint64_t)row * MM_FFN * 3 / 64;
    const auto *plane = (const uint32_t *)m.plane + (uint64_t)row * MM_FFN / 256;
    const auto *res = RES ? (const uint32_t *)m.res + (uint64_t)row * MM_FFN / 32 : nullptr;
    const glmvq_decode_block head = {main[lane], main[lane + 32], main[lane + 64], lane < 8 ? plane[lane] : 0u,
        RES ? res[lane] : 0u, RES ? res[lane + 32] : 0u};
    const glmvq_decode_block tail = {main[96 + lane], lane < 16 ? main[128 + lane] : 0u, 0u,
        lane < 4 ? plane[8 + lane] : 0u, RES ? res[64 + lane] : 0u, 0u};
    float sum = 0.0f;
    #pragma unroll
    for (uint32_t part = 0; part < 2; part++) {
        const glmvq_decode_block &cur = part ? tail : head;
        const uint32_t steps = part ? 4u : 8u;
        #pragma unroll
        for (uint32_t k = 0; k < 8; k++) {
            if (k >= steps) { break; }
            const uint32_t first = k * 12;
            const uint32_t lr = (first & 31u) > 20 ? (first + 31 - lane) / 32 : first / 32;
            const uint32_t hr = ((first + 1) & 31u) > 20 ? (first + 32 - lane) / 32 : (first + 1) / 32;
            const uint32_t lo = __shfl_sync(0xffffffffu, glmvq_decode_word(cur, lr), first + offset);
            const uint32_t hi = __shfl_sync(0xffffffffu, glmvq_decode_word(cur, hr), first + offset + 1);
            uint32_t index = __funnelshift_r(lo, hi, shift) & 4095u;
            index |= ((__shfl_sync(0xffffffffu, cur.plane, k) >> lane) & 1u) << 12;
            const uint4 values = *(const uint4 *)(x + part * 2048 + k * 256 + lane * GLMVQ_VECTOR);
            if constexpr (RES) {
                sum = glmvq_decode_word8_res<GLMVQ_GAIN_DOT>(book[index], book2[glmvq_decode_res(cur, k)], values,
                    gain, sum);
            } else {
                sum = glmvq_decode_word8<GLMVQ_GAIN_DOT>(book[index], values, gain, sum);
            }
        }
    }
    for (uint32_t stride = 16; stride; stride >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, stride); }
    return glmvq_round(glmvq_mul(sum, gain));
}

// gate/up of the four experts: one warp per (expert, row); the layer book
// (and residual book) stays in shared memory of a persistent CTA.
template <bool RES>
__global__ void __launch_bounds__(MM_WARPS * 32, 1) mm_decode_gu(const glmvq_decode_item *items,
        const uint4 *book, const uint4 *book2, const float *input, __nv_bfloat16 *mid) {
    extern __shared__ __align__(16) uint8_t mm_stage[];
    constexpr uint32_t books = glmvq_decode_books<13, RES>();
    auto *x = (__nv_bfloat16 *)(mm_stage + books);
    const auto *res_book = (const uint2 *)(mm_stage + MM_WORDS * GLMVQ_VECTOR);
    glmvq_pdl_open();
    glmvq_decode_book<13, RES>((uint4 *)mm_stage, book, book2);
    glmvq_pdl_wait();
    for (uint32_t i = threadIdx.x; i < MM_HIDDEN; i += blockDim.x) { x[i] = __float2bfloat16_rn(input[i]); }
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31, warp = blockIdx.x * MM_WARPS + threadIdx.x / 32;
    for (uint32_t task = warp; task < MM_USED * MM_FFN; task += gridDim.x * MM_WARPS) {
        const uint32_t e = task / MM_FFN, row = task % MM_FFN;
        const glmvq_decode_item item = items[e];
        const auto *gm = (const uint32_t *)item.gate.main + (uint64_t)row * MM_HIDDEN * 3 / 64;
        const auto *gp = (const uint32_t *)item.gate.plane + (uint64_t)row * MM_HIDDEN / 256;
        const auto *um = (const uint32_t *)item.up.main + (uint64_t)row * MM_HIDDEN * 3 / 64;
        const auto *up = (const uint32_t *)item.up.plane + (uint64_t)row * MM_HIDDEN / 256;
        const auto *gr = RES ? (const uint32_t *)item.gate.res + (uint64_t)row * MM_HIDDEN / 32 : nullptr;
        const auto *ur = RES ? (const uint32_t *)item.up.res + (uint64_t)row * MM_HIDDEN / 32 : nullptr;
        auto carry = glmvq_decode_load<13, 8, RES>(gm, gp, gr);
        const float g = glmvq_decode_dot<MM_HIDDEN, 13, GLMVQ_GAIN_DOT, RES>(item.gate, row, x,
            (const uint2 *)mm_stage, &carry, um, up, res_book, ur);
        const float u = glmvq_decode_dot<MM_HIDDEN, 13, GLMVQ_GAIN_DOT, RES>(item.up, row, x,
            (const uint2 *)mm_stage, &carry, nullptr, nullptr, res_book, nullptr);
        if (!lane) { mid[task] = __float2bfloat16_rn(glmvq_swiglu_value<GLMVQ_CLAMP_OAI>(g, u, MM_LIMIT)); }
    }
}

// down of the four experts over their staged mid rows (4 x 3072 BF16 fit
// beside the 64 KiB book: 90,112 / 92,160 of 101,376 B).
template <bool RES>
__global__ void __launch_bounds__(MM_WARPS * 32, 1) mm_decode_down(const glmvq_decode_item *items,
        const uint4 *book, const uint4 *book2, const __nv_bfloat16 *mid, float *partial) {
    extern __shared__ __align__(16) uint8_t mm_stage[];
    constexpr uint32_t books = glmvq_decode_books<13, RES>();
    auto *x = (__nv_bfloat16 *)(mm_stage + books);
    const auto *res_book = (const uint2 *)(mm_stage + MM_WORDS * GLMVQ_VECTOR);
    glmvq_pdl_open();
    glmvq_decode_book<13, RES>((uint4 *)mm_stage, book, book2);
    glmvq_pdl_wait();
    for (uint32_t i = threadIdx.x; i < MM_USED * MM_FFN / 8; i += blockDim.x) {
        ((uint4 *)x)[i] = ((const uint4 *)mid)[i];
    }
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31, warp = blockIdx.x * MM_WARPS + threadIdx.x / 32;
    for (uint32_t task = warp; task < MM_USED * MM_HIDDEN; task += gridDim.x * MM_WARPS) {
        const uint32_t e = task / MM_HIDDEN, row = task % MM_HIDDEN;
        const float value = mm_down_dot<RES>(items[e].down, row, x + e * MM_FFN, (const uint2 *)mm_stage, res_book);
        if (!lane) { partial[task] = value; }
    }
}

#endif
