#ifndef DS4_GLM_VQ_DECODE_CUH
#define DS4_GLM_VQ_DECODE_CUH

#include "glm_vq_ops.cuh"

enum { GLMVQ_DEC_HIDDEN = 4096, GLMVQ_DEC_FFN = 2048, GLMVQ_DEC_USED = 8 };
enum glmvq_decode_output { GLMVQ_DEC_NORMAL, GLMVQ_DEC_PROOF };

struct glmvq_decode_item {
    glmvq_matrix gate, up, down;
    uint32_t pair;
};

struct glmvq_decode_block { uint32_t a, b, c, plane; };

// YoungAi 7fb26cc cuda_vq_row.inc.cu:146-280: a warp loads three complete
// 128-byte lines (256 indices), with a separate 32-byte thirteenth plane.
template <uint32_t BITS>
__device__ __forceinline__ glmvq_decode_block glmvq_decode_load(
        const uint32_t *main, const uint32_t *plane) {
    const uint32_t lane = threadIdx.x & 31;
    return {main[lane], main[lane + 32], main[lane + 64],
        BITS == 13 && lane < 8 ? plane[lane] : 0u};
}

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

template <uint32_t COLS, uint32_t BITS>
__device__ __forceinline__ float glmvq_decode_dot(const glmvq_matrix &m,
        uint32_t row, const __nv_bfloat16 *x, const uint2 *book,
        glmvq_decode_block *carry, const uint32_t *next, const uint32_t *next_plane) {
    const uint32_t lane = threadIdx.x & 31, offset = lane * 12 / 32, shift = lane * 12 % 32;
    const auto *main = (const uint32_t *)m.main + (uint64_t)row * COLS * 3 / 64;
    const auto *plane = BITS == 13 ? (const uint32_t *)m.plane + (uint64_t)row * COLS / 256 : nullptr;
    const float gain = __half2float(m.gain[row]);
    glmvq_decode_block cur = *carry;
    float sum = 0;
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
            sum = glmvq_decode_pair(word.x, values.x, gain, sum);
            sum = glmvq_decode_pair(word.x >> 16, values.y, gain, sum);
            sum = glmvq_decode_pair(word.y, values.z, gain, sum);
            sum = glmvq_decode_pair(word.y >> 16, values.w, gain, sum);
        }
        cur = ahead;
    }
    *carry = cur;
    for (uint32_t stride = 16; stride; stride >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, stride); }
    return glmvq_round(sum);
}

template <uint32_t BITS>
__device__ __forceinline__ void glmvq_decode_book(uint4 *dst, const uint4 *src) {
    for (uint32_t i = threadIdx.x; i < (1u << BITS) * GLMVQ_VECTOR / sizeof(uint4); i += blockDim.x) {
        dst[i] = src[i];
    }
}

// YoungAi cuda_vq_persist.inc.cu:1-210: each persistent CTA retains one
// layer book. E4M3 stays packed: even the 13-bit book needs only 64 KiB.
template <uint32_t BITS, uint32_t WARPS, glmvq_decode_output OUTPUT>
__global__ void __launch_bounds__(WARPS * 32, 1) glmvq_decode_gu(
        const glmvq_decode_item *items, const uint4 *book, const float *input,
        __nv_bfloat16 *mid, float clamp, float *gate, float *up) {
    extern __shared__ __align__(16) uint8_t decode_stage[];
    constexpr uint32_t book_bytes = (1u << BITS) * GLMVQ_VECTOR;
    auto *x = (__nv_bfloat16 *)(decode_stage + book_bytes);
    glmvq_decode_book<BITS>((uint4 *)decode_stage, book);
    for (uint32_t i = threadIdx.x; i < GLMVQ_DEC_HIDDEN; i += blockDim.x) {
        x[i] = __float2bfloat16_rn(input[i]);
    }
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31, warp = blockIdx.x * WARPS + threadIdx.x / 32;
    for (uint32_t task = warp; task < GLMVQ_DEC_USED * GLMVQ_DEC_FFN; task += gridDim.x * WARPS) {
        const uint32_t e = task / GLMVQ_DEC_FFN, row = task % GLMVQ_DEC_FFN;
        const glmvq_decode_item item = items[e];
        const auto *gm = (const uint32_t *)item.gate.main + row * GLMVQ_DEC_HIDDEN * 3 / 64;
        const auto *gp = BITS == 13 ? (const uint32_t *)item.gate.plane + row * GLMVQ_DEC_HIDDEN / 256 : nullptr;
        const auto *um = (const uint32_t *)item.up.main + row * GLMVQ_DEC_HIDDEN * 3 / 64;
        const auto *up_plane = BITS == 13 ? (const uint32_t *)item.up.plane + row * GLMVQ_DEC_HIDDEN / 256 : nullptr;
        auto carry = glmvq_decode_load<BITS>(gm, gp);
        const float g = glmvq_decode_dot<GLMVQ_DEC_HIDDEN, BITS>(item.gate, row, x,
            (const uint2 *)decode_stage, &carry, um, up_plane);
        const float u = glmvq_decode_dot<GLMVQ_DEC_HIDDEN, BITS>(item.up, row, x,
            (const uint2 *)decode_stage, &carry, nullptr, nullptr);
        if (!lane) {
            mid[task] = __float2bfloat16_rn(glmvq_swiglu_value(g, u, clamp));
            if constexpr (OUTPUT == GLMVQ_DEC_PROOF) { gate[task] = g; up[task] = u; }
        }
    }
}

template <uint32_t BITS, uint32_t WARPS>
__global__ void __launch_bounds__(WARPS * 32, 1) glmvq_decode_down(
        const glmvq_decode_item *items, const uint4 *book, const __nv_bfloat16 *mid, float *output) {
    extern __shared__ __align__(16) uint8_t decode_stage[];
    constexpr uint32_t book_bytes = (1u << BITS) * GLMVQ_VECTOR;
    auto *x = (__nv_bfloat16 *)(decode_stage + book_bytes);
    glmvq_decode_book<BITS>((uint4 *)decode_stage, book);
    for (uint32_t i = threadIdx.x; i < GLMVQ_DEC_USED * GLMVQ_DEC_FFN / 8; i += blockDim.x) {
        ((uint4 *)x)[i] = ((const uint4 *)mid)[i];
    }
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31, warp = blockIdx.x * WARPS + threadIdx.x / 32;
    for (uint32_t task = warp; task < GLMVQ_DEC_USED * GLMVQ_DEC_HIDDEN; task += gridDim.x * WARPS) {
        const uint32_t e = task / GLMVQ_DEC_HIDDEN, row = task % GLMVQ_DEC_HIDDEN;
        const glmvq_matrix m = items[e].down;
        const auto *main = (const uint32_t *)m.main + row * GLMVQ_DEC_FFN * 3 / 64;
        const auto *plane = BITS == 13 ? (const uint32_t *)m.plane + row * GLMVQ_DEC_FFN / 256 : nullptr;
        auto carry = glmvq_decode_load<BITS>(main, plane);
        const float value = glmvq_decode_dot<GLMVQ_DEC_FFN, BITS>(m, row,
            x + e * GLMVQ_DEC_FFN, (const uint2 *)decode_stage, &carry, nullptr, nullptr);
        if (!lane) { output[task] = value; }
    }
}

__global__ void glmvq_decode_reduce(float *out, const float *partial,
        const float *weights, const glmvq_decode_item *items) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= GLMVQ_DEC_HIDDEN) { return; }
    float sum = 0;
    // Items are in ascending expert order, matching the existing scatter.
    for (uint32_t e = 0; e < GLMVQ_DEC_USED; e++) {
        const float value = glmvq_round(glmvq_mul(partial[e * GLMVQ_DEC_HIDDEN + row], weights[items[e].pair]));
        sum = glmvq_round(glmvq_add(sum, value));
    }
    out[row] = sum;
}

template <uint32_t BITS, uint32_t WARPS, glmvq_decode_output OUTPUT>
static cudaError_t glmvq_decode_prepare(uint32_t *blocks) {
    int device, shared, sms;
    cudaError_t err = cudaGetDevice(&device);
    if (err == cudaSuccess) { err = cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device); }
    if (err == cudaSuccess) { err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device); }
    if (err != cudaSuccess) { return err; }
    constexpr uint32_t book = (1u << BITS) * GLMVQ_VECTOR;
    constexpr uint32_t gu = book + GLMVQ_DEC_HIDDEN * sizeof(__nv_bfloat16);
    constexpr uint32_t dn = book + GLMVQ_DEC_USED * GLMVQ_DEC_FFN * sizeof(__nv_bfloat16);
    if (dn > (uint32_t)shared) { return cudaErrorNotSupported; }
    err = cudaFuncSetAttribute(glmvq_decode_gu<BITS, WARPS, OUTPUT>, cudaFuncAttributeMaxDynamicSharedMemorySize, gu);
    if (err == cudaSuccess) {
        err = cudaFuncSetAttribute(glmvq_decode_down<BITS, WARPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, dn);
    }
    if (err == cudaSuccess) { *blocks = (uint32_t)sms; }
    return err;
}

template <uint32_t BITS, uint32_t WARPS, glmvq_decode_output OUTPUT>
static cudaError_t glmvq_decode_launch(const glmvq_decode_item *items, const void *book,
        const float *input, __nv_bfloat16 *mid, float *partial, float *out,
        const float *weights, float clamp, uint32_t blocks, cudaStream_t stream,
        float *gate = nullptr, float *up = nullptr) {
    constexpr uint32_t cb = (1u << BITS) * GLMVQ_VECTOR;
    glmvq_decode_gu<BITS, WARPS, OUTPUT><<<blocks, WARPS * 32,
        cb + GLMVQ_DEC_HIDDEN * sizeof(__nv_bfloat16), stream>>>(items, (const uint4 *)book, input, mid, clamp, gate, up);
    glmvq_decode_down<BITS, WARPS><<<blocks, WARPS * 32,
        cb + GLMVQ_DEC_USED * GLMVQ_DEC_FFN * sizeof(__nv_bfloat16), stream>>>(items, (const uint4 *)book, mid, partial);
    glmvq_decode_reduce<<<GLMVQ_DEC_HIDDEN / GLMVQ_THREADS, GLMVQ_THREADS, 0, stream>>>(out, partial, weights, items);
    return cudaGetLastError();
}

#endif
