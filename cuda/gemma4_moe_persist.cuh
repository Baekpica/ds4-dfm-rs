#ifndef DS4_GEMMA4_MOE_PERSIST_CUH
#define DS4_GEMMA4_MOE_PERSIST_CUH

/* Gemma-4 batch-1 persistent layer-book MoE decode.
 *
 * The per-expert decode loop launches gather, gate, up, gelu, down and
 * scatter per expert (48 launches per layer, 1,440 per token) and reads the
 * eight selected routes back to the host once per layer. This header is the
 * one-launch-per-pass replacement: every routed expert of the layer is
 * walked by persistent CTAs that keep the layer's 64 KiB book (and its 2 KiB
 * residual book) in shared memory, the input row is staged once as BF16, and
 * the route order is sorted in shared memory from the device's `selected`
 * row, so no route leaves the device.
 *
 * The shape is MiniMax-M3.1's mm_decode_gu (cuda/minimax_decode.cuh:66-106):
 * 48 CTAs of 16 warps, one CTA per SM, the book staged by the whole block,
 * one warp per (expert, row) task. The arithmetic is not new:
 *   - the index unpack and plane bit are g4_vq_index
 *     (cuda/gemma_vq_decode.cuh:58-66);
 *   - the eight-weight dot is g4_moe_word8 / _res, Gemma's own packed form of
 *     g4_vq_dot_vector (cuda/gemma_vq_decode.cuh:74-93): the same bytes and
 *     rounding, with two E4M3 codewords converted at once (the order
 *     cuda/glm_vq_decode.cuh:99-132 uses, cited, not included);
 *   - the gate/up epilogue is the fused kernel's
 *     round(gelu(gate) * round(up)) (cuda/gemma4_moe_fused.cuh:44);
 *   - the routed sum is the scatter's one rounding per expert, in ascending
 *     expert order (ds4_gemma4_gpu.cuh g4_moe_scatter_kernel), which is what
 *     the reference's index_add_ does.
 *
 * Numerical class: the down pass keeps the current warp-per-row kernel's
 * lane assignment and reduction tree, so each expert's down row and the
 * weighted sum are bit-identical to the per-expert loop. The gate/up pass
 * replaces the block-per-row kernel's 256-thread striding with a warp's
 * 32-lane striding, which reorders the FP32 row sum (Class B, the split-K
 * precedent); the test measures the distance and the model gate bounds it.
 *
 * Shared memory per CTA: gate/up 73,248 B, down 78,880 B, both under the
 * 99 KiB opt-in limit and therefore one CTA per SM. */

#include "gemma_vq_decode.cuh"
#include "gemma4_primitives.cuh"

enum : uint32_t {
    G4_MOE_WARPS = 16,
    G4_MOE_THREADS = G4_MOE_WARPS * 32u,
    G4_MOE_BOOK_BYTES = G4_WORDS * GLMVQ_VECTOR,          /* 65,536 */
    G4_MOE_RES_BYTES = GLMVQ_RES_WORDS * GLMVQ_VECTOR,    /* 2,048 */
    G4_MOE_ORDER_BYTES = G4_USED * sizeof(uint32_t),      /* 32 */
    G4_MOE_GU_STAGE = G4_MOE_BOOK_BYTES + G4_MOE_RES_BYTES +
        G4_EMBED * sizeof(__nv_bfloat16) + G4_MOE_ORDER_BYTES,
    G4_MOE_DOWN_STAGE = G4_MOE_BOOK_BYTES + G4_MOE_RES_BYTES +
        G4_USED * G4_FF * sizeof(__nv_bfloat16) + G4_MOE_ORDER_BYTES,
};

/* The layer's route order, sorted by (expert, slot) and packed as
 * (expert << 8) | slot. Each of the first G4_USED threads counts the routes
 * that precede its own, the rank-count v41_vq_order_kernel uses
 * (cuda/ds41_vq_decode.cuh:189-196); ties keep the slot order, which is the
 * grouped list's order for one token. An out-of-range id (the router never
 * emits one) is treated as expert 0 so no table read can leave the layer. */
__device__ __forceinline__ void g4_moe_order_shared(uint32_t *order, const uint32_t *selected) {
    const uint32_t p = threadIdx.x;
    if (p >= G4_USED) { return; }
    const uint32_t e = selected[p] < G4_EXPERTS ? selected[p] : 0u;
    uint32_t rank = 0;
    #pragma unroll
    for (uint32_t q = 0; q < G4_USED; q++) {
        const uint32_t eq = selected[q] < G4_EXPERTS ? selected[q] : 0u;
        if (eq < e || (eq == e && q < p)) { rank++; }
    }
    order[rank] = (e << 8) | p;
}

/* One 8-weight vector dotted with the staged BF16 row. Same order and
 * rounding as g4_vq_dot_vector, with the book word and the residual word
 * read from shared memory instead of the matrix's global pointers.
 *
 * The two-word arithmetic is Gemma's own copy of the decode dot: two E4M3
 * codewords converted at once (the same values glmvq_e4m3 returns one byte
 * at a time), the gain applied per weight in FP32 and rounded to BF16, then
 * an FMA chain in index order. cuda/gemma_vq_decode.cuh:74-93 is the
 * definition this reproduces; cuda/glm_vq_decode.cuh:99-132 is the same
 * order with the packed conversions. */
__device__ __forceinline__ float2 g4_moe_e4m3x2(uint32_t bits) {
    return __half22float2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)bits, __NV_E4M3));
}

/* The restore's FP32 multiply and add, kept out of fast-math's flush: two
 * E4M3 codewords add exactly, and the gain then scales the exact sum.
 * cuda/glm_vq.cuh:56 and cuda/glm_vq_ops.cuh:12 are the shared forms this
 * reproduces, instruction for instruction; Gemma owns the copies so a
 * sibling's change cannot move this kernel's arithmetic. */
__device__ __forceinline__ float g4_moe_mul(float a, float b) {
    float value;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(value) : "f"(a), "f"(b));
    return value;
}

__device__ __forceinline__ float g4_moe_add(float a, float b) {
    float value;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(value) : "f"(a), "f"(b));
    return value;
}

__device__ __forceinline__ float g4_moe_word8(uint2 word, const uint4 &values,
        float gain, float sum) {
    const uint32_t ws[4] = {word.x, word.x >> 16, word.y, word.y >> 16};
    const uint32_t xs[4] = {values.x, values.y, values.z, values.w};
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        const float2 w = g4_moe_e4m3x2(ws[i]);
        sum = fmaf(g4_vq_bf16(g4_moe_mul(gain, w.x)), __uint_as_float(xs[i] << 16), sum);
        sum = fmaf(g4_vq_bf16(g4_moe_mul(gain, w.y)),
            __uint_as_float(xs[i] & 0xffff0000u), sum);
    }
    return sum;
}

__device__ __forceinline__ float g4_moe_word8_res(uint2 w1, uint2 w2, const uint4 &values,
        float gain, float sum) {
    const uint32_t a[4] = {w1.x, w1.x >> 16, w1.y, w1.y >> 16};
    const uint32_t b[4] = {w2.x, w2.x >> 16, w2.y, w2.y >> 16};
    const uint32_t xs[4] = {values.x, values.y, values.z, values.w};
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        const float2 p = g4_moe_e4m3x2(a[i]), q = g4_moe_e4m3x2(b[i]);
        sum = fmaf(g4_vq_bf16(g4_moe_mul(gain, g4_moe_add(p.x, q.x))),
            __uint_as_float(xs[i] << 16), sum);
        sum = fmaf(g4_vq_bf16(g4_moe_mul(gain, g4_moe_add(p.y, q.y))),
            __uint_as_float(xs[i] & 0xffff0000u), sum);
    }
    return sum;
}

__device__ __forceinline__ float g4_moe_book_dot(const glmvq_matrix &m, uint64_t vector,
        uint32_t index, const uint4 *x, uint32_t vector_col, const uint2 *book,
        const uint2 *res_book) {
    const uint32_t row = (uint32_t)(vector / (m.cols / G4_VQ_VECTOR));
    const float gain = __half2float(m.gain[row]);
    const uint2 word = book[index];
    if (m.res) {
        return g4_moe_word8_res(word, res_book[m.res[vector]], x[vector_col], gain, 0.0f);
    }
    return g4_moe_word8(word, x[vector_col], gain, 0.0f);
}

/* Stage a book (or residual book) into shared as 16-byte words. */
__device__ __forceinline__ void g4_moe_stage_book(uint4 *dst, const uint4 *src,
        uint32_t words) {
    for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) { dst[i] = src[i]; }
}

/* gate/up/gelu for every routed expert in one launch: 8 x 704 row tasks over
 * 48 x 16 warps. `tab` is the layer table (gate[128], up[128], down[128]);
 * the eight active experts are resolved from `selected` in shared memory. */
template <uint32_t WARPS>
__global__ void __launch_bounds__(WARPS * 32, 1) g4_moe_gu_kernel(
        const glmvq_matrix *tab, uint32_t layer, const uint32_t *selected,
        const float *input, __nv_bfloat16 *mid) {
    extern __shared__ __align__(16) uint8_t stage[];
    const glmvq_matrix *gate = tab + (uint64_t)layer * 3u * G4_EXPERTS;
    const glmvq_matrix *up = gate + G4_EXPERTS;
    auto *book = (uint2 *)stage;
    auto *res_book = (uint2 *)(stage + G4_MOE_BOOK_BYTES);
    auto *x = (__nv_bfloat16 *)(stage + G4_MOE_BOOK_BYTES + G4_MOE_RES_BYTES);
    auto *order = (uint32_t *)(stage + G4_MOE_BOOK_BYTES + G4_MOE_RES_BYTES +
        G4_EMBED * sizeof(__nv_bfloat16));
    g4_moe_stage_book((uint4 *)book, (const uint4 *)gate[0].book,
        G4_MOE_BOOK_BYTES / sizeof(uint4));
    g4_moe_stage_book((uint4 *)res_book, (const uint4 *)gate[0].res_book,
        G4_MOE_RES_BYTES / sizeof(uint4));
    for (uint32_t i = threadIdx.x; i < G4_EMBED; i += blockDim.x) {
        x[i] = __float2bfloat16_rn(input[i]);
    }
    g4_moe_order_shared(order, selected);
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = blockIdx.x * WARPS + threadIdx.x / 32u;
    const uint32_t nvec = G4_EMBED / G4_VQ_VECTOR;
    for (uint32_t task = warp; task < G4_USED * G4_FF; task += gridDim.x * WARPS) {
        const uint32_t e = task / G4_FF, row = task % G4_FF;
        const uint32_t expert = order[e] >> 8;
        const glmvq_matrix gm = gate[expert], um = up[expert];
        const uint64_t base = (uint64_t)row * nvec;
        float g = 0.0f, u = 0.0f;
        for (uint32_t j = lane; j < nvec; j += 32u) {
            const uint64_t vector = base + j;
            g += g4_moe_book_dot(gm, vector, g4_vq_index(gm, vector), (const uint4 *)x, j,
                book, res_book);
            u += g4_moe_book_dot(um, vector, g4_vq_index(um, vector), (const uint4 *)x, j,
                book, res_book);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            g += __shfl_down_sync(0xffffffffu, g, off);
            u += __shfl_down_sync(0xffffffffu, u, off);
        }
        if (lane == 0) {
            mid[(uint64_t)e * G4_FF + row] =
                __float2bfloat16_rn(g4_round(g4_gelu(g) * g4_round(u)));
        }
    }
}

/* down and the routed sum for every routed expert in one launch: one warp
 * per output row, the eight experts' down rows walked in ascending expert
 * order and accumulated with the scatter's one rounding per expert. The
 * per-expert dot keeps the warp<8> kernel's lane walk and reduction
 * (cuda/gemma_vq_decode.cuh:95-112), so it is bit-identical to it. */
template <uint32_t WARPS>
__global__ void __launch_bounds__(WARPS * 32, 1) g4_moe_down_kernel(
        const glmvq_matrix *tab, uint32_t layer, const uint32_t *selected,
        const __nv_bfloat16 *mid, const float *weights, float *out) {
    extern __shared__ __align__(16) uint8_t stage[];
    const glmvq_matrix *down = tab + (uint64_t)layer * 3u * G4_EXPERTS + 2u * G4_EXPERTS;
    auto *book = (uint2 *)stage;
    auto *res_book = (uint2 *)(stage + G4_MOE_BOOK_BYTES);
    auto *x = (__nv_bfloat16 *)(stage + G4_MOE_BOOK_BYTES + G4_MOE_RES_BYTES);
    auto *order = (uint32_t *)(stage + G4_MOE_BOOK_BYTES + G4_MOE_RES_BYTES +
        G4_USED * G4_FF * sizeof(__nv_bfloat16));
    g4_moe_stage_book((uint4 *)book, (const uint4 *)down[0].book,
        G4_MOE_BOOK_BYTES / sizeof(uint4));
    g4_moe_stage_book((uint4 *)res_book, (const uint4 *)down[0].res_book,
        G4_MOE_RES_BYTES / sizeof(uint4));
    for (uint32_t i = threadIdx.x; i < G4_USED * G4_FF / 8u; i += blockDim.x) {
        ((uint4 *)x)[i] = ((const uint4 *)mid)[i];
    }
    g4_moe_order_shared(order, selected);
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = blockIdx.x * WARPS + threadIdx.x / 32u;
    const uint32_t nvec = G4_FF / G4_VQ_VECTOR;
    for (uint32_t row = warp; row < G4_EMBED; row += gridDim.x * WARPS) {
        const uint64_t base = (uint64_t)row * nvec;
        float acc = 0.0f;
        #pragma unroll
        for (uint32_t e = 0; e < G4_USED; e++) {
            const uint32_t packed = order[e];
            const glmvq_matrix dm = down[packed >> 8];
            float val = 0.0f;
            for (uint32_t j = lane; j < nvec; j += 32u) {
                const uint64_t vector = base + j;
                val += g4_moe_book_dot(dm, vector, g4_vq_index(dm, vector),
                    (const uint4 *)(x + e * G4_FF), j, book, res_book);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                val += __shfl_down_sync(0xffffffffu, val, off);
            }
            val = __shfl_sync(0xffffffffu, val, 0);
            acc = g4_round(acc + val * weights[packed & 255u]);
        }
        if (lane == 0) { out[row] = acc; }
    }
}

#endif
