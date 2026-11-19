#pragma once
#include <cuda_bf16.h>

enum class SwiGLUOutput { F32, NaiveBF16, Hy4Clamp };

/* Hy4's SwiGLU (hy4::swiglu: limit 10, BF16 boundaries) straight into the
 * Q5_K down's DS4 operand, so the [assignments x width] F32 mid is never
 * written or read back. Lane mapping, shuffle order and rounding mirror
 * quantize_mmq_q8_1<DS4>; the operand is byte-identical. */
static __global__ void hy4_swiglu_q8(
        const float *gate, const float *up, const int32_t *ids,
        block_q8_1_mmq *out, int width, int rows) {
    constexpr float limit = 10.0f;
    const int col = ((int)blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (col >= width) { return; }
    const int sorted = blockIdx.x;
    const size_t at = (size_t)ids[sorted] * width + col;
    const float4 g4 = *(const float4 *)(gate + at);
    const float4 u4 = *(const float4 *)(up + at);
    const float *gp = (const float *)&g4, *upp = (const float *)&u4;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float g = fminf(__bfloat162float(__float2bfloat16(gp[j])), limit);
        const float u = fmaxf(-limit, fminf(__bfloat162float(__float2bfloat16(upp[j])), limit));
        const float silu = __bfloat162float(__float2bfloat16(g * (1.f / (1.f + expf(-g)))));
        v[j] = __bfloat162float(__float2bfloat16(silu * u));
    }
    float amax = fabsf(v[0]);
    amax = fmaxf(amax, fabsf(v[1]));
    amax = fmaxf(amax, fabsf(v[2]));
    amax = fmaxf(amax, fabsf(v[3]));
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset, 32));
    }
    float sum = v[0] + v[1] + v[2] + v[3];
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset, 32);
    }
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(v[0] * d_inv);
    q.y = roundf(v[1] * d_inv);
    q.z = roundf(v[2] * d_inv);
    q.w = roundf(v[3] * d_inv);
    block_q8_1_mmq &block = out[(size_t)(col / 128) * rows + sorted];
    ((char4 *)block.qs)[(col % 128) / 4] = q;
    if (col % 32 == 0) { block.ds4[(col % 128) / 32] = make_half2(1.0f / d_inv, sum); }
}

/* MiMo applies route weights after down projection. Emit unweighted SwiGLU
 * directly in the IQ2_XS consumer's sorted D4 layout, without an F32 mid. */
template<SwiGLUOutput OUTPUT = SwiGLUOutput::F32>
static __global__ void mimo2_swiglu_q8(
        const float *gate, const float *up, const int32_t *ids,
        block_q8_1_mmq *out, int width, int rows) {
    const int col = ((int)blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (col >= width) { return; }
    const int sorted = blockIdx.x;
    const int src = ids[sorted];
    const size_t at = (size_t)src * width + col;
    const float4 g = *(const float4 *)(gate + at);
    const float4 u = *(const float4 *)(up + at);
    const float *gp = (const float *)&g, *uptr = (const float *)&u;
    float value[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        if constexpr (OUTPUT == SwiGLUOutput::NaiveBF16) {
            // Match naive_swiglu's input, SiLU and product BF16 boundaries.
            const float gate = __bfloat162float(__float2bfloat16_rn(gp[j]));
            const float up = __bfloat162float(__float2bfloat16_rn(uptr[j]));
            const float silu = __bfloat162float(__float2bfloat16_rn(
                gate / (1.0f + expf(-gate))));
            value[j] = __bfloat162float(__float2bfloat16_rn(silu * up));
        } else {
            value[j] = (gp[j] / (1.0f + expf(-gp[j]))) * uptr[j];
        }
    }
    float amax = fabsf(value[0]);
#pragma unroll
    for (int j = 1; j < 4; j++) { amax = fmaxf(amax, fabsf(value[j])); }
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset, 32));
    }
    // Preserve quantize_mmq_q8_1<D4>'s scale and rounding, including zeros.
    const float inverse = 127.0f / amax;
    const char4 q = make_char4(roundf(value[0] * inverse), roundf(value[1] * inverse),
                              roundf(value[2] * inverse), roundf(value[3] * inverse));
    block_q8_1_mmq &block = out[(size_t)(col / 128) * rows + sorted];
    ((char4 *)block.qs)[(col % 128) / 4] = q;
    if (col % 32 == 0) { block.d4[(col % 128) / 32] = 1.0f / inverse; }
}
