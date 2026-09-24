#pragma once

/* MiMo applies route weights after down projection. Emit unweighted SwiGLU
 * directly in the IQ2_XS consumer's sorted D4 layout, without an F32 mid. */
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
        value[j] = (gp[j] / (1.0f + expf(-gp[j]))) * uptr[j];
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
