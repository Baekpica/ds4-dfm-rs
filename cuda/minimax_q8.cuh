#ifndef DS4_MINIMAX_Q8_CUH
#define DS4_MINIMAX_Q8_CUH

#include <cuda_runtime.h>
#include "../ds4_minimax.h"

// quantize_mmq_q8_1<D4>: four F32 scales and 128 signed codes, tile-major.
struct alignas(16) mm_norm_q8 { float scale[4]; int8_t code[128]; };
static_assert(sizeof(mm_norm_q8) == 144, "MMQ D4 block layout");
enum { MM_NORM_Q8_TILE = 128, MM_NORM_Q8_GROUP = 32, MM_NORM_Q8_SLACK = 128 };

__host__ __device__ constexpr uint64_t mm_norm_q8_bytes(uint32_t rows) {
    return ((uint64_t)rows * (DS4_MINIMAX_HIDDEN / MM_NORM_Q8_TILE) + MM_NORM_Q8_SLACK) * sizeof(mm_norm_q8);
}

// A norm CTA already holds columns tid + j*256. A warp owns one D4 scale;
// finite maxima match the canonical quantizer's eight float4 lanes exactly.
template <uint32_t VALUES>
__device__ __forceinline__ void mm_row_q8(const float *value, mm_norm_q8 *out, uint32_t rows) {
    const uint32_t lane = threadIdx.x & 31;
    #pragma unroll
    for (uint32_t j = 0; j < VALUES; j++) {
        const uint32_t col = threadIdx.x + j * blockDim.x;
        float peak = fabsf(value[j]);
        #pragma unroll
        for (uint32_t stride = 16; stride; stride >>= 1) {
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, stride));
        }
        const float inverse = 127.0f / peak;
        mm_norm_q8 &block = out[(uint64_t)(col / MM_NORM_Q8_TILE) * rows + blockIdx.x];
        block.code[col % MM_NORM_Q8_TILE] = (int8_t)roundf(value[j] * inverse);
        if (!lane) { block.scale[(col % MM_NORM_Q8_TILE) / MM_NORM_Q8_GROUP] = 1.0f / inverse; }
    }
}

#endif
