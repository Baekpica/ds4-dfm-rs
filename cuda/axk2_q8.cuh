#ifndef DS4_AXK2_Q8_CUH
#define DS4_AXK2_Q8_CUH

#include "glm_vq.cuh"

/* Q8_0 decode GEMV (MiniMax's mm_q8_gemv order): warp o first stages
 * output o's row in shared memory by 16-byte loads (byte loads per block
 * held MiniMax's at ~17% of the DRAM rate), then lane l takes element l of
 * every 34-byte block (d * q exact in F32) against the F32 input, and a
 * fixed xor tree sums the lanes. */
enum { AX_Q8_BLOCK = 32, AX_Q8_BYTES = 34, AX_Q8_WARPS = 8, AX_TYPE_Q8_0 = 8 };

enum class ax_q8_path { reference, stream };

template <ax_q8_path PATH, uint32_t WARPS = AX_Q8_WARPS>
__global__ void __launch_bounds__(WARPS * 32) ax_q8_gemv(float *out, const uint8_t *weight, const float *x,
        uint32_t outs, uint32_t cols) {
    extern __shared__ __align__(16) uint8_t ax_q8_rows[];
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32, o = blockIdx.x * WARPS + warp;
    const uint32_t blocks = cols / AX_Q8_BLOCK, row_bytes = blocks * AX_Q8_BYTES;
    uint8_t *w = ax_q8_rows + warp * row_bytes;
    glmvq_pdl_open();
    // The weights are constant: they stage while the predecessor drains.
    if (o < outs) {
        const uint4 *src = (const uint4 *)(weight + (uint64_t)o * row_bytes);
        for (uint32_t i = lane; i < row_bytes / 16; i += 32) {
            if constexpr (PATH == ax_q8_path::stream) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                    "r"((uint32_t)__cvta_generic_to_shared((uint4 *)w + i)), "l"(src + i) : "memory");
            } else { ((uint4 *)w)[i] = src[i]; }
        }
    }
    if constexpr (PATH == ax_q8_path::stream) {
        asm volatile("cp.async.commit_group;" ::: "memory");
        glmvq_pdl_wait();
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncwarp();
    } else {
        __syncwarp();
        glmvq_pdl_wait();
    }
    if (o >= outs) { return; }
    float acc = 0.0f;
    // More independent input loads hide the long dot loop's latency.
    constexpr uint32_t UNROLL = PATH == ax_q8_path::stream ? 16 : 4;
    #pragma unroll UNROLL
    for (uint32_t b = 0; b < blocks; b++) {
        const uint8_t *block = w + b * AX_Q8_BYTES;
        const float value = __half2float(__ushort_as_half((uint16_t)(block[0] | block[1] << 8))) *
            (float)(int8_t)block[2 + lane];
        acc = fmaf(value, x[b * AX_Q8_BLOCK + lane], acc);
    }
    for (uint32_t stride = 16; stride; stride >>= 1) { acc += __shfl_xor_sync(0xffffffffu, acc, stride); }
    if (!lane) { out[o] = acc; }
}

#endif
