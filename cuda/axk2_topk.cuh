#ifndef DS4_AXK2_TOPK_CUH
#define DS4_AXK2_TOPK_CUH

#include <cuda_runtime.h>
#include <stdint.h>

/* Exact top-K of each row's scores by radix select, one 1,024-thread block
 * per row. Four 8-bit digit passes (per-warp histograms, so no bin takes
 * every atomic) find the K-th largest key; a counting pass and a writing
 * pass then place the keys above it and, lowest index first, the ties at
 * it. Warp w owns a contiguous slice of the row and walks it 32 keys at a
 * time, so every pass reads coalesced and the output order is the index
 * order within each class. Consumers do not depend on the order.
 *
 *   key(x) = monotone u32 of the float: larger score, larger key
 *   pass:  hist of digit (key >> shift) & 255 among keys matching `prefix`
 *          -> digit d with  #(> d) < need <= #(>= d);  need -= #(> d)
 *
 * The DeepSeek indexer kernels sort (score, index) pairs per chunk and
 * merge; a decode row of 8K keys spent ~0.2 ms per layer there. */
enum { AXK2_TOPK_THREADS = 1024, AXK2_TOPK_WARPS = AXK2_TOPK_THREADS / 32, AXK2_TOPK_BINS = 256 };

__device__ __forceinline__ uint32_t axk2_topk_key(float v) {
    const uint32_t u = __float_as_uint(v);
    return u & 0x80000000u ? ~u : u | 0x80000000u;
}

// Inclusive warp scan; lane 31 holds the total.
__device__ __forceinline__ uint32_t axk2_warp_scan(uint32_t v, uint32_t lane) {
    #pragma unroll
    for (uint32_t s = 1; s < 32; s <<= 1) {
        const uint32_t up = __shfl_up_sync(0xffffffffu, v, s);
        if (lane >= s) { v += up; }
    }
    return v;
}

template <uint32_t K>
__global__ void __launch_bounds__(AXK2_TOPK_THREADS) axk2_topk_kernel(uint32_t *selected, const float *scores,
        uint32_t n, uint32_t stride) {
    __shared__ uint32_t hist[AXK2_TOPK_WARPS][AXK2_TOPK_BINS], total[AXK2_TOPK_BINS];
    __shared__ uint32_t above[AXK2_TOPK_WARPS], equal[AXK2_TOPK_WARPS];
    __shared__ uint32_t chosen_prefix, chosen_need;
    const uint32_t t = threadIdx.x, lane = t & 31u, warp = t >> 5;
    const float *row = scores + (uint64_t)blockIdx.x * stride;
    uint32_t *out = selected + (uint64_t)blockIdx.x * K;
    const uint32_t slice = (n + AXK2_TOPK_WARPS - 1) / AXK2_TOPK_WARPS;
    const uint32_t lo = min(n, warp * slice), hi = min(n, lo + slice);

    uint32_t prefix = 0, mask = 0, need = K;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint32_t b = lane; b < AXK2_TOPK_BINS; b += 32) { hist[warp][b] = 0; }
        __syncwarp();
        for (uint32_t i = lo + lane; i < hi; i += 32) {
            const uint32_t key = axk2_topk_key(row[i]);
            if ((key & mask) == prefix) { atomicAdd(&hist[warp][(key >> shift) & 255u], 1u); }
        }
        __syncthreads();
        if (t < AXK2_TOPK_BINS) {
            uint32_t c = 0;
            for (uint32_t w = 0; w < AXK2_TOPK_WARPS; w++) { c += hist[w][t]; }
            total[t] = c;
        }
        __syncthreads();
        // Warp 0: bins from the top, eight per lane.
        if (warp == 0) {
            uint32_t count[8], local = 0;
            #pragma unroll
            for (uint32_t j = 0; j < 8; j++) {
                count[j] = total[AXK2_TOPK_BINS - 1u - (lane * 8u + j)];
                local += count[j];
            }
            const uint32_t incl = axk2_warp_scan(local, lane), excl = incl - local;
            if (excl < need && need <= incl) {
                uint32_t run = excl;
                #pragma unroll
                for (uint32_t j = 0; j < 8; j++) {
                    if (run + count[j] >= need) {
                        chosen_prefix = prefix | ((AXK2_TOPK_BINS - 1u - (lane * 8u + j)) << shift);
                        chosen_need = need - run;
                        break;
                    }
                    run += count[j];
                }
            }
        }
        __syncthreads();
        prefix = chosen_prefix;
        need = chosen_need;
        mask |= 255u << shift;
    }

    // `prefix` is the K-th largest key; `need` of the keys equal to it go in.
    uint32_t gt = 0, eq = 0;
    for (uint32_t i = lo + lane; i < hi; i += 32) {
        const uint32_t key = axk2_topk_key(row[i]);
        gt += key > prefix;
        eq += key == prefix;
    }
    for (uint32_t s = 16; s; s >>= 1) {
        gt += __shfl_xor_sync(0xffffffffu, gt, s);
        eq += __shfl_xor_sync(0xffffffffu, eq, s);
    }
    if (!lane) { above[warp] = gt; equal[warp] = eq; }
    __syncthreads();
    uint32_t gt_at = 0, eq_at = 0;
    for (uint32_t w = 0; w < warp; w++) { gt_at += above[w]; eq_at += equal[w]; }
    const uint32_t ties_at = K - need;
    for (uint32_t base = lo; base < hi; base += 32) {
        const uint32_t i = base + lane;
        const uint32_t key = i < hi ? axk2_topk_key(row[i]) : 0u;
        const bool g = i < hi && key > prefix, e = i < hi && key == prefix;
        const uint32_t gb = __ballot_sync(0xffffffffu, g), eb = __ballot_sync(0xffffffffu, e);
        const uint32_t below = (1u << lane) - 1u;
        if (g) { out[gt_at + __popc(gb & below)] = i; }
        if (e) {
            const uint32_t rank = eq_at + __popc(eb & below);
            if (rank < need) { out[ties_at + rank] = i; }
        }
        gt_at += __popc(gb);
        eq_at += __popc(eb);
    }
}

#endif
