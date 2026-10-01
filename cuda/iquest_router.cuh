#pragma once
#include "iquest_primitives.cuh"

namespace iq_router {
enum { WARP = 32, VALUES = IQ_EXPERTS / WARP, INVALID = IQ_EXPERTS };

__device__ __forceinline__ static bool better(float score, unsigned id,
                                             float best, unsigned best_id) {
    return id != INVALID && (best_id == INVALID || score > best ||
                             (score == best && id < best_id));
}

__global__ static void select(unsigned *ids, float *weights, const float *logits) {
    const unsigned lane = threadIdx.x, row = blockIdx.x;
    unsigned *selected = ids + (uint64_t)row * IQ_USED;
    float *weight = weights + (uint64_t)row * IQ_USED;
    const float *scores = logits + (uint64_t)row * IQ_EXPERTS;
    float values[VALUES];
    #pragma unroll
    for (unsigned i = 0; i < VALUES; i++) { values[i] = scores[lane + i * WARP]; }
    unsigned used = 0;
    for (unsigned slot = 0; slot < IQ_USED; slot++) {
        unsigned first = INVALID, best = INVALID;
        float first_score = 0, best_score = 0;
        #pragma unroll
        for (unsigned i = 0; i < VALUES; i++) {
            if (used & (1u << i)) { continue; }
            const unsigned id = lane + i * WARP;
            const float value = values[i];
            if (id < first) { first = id; first_score = value; }
            if (!isnan(value) && better(value, id, best_score, best)) {
                best = id; best_score = value;
            }
        }
        // The serial scan keeps an initial NaN but ignores later NaNs.
        // Reduce first-unused and numeric argmax separately to preserve it.
        #pragma unroll
        for (unsigned step = WARP / 2; step; step >>= 1) {
            const unsigned other_first = __shfl_xor_sync(UINT32_MAX, first, step);
            const float other_first_score = __shfl_xor_sync(UINT32_MAX, first_score, step);
            const unsigned other_best = __shfl_xor_sync(UINT32_MAX, best, step);
            const float other_best_score = __shfl_xor_sync(UINT32_MAX, best_score, step);
            if (other_first < first) { first = other_first; first_score = other_first_score; }
            if (better(other_best_score, other_best, best_score, best)) {
                best = other_best; best_score = other_best_score;
            }
        }
        const unsigned chosen = isnan(first_score) ? first : best;
        if (chosen % WARP == lane) { used |= 1u << (chosen / WARP); }
        if (!lane) { selected[slot] = chosen; }
    }
    if (lane) { return; }
    // Keep the original exp, ascending eight-term sum and division order.
    float sum = 0;
    for (unsigned k = 0; k < IQ_USED; k++) {
        weight[k] = expf(scores[selected[k]] - scores[selected[0]]);
        sum += weight[k];
    }
    for (unsigned k = 0; k < IQ_USED; k++) { weight[k] /= sum; }
}
} // namespace iq_router
