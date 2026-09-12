#pragma once
#include <stdint.h>

/* Step 3.7 differs from the legacy pre-SiLU clamp and router sum floor.
 * These operators preserve the published FP32 language graph. */
__global__ static void step37_swiglu(
        float *out, const float *gate, const float *up, const float *weights,
        unsigned width, uint64_t count, float limit) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    float activated = gate[i] / (1.0f + expf(-gate[i]));
    float value = up[i];
    if (limit > 0) {
        activated = fminf(activated, limit);
        value = fminf(fmaxf(value, -limit), limit);
    }
    out[i] = activated * value * (weights ? weights[i / width] : 1.0f);
}

__global__ static void step37_attn_gate(
        float *values, const float *gate, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    values[i] *= 1.0f / (1.0f + expf(-gate[i / 128]));
}

__global__ static void step37_router(
        int *ids, float *weights, const float *logits, const float *bias) {
    enum { EXPERTS = 288, USED = 8 };
    extern __shared__ float scratch[];
    float *prob = scratch, *score = scratch + EXPERTS;
    const unsigned row = blockIdx.x, tid = threadIdx.x;
    for (unsigned e = tid; e < EXPERTS; e += blockDim.x) {
        const float x = logits[row * EXPERTS + e];
        // Use the source sigmoid, including its FP32 overflow/underflow.
        prob[e] = 1.0f / (1.0f + expf(-x));
        score[e] = prob[e] + (bias ? bias[e] : 0.0f);
    }
    __syncthreads();
    if (tid) { return; }
    ids += row * USED;
    weights += row * USED;
    for (unsigned e = 0; e < EXPERTS; e++) {
        if (!isfinite(score[e])) {
            for (unsigned k = 0; k < USED; k++) { ids[k] = k; weights[k] = NAN; }
            return;
        }
    }
    float sum = 0;
    for (unsigned k = 0; k < USED; k++) {
        unsigned best = 0;
        // Make equal-score routing deterministic with ascending expert IDs.
        for (unsigned e = 1; e < EXPERTS; e++) {
            if (score[e] > score[best]) { best = e; }
        }
        ids[k] = best;
        weights[k] = prob[best];
        sum += weights[k];
        score[best] = -INFINITY;
    }
    for (unsigned k = 0; k < USED; k++) { weights[k] = weights[k] / (sum + 1e-20f) * 3.0f; }
}

/* Prepare once per attention geometry, shared by every head and layer.
 * Frequencies incorporate the full-layer factors before upload. Keep the
 * source FP32 phase, then reduce large angles accurately without fast-math
 * intrinsics. The small table avoids repeating trigonometry per head. */
__global__ static void step37_rope_table(
        float2 *table, const float *frequency, const unsigned *positions,
        unsigned half, unsigned rows) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * half) { return; }
    const float phase = positions[i / half] * frequency[i % half];
    double sine, cosine;
    sincos((double)phase, &sine, &cosine);
    table[i] = make_float2((float)cosine, (float)sine);
}

/* One block per token/head. GGUF RMS weights already include the learned +1. */
__global__ static void step37_qk_rope(
        float *out, const float *x, const float *norm, const float2 *table,
        unsigned heads, unsigned rotary) {
    enum { DIM = 128 };
    __shared__ float squares[DIM];
    __shared__ float values[DIM];
    const unsigned d = threadIdx.x, row = blockIdx.x;
    const uint64_t i = (uint64_t)row * DIM + d;
    const float value = x[i];
    squares[d] = value * value;
    __syncthreads();
    for (unsigned step = DIM / 2; step; step /= 2) {
        if (d < step) { squares[d] += squares[d + step]; }
        __syncthreads();
    }
    values[d] = value * rsqrtf(squares[0] / DIM + 1e-5f) * norm[d];
    __syncthreads();
    if (d >= rotary) { out[i] = values[d]; return; }
    const unsigned half = rotary / 2, j = d % half;
    const float2 angle = table[(row / heads) * half + j];
    const unsigned mate = d < half ? d + half : d - half;
    out[i] = values[d] * angle.x + (d < half ? -values[mate] : values[mate]) * angle.y;
}
