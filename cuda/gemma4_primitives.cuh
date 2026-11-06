#ifndef DS4_GEMMA4_PRIMITIVES_CUH
#define DS4_GEMMA4_PRIMITIVES_CUH

/* Gemma-4 primitives: the operations the shared GVQ runtime does not have.
 * Each one restates the reference (transformers cc1451df, modeling_gemma4.py)
 * and is gated against the matching g4_ref_* function in ds4_gemma4_ref.h.
 *
 * Rounding is part of the contract here: the reference is a stack of bf16
 * elementwise torch ops, so a norm, a rope and the softcap each round more
 * than once, and this file reproduces the same sequence. Do not "optimise"
 * a rounding away without a measurement against the reference. */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../ds4_gemma4_ref.h"

/* Round to nearest even bf16 and return the widened value, which is how a
 * float row carries a rounded tensor in this runtime. */
__device__ __forceinline__ static float g4_round(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

/* A bf16 value's low/high 16 bits widened to float, the two operations a
 * uint4 weight load turns into eight floats. */
__device__ __forceinline__ static float g4_bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ static float g4_bf16_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

/* The reference applies gelu in float32 with torch's accurate tanh. This
 * file is built with --use_fast_math, which would route tanhf to the fast
 * approximation and drift past a bf16 rounding; the double transcendental
 * then a float cast keeps the result within the reference's own accuracy.
 * Same reasoning for the rope's cos/sin, the softcap and the softmax. */
__device__ __forceinline__ static float g4_gelu(float x) {
    const float v = g4_round(x);
    const float cube = 0.044715f * v * v * v;
    return g4_round(0.5f * v * (1.0f + (float)tanh(0.7978845608028654 * (double)(v + cube))));
}

/* RMSNorm: one block per row, float32 square sum, optional weight (3.2).
 * The reference computes pow(mean, -0.5) in float32; rsqrtf differs by ULPs,
 * which the golden tolerance absorbs.
 *
 * The weight is the container's BF16 tensor (input_layernorm.weight is
 * type 30 in the artifact), so the kernel widens it per element. Reading it
 * as float32 gives NaN: the first half of the row then looks plausible
 * (a float whose high half is the bf16 weight) and the rest reads past the
 * tensor into its neighbours. */
template <unsigned BLOCK>
__global__ static void g4_rms_kernel(float *out, const float *in, const __nv_bfloat16 *weight,
        uint32_t rows, uint32_t width, float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) { return; }
    const float *x = in + (uint64_t)row * width;
    float sum = 0.0f;
    for (uint32_t d = threadIdx.x; d < width; d += BLOCK) {
        const float v = x[d];
        sum += v * v;
    }
    __shared__ float scratch[BLOCK];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = BLOCK / 2u; stride; stride >>= 1) {
        if (threadIdx.x < stride) { scratch[threadIdx.x] += scratch[threadIdx.x + stride]; }
        __syncthreads();
    }
    const float inv = (float)(1.0 / sqrt((double)(scratch[0] / (float)width + eps)));
    __syncthreads();
    float *o = out + (uint64_t)row * width;
    for (uint32_t d = threadIdx.x; d < width; d += BLOCK) {
        float y = x[d] * inv;
        if (weight) { y *= __bfloat162float(weight[d]); }
        o[d] = g4_round(y);
    }
}

/* The router's input transform: RMS without a weight, then the per-dimension
 * router.scale (a bf16 product, so a rounding), then hidden_size**-0.5 in
 * float32 with one final rounding (3.8). The scale is BF16 in the container. */
template <unsigned BLOCK>
__global__ static void g4_router_input_kernel(float *out, const float *in, const __nv_bfloat16 *scale,
        uint32_t rows, uint32_t width, float eps, float root) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) { return; }
    const float *x = in + (uint64_t)row * width;
    float sum = 0.0f;
    for (uint32_t d = threadIdx.x; d < width; d += BLOCK) {
        const float v = x[d];
        sum += v * v;
    }
    __shared__ float scratch[BLOCK];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = BLOCK / 2u; stride; stride >>= 1) {
        if (threadIdx.x < stride) { scratch[threadIdx.x] += scratch[threadIdx.x + stride]; }
        __syncthreads();
    }
    const float inv = (float)(1.0 / sqrt((double)(scratch[0] / (float)width + eps)));
    __syncthreads();
    float *o = out + (uint64_t)row * width;
    for (uint32_t d = threadIdx.x; d < width; d += BLOCK) {
        const float normed = g4_round(x[d] * inv);
        const float scaled = g4_round(normed * __bfloat162float(scale[d]));
        o[d] = g4_round(scaled * root);
    }
}

/* Half-split RoPE over [rows, heads, head_dim]. `inv` holds head_dim/2
 * inverse frequencies with zeros beyond the rotating pairs, so the same
 * kernel serves both geometries; cos/sin and every product round to bf16,
 * as in the reference (3.3, 3.5). */
__global__ static void g4_rope_kernel(float *x, const uint32_t *positions, const float *inv,
        uint32_t rows, uint32_t heads, uint32_t head_dim, uint32_t pairs) {
    const uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (uint64_t)rows * heads) { return; }
    const uint32_t row = (uint32_t)(index / heads);
    const uint32_t position = positions[row];
    const uint32_t half = head_dim / 2u;
    float *v = x + index * head_dim;
    for (uint32_t i = 0; i < pairs; i++) {
        const float angle = (float)((double)position * (double)inv[i]);
        const float c = g4_round((float)cos((double)angle));
        const float s = g4_round((float)sin((double)angle));
        const float a = v[i], b = v[i + half];
        v[i] = g4_round(g4_round(a * c) + g4_round(-b * s));
        v[i + half] = g4_round(g4_round(b * c) + g4_round(a * s));
    }
}

/* Fused RMSNorm + RoPE, one block per (row, head). The norm is g4_rms_kernel's
 * exact reduction and rounding, staged in shared; the rope pairs then rotate
 * from the staged row with g4_rope_kernel's exact expression, so the result is
 * the two-kernel sequence bit for bit while the row is read and written once.
 * `pairs` is the rotary pair count (full for the sliding geometry, 64 of 256
 * for the full layers); indices outside the rotated ranges keep the normalized
 * value. */
template <unsigned BLOCK>
__global__ static void g4_rms_rope_kernel(float *out, const float *in, const __nv_bfloat16 *weight,
        const uint32_t *positions, const float *inv, uint32_t rows, uint32_t heads,
        uint32_t dim, uint32_t pairs, float eps) {
    const uint32_t head = blockIdx.x, row = blockIdx.y;
    if (head >= heads || row >= rows) { return; }
    const uint64_t offset = ((uint64_t)row * heads + head) * dim;
    const float *x = in + offset;
    float sum = 0.0f;
    for (uint32_t d = threadIdx.x; d < dim; d += BLOCK) {
        const float v = x[d];
        sum += v * v;
    }
    /* BLOCK entries serve the reduction; dim up to 512 serves the staging. */
    __shared__ float scratch[BLOCK > 512u ? BLOCK : 512u];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = BLOCK / 2u; stride; stride >>= 1) {
        if (threadIdx.x < stride) { scratch[threadIdx.x] += scratch[threadIdx.x + stride]; }
        __syncthreads();
    }
    const float inv_norm = (float)(1.0 / sqrt((double)(scratch[0] / (float)dim + eps)));
    __syncthreads();
    float *o = out + offset;
    for (uint32_t d = threadIdx.x; d < dim; d += BLOCK) {
        float y = x[d] * inv_norm;
        if (weight) { y *= __bfloat162float(weight[d]); }
        scratch[d] = g4_round(y);
    }
    __syncthreads();
    const uint32_t half = dim / 2u;
    const uint32_t t = threadIdx.x;
    for (uint32_t i = t; i < pairs; i += BLOCK) {
        const float angle = (float)((double)positions[row] * (double)inv[i]);
        const float c = g4_round((float)cos((double)angle));
        const float s = g4_round((float)sin((double)angle));
        const float a = scratch[i], b = scratch[i + half];
        o[i] = g4_round(g4_round(a * c) + g4_round(-b * s));
        o[i + half] = g4_round(g4_round(b * c) + g4_round(a * s));
    }
    for (uint32_t d = t; d < dim; d += BLOCK) {
        if (d >= pairs && (d < half || d >= half + pairs)) { o[d] = scratch[d]; }
    }
}

/* act(gate) * up with the reference's two roundings (3.7). */
__global__ static void g4_gelu_kernel(float *out, const float *gate, const float *up,
        uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    out[i] = g4_round(g4_gelu(gate[i]) * g4_round(up[i]));
}

/* The embedding lookup's scale: a bf16 53.0 (3.1). */
__global__ static void g4_embed_kernel(float *out, const float *rows, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    out[i] = g4_round(g4_round(rows[i]) * G4_REF_EMBED_SCALE_BF16);
}

/* tanh(logits/cap)*cap on packed positions, one rounding per op (3.9). */
__global__ static void g4_softcap_kernel(float *logits, uint64_t count, float cap) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const float divided = g4_round(logits[i] / cap);
    const float squashed = g4_round((float)tanh((double)divided));
    logits[i] = g4_round(squashed * cap);
}

/* The block output scale, a bf16 buffer value (3.6). */
__global__ static void g4_layer_scale_kernel(float *x, uint64_t count, float scalar) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    x[i] = g4_round(x[i] * g4_round(scalar));
}

/* Router selection: one block per row with one thread per expert (128), so
 * the reductions are exact block scans. Softmax in float32 over the bf16
 * logits, top-k keeping the lower index on ties, renormalise, then
 * per_expert_scale (3.8, a BF16 tensor in the container).
 *
 * PAR replaces the one-thread top-k scan with k block max passes: every
 * thread carries its probability and index through the same tree, the winner
 * is marked used, and the k values are summed in pick order. The selected
 * ids, the weights and their order are the serial scan's, so the route is
 * bit-identical. */
template <bool PAR>
__global__ static void g4_router_kernel(uint32_t *selected, float *weights, const float *logits,
        const __nv_bfloat16 *per_expert, uint32_t rows, uint32_t experts, uint32_t k) {
    const uint32_t row = blockIdx.x;
    const uint32_t t = threadIdx.x;
    if (row >= rows || t >= experts) { return; }
    const float *row_logits = logits + (uint64_t)row * experts;
    __shared__ float scratch[G4_REF_EXPERTS];
    __shared__ float p[G4_REF_EXPERTS];
    __shared__ uint32_t picked[G4_REF_USED];
    __shared__ float picked_p[G4_REF_USED];
    __shared__ uint32_t order[G4_REF_EXPERTS];
    const float local = g4_round(row_logits[t]);
    scratch[t] = local;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2u; stride; stride >>= 1) {
        if (t < stride) { scratch[t] = fmaxf(scratch[t], scratch[t + stride]); }
        __syncthreads();
    }
    const float maximum = scratch[0];
    __syncthreads();
    const float e = (float)exp((double)(local - maximum));
    scratch[t] = e;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2u; stride; stride >>= 1) {
        if (t < stride) { scratch[t] += scratch[t + stride]; }
        __syncthreads();
    }
    p[t] = e / scratch[0];
    __syncthreads();
    if constexpr (PAR) {
        for (uint32_t s = 0; s < k; s++) {
            scratch[t] = p[t];
            order[t] = t;
            __syncthreads();
            for (unsigned stride = blockDim.x / 2u; stride; stride >>= 1) {
                if (t < stride) {
                    const float a = scratch[t], b = scratch[t + stride];
                    const uint32_t ia = order[t], ib = order[t + stride];
                    if (b > a || (b == a && ib < ia)) { scratch[t] = b; order[t] = ib; }
                }
                __syncthreads();
            }
            if (t == 0) {
                picked[s] = order[0];
                picked_p[s] = scratch[0];
                p[order[0]] = -1.0f;
            }
            __syncthreads();
        }
        if (t == 0) {
            float sum = 0.0f;
            for (uint32_t s = 0; s < k; s++) { sum += picked_p[s]; }
            uint32_t *out_index = selected + (uint64_t)row * k;
            float *out_weight = weights + (uint64_t)row * k;
            for (uint32_t s = 0; s < k; s++) {
                out_index[s] = picked[s];
                out_weight[s] = (picked_p[s] / sum) * g4_round(__bfloat162float(per_expert[picked[s]]));
            }
        }
    } else {
        if (t == 0) {
            for (uint32_t s = 0; s < k; s++) {
                uint32_t best = UINT32_MAX;
                for (uint32_t x = 0; x < experts; x++) {
                    int taken = 0;
                    for (uint32_t j = 0; j < s; j++) { if (picked[j] == x) { taken = 1; } }
                    if (!taken && (best == UINT32_MAX || p[x] > p[best])) { best = x; }
                }
                picked[s] = best;
            }
            float sum = 0.0f;
            for (uint32_t s = 0; s < k; s++) { sum += p[picked[s]]; }
            uint32_t *out_index = selected + (uint64_t)row * k;
            float *out_weight = weights + (uint64_t)row * k;
            for (uint32_t s = 0; s < k; s++) {
                out_index[s] = picked[s];
                out_weight[s] = (p[picked[s]] / sum) * g4_round(__bfloat162float(per_expert[picked[s]]));
            }
        }
    }
}

#endif
