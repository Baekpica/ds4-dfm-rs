#pragma once
#include <stdint.h>
#include <cuda_bf16.h>

/* Ling-3.0-flash-VL numerical primitives that no existing family supplies:
 * group-limited sigmoid routing, the VL M-RoPE, the fused-row latent RMSNorm,
 * and the BF16 MLA absorb pair.  Everything else in the language graph
 * reuses the Solar/GLM KDA recurrence, the Motif latent attention and the
 * Step routed-MoE operators. */

enum {
    LING_EXPERTS = 512u,
    LING_USED = 8u,
    LING_GROUPS = 8u,
    LING_GROUPS_USED = 4u,
    LING_PER_GROUP = LING_EXPERTS / LING_GROUPS,
    LING_ROUTER_LANES = 32u,
    LING_PER_LANE = LING_EXPERTS / LING_ROUTER_LANES,
    LING_MLA_LAYERS = 7u
};

/* The shared ds4 rule: a higher score wins, and equal scores resolve to the
 * lower index so routing is deterministic across launches. */
__device__ __forceinline__ static bool ling3vl_better(
        float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__device__ __forceinline__ static float4 ling3vl_bf16x4(
        const __nv_bfloat16 *p) {
    const uint2 u = *reinterpret_cast<const uint2 *>(p);
    const float2 lo = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162 *>(&u.x));
    const float2 hi = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162 *>(&u.y));
    return make_float4(lo.x, lo.y, hi.x, hi.y);
}

/* Group-limited sigmoid routing (DeepSeek-V3 "noaux_tc").  The correction
 * bias only selects: it picks the four groups whose two best biased scores
 * sum highest, then the eight best biased experts inside that mask.  The
 * emitted weights are the UNBIASED sigmoid probabilities, renormalized over
 * the selected set and scaled by routed_scaling_factor.
 *
 *   scores   e0 e1 .. e63 | e64 .. e127 | ...   8 groups of 64
 *   group g  -> max1(g) + max2(g)
 *   keep the top 4 groups, mask the rest, then take the top 8 experts. */
__global__ static void ling3vl_router(
        int *ids, float *weights, const float *logits, const float *bias,
        float weight_scale) {
    __shared__ float prob[LING_EXPERTS];
    __shared__ float score[LING_EXPERTS];
    __shared__ float group_score[LING_GROUPS];
    __shared__ int group_keep[LING_GROUPS];
    __shared__ int bad;

    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (!tid) { bad = 0; }
    __syncthreads();

    for (unsigned e = tid; e < LING_EXPERTS; e += blockDim.x) {
        const float x = logits[(uint64_t)row * LING_EXPERTS + e];
        // Use the source sigmoid, including its FP32 overflow/underflow.
        const float p = 1.0f / (1.0f + expf(-x));
        const float s = p + bias[e];
        prob[e] = p;
        score[e] = s;
        if (!isfinite(s)) { bad = 1; }
    }
    __syncthreads();

    if (bad) {
        // Surface an upstream non-finite logit instead of routing on it.
        if (tid < LING_USED) {
            ids[(uint64_t)row * LING_USED + tid] = (int)tid;
            weights[(uint64_t)row * LING_USED + tid] = NAN;
        }
        return;
    }

    if (tid < LING_GROUPS) {
        const float *g = score + tid * LING_PER_GROUP;
        float best = -INFINITY, second = -INFINITY;
        for (unsigned i = 0; i < LING_PER_GROUP; i++) {
            if (g[i] > best) { second = best; best = g[i]; }
            else if (g[i] > second) { second = g[i]; }
        }
        group_score[tid] = best + second;
        group_keep[tid] = 0;
    }
    __syncthreads();

    if (!tid) {
        float remaining[LING_GROUPS];
        for (unsigned g = 0; g < LING_GROUPS; g++) { remaining[g] = group_score[g]; }
        for (unsigned k = 0; k < LING_GROUPS_USED; k++) {
            unsigned best = 0;
            for (unsigned g = 1; g < LING_GROUPS; g++) {
                if (ling3vl_better(remaining[g], g, remaining[best], best)) { best = g; }
            }
            group_keep[best] = 1;
            remaining[best] = -INFINITY;
        }
    }
    __syncthreads();

    if (tid >= LING_ROUTER_LANES) { return; }

    float local_score[LING_PER_LANE];
    float local_prob[LING_PER_LANE];
#pragma unroll
    for (unsigned j = 0; j < LING_PER_LANE; j++) {
        const unsigned e = tid + j * LING_ROUTER_LANES;
        local_prob[j] = prob[e];
        local_score[j] = group_keep[e / LING_PER_GROUP] ? score[e] : -INFINITY;
    }

    float out_prob[LING_USED];
    unsigned out_idx[LING_USED];
    for (unsigned k = 0; k < LING_USED; k++) {
        float best_score = -INFINITY, best_prob = 0.0f;
        unsigned best_idx = UINT32_MAX;
#pragma unroll
        for (unsigned j = 0; j < LING_PER_LANE; j++) {
            const unsigned e = tid + j * LING_ROUTER_LANES;
            if (ling3vl_better(local_score[j], e, best_score, best_idx)) {
                best_score = local_score[j];
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        for (unsigned mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
            const float other_prob = __shfl_xor_sync(0xffffffffu, best_prob, mask);
            const unsigned other_idx = __shfl_xor_sync(0xffffffffu, best_idx, mask);
            if (ling3vl_better(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
#pragma unroll
        for (unsigned j = 0; j < LING_PER_LANE; j++) {
            if (tid + j * LING_ROUTER_LANES == best_idx) { local_score[j] = -INFINITY; }
        }
        out_idx[k] = best_idx;
        out_prob[k] = best_prob;
    }

    if (tid) { return; }
    int *sel = ids + (uint64_t)row * LING_USED;
    float *w = weights + (uint64_t)row * LING_USED;
    float sum = 0.0f;
    for (unsigned k = 0; k < LING_USED; k++) {
        sel[k] = (int)out_idx[k];
        w[k] = out_prob[k];
        sum += out_prob[k];
    }
    // norm_topk_prob, with the source's F16-smallest divisor floor.
    sum = fmaxf(sum, 6.103515625e-5f);
    for (unsigned k = 0; k < LING_USED; k++) { w[k] = w[k] / sum * weight_scale; }
}

/* Interleaved-pair (NORM) rotary over one contiguous slice of each head row.
 *
 * The 32 frequency pairs are split contiguously into T, H and W sections, so
 * text positions -- which set all three axes to the same index -- reduce this
 * to ordinary 1-D RoPE.  Only an image or video span separates the axes.
 * Pairs are adjacent (2p, 2p+1), matching `rope_interleave` on this family,
 * not the half-offset NeoX layout. */
__global__ static void ling3vl_mrope(
        float *x, const int32_t *positions, const float *inv_freq,
        unsigned heads, unsigned head_stride, unsigned offset,
        unsigned half, unsigned section_t, unsigned section_th,
        uint64_t pairs, float attn_factor) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) { return; }
    const unsigned pair = (unsigned)(i % half);
    const uint64_t head_row = i / half;
    const unsigned row = (unsigned)(head_row / heads);
    const unsigned axis = pair < section_t ? 0u : (pair < section_th ? 1u : 2u);
    const double phase =
        (double)positions[(uint64_t)row * 3u + axis] * (double)inv_freq[pair];
    double sine, cosine;
    // Positions reach the YaRN 262144-token cap; reduce the angle in double
    // before narrowing, as the other families' RoPE tables do.
    sincos(phase, &sine, &cosine);
    const float c = (float)cosine * attn_factor, s = (float)sine * attn_factor;
    const uint64_t base = head_row * head_stride + offset + 2u * pair;
    const float x0 = x[base], x1 = x[base + 1u];
    x[base] = x0 * c - x1 * s;
    x[base + 1u] = x0 * s + x1 * c;
}

/* q_nope[head] (qk_nope wide) absorbed through k_b[head] into the latent
 * basis, so attention scores against the stored latent row directly.
 * k_b arrives as [qk_nope, latent, head]: each (head, j) is one contiguous
 * qk_nope-wide BF16 row. */
__global__ static void ling3vl_qk_absorb_bf16(
        float *out, const float *q, const __nv_bfloat16 *k_b,
        unsigned rows, unsigned heads, unsigned key_dim,
        unsigned qk_nope, unsigned latent_dim) {
    extern __shared__ float q_nope[];
    const unsigned head = blockIdx.x, token = blockIdx.y;
    if (head >= heads || token >= rows) { return; }
    const float *qh = q + ((uint64_t)token * heads + head) * key_dim;
    for (unsigned d = threadIdx.x; d < qk_nope; d += blockDim.x) {
        q_nope[d] = qh[d];
    }
    __syncthreads();

    const unsigned warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const unsigned warps = blockDim.x >> 5u;
    const unsigned chunk = qk_nope / 32u / 4u;   /* float4 chunks per lane */
    float *dst = out + ((uint64_t)token * heads + head) * latent_dim;
    for (unsigned j = warp; j < latent_dim; j += warps) {
        const __nv_bfloat16 *row =
            k_b + ((uint64_t)head * latent_dim + j) * qk_nope;
        float sum = 0.0f;
        for (unsigned c = 0; c < chunk; c++) {
            const unsigned d = (c * 32u + lane) * 4u;
            const float4 w = ling3vl_bf16x4(row + d);
            sum += q_nope[d] * w.x + q_nope[d + 1u] * w.y +
                   q_nope[d + 2u] * w.z + q_nope[d + 3u] * w.w;
        }
        for (unsigned off = 16u; off > 0u; off >>= 1u) {
            sum += __shfl_xor_sync(0xffffffffu, sum, off);
        }
        if (!lane) { dst[j] = sum; }
    }
}

/* The attention output leaves the kernel in the latent basis; v_b expands it
 * back to the per-head value width.  v_b arrives as [latent, value, head]:
 * each (head, v) is one contiguous latent-wide BF16 row. */
__global__ static void ling3vl_value_project_bf16(
        float *out, const float *latent, const __nv_bfloat16 *v_b,
        unsigned rows, unsigned heads, unsigned latent_dim,
        unsigned value_dim) {
    extern __shared__ float latent_row[];
    const unsigned head = blockIdx.x, token = blockIdx.y;
    if (head >= heads || token >= rows) { return; }
    const float *src = latent + ((uint64_t)token * heads + head) * latent_dim;
    for (unsigned l = threadIdx.x; l < latent_dim; l += blockDim.x) {
        latent_row[l] = src[l];
    }
    __syncthreads();

    const unsigned warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const unsigned warps = blockDim.x >> 5u;
    const unsigned chunk = latent_dim / 32u / 4u;
    float *dst = out + ((uint64_t)token * heads + head) * value_dim;
    for (unsigned v = warp; v < value_dim; v += warps) {
        const __nv_bfloat16 *row =
            v_b + ((uint64_t)head * value_dim + v) * latent_dim;
        float sum = 0.0f;
        for (unsigned c = 0; c < chunk; c++) {
            const unsigned l = (c * 32u + lane) * 4u;
            const float4 w = ling3vl_bf16x4(row + l);
            sum += latent_row[l] * w.x + latent_row[l + 1u] * w.y +
                   latent_row[l + 2u] * w.z + latent_row[l + 3u] * w.w;
        }
        for (unsigned off = 16u; off > 0u; off >>= 1u) {
            sum += __shfl_xor_sync(0xffffffffu, sum, off);
        }
        if (!lane) { dst[v] = sum; }
    }
}

/* Expanded-MLA prefill: the single rotated key tail is shared by every head,
 * so broadcast k_pe[slot0 + t] into k_full[t][h][qk_nope..key_dim) for all
 * heads.  The GEMM that fills the leading qk_nope columns runs beside it.
 * The scratch is BF16 (a straight copy of the cache row) or FP32. */
template <typename T>
__device__ __forceinline__ static void ling3vl_store_bf16x4(
        T *dst, const __nv_bfloat16 *src);
template <>
__device__ __forceinline__ void ling3vl_store_bf16x4<float>(
        float *dst, const __nv_bfloat16 *src) {
    *reinterpret_cast<float4 *>(dst) = ling3vl_bf16x4(src);
}
template <>
__device__ __forceinline__ void ling3vl_store_bf16x4<__nv_bfloat16>(
        __nv_bfloat16 *dst, const __nv_bfloat16 *src) {
    *reinterpret_cast<uint2 *>(dst) = *reinterpret_cast<const uint2 *>(src);
}

template <typename T>
__global__ static void ling3vl_expand_k_pe(
        T *k_full, const __nv_bfloat16 *k_pe, unsigned slot0,
        unsigned rows, unsigned heads, unsigned key_dim, unsigned qk_nope,
        unsigned qk_rope) {
    const unsigned quads = qk_rope / 4u;
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)rows * heads * quads) { return; }
    const unsigned c = (unsigned)(idx % quads) * 4u;
    const unsigned h = (unsigned)((idx / quads) % heads);
    const unsigned t = (unsigned)(idx / quads / heads);
    ling3vl_store_bf16x4<T>(
        k_full + ((uint64_t)t * heads + h) * key_dim + qk_nope + c,
        k_pe + (uint64_t)(slot0 + t) * qk_rope + c);
}

/* RMSNorm over the leading `dim` of a fused kv_a_mqa row.  The row is
 * 576 wide (512 latent + 64 RoPE); `in_stride` is that fused width.  Using
 * `dim` as the input stride mixes the previous RoPE tail into the next
 * latent on every prefill with n > 1. */
__global__ static void ling3vl_rms_norm(
        float *out, const float *in, const float *weight,
        unsigned dim, unsigned in_stride, unsigned rows, float eps) {
    const unsigned row = blockIdx.x;
    if (row >= rows) { return; }
    const float *src = in + (uint64_t)row * in_stride;
    float *dst = out + (uint64_t)row * dim;
    __shared__ float red[256];
    float acc = 0.0f;
    for (unsigned d = threadIdx.x; d < dim; d += blockDim.x) {
        const float v = src[d];
        acc += v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned off = blockDim.x >> 1u; off; off >>= 1u) {
        if (threadIdx.x < off) { red[threadIdx.x] += red[threadIdx.x + off]; }
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)dim + eps);
    for (unsigned d = threadIdx.x; d < dim; d += blockDim.x) {
        dst[d] = src[d] * inv * weight[d];
    }
}

/* Every MLA block is full attention, so the latent cache is linear: row
 * pos0+t holds the normalized latent and its already-rotated k_pe.  Both are
 * stored BF16, which is what the attention kernel reads. */
__global__ static void ling3vl_store_latent(
        __nv_bfloat16 *latent_cache, __nv_bfloat16 *k_pe_cache,
        const float *kv_norm, const float *kv_raw,
        unsigned rows, unsigned pos0, unsigned cache_cap,
        unsigned kv_raw_dim, unsigned latent_dim, unsigned rope_dim) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t width = latent_dim + rope_dim;
    if (i >= (uint64_t)rows * width) { return; }
    const unsigned token = (unsigned)(i / width);
    const unsigned d = (unsigned)(i % width);
    const unsigned slot = pos0 + token;
    if (slot >= cache_cap) { return; }
    if (d < latent_dim) {
        latent_cache[(uint64_t)slot * latent_dim + d] =
            __float2bfloat16(kv_norm[(uint64_t)token * latent_dim + d]);
        return;
    }
    const unsigned r = d - latent_dim;
    k_pe_cache[(uint64_t)slot * rope_dim + r] = __float2bfloat16(
        kv_raw[(uint64_t)token * kv_raw_dim + latent_dim + r]);
}

/* Qwen3-VL learned position table, bilinearly sampled on its 48x48 grid.
 * Identical to the Qwen4Exp entry except that this artifact keeps the table
 * in F32 rather than Q8_0. */
__global__ static void ling3vl_vision_patch_position(
        float *hidden, const float *bias, const float *position,
        const int32_t *indices, const float *weights,
        unsigned rows, unsigned dim) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t count = (uint64_t)rows * dim;
    if (i >= count) { return; }
    const unsigned row = (unsigned)(i / dim);
    const unsigned d = (unsigned)(i - (uint64_t)row * dim);
    float p = 0.0f;
    for (unsigned k = 0; k < 4u; k++) {
        const unsigned index = (unsigned)indices[4u * row + k];
        p += weights[4u * row + k] * position[(uint64_t)index * dim + d];
    }
    hidden[i] += bias[d] + p;
}
