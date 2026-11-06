#ifndef DS4_GEMMA4_REF_H
#define DS4_GEMMA4_REF_H

/* Host reference for the Gemma-4 primitives, in double, transcribed from
 * transformers cc1451df (modeling_gemma4.py). Every device kernel is gated
 * against these functions, and each function restates the reference's
 * rounding order, because several of them round more than once (a bf16
 * elementwise op per torch call).
 *
 * Nothing here is a performance path: it is the golden-vector source. */

#include <math.h>
#include <stdint.h>
#include <string.h>

enum { G4_REF_EMBED = 2816, G4_REF_VOCAB = 262144, G4_REF_HEADS = 16,
       G4_REF_KV_HEADS = 8, G4_REF_GLOBAL_KV_HEADS = 2, G4_REF_HEAD_DIM = 256,
       G4_REF_GLOBAL_HEAD_DIM = 512, G4_REF_ROT_PAIRS = 64, G4_REF_EXPERTS = 128,
       G4_REF_USED = 8, G4_REF_FF = 704, G4_REF_DENSE = 2112, G4_REF_WINDOW = 1024 };

/* The embedding's scale is a bf16 tensor holding sqrt(2816), so the factor is
 * 53.0 exactly, not 53.065996645686 (semantics 3.1). */
#define G4_REF_SQRT2_OVER_PI 0.7978845608028654
#define G4_REF_EMBED_SCALE_BF16 53.0f
#define G4_REF_SOFTCAP 30.0

static inline float g4_ref_bf16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    memcpy(&x, &bits, sizeof(bits));
    return x;
}

/* gelu(approximate="tanh"), then one rounding (semantics 3.7). */
static inline float g4_ref_gelu(float x) {
    const double v = (double)x;
    const double cube = 0.044715 * v * v * v;
    return g4_ref_bf16((float)(0.5 * v * (1.0 + tanh(G4_REF_SQRT2_OVER_PI * (v + cube)))));
}

/* RMSNorm in float32, weight applied in float32, one rounding (3.2).
 * weight may be NULL (v_norm, router norm). */
static inline void g4_ref_rms(float *out, const float *in, const float *weight,
        uint32_t rows, uint32_t width, float eps) {
    for (uint32_t r = 0; r < rows; r++) {
        const float *x = in + (uint64_t)r * width;
        double sum = 0.0;
        for (uint32_t d = 0; d < width; d++) { sum += (double)x[d] * (double)x[d]; }
        const double inv = pow(sum / width + eps, -0.5);
        for (uint32_t d = 0; d < width; d++) {
            double y = (double)x[d] * inv;
            if (weight) { y *= (double)weight[d]; }
            out[(uint64_t)r * width + d] = g4_ref_bf16((float)y);
        }
    }
}

/* The router's input transform: RMS without a weight, one rounding; times
 * router.scale, a rounding; times hidden_size**-0.5 in float32, a rounding
 * (3.8). */
static inline void g4_ref_router_input(float *out, const float *in, const float *scale,
        uint32_t rows, uint32_t width, float eps, float root) {
    for (uint32_t r = 0; r < rows; r++) {
        const float *x = in + (uint64_t)r * width;
        double sum = 0.0;
        for (uint32_t d = 0; d < width; d++) { sum += (double)x[d] * (double)x[d]; }
        const double inv = pow(sum / width + eps, -0.5);
        for (uint32_t d = 0; d < width; d++) {
            const float normed = g4_ref_bf16((float)((double)x[d] * inv));
            const float scaled = g4_ref_bf16(normed * scale[d]);
            out[(uint64_t)r * width + d] = g4_ref_bf16((float)((double)scaled * root));
        }
    }
}

/* inv_freq for one layer type (3.5). pairs receives the number of rotating
 * half-pairs: head_dim/2 for the default rope, 64 for the proportional one.
 * Writes pairs entries and zeroes the rest up to head_dim/2. */
static inline uint32_t g4_ref_rope_inv(float *inv, uint32_t head_dim, int full, double theta) {
    const uint32_t pairs = full ? G4_REF_ROT_PAIRS : head_dim / 2u;
    for (uint32_t i = 0; i < head_dim / 2u; i++) {
        inv[i] = i < pairs ? (float)(1.0 / pow(theta, (double)(2u * i) / head_dim)) : 0.0f;
    }
    return pairs;
}

/* One rope application, half-split, cos/sin rounded to bf16 and every product
 * rounded to bf16 (3.3, 3.5). */
static inline void g4_ref_rope(float *x, uint32_t rows, uint32_t heads,
        uint32_t head_dim, const uint32_t *positions, const float *inv, uint32_t pairs) {
    const uint32_t half = head_dim / 2u;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t h = 0; h < heads; h++) {
            float *v = x + ((uint64_t)r * heads + h) * head_dim;
            for (uint32_t i = 0; i < pairs; i++) {
                const float angle = (float)((double)positions[r] * (double)inv[i]);
                const float c = g4_ref_bf16(cosf(angle));
                const float s = g4_ref_bf16(sinf(angle));
                const float a = v[i], b = v[i + half];
                v[i] = g4_ref_bf16(g4_ref_bf16(a * c) + g4_ref_bf16(-b * s));
                v[i + half] = g4_ref_bf16(g4_ref_bf16(b * c) + g4_ref_bf16(a * s));
            }
        }
    }
}

/* Router: softmax in float32 over bf16 logits, top-8, renormalise, then the
 * per-expert scale (3.8). Ties keep the lower index, as torch.topk does. */
static inline void g4_ref_router(uint32_t *selected, float *weights, const float *logits,
        const float *per_expert, uint32_t experts, uint32_t k) {
    double maximum = -INFINITY;
    for (uint32_t e = 0; e < experts; e++) {
        if ((double)logits[e] > maximum) { maximum = logits[e]; }
    }
    double total = 0.0;
    float p[G4_REF_EXPERTS];
    for (uint32_t e = 0; e < experts; e++) {
        p[e] = (float)exp((double)logits[e] - maximum);
        total += p[e];
    }
    for (uint32_t e = 0; e < experts; e++) { p[e] = (float)((double)p[e] / total); }
    for (uint32_t s = 0; s < k; s++) {
        uint32_t best = UINT32_MAX;
        for (uint32_t e = 0; e < experts; e++) {
            int taken = 0;
            for (uint32_t j = 0; j < s; j++) { if (selected[j] == e) { taken = 1; } }
            if (!taken && (best == UINT32_MAX || p[e] > p[best])) { best = e; }
        }
        selected[s] = best;
        weights[s] = p[best];
    }
    double sum = 0.0;
    for (uint32_t s = 0; s < k; s++) { sum += weights[s]; }
    for (uint32_t s = 0; s < k; s++) {
        weights[s] = (float)((double)weights[s] / sum) * per_expert[selected[s]];
    }
}

/* tanh(logits/30)*30 with one rounding per operation (3.9). */
static inline float g4_ref_softcap(float logit) {
    const float divided = g4_ref_bf16(logit / (float)G4_REF_SOFTCAP);
    const float squashed = g4_ref_bf16(tanhf(divided));
    return g4_ref_bf16(squashed * (float)G4_REF_SOFTCAP);
}

/* The block output scale: x times a bf16 buffer, one rounding (3.6). */
static inline float g4_ref_layer_scale(float x, float scalar) {
    return g4_ref_bf16(x * g4_ref_bf16(scalar));
}

/* Expert and dense FFN activation product: act(gate) rounded, then times up,
 * rounded (3.7). */
static inline float g4_ref_gate_up(float gate, float up) {
    return g4_ref_bf16(g4_ref_gelu(g4_ref_bf16(gate)) * g4_ref_bf16(up));
}

static inline float g4_ref_embed(float row) {
    return g4_ref_bf16(g4_ref_bf16(row) * G4_REF_EMBED_SCALE_BF16);
}

/* Causal attention, scale 1.0, grouped KV, optional sliding window that
 * includes the query (3.3, 3.4). `ring` is the cache's row count when rows
 * wrap (slot = position % ring); 0 means the cache is indexed by absolute
 * position and `capacity` is first + rows. `keys` is how many keys the cache
 * holds (0 = first + rows); the drafter's query sits one past the committed
 * rows, so it passes `keys = query position` (candidate_generator.py:1316-1327)
 * and the window's begin stays relative to the query's own position.
 * Accumulation in double: this is the golden value, not the device's
 * arithmetic. */
static inline void g4_ref_attn(float *out, const float *query, const float *key, const float *value,
        uint32_t rows, uint32_t first, uint32_t heads, uint32_t kv_heads, uint32_t dim,
        uint32_t window, uint32_t capacity, uint32_t ring, uint32_t keys) {
    const uint32_t group = heads / kv_heads;
    const uint64_t kv_stride = (uint64_t)kv_heads * dim;
    for (uint32_t r = 0; r < rows; r++) {
        const uint32_t position = first + r;
        const uint32_t begin = window && position + 1 > window ? position + 1 - window : 0u;
        const uint32_t last = keys && keys <= position ? keys - 1u : position;
        for (uint32_t h = 0; h < heads; h++) {
            const uint32_t kh = h / group;
            const float *q = query + ((uint64_t)r * heads + h) * dim;
            double maximum = -INFINITY;
            for (uint32_t k = begin; k <= last; k++) {
                const uint64_t slot = ring ? (uint64_t)(k % ring) : k;
                const float *kv = key + slot * kv_stride + (uint64_t)kh * dim;
                double dot = 0.0;
                for (uint32_t d = 0; d < dim; d++) { dot += (double)q[d] * (double)kv[d]; }
                if (dot > maximum) { maximum = dot; }
            }
            double total = 0.0;
            double acc[512];
            for (uint32_t d = 0; d < dim; d++) { acc[d] = 0.0; }
            for (uint32_t k = begin; k <= last; k++) {
                const uint64_t slot = ring ? (uint64_t)(k % ring) : k;
                const float *kv = key + slot * kv_stride + (uint64_t)kh * dim;
                double dot = 0.0;
                for (uint32_t d = 0; d < dim; d++) { dot += (double)q[d] * (double)kv[d]; }
                const double p = exp(dot - maximum);
                total += p;
                const float *vv = value + slot * kv_stride + (uint64_t)kh * dim;
                for (uint32_t d = 0; d < dim; d++) { acc[d] += p * (double)vv[d]; }
            }
            float *o = out + ((uint64_t)r * heads + h) * dim;
            for (uint32_t d = 0; d < dim; d++) { o[d] = (float)(acc[d] / total); }
        }
    }
    (void)capacity;
    (void)ring;
}

#endif
