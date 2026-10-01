#ifndef DS4_IQUEST_REF_H
#define DS4_IQUEST_REF_H

/* Model-free numerical oracle for the IQuest-Q1 native integration.
 * FP32 mechanics are explicit; BF16 projection rounding is a separate gate. */
#include <math.h>
#include <stdint.h>
#include <string.h>

enum { IQ_EMBED = 3072, IQ_HEADS = 48, IQ_KV_HEADS = 8, IQ_HEAD = 128,
       IQ_ROT = 32, IQ_EXPERTS = 256, IQ_USED = 8, IQ_LAYERS = 88,
       IQ_WINDOW = 4096, IQ_CONTEXT = 524288, IQ_DRAFT_WINDOW = 512,
       IQ_DRAFT_SLOTS = 7, IQ_Q8_BLOCK = 32, IQ_Q8_ROW_BLOCKS = 64,
       IQ_VOCAB = 160000, IQ_DENSE = 12288, IQ_FF = 1536,
       IQ_PREFILL = 128, IQ_PREFILL_MAX = 8192 };

typedef enum { IQ_KEEP_F32, IQ_ROUND_BF16 } iquest_round_mode;
typedef enum { IQ_DENSE_LAYER, IQ_MOE_LAYER, IQ_DRAFT_LAYER } iquest_layer_kind;
typedef enum { IQ_MTP_OFF, IQ_MTP_ON } iquest_mtp_mode;
typedef enum { IQ_NO_LOGITS, IQ_HAS_LOGITS } iquest_logits_mode;

typedef struct { uint16_t d; int8_t qs[IQ_Q8_BLOCK]; } iquest_q8;
#ifdef __cplusplus
static_assert(sizeof(iquest_q8) == 34, "Q8_0 block layout");
#else
_Static_assert(sizeof(iquest_q8) == 34, "Q8_0 block layout");
#endif

static inline float iquest_half(uint16_t value) {
    unsigned exponent = (value >> 10) & 31u, fraction = value & 1023u;
    uint32_t bits = (uint32_t)(value & 32768u) << 16;
    if (exponent == 31) { bits |= 0x7f800000u | (fraction << 13); }
    else if (exponent) { bits |= ((exponent + 112u) << 23) | (fraction << 13); }
    else if (fraction) {
        unsigned shift = 0;
        while (!(fraction & 1024u)) { fraction <<= 1; shift++; }
        bits |= ((113u - shift) << 23) | ((fraction & 1023u) << 13);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static inline uint16_t iquest_half_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = (uint16_t)((bits >> 16) & 32768u);
    const unsigned exponent = (bits >> 23) & 255u, fraction = bits & 0x7fffffu;
    if (exponent == 255u) { return sign | 0x7c00u | (fraction ? 0x200u : 0); }
    if (exponent >= 143u) { return sign | 0x7c00u; }
    if (exponent < 102u) { return sign; }
    if (exponent <= 112u) {
        const unsigned shift = 126u - exponent;
        const unsigned mantissa = fraction | 0x800000u;
        const unsigned low = mantissa & ((1u << shift) - 1u);
        unsigned high = mantissa >> shift;
        const unsigned half = 1u << (shift - 1u);
        high += low > half || (low == half && (high & 1u));
        return sign | (uint16_t)high;
    }
    const unsigned rounded = fraction + 0xfffu + ((fraction >> 13) & 1u);
    return sign | (uint16_t)(((exponent - 112u) << 10) + (rounded >> 13));
}

static inline float iquest_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) { bits |= 0x00400000u; }
    else { bits += 0x7fffu + ((bits >> 16) & 1u); }
    bits &= 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static inline int iquest_full(unsigned layer) {
    return layer < IQ_LAYERS && (layer == 0 || layer >= 85 || (layer - 1) % 4 == 0);
}

static inline void iquest_rms(float *out, const float *x,
                              const float *weight, unsigned n) {
    double square = 0;
    for (unsigned d = 0; d < n; d++) { square += (double)x[d] * x[d]; }
    const float r = 1.0f / sqrtf((float)(square / n) + 1e-6f);
    for (unsigned d = 0; d < n; d++) { out[d] = x[d] * r * weight[d]; }
}

static inline void iquest_rope(float *x, unsigned heads,
                               unsigned position, float theta) {
    for (unsigned h = 0; h < heads; h++) {
        float *row = x + h * IQ_HEAD;
        for (unsigned d = 0; d < IQ_ROT / 2; d++) {
            const float inv_freq = 1.0f / powf(theta, 2.0f * d / IQ_ROT);
            const float angle = position * inv_freq;
            const float c = cosf(angle), s = sinf(angle);
            const float first = row[d], second = row[d + IQ_ROT / 2];
            row[d] = first * c - second * s;
            row[d + IQ_ROT / 2] = second * c + first * s;
        }
    }
}

static inline void iquest_router(unsigned *ids, float *weights,
                                 const float *logits) {
    // Top-k of softmax is top-k of logits. Renormalizing cancels other experts.
    for (unsigned k = 0; k < IQ_USED; k++) {
        unsigned best = IQ_EXPERTS;
        for (unsigned expert = 0; expert < IQ_EXPERTS; expert++) {
            int selected = 0;
            for (unsigned j = 0; j < k; j++) { selected |= ids[j] == expert; }
            if (selected) { continue; }
            if (best == IQ_EXPERTS || logits[expert] > logits[best]) { best = expert; }
        }
        ids[k] = best;
    }
    float sum = 0;
    for (unsigned k = 0; k < IQ_USED; k++) {
        weights[k] = expf(logits[ids[k]] - logits[ids[0]]);
        sum += weights[k];
    }
    for (unsigned k = 0; k < IQ_USED; k++) { weights[k] /= sum; }
}

static inline void iquest_pack(iquest_q8 *out, const float *x, unsigned n) {
    for (unsigned b = 0; b < n / IQ_Q8_BLOCK; b++) {
        float maximum = 0;
        for (unsigned d = 0; d < IQ_Q8_BLOCK; d++) {
            maximum = fmaxf(maximum, fabsf(x[b * IQ_Q8_BLOCK + d]));
        }
        const float scale = maximum / 127.0f;
        out[b].d = iquest_half_bits(scale);
        const float inverse = scale == 0 ? 0 : 1.0f / scale;
        for (unsigned d = 0; d < IQ_Q8_BLOCK; d++) {
            out[b].qs[d] = (int8_t)roundf(x[b * IQ_Q8_BLOCK + d] * inverse);
        }
    }
}

static inline float iquest_read(const iquest_q8 *row, unsigned d) {
    return iquest_half(row[d / IQ_Q8_BLOCK].d) * row[d / IQ_Q8_BLOCK].qs[d % IQ_Q8_BLOCK];
}

static inline void iquest_store(iquest_q8 *cache, const float *key,
                                const float *value, unsigned position,
                                unsigned capacity) {
    iquest_q8 *row = cache + (position % capacity) * IQ_Q8_ROW_BLOCKS;
    iquest_pack(row, key, IQ_KV_HEADS * IQ_HEAD);
    iquest_pack(row + IQ_Q8_ROW_BLOCKS / 2, value, IQ_KV_HEADS * IQ_HEAD);
}

static inline void iquest_attn(float *out, const float *query,
                               const iquest_q8 *cache, const float *sink,
                               unsigned position, unsigned capacity,
                               unsigned window) {
    const unsigned begin = window && position + 1 > window ? position + 1 - window : 0;
    const float scale = 1.0f / sqrtf(IQ_HEAD);
    for (unsigned h = 0; h < IQ_HEADS; h++) {
        const unsigned kh = h / (IQ_HEADS / IQ_KV_HEADS);
        const float *q = query + h * IQ_HEAD;
        float *acc = out + h * IQ_HEAD;
        float sink_score = 0;
        for (unsigned d = 0; d < IQ_HEAD; d++) {
            sink_score += q[d] * sink[kh * IQ_HEAD + d];
            acc[d] = 0;
        }
        sink_score *= scale;
        float maximum = -INFINITY, total = 0;
        // Official sink correction reads an ordinary BF16 attention output.
        for (unsigned t = begin; t <= position; t++) {
            const iquest_q8 *row = cache + (t % capacity) * IQ_Q8_ROW_BLOCKS;
            float score = 0;
            for (unsigned d = 0; d < IQ_HEAD; d++) { score += q[d] * iquest_read(row, kh * IQ_HEAD + d); }
            score *= scale;
            const float next = fmaxf(maximum, score);
            const float prior = expf(maximum - next), weight = expf(score - next);
            for (unsigned d = 0; d < IQ_HEAD; d++) {
                acc[d] = prior * acc[d] + weight * iquest_read(row + IQ_Q8_ROW_BLOCKS / 2, kh * IQ_HEAD + d);
            }
            total = prior * total + weight;
            maximum = next;
        }
        const float normal_lse = maximum + logf(total);
        const float factor = 1.0f / (1.0f + expf(sink_score - normal_lse));
        for (unsigned d = 0; d < IQ_HEAD; d++) {
            acc[d] = iquest_bf16(iquest_bf16(acc[d] / total) * factor);
        }
    }
}

static inline void iquest_attn_add(float *out, const float *raw,
                                   const float *normalized, const float *attn,
                                   unsigned n, unsigned first) {
    // First layer and predictor preserve raw input; later layers preserve RMS.
    const float *residual = first ? raw : normalized;
    for (unsigned d = 0; d < n; d++) { out[d] = residual[d] + attn[d]; }
}

static inline void iquest_ffn_add(float *out, const float *residual,
                                  const float *ffn, unsigned n, unsigned first) {
    const float scale = first ? 1.0f : 0.53881590608f;
    for (unsigned d = 0; d < n; d++) { out[d] = residual[d] + scale * ffn[d]; }
}

#endif
