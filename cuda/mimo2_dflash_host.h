/* Host DFlash attention. The GPU path is checked against these loops.
 * DS4_MIMO2_DFLASH_CPU=1 runs them instead of the device kernels. */
#ifndef MIMO2_DFLASH_HOST_H
#define MIMO2_DFLASH_HOST_H

#include <math.h>
#include <string.h>

enum { DF_H = 4096, DF_Q = 8192, DF_KV = 1024, DF_HD = 128,
       DF_QH = 64, DF_KH = 8, DF_WIN = 1024 };

#ifdef MIMO2_DFLASH_HOST_FNS
static void df_rms(float *row, const float *weight, unsigned width) {
    double sum = 0.0;
    unsigned i;
    float scale;
    for (i = 0; i < width; i++) { sum += (double)row[i] * row[i]; }
    scale = 1.0f / sqrtf((float)(sum / width) + 1e-6f);
    for (i = 0; i < width; i++) { row[i] *= scale * weight[i]; }
}

/* Qwen3 neoX partial RoPE. The first 64 dims rotate as two halves that
 * share one frequency; the remaining 64 stay put. Adjacent-pair rotation
 * makes every mask row predict the same token. */
static void df_rope(float *row, unsigned heads, unsigned pos) {
    float c[32], s[32];
    unsigned i, h;
    for (i = 0; i < 32; i++) {
        const float freq = (float)pos / powf(10000.0f, (2.0f * (float)i) / 64.0f);
        c[i] = cosf(freq);
        s[i] = sinf(freq);
    }
    for (h = 0; h < heads; h++) {
        float *x = row + (size_t)h * DF_HD;
        float rot[64];
        for (i = 0; i < 32; i++) {
            const float a = x[i], b = x[32 + i];
            rot[i] = a * c[i] - b * s[i];
            rot[32 + i] = b * c[i] + a * s[i];
        }
        memcpy(x, rot, sizeof(rot));
    }
}

static void df_attn(float *out, const float *q, const float *k, const float *v,
                     const float *sinks, const unsigned *k_pos, unsigned q0,
                     unsigned q_len, unsigned kv_len) {
    const float value_scale = 0.612f;
    unsigned qi, head, ki, d;
    for (qi = 0; qi < q_len; qi++) {
        const unsigned q_pos = q0 + qi;
        for (head = 0; head < DF_QH; head++) {
            const float *qq = q + ((size_t)qi * DF_QH + head) * DF_HD;
            const unsigned kv_head = head / (DF_QH / DF_KH);
            float max_score = sinks ? sinks[head] : -INFINITY;
            for (ki = 0; ki < kv_len; ki++) {
                const unsigned kp = k_pos[ki];
                const float *kk;
                float dot = 0.0f;
                if (q_pos > kp && q_pos - kp >= DF_WIN) { continue; }
                if (kp > q_pos && kp - q_pos >= DF_WIN) { continue; }
                kk = k + ((size_t)ki * DF_KH + kv_head) * DF_HD;
                for (d = 0; d < DF_HD; d++) { dot += qq[d] * kk[d]; }
                dot /= sqrtf((float)DF_HD);
                if (dot > max_score) { max_score = dot; }
            }
            float denom = sinks ? expf(sinks[head] - max_score) : 0.0f;
            for (ki = 0; ki < kv_len; ki++) {
                const unsigned kp = k_pos[ki];
                const float *kk;
                float dot = 0.0f;
                if (q_pos > kp && q_pos - kp >= DF_WIN) { continue; }
                if (kp > q_pos && kp - q_pos >= DF_WIN) { continue; }
                kk = k + ((size_t)ki * DF_KH + kv_head) * DF_HD;
                for (d = 0; d < DF_HD; d++) { dot += qq[d] * kk[d]; }
                dot /= sqrtf((float)DF_HD);
                denom += expf(dot - max_score);
            }
            float *dst = out + ((size_t)qi * DF_QH + head) * DF_HD;
            for (d = 0; d < DF_HD; d++) { dst[d] = 0.0f; }
            for (ki = 0; ki < kv_len; ki++) {
                const unsigned kp = k_pos[ki];
                const float *kk;
                const float *vv;
                float dot = 0.0f;
                float w;
                if (q_pos > kp && q_pos - kp >= DF_WIN) { continue; }
                if (kp > q_pos && kp - q_pos >= DF_WIN) { continue; }
                kk = k + ((size_t)ki * DF_KH + kv_head) * DF_HD;
                vv = v + ((size_t)ki * DF_KH + kv_head) * DF_HD;
                for (d = 0; d < DF_HD; d++) { dot += qq[d] * kk[d]; }
                dot /= sqrtf((float)DF_HD);
                w = expf(dot - max_score) / denom;
                for (d = 0; d < DF_HD; d++) { dst[d] += w * vv[d] * value_scale; }
            }
        }
    }
}
#endif

#endif
