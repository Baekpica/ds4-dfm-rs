/*
 * DS4 quantization facade.
 *
 * These are the small GGUF quantization pieces needed by our DeepSeek V4
 * Flash recipes: float conversion, q8_0, q2_K, q4_K, and iq2_xxs.  The code is
 * local C and deliberately narrow; other GGUF type IDs are named for metadata
 * compatibility, but cannot be emitted by this tool.
 *
 * The quantized block layouts and search procedures are derived from the
 * MIT-licensed GGML/llama.cpp quantizers.  Keep changes conservative: byte
 * layout compatibility is more important here than generality.
 */

#include "quants.h"
#include "onebit_quant.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static pthread_mutex_t ds4q_init_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    const char *name;
    int64_t block_size;
    size_t type_size;
    bool can_quantize;
    bool requires_imatrix;
} ds4q_traits;

#define QK_K 256
#define DS4Q_GROUP_MAX_EPS 1e-15f
#define DS4Q_MIN(a, b) ((a) < (b) ? (a) : (b))
#define DS4Q_MAX(a, b) ((a) > (b) ? (a) : (b))

static const ds4q_traits ds4q_type_traits[DS4Q_TYPE_COUNT] = {
    [DS4Q_TYPE_F32]     = { "f32",      1,   4, false, false },
    [DS4Q_TYPE_F16]     = { "f16",      1,   2, false, false },
    [DS4Q_TYPE_Q4_0]    = { "q4_0",    32,  18, false, false },
    [DS4Q_TYPE_Q4_1]    = { "q4_1",    32,  20, false, false },
    [DS4Q_TYPE_Q5_0]    = { "q5_0",    32,  22, false, false },
    [DS4Q_TYPE_Q5_1]    = { "q5_1",    32,  24, false, false },
    [DS4Q_TYPE_Q8_0]    = { "q8_0",    32,  34, true,  false },
    [DS4Q_TYPE_Q8_1]    = { "q8_1",    32,  36, false, false },
    [DS4Q_TYPE_Q2_K]    = { "q2_K",  QK_K,  84, true,  false },
    [DS4Q_TYPE_Q3_K]    = { "q3_K",  QK_K, 110, false, false },
    [DS4Q_TYPE_Q4_K]    = { "q4_K",  QK_K, 144, true,  false },
    [DS4Q_TYPE_Q5_K]    = { "q5_K",  QK_K, 176, false, false },
    [DS4Q_TYPE_Q6_K]    = { "q6_K",  QK_K, 210, false, false },
    [DS4Q_TYPE_Q8_K]    = { "q8_K",  QK_K, 292, false, false },
    [DS4Q_TYPE_IQ2_XXS] = { "iq2_xxs", QK_K,  66, true,  true  },
    [DS4Q_TYPE_IQ2_XS]  = { "iq2_xs",  QK_K,  74, false, true  },
    [DS4Q_TYPE_IQ3_XXS] = { "iq3_xxs", QK_K,  98, false, false },
    [DS4Q_TYPE_IQ1_S]   = { "iq1_s",   QK_K,  50, false, true  },
    [DS4Q_TYPE_IQ4_NL]  = { "iq4_nl",     32,  18, false, false },
    [DS4Q_TYPE_IQ3_S]   = { "iq3_s",   QK_K, 110, false, false },
    [DS4Q_TYPE_IQ2_S]   = { "iq2_s",   QK_K,  82, false, false },
    [DS4Q_TYPE_IQ4_XS]  = { "iq4_xs",  QK_K, 136, false, false },
    [DS4Q_TYPE_I8]      = { "i8",          1,   1, false, false },
    [DS4Q_TYPE_I16]     = { "i16",         1,   2, false, false },
    [DS4Q_TYPE_I32]     = { "i32",         1,   4, false, false },
    [DS4Q_TYPE_I64]     = { "i64",         1,   8, false, false },
    [DS4Q_TYPE_F64]     = { "f64",         1,   8, false, false },
    [DS4Q_TYPE_IQ1_M]   = { "iq1_m",   QK_K,  56, false, false },
    [DS4Q_TYPE_BF16]    = { "bf16",        1,   2, false, false },
    [DS4Q_TYPE_TQ1_0]   = { "tq1_0",   QK_K,  54, false, false },
    [DS4Q_TYPE_TQ2_0]   = { "tq2_0",   QK_K,  66, false, false },
    [DS4Q_TYPE_MXFP4]   = { "mxfp4",      32,  17, false, false },
    [DS4Q_TYPE_Q1_0]    = { "q1_0",      128,  18, false, false },
    /* GO1B (per-ROW) is a variable-length format (one fp16 scale + packed sign words
     * per row); its byte size is special-cased in ds4q_row_size, NOT
     * type_size*(ne/block_size). block_size=32 only enforces ncols%32==0; type_size
     * here is nominal/unused. CLI name "go1b_row" so the emitted BLOCK form (slot 40)
     * owns the canonical "go1b" name. Research/calibration encoder only. */
    [DS4Q_TYPE_GO1B]    = { "go1b_row",   32,   4, true,  false },
    /* GO1B_BLK is the runtime on-disk format (Metal `block_go1b`): a clean fixed
     * 256-element block of 34 bytes = fp16 row scale (replicated into every block)
     * + 32 sign bytes. Uniform type_size*(ne/block_size) sizing (34*(ne/256)), so it
     * needs NO ds4q_row_size special case. PINNED ggml disk type 40 (was the unused
     * NVFP4 placeholder). CLI name "go1b"; this is what `--experts go1b` emits. */
    [DS4Q_TYPE_GO1B_BLK] = { "go1b",     256,  34, true,  false },
};

static float ds4q_f32_from_bits(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } v = { .u = bits };
    return v.f;
}

static uint32_t ds4q_f32_to_bits(float f) {
    union {
        float f;
        uint32_t u;
    } v = { .f = f };
    return v.u;
}

static uint16_t ds4q_f32_to_f16(float f) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = ds4q_f32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) bias = UINT32_C(0x71000000);

    base = ds4q_f32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t out = ds4q_f32_to_bits(base);
    const uint32_t exp_bits = (out >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = out & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (uint16_t)((sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
}

static int ds4q_nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(i));
    return (i & 0x007fffff) - 0x00400000;
}

static float ds4q_make_qkx2_quants(int n, int nmax, const float *x, const float *weights,
                                   uint8_t *L, float *the_min, uint8_t *Laux,
                                   float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        memset(L, 0, (size_t)n);
        *the_min = -min;
        return 0.0f;
    }
    float iscale = nmax / (max - min);
    float scale = 1 / iscale;
    float best_error = 0;
    for (int i = 0; i < n; i++) {
        int l = ds4q_nearest_int(iscale * (x[i] - min));
        L[i] = DS4Q_MAX(0, DS4Q_MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        best_error += weights[i] * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; is++) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; i++) {
            int l = ds4q_nearest_int(iscale * (x[i] - min));
            l = DS4Q_MAX(0, DS4Q_MIN(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; i++) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                cur_error += weights[i] * diff;
            }
            if (cur_error < best_error) {
                memcpy(L, Laux, (size_t)n);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static float ds4q_make_qkx3_quants(int n, int nmax, const float *x, const float *weights,
                                   uint8_t *L, float *the_min, uint8_t *Laux,
                                   float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights ? weights[0] : x[0] * x[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights ? weights[i] : x[i] * x[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max <= min) {
        memset(L, 0, (size_t)n);
        *the_min = -min;
        return 0.0f;
    }
    float iscale = nmax / (max - min);
    float scale = 1 / iscale;
    float best_mad = 0;
    for (int i = 0; i < n; i++) {
        int l = ds4q_nearest_int(iscale * (x[i] - min));
        L[i] = DS4Q_MAX(0, DS4Q_MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights ? weights[i] : x[i] * x[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; is++) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; i++) {
            int l = ds4q_nearest_int(iscale * (x[i] - min));
            l = DS4Q_MAX(0, DS4Q_MIN(nmax, l));
            Laux[i] = l;
            float w = weights ? weights[i] : x[i] * x[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; i++) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights ? weights[i] : x[i] * x[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                memcpy(L, Laux, (size_t)n);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static float ds4q_make_qp_quants(int n, int nmax, const float *x, uint8_t *L, const float *quant_weights) {
    float max = 0;
    for (int i = 0; i < n; i++) max = DS4Q_MAX(max, x[i]);
    if (max < DS4Q_GROUP_MAX_EPS) {
        memset(L, 0, (size_t)n);
        return 0.0f;
    }
    float iscale = nmax / max;
    for (int i = 0; i < n; i++) L[i] = ds4q_nearest_int(iscale * x[i]);
    float scale = 1 / iscale;
    float best_mse = 0;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - scale * L[i];
        best_mse += quant_weights[i] * diff * diff;
    }
    for (int is = -4; is <= 4; is++) {
        if (is == 0) continue;
        float iscale_is = (0.1f * is + nmax) / max;
        float scale_is = 1 / iscale_is;
        float mse = 0;
        for (int i = 0; i < n; i++) {
            int l = ds4q_nearest_int(iscale_is * x[i]);
            l = DS4Q_MIN(nmax, l);
            float diff = x[i] - scale_is * l;
            mse += quant_weights[i] * diff * diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    float sumlx = 0, suml2 = 0;
    for (int i = 0; i < n; i++) {
        int l = ds4q_nearest_int(iscale * x[i]);
        l = DS4Q_MIN(nmax, l);
        L[i] = l;
        float w = quant_weights[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    for (int itry = 0; itry < 5; itry++) {
        int n_changed = 0;
        for (int i = 0; i < n; i++) {
            float w = quant_weights[i];
            float slx = sumlx - w * x[i] * L[i];
            float sl2 = suml2 - w * L[i] * L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = ds4q_nearest_int(x[i] * sl2 / slx);
                new_l = DS4Q_MIN(nmax, new_l);
                if (new_l != L[i]) {
                    slx += w * x[i] * new_l;
                    sl2 += w * new_l * new_l;
                    if (slx * slx * suml2 > sumlx * sumlx * sl2) {
                        L[i] = new_l;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed++;
                    }
                }
            }
        }
        if (!n_changed) break;
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

static void ds4q_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static size_t ds4q_quantize_q8_0(const float *src, void *dst, int64_t start,
                                 int64_t nrows, int64_t ncols) {
    const int64_t qk = 32;
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_Q8_0, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t nblocks = nrows * (ncols / qk);

    for (int64_t b = 0; b < nblocks; b++) {
        const float *x = src + start + (size_t)b * qk;
        float amax = 0.0f;
        for (int j = 0; j < qk; j++) {
            const float av = fabsf(x[j]);
            if (av > amax) amax = av;
        }

        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        const uint16_t hd = ds4q_f32_to_f16(d);
        memcpy(out, &hd, sizeof(hd));

        int8_t *qs = (int8_t *)(out + sizeof(hd));
        for (int j = 0; j < qk; j++) qs[j] = (int8_t)roundf(x[j] * id);
        out += sizeof(hd) + qk;
    }
    return (size_t)nrows * row_size;
}

static void ds4q_write_q4_k_block_ref(const float *x, uint8_t *y) {
    enum { scales_off = 4, qs_off = 16 };
    uint8_t L[QK_K];
    uint8_t Laux[32];
    float weights[32];
    float mins[QK_K / 32];
    float scales[QK_K / 32];
    uint8_t *scales_out = y + scales_off;
    uint8_t *qs_out = y + qs_off;

    float max_scale = 0;
    float max_min = 0;
    for (int j = 0; j < QK_K / 32; j++) {
        float sum_x2 = 0;
        for (int l = 0; l < 32; l++) sum_x2 += x[32 * j + l] * x[32 * j + l];
        float av_x = sqrtf(sum_x2 / 32);
        for (int l = 0; l < 32; l++) weights[l] = av_x + fabsf(x[32 * j + l]);
        scales[j] = ds4q_make_qkx2_quants(32, 15, x + 32 * j, weights, L + 32 * j,
                                           &mins[j], Laux, -1.0f, 0.1f, 20, false);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }

    float inv_scale = max_scale > 0 ? 63.0f / max_scale : 0.0f;
    float inv_min = max_min > 0 ? 63.0f / max_min : 0.0f;
    for (int j = 0; j < QK_K / 32; j++) {
        uint8_t ls = ds4q_nearest_int(inv_scale * scales[j]);
        uint8_t lm = ds4q_nearest_int(inv_min * mins[j]);
        ls = DS4Q_MIN(63, ls);
        lm = DS4Q_MIN(63, lm);
        if (j < 4) {
            scales_out[j] = ls;
            scales_out[j + 4] = lm;
        } else {
            scales_out[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scales_out[j - 4] |= ((ls >> 4) << 6);
            scales_out[j - 0] |= ((lm >> 4) << 6);
        }
    }

    uint16_t d = ds4q_f32_to_f16(max_scale / 63.0f);
    uint16_t dmin = ds4q_f32_to_f16(max_min / 63.0f);
    memcpy(y + 0, &d, sizeof(d));
    memcpy(y + 2, &dmin, sizeof(dmin));

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; j++) {
        ds4q_get_scale_min_k4(j, scales_out, &sc, &m);
        const float dd = ds4q_f16_to_f32(d) * sc;
        if (!dd) continue;
        const float dm = ds4q_f16_to_f32(dmin) * m;
        for (int ii = 0; ii < 32; ii++) {
            int l = ds4q_nearest_int((x[32 * j + ii] + dm) / dd);
            l = DS4Q_MAX(0, DS4Q_MIN(15, l));
            L[32 * j + ii] = l;
        }
    }

    uint8_t *q = qs_out;
    for (int j = 0; j < QK_K; j += 64) {
        for (int l = 0; l < 32; l++) q[l] = L[j + l] | (L[j + l + 32] << 4);
        q += 32;
    }
}

