/* GLM mixed MoE: independent F32 dots, high expert IDs and padded SSD slots. */
#include "ds4_gpu.h"
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "../cuda/mmq/ggml-common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { WIDTH = 256, HIDDEN = 256, OUTPUT = 32, EXPERTS = 288,
       CACHE_SLOTS = 3072, USED = 8, ROWS = 17, IQ2_XXS = 16, IQ2_XS = 17, Q2_K = 10 };
static const int32_t expert_ids[USED] = {287, 256, 270, 3, 128, 255, 1, 281};
static const int32_t cache_ids[USED] = {3071, 1024, 2048, 7, 1, 1023, 2047, 0};
static uint32_t rng = 71u;
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GLM mixed FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static uint8_t rand_byte(void) {
    rng = rng * 1664525u + 1013904223u;
    return (uint8_t)(rng >> 24u);
}

static float half_float(uint16_t h) {
    const float sign = h & 0x8000u ? -1.0f : 1.0f;
    const unsigned exp = (h >> 10u) & 31u;
    return sign * ldexpf((float)(1024u + (h & 1023u)), (int)exp - 25);
}

static size_t block_bytes(uint32_t type) {
    return type == IQ2_XXS ? sizeof(block_iq2_xxs) :
        type == IQ2_XS ? sizeof(block_iq2_xs) : sizeof(block_q2_K);
}

static void dequant(float *out, const void *p, uint32_t type) {
    if (type == Q2_K) {
        const block_q2_K *b = p;
        for (unsigned i = 0u; i < WIDTH; i++) {
            const unsigned group = i / 16u;
            const unsigned qbase = 32u * (group / 8u) + 16u * (group & 1u);
            const unsigned shift = ((group / 2u) & 3u) * 2u;
            const unsigned q = (b->qs[qbase + (i & 15u)] >> shift) & 3u;
            out[i] = half_float(b->d) * (b->scales[group] & 15u) * q -
                half_float(b->dmin) * (b->scales[group] >> 4u);
        }
        return;
    }

    const block_iq2_xxs *xxs = p;
    const block_iq2_xs *xs = p;
    for (unsigned g = 0u; g < 8u; g++) {
        uint32_t packed[2];
        memcpy(packed, xxs->qs + 4u * g, sizeof(packed));
        const uint8_t *codes = (const uint8_t *)packed;
        for (unsigned j = 0u; j < 4u; j++) {
            const unsigned code = type == IQ2_XS ? xs->qs[4u * g + j] : codes[j];
            const uint8_t *grid = (const uint8_t *)(type == IQ2_XS
                ? iq2xs_grid + (code & 511u) : iq2xxs_grid + code);
            const unsigned sign_id = type == IQ2_XS ? code >> 9u :
                (packed[1] >> (7u * j)) & 127u;
            const unsigned scale = type == IQ2_XS
                ? (xs->scales[g] >> (4u * (j / 2u))) & 15u : packed[1] >> 28u;
            const float d = half_float(xxs->d) * (0.5f + scale) * 0.25f;
            for (unsigned k = 0u; k < 8u; k++) {
                const float sign = ksigns_iq2xs[sign_id] & kmask_iq2xs[k] ? -1.0f : 1.0f;
                out[g * 32u + j * 8u + k] = d * grid[k] * sign;
            }
        }
    }
}

static float dot(const uint8_t *w, const float *x, uint32_t type) {
    float values[WIDTH], sum = 0.0f;
    dequant(values, w, type);
    for (unsigned i = 0u; i < WIDTH; i++) { sum += values[i] * x[i]; }
    return sum;
}

static void fill_weights(uint8_t *w, size_t bytes, uint32_t type, size_t unit) {
    const size_t block = block_bytes(type);
    for (size_t p = 0u; p < bytes; p++) { w[p] = rand_byte(); }
    for (size_t p = 0u; p < bytes; p += block) {
        const uint16_t scale = (uint16_t)(0x1400u + (p / unit % 8u) * 128u);
        if (type == Q2_K) {
            block_q2_K *b = (block_q2_K *)(w + p);
            b->d = scale;
            b->dmin = scale;
        } else {
            ((block_iq2_xxs *)(w + p))->d = scale;
        }
    }
}

static ds4_gpu_tensor *tensor(size_t bytes, const void *data) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    CHECK(t);
    if (data) { CHECK(ds4_gpu_tensor_write(t, 0u, data, bytes)); }
    return t;
}

static void compare(const char *label, const float *got, const float *want,
                    unsigned rows, double bound) {
    double error = 0.0, signal = 0.0;
    for (unsigned i = 0u; i < rows * OUTPUT; i++) {
        CHECK(isfinite(got[i]) && isfinite(want[i]));
        const double delta = (double)got[i] - want[i];
        error += delta * delta;
        signal += (double)want[i] * want[i];
    }
    const double rms = sqrt(error / fmax(signal, 1.0e-30));
    printf("GLM mixed %s rows=%u rel_rms=%.8g\n", label, rows, rms);
    CHECK(rms < bound);
}

static void run(uint32_t gt, uint32_t dt) {
    const size_t gu = HIDDEN * block_bytes(gt), du = OUTPUT * block_bytes(dt);
    const size_t off[3] = {64u, 64u + EXPERTS * gu, 64u + 2u * EXPERTS * gu};
    const size_t size = off[2] + EXPERTS * du;
    uint8_t *map = calloc(1u, size);
    CHECK(map);
    fill_weights(map + off[0], EXPERTS * gu, gt, gu);
    fill_weights(map + off[1], EXPERTS * gu, gt, gu);
    fill_weights(map + off[2], EXPERTS * du, dt, du);

    /* Padding is a whole quant block; reads through the wrong stride see poison. */
    const size_t gs = gu + 3u * block_bytes(gt), ds = du + 5u * block_bytes(dt);
    uint8_t *gw = malloc(CACHE_SLOTS * gs), *uw = malloc(CACHE_SLOTS * gs), *dw = malloc(CACHE_SLOTS * ds);
    CHECK(gw && uw && dw);
    memset(gw, 0xff, CACHE_SLOTS * gs);
    memset(uw, 0xff, CACHE_SLOTS * gs);
    memset(dw, 0xff, CACHE_SLOTS * ds);
    for (unsigned s = 0u; s < USED; s++) {
        memcpy(gw + cache_ids[s] * gs, map + off[0] + expert_ids[s] * gu, gu);
        memcpy(uw + cache_ids[s] * gs, map + off[1] + expert_ids[s] * gu, gu);
        memcpy(dw + cache_ids[s] * ds, map + off[2] + expert_ids[s] * du, du);
    }

    float x[ROWS * WIDTH], weights[ROWS * USED], want[ROWS * OUTPUT] = {0};
    int32_t ids[ROWS * USED], slots[ROWS * USED];
    const float input_values[] = {-127.0f, -63.0f, 0.0f, 63.0f, 127.0f};
    for (unsigned r = 0u; r < ROWS; r++) {
        for (unsigned i = 0u; i < WIDTH; i++) { x[r * WIDTH + i] = input_values[(i + 3u * r) % 5u]; }
        for (unsigned s = 0u; s < USED; s++) {
            ids[r * USED + s] = expert_ids[s];
            slots[r * USED + s] = cache_ids[s];
            weights[r * USED + s] = (float)(s + 1u) / 36.0f;
            float hidden[HIDDEN];
            for (unsigned j = 0u; j < HIDDEN; j++) {
                float g = dot(gw + cache_ids[s] * gs + j * block_bytes(gt), x + r * WIDTH, gt);
                float u = dot(uw + cache_ids[s] * gs + j * block_bytes(gt), x + r * WIDTH, gt);
                g = fminf(g, 3.0f);
                u = fmaxf(-3.0f, fminf(u, 3.0f));
                hidden[j] = g / (1.0f + expf(-g)) * u * weights[r * USED + s];
            }
            for (unsigned j = 0u; j < OUTPUT; j++) {
                want[r * OUTPUT + j] += dot(dw + cache_ids[s] * ds + j * block_bytes(dt), hidden, dt);
            }
        }
    }

    CHECK(ds4_gpu_init());
    for (unsigned i = 0u; i < 3u; i++) {
        CHECK(ds4_gpu_cache_model_range(map, size, off[i], EXPERTS * (i == 2u ? du : gu), "glm-mixed-test"));
    }
    ds4_gpu_tensor *tx = tensor(sizeof(x), x), *tw = tensor(sizeof(weights), weights);
    ds4_gpu_tensor *ti = tensor(sizeof(ids), ids), *ts = tensor(sizeof(slots), slots);
    ds4_gpu_tensor *tg = tensor(CACHE_SLOTS * gs, gw), *tu = tensor(CACHE_SLOTS * gs, uw), *td = tensor(CACHE_SLOTS * ds, dw);
    ds4_gpu_tensor *out = tensor((ROWS + 1u) * OUTPUT * sizeof(float), NULL);
    ds4_gpu_tensor *gate = tensor(ROWS * USED * HIDDEN * sizeof(float), NULL);
    ds4_gpu_tensor *up = tensor(ROWS * USED * HIDDEN * sizeof(float), NULL);
    ds4_gpu_tensor *mid = tensor(ROWS * USED * HIDDEN * sizeof(float), NULL);
    ds4_gpu_tensor *down = tensor(ROWS * USED * OUTPUT * sizeof(float), NULL);
    const unsigned counts[] = {1u, 2u, ROWS};
    for (unsigned c = 0u; c < sizeof(counts) / sizeof(counts[0]); c++) {
        const unsigned n = counts[c];
        float raw[(ROWS + 1u) * OUTPUT], owned[(ROWS + 1u) * OUTPUT];
        bool half = true;
        CHECK(ds4_gpu_tensor_fill_f32(out, 12345.0f, (n + 1u) * OUTPUT));
        CHECK(ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down,
            map, size, off[0], off[1], off[2], gt, dt, gu, block_bytes(gt), du,
            block_bytes(dt), WIDTH, HIDDEN, OUTPUT, ti, tw, EXPERTS, USED, 3.0f,
            tx, 3u, n, &half));
        CHECK(!half && ds4_gpu_tensor_read(out, 0u, raw, (n + 1u) * OUTPUT * sizeof(float)));
        CHECK(ds4_gpu_tensor_fill_f32(out, 12345.0f, (n + 1u) * OUTPUT));
        CHECK(ds4_gpu_glm53_moe_owned(out, gate, up, mid, down, tg, tu, td,
            gt, dt, gs, ds, WIDTH, HIDDEN, OUTPUT, ts, tw, CACHE_SLOTS, USED, n, 3.0f, tx));
        CHECK(ds4_gpu_tensor_read(out, 0u, owned, (n + 1u) * OUTPUT * sizeof(float)));
        for (unsigned i = 0u; i < OUTPUT; i++) {
            CHECK(raw[n * OUTPUT + i] == 12345.0f && owned[n * OUTPUT + i] == 12345.0f);
        }
        compare("CPU-F32", raw, want, n, 0.035);
        compare("padded-owned", owned, raw, n, 0.0001);
    }
    ds4_gpu_tensor *all[] = {tx, tw, ti, ts, tg, tu, td, out, gate, up, mid, down};
    for (unsigned i = 0u; i < sizeof(all) / sizeof(all[0]); i++) { ds4_gpu_tensor_free(all[i]); }
    ds4_gpu_cleanup();
    free(gw); free(uw); free(dw); free(map);
}

int main(void) {
    run(IQ2_XXS, IQ2_XS);
    run(IQ2_XS, Q2_K);
    puts("GLM mixed: CUDA and padded owned recipes passed");
    return 0;
}
