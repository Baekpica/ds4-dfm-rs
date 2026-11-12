/* test_ds41_indexer.cu — the V4.1 indexer score on tensor cores against the
 * scalar kernel (model-free).
 *
 * A prefill-width launch (the default dispatch, cuda/ds41_indexer.cuh
 * v41_idx_mma_use) is compared with one scalar launch per row (width 1).
 * Inputs follow the model's formats: q is an E2M1 mantissa times a
 * power-of-two scale per 32-dim block (act_quant mode 1), keys are packed
 * MXFP4 rows, weights are bf16 values.
 *
 *   near   block scales within 2^4 on each side: every sum is exact in f32,
 *          so the two kernels must agree bit for bit;
 *   wide   block scales 2^24 apart: the four block sums add in a different
 *          order, so a dot may land on the other side of a bf16 boundary;
 *          few scores may differ, each by a few bf16 steps of its row's
 *          largest score, never more.
 * Both run with and without a compact candidate list.
 *
 * Build and run: make test-ds41-indexer */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"
#include "ds41_kvfmt.h"

enum { HEADS = 32, DIM = 128, BLOCK = 32, ROWS = 24, GROUPS = 1000, RATIO = 2, POS0 = 1990,
       CAND_BS = 8, CAND_CAP = 40, E8M0_BIAS = 127 };
static const float E2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
/* Wide case: a score differs by at most this many bf16 steps of the row's
 * largest score, and at most this share of the scores differs at all. */
static const float WIDE_STEPS = 4.0f;
static const float WIDE_SHARE = 0.01f;

static uint32_t g_seed = 0x2545F491u;
static uint32_t rnd(void) { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }

static float bf16(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    memcpy(&v, &u, 4);
    return v;
}

/* Block scale exponents: `spread` octaves below `top`. */
static int scale_exp(int top, int spread) { return top - (int)(rnd() % (uint32_t)(spread + 1)); }

static void fill_q(float *q, int spread) {
    for (uint32_t rh = 0; rh < ROWS * HEADS; rh++) {
        for (uint32_t b = 0; b < DIM / BLOCK; b++) {
            const float s = ldexpf(1.0f, scale_exp(-2, spread));
            for (uint32_t d = 0; d < BLOCK; d++) {
                const uint32_t n = rnd() & 15u;
                q[(size_t)rh * DIM + b * BLOCK + d] = ((n & 8u) ? -1.0f : 1.0f) * E2M1[n & 7u] * s;
            }
        }
    }
}

static void fill_keys(uint8_t *k, int spread) {
    for (uint32_t g = 0; g < GROUPS; g++) {
        uint8_t *row = k + (size_t)g * DS4_V41_IDXK_BYTES;
        for (uint32_t i = 0; i < DS4_V41_IDXK_NIB; i++) { row[i] = (uint8_t)rnd(); }
        for (uint32_t b = 0; b < DIM / BLOCK; b++) { row[DS4_V41_IDXK_NIB + b] = (uint8_t)(E8M0_BIAS + scale_exp(1, spread)); }
    }
}

static void fill_weights(float *w) {
    for (uint32_t i = 0; i < ROWS * HEADS; i++) { w[i] = bf16(((float)(rnd() % 2001u) - 1000.0f) / 4096.0f); }
}

/* Per row: a count, then ascending block numbers (the C2 compact list). */
static void fill_cand(int32_t *cand) {
    const uint32_t blocks = GROUPS / CAND_BS;
    for (uint32_t r = 0; r < ROWS; r++) {
        int32_t *list = cand + (size_t)r * (1 + CAND_CAP);
        uint32_t n = 0;
        // Some rows fill fewer blocks than the cap: their tail items are off-list.
        const uint32_t cap = CAND_CAP - r % 5u;
        for (uint32_t b = r % 3u; b < blocks && n < cap; b += 1u + rnd() % 4u) { list[1 + n++] = (int32_t)b; }
        list[0] = (int32_t)n;
    }
}

static int run_case(const char *name, int spread, int use_cand) {
    const uint32_t ns = use_cand ? CAND_CAP * CAND_BS : GROUPS;
    const size_t qb = (size_t)ROWS * HEADS * DIM * 4, kb = (size_t)GROUPS * DS4_V41_IDXK_BYTES;
    const size_t wb = (size_t)ROWS * HEADS * 4, cb = (size_t)ROWS * (1 + CAND_CAP) * 4, sb = (size_t)ROWS * ns * 4;
    float *q = (float *)malloc(qb), *w = (float *)malloc(wb);
    float *wide = (float *)malloc(sb), *one = (float *)malloc(sb);
    uint8_t *k = (uint8_t *)calloc(1, kb);
    int32_t *cand = (int32_t *)malloc(cb);
    fill_q(q, spread); fill_keys(k, spread); fill_weights(w); fill_cand(cand);

    ds4_gpu_tensor *q_t = ds4_gpu_tensor_alloc(qb), *k_t = ds4_gpu_tensor_alloc(kb), *w_t = ds4_gpu_tensor_alloc(wb);
    ds4_gpu_tensor *c_t = ds4_gpu_tensor_alloc(cb), *s_t = ds4_gpu_tensor_alloc(sb);
    int ok = q_t && k_t && w_t && c_t && s_t &&
        ds4_gpu_tensor_write(q_t, 0, q, qb) && ds4_gpu_tensor_write(k_t, 0, k, kb) &&
        ds4_gpu_tensor_write(w_t, 0, w, wb) && ds4_gpu_tensor_write(c_t, 0, cand, cb);

    /* The prefill-width launch: the tensor-core kernel by default. */
    ok = ok && ds4_gpu_v41_indexer_score_tensor(s_t, q_t, k_t, w_t, use_cand ? c_t : NULL, CAND_BS, CAND_CAP,
                                                 ROWS, POS0, GROUPS, HEADS, DIM, RATIO, NULL) &&
         ds4_gpu_synchronize() && ds4_gpu_tensor_read(s_t, 0, wide, sb);

    /* The scalar kernel, one row per launch. */
    for (uint32_t r = 0; ok && r < ROWS; r++) {
        ds4_gpu_tensor *qv = ds4_gpu_tensor_view(q_t, (uint64_t)r * HEADS * DIM * 4, (uint64_t)HEADS * DIM * 4);
        ds4_gpu_tensor *wv = ds4_gpu_tensor_view(w_t, (uint64_t)r * HEADS * 4, (uint64_t)HEADS * 4);
        ds4_gpu_tensor *cv = ds4_gpu_tensor_view(c_t, (uint64_t)r * (1 + CAND_CAP) * 4, (uint64_t)(1 + CAND_CAP) * 4);
        ok = qv && wv && cv &&
            ds4_gpu_v41_indexer_score_tensor(s_t, qv, k_t, wv, use_cand ? cv : NULL, CAND_BS, CAND_CAP,
                                             1, POS0 + r, GROUPS, HEADS, DIM, RATIO, NULL) &&
            ds4_gpu_synchronize() && ds4_gpu_tensor_read(s_t, 0, one + (size_t)r * ns, (uint64_t)ns * 4);
        ds4_gpu_tensor_free(qv); ds4_gpu_tensor_free(wv); ds4_gpu_tensor_free(cv);
    }
    if (!ok) { printf("%-12s FAIL (launch)\n", name); return 1; }

    uint32_t finite = 0, masked = 0, differ = 0, bad = 0;
    float worst = 0.0f;
    for (uint32_t r = 0; r < ROWS; r++) {
        const float *a = wide + (size_t)r * ns, *b = one + (size_t)r * ns;
        float peak = 0.0f;
        for (uint32_t i = 0; i < ns; i++) { if (isfinite(b[i]) && fabsf(b[i]) > peak) { peak = fabsf(b[i]); } }

        for (uint32_t i = 0; i < ns; i++) {
            const int same = !memcmp(&a[i], &b[i], 4);
            if (!isfinite(b[i])) { masked++; bad += !same; continue; }
            finite++;
            if (same) { continue; }
            differ++;
            const float steps = fabsf(a[i] - b[i]) / (peak * ldexpf(1.0f, -8));
            if (steps > worst) { worst = steps; }
        }
    }
    const int exact = spread <= 4;
    const int close = worst <= WIDE_STEPS && (float)differ <= WIDE_SHARE * (float)finite;
    const int pass = !bad && finite && masked && (exact ? differ == 0 : close);
    printf("%-12s %s: %u scores, %u masked, %u differ, worst %.2f bf16 steps of the row peak\n", name, pass ? "PASS" : "FAIL",
           finite, masked, differ, (double)worst);

    ds4_gpu_tensor_free(q_t); ds4_gpu_tensor_free(k_t); ds4_gpu_tensor_free(w_t);
    ds4_gpu_tensor_free(c_t); ds4_gpu_tensor_free(s_t);
    free(q); free(w); free(wide); free(one); free(k); free(cand);
    return !pass;
}

int main(void) {
    if (!ds4_gpu_init()) { fprintf(stderr, "test_ds41_indexer: ds4_gpu_init failed\n"); return 2; }
    int fail = 0;
    fail |= run_case("near", 4, 0);
    fail |= run_case("near+list", 4, 1);
    fail |= run_case("wide", 24, 0);
    fail |= run_case("wide+list", 24, 1);
    printf("DS41 indexer score: %s\n", fail ? "FAIL" : "PASS");
    ds4_gpu_cleanup();
    return fail;
}
