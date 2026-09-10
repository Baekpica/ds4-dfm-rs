/* Synthetic source GQA equations, independent FP64 softmax, live KV state. */
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { QH = 32, KH = 8, DIM = 128, QWIDTH = QH * DIM, KWIDTH = KH * DIM,
       LOCAL = 512, GLOBAL = 1024, TOKENS = 8201, KV_ROW = 2 * KWIDTH,
       BF_SHIFT = 16, BF_HALF = 0x7fff, POISON = 0x7fc1,
       CAPTURE_ROWS = 3, CAPTURE_START = 508, CAPTURE_STEPS = 16,
       VERIFY_ROWS = 9, ACCEPT_ROWS = 3, TIMING_START = 7680, TIMING_REPEATS = 10 };

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); \
        exit(1); \
    } \
} while (0)

struct fixture {
    unsigned extent, cap;
    float *q, *k, *v, *rel, *baseline;
    uint16_t *poison;
    ds4_gpu_tensor *dq, *dk, *dv, *dr, *out, *cache, *position;
};

static uint16_t bf_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return (u + BF_HALF + ((u >> BF_SHIFT) & 1)) >> BF_SHIFT;
}

static float bf(float x) {
    uint32_t u = (uint32_t)bf_bits(x) << BF_SHIFT;
    memcpy(&x, &u, sizeof(x));
    return x;
}

static ds4_gpu_tensor *upload(const void *x, size_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    CHECK(t);
    if (x) { CHECK(ds4_gpu_tensor_write(t, 0, x, bytes)); }
    return t;
}

static ds4_gpu_tensor *view(const ds4_gpu_tensor *base, unsigned start,
                            unsigned rows, unsigned width) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_view(base, (uint64_t)start * width * sizeof(float),
                                          (uint64_t)rows * width * sizeof(float));
    CHECK(t);
    return t;
}

static void exact(const float *got, const float *want, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(got[i]) || got[i] != want[i]) {
            fprintf(stderr, "at %zu: %.9g != %.9g\n", i, got[i], want[i]);
            exit(1);
        }
    }
}

static void init(struct fixture *f, unsigned extent) {
    memset(f, 0, sizeof(*f));
    f->extent = extent;
    f->cap = extent == LOCAL ? LOCAL : TOKENS;
    size_t qn = (size_t)TOKENS * QWIDTH, kn = (size_t)TOKENS * KWIDTH;
    size_t rn = (size_t)TOKENS * QH * extent;
    f->q = malloc(qn * sizeof(float)); f->k = malloc(kn * sizeof(float));
    f->v = malloc(kn * sizeof(float)); f->rel = malloc(rn * sizeof(float));
    f->baseline = malloc(qn * sizeof(float));
    f->poison = malloc((size_t)f->cap * KV_ROW * sizeof(uint16_t));
    CHECK(f->q && f->k && f->v && f->rel && f->baseline && f->poison);
    for (size_t i = 0; i < qn; i++) { f->q[i] = ((int)(i * 19 % 127) - 63) / 16.0f + 0.00003f; }
    for (size_t i = 0; i < kn; i++) {
        f->k[i] = ((int)(i * 31 % 139) - 69) / 32.0f + 0.00003f;
        f->v[i] = ((int)(i * 43 % 151) - 75) / 64.0f + 0.00003f;
    }
    for (unsigned t = 0; t < TOKENS; t++) {
        for (unsigned h = 0; h < QH; h++) {
            for (unsigned d = 0; d < extent; d++) {
                f->rel[((size_t)t * QH + h) * extent + d] =
                    ((int)((t * 3 + h * 11 + d * 17) % 113) - 56) / 32.0f;
            }
        }
    }
    for (size_t i = 0; i < (size_t)f->cap * KV_ROW; i++) { f->poison[i] = POISON; }
    f->dq = upload(f->q, qn * sizeof(float)); f->dk = upload(f->k, kn * sizeof(float));
    f->dv = upload(f->v, kn * sizeof(float)); f->dr = upload(f->rel, rn * sizeof(float));
    f->out = upload(NULL, qn * sizeof(float));
    f->cache = upload(f->poison, (size_t)f->cap * KV_ROW * sizeof(uint16_t));
    f->position = upload(NULL, sizeof(uint32_t));
}

static void destroy(struct fixture *f) {
    ds4_gpu_tensor_free(f->dq); ds4_gpu_tensor_free(f->dk); ds4_gpu_tensor_free(f->dv);
    ds4_gpu_tensor_free(f->dr); ds4_gpu_tensor_free(f->out); ds4_gpu_tensor_free(f->cache);
    ds4_gpu_tensor_free(f->position);
    free(f->q); free(f->k); free(f->v); free(f->rel); free(f->baseline); free(f->poison);
}

static void check_cache(struct fixture *f, unsigned committed) {
    size_t bytes = (size_t)f->cap * KV_ROW * sizeof(uint16_t);
    uint16_t *want = malloc(bytes), *got = malloc(bytes);
    CHECK(want && got);
    memcpy(want, f->poison, bytes);
    for (unsigned t = 0; t < committed; t++) {
        size_t dst = (size_t)(t % f->cap) * KV_ROW, src = (size_t)t * KWIDTH;
        for (unsigned d = 0; d < KWIDTH; d++) {
            want[dst + d] = bf_bits(f->k[src + d]);
            want[dst + KWIDTH + d] = bf_bits(f->v[src + d]);
        }
    }
    CHECK(ds4_gpu_tensor_read(f->cache, 0, got, bytes));
    CHECK(memcmp(got, want, bytes) == 0);
    free(want); free(got);
}

static void reference(struct fixture *f, unsigned t) {
    unsigned first = f->extent == LOCAL && t + 1 > LOCAL ? t + 1 - LOCAL : 0;
    double scores[TOKENS], sums[DIM];
    for (unsigned h = 0; h < QH; h++) {
        double maximum = -INFINITY;
        for (unsigned j = first; j <= t; j++) {
            double dot = 0.0;
            for (unsigned d = 0; d < DIM; d++) {
                dot += (double)bf(f->q[(size_t)t * QWIDTH + h * DIM + d]) *
                       bf(f->k[(size_t)j * KWIDTH + (h / (QH / KH)) * DIM + d]);
            }
            unsigned distance = t - j;
            double bias = distance < f->extent ? bf(f->rel[((size_t)t * QH + h) * f->extent + distance]) : 0;
            scores[j] = dot / DIM + bias;
            maximum = fmax(maximum, scores[j]);
        }
        double denom = 0;
        memset(sums, 0, sizeof(sums));
        for (unsigned j = first; j <= t; j++) {
            double p = exp(scores[j] - maximum);
            denom += p;
            for (unsigned d = 0; d < DIM; d++) {
                sums[d] += p * bf(f->v[(size_t)j * KWIDTH + (h / (QH / KH)) * DIM + d]);
            }
        }
        for (unsigned d = 0; d < DIM; d++) {
            double want = sums[d] / denom;
            float got = f->baseline[(size_t)t * QWIDTH + h * DIM + d];
            /* Half a BF16 ulp plus an FP32 softmax/reduction allowance. */
            double ulp = ldexp(1.0, ilogb(fmax(fabs(want), 0x1p-126)) - 7);
            CHECK(isfinite(got) && fabs(got - want) <= 0.501 * ulp + 2e-6);
        }
    }
}

static void run_chunks(struct fixture *f, unsigned chunk) {
    size_t bytes = (size_t)TOKENS * QWIDTH * sizeof(float);
    float *got = malloc(bytes);
    CHECK(got);
    CHECK(ds4_gpu_tensor_write(f->cache, 0, f->poison, (size_t)f->cap * KV_ROW * sizeof(uint16_t)));
    for (uint32_t at = 0; at < TOKENS; at += chunk) {
        unsigned rows = TOKENS - at < chunk ? TOKENS - at : chunk;
        CHECK(ds4_gpu_tensor_write(f->position, 0, &at, sizeof(at)));
        ds4_gpu_tensor *q = view(f->dq, at, rows, QWIDTH), *r = view(f->dr, at, rows, QH * f->extent);
        ds4_gpu_tensor *k = view(f->dk, at, rows, KWIDTH), *v = view(f->dv, at, rows, KWIDTH);
        ds4_gpu_tensor *out = view(f->out, at, rows, QWIDTH);
        CHECK(ds4_gpu_inkling_attention(out, q, r, k, v, f->cache, f->position,
                                       rows, f->cap, f->extent));
        CHECK(ds4_gpu_inkling_kv_store(f->cache, k, v, f->position, rows, f->cap));
        ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(r); ds4_gpu_tensor_free(k);
        ds4_gpu_tensor_free(v); ds4_gpu_tensor_free(out);
    }
    CHECK(ds4_gpu_tensor_read(f->out, 0, got, bytes));
    if (chunk == TOKENS) {
        memcpy(f->baseline, got, bytes);
        const unsigned probes[] = {0, 1, LOCAL - 1, LOCAL, GLOBAL - 1, GLOBAL, TOKENS - 1};
        for (unsigned i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) { reference(f, probes[i]); }
    } else {
        exact(got, f->baseline, bytes / sizeof(float));
    }
    check_cache(f, TOKENS);
    free(got);
    printf("attention extent=%u chunk=%u outputs/cache passed\n", f->extent, chunk);
}

static void seed_prefix(struct fixture *f, uint32_t count) {
    uint32_t start = 0;
    CHECK(ds4_gpu_tensor_write(f->cache, 0, f->poison, (size_t)f->cap * KV_ROW * sizeof(uint16_t)));
    CHECK(ds4_gpu_tensor_write(f->position, 0, &start, sizeof(start)));
    CHECK(ds4_gpu_inkling_kv_store(f->cache, f->dk, f->dv, f->position, count, f->cap));
}

static void captured(struct fixture *f) {
    size_t qb = CAPTURE_ROWS * QWIDTH * sizeof(float), kb = CAPTURE_ROWS * KWIDTH * sizeof(float);
    size_t rb = (size_t)CAPTURE_ROWS * QH * f->extent * sizeof(float);
    ds4_gpu_tensor *q = upload(NULL, qb), *k = upload(NULL, kb), *v = upload(NULL, kb);
    ds4_gpu_tensor *r = upload(NULL, rb), *out = upload(NULL, qb);
    float *got = malloc(qb);
    CHECK(got);
    seed_prefix(f, CAPTURE_START);
    struct ds4_layer_graph_key key = {0};
    key.n_tok = CAPTURE_ROWS; key.cur_hc = q; key.q = r; key.kv = f->cache; key.heads = out;
    unsigned captures = 0, replays = 0;
    for (uint32_t step = 0; step < CAPTURE_STEPS; step++) {
        uint32_t at = CAPTURE_START + step * CAPTURE_ROWS;
        CHECK(ds4_gpu_tensor_write(f->position, 0, &at, sizeof(at)));
        CHECK(ds4_gpu_tensor_write(q, 0, f->q + (size_t)at * QWIDTH, qb));
        CHECK(ds4_gpu_tensor_write(k, 0, f->k + (size_t)at * KWIDTH, kb));
        CHECK(ds4_gpu_tensor_write(v, 0, f->v + (size_t)at * KWIDTH, kb));
        CHECK(ds4_gpu_tensor_write(r, 0, f->rel + (size_t)at * QH * f->extent, rb));
        int mode = ds4_cuda_layer_graph_begin_or_replay(f->extent == LOCAL ? 0 : 1, &key);
        if (mode != 1) {
            CHECK(ds4_gpu_inkling_attention(out, q, r, k, v, f->cache, f->position,
                                           CAPTURE_ROWS, f->cap, f->extent));
            CHECK(ds4_gpu_inkling_kv_store(f->cache, k, v, f->position, CAPTURE_ROWS, f->cap));
        }
        if (mode == 0) {
            ds4_cuda_layer_graph_end_or_commit(f->extent == LOCAL ? 0 : 1);
            captures++;
        }
        replays += mode == 1;
        CHECK(ds4_gpu_tensor_read(out, 0, got, qb));
        exact(got, f->baseline + (size_t)at * QWIDTH, qb / sizeof(float));
        check_cache(f, at + CAPTURE_ROWS);
    }
    CHECK(captures == 1 && replays == CAPTURE_STEPS - 2);
    ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(k); ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(r); ds4_gpu_tensor_free(out); free(got);
    printf("attention extent=%u capture=%u replay=%u live position/ring passed\n", f->extent, captures, replays);
}

static void rejected(struct fixture *f) {
    uint32_t start = TOKENS - VERIFY_ROWS;
    seed_prefix(f, start);
    CHECK(ds4_gpu_tensor_write(f->position, 0, &start, sizeof(start)));
    ds4_gpu_tensor *q = view(f->dq, start, VERIFY_ROWS, QWIDTH);
    ds4_gpu_tensor *r = view(f->dr, start, VERIFY_ROWS, QH * f->extent);
    size_t kb = VERIFY_ROWS * KWIDTH * sizeof(float), qb = VERIFY_ROWS * QWIDTH * sizeof(float);
    ds4_gpu_tensor *k = upload(f->k + (size_t)start * KWIDTH, kb);
    ds4_gpu_tensor *v = upload(f->v + (size_t)start * KWIDTH, kb);
    ds4_gpu_tensor *out = upload(NULL, qb);
    float *got = malloc(qb), *changed = malloc(kb);
    CHECK(got && changed);
    CHECK(ds4_gpu_inkling_kv_store(f->cache, k, v, f->position, 0, f->cap));
    check_cache(f, start);
    for (unsigned pass = 0; pass < 2; pass++) {
        CHECK(ds4_gpu_inkling_attention(out, q, r, k, v, f->cache, f->position,
                                       VERIFY_ROWS, f->cap, f->extent));
        CHECK(ds4_gpu_tensor_read(out, 0, got, qb));
        exact(got, f->baseline + (size_t)start * QWIDTH, ACCEPT_ROWS * QWIDTH);
        check_cache(f, start); /* Forward never commits candidate KV. */
        memcpy(changed, f->k + (size_t)start * KWIDTH, kb);
        for (unsigned i = ACCEPT_ROWS * KWIDTH; i < VERIFY_ROWS * KWIDTH; i++) { changed[i] = 64; }
        CHECK(ds4_gpu_tensor_write(k, 0, changed, kb));
        memcpy(changed, f->v + (size_t)start * KWIDTH, kb);
        for (unsigned i = ACCEPT_ROWS * KWIDTH; i < VERIFY_ROWS * KWIDTH; i++) { changed[i] = -64; }
        CHECK(ds4_gpu_tensor_write(v, 0, changed, kb));
    }
    CHECK(ds4_gpu_inkling_kv_store(f->cache, k, v, f->position, ACCEPT_ROWS, f->cap));
    check_cache(f, start + ACCEPT_ROWS);
    CHECK(!ds4_gpu_inkling_attention(q, q, r, k, v, f->cache, f->position,
                                    VERIFY_ROWS, f->cap, f->extent));
    CHECK(!ds4_gpu_inkling_attention(out, q, r, k, v, f->cache, f->position,
                                    VERIFY_ROWS + 1, f->cap, f->extent));
    CHECK(!ds4_gpu_inkling_attention(out, q, r, k, v, f->cache, f->position,
                                    VERIFY_ROWS, f->cap, 1));
    CHECK(!ds4_gpu_inkling_kv_store(f->cache, k, v, f->position, VERIFY_ROWS + 1, f->cap));
    CHECK(!ds4_gpu_inkling_kv_store(f->cache, k, v, f->position, 1, 0));
    CHECK(!ds4_gpu_inkling_kv_store(f->cache, f->cache, v, f->position, 1, f->cap));
    /* Invalid live positions cannot wrap a write or read outside global KV. */
    const uint32_t invalid[] = {UINT32_MAX, f->cap};
    const unsigned invalid_cases = f->extent == GLOBAL ? 2 : 1;
    for (unsigned i = 0; i < invalid_cases; i++) {
        CHECK(ds4_gpu_tensor_write(f->position, 0, invalid + i, sizeof(uint32_t)));
        CHECK(ds4_gpu_inkling_attention(out, q, r, k, v, f->cache, f->position,
                                       VERIFY_ROWS, f->cap, f->extent));
        CHECK(ds4_gpu_tensor_read(out, 0, got, qb));
        for (unsigned j = 0; j < VERIFY_ROWS * QWIDTH; j++) { CHECK(isnan(got[j])); }
    }
    CHECK(ds4_gpu_tensor_write(f->position, 0, invalid, sizeof(uint32_t)));
    CHECK(ds4_gpu_inkling_kv_store(f->cache, k, v, f->position, VERIFY_ROWS, f->cap));
    check_cache(f, start + ACCEPT_ROWS);
    ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(r); ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(v); ds4_gpu_tensor_free(out); free(got); free(changed);
    printf("attention extent=%u reject suffix / accepted-prefix KV passed\n", f->extent);
}

#include <time.h>
static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Time the grouped prefill kernel against its rollback on production widths
 * at a late position; both paths already matched the baseline above. */
static void timing(struct fixture *f, unsigned rows) {
    uint32_t start = TIMING_START;
    seed_prefix(f, start);
    CHECK(ds4_gpu_tensor_write(f->position, 0, &start, sizeof(start)));
    ds4_gpu_tensor *q = view(f->dq, start, rows, QWIDTH), *r = view(f->dr, start, rows, QH * f->extent);
    ds4_gpu_tensor *k = view(f->dk, start, rows, KWIDTH), *v = view(f->dv, start, rows, KWIDTH);
    ds4_gpu_tensor *out = view(f->out, start, rows, QWIDTH);
    float *got = malloc((size_t)rows * QWIDTH * sizeof(float));
    CHECK(got);
    double elapsed[2];
    for (unsigned mode = 0; mode < 2; mode++) {
        if (mode == 1) { CHECK(setenv("DS4_INKLING_NO_ATTN_GROUP", "1", 1) == 0); }
        else { CHECK(unsetenv("DS4_INKLING_NO_ATTN_GROUP") == 0); }
        CHECK(ds4_gpu_inkling_attention(out, q, r, k, v, f->cache, f->position, rows, f->cap, f->extent));
        CHECK(ds4_gpu_synchronize());
        const double begin = now();
        for (unsigned i = 0; i < TIMING_REPEATS; i++) {
            CHECK(ds4_gpu_inkling_attention(out, q, r, k, v, f->cache, f->position, rows, f->cap, f->extent));
        }
        CHECK(ds4_gpu_synchronize()); elapsed[mode] = (now() - begin) / TIMING_REPEATS;
        CHECK(ds4_gpu_tensor_read(out, 0, got, (size_t)rows * QWIDTH * sizeof(float)));
        exact(got, f->baseline + (size_t)start * QWIDTH, (size_t)rows * QWIDTH);
    }
    CHECK(unsetenv("DS4_INKLING_NO_ATTN_GROUP") == 0);
    printf("attention extent=%u rows=%u at %u exact; selected=%.3f us no-group-control=%.3f us\n",
           f->extent, rows, start, elapsed[0] * 1e6, elapsed[1] * 1e6);
    ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(r); ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(v); ds4_gpu_tensor_free(out); free(got);
}

/* Local rows ending exactly at UINT32_MAX are valid: the grouped kernel must
 * match the per-head kernel there instead of wrapping its key arithmetic. */
static void wrap_boundary(struct fixture *f) {
    const unsigned rows = 16;
    const uint32_t seed = UINT32_MAX - rows + 1 - LOCAL, start = UINT32_MAX - rows + 1;
    CHECK(ds4_gpu_tensor_write(f->cache, 0, f->poison, (size_t)f->cap * KV_ROW * sizeof(uint16_t)));
    CHECK(ds4_gpu_tensor_write(f->position, 0, &seed, sizeof(seed)));
    CHECK(ds4_gpu_inkling_kv_store(f->cache, f->dk, f->dv, f->position, LOCAL, f->cap));
    CHECK(ds4_gpu_tensor_write(f->position, 0, &start, sizeof(start)));
    ds4_gpu_tensor *q = view(f->dq, LOCAL, rows, QWIDTH), *r = view(f->dr, LOCAL, rows, QH * f->extent);
    ds4_gpu_tensor *k = view(f->dk, LOCAL, rows, KWIDTH), *v = view(f->dv, LOCAL, rows, KWIDTH);
    ds4_gpu_tensor *out = view(f->out, 0, rows, QWIDTH);
    const size_t bytes = (size_t)rows * QWIDTH * sizeof(float);
    float *grouped = malloc(bytes), *control = malloc(bytes);
    CHECK(grouped && control);
    CHECK(ds4_gpu_inkling_attention(out, q, r, k, v, f->cache, f->position, rows, f->cap, f->extent));
    CHECK(ds4_gpu_tensor_read(out, 0, grouped, bytes));
    CHECK(setenv("DS4_INKLING_NO_ATTN_GROUP", "1", 1) == 0);
    CHECK(ds4_gpu_inkling_attention(out, q, r, k, v, f->cache, f->position, rows, f->cap, f->extent));
    CHECK(unsetenv("DS4_INKLING_NO_ATTN_GROUP") == 0);
    CHECK(ds4_gpu_tensor_read(out, 0, control, bytes));
    exact(grouped, control, bytes / sizeof(float));
    ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(r); ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(v); ds4_gpu_tensor_free(out); free(grouped); free(control);
    printf("attention extent=%u rows=%u ending at UINT32_MAX exact\n", f->extent, rows);
}

int main(void) {
    CHECK(unsetenv("DS4_INKLING_NO_ATTN_GROUP") == 0);
    CHECK(ds4_gpu_init());
    const unsigned extents[] = {LOCAL, GLOBAL}, chunks[] = {TOKENS, 1, 7, 15, 16, 63, 257, 700, 8192};
    for (unsigned e = 0; e < sizeof(extents) / sizeof(extents[0]); e++) {
        struct fixture f;
        init(&f, extents[e]);
        for (unsigned c = 0; c < sizeof(chunks) / sizeof(chunks[0]); c++) { run_chunks(&f, chunks[c]); }
        /* The rollback kernel must reproduce the grouped baseline exactly. */
        CHECK(setenv("DS4_INKLING_NO_ATTN_GROUP", "1", 1) == 0);
        run_chunks(&f, 8192); run_chunks(&f, 257);
        CHECK(unsetenv("DS4_INKLING_NO_ATTN_GROUP") == 0);
        captured(&f); rejected(&f);
        if (extents[e] == LOCAL) { wrap_boundary(&f); }
        timing(&f, 512); timing(&f, 16);
        destroy(&f);
    }
    ds4_gpu_cleanup();
    puts("Inkling attention/KV checks passed");
    return 0;
}
