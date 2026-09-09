/* FP64 RMSNorm oracle with source BF16 inputs/weights/output boundaries. */
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { WIDTH_MAX = 9600, OFFSET = 4096, Q8_WIDTH = 512, Q8_ROWS = 64, Q8_OUTPUT = 128,
       Q8_BLOCK = 32, Q8_BYTES = 34, HALF_ONE = 0x3c00,
       F32_OFFSET = OFFSET + WIDTH_MAX * sizeof(uint16_t),
       Q8_OFFSET = F32_OFFSET + Q8_WIDTH * sizeof(float),
       MAP_SIZE = Q8_OFFSET + Q8_WIDTH / Q8_BLOCK * Q8_OUTPUT * Q8_BYTES,
       BF_SHIFT = 16, BF_HALF = 0x7fff, POINT_COUNT = 257, CAPTURE_STEPS = 9 };
static const double EPS = 1e-6;
#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); exit(1); \
} } while (0)

static uint16_t bits(float x) {
    uint32_t u; memcpy(&u, &x, sizeof(u));
    return (u + BF_HALF + ((u >> BF_SHIFT) & 1)) >> BF_SHIFT;
}
static float bf(float x) {
    uint32_t u = (uint32_t)bits(x) << BF_SHIFT;
    memcpy(&x, &u, sizeof(x)); return x;
}
static float weight(unsigned i) { return bf(((int)(i * 17 % 101) - 50) / 37.0f); }
static ds4_gpu_tensor *upload(const void *data, size_t n) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n); CHECK(t);
    if (data) { CHECK(ds4_gpu_tensor_write(t, 0, data, n)); }
    return t;
}

static void reference(const float *got, const float *x, unsigned rows, unsigned width) {
    for (unsigned t = 0; t < rows; t++) {
        double sum = 0;
        for (unsigned i = 0; i < width; i++) {
            double v = bf(x[(size_t)t * width + i]); sum += v * v;
        }
        double scale = 1.0 / sqrt(sum / width + EPS);
        for (unsigned i = 0; i < width; i++) {
            size_t j = (size_t)t * width + i;
            double want = bf(x[j]) * scale * weight(i);
            double ulp = ldexp(1.0, ilogb(fmax(fabs(want), 0x1p-126)) - 7);
            CHECK(isfinite(got[j]) && got[j] == bf(got[j]));
            CHECK(fabs(got[j] - want) <= 0.501 * ulp + 2e-7 * fabs(want));
            if (want == 0) { CHECK(got[j] == 0); }
        }
    }
}

static void norm_case(const void *map, unsigned rows, unsigned width) {
    size_t n = (size_t)rows * width, bytes = n * sizeof(float);
    float *x = malloc(bytes), *got = malloc(bytes); CHECK(x && got);
    for (unsigned t = 0; t < rows; t++) {
        for (unsigned c = 0; c < width; c++) {
            float v = ((int)((t * 19 + c * 31) % 251) - 125) / 53.0f + 0.00003f;
            if (t % 4 == 1) { v *= 1e-8f; }
            if (t % 4 == 2) { v *= 65536.0f; }
            if (t % 4 == 3) { v = 0; }
            x[(size_t)t * width + c] = v;
        }
    }
    ds4_gpu_tensor *dx = upload(x, bytes), *out = upload(NULL, bytes);
    CHECK(ds4_gpu_inkling_norm(out, dx, map, MAP_SIZE, OFFSET, width, rows));
    CHECK(ds4_gpu_tensor_read(out, 0, got, bytes)); reference(got, x, rows, width);
    CHECK(ds4_gpu_inkling_norm(dx, dx, map, MAP_SIZE, OFFSET, width, rows));
    CHECK(ds4_gpu_tensor_read(dx, 0, got, bytes)); reference(got, x, rows, width);
    CHECK(!ds4_gpu_inkling_norm(out, dx, map, MAP_SIZE, OFFSET, width, rows + 1));
    CHECK(!ds4_gpu_inkling_norm(out, dx, map, OFFSET, OFFSET, width, rows));
    CHECK(!ds4_gpu_inkling_norm(out, dx, map, MAP_SIZE, OFFSET + 1, width, rows));
    CHECK(!ds4_gpu_inkling_norm(out, dx, map, MAP_SIZE, OFFSET, 0, rows));
    ds4_gpu_tensor_free(dx); ds4_gpu_tensor_free(out); free(x); free(got);
    printf("norm rows=%u width=%u BF16 reference passed\n", rows, width);
}

static void pointwise(void) {
    float a[POINT_COUNT], b[POINT_COUNT], got[POINT_COUNT];
    for (unsigned i = 0; i < POINT_COUNT; i++) {
        a[i] = ((int)(i * 43 % 137) - 68) / 11.0f;
        b[i] = ((int)(i * 29 % 101) - 50) / 17.0f;
    }
    ds4_gpu_tensor *da = upload(a, sizeof(a)), *db = upload(b, sizeof(b));
    ds4_gpu_tensor *out = upload(NULL, sizeof(got));
    const float scales[] = {1.0f, 0.0625f, 0.875f, -0.5f, 0.0f};
    for (unsigned add = 0; add < 2; add++) {
        for (unsigned s = 0; s < sizeof(scales) / sizeof(scales[0]); s++) {
            CHECK(ds4_gpu_tensor_write(da, 0, a, sizeof(a)));
            CHECK(ds4_gpu_inkling_add_scale(out, da, add ? db : NULL, scales[s], POINT_COUNT));
            CHECK(ds4_gpu_tensor_read(out, 0, got, sizeof(got)));
            for (unsigned i = 0; i < POINT_COUNT; i++) {
                CHECK(got[i] == bf(bf(a[i]) * scales[s] + (add ? bf(b[i]) : 0.0f)));
            }
            CHECK(ds4_gpu_inkling_add_scale(da, da, add ? db : NULL, scales[s], POINT_COUNT));
            float inplace[POINT_COUNT];
            CHECK(ds4_gpu_tensor_read(da, 0, inplace, sizeof(inplace)));
            CHECK(memcmp(got, inplace, sizeof(got)) == 0);
            if (add) {
                CHECK(ds4_gpu_tensor_write(da, 0, a, sizeof(a)));
                CHECK(ds4_gpu_inkling_add_scale(db, da, db, scales[s], POINT_COUNT));
                CHECK(ds4_gpu_tensor_read(db, 0, inplace, sizeof(inplace)));
                CHECK(memcmp(got, inplace, sizeof(got)) == 0);
                CHECK(ds4_gpu_tensor_write(db, 0, b, sizeof(b)));
            }
        }
    }
    CHECK(!ds4_gpu_inkling_add_scale(out, da, db, 1, POINT_COUNT + 1));
    CHECK(!ds4_gpu_inkling_add_scale(out, da, db, 1, 0));
    CHECK(!ds4_gpu_inkling_add_scale(out, da, db, NAN, POINT_COUNT));
    CHECK(!ds4_gpu_inkling_add_scale(out, da, db, 1, UINT64_MAX));
    ds4_gpu_tensor_free(da); ds4_gpu_tensor_free(db); ds4_gpu_tensor_free(out);
    puts("BF16 scale/residual and in-place boundaries exact");
}

static void aliases(const void *map) {
    const size_t bytes = POINT_COUNT * sizeof(float);
    ds4_gpu_tensor *base = upload(NULL, bytes + sizeof(float));
    ds4_gpu_tensor *low = ds4_gpu_tensor_view(base, 0, bytes);
    ds4_gpu_tensor *high = ds4_gpu_tensor_view(base, sizeof(float), bytes);
    ds4_gpu_tensor *other = upload(NULL, bytes);
    CHECK(low && high);
    CHECK(!ds4_gpu_inkling_norm(high, low, map, MAP_SIZE, OFFSET, POINT_COUNT, 1));
    CHECK(!ds4_gpu_inkling_add_scale(high, low, NULL, 1, POINT_COUNT));
    CHECK(!ds4_gpu_inkling_add_scale(high, other, low, 1, POINT_COUNT));
    ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(high);
    ds4_gpu_tensor_free(base); ds4_gpu_tensor_free(other);
}

static void captured(const void *map) {
    float x[POINT_COUNT], got[POINT_COUNT], normed[POINT_COUNT];
    ds4_gpu_tensor *dx = upload(NULL, sizeof(x)), *mid = upload(NULL, sizeof(x));
    ds4_gpu_tensor *out = upload(NULL, sizeof(x));
    struct ds4_layer_graph_key key = {0};
    key.n_tok = 1; key.cur_hc = dx; key.q = mid; key.heads = out;
    unsigned captures = 0, replays = 0;
    for (unsigned t = 0; t < CAPTURE_STEPS; t++) {
        for (unsigned i = 0; i < POINT_COUNT; i++) { x[i] = ((int)((i * 41 + t * 79) % 131) - 65) / 17.0f; }
        CHECK(ds4_gpu_tensor_write(dx, 0, x, sizeof(x)));
        int mode = ds4_cuda_layer_graph_begin_or_replay(0, &key);
        if (mode != 1) {
            CHECK(ds4_gpu_inkling_norm(mid, dx, map, MAP_SIZE, OFFSET, POINT_COUNT, 1));
            CHECK(ds4_gpu_inkling_add_scale(out, mid, dx, 1.0f, POINT_COUNT));
        }
        if (mode == 0) { ds4_cuda_layer_graph_end_or_commit(0); captures++; }
        replays += mode == 1;
        CHECK(ds4_gpu_tensor_read(mid, 0, normed, sizeof(normed)));
        CHECK(ds4_gpu_tensor_read(out, 0, got, sizeof(got)));
        reference(normed, x, 1, POINT_COUNT);
        for (unsigned i = 0; i < POINT_COUNT; i++) { CHECK(got[i] == bf(normed[i] + bf(x[i]))); }
    }
    CHECK(captures == 1 && replays == CAPTURE_STEPS - 2);
    ds4_gpu_tensor_free(dx); ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(out);
    printf("norm/residual capture=%u replay=%u live inputs passed\n", captures, replays);
}

static void q8_reuse(const void *map) {
    const size_t bytes = Q8_WIDTH * Q8_ROWS * sizeof(float);
    const size_t out_bytes = Q8_OUTPUT * Q8_ROWS * sizeof(float);
    float *x = malloc(bytes), *y = malloc(bytes), *got = malloc(out_bytes), *want = malloc(out_bytes);
    CHECK(x && y && got && want);
    for (unsigned i = 0; i < Q8_WIDTH * Q8_ROWS; i++) { x[i] = ((int)(i * 37 % 127) - 63) / 19.0f; }
    ds4_gpu_tensor *dx = upload(x, bytes), *dy = upload(NULL, bytes), *fresh = upload(NULL, bytes);
    ds4_gpu_tensor *out = upload(NULL, out_bytes), *ref = upload(NULL, out_bytes);
    unsigned mismatches = 0;
    for (unsigned op = 0; op < 2; op++) {
        CHECK(ds4_gpu_rms_norm_weight_rows_q8_tensor(dy, dx, map, MAP_SIZE,
                                                    F32_OFFSET, Q8_WIDTH, Q8_ROWS, EPS));
        if (op == 0) {
            CHECK(ds4_gpu_inkling_norm(dy, dx, map, MAP_SIZE, OFFSET, Q8_WIDTH, Q8_ROWS));
        } else {
            CHECK(ds4_gpu_inkling_add_scale(dy, dx, NULL, 0.5f, Q8_WIDTH * Q8_ROWS));
        }
        CHECK(ds4_gpu_tensor_read(dy, 0, y, bytes));
        CHECK(ds4_gpu_tensor_write(fresh, 0, y, bytes));
        CHECK(ds4_gpu_matmul_q8_0_tensor(out, map, MAP_SIZE, Q8_OFFSET, Q8_WIDTH, Q8_OUTPUT, dy, Q8_ROWS));
        CHECK(ds4_gpu_matmul_q8_0_tensor(ref, map, MAP_SIZE, Q8_OFFSET, Q8_WIDTH, Q8_OUTPUT, fresh, Q8_ROWS));
        CHECK(ds4_gpu_tensor_read(out, 0, got, out_bytes));
        CHECK(ds4_gpu_tensor_read(ref, 0, want, out_bytes));
        unsigned same = memcmp(got, want, out_bytes) == 0;
        fprintf(stderr, "norm-Q8 reuse after Inkling %s: %s\n", op ? "arithmetic" : "norm", same ? "pass" : "FAIL");
        mismatches += !same;
    }
    CHECK(mismatches == 0);
    ds4_gpu_tensor_free(dx); ds4_gpu_tensor_free(dy); ds4_gpu_tensor_free(fresh);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(ref);
    free(x); free(y); free(got); free(want);
}

int main(void) {
    /* The Q8 invalidation regression must exercise the producer/consumer path. */
    CHECK(unsetenv("DS4_CUDA_NO_NORM_Q8EMIT") == 0);
    CHECK(setenv("DS4_CUDA_PREFILL_PATH", "mmq", 1) == 0);
    CHECK(ds4_gpu_init());
    void *map = NULL; CHECK(posix_memalign(&map, OFFSET, MAP_SIZE) == 0);
    memset(map, 0, MAP_SIZE);
    uint16_t *w = (uint16_t *)((char *)map + OFFSET);
    for (unsigned i = 0; i < WIDTH_MAX; i++) { w[i] = bits(weight(i)); }
    float *f32 = (float *)((char *)map + F32_OFFSET);
    for (unsigned i = 0; i < Q8_WIDTH; i++) { f32[i] = 1; }
    uint8_t *q8 = (uint8_t *)map + Q8_OFFSET;
    for (unsigned o = 0; o < Q8_OUTPUT; o++) {
        for (unsigned b = 0; b < Q8_WIDTH / Q8_BLOCK; b++) {
            uint8_t *block = q8 + (o * (Q8_WIDTH / Q8_BLOCK) + b) * Q8_BYTES;
            uint16_t scale = HALF_ONE; memcpy(block, &scale, sizeof(scale));
            if (o / Q8_BLOCK == b) { block[sizeof(scale) + o % Q8_BLOCK] = 1; }
        }
    }
    CHECK(ds4_gpu_set_model_map(map, MAP_SIZE));
    q8_reuse(map);
    const unsigned widths[] = {17, 128, 320, 4096, 4800, WIDTH_MAX}, rows[] = {1, 7, 65};
    for (unsigned widx = 0; widx < sizeof(widths) / sizeof(widths[0]); widx++) {
        for (unsigned ridx = 0; ridx < sizeof(rows) / sizeof(rows[0]); ridx++) { norm_case(map, rows[ridx], widths[widx]); }
    }
    pointwise(); aliases(map); captured(map); ds4_gpu_cleanup(); free(map);
    puts("Inkling norm/residual checks passed"); return 0;
}
