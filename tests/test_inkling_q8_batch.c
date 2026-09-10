/* Aligned Q8 prefill tiles must retain the one-row projection arithmetic. */
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { HIDDEN = 4096, MIDDLE = 16384, QK = 32, Q8_BYTES = 34,
       Q8_TYPE = 8, OFFSET = 4096, REPEATS = 3,
       UP_BYTES = HIDDEN * (2 * MIDDLE / QK) * Q8_BYTES,
       DOWN_OFFSET = OFFSET + UP_BYTES,
       DOWN_BYTES = MIDDLE * (HIDDEN / QK) * Q8_BYTES,
       MAP_BYTES = DOWN_OFFSET + DOWN_BYTES };
typedef struct { uint16_t d; int8_t qs[QK]; } q8_block;
#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); exit(1); \
} } while (0)

static ds4_gpu_tensor *upload(const void *data, size_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes); CHECK(t);
    if (data) { CHECK(ds4_gpu_tensor_write(t, 0, data, bytes)); }
    return t;
}

static double now(void) {
    struct timespec t; CHECK(clock_gettime(CLOCK_MONOTONIC, &t) == 0);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void reference(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
                       const void *map, const ds4_gpu_tensor_record *w, unsigned rows) {
    const size_t in = w->dims[0] * sizeof(float), dest = w->dims[1] * sizeof(float);
    for (unsigned row = 0; row < rows; row++) {
        ds4_gpu_tensor *a = ds4_gpu_tensor_view(x, row * in, in);
        ds4_gpu_tensor *b = ds4_gpu_tensor_view(out, row * dest, dest);
        CHECK(a && b && ds4_gpu_matmul_q8_0_tensor(b, map, MAP_BYTES,
            w->offset, w->dims[0], w->dims[1], a, 1));
        ds4_gpu_tensor_free(a); ds4_gpu_tensor_free(b);
    }
}

static void batch_case(const void *map, const ds4_gpu_tensor_record *w, unsigned rows) {
    const unsigned k = w->dims[0], m = w->dims[1];
    const size_t in_bytes = (size_t)rows * k * sizeof(float);
    const size_t out_bytes = (size_t)rows * m * sizeof(float);
    float *x = malloc(in_bytes), *got = malloc(out_bytes), *want = malloc(out_bytes);
    CHECK(x && got && want);
    for (size_t i = 0; i < in_bytes / sizeof(float); i++) {
        x[i] = ((int)(i * 37 % 251) - 125) / 53.0f + 0.000031f;
        if ((i / k) % 5 == 0) { x[i] = 0; }
    }
    ds4_gpu_tensor *dx = upload(x, in_bytes), *out = upload(NULL, out_bytes);
    reference(out, dx, map, w, rows);
    CHECK(ds4_gpu_tensor_read(out, 0, want, out_bytes));
    CHECK(ds4_gpu_inkling_q8(out, dx, map, MAP_BYTES, w->offset,
        w->bytes, k, m, rows) == 1);
    CHECK(ds4_gpu_tensor_read(out, 0, got, out_bytes));
    CHECK(memcmp(got, want, out_bytes) == 0);

    CHECK(ds4_gpu_inkling_q8(out, dx, map, MAP_BYTES, w->offset,
        w->bytes, k, m, 1) == 0);
    CHECK(ds4_gpu_inkling_q8(out, dx, map, MAP_BYTES, w->offset,
        w->bytes - 1, k, m, rows) == -1);
    CHECK(ds4_gpu_inkling_q8(out, dx, map, MAP_BYTES, UINT64_MAX,
        w->bytes, k, m, rows) == -1);
    CHECK(ds4_gpu_inkling_q8(out, dx, map, MAP_BYTES, w->offset,
        w->bytes, k, m, rows + 1) == (rows == 8192 ? 0 : -1));
    CHECK(ds4_gpu_inkling_q8(out, out, map, MAP_BYTES, w->offset,
        w->bytes, k, m, rows) == -1);
    const char *kills[] = {"DS4_INKLING_NO_Q8_BATCH", "DS4_CUDA_NO_Q8_ALIGNED_NC"};
    for (unsigned i = 0; i < sizeof(kills) / sizeof(kills[0]); i++) {
        CHECK(setenv(kills[i], "1", 1) == 0);
        CHECK(ds4_gpu_inkling_q8(out, dx, map, MAP_BYTES, w->offset,
            w->bytes, k, m, rows) == 0);
        CHECK(unsetenv(kills[i]) == 0);
    }
    CHECK(ds4_gpu_tensor_read(out, 0, got, out_bytes));
    CHECK(memcmp(got, want, out_bytes) == 0);

    double elapsed[2] = {0};
    if (rows == 64) {
        for (unsigned mode = 0; mode < 2; mode++) {
            CHECK(ds4_gpu_synchronize()); const double start = now();
            for (unsigned repeat = 0; repeat < REPEATS; repeat++) {
                if (mode == 0) { reference(out, dx, map, w, rows); }
                else { CHECK(ds4_gpu_inkling_q8(out, dx, map, MAP_BYTES, w->offset,
                    w->bytes, k, m, rows) == 1); }
            }
            CHECK(ds4_gpu_synchronize()); elapsed[mode] = (now() - start) / REPEATS;
        }
    }
    printf("dense Q8 k=%u m=%u rows=%u exact; %.3f -> %.3f ms\n",
           k, m, rows, elapsed[0] * 1e3, elapsed[1] * 1e3);
    ds4_gpu_tensor_free(dx); ds4_gpu_tensor_free(out);
    free(x); free(got); free(want);
}

int main(void) {
    CHECK(sizeof(q8_block) == Q8_BYTES);
    CHECK(unsetenv("DS4_INKLING_NO_Q8_BATCH") == 0);
    CHECK(unsetenv("DS4_CUDA_NO_Q8_ALIGNED_NC") == 0);
    CHECK(ds4_gpu_init());
    void *map = NULL; CHECK(posix_memalign(&map, OFFSET, MAP_BYTES) == 0);
    q8_block *w = (q8_block *)((char *)map + OFFSET);
    for (size_t b = 0; b < (UP_BYTES + DOWN_BYTES) / Q8_BYTES; b++) {
        w[b].d = (b % 2 ? 0xa400 : 0x2400) + b % 16;
        for (unsigned j = 0; j < QK; j++) { w[b].qs[j] = (int)((b * 17 + j * 37) % 255) - 127; }
    }
    ds4_gpu_tensor_record weights[] = {
        {.name="inkling_up.weight", .name_len=17, .type=Q8_TYPE, .ndim=2,
         .dims={HIDDEN, 2 * MIDDLE}, .offset=OFFSET, .bytes=UP_BYTES},
        {.name="inkling_down.weight", .name_len=19, .type=Q8_TYPE, .ndim=2,
         .dims={MIDDLE, HIDDEN}, .offset=DOWN_OFFSET, .bytes=DOWN_BYTES},
    };
    ds4_gpu_tensor *input = upload(NULL, 2 * HIDDEN * sizeof(float));
    float *sentinel = malloc(4 * MIDDLE * sizeof(float)); CHECK(sentinel);
    for (unsigned i = 0; i < 4 * MIDDLE; i++) { sentinel[i] = -19.25f; }
    ds4_gpu_tensor *output = upload(sentinel, 4 * MIDDLE * sizeof(float));
    CHECK(ds4_gpu_inkling_q8(output, input, map, MAP_BYTES, OFFSET,
        UP_BYTES, HIDDEN, 2 * MIDDLE, 2) == 0);
    float *unchanged = malloc(4 * MIDDLE * sizeof(float)); CHECK(unchanged);
    CHECK(ds4_gpu_tensor_read(output, 0, unchanged, 4 * MIDDLE * sizeof(float)));
    CHECK(memcmp(sentinel, unchanged, 4 * MIDDLE * sizeof(float)) == 0);
    free(sentinel); free(unchanged); ds4_gpu_tensor_free(input); ds4_gpu_tensor_free(output);
    CHECK(ds4_gpu_build_derived_artifacts_from_records(map, MAP_BYTES, weights, 2) == 2);
    CHECK(ds4_gpu_set_model_map(map, MAP_BYTES));
    const unsigned rows[] = {2, 3, 7, 8, 9, 15, 64, 65, 2048, 8191, 8192};
    for (unsigned i = 0; i < 2; i++) {
        for (unsigned r = 0; r < sizeof(rows) / sizeof(rows[0]); r++) {
            batch_case(map, &weights[i], rows[r]);
        }
    }
    ds4_gpu_cleanup(); free(map); puts("Inkling aligned Q8 batch checks passed"); return 0;
}
