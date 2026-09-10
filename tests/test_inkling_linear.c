/* Compare the fused projection with the existing stable BF16 + store path. */
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { OFFSET = 4096, WIDTH = 4096, MTP_OUTPUT = 32768,
       MAP_BYTES = OFFSET + WIDTH * MTP_OUTPUT * sizeof(uint16_t),
       REPEATS = 12, BF_SHIFT = 16, BF_HALF = 0x7fff };
#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); exit(1); \
} } while (0)

static uint16_t bits(float x) {
    uint32_t u; memcpy(&u, &x, sizeof(u));
    return (u + BF_HALF + ((u >> BF_SHIFT) & 1)) >> BF_SHIFT;
}

static ds4_gpu_tensor *upload(const void *data, size_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes); CHECK(t);
    if (data) { CHECK(ds4_gpu_tensor_write(t, 0, data, bytes)); }
    return t;
}

static void reference(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
                       const void *map, unsigned k, unsigned m, unsigned rows) {
    CHECK(ds4_gpu_matmul_bf16_stable_rows_tensor(out, map, MAP_BYTES,
                                                OFFSET, k, m, x, rows));
    CHECK(ds4_gpu_inkling_add_scale(out, out, NULL, 1, (uint64_t)rows * m));
}

static double now(void) {
    struct timespec t; CHECK(clock_gettime(CLOCK_MONOTONIC, &t) == 0);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void linear_case(const void *map, unsigned k, unsigned m, unsigned rows) {
    const size_t in_bytes = (size_t)k * rows * sizeof(float);
    const size_t out_bytes = (size_t)m * rows * sizeof(float);
    float *x = malloc(in_bytes), *got = malloc(out_bytes), *want = malloc(out_bytes);
    CHECK(x && got && want);
    for (size_t i = 0; i < in_bytes / sizeof(float); i++) {
        x[i] = ((int)(i * 37 % 251) - 125) / 53.0f + 0.000031f;
        if ((i / k) % 3 == 1) { x[i] *= 1e-30f; }
        if ((i / k) % 3 == 2) { x[i] *= 1e12f; }
    }
    ds4_gpu_tensor *dx = upload(x, in_bytes), *out = upload(NULL, out_bytes);
    reference(out, dx, map, k, m, rows);
    CHECK(ds4_gpu_tensor_read(out, 0, want, out_bytes));
    CHECK(ds4_gpu_inkling_linear(out, dx, map, MAP_BYTES, OFFSET, k, m, rows) == 1);
    CHECK(ds4_gpu_tensor_read(out, 0, got, out_bytes));
    CHECK(memcmp(got, want, out_bytes) == 0);
    CHECK(setenv("DS4_INKLING_NO_LINEAR_TILE", "1", 1) == 0);
    CHECK(ds4_gpu_inkling_linear(out, dx, map, MAP_BYTES, OFFSET, k, m, rows) == 1);
    CHECK(unsetenv("DS4_INKLING_NO_LINEAR_TILE") == 0);
    CHECK(ds4_gpu_tensor_read(out, 0, got, out_bytes));
    CHECK(memcmp(got, want, out_bytes) == 0);

    // Convert every input before any output, including partial aliases.
    const size_t storage = (in_bytes > out_bytes ? in_bytes : out_bytes) + sizeof(float);
    ds4_gpu_tensor *base = upload(NULL, storage);
    ds4_gpu_tensor *input = ds4_gpu_tensor_view(base, 0, in_bytes);
    CHECK(input);
    for (unsigned shift = 0; shift <= sizeof(float); shift += sizeof(float)) {
        ds4_gpu_tensor *alias = ds4_gpu_tensor_view(base, shift, out_bytes); CHECK(alias);
        CHECK(ds4_gpu_tensor_write(input, 0, x, in_bytes));
        CHECK(ds4_gpu_inkling_linear(alias, input, map, MAP_BYTES, OFFSET, k, m, rows) == 1);
        CHECK(ds4_gpu_tensor_read(alias, 0, got, out_bytes));
        CHECK(memcmp(got, want, out_bytes) == 0);
        ds4_gpu_tensor_free(alias);
    }

    double elapsed[3];
    for (unsigned mode = 0; mode < 3; mode++) {
        if (mode == 1) { CHECK(setenv("DS4_INKLING_NO_LINEAR_TILE", "1", 1) == 0); }
        else { CHECK(unsetenv("DS4_INKLING_NO_LINEAR_TILE") == 0); }
        CHECK(ds4_gpu_synchronize());
        const double start = now();
        for (unsigned i = 0; i < REPEATS; i++) {
            if (mode == 0) { reference(out, dx, map, k, m, rows); }
            else { CHECK(ds4_gpu_inkling_linear(out, dx, map, MAP_BYTES, OFFSET, k, m, rows) == 1); }
        }
        CHECK(ds4_gpu_synchronize()); elapsed[mode] = (now() - start) / REPEATS;
    }
    printf("linear k=%u m=%u rows=%u exact; reference=%.3f us grouped=%.3f us tiled=%.3f us\n",
           k, m, rows, elapsed[0] * 1e6, elapsed[1] * 1e6, elapsed[2] * 1e6);
    ds4_gpu_tensor_free(input); ds4_gpu_tensor_free(base);
    ds4_gpu_tensor_free(dx); ds4_gpu_tensor_free(out);
    free(x); free(got); free(want);
}

static void unsupported(const void *map) {
    float x[WIDTH] = {0}, got[WIDTH];
    for (unsigned i = 0; i < WIDTH; i++) { x[i] = -123.25f; }
    ds4_gpu_tensor *dx = upload(x, sizeof(x)), *out = upload(x, sizeof(x));
    CHECK(ds4_gpu_inkling_linear(out, dx, map, MAP_BYTES, OFFSET, 75, 320, 1) == 0);
    CHECK(ds4_gpu_inkling_linear(out, dx, map, MAP_BYTES, OFFSET + 2, WIDTH, WIDTH, 1) == 0);
    const char *kills[] = {"DS4_INKLING_NO_LINEAR", "DS4_CUDA_NO_BF16_ROWS_WARP"};
    for (unsigned i = 0; i < sizeof(kills) / sizeof(kills[0]); i++) {
        CHECK(setenv(kills[i], "1", 1) == 0);
        CHECK(ds4_gpu_inkling_linear(out, dx, map, MAP_BYTES, OFFSET, WIDTH, WIDTH, 1) == 0);
        CHECK(unsetenv(kills[i]) == 0);
    }
    CHECK(ds4_gpu_inkling_linear(out, dx, map, MAP_BYTES, OFFSET, WIDTH, WIDTH, 2) == -1);
    CHECK(ds4_gpu_inkling_linear(out, dx, map, OFFSET, OFFSET, WIDTH, WIDTH, 1) == -1);
    CHECK(ds4_gpu_inkling_linear(out, dx, map, MAP_BYTES, UINT64_MAX, WIDTH, WIDTH, 1) == -1);
    CHECK(ds4_gpu_inkling_linear(out, dx, map, MAP_BYTES, OFFSET, 0, WIDTH, 1) == -1);
    CHECK(ds4_gpu_inkling_linear(NULL, dx, map, MAP_BYTES, OFFSET, WIDTH, WIDTH, 1) == -1);
    CHECK(ds4_gpu_tensor_read(out, 0, got, sizeof(got)));
    CHECK(memcmp(got, x, sizeof(got)) == 0);
    ds4_gpu_tensor_free(dx); ds4_gpu_tensor_free(out);
    puts("unsupported/invalid/disabled paths leave output untouched");
}

int main(void) {
    CHECK(unsetenv("DS4_INKLING_NO_LINEAR") == 0);
    CHECK(unsetenv("DS4_INKLING_NO_LINEAR_TILE") == 0);
    CHECK(unsetenv("DS4_CUDA_NO_BF16_ROWS_WARP") == 0);
    CHECK(ds4_gpu_init());
    void *map = NULL; CHECK(posix_memalign(&map, OFFSET, MAP_BYTES) == 0);
    memset(map, 0, MAP_BYTES);
    uint16_t *weight = (uint16_t *)((char *)map + OFFSET);
    for (size_t i = 0; i < (MAP_BYTES - OFFSET) / sizeof(uint16_t); i++) {
        weight[i] = bits(((int)(i * 17 % 257) - 128) / 127.0f);
    }
    CHECK(ds4_gpu_set_model_map(map, MAP_BYTES));
    const unsigned rows[] = {1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 64, 65, 129};
    for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); i++) {
        linear_case(map, WIDTH, WIDTH, rows[i]);
    }
    linear_case(map, WIDTH, 1024, 64); linear_case(map, WIDTH, 512, 1);
    linear_case(map, 512, 258, 9); linear_case(map, WIDTH, MTP_OUTPUT, 9);
    linear_case(map, WIDTH, MTP_OUTPUT, 129); linear_case(map, WIDTH, 512, 512);
    linear_case(map, 512, 320, 2048); linear_case(map, 4800, WIDTH, 16);
    unsupported(map); ds4_gpu_cleanup(); free(map);
    puts("Inkling linear checks passed"); return 0;
}
