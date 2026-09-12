/* Solar K-FP8/V-FP4 prefill attention: the warp-specialized kernel must
 * write the same bytes as the GQA-pair kernel at ragged, windowed, ring
 * and deep production shapes.  Synthetic packed rows keep the fixture
 * small and independent of weights. */
#include "ds4_gpu.h"
#include "cuda/mmq/ds4_mmq.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { HEAD_DIM = 128, HEAD_GROUP = 8, REPEATS = 3 };

#define CHECK(ok) do { \
    if (!(ok)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #ok); \
        exit(1); \
    } \
} while (0)

static const char *const CONTROL = "DS4_SOLAR_FATTN_WS";

static uint32_t random_word(uint32_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static double seconds(void) {
    struct timespec now;
    CHECK(clock_gettime(CLOCK_MONOTONIC, &now) == 0);
    return now.tv_sec + now.tv_nsec * 1e-9;
}

static void run_shape(uint32_t rows, uint32_t pos, uint32_t cap,
                      uint32_t window, uint32_t kv_heads) {
    const uint32_t heads = HEAD_GROUP * kv_heads;
    const size_t count = (size_t)rows * heads * HEAD_DIM;
    const size_t bytes = count * sizeof(float);
    const uint64_t row_bytes = ds4_gpu_solar_kv_row_bytes(
        DS4_SOLAR_KV_KFP8_VFP4, kv_heads, HEAD_DIM);
    const size_t cache_bytes = cap * row_bytes;
    const size_t kv_dim = (size_t)kv_heads * HEAD_DIM;
    float *query = malloc(bytes), *reference = malloc(bytes);
    float *result = malloc(bytes);
    uint8_t *cache = calloc(1, cache_bytes);
    CHECK(query && reference && result && cache);
    uint32_t seed = 42;
    for (size_t i = 0; i < count; i++) {
        query[i] = ((int)(random_word(&seed) % 2001) - 1000) * 0.001f;
    }
    for (uint32_t r = 0; r < cap; r++) {
        uint8_t *row = cache + r * row_bytes;
        for (size_t i = 0; i < kv_dim; i++) {
            /* Finite e4m3 codes with both signs; every FP4 code. */
            row[i] = (random_word(&seed) % 0x70) |
                     (random_word(&seed) & 0x80);
        }
        for (size_t i = 0; i < kv_dim / 2; i++) {
            row[kv_dim + i] = (uint8_t)random_word(&seed);
        }
        uint16_t *scales = (uint16_t *)(row + kv_dim * 3 / 2);
        for (uint32_t h = 0; h < kv_heads * 2; h++) {
            scales[h] = r % 31 == 0 ? 0 : 0x2800 + ((r + h) % 4) * 0x400;
        }
    }
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(cache_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(bytes);
    CHECK(q && kv && out);
    CHECK(ds4_gpu_tensor_write(q, 0, query, bytes));
    CHECK(ds4_gpu_tensor_write(kv, 0, cache, cache_bytes));
    CHECK(ds4_gpu_synchronize());
    double times[2];
    for (int variant = 0; variant < 2; variant++) {
        CHECK(setenv(CONTROL, variant ? "1" : "0", 1) == 0);
        /* Poison outputs so an omitted write cannot pass the compare. */
        for (size_t i = 0; i < count; i++) { result[i] = NAN; }
        CHECK(ds4_gpu_tensor_write(out, 0, result, bytes));
        CHECK(ds4_gpu_synchronize());
        /* The direct entry fails on a launch error instead of falling back. */
        CHECK(ds4_mmq_solar_prefill_attn_hmma(
            (float *)ds4_gpu_tensor_ptr(out), ds4_gpu_tensor_ptr(q),
            ds4_gpu_tensor_ptr(kv), DS4_SOLAR_KV_KFP8_VFP4, row_bytes,
            rows, pos, heads, kv_heads, HEAD_DIM, cap, window,
            1.0f / sqrtf(HEAD_DIM), NULL) == 0);
        CHECK(ds4_gpu_synchronize());
        const double start = seconds();
        for (int rep = 0; rep < REPEATS; rep++) {
            CHECK(ds4_mmq_solar_prefill_attn_hmma(
                (float *)ds4_gpu_tensor_ptr(out), ds4_gpu_tensor_ptr(q),
                ds4_gpu_tensor_ptr(kv), DS4_SOLAR_KV_KFP8_VFP4, row_bytes,
                rows, pos, heads, kv_heads, HEAD_DIM, cap, window,
                1.0f / sqrtf(HEAD_DIM), NULL) == 0);
        }
        CHECK(ds4_gpu_synchronize());
        times[variant] = (seconds() - start) * 1000.0 / REPEATS;
        CHECK(ds4_gpu_tensor_read(out, 0, result, bytes));
        for (size_t i = 0; i < count; i++) { CHECK(isfinite(result[i])); }
        if (variant == 0) { memcpy(reference, result, bytes); }
    }
    size_t different = 0;
    for (size_t i = 0; i < count; i++) {
        different += memcmp(result + i, reference + i, sizeof(float)) != 0;
    }
    printf("rows=%u pos=%u cap=%u window=%u heads=%u "
           "pair=%.3f ms ws=%.3f ms different=%zu\n",
           rows, pos, cap, window, heads, times[0], times[1], different);
    CHECK(different == 0);
    CHECK(unsetenv(CONTROL) == 0);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(q);
    free(cache);
    free(result);
    free(reference);
    free(query);
}

int main(void) {
    /* These gates would otherwise make both trials use a fallback. */
    CHECK(setenv("DS4_SOLAR_KV_PREFILL_HMMA", "1", 1) == 0);
    CHECK(setenv("DS4_SOLAR_FATTN_GQA2", "1", 1) == 0);
    CHECK(setenv("DS4_FATTN_HMMA_LDSM", "1", 1) == 0);
    CHECK(ds4_gpu_init());
    CHECK(ds4_mmq_init(0) == 0);
    const uint32_t widths[] = {1, 63, 64, 65, 127, 257};
    for (size_t i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) {
        /* 2 KV heads: 392-byte rows take the 8-byte copies. */
        run_shape(widths[i], 0, 320, 0, 2);
        run_shape(widths[i], 901, 512, 127, 2);
        /* 4 KV heads: 784-byte rows take the 16-byte copies. */
        run_shape(widths[i], 901, 1200, 0, 4);
    }
    run_shape(257, 4096, 4353, 0, 2);
    run_shape(129, 4096, 4353, 33, 8);
    run_shape(4096, 0, 4160, 0, 8);
    run_shape(4096, 4096, 8257, 0, 8);
    run_shape(4096, 61440, 65601, 0, 8);
    puts("Solar attention rollback parity passed");
    return 0;
}
