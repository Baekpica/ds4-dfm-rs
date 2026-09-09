/* Source 4-tap residual convolution: orientation, chunking and live history.
 * No model weights are loaded. FP64 reference sums use BF16 inputs/weights. */
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TAPS = 4, HISTORY = TAPS - 1, MAX_CHANNELS = 4096, MAP_OFFSET = 4096 };

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); \
        exit(1); \
    } \
} while (0)

static uint16_t to_bf16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float from_bf16(uint16_t x) {
    uint32_t bits = (uint32_t)x << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static float rounded(float x) {
    return from_bf16(to_bf16(x));
}

static void reference(float *out, const float *x, const float *history,
                      const uint16_t *weight, unsigned channels, unsigned rows) {
    for (unsigned t = 0; t < rows; t++) {
        for (unsigned c = 0; c < channels; c++) {
            double sum = 0;
            for (unsigned tap = 0; tap < TAPS; tap++) {
                int pos = (int)t - HISTORY + (int)tap;
                float input = pos < 0 ? history[(pos + HISTORY) * channels + c]
                                      : x[(size_t)pos * channels + c];
                sum += (double)rounded(input) * from_bf16(weight[c * TAPS + tap]);
            }
            sum += rounded(x[(size_t)t * channels + c]);
            out[(size_t)t * channels + c] = rounded((float)sum);
        }
    }
}

static ds4_gpu_tensor *upload(const void *data, size_t bytes) {
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(bytes);
    CHECK(out);
    if (data) {
        CHECK(ds4_gpu_tensor_write(out, 0, data, bytes));
    }
    return out;
}

static void compare(const char *label, const float *got, const float *want, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (got[i] != want[i] || !isfinite(got[i])) {
            fprintf(stderr, "%s[%zu]: got %.9g want %.9g\n", label, i, got[i], want[i]);
            exit(1);
        }
    }
}

static void run_shape(const void *map, size_t map_size, unsigned channels, unsigned rows) {
    const size_t count = (size_t)channels * rows;
    const size_t bytes = count * sizeof(float);
    const size_t history_count = (size_t)HISTORY * channels;
    const size_t history_bytes = history_count * sizeof(float);
    float *x = malloc(bytes), *got = malloc(bytes), *want = malloc(bytes);
    float *history = malloc(history_bytes), *next = malloc(history_bytes);
    CHECK(x && got && want && history && next);
    for (size_t i = 0; i < count; i++) {
        x[i] = (float)((int)((i * 73 + 19) % 509) - 254) / 113.0f;
    }
    for (size_t i = 0; i < history_count; i++) {
        history[i] = (float)((int)(i % 19) - 9) / 7.0f;
    }
    const uint16_t *weight = (const uint16_t *)((const char *)map + MAP_OFFSET);
    reference(want, x, history, weight, channels, rows);
    ds4_gpu_tensor *dx = upload(x, bytes), *dy = upload(NULL, bytes);
    ds4_gpu_tensor *dh = upload(history, history_bytes), *dn = upload(NULL, history_bytes);

    /* Read-only input history and a separate output state support snapshots. */
    CHECK(ds4_gpu_inkling_sconv(dy, dn, dx, dh, map, map_size, MAP_OFFSET, channels, rows));
    CHECK(ds4_gpu_tensor_read(dy, 0, got, bytes));
    compare("full", got, want, count);
    CHECK(ds4_gpu_tensor_read(dh, 0, next, history_bytes));
    compare("input history", next, history, history_count);
    CHECK(ds4_gpu_tensor_read(dn, 0, next, history_bytes));
    for (unsigned h = 0; h < HISTORY; h++) {
        int pos = (int)rows - HISTORY + (int)h;
        for (unsigned c = 0; c < channels; c++) {
            float expected = pos < 0 ? rounded(history[(pos + HISTORY) * channels + c])
                                      : rounded(x[(size_t)pos * channels + c]);
            CHECK(next[h * channels + c] == expected);
        }
    }

    /* Widths below the history length exercise overlapping in-place shifts. */
    const unsigned chunks[] = {1, 2, 3, 7};
    for (unsigned k = 0; k < sizeof(chunks) / sizeof(chunks[0]); k++) {
        CHECK(ds4_gpu_tensor_write(dh, 0, history, history_bytes));
        for (unsigned at = 0; at < rows;) {
            unsigned n = rows - at < chunks[k] ? rows - at : chunks[k];
            size_t offset = (size_t)at * channels * sizeof(float);
            size_t span = (size_t)n * channels * sizeof(float);
            ds4_gpu_tensor *ix = ds4_gpu_tensor_view(dx, offset, span);
            ds4_gpu_tensor *oy = ds4_gpu_tensor_view(dy, offset, span);
            CHECK(ix && oy);
            CHECK(ds4_gpu_inkling_sconv(oy, dh, ix, dh, map, map_size,
                                        MAP_OFFSET, channels, n));
            ds4_gpu_tensor_free(ix);
            ds4_gpu_tensor_free(oy);
            at += n;
        }
        CHECK(ds4_gpu_tensor_read(dy, 0, got, bytes));
        compare("chunk", got, want, count);
        CHECK(ds4_gpu_tensor_read(dh, 0, history, history_bytes));
        compare("chunk history", history, next, history_count);
        for (size_t i = 0; i < history_count; i++) {
            history[i] = (float)((int)(i % 19) - 9) / 7.0f;
        }
    }

    CHECK(!ds4_gpu_inkling_sconv(dx, dn, dx, dh, map, map_size, MAP_OFFSET, channels, rows));
    CHECK(!ds4_gpu_inkling_sconv(dy, dn, dx, dh, map, MAP_OFFSET, MAP_OFFSET, channels, rows));
    CHECK(!ds4_gpu_inkling_sconv(dy, dn, dx, dh, map, map_size, MAP_OFFSET, channels, 0));
    CHECK(!ds4_gpu_inkling_sconv(dy, dn, dx, dh, map, map_size, MAP_OFFSET, 0, rows));
    CHECK(!ds4_gpu_inkling_sconv(dy, dn, dx, dh, map, map_size, MAP_OFFSET, channels, rows + 1));
    ds4_gpu_tensor_free(dx);
    ds4_gpu_tensor_free(dy);
    ds4_gpu_tensor_free(dh);
    ds4_gpu_tensor_free(dn);
    free(x); free(got); free(want); free(history); free(next);
    printf("sconv channels=%u rows=%u full/chunk/decode/history exact\n", channels, rows);
}

static void capture_history(const void *map, size_t map_size) {
    enum { CHANNELS = 7, STEPS = 9 };
    float history[HISTORY * CHANNELS] = {0}, x[CHANNELS], got[CHANNELS], want[CHANNELS];
    ds4_gpu_tensor *dh = upload(history, sizeof(history));
    ds4_gpu_tensor *dx = upload(NULL, sizeof(x)), *dy = upload(NULL, sizeof(x));
    struct ds4_layer_graph_key key = {0};
    key.n_tok = 1;
    key.cur_hc = dx;
    key.after_ffn_hc = dy;
    key.raw_cache = dh;
    unsigned captures = 0, replays = 0;
    const uint16_t *weight = (const uint16_t *)((const char *)map + MAP_OFFSET);
    for (unsigned t = 0; t < STEPS; t++) {
        for (unsigned c = 0; c < CHANNELS; c++) {
            x[c] = (float)((int)(t * CHANNELS + c) - 17) / 11.0f;
        }
        reference(want, x, history, weight, CHANNELS, 1);
        CHECK(ds4_gpu_tensor_write(dx, 0, x, sizeof(x)));
        int mode = ds4_cuda_layer_graph_begin_or_replay(0, &key);
        if (mode != 1) {
            CHECK(ds4_gpu_inkling_sconv(dy, dh, dx, dh, map, map_size,
                                        MAP_OFFSET, CHANNELS, 1));
        }
        if (mode == 0) {
            ds4_cuda_layer_graph_end_or_commit(0);
            captures++;
        }
        replays += mode == 1;
        CHECK(ds4_gpu_tensor_read(dy, 0, got, sizeof(got)));
        compare("captured decode", got, want, CHANNELS);
        memmove(history, history + CHANNELS, (HISTORY - 1) * CHANNELS * sizeof(float));
        for (unsigned c = 0; c < CHANNELS; c++) {
            history[(HISTORY - 1) * CHANNELS + c] = rounded(x[c]);
        }
    }
    float state[HISTORY * CHANNELS];
    CHECK(ds4_gpu_tensor_read(dh, 0, state, sizeof(state)));
    compare("captured history", state, history, HISTORY * CHANNELS);
    CHECK(captures == 1 && replays == STEPS - 2);
    printf("sconv capture=%u replay=%u live-history exact\n", captures, replays);
    ds4_gpu_tensor_free(dx); ds4_gpu_tensor_free(dy); ds4_gpu_tensor_free(dh);
}

int main(void) {
    CHECK(ds4_gpu_init());
    const size_t map_size = MAP_OFFSET + MAX_CHANNELS * TAPS * sizeof(uint16_t);
    void *map = NULL;
    CHECK(posix_memalign(&map, MAP_OFFSET, map_size) == 0);
    memset(map, 0, map_size);
    uint16_t *weight = (uint16_t *)((char *)map + MAP_OFFSET);
    for (unsigned i = 0; i < MAX_CHANNELS * TAPS; i++) {
        weight[i] = to_bf16((float)((int)((i * 13) % 29) - 14) / 31.0f);
    }
    CHECK(ds4_gpu_set_model_map(map, map_size));
    const unsigned widths[] = {7, 1024, 4096}, lengths[] = {1, 2, 3, 4, 9, 33};
    for (unsigned c = 0; c < sizeof(widths) / sizeof(widths[0]); c++) {
        for (unsigned t = 0; t < sizeof(lengths) / sizeof(lengths[0]); t++) {
            run_shape(map, map_size, widths[c], lengths[t]);
        }
    }
    capture_history(map, map_size);
    ds4_gpu_cleanup();
    free(map);
    puts("Inkling convolution checks passed");
    return 0;
}
