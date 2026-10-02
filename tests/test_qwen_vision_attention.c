/* FP32 vision attention: real head geometry, ragged image segments, and
 * sampled F64 reference rows. No GGUF or weight owner is needed. */
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", \
    __FILE__, __LINE__, #x); exit(1); } } while (0)

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec * 1e-6;
}

enum segments { UNIFORM_SEGMENTS, IMAGE_SEGMENTS };
static const uint32_t image_rows[] = {256u, 3072u, 1024u, 6144u};

static void check_case(uint32_t rows, uint32_t heads, uint32_t dim,
                       uint32_t segment, float scale, enum segments layout) {
    const size_t width = heads * dim, n = rows * width;
    float *qkv = malloc(3 * n * sizeof(float));
    float *got = malloc(n * sizeof(float)), *old = malloc(n * sizeof(float));
    int32_t *begin = malloc(rows * sizeof(int32_t));
    int32_t *end = malloc(rows * sizeof(int32_t));
    double *scores = malloc(rows * sizeof(double));
    CHECK(qkv && got && old && begin && end && scores);
    uint32_t seed = 42;
    for (size_t i = 0; i < 3 * n; i++) {
        seed = 1664525u * seed + 1013904223u;
        qkv[i] = ((float)(seed >> 8) / 8388608.0f - 1.0f) * scale;
    }
    uint32_t base = 0, image = 0;
    for (uint32_t r = 0; r < rows; r++) {
        if (layout == IMAGE_SEGMENTS) {
            if (r == base + image_rows[image]) {
                base += image_rows[image++];
                CHECK(image < sizeof(image_rows) / sizeof(image_rows[0]));
            }
            begin[r] = base;
            end[r] = base + image_rows[image];
            continue;
        }
        begin[r] = r / segment * segment;
        end[r] = begin[r] + segment < rows ? begin[r] + segment : rows;
    }
    /* Empty segments are legal at the attention seam and produce zeros. */
    if (layout == UNIFORM_SEGMENTS && segment < rows) {
        begin[rows - 1] = end[rows - 1] = rows;
    }
    CHECK(end[rows - 1] == (int32_t)rows);
    ds4_gpu_tensor *dq = ds4_gpu_tensor_alloc(3 * n * sizeof(float));
    ds4_gpu_tensor *dout = ds4_gpu_tensor_alloc(n * sizeof(float));
    ds4_gpu_tensor *db = ds4_gpu_tensor_alloc(rows * sizeof(int32_t));
    ds4_gpu_tensor *de = ds4_gpu_tensor_alloc(rows * sizeof(int32_t));
    CHECK(dq && dout && db && de);
    CHECK(ds4_gpu_tensor_write(dq, 0, qkv, 3 * n * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(db, 0, begin, rows * sizeof(int32_t)));
    CHECK(ds4_gpu_tensor_write(de, 0, end, rows * sizeof(int32_t)));
    CHECK(setenv("DS4_QWEN_VISION_LEGACY", "1", 1) == 0);
    double start = now_ms();
    CHECK(ds4_gpu_qwen4exp_vision_attention_tensor(dout, dq, db, de, rows, heads, dim));
    CHECK(ds4_gpu_synchronize());
    const double old_ms = now_ms() - start;
    CHECK(ds4_gpu_tensor_read(dout, 0, old, n * sizeof(float)));
    CHECK(unsetenv("DS4_QWEN_VISION_LEGACY") == 0);
    CHECK(setenv("DS4_QWEN_VISION_QUAD", "0", 1) == 0);
    start = now_ms();
    CHECK(ds4_gpu_qwen4exp_vision_attention_tensor(dout, dq, db, de, rows, heads, dim));
    CHECK(ds4_gpu_synchronize());
    const double pair_ms = now_ms() - start;
    CHECK(ds4_gpu_tensor_read(dout, 0, got, n * sizeof(float)));
    CHECK(memcmp(old, got, n * sizeof(float)) == 0);
    CHECK(setenv("DS4_QWEN_VISION_QUAD", "1", 1) == 0);
    start = now_ms();
    CHECK(ds4_gpu_qwen4exp_vision_attention_tensor(dout, dq, db, de, rows, heads, dim));
    CHECK(ds4_gpu_synchronize());
    const double quad_ms = now_ms() - start;
    CHECK(ds4_gpu_tensor_read(dout, 0, got, n * sizeof(float)));
    CHECK(memcmp(old, got, n * sizeof(float)) == 0);
    CHECK(unsetenv("DS4_QWEN_VISION_QUAD") == 0);
    CHECK(ds4_gpu_qwen4exp_vision_attention_tensor(dout, dq, db, de, rows, heads, dim));
    CHECK(ds4_gpu_tensor_read(dout, 0, got, n * sizeof(float)));
    double max_old = 0, max_ref = 0, err = 0, ref = 0;
    for (size_t i = 0; i < n; i++) {
        CHECK(isfinite(got[i]));
        max_old = fmax(max_old, fabs((double)got[i] - old[i]));
    }
    CHECK(memcmp(old, got, n * sizeof(float)) == 0);
    uint32_t samples[16] = {0, 1, 3, rows / 2, rows - 2, rows - 1};
    size_t sample_count = 6;
    if (layout == IMAGE_SEGMENTS) {
        base = 0;
        for (size_t i = 0; i < sizeof(image_rows) / sizeof(image_rows[0]); i++) {
            samples[sample_count++] = base;
            samples[sample_count++] = base + image_rows[i] - 1u;
            base += image_rows[i];
        }
    } else if (segment < rows) {
        samples[sample_count++] = segment - 1u;
        samples[sample_count++] = segment;
    }
    for (size_t si = 0; si < sample_count; si++) {
        const uint32_t r = samples[si];
        for (uint32_t h = 0; h < heads; h++) {
            double max = -INFINITY, sum = 0;
            for (int32_t k = begin[r]; k < end[r]; k++) {
                double dot = 0;
                for (uint32_t d = 0; d < dim; d++) {
                    dot += (double)qkv[r * 3 * width + h * dim + d] *
                        qkv[k * 3 * width + width + h * dim + d];
                }
                scores[k] = dot / sqrt((double)dim);
                max = fmax(max, scores[k]);
            }
            for (int32_t k = begin[r]; k < end[r]; k++) {
                scores[k] = exp(scores[k] - max);
                sum += scores[k];
            }
            for (uint32_t d = 0; d < dim; d++) {
                double value = 0;
                for (int32_t k = begin[r]; k < end[r]; k++) {
                    value += scores[k] * qkv[k * 3 * width + 2 * width + h * dim + d];
                }
                value = sum > 0 ? value / sum : 0;
                const double delta = got[r * width + h * dim + d] - value;
                max_ref = fmax(max_ref, fabs(delta));
                err += delta * delta; ref += value * value;
            }
        }
    }
    CHECK(max_ref < 5e-5 && sqrt(err / fmax(ref, 1e-30)) < 5e-5);
    CHECK(ds4_gpu_qwen4exp_vision_attention_tensor(dout, dq, db, de, rows, heads, dim));
    CHECK(ds4_gpu_tensor_read(dout, 0, got, n * sizeof(float)));
    CHECK(memcmp(old, got, n * sizeof(float)) == 0);
    if (layout == IMAGE_SEGMENTS) {
        /* Another image may change without affecting this image's attention. */
        const size_t first = image_rows[0] * width;
        const size_t last = (image_rows[0] + image_rows[1]) * width;
        for (size_t i = 3 * first; i < 3 * last; i++) {
            qkv[i] = -2.0f * qkv[i];
        }
        CHECK(ds4_gpu_tensor_write(dq, 0, qkv, 3 * n * sizeof(float)));
        CHECK(ds4_gpu_qwen4exp_vision_attention_tensor(dout, dq, db, de, rows, heads, dim));
        CHECK(ds4_gpu_tensor_read(dout, 0, got, n * sizeof(float)));
        CHECK(memcmp(old, got, first * sizeof(float)) == 0);
        CHECK(memcmp(old + last, got + last, (n - last) * sizeof(float)) == 0);
        CHECK(memcmp(old + first, got + first, (last - first) * sizeof(float)) != 0);
    }
    printf("PASS rows=%u heads=%u dim=%u segment=%u scale=%.1f legacy=%.3f ms pair=%.3f ms quad=%.3f ms max_old=%.3g max_f64=%.3g rel_rms=%.3g\n",
        rows, heads, dim, segment, scale, old_ms, pair_ms, quad_ms, max_old, max_ref,
        sqrt(err / fmax(ref, 1e-30)));
    ds4_gpu_tensor_free(de); ds4_gpu_tensor_free(db);
    ds4_gpu_tensor_free(dout); ds4_gpu_tensor_free(dq);
    free(scores); free(end); free(begin); free(old); free(got); free(qkv);
}

int main(void) {
    CHECK(unsetenv("DS4_CUDA_NO_QWEN_VISION_TILE") == 0);
    CHECK(ds4_gpu_init());
    check_case(67, 16, 72, 36, 1.0f, UNIFORM_SEGMENTS);
    check_case(256, 16, 72, 256, 1.0f, UNIFORM_SEGMENTS);
    check_case(259, 16, 72, 37, 4.0f, UNIFORM_SEGMENTS);
    check_case(511, 16, 72, 511, 1.0f, UNIFORM_SEGMENTS);
    check_case(512, 16, 72, 512, 1.0f, UNIFORM_SEGMENTS);
    check_case(513, 16, 72, 513, 1.0f, UNIFORM_SEGMENTS);
    check_case(513, 16, 72, 31, 4.0f, UNIFORM_SEGMENTS);
    check_case(1031, 16, 72, 259, 4.0f, UNIFORM_SEGMENTS);
    check_case(3072, 16, 72, 3072, 1.0f, UNIFORM_SEGMENTS);
    check_case(8193, 16, 72, 2051, 4.0f, UNIFORM_SEGMENTS);
    uint32_t total = 0;
    for (size_t i = 0; i < sizeof(image_rows) / sizeof(image_rows[0]); i++) {
        total += image_rows[i];
    }
    check_case(total, 16, 72, 0, 1.0f, IMAGE_SEGMENTS);
    check_case(257, 1, 64, 129, 1.0f, UNIFORM_SEGMENTS);
    check_case(257, 1, 96, 129, 1.0f, UNIFORM_SEGMENTS);
    ds4_gpu_cleanup();
    puts("ALL PASS");
    return 0;
}
