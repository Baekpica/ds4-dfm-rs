/* Full QKV parity across bias/RoPE fusion, including V and FP32 rounding. */
#include "ds4_gpu.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", \
    __FILE__, __LINE__, #x); exit(1); } } while (0)

enum input_kind { RANDOM_INPUT, ROUNDING_INPUT };
enum { BIAS_PREFIX = 16, GUARD_VALUES = 16, PATCH_WIDTH = 64 };
static const char *fuse_control = "DS4_QWEN_VISION_FUSE_ROPE";
static const float rounding[][2] = {
    {0x1p0f, 0x1p-24f}, {0x1.000002p0f, 0x1p-24f},
    {0x1p24f, 1.0f}, {0x1.000002p24f, 1.0f},
    {-0x1p0f, -0x1p-24f}, {-0x1.000002p0f, -0x1p-24f},
    {FLT_MIN, -0x1.000002p-126f}, {-FLT_MIN, 0x1.000002p-126f},
    {0x1p-149f, FLT_MIN}, {-0x1p-149f, -FLT_MIN},
    {FLT_MIN, 0x1p-149f}, {-FLT_MIN, -0x1p-149f},
    {0.0f, -0.0f}, {-0.0f, -0.0f},
    {0x1.fffffep-1f, 0x1p-25f}, {-0x1.fffffep-1f, -0x1p-25f},
};

static float random_value(uint32_t *seed) {
    *seed = 1664525u * *seed + 1013904223u;
    return (float)(*seed >> 8) / 8388608.0f - 1.0f;
}

static void check_case(uint32_t rows, uint32_t heads, uint32_t dim,
                       float frequency, enum input_kind kind) {
    const uint32_t width = 3u * heads * dim;
    const size_t n = (size_t)rows * width, bytes = (n + GUARD_VALUES) * sizeof(float);
    const size_t map_bytes = (BIAS_PREFIX + width) * sizeof(float);
    const uint64_t offset = BIAS_PREFIX * sizeof(float);
    float *weights = calloc(BIAS_PREFIX + width, sizeof(float));
    float *input = malloc(bytes), *first = malloc(bytes), *got = malloc(bytes);
    int32_t *height = malloc(rows * sizeof(int32_t));
    int32_t *position = malloc(rows * sizeof(int32_t));
    CHECK(weights && input && first && got && height && position);
    uint32_t seed = 42;
    for (uint32_t d = 0; d < width; d++) {
        weights[BIAS_PREFIX + d] = kind == ROUNDING_INPUT
            ? rounding[d % (sizeof(rounding) / sizeof(rounding[0]))][1]
            : random_value(&seed);
    }
    for (size_t i = 0; i < n; i++) {
        input[i] = kind == ROUNDING_INPUT
            ? rounding[(i % width) % (sizeof(rounding) / sizeof(rounding[0]))][0]
            : random_value(&seed) * 4.0f;
    }
    for (size_t i = n; i < n + GUARD_VALUES; i++) { input[i] = 12345.0f; }
    /* Match image preprocessing's merge-2 order: (h,w),(h,w+1),
     * (h+1,w),(h+1,w+1). Ragged final groups exercise the tensor seam. */
    for (uint32_t r = 0; r < rows; r++) {
        const uint32_t group = r / 4u;
        height[r] = (int32_t)(group / (PATCH_WIDTH / 2u) * 2u + r % 4u / 2u);
        position[r] = (int32_t)(group % (PATCH_WIDTH / 2u) * 2u + r % 2u);
    }
    CHECK(ds4_gpu_set_model_map(weights, map_bytes));
    ds4_gpu_tensor *qkv = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *dh = ds4_gpu_tensor_alloc(rows * sizeof(int32_t));
    ds4_gpu_tensor *dw = ds4_gpu_tensor_alloc(rows * sizeof(int32_t));
    CHECK(qkv && dh && dw);
    CHECK(ds4_gpu_tensor_write(dh, 0, height, rows * sizeof(int32_t)));
    CHECK(ds4_gpu_tensor_write(dw, 0, position, rows * sizeof(int32_t)));
    const char *controls[] = {"0", "0", "1", "1", NULL};
    for (size_t pass = 0; pass < sizeof(controls) / sizeof(controls[0]); pass++) {
        if (controls[pass]) {
            CHECK(setenv(fuse_control, controls[pass], 1) == 0);
        } else {
            CHECK(unsetenv(fuse_control) == 0);
        }
        CHECK(ds4_gpu_tensor_write(qkv, 0, input, bytes));
        CHECK(ds4_gpu_qwen4exp_vision_qkv_rope_tensor(qkv, weights, map_bytes,
            offset, dh, dw, rows, heads, dim, frequency));
        CHECK(ds4_gpu_tensor_read(qkv, 0, got, bytes));
        for (size_t i = 0; i < n; i++) { CHECK(isfinite(got[i])); }
        CHECK(memcmp(got + n, input + n, GUARD_VALUES * sizeof(float)) == 0);
        if (pass == 0) { memcpy(first, got, bytes); }
        else { CHECK(memcmp(first, got, bytes) == 0); }
    }
    CHECK(memcmp(first, input, n * sizeof(float)) != 0);
    if (kind == RANDOM_INPUT) {
        for (uint32_t r = 0; r < rows; r++) {
            for (uint32_t d = 2u * heads * dim; d < width; d++) {
                const size_t i = (size_t)r * width + d;
                const float expected = input[i] + weights[BIAS_PREFIX + d];
                CHECK(memcmp(first + i, &expected, sizeof(float)) == 0);
            }
        }
    }
    /* A second in-place application must also consume only its own inputs. */
    CHECK(setenv(fuse_control, "0", 1) == 0);
    CHECK(ds4_gpu_tensor_write(qkv, 0, first, bytes));
    CHECK(ds4_gpu_qwen4exp_vision_qkv_rope_tensor(qkv, weights, map_bytes,
        offset, dh, dw, rows, heads, dim, frequency));
    CHECK(ds4_gpu_tensor_read(qkv, 0, input, bytes));
    CHECK(setenv(fuse_control, "1", 1) == 0);
    CHECK(ds4_gpu_tensor_write(qkv, 0, first, bytes));
    CHECK(ds4_gpu_qwen4exp_vision_qkv_rope_tensor(qkv, weights, map_bytes,
        offset, dh, dw, rows, heads, dim, frequency));
    CHECK(ds4_gpu_tensor_read(qkv, 0, got, bytes));
    CHECK(memcmp(input, got, bytes) == 0);
    if (rows == 512u && heads == 16u && dim == 72u && kind == RANDOM_INPUT) {
        /* Reject malformed weight/tensor ranges before either launch. */
        ds4_gpu_tensor *short_pos = ds4_gpu_tensor_alloc((rows - 1u) * sizeof(int32_t));
        CHECK(short_pos);
        CHECK(!ds4_gpu_qwen4exp_vision_qkv_rope_tensor(qkv, weights, map_bytes - 1u,
            offset, dh, dw, rows, heads, dim, frequency));
        CHECK(!ds4_gpu_qwen4exp_vision_qkv_rope_tensor(qkv, weights, map_bytes,
            UINT64_MAX, dh, dw, rows, heads, dim, frequency));
        CHECK(!ds4_gpu_qwen4exp_vision_qkv_rope_tensor(qkv, NULL, map_bytes,
            offset, dh, dw, rows, heads, dim, frequency));
        CHECK(!ds4_gpu_qwen4exp_vision_qkv_rope_tensor(qkv, weights, map_bytes,
            offset, short_pos, dw, rows, heads, dim, frequency));
        CHECK(!ds4_gpu_qwen4exp_vision_qkv_rope_tensor(qkv, weights, map_bytes,
            offset, dh, dw, rows, heads, dim, NAN));
        CHECK(!ds4_gpu_qwen4exp_vision_qkv_rope_tensor(qkv, weights, map_bytes,
            offset, dh, dw, rows, heads, dim, 0.0f));
        ds4_gpu_tensor_free(short_pos);
    }
    CHECK(unsetenv(fuse_control) == 0);
    printf("PASS rows=%u heads=%u dim=%u freq=%g input=%s bytes=%zu exact/repeat/default\n",
        rows, heads, dim, frequency, kind == ROUNDING_INPUT ? "rounding" : "random", bytes);
    ds4_gpu_tensor_free(dw); ds4_gpu_tensor_free(dh); ds4_gpu_tensor_free(qkv);
    ds4_gpu_unregister_model_map(weights);
    free(position); free(height); free(got); free(first); free(input); free(weights);
}

int main(void) {
    CHECK(unsetenv(fuse_control) == 0);
    CHECK(ds4_gpu_init());
    check_case(1, 16, 72, 10000.0f, RANDOM_INPUT);
    check_case(511, 16, 72, 10000.0f, RANDOM_INPUT);
    check_case(512, 16, 72, 10000.0f, RANDOM_INPUT);
    check_case(513, 16, 72, 10000.0f, RANDOM_INPUT);
    check_case(543, 16, 72, 10000.0f, RANDOM_INPUT);
    check_case(544, 16, 72, 10000.0f, RANDOM_INPUT);
    check_case(545, 16, 72, 10000.0f, RANDOM_INPUT);
    check_case(3072, 16, 72, 10000.0f, RANDOM_INPUT);
    check_case(512, 1, 72, 10000.0f, RANDOM_INPUT);
    check_case(577, 3, 72, 1000.0f, RANDOM_INPUT);
    check_case(513, 16, 72, 500000.0f, RANDOM_INPUT);
    check_case(512, 16, 72, 0.5f, RANDOM_INPUT);
    check_case(513, 1, 64, 10000.0f, RANDOM_INPUT);
    check_case(513, 1, 96, 10000.0f, RANDOM_INPUT);
    check_case(512, 1, 72, 10000.0f, ROUNDING_INPUT);
    check_case(513, 16, 72, 10000.0f, ROUNDING_INPUT);
    ds4_gpu_cleanup();
    puts("VISION ROPE PASS");
    return 0;
}
