/* Solar Open 2 CUDA output-gate parity tests.
 *
 * Covers the production 64 x 128 layout, several tokens, GQA's plain
 * sigmoid gate, and KDA's per-head RMSNorm+sigmoid gate. Both operations are
 * checked out-of-place and in-place because the runtime reuses attention
 * scratch in place before the output projection.
 */
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    T_TOKENS = 7,
    T_HEAD = 64,
    T_DIM = 128,
    T_COUNT = T_TOKENS * T_HEAD * T_DIM,
};

static int failures;

#define CHECK(c, m) do {                                                      \
    if (!(c)) {                                                               \
        fprintf(stderr, "FAIL: %s\n", (m));                                 \
        exit(1);                                                              \
    }                                                                         \
} while (0)

static float sigmoid_ref(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static void compare(const char *label, const float *got, const float *want,
                    size_t n, double atol, double rtol) {
    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < n; i++) {
        const double delta = fabs((double)got[i] - want[i]);
        const double scale = fmax(fabs((double)got[i]), fabs((double)want[i]));
        if (delta > max_abs) {
            max_abs = delta;
            worst = i;
        }
        if (scale > 1.0e-12 && delta / scale > max_rel) {
            max_rel = delta / scale;
        }
    }
    const int ok = max_abs <= atol || max_rel <= rtol;
    if (!ok) failures++;
    printf("%-34s abs=%.3e rel=%.3e worst=%zu %s\n",
           label, max_abs, max_rel, worst, ok ? "ok" : "FAIL");
}

static void make_fixture(float *x, float *gate, float *weight) {
    for (uint32_t d = 0; d < T_DIM; d++) {
        weight[d] = 0.75f + 0.003f * (float)d +
                    0.04f * sinf(0.19f * (float)d);
    }
    for (uint32_t i = 0; i < T_COUNT; i++) {
        const float fi = (float)(i + 1u);
        x[i] = 0.9f * sinf(0.013f * fi) +
               0.35f * cosf(0.021f * fi);
        gate[i] = 1.7f * sinf(0.017f * fi) -
                  0.6f * cosf(0.011f * fi);
    }
    gate[0] = -40.0f;
    gate[1] = 40.0f;
    gate[T_DIM] = -18.0f;
    gate[T_DIM + 1u] = 18.0f;
}

static void reference_sigmoid(float *out, const float *x, const float *gate) {
    for (uint32_t i = 0; i < T_COUNT; i++) {
        out[i] = x[i] * sigmoid_ref(gate[i]);
    }
}

static void reference_head_rms(float *out, const float *x, const float *gate,
                               const float *weight, float eps) {
    const uint32_t rows = T_TOKENS * T_HEAD;
    for (uint32_t row = 0; row < rows; row++) {
        const size_t base = (size_t)row * T_DIM;
        float sumsq = 0.0f;
        for (uint32_t d = 0; d < T_DIM; d++) {
            sumsq += x[base + d] * x[base + d];
        }
        const float scale = 1.0f / sqrtf(sumsq / (float)T_DIM + eps);
        for (uint32_t d = 0; d < T_DIM; d++) {
            out[base + d] = x[base + d] * scale * weight[d] *
                            sigmoid_ref(gate[base + d]);
        }
    }
}

static void test_moe_residual(uint32_t rows, uint32_t hidden) {
    const uint32_t used = 8u;
    const uint32_t count = rows * hidden;
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    float *down = malloc(bytes * used);
    float *x = malloc(bytes), *shared = malloc(bytes);
    float *want = malloc(bytes), *got = malloc(bytes);
    CHECK(down && x && shared && want && got, "MoE host allocation");
    for (uint64_t i = 0; i < (uint64_t)count * used; i++) {
        uint32_t bits = (uint32_t)i * 1664525u + 1013904223u;
        down[i] = ((float)(bits & 65535u) - 32768.0f) / 4096.0f;
    }
    for (uint32_t i = 0; i < count; i++) {
        x[i] = (float)(i % 127u) / 32.0f - 2.0f;
        shared[i] = (float)(i % 79u) / 64.0f - 0.5f;
    }
    /* Guard nonfinite routed values, and pin the left-to-right slot sum
     * with cancellation too large for an alternate association to hide. */
    down[0] = NAN;
    down[hidden] = INFINITY;
    down[2u * hidden] = -INFINITY;
    down[1] = 1.0e20f;
    down[hidden + 1u] = -1.0e20f;
    down[2u * hidden + 1u] = 3.25f;

    ds4_gpu_tensor *dx = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *dd = ds4_gpu_tensor_alloc(bytes * used);
    ds4_gpu_tensor *ds = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *tmp = ds4_gpu_tensor_alloc(bytes);
    CHECK(dx && dd && ds && tmp, "MoE device allocation");
    CHECK(ds4_gpu_tensor_write(dx, 0, x, bytes), "write residual");
    CHECK(ds4_gpu_tensor_write(dd, 0, down, bytes * used), "write experts");
    CHECK(ds4_gpu_tensor_write(ds, 0, shared, bytes), "write shared");
    CHECK(ds4_gpu_moe_sum_tensor(tmp, dd, hidden, used, rows), "reference sum");
    CHECK(ds4_gpu_add_tensor(tmp, tmp, ds, count), "reference shared add");
    CHECK(ds4_gpu_add_tensor(tmp, dx, tmp, count), "reference residual add");
    CHECK(ds4_gpu_tensor_read(tmp, 0, want, bytes), "reference read");
    CHECK(ds4_gpu_solar_moe_residual_tensor(dx, dd, ds, hidden, used, rows) == 1,
          "fused MoE residual");
    CHECK(ds4_gpu_tensor_read(dx, 0, got, bytes), "fused read");
    CHECK(memcmp(got, want, bytes) == 0, "MoE residual byte parity");
    printf("MoE residual %u x %u x %u: byte-identical\n", rows, used, hidden);

    CHECK(ds4_gpu_solar_moe_residual_tensor(NULL, dd, ds, hidden, used, rows) == 0,
          "reject null residual");
    CHECK(ds4_gpu_solar_moe_residual_tensor(dx, NULL, ds, hidden, used, rows) == 0,
          "reject null experts");
    CHECK(ds4_gpu_solar_moe_residual_tensor(dx, dd, NULL, hidden, used, rows) == 0,
          "reject null shared");
    CHECK(ds4_gpu_solar_moe_residual_tensor(dx, dd, ds, hidden, used, 0) == 0,
          "reject empty rows");
    CHECK(ds4_gpu_solar_moe_residual_tensor(dx, dx, ds, hidden, used, rows) == 0,
          "reject short expert buffer");
    ds4_gpu_tensor *overlap = ds4_gpu_tensor_view(dd, sizeof(float), bytes);
    CHECK(overlap, "overlapping view");
    CHECK(ds4_gpu_solar_moe_residual_tensor(overlap, dd, ds, hidden, used, rows) == 0,
          "reject overlapping expert view");
    ds4_gpu_tensor_free(overlap);
    CHECK(ds4_gpu_solar_moe_residual_tensor(dx, dd, dx, hidden, used, rows) == 0,
          "reject shared residual alias");
    ds4_gpu_tensor_free(dx); ds4_gpu_tensor_free(dd);
    ds4_gpu_tensor_free(ds); ds4_gpu_tensor_free(tmp);
    free(down); free(x); free(shared); free(want); free(got);
}

int main(void) {
    CHECK(ds4_gpu_init(), "CUDA init");
    test_moe_residual(1u, 4096u);
    test_moe_residual(7u, 259u);
    test_moe_residual(257u, 4096u);
    test_moe_residual(4096u, 4096u);
    const uint64_t bytes = (uint64_t)T_COUNT * sizeof(float);
    float *x = malloc(bytes);
    float *gate = malloc(bytes);
    float *want = malloc(bytes);
    float *got = malloc(bytes);
    float weight[T_DIM];
    CHECK(x && gate && want && got, "host arrays");
    make_fixture(x, gate, weight);

    ds4_gpu_tensor *dx = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *dg = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *dout = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *dw = ds4_gpu_tensor_alloc(sizeof(weight));
    CHECK(dx && dg && dout && dw, "device arrays");
    CHECK(ds4_gpu_tensor_write(dg, 0, gate, bytes), "write gates");
    CHECK(ds4_gpu_tensor_write(dw, 0, weight, sizeof(weight)), "write norm weight");

    puts("== Solar Open 2 CUDA output gates ==");
    reference_sigmoid(want, x, gate);
    CHECK(ds4_gpu_tensor_write(dx, 0, x, bytes), "write GQA input");
    CHECK(ds4_gpu_solar_sigmoid_gate_tensor(dout, dx, dg, T_COUNT),
          "GQA gate out-of-place");
    CHECK(ds4_gpu_tensor_read(dout, 0, got, bytes), "read GQA output");
    compare("GQA sigmoid gate out-of-place", got, want, T_COUNT,
            3.0e-6, 3.0e-6);

    CHECK(ds4_gpu_tensor_write(dx, 0, x, bytes), "rewrite GQA input");
    CHECK(ds4_gpu_solar_sigmoid_gate_tensor(dx, dx, dg, T_COUNT),
          "GQA gate in-place");
    CHECK(ds4_gpu_tensor_read(dx, 0, got, bytes), "read in-place GQA output");
    compare("GQA sigmoid gate in-place", got, want, T_COUNT,
            3.0e-6, 3.0e-6);

    const float eps = 1.0e-5f;
    reference_head_rms(want, x, gate, weight, eps);
    CHECK(ds4_gpu_tensor_write(dx, 0, x, bytes), "write KDA input");
    CHECK(ds4_gpu_solar_head_rms_sigmoid_gate_tensor(
              dout, dx, dg, dw, T_TOKENS, T_HEAD, T_DIM, eps),
          "KDA gated RMS out-of-place");
    CHECK(ds4_gpu_tensor_read(dout, 0, got, bytes), "read KDA gated RMS");
    compare("KDA head RMS sigmoid out-of-place", got, want, T_COUNT,
            2.0e-5, 3.0e-5);

    CHECK(ds4_gpu_tensor_write(dx, 0, x, bytes), "rewrite KDA input");
    CHECK(ds4_gpu_solar_head_rms_sigmoid_gate_tensor(
              dx, dx, dg, dw, T_TOKENS, T_HEAD, T_DIM, eps),
          "KDA gated RMS in-place");
    CHECK(ds4_gpu_tensor_read(dx, 0, got, bytes), "read in-place KDA gated RMS");
    compare("KDA head RMS sigmoid in-place", got, want, T_COUNT,
            2.0e-5, 3.0e-5);

    ds4_gpu_tensor_free(dx);
    ds4_gpu_tensor_free(dg);
    ds4_gpu_tensor_free(dout);
    ds4_gpu_tensor_free(dw);
    free(x);
    free(gate);
    free(want);
    free(got);
    ds4_gpu_cleanup();
    puts(failures ? "Solar gate checks FAILED" : "all Solar gate checks passed");
    return failures ? 1 : 0;
}
