/* Regression for mean/variance shared-buffer reuse across eight warps. */
#include "ds4_gpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", \
    __FILE__, __LINE__, #x); exit(1); } } while (0)

static void check_norm(const float *weights, unsigned dim) {
    const unsigned rows = 67, max_dim = 4608;
    const size_t n = rows * dim, bytes = n * sizeof(float);
    float *input = malloc(bytes), *got = malloc(bytes), *first = malloc(bytes);
    double *reference = malloc(n * sizeof(double));
    CHECK(input && got && first && reference);
    for (size_t i = 0; i < n; i++) { input[i] = (i % 97) * 0.13f + 1.0f; }
    for (unsigned r = 0; r < rows; r++) {
        double mean = 0, var = 0;
        for (unsigned d = 0; d < dim; d++) { mean += input[r * dim + d]; }
        mean /= dim;
        for (unsigned d = 0; d < dim; d++) {
            const double x = input[r * dim + d] - mean;
            var += x * x;
        }
        const double inv = 1.0 / sqrt(var / dim + 1e-6);
        for (unsigned d = 0; d < dim; d++) {
            reference[r * dim + d] = (input[r * dim + d] - mean) * inv;
        }
    }
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(bytes), *y = ds4_gpu_tensor_alloc(bytes);
    CHECK(x && y && ds4_gpu_tensor_write(x, 0, input, bytes));
    double max_error = 0;
    for (int pass = 0; pass < 64; pass++) {
        CHECK(ds4_gpu_qwen4exp_vision_layernorm_tensor(y, x, weights,
            2 * max_dim * sizeof(float), 0, max_dim * sizeof(float), rows, dim, 1e-6f));
        CHECK(ds4_gpu_tensor_read(y, 0, got, bytes));
        for (size_t i = 0; i < n; i++) {
            CHECK(isfinite(got[i]));
            max_error = fmax(max_error, fabs(got[i] - reference[i]));
        }
        if (pass == 0) { memcpy(first, got, bytes); }
        else { CHECK(memcmp(first, got, bytes) == 0); }
    }
    printf("vision LayerNorm dim=%u repeats=64 max_f64=%.9g\n", dim, max_error);
    CHECK(max_error < 2e-5);
    ds4_gpu_tensor_free(y); ds4_gpu_tensor_free(x);
    free(reference); free(first); free(got); free(input);
}

int main(void) {
    const unsigned dim = 4608;
    float *weights = calloc(2 * dim, sizeof(float));
    CHECK(weights);
    for (unsigned i = 0; i < dim; i++) { weights[i] = 1.0f; }
    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_set_model_map(weights, 2 * dim * sizeof(float)));
    check_norm(weights, 1152);
    check_norm(weights, 4608);
    ds4_gpu_unregister_model_map(weights);
    ds4_gpu_cleanup(); free(weights);
    puts("VISION NORM PASS");
    return 0;
}
