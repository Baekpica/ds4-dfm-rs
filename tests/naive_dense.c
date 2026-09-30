/* MQ87 Q and attention-output projections are Q6_K at every layer. */
#include "../ds4.c"
#include <assert.h>

int main(void) {
    enum { K = N05_EMBED, M = 256, ROWS = 32, BLOCKS = K / QK_K };
    const uint64_t bytes = (uint64_t)M * BLOCKS * sizeof(block_q6_K);
    block_q6_K *weights = xmalloc(bytes);
    for (unsigned i = 0; i < M * BLOCKS; i++) {
        block_q6_K *b = &weights[i];
        b->d = f32_to_f16(1.0f / 64);
        for (unsigned j = 0; j < sizeof(b->ql); j++) { b->ql[j] = (uint8_t)(i + j * 7); }
        for (unsigned j = 0; j < sizeof(b->qh); j++) { b->qh[j] = (uint8_t)(i * 3 + j); }
        for (unsigned j = 0; j < sizeof(b->scales); j++) { b->scales[j] = (int8_t)(1 + j % 4); }
    }
    float *input = xmalloc(ROWS * K * sizeof(float)), output[ROWS * M];
    for (unsigned r = 0; r < ROWS; r++) {
        for (unsigned k = 0; k < K; k++) { input[r * K + k] = (k + r) % 3 ? 1.0f : -1.0f; }
    }
    ds4_model model = {.map = (uint8_t *)weights, .size = bytes};
    ds4_tensor weight = {.type = DS4_TENSOR_Q6_K, .bytes = bytes};
    assert(ds4_gpu_init() && ds4_gpu_set_model_map(weights, bytes));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(ROWS * K * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(ROWS * M * sizeof(float));
    ds4_gpu_tensor *ids = ds4_gpu_tensor_alloc(ROWS * sizeof(int));
    assert(x && out && ids && ds4_gpu_tensor_write(x, 0, input, ROWS * K * sizeof(float)));
    const unsigned rows[] = {1, 7, ROWS};
    for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); i++) {
        const unsigned n = rows[i];
        assert(naive_dense(out, &model, &weight, K, M, x, ids, n));
        assert(ds4_gpu_tensor_read(out, 0, output, n * M * sizeof(float)));
        double max_error = 0;
        for (unsigned r = 0; r < n; r++) {
            for (unsigned m = 0; m < M; m++) {
                const float want = ds4_vec_dot_q6_K_f32(K, weights + m * BLOCKS, input + r * K);
                const float error = fabsf(output[r * M + m] - want);
                assert(isfinite(output[r * M + m]) && error <= .005f * (fabsf(want) + 1));
                if (error > max_error) { max_error = error; }
            }
        }
        printf("Naive Q6_K rows=%u max_abs=%.9g\n", n, max_error);
    }
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(ids);
    free(weights); free(input);
    return 0;
}
