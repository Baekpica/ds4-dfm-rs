#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    N_TOKEN = 2,
    N_HEAD = 2,
    LORA_DIM = 32,
    HEAD_DIM = 32,
    Q8_ROW_BYTES = 34,
};

#define CHECK(expr, what) do {                                                \
    if (!(expr)) { fprintf(stderr, "FAIL: %s\n", what); return 1; }         \
} while (0)

int main(void) {
    const uint64_t weight_bytes =
        (uint64_t)N_HEAD * LORA_DIM * Q8_ROW_BYTES;
    const uint64_t map_size = 4096u;
    unsigned char *map = NULL;
    CHECK(posix_memalign((void **)&map, 4096u, map_size) == 0, "model map");
    memset(map, 0, map_size);

    for (uint32_t h = 0; h < N_HEAD; h++) {
        for (uint32_t j = 0; j < LORA_DIM; j++) {
            unsigned char *row = map +
                ((uint64_t)h * LORA_DIM + j) * Q8_ROW_BYTES;
            const uint16_t half_scale = 0x3000u; /* 0.125 */
            memcpy(row, &half_scale, sizeof(half_scale));
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                ((int8_t *)(row + 2))[d] =
                    (int8_t)((int)((h * 3u + j + d) % 7u) - 3);
            }
        }
    }
    CHECK(weight_bytes <= map_size, "fixture bounds");

    float input[N_TOKEN * LORA_DIM];
    float expected[N_TOKEN * N_HEAD * HEAD_DIM];
    float got[N_TOKEN * N_HEAD * HEAD_DIM];
    for (uint32_t t = 0; t < N_TOKEN; t++) {
        for (uint32_t j = 0; j < LORA_DIM; j++)
            input[t * LORA_DIM + j] =
                0.02f * (float)((int)(t * 5u + j) - 11);
        for (uint32_t h = 0; h < N_HEAD; h++) {
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                float sum = 0.0f;
                for (uint32_t j = 0; j < LORA_DIM; j++) {
                    const int q = (int)((h * 3u + j + d) % 7u) - 3;
                    sum += 0.125f * (float)q * input[t * LORA_DIM + j];
                }
                expected[((t * N_HEAD + h) * HEAD_DIM) + d] = sum;
            }
        }
    }

    CHECK(ds4_gpu_init(), "CUDA init");
    CHECK(ds4_gpu_set_model_map(map, map_size), "register model map");
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(input));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(x && out, "device tensors");
    CHECK(ds4_gpu_tensor_write(x, 0, input, sizeof(input)), "write input");
    CHECK(ds4_gpu_glm53_k_b_project_tensor(
              out, x, map, map_size, 0u, N_TOKEN,
              LORA_DIM, HEAD_DIM, N_HEAD),
          "GLM 5.3 K-b projection");
    CHECK(ds4_gpu_tensor_read(out, 0, got, sizeof(got)), "read output");

    float max_err = 0.0f;
    for (size_t i = 0; i < sizeof(got) / sizeof(got[0]); i++)
        max_err = fmaxf(max_err, fabsf(got[i] - expected[i]));
    CHECK(max_err < 2.0e-5f, "CPU parity");
    printf("GLM 5.3 DSA K-b projection: valid (max_err %.3g)\n", max_err);

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_unregister_model_map(map);
    free(map);
    return 0;
}
