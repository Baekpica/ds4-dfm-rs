/* Preserve the original 1024-thread norm and per-consumer Q8 arithmetic. */
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { WIDTH = 4096, ROWS = 64, OUTPUT = 128, OFFSET = 4096,
       QOFFSET = OFFSET + WIDTH * sizeof(float),
       MAP_BYTES = QOFFSET + WIDTH / 32 * OUTPUT * 34 };
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "Step norm FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)

int main(void) {
    CHECK(!setenv("DS4_CUDA_PREFILL_PATH", "mmq", 1));
    CHECK(ds4_gpu_init());
    void *map = NULL;
    CHECK(!posix_memalign(&map, OFFSET, MAP_BYTES));
    memset(map, 0, MAP_BYTES);
    float *weight = (float *)((char *)map + OFFSET);
    for (unsigned i = 0; i < WIDTH; i++) { weight[i] = 0.5f + (i % 31) / 29.0f; }
    for (unsigned o = 0; o < OUTPUT; o++) {
        for (unsigned b = 0; b < WIDTH / 32; b++) {
            uint8_t *block = (uint8_t *)map + QOFFSET + (o * (WIDTH / 32) + b) * 34;
            uint16_t one = 0x3c00;
            memcpy(block, &one, sizeof(one));
            for (unsigned i = 0; i < 32; i++) { block[2 + i] = (uint8_t)((int)((o + b + i) % 15) - 7); }
        }
    }
    CHECK(ds4_gpu_set_model_map(map, MAP_BYTES));
    const size_t bytes = WIDTH * ROWS * sizeof(float), obytes = OUTPUT * ROWS * sizeof(float);
    float *input = malloc(bytes), *a = malloc(bytes), *b = malloc(bytes);
    CHECK(input && a && b);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(bytes), *norm = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(obytes), *refout = ds4_gpu_tensor_alloc(obytes);
    CHECK(x && norm && reference && out && refout);
    const unsigned widths[] = {1, 4, 63, ROWS};
    for (unsigned pass = 0; pass < 2; pass++) {
        for (unsigned i = 0; i < WIDTH * ROWS; i++) {
            input[i] = ((int)((i * 37 + pass * 11) % 127) - 63) / 19.0f;
        }
        CHECK(ds4_gpu_tensor_write(x, 0, input, bytes));
        for (unsigned r = 0; r < sizeof(widths) / sizeof(widths[0]); r++) {
            const unsigned rows = widths[r];
            CHECK(ds4_gpu_step37_norm(norm, x, map, MAP_BYTES, OFFSET, WIDTH, rows, 1e-5f));
            CHECK(ds4_gpu_exaone_rms_norm_tensor(reference, x, map, MAP_BYTES, OFFSET, WIDTH, rows, 1e-5f));
            CHECK(ds4_gpu_tensor_read(norm, 0, a, rows * WIDTH * sizeof(float)));
            CHECK(ds4_gpu_tensor_read(reference, 0, b, rows * WIDTH * sizeof(float)));
            CHECK(!memcmp(a, b, rows * WIDTH * sizeof(float)));
            CHECK(ds4_gpu_matmul_q8_0_tensor(out, map, MAP_BYTES, QOFFSET, WIDTH, OUTPUT, norm, rows));
            CHECK(ds4_gpu_matmul_q8_0_tensor(refout, map, MAP_BYTES, QOFFSET, WIDTH, OUTPUT, reference, rows));
            CHECK(ds4_gpu_tensor_read(out, 0, a, rows * OUTPUT * sizeof(float)));
            CHECK(ds4_gpu_tensor_read(refout, 0, b, rows * OUTPUT * sizeof(float)));
            CHECK(!memcmp(a, b, rows * OUTPUT * sizeof(float)));
        }
    }
    CHECK(!ds4_gpu_step37_norm(norm, x, map, OFFSET, OFFSET, WIDTH, ROWS, 1e-5f));
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(norm); ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(refout);
    ds4_gpu_cleanup(); free(map); free(input); free(a); free(b);
    puts("Step norm/Q8 reuse: full F32 norm and Q8 consumer outputs exact");
    return 0;
}
