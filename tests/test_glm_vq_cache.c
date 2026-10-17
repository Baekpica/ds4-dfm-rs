#include "ds4_gpu.h"
#include "ds4_glmvq.h"
#include <stdio.h>
#include <string.h>

enum { WIDTH = 512, CAPACITY = 5, ROWS = 3, SENTINEL = 0xa55a };

int main(void) {
    if (!ds4_gpu_init()) { return 1; }
    float source[ROWS * WIDTH], restored[CAPACITY * WIDTH];
    uint16_t before[CAPACITY * WIDTH], after[CAPACITY * WIDTH];
    for (unsigned i = 0; i < ROWS * WIDTH; i++) {
        // Exactly representable BF16, including values outside FP16's range.
        const uint32_t bits = ((i & 1u) ? 0x80000000u : 0u) |
            ((100u + i % 55u) << 23) | ((i % 128u) << 16);
        memcpy(&source[i], &bits, sizeof(bits));
    }
    for (unsigned i = 0; i < CAPACITY * WIDTH; i++) { before[i] = SENTINEL; }
    ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(sizeof(source));
    ds4_gpu_tensor *cache = ds4_gpu_tensor_alloc(sizeof(before));
    ds4_gpu_tensor *output = ds4_gpu_tensor_alloc(sizeof(restored));
    ds4_gpu_tensor *tail = ds4_gpu_tensor_view(input, 2u * WIDTH * sizeof(float), WIDTH * sizeof(float));
    if (!input || !cache || !output || !tail ||
        !ds4_gpu_tensor_write(input, 0u, source, sizeof(source)) ||
        !ds4_gpu_tensor_write(cache, 0u, before, sizeof(before)) ||
        !ds4_gpu_gvq_store(cache, input, 2u, 1u, CAPACITY, WIDTH) ||
        !ds4_gpu_gvq_store(cache, tail, 1u, 3u, CAPACITY, WIDTH) ||
        !ds4_gpu_gvq_unpack(output, cache, CAPACITY * WIDTH) ||
        !ds4_gpu_tensor_read(cache, 0u, after, sizeof(after)) ||
        !ds4_gpu_tensor_read(output, 0u, restored, sizeof(restored))) { return 1; }
    for (unsigned row = 0; row < CAPACITY; row++) {
        for (unsigned col = 0; col < WIDTH; col++) {
            const unsigned index = row * WIDTH + col;
            uint16_t expected = SENTINEL;
            if (row >= 1u && row <= ROWS) {
                uint32_t bits;
                memcpy(&bits, &source[(row - 1u) * WIDTH + col], sizeof(bits));
                expected = (uint16_t)(bits >> 16);
                if (memcmp(&source[(row - 1u) * WIDTH + col], &restored[index], sizeof(float))) {
                    fprintf(stderr, "BF16 latent roundtrip failed at %u: %.9g vs %.9g\n",
                        index, restored[index], source[(row - 1u) * WIDTH + col]);
                    return 1;
                }
            }
            if (after[index] != expected) { return 1; }
        }
    }
    if (ds4_gpu_gvq_store(cache, input, 2u, CAPACITY - 1u, CAPACITY, WIDTH)) { return 1; }
    ds4_gpu_tensor_free(tail); ds4_gpu_tensor_free(output);
    ds4_gpu_tensor_free(cache); ds4_gpu_tensor_free(input);
    ds4_gpu_cleanup();
    puts("GLM VQ BF16 latent append/roundtrip/bounds PASS");
    return 0;
}
