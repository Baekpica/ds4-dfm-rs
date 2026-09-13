/* Run each scope in a fresh process: dispatch controls are cached at init. */
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { K = 4096, M = 1024, QK = 32, Q8_BYTES = 34, OFFSET = 4096,
       WEIGHT_BYTES = K / QK * M * Q8_BYTES,
       MAP_BYTES = OFFSET + WEIGHT_BYTES };
typedef struct { uint16_t d; int8_t q[QK]; } block_q8;
#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); return 1; \
} } while (0)

int main(int argc, char **argv) {
    CHECK(argc == 2 && (!strcmp(argv[1], "0") || !strcmp(argv[1], "1")));
    const int expected = argv[1][0] - '0';
    CHECK(sizeof(block_q8) == Q8_BYTES && ds4_gpu_init());
    void *map = NULL;
    CHECK(posix_memalign(&map, OFFSET, MAP_BYTES) == 0);
    memset(map, 0, MAP_BYTES);
    block_q8 *w = (block_q8 *)((char *)map + OFFSET);
    for (unsigned b = 0; b < K / QK * M; b++) {
        w[b].d = 0x2800; /* Exactly 1/32. */
        for (unsigned j = 0; j < QK; j++) { w[b].q[j] = (int)((b * 3 + j) % 15) - 7; }
    }
    const char name[] = "blk.0.attn_q.weight";
    ds4_gpu_tensor_record record = {
        .name = name, .name_len = sizeof(name) - 1, .type = 8, .ndim = 2,
        .dims = {K, M}, .offset = OFFSET, .bytes = WEIGHT_BYTES,
    };
    CHECK(ds4_gpu_build_derived_artifacts_from_records(map, MAP_BYTES, &record, 1) == expected);
    int source = -1; uint64_t count = UINT64_MAX, bytes = UINT64_MAX;
    ds4_gpu_derived_artifact_stats(&source, &count, &bytes, NULL);
    CHECK(count == (unsigned)expected && source == expected * 2);
    CHECK(expected ? bytes > 0 : bytes == 0);
    CHECK(!ds4_gpu_model_map_replacements_complete(map)); /* Q8 is additive. */
    /* The dummy manifest tests producer selection, not an actual import.
     * Register local raw storage for the projection in every case. */
    CHECK(unsetenv("DS4_CUDA_WEIGHT_IPC_MANIFEST") == 0);
    CHECK(ds4_gpu_set_model_map(map, MAP_BYTES));

    /* Integer products scaled by 1/1024 make the CPU and GPU sums exact;
     * every input block has max=127/32, so Q8_1 quantization is exact too. */
    float x[K], got[M], want[M];
    for (unsigned k = 0; k < K; k++) {
        x[k] = (k % QK == 0 ? 127 : (int)(k % 11) - 5) / 32.0f;
    }
    for (unsigned m = 0; m < M; m++) {
        want[m] = 0;
        for (unsigned k = 0; k < K; k++) {
            want[m] += w[(m * K + k) / QK].q[k % QK] * x[k] / 32.0f;
        }
    }
    ds4_gpu_tensor *dx = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *dy = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(dx && dy && ds4_gpu_tensor_write(dx, 0, x, sizeof(x)));
    CHECK(ds4_gpu_matmul_q8_0_tensor(dy, map, MAP_BYTES, OFFSET, K, M, dx, 1));
    CHECK(ds4_gpu_tensor_read(dy, 0, got, sizeof(got)));
    CHECK(memcmp(got, want, sizeof(got)) == 0);
    ds4_gpu_tensor_free(dx); ds4_gpu_tensor_free(dy);
    ds4_gpu_cleanup(); free(map);
    puts("PASS artifact scope and complete Q8 output");
    return 0;
}
