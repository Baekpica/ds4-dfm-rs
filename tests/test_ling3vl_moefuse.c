/* Ling n=1 Q4_K fused silu(gate)*up vs unfused gate/up + SwiGLU. */
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    QK_K = 256,
    OFFSET = 4096,
    K = 256,
    M = 256,
    NEXP = 8,
    USED = 8,
    TYPE_Q4_K = 12u
};

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
} block_q4_k;

_Static_assert(sizeof(block_q4_k) == 144, "Q4_K block");

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static void fill_q4_k(block_q4_k *blocks, uint64_t count) {
    uint32_t state = 0x51ed270bu;
    for (uint64_t b = 0; b < count; b++) {
        blocks[b].d = 0x2000u;
        blocks[b].dmin = 0x1c00u;
        for (uint32_t i = 0; i < 12u; i++) {
            state = state * 1664525u + 1013904223u;
            blocks[b].scales[i] = (uint8_t)(state >> 24u);
        }
        for (uint32_t i = 0; i < QK_K / 2; i++) {
            state = state * 1664525u + 1013904223u;
            blocks[b].qs[i] = (uint8_t)(state >> 24u);
        }
    }
}

int main(void) {
    const uint64_t blocks = (uint64_t)NEXP * M * (K / QK_K);
    const uint64_t bytes = blocks * sizeof(block_q4_k);
    const uint64_t map_bytes = OFFSET + 2u * bytes;
    void *map = NULL;
    CHECK(!posix_memalign(&map, OFFSET, map_bytes));
    memset(map, 0, map_bytes);
    fill_q4_k((block_q4_k *)((char *)map + OFFSET), blocks);
    fill_q4_k((block_q4_k *)((char *)map + OFFSET + bytes), blocks);

    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_set_model_map(map, map_bytes));

    float x[K];
    int32_t ids[USED];
    for (unsigned i = 0; i < K; i++) {
        x[i] = ((int)(i * 13 % 29) - 14) / 7.0f;
    }
    for (unsigned i = 0; i < USED; i++) {
        ids[i] = (int32_t)i;
    }

    ds4_gpu_tensor *dx = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *di = ds4_gpu_tensor_alloc(sizeof(ids));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(USED * M * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(USED * M * sizeof(float));
    ds4_gpu_tensor *mid_u = ds4_gpu_tensor_alloc(USED * M * sizeof(float));
    ds4_gpu_tensor *mid_f = ds4_gpu_tensor_alloc(USED * M * sizeof(float));
    CHECK(dx && di && gate && up && mid_u && mid_f);
    CHECK(ds4_gpu_tensor_write(dx, 0, x, sizeof(x)));
    CHECK(ds4_gpu_tensor_write(di, 0, ids, sizeof(ids)));

    CHECK(ds4_gpu_routed_gate_up_tensor(
        gate, up, dx, di, map, map_bytes, OFFSET, bytes, OFFSET + bytes, bytes,
        TYPE_Q4_K, K, M, NEXP, 1u, USED));
    CHECK(ds4_gpu_step37_swiglu(mid_u, gate, up, NULL, M, USED, 0.0f));
    CHECK(ds4_gpu_routed_silu_mid_tensor(
        mid_f, dx, di, map, map_bytes, OFFSET, bytes, OFFSET + bytes, bytes,
        TYPE_Q4_K, K, M, NEXP, 1u, USED));
    CHECK(!setenv("DS4_LING3VL_NO_MOE_FUSE", "1", 1));
    CHECK(!ds4_gpu_routed_silu_mid_tensor(
        mid_f, dx, di, map, map_bytes, OFFSET, bytes, OFFSET + bytes, bytes,
        TYPE_Q4_K, K, M, NEXP, 1u, USED));
    CHECK(!unsetenv("DS4_LING3VL_NO_MOE_FUSE"));
    CHECK(ds4_gpu_routed_silu_mid_tensor(
        mid_f, dx, di, map, map_bytes, OFFSET, bytes, OFFSET + bytes, bytes,
        TYPE_Q4_K, K, M, NEXP, 1u, USED));

    const size_t ny = (size_t)USED * M * sizeof(float);
    float *ru = malloc(ny), *rf = malloc(ny);
    CHECK(ru && rf);
    CHECK(ds4_gpu_tensor_read(mid_u, 0, ru, ny));
    CHECK(ds4_gpu_tensor_read(mid_f, 0, rf, ny));
    double num = 0.0, den = 0.0;
    for (unsigned i = 0; i < USED * M; i++) {
        CHECK(isfinite(ru[i]) && isfinite(rf[i]));
        const double d = (double)rf[i] - (double)ru[i];
        num += d * d;
        den += (double)ru[i] * (double)ru[i];
    }
    const double rel = sqrt(num) / (sqrt(den) + 1e-12);
    CHECK(rel < 1e-5);

    ds4_gpu_tensor_free(dx);
    ds4_gpu_tensor_free(di);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid_u);
    ds4_gpu_tensor_free(mid_f);
    ds4_gpu_cleanup();
    free(ru);
    free(rf);
    free(map);
    printf("Ling Q4_K n=1 fused silu mmvq: rel-rms vs unfused SwiGLU < 1e-5 "
           "(kill DS4_LING3VL_NO_MOE_FUSE)\n");
    return 0;
}
