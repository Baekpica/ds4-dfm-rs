/* Ling n=1 Q5_K gate/up pair vs two singles. */
#include "ds4_gpu.h"
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
    TYPE_Q5_K = 13u
};

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
} block_q5_k;

_Static_assert(sizeof(block_q5_k) == 176, "Q5_K block");

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static void fill_q5_k(block_q5_k *blocks, uint64_t count) {
    uint32_t state = 0x5e1c93a7u;
    for (uint64_t b = 0; b < count; b++) {
        blocks[b].d = 0x2000u;
        blocks[b].dmin = 0x1c00u;
        for (uint32_t i = 0; i < 12u; i++) {
            state = state * 1664525u + 1013904223u;
            blocks[b].scales[i] = (uint8_t)(state >> 24u);
        }
        for (uint32_t i = 0; i < QK_K / 8; i++) {
            state = state * 1664525u + 1013904223u;
            blocks[b].qh[i] = (uint8_t)(state >> 24u);
        }
        for (uint32_t i = 0; i < QK_K / 2; i++) {
            state = state * 1664525u + 1013904223u;
            blocks[b].qs[i] = (uint8_t)(state >> 24u);
        }
    }
}

int main(void) {
    const uint64_t blocks = (uint64_t)NEXP * M * (K / QK_K);
    const uint64_t bytes = blocks * sizeof(block_q5_k);
    const uint64_t map_bytes = OFFSET + 2u * bytes;
    void *map = NULL;
    CHECK(!posix_memalign(&map, OFFSET, map_bytes));
    memset(map, 0, map_bytes);
    fill_q5_k((block_q5_k *)((char *)map + OFFSET), blocks);
    fill_q5_k((block_q5_k *)((char *)map + OFFSET + bytes), blocks);

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
    ds4_gpu_tensor *g0 = ds4_gpu_tensor_alloc(USED * M * sizeof(float));
    ds4_gpu_tensor *u0 = ds4_gpu_tensor_alloc(USED * M * sizeof(float));
    ds4_gpu_tensor *g1 = ds4_gpu_tensor_alloc(USED * M * sizeof(float));
    ds4_gpu_tensor *u1 = ds4_gpu_tensor_alloc(USED * M * sizeof(float));
    CHECK(dx && di && g0 && u0 && g1 && u1);
    CHECK(ds4_gpu_tensor_write(dx, 0, x, sizeof(x)));
    CHECK(ds4_gpu_tensor_write(di, 0, ids, sizeof(ids)));

    CHECK(!setenv("DS4_MMQ_Q5_PAIR", "0", 1));
    CHECK(ds4_gpu_routed_gate_up_tensor(
        g0, u0, dx, di, map, map_bytes, OFFSET, bytes, OFFSET + bytes, bytes,
        TYPE_Q5_K, K, M, NEXP, 1u, USED));
    CHECK(!unsetenv("DS4_MMQ_Q5_PAIR"));
    CHECK(ds4_gpu_routed_gate_up_tensor(
        g1, u1, dx, di, map, map_bytes, OFFSET, bytes, OFFSET + bytes, bytes,
        TYPE_Q5_K, K, M, NEXP, 1u, USED));

    const size_t ny = (size_t)USED * M * sizeof(float);
    float *sg = malloc(ny), *su = malloc(ny), *pg = malloc(ny), *pu = malloc(ny);
    CHECK(sg && su && pg && pu);
    CHECK(ds4_gpu_tensor_read(g0, 0, sg, ny));
    CHECK(ds4_gpu_tensor_read(u0, 0, su, ny));
    CHECK(ds4_gpu_tensor_read(g1, 0, pg, ny));
    CHECK(ds4_gpu_tensor_read(u1, 0, pu, ny));
    CHECK(!memcmp(sg, pg, ny));
    CHECK(!memcmp(su, pu, ny));

    CHECK(!setenv("DS4_MMQ_VEC_SANITIZE", "1", 1));
    CHECK(ds4_gpu_routed_gate_up_tensor(
        g0, u0, dx, di, map, map_bytes, OFFSET, bytes, OFFSET + bytes, bytes,
        TYPE_Q5_K, K, M, NEXP, 1u, USED));
    CHECK(!unsetenv("DS4_MMQ_VEC_SANITIZE"));
    CHECK(ds4_gpu_tensor_read(g0, 0, sg, ny));
    CHECK(ds4_gpu_tensor_read(u0, 0, su, ny));
    CHECK(!memcmp(sg, pg, ny));
    CHECK(!memcmp(su, pu, ny));

    ds4_gpu_tensor_free(dx);
    ds4_gpu_tensor_free(di);
    ds4_gpu_tensor_free(g0);
    ds4_gpu_tensor_free(u0);
    ds4_gpu_tensor_free(g1);
    ds4_gpu_tensor_free(u1);
    ds4_gpu_cleanup();
    free(sg);
    free(su);
    free(pg);
    free(pu);
    free(map);
    printf("Ling Q5_K n=1 gate/up pair: matches two singles (kill DS4_MMQ_Q5_PAIR=0)\n");
    printf("Ling mmvq vec sanitize skip: matches DS4_MMQ_VEC_SANITIZE=1\n");
    return 0;
}
