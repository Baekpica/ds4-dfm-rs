/* Independent FP64 relative projection and post-projection tau reference. */
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { HEADS = 32, DIM = 128, REL_DIM = 16, LOCAL = 512, GLOBAL = 1024,
       TAU_FLOOR = 128000, MAP_OFFSET = 4096,
       MAP_SIZE = MAP_OFFSET + REL_DIM * GLOBAL * sizeof(uint16_t) };
static const float TAU_ALPHA = 0.1f;

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); \
        exit(1); \
    } \
} while (0)

static uint16_t bf_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
}

static float bf(float x) {
    uint32_t u = (uint32_t)bf_bits(x) << 16;
    memcpy(&x, &u, sizeof(x));
    return x;
}

static float weight(unsigned d, unsigned e) {
    return ((int)((d * 31 + e * 17) % 127) - 63) / 256.0f;
}

static ds4_gpu_tensor *upload(const void *x, size_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    CHECK(t);
    if (x) { CHECK(ds4_gpu_tensor_write(t, 0, x, bytes)); }
    return t;
}

static float tau(uint32_t pos, unsigned extent) {
    if (extent == LOCAL) { return 1.0f; }
    float n = (float)((uint64_t)pos + 1);
    return 1.0f + TAU_ALPHA * logf(fmaxf(n / TAU_FLOOR, 1.0f));
}

static void check(float got, float want) {
    if (!isfinite(got) || got != want) {
        fprintf(stderr, "got %.9g, want %.9g\n", got, want);
        exit(1);
    }
}

static void verify(const float *qout, const float *rel, const float *q,
                   const float *r, const uint32_t *pos, unsigned rows, unsigned extent) {
    for (unsigned t = 0; t < rows; t++) {
        float scale = tau(pos[t], extent);
        for (unsigned h = 0; h < HEADS; h++) {
            size_t row = (size_t)t * HEADS + h;
            for (unsigned c = 0; c < DIM; c++) {
                check(qout[row * DIM + c], bf(bf(q[row * DIM + c]) * scale));
            }
            for (unsigned e = 0; e < extent; e++) {
                double sum = 0.0;
                for (unsigned d = 0; d < REL_DIM; d++) {
                    sum += (double)bf(r[row * REL_DIM + d]) * weight(d, e);
                }
                check(rel[row * extent + e], bf(bf((float)sum) * scale));
            }
        }
    }
}

static void run_case(const void *map, unsigned rows, unsigned extent) {
    size_t qbytes = (size_t)rows * HEADS * DIM * sizeof(float);
    size_t rbytes = (size_t)rows * HEADS * REL_DIM * sizeof(float);
    size_t obytes = (size_t)rows * HEADS * extent * sizeof(float);
    size_t pbytes = rows * sizeof(uint32_t);
    float *q = malloc(qbytes), *r = malloc(rbytes), *qo = malloc(qbytes), *o = malloc(obytes);
    uint32_t *pos = malloc(pbytes);
    CHECK(q && r && qo && o && pos);
    for (size_t i = 0; i < qbytes / sizeof(float); i++) {
        q[i] = ((int)(i * 17 % 251) - 125) / 31.0f;
    }
    for (size_t i = 0; i < rbytes / sizeof(float); i++) {
        r[i] = ((int)(i * 23 % 257) - 128) / 64.0f + 0.00003f;
    }
    const uint32_t positions[] = {0, LOCAL - 1, TAU_FLOOR - 2, TAU_FLOOR - 1,
                                  TAU_FLOOR, 2 * TAU_FLOOR, 1048575, UINT32_MAX};
    for (unsigned t = 0; t < rows; t++) {
        pos[t] = positions[t % (sizeof(positions) / sizeof(positions[0]))];
    }
    ds4_gpu_tensor *dq = upload(q, qbytes), *dr = upload(r, rbytes), *dp = upload(pos, pbytes);
    ds4_gpu_tensor *dqo = upload(NULL, qbytes), *drel = upload(NULL, obytes);
    CHECK(ds4_gpu_inkling_attn_prep(dqo, drel, dq, dr, dp, map, MAP_SIZE,
                                   MAP_OFFSET, rows, extent));
    CHECK(ds4_gpu_tensor_read(dqo, 0, qo, qbytes));
    CHECK(ds4_gpu_tensor_read(drel, 0, o, obytes));
    verify(qo, o, q, r, pos, rows, extent);
    /* Q may update in place; source R/positions and relative output may not alias. */
    CHECK(ds4_gpu_inkling_attn_prep(dq, drel, dq, dr, dp, map, MAP_SIZE,
                                   MAP_OFFSET, rows, extent));
    CHECK(ds4_gpu_tensor_read(dq, 0, qo, qbytes));
    verify(qo, o, q, r, pos, rows, extent);
    CHECK(!ds4_gpu_inkling_attn_prep(dqo, drel, dq, dr, dp, map, MAP_SIZE,
                                    MAP_OFFSET, rows + 1, extent));
    CHECK(!ds4_gpu_inkling_attn_prep(dqo, drel, dq, dr, dp, map, MAP_SIZE,
                                    MAP_OFFSET, rows, REL_DIM));
    CHECK(!ds4_gpu_inkling_attn_prep(dqo, drel, dq, dr, dp, map, MAP_OFFSET,
                                    MAP_OFFSET, rows, extent));
    CHECK(!ds4_gpu_inkling_attn_prep(dqo, drel, dq, dr, dp, map, MAP_SIZE,
                                    MAP_OFFSET + 1, rows, extent));
    CHECK(!ds4_gpu_inkling_attn_prep(drel, drel, dq, dr, dp, map, MAP_SIZE,
                                    MAP_OFFSET, rows, extent));
    ds4_gpu_tensor_free(dq); ds4_gpu_tensor_free(dr); ds4_gpu_tensor_free(dp);
    ds4_gpu_tensor_free(dqo); ds4_gpu_tensor_free(drel);
    free(q); free(r); free(qo); free(o); free(pos);
    printf("relative prep rows=%u extent=%u BF16 exact\n", rows, extent);
}

static void capture(const void *map) {
    float q[HEADS * DIM], r[HEADS * REL_DIM], qo[HEADS * DIM], rel[HEADS * GLOBAL];
    for (unsigned i = 0; i < HEADS * DIM; i++) { q[i] = (int)(i % 29) / 17.0f; }
    for (unsigned i = 0; i < HEADS * REL_DIM; i++) { r[i] = (int)(i % 13) / 32.0f; }
    ds4_gpu_tensor *dq = upload(q, sizeof(q)), *dr = upload(r, sizeof(r));
    ds4_gpu_tensor *dp = upload(NULL, sizeof(uint32_t));
    ds4_gpu_tensor *dqo = upload(NULL, sizeof(qo)), *drel = upload(NULL, sizeof(rel));
    struct ds4_layer_graph_key key = {0};
    key.n_tok = 1; key.cur_hc = dq; key.q = dr; key.kv = dp; key.heads = drel;
    unsigned captures = 0, replays = 0;
    const uint32_t positions[] = {0, 1, 2 * TAU_FLOOR, 1048575, TAU_FLOOR - 1, TAU_FLOOR, 256000};
    for (unsigned t = 0; t < sizeof(positions) / sizeof(positions[0]); t++) {
        CHECK(ds4_gpu_tensor_write(dp, 0, positions + t, sizeof(uint32_t)));
        int mode = ds4_cuda_layer_graph_begin_or_replay(0, &key);
        if (mode != 1) {
            CHECK(ds4_gpu_inkling_attn_prep(dqo, drel, dq, dr, dp, map, MAP_SIZE,
                                           MAP_OFFSET, 1, GLOBAL));
        }
        if (mode == 0) { ds4_cuda_layer_graph_end_or_commit(0); captures++; }
        replays += mode == 1;
        CHECK(ds4_gpu_tensor_read(dqo, 0, qo, sizeof(qo)));
        CHECK(ds4_gpu_tensor_read(drel, 0, rel, sizeof(rel)));
        verify(qo, rel, q, r, positions + t, 1, GLOBAL);
    }
    CHECK(captures == 1 && replays == 5);
    ds4_gpu_tensor_free(dq); ds4_gpu_tensor_free(dr); ds4_gpu_tensor_free(dp);
    ds4_gpu_tensor_free(dqo); ds4_gpu_tensor_free(drel);
    printf("relative prep capture=%u replay=%u live positions passed\n", captures, replays);
}

int main(void) {
    void *map = NULL;
    CHECK(posix_memalign(&map, MAP_OFFSET, MAP_SIZE) == 0);
    memset(map, 0, MAP_SIZE);
    uint16_t *w = (uint16_t *)((char *)map + MAP_OFFSET);
    const unsigned extents[] = {LOCAL, GLOBAL}, rows[] = {1, 2, 17, 65};
    for (unsigned e = 0; e < sizeof(extents) / sizeof(extents[0]); e++) {
        unsigned extent = extents[e];
        for (unsigned d = 0; d < REL_DIM; d++) {
            for (unsigned j = 0; j < extent; j++) { w[d * extent + j] = bf_bits(weight(d, j)); }
        }
        CHECK(ds4_gpu_init());
        CHECK(ds4_gpu_set_model_map(map, MAP_SIZE));
        for (unsigned t = 0; t < sizeof(rows) / sizeof(rows[0]); t++) { run_case(map, rows[t], extent); }
        if (extent == GLOBAL) { capture(map); }
        ds4_gpu_cleanup();
    }
    free(map);
    puts("Inkling attention preparation checks passed");
    return 0;
}
