/* PyTorch source-equation fixtures; no model weights or Python at test time. */
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fixtures/inkling/moe-vectors.h"

enum { ROUTED = 256, USED = 6, SHARED = 2, ACTIVE = USED + SHARED,
       LOGITS = ROUTED + SHARED, BIAS_OFFSET = 4096,
       SCALE_OFFSET = BIAS_OFFSET + REF_CASES * ROUTED * sizeof(float),
       MAP_SIZE = SCALE_OFFSET + REF_CASES * sizeof(float) };

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); \
        exit(1); \
    } \
} while (0)

static ds4_gpu_tensor *upload(const void *data, size_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    CHECK(t);
    if (data) { CHECK(ds4_gpu_tensor_write(t, 0, data, bytes)); }
    return t;
}

static void exact(const char *label, const float *got, const float *want, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (got[i] != want[i] || !isfinite(got[i])) {
            fprintf(stderr, "%s[%zu]: %.9g != %.9g\n", label, i, got[i], want[i]);
            exit(1);
        }
    }
}

static void check_route(const int32_t *ids, const float *routed, const float *shared,
                         unsigned rows, unsigned fixture) {
    for (unsigned t = 0; t < rows; t++) {
        for (unsigned k = 0; k < USED; k++) {
            CHECK(ids[t * USED + k] == ref_ids[fixture * USED + k]);
        }
        double sum = 0.0;
        for (unsigned k = 0; k < ACTIVE; k++) {
            float got = k < USED ? routed[t * USED + k] : shared[t * SHARED + k - USED];
            float want = ref_weights[fixture * ACTIVE + k];
            CHECK(isfinite(got));
            CHECK(fabsf(got - want) <= 2e-6f + 3e-6f * fabsf(want));
            sum += got;
        }
        /* Logsumexp near -1000 has FP32 cancellation in the source too. */
        CHECK(fabs(sum - ACTIVE * ref_scale[fixture]) < 4e-4);
    }
}

static void router(const void *map, unsigned rows, unsigned stride) {
    size_t xbytes = (size_t)rows * stride * sizeof(float);
    size_t rbytes = (size_t)rows * USED * sizeof(float);
    size_t sbytes = (size_t)rows * SHARED * sizeof(float);
    float *x = malloc(xbytes), *r = malloc(rbytes), *s = malloc(sbytes);
    int32_t *ids = malloc(rbytes);
    CHECK(x && r && s && ids);
    ds4_gpu_tensor *dx = upload(NULL, xbytes), *di = upload(NULL, rbytes);
    ds4_gpu_tensor *dr = upload(NULL, rbytes), *ds = upload(NULL, sbytes);
    for (unsigned f = 0; f < REF_CASES; f++) {
        for (unsigned t = 0; t < rows; t++) {
            memcpy(x + (size_t)t * stride, ref_logits + f * LOGITS, LOGITS * sizeof(float));
            for (unsigned j = LOGITS; j < stride; j++) { x[(size_t)t * stride + j] = NAN; }
        }
        CHECK(ds4_gpu_tensor_write(dx, 0, x, xbytes));
        CHECK(ds4_gpu_inkling_route(di, dr, ds, dx, map, MAP_SIZE,
                                    BIAS_OFFSET + f * ROUTED * sizeof(float),
                                    SCALE_OFFSET + f * sizeof(float), rows, stride));
        CHECK(ds4_gpu_tensor_read(di, 0, ids, rbytes));
        CHECK(ds4_gpu_tensor_read(dr, 0, r, rbytes));
        CHECK(ds4_gpu_tensor_read(ds, 0, s, sbytes));
        check_route(ids, r, s, rows, f);
    }
    CHECK(!ds4_gpu_inkling_route(di, dr, ds, dx, map, MAP_SIZE, BIAS_OFFSET, SCALE_OFFSET, 0, stride));
    CHECK(!ds4_gpu_inkling_route(di, dr, ds, dx, map, MAP_SIZE, BIAS_OFFSET, SCALE_OFFSET, rows, LOGITS - 1));
    CHECK(!ds4_gpu_inkling_route(di, dr, ds, dx, map, SCALE_OFFSET, BIAS_OFFSET, SCALE_OFFSET, rows, stride));
    CHECK(!ds4_gpu_inkling_route(di, di, ds, dx, map, MAP_SIZE, BIAS_OFFSET, SCALE_OFFSET, rows, stride));
    CHECK(!ds4_gpu_inkling_route(di, dr, ds, dx, map, MAP_SIZE, BIAS_OFFSET, SCALE_OFFSET, rows + 1, stride));
    ds4_gpu_tensor_free(dx); ds4_gpu_tensor_free(di);
    ds4_gpu_tensor_free(dr); ds4_gpu_tensor_free(ds);
    free(x); free(r); free(s); free(ids);
    printf("router rows=%u stride=%u %u source cases passed\n", rows, stride, REF_CASES);
}

static void swiglu(unsigned rows, unsigned width) {
    size_t count = (size_t)rows * width, bytes = count * sizeof(float);
    float *x = malloc(2 * bytes), *got = malloc(bytes), *want = malloc(bytes);
    float *gamma = malloc(rows * sizeof(float));
    CHECK(x && got && want && gamma);
    for (unsigned t = 0; t < rows; t++) {
        gamma[t] = ref_gammas[t % REF_CASES];
        for (unsigned c = 0; c < width; c++) {
            size_t i = (size_t)t * width + c;
            unsigned src = (t % REF_CASES) * REF_ACT_WIDTH + c % REF_ACT_WIDTH;
            x[2 * i] = ref_gateup[2 * src];
            x[2 * i + 1] = ref_gateup[2 * src + 1];
        }
    }
    ds4_gpu_tensor *dx = upload(x, 2 * bytes), *dy = upload(NULL, bytes);
    ds4_gpu_tensor *dg = upload(gamma, rows * sizeof(float));
    for (unsigned weighted = 0; weighted < 2; weighted++) {
        for (unsigned t = 0; t < rows; t++) {
            for (unsigned c = 0; c < width; c++) {
                unsigned src = (t % REF_CASES) * REF_ACT_WIDTH + c % REF_ACT_WIDTH;
                want[(size_t)t * width + c] = (weighted ? ref_act_weighted : ref_act_plain)[src];
            }
        }
        CHECK(ds4_gpu_inkling_swiglu(dy, dx, weighted ? dg : NULL, width, rows));
        CHECK(ds4_gpu_tensor_read(dy, 0, got, bytes));
        exact("swiglu", got, want, count);
    }
    CHECK(!ds4_gpu_inkling_swiglu(dx, dx, dg, width, rows));
    CHECK(!ds4_gpu_inkling_swiglu(dy, dx, dg, 0, rows));
    CHECK(!ds4_gpu_inkling_swiglu(dy, dx, dg, width, rows + 1));
    ds4_gpu_tensor_free(dx); ds4_gpu_tensor_free(dy); ds4_gpu_tensor_free(dg);
    free(x); free(got); free(want); free(gamma);
    printf("swiglu rows=%u width=%u plain/shared-gamma BF16 exact\n", rows, width);
}

static void combine(unsigned rows, unsigned width) {
    size_t count = (size_t)rows * width, bytes = count * sizeof(float);
    float *r = malloc(USED * bytes), *s = malloc(SHARED * bytes);
    float *weights = malloc((size_t)rows * USED * sizeof(float));
    float *got = malloc(bytes), *want = malloc(bytes);
    CHECK(r && s && weights && got && want);
    for (unsigned t = 0; t < rows; t++) {
        unsigned f = t % REF_CASES;
        memcpy(weights + t * USED, ref_weights + f * ACTIVE, USED * sizeof(float));
        for (unsigned c = 0; c < width; c++) {
            want[(size_t)t * width + c] = ref_combined[f * REF_WIDTH + c % REF_WIDTH];
            for (unsigned k = 0; k < USED; k++) {
                r[((size_t)t * USED + k) * width + c] = ref_routed[(f * USED + k) * REF_WIDTH + c % REF_WIDTH];
            }
            for (unsigned k = 0; k < SHARED; k++) {
                s[((size_t)t * SHARED + k) * width + c] = ref_shared[(f * SHARED + k) * REF_WIDTH + c % REF_WIDTH];
            }
        }
    }
    ds4_gpu_tensor *dr = upload(r, USED * bytes), *ds = upload(s, SHARED * bytes);
    ds4_gpu_tensor *dw = upload(weights, (size_t)rows * USED * sizeof(float));
    ds4_gpu_tensor *dy = upload(NULL, bytes);
    CHECK(ds4_gpu_inkling_combine(dy, dr, ds, dw, width, rows));
    CHECK(ds4_gpu_tensor_read(dy, 0, got, bytes));
    exact("combine", got, want, count);
    CHECK(!ds4_gpu_inkling_combine(dr, dr, ds, dw, width, rows));
    CHECK(!ds4_gpu_inkling_combine(dy, dr, ds, dw, width, rows + 1));
    CHECK(!ds4_gpu_inkling_combine(dy, dr, ds, dw, width, 0));
    ds4_gpu_tensor_free(dr); ds4_gpu_tensor_free(ds);
    ds4_gpu_tensor_free(dw); ds4_gpu_tensor_free(dy);
    free(r); free(s); free(weights); free(got); free(want);
    printf("combine rows=%u width=%u routed/shared BF16 boundaries exact\n", rows, width);
}

static void capture_route(const void *map) {
    float r[USED], s[SHARED];
    int32_t ids[USED];
    ds4_gpu_tensor *dx = upload(NULL, LOGITS * sizeof(float));
    ds4_gpu_tensor *di = upload(NULL, sizeof(ids)), *dr = upload(NULL, sizeof(r));
    ds4_gpu_tensor *ds = upload(NULL, sizeof(s));
    struct ds4_layer_graph_key key = {0};
    key.n_tok = 1; key.cur_hc = dx; key.q = di; key.kv = dr; key.heads = ds;
    unsigned captures = 0, replays = 0;
    for (unsigned t = 0; t < REF_CASES; t++) {
        float x[LOGITS];
        /* Cases 1, 5 and 7 use zero bias and scale 1; IDs and weights change. */
        const unsigned fixtures[] = {1, 5, 7};
        unsigned f = fixtures[t % (sizeof(fixtures) / sizeof(fixtures[0]))];
        memcpy(x, ref_logits + f * LOGITS, sizeof(x));
        CHECK(ds4_gpu_tensor_write(dx, 0, x, sizeof(x)));
        int mode = ds4_cuda_layer_graph_begin_or_replay(0, &key);
        if (mode != 1) {
            CHECK(ds4_gpu_inkling_route(di, dr, ds, dx, map, MAP_SIZE,
                                        BIAS_OFFSET + ROUTED * sizeof(float),
                                        SCALE_OFFSET + sizeof(float), 1, LOGITS));
        }
        if (mode == 0) { ds4_cuda_layer_graph_end_or_commit(0); captures++; }
        replays += mode == 1;
        CHECK(ds4_gpu_tensor_read(di, 0, ids, sizeof(ids)));
        CHECK(ds4_gpu_tensor_read(dr, 0, r, sizeof(r)));
        CHECK(ds4_gpu_tensor_read(ds, 0, s, sizeof(s)));
        check_route(ids, r, s, 1, f);
    }
    CHECK(captures == 1 && replays == REF_CASES - 2);
    ds4_gpu_tensor_free(dx); ds4_gpu_tensor_free(di);
    ds4_gpu_tensor_free(dr); ds4_gpu_tensor_free(ds);
    printf("router capture=%u replay=%u changing IDs/weights passed\n", captures, replays);
}

int main(void) {
    CHECK(ds4_gpu_init());
    void *map = NULL;
    CHECK(posix_memalign(&map, BIAS_OFFSET, MAP_SIZE) == 0);
    memset(map, 0, MAP_SIZE);
    memcpy((char *)map + BIAS_OFFSET, ref_bias, sizeof(ref_bias));
    memcpy((char *)map + SCALE_OFFSET, ref_scale, sizeof(ref_scale));
    CHECK(ds4_gpu_set_model_map(map, MAP_SIZE));
    const unsigned rows[] = {1, 7, 64, 513}, strides[] = {LOGITS, 264};
    for (unsigned t = 0; t < sizeof(rows) / sizeof(rows[0]); t++) {
        for (unsigned s = 0; s < sizeof(strides) / sizeof(strides[0]); s++) {
            router(map, rows[t], strides[s]);
        }
    }
    const unsigned act_widths[] = {17, 2048, 16384}, act_rows[] = {1, 8, 257};
    for (unsigned c = 0; c < sizeof(act_widths) / sizeof(act_widths[0]); c++) {
        for (unsigned t = 0; t < sizeof(act_rows) / sizeof(act_rows[0]); t++) {
            swiglu(act_rows[t], act_widths[c]);
        }
    }
    combine(1, REF_WIDTH); combine(9, 4096); combine(513, 4096);
    capture_route(map);
    ds4_gpu_cleanup();
    free(map);
    puts("Inkling MoE primitive checks passed");
    return 0;
}
