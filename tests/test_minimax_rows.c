/* Model-free production residual/norm probe. The CPU checks the BF16 add;
 * saved F32-path outputs gate storage variants without changing reductions. */
#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_minimax.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

enum { H = DS4_MINIMAX_HIDDEN, BF16_TYPE = 30, MAX_ROWS = 4096,
       Q8_TYPE = 8, Q8_BLOCK = 32, Q8_BYTES = 34, IQ = 512, IK = 128 };
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "MiniMax rows FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static uint16_t bf16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static float widen(uint16_t x) {
    const uint32_t bits = (uint32_t)x << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static ds4_gpu_tensor *tensor(uint64_t bytes, const void *data) {
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(bytes);
    CHECK(out && (!data || ds4_gpu_tensor_write(out, 0, data, bytes)));
    return out;
}

static void run(ds4_gvq_ctx *ctx, unsigned rows) {
    const size_t count = (size_t)rows * H, bytes = count * sizeof(float);
    float *x = malloc(bytes), *branch = malloc(bytes), *got = malloc(bytes), *norm = malloc(bytes);
    uint16_t *packed = malloc(count * sizeof(uint16_t)), *x16 = malloc(count * sizeof(uint16_t));
    uint16_t *qh_ref = malloc(count * sizeof(uint16_t)), *qh_got = malloc(count * sizeof(uint16_t));
    float *qs_ref = malloc(rows * sizeof(float)), *qs_got = malloc(rows * sizeof(float));
    CHECK(x && branch && got && norm && packed && x16 && qh_ref && qh_got && qs_ref && qs_got);
    for (size_t i = 0; i < count; i++) {
        x[i] = widen(bf16(((int)((i * 37 + i / H) % 1021) - 510) / 64.0f));
        branch[i] = ((int)((i * 19 + i / H * 3) % 509) - 254) / 97.0f;
        packed[i] = bf16(branch[i]);
        x16[i] = bf16(x[i]);
    }
    ds4_gpu_tensor *cur = tensor(bytes, x), *b = tensor(bytes, NULL), *n = tensor(bytes, NULL);
    ds4_gpu_tensor *embed = tensor(count * sizeof(uint16_t), x16);
    ds4_gpu_tensor *qh = tensor(count * sizeof(uint16_t), NULL), *qs = tensor(rows * sizeof(float), NULL);
    const size_t iq_bytes = (size_t)rows * IQ * sizeof(float), ik_bytes = (size_t)rows * IK * sizeof(float);
    ds4_gpu_tensor *iq = rows > DS4_MINIMAX_SPLIT_ROWS ? tensor(iq_bytes, NULL) : NULL;
    ds4_gpu_tensor *ik = rows > DS4_MINIMAX_SPLIT_ROWS ? tensor(ik_bytes, NULL) : NULL;
    float *iq_ref = iq ? malloc(iq_bytes) : NULL, *ik_ref = ik ? malloc(ik_bytes) : NULL;
    CHECK(!iq || (iq_ref && ik_ref));
    const size_t tap_count = rows <= 128 ? count * 5 : 0;
    uint16_t *tap = tap_count ? malloc(tap_count * sizeof(uint16_t)) : NULL;
    ds4_gpu_tensor *taps = tap_count ? tensor(tap_count * sizeof(uint16_t), NULL) : NULL;
    CHECK(!tap_count || tap);
    for (unsigned compact = 1; compact <= 2; compact++) {
        const int branch_bf16 = compact == 1;
        CHECK(ds4_gpu_tensor_write(b, 0, branch_bf16 ? (void *)packed : (void *)branch,
            count * (branch_bf16 ? sizeof(uint16_t) : sizeof(float))));
        for (unsigned format = DS4_GVQ_ROWS_F32; format <= DS4_GVQ_ROWS_BF16; format++) {
            CHECK(ds4_gpu_minimax_embed(cur, embed, rows, (ds4_gvq_rows)format));
            if (format == DS4_GVQ_ROWS_F32) {
                CHECK(ds4_gpu_tensor_read(cur, 0, got, bytes) && !memcmp(got, x, bytes));
            } else {
                CHECK(ds4_gpu_tensor_read(cur, 0, qh_got, count * sizeof(uint16_t)));
                CHECK(!memcmp(qh_got, x16, count * sizeof(uint16_t)));
            }
            CHECK(ds4_gpu_minimax_residual(ctx, cur, b, n, 0, rows, qh, qs, branch_bf16, (ds4_gvq_rows)format,
                DS4_MINIMAX_NORM_F32));
            CHECK(ds4_gpu_tensor_read(cur, 0, got, count * (format ? sizeof(uint16_t) : sizeof(float))));
            for (size_t i = 0; i < count; i++) {
                const uint16_t expected = bf16(x[i] + widen(packed[i]));
                CHECK(format ? ((uint16_t *)got)[i] == expected : got[i] == widen(expected));
            }
            CHECK(ds4_gpu_tensor_read(n, 0, got, bytes));
            CHECK(ds4_gpu_tensor_read(qh, 0, qh_got, count * sizeof(uint16_t)));
            CHECK(ds4_gpu_tensor_read(qs, 0, qs_got, rows * sizeof(float)));
            if (format == DS4_GVQ_ROWS_F32) {
                memcpy(norm, got, bytes);
                memcpy(qh_ref, qh_got, count * sizeof(uint16_t));
                memcpy(qs_ref, qs_got, rows * sizeof(float));
                for (size_t i = 0; i < count; i++) { CHECK(isfinite(norm[i])); }
            } else {
                CHECK(!memcmp(norm, got, bytes));
                CHECK(!memcmp(qh_ref, qh_got, count * sizeof(uint16_t)));
                CHECK(!memcmp(qs_ref, qs_got, rows * sizeof(float)));
                CHECK(!ds4_gpu_minimax_rms(ctx, cur, cur, 0, rows, DS4_GVQ_ROWS_BF16));
            }
            CHECK(ds4_gpu_minimax_rms(ctx, n, cur, 0, rows, (ds4_gvq_rows)format));
            CHECK(ds4_gpu_tensor_read(n, 0, got, bytes) && !memcmp(norm, got, bytes));
            if (tap_count) {
                CHECK(ds4_gpu_minimax_draft_tap(taps, cur, 3, rows, (ds4_gvq_rows)format));
                CHECK(ds4_gpu_tensor_read(taps, 0, tap, tap_count * sizeof(uint16_t)));
                for (size_t i = 0; i < count; i++) {
                    CHECK(tap[(i / H * 5 + 3) * H + i % H] == bf16(x[i] + widen(packed[i])));
                }
            }
            if (rows > DS4_MINIMAX_SPLIT_ROWS) {
                CHECK(ds4_gpu_minimax_q8_pair(ctx, iq, ik, 1, 2, n, H, IQ, IK, rows, DS4_MINIMAX_NORM_F32));
                CHECK(ds4_gpu_tensor_read(iq, 0, iq_ref, iq_bytes));
                CHECK(ds4_gpu_tensor_read(ik, 0, ik_ref, ik_bytes));
                CHECK(ds4_gpu_minimax_embed(cur, embed, rows, (ds4_gvq_rows)format));
                memset(got, 0xa5, bytes);
                CHECK(ds4_gpu_tensor_write(n, 0, got, bytes));
                CHECK(ds4_gpu_minimax_residual(ctx, cur, b, n, 0, rows, qh, qs, branch_bf16,
                    (ds4_gvq_rows)format, DS4_MINIMAX_NORM_Q8));
                CHECK(ds4_gpu_tensor_read(cur, 0, got, count * (format ? sizeof(uint16_t) : sizeof(float))));
                for (size_t i = 0; i < count; i++) {
                    const uint16_t expected = bf16(x[i] + widen(packed[i]));
                    CHECK(format ? ((uint16_t *)got)[i] == expected : got[i] == widen(expected));
                }
                CHECK(ds4_gpu_tensor_read(qh, 0, qh_got, count * sizeof(uint16_t)));
                CHECK(ds4_gpu_tensor_read(qs, 0, qs_got, rows * sizeof(float)));
                CHECK(!memcmp(qh_ref, qh_got, count * sizeof(uint16_t)));
                CHECK(!memcmp(qs_ref, qs_got, rows * sizeof(float)));
                CHECK(ds4_gpu_tensor_read(n, 0, got, bytes));
                for (size_t i = 0; i < bytes; i++) { CHECK(((uint8_t *)got)[i] == 0xa5); }
                // NULL input proves the prepared consumer cannot re-quantize
                // the deliberately stale F32 norm instead of reading scratch.
                CHECK(ds4_gpu_minimax_q8_pair(ctx, iq, ik, 1, 2, NULL, H, IQ, IK, rows, DS4_MINIMAX_NORM_Q8));
                CHECK(ds4_gpu_tensor_read(iq, 0, got, iq_bytes) && !memcmp(got, iq_ref, iq_bytes));
                CHECK(ds4_gpu_tensor_read(ik, 0, got, ik_bytes) && !memcmp(got, ik_ref, ik_bytes));
                printf("MINIMAX_ROWS rows=%u branch_bf16=%d cur_bf16=%u prepared_q8 PASS\n",
                    rows, branch_bf16, format);
            }
            printf("MINIMAX_ROWS rows=%u branch_bf16=%d cur_bf16=%u PASS\n", rows, branch_bf16, format);
        }
    }
    ds4_gpu_tensor_free(ik); ds4_gpu_tensor_free(iq); free(ik_ref); free(iq_ref);
    ds4_gpu_tensor_free(taps); ds4_gpu_tensor_free(embed);
    ds4_gpu_tensor_free(qs); ds4_gpu_tensor_free(qh); ds4_gpu_tensor_free(n);
    ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(cur);
    free(tap); free(qs_got); free(qs_ref); free(qh_got); free(qh_ref); free(x16);
    free(packed); free(norm); free(got); free(branch); free(x);
}

int main(int argc, char **argv) {
    CHECK(argc <= 2 && ds4_gpu_init());
    struct timespec before, after;
    CHECK(!clock_gettime(CLOCK_MONOTONIC, &before));
    CHECK(ds4_gpu_minimax_prepare_decode());
    CHECK(!clock_gettime(CLOCK_MONOTONIC, &after));
    printf("MINIMAX_PRELOAD ms=%.6f\n", (after.tv_sec - before.tv_sec) * 1000.0 +
        (after.tv_nsec - before.tv_nsec) / 1000000.0);
    const size_t page = (size_t)sysconf(_SC_PAGESIZE), wb = H * sizeof(uint16_t);
    const size_t qb0 = (size_t)IQ * (H / Q8_BLOCK) * Q8_BYTES;
    const size_t qb1 = (size_t)IK * (H / Q8_BLOCK) * Q8_BYTES;
    const size_t weight_bytes = wb + qb0 + qb1;
    const size_t bytes = (weight_bytes + page - 1) / page * page;
    uint16_t *map = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(map != MAP_FAILED);
    for (unsigned i = 0; i < H; i++) { map[i] = bf16(((int)(i % 31) - 15) / 64.0f); }
    uint8_t *q8 = (uint8_t *)map + wb;
    uint32_t state = 0x9e3779b9u;
    for (size_t b = 0; b < (qb0 + qb1) / Q8_BYTES; b++) {
        q8[b * Q8_BYTES] = 0; q8[b * Q8_BYTES + 1] = 0x20;  // F16 2^-7
        for (unsigned k = 0; k < Q8_BLOCK; k++) {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            q8[b * Q8_BYTES + 2 + k] = (uint8_t)((int)(state % 255) - 127);
        }
    }
    CHECK(!mprotect(map, bytes, PROT_READ));
    CHECK(ds4_gpu_model_source_bind(map, bytes, DS4_MSRC_ROLE_PRIMARY, -1,
        DS4_RESIDENCY_EAGER_DEVICE, "base", "MiniMax rows test") >= 0);
    CHECK(ds4_gpu_set_model_map_range(map, bytes, 0, bytes, weight_bytes));
    ds4_unit_tensor_in input[] = {{0, wb, 0, 1}, {wb, qb0, 0, 2}, {wb + qb0, qb1, 0, 3}};
    ds4_unit_compile_params params = {0};
    params.max_unit_bytes = bytes; params.vmm_granularity = 256; params.device_promote = 1;
    ds4_phys_unit unit[3];
    uint64_t populated = 0;
    CHECK(ds4_units_compile(input, 3, &params, unit) == 1);
    CHECK(!ds4_gpu_model_units_bind(map, unit, 1));
    CHECK(ds4_gpu_materialize_model_plan(map, bytes, &populated));
    const ds4_gvq_record record[] = {
        {0, wb, BF16_TYPE, 1, H, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK},
        {wb, qb0, Q8_TYPE, IQ, H, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK},
        {wb + qb0, qb1, Q8_TYPE, IK, H, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK}
    };
    ds4_gvq_model *model = ds4_gpu_gvq_bind(map, bytes, record, 3);
    CHECK(model);
    ds4_gvq_ctx *ctx = ds4_gpu_gvq_create_linear(model, ds4_gpu_gvq_workspace_linear(MAX_ROWS), MAX_ROWS);
    CHECK(ctx);
    if (argc == 2) {
        const unsigned rows = (unsigned)strtoul(argv[1], NULL, 10);
        CHECK(rows && rows <= MAX_ROWS);
        run(ctx, rows);
    } else {
        const unsigned widths[] = {1, 8, 9, 128, MAX_ROWS};
        for (unsigned i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) { run(ctx, widths[i]); }
    }
    ds4_gpu_gvq_free(ctx); ds4_gpu_gvq_unbind(model);
    ds4_gpu_unregister_model_map(map); ds4_gpu_cleanup();
    CHECK(!munmap(map, bytes));
    puts("MINIMAX_ROWS_PASS");
    return 0;
}
