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

/* Joint decode projection at native shapes; distinct layer sets keep its
 * 36 MB weight stream outside L2. Separate production APIs are the oracle. */
static void run_proj(void) {
    enum { SETS = 4, MATRICES = 5, TOKENS = 8, Q4_TYPE = 12 };
    const unsigned outputs[MATRICES] = {8192, 512, 512, IQ, IK};
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    size_t weight_bytes = 0;
    ds4_gvq_record records[SETS * MATRICES];
    ds4_unit_tensor_in inputs[SETS * MATRICES];
    for (unsigned s = 0; s < SETS; s++) {
        for (unsigned c = 0; c < MATRICES; c++) {
            const unsigned i = s * MATRICES + c;
            const size_t wb = (size_t)outputs[c] * H / (c < 3 ? 256 : Q8_BLOCK) * (c < 3 ? 144 : Q8_BYTES);
            records[i] = (ds4_gvq_record){weight_bytes, wb, c < 3 ? Q4_TYPE : Q8_TYPE, outputs[c], H,
                DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK};
            inputs[i] = (ds4_unit_tensor_in){weight_bytes, wb, 0, i + 1};
            weight_bytes += wb;
        }
    }
    const size_t bytes = (weight_bytes + page - 1) / page * page;
    uint8_t *map = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(map != MAP_FAILED);
    uint32_t state = 0x92731ab9u;
    for (unsigned i = 0; i < SETS * MATRICES; i++) {
        const ds4_gvq_record *r = &records[i];
        uint8_t *w = map + r->offset;
        for (size_t b = 0; b < r->bytes; b++) {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            w[b] = (uint8_t)state;
        }
        const unsigned block = i % MATRICES < 3 ? 144 : Q8_BYTES;
        for (size_t b = 0; b < r->bytes; b += block) {
            const uint16_t d = 0x2000, dmin = 0x1800;
            memcpy(w + b, &d, 2);
            if (block == 144) { memcpy(w + b + 2, &dmin, 2); }
        }
    }
    CHECK(!mprotect(map, bytes, PROT_READ));
    CHECK(ds4_gpu_model_source_bind(map, bytes, DS4_MSRC_ROLE_PRIMARY, -1,
        DS4_RESIDENCY_EAGER_DEVICE, "base", "MiniMax joint projection test") >= 0);
    CHECK(ds4_gpu_set_model_map_range(map, bytes, 0, bytes, weight_bytes));
    ds4_unit_compile_params params = {0};
    params.max_unit_bytes = bytes; params.vmm_granularity = 256; params.device_promote = 1;
    ds4_phys_unit units[SETS * MATRICES];
    uint64_t populated = 0;
    CHECK(ds4_units_compile(inputs, SETS * MATRICES, &params, units) == 1);
    CHECK(!ds4_gpu_model_units_bind(map, units, 1));
    CHECK(ds4_gpu_materialize_model_plan(map, bytes, &populated));
    ds4_gvq_model *model = ds4_gpu_gvq_bind(map, bytes, records, SETS * MATRICES);
    CHECK(model);
    ds4_gvq_ctx *ctx = ds4_gpu_gvq_create_linear(model, ds4_gpu_gvq_workspace_linear(TOKENS), TOKENS);
    CHECK(ctx);
    float x[TOKENS * H];
    for (unsigned i = 0; i < TOKENS * H; i++) { x[i] = widen(bf16(((int)(i * 37 % 1021) - 510) / 1024.0f)); }
    ds4_gpu_tensor *input = tensor(sizeof(x), x), *out[MATRICES];
    float *expected[MATRICES], *got = malloc(((size_t)TOKENS * outputs[0] + 16) * sizeof(float));
    CHECK(got);
    for (unsigned c = 0; c < MATRICES; c++) {
        const size_t cap = ((size_t)TOKENS * outputs[c] + 16) * sizeof(float);
        expected[c] = malloc(cap); CHECK(expected[c]);
        memset(expected[c], 0xa5, cap);
        out[c] = tensor(cap, expected[c]);
    }
    const unsigned widths[] = {1, TOKENS};
    for (unsigned j = 0; j < sizeof(widths) / sizeof(widths[0]); j++) {
        const unsigned rows = widths[j];
        for (unsigned s = 0; s < SETS; s++) {
            const uint32_t w[] = {s * MATRICES, s * MATRICES + 1, s * MATRICES + 2};
            CHECK(ds4_gpu_gvq_q4_decode_many(ctx, out, w, 3, input, rows));
            CHECK(ds4_gpu_minimax_q8_pair(ctx, out[3], out[4], s * MATRICES + 3, s * MATRICES + 4,
                input, H, IQ, IK, rows, DS4_MINIMAX_NORM_F32));
            for (unsigned c = 0; c < MATRICES; c++) {
                const size_t cap = ((size_t)TOKENS * outputs[c] + 16) * sizeof(float);
                CHECK(ds4_gpu_tensor_read(out[c], 0, expected[c], cap));
                for (size_t k = 0; k < (size_t)rows * outputs[c]; k++) { CHECK(isfinite(expected[c][k])); }
                memset(got, 0xa5, cap); CHECK(ds4_gpu_tensor_write(out[c], 0, got, cap));
            }
            CHECK(ds4_gpu_minimax_proj(ctx, out, w, out[3], out[4], s * MATRICES + 3, s * MATRICES + 4,
                IQ, IK, input, rows));
            for (unsigned c = 0; c < MATRICES; c++) {
                const size_t cap = ((size_t)TOKENS * outputs[c] + 16) * sizeof(float);
                CHECK(ds4_gpu_tensor_read(out[c], 0, got, cap) && !memcmp(got, expected[c], cap));
            }
        }
        printf("MINIMAX_PROJ rows=%u sets=%u all_bits_and_guards_exact\n", rows, SETS);
        if (rows != 1) { continue; }
        for (unsigned sample = 0; sample < 7; sample++) {
            struct timespec before, after;
            CHECK(ds4_gpu_synchronize() && !clock_gettime(CLOCK_MONOTONIC, &before));
            for (unsigned it = 0; it < 32; it++) {
                const unsigned s = it % SETS;
                const uint32_t w[] = {s * MATRICES, s * MATRICES + 1, s * MATRICES + 2};
                CHECK(ds4_gpu_minimax_proj(ctx, out, w, out[3], out[4], s * MATRICES + 3, s * MATRICES + 4,
                    IQ, IK, input, rows));
            }
            CHECK(ds4_gpu_synchronize() && !clock_gettime(CLOCK_MONOTONIC, &after));
            printf("MINIMAX_PROJ sample=%u us=%.6f\n", sample,
                ((after.tv_sec - before.tv_sec) * 1e6 + (after.tv_nsec - before.tv_nsec) / 1e3) / 32);
        }
    }
    for (unsigned c = 0; c < MATRICES; c++) { ds4_gpu_tensor_free(out[c]); free(expected[c]); }
    ds4_gpu_tensor_free(input); free(got);
    ds4_gpu_gvq_free(ctx); ds4_gpu_gvq_unbind(model);
    ds4_gpu_unregister_model_map(map); CHECK(!munmap(map, bytes));
}

/* Alias one valid gate/up and down matrix across experts: this exercises
 * production validation and epilogues, not expert bandwidth. */
static void run_moe(void) {
    enum { BOOK = 0x47564342, VQ = 0x47565138, F = DS4_MINIMAX_FFN, EXPERTS = DS4_MINIMAX_EXPERTS };
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    ds4_gvq_record record[4] = {
        {0, H * 2, BF16_TYPE, 1, H, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK},
        {H * 2, 8192 * 8, BOOK, 8192, 8, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK}
    };
    size_t used = H * 2 + 8192 * 8;
    for (unsigned i = 2; i < 4; i++) {
        const unsigned rows = i == 2 ? F : H, cols = i == 2 ? H : F;
        const size_t wb = 128 + rows * 2 + (size_t)rows * cols * 13 / 64;
        record[i] = (ds4_gvq_record){used, wb, VQ, rows, cols, DS4_GVQ_NO_LINK, 1, DS4_GVQ_NO_LINK};
        used = (used + wb + 255) / 256 * 256;
    }
    const size_t bytes = (used + page - 1) / page * page;
    uint8_t *map = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(map != MAP_FAILED);
    memset(map + record[1].offset, 0x38, record[1].bytes);
    for (unsigned i = 2; i < 4; i++) {
        uint8_t *h = map + record[i].offset;
        memcpy(h, "GVQ1", 4); h[4] = 1; h[6] = 13; h[7] = 8;
        const uint32_t dims[] = {record[i].rows, record[i].cols, 8192, 1};
        memcpy(h + 8, dims, sizeof(dims));
        const uint64_t offset[] = {128, 128 + record[i].rows * 2,
            128 + record[i].rows * 2 + (uint64_t)record[i].rows * record[i].cols * 3 / 16, record[i].bytes};
        memcpy(h + 32, offset, sizeof(offset));
        uint16_t *gain = (uint16_t *)(h + 128);
        for (unsigned r = 0; r < record[i].rows; r++) { gain[r] = 0x2800; }
    }
    CHECK(!mprotect(map, bytes, PROT_READ));
    CHECK(ds4_gpu_model_source_bind(map, bytes, DS4_MSRC_ROLE_PRIMARY, -1,
        DS4_RESIDENCY_EAGER_DEVICE, "base", "MiniMax decode residual API test") >= 0);
    CHECK(ds4_gpu_set_model_map_range(map, bytes, 0, bytes, used));
    ds4_unit_tensor_in span = {0, used, 0, 1};
    ds4_unit_compile_params params = {0};
    params.max_unit_bytes = bytes; params.vmm_granularity = 256; params.device_promote = 1;
    ds4_phys_unit unit; uint64_t populated = 0;
    CHECK(ds4_units_compile(&span, 1, &params, &unit) == 1);
    CHECK(!ds4_gpu_model_units_bind(map, &unit, 1));
    CHECK(ds4_gpu_materialize_model_plan(map, bytes, &populated));
    ds4_gvq_model *model = ds4_gpu_gvq_bind(map, bytes, record, 4); CHECK(model);
    ds4_gvq_ctx *ctx = ds4_gpu_gvq_create_linear(model, ds4_gpu_gvq_workspace_linear(1), 1); CHECK(ctx);
    ds4_minimax_moe *moe = ds4_gpu_minimax_moe_create(ctx, 1); CHECK(moe);
    float x[H], shared[H], expected[H], expected_norm[H], got[H];
    for (unsigned i = 0; i < H; i++) {
        x[i] = widen(bf16(((int)(i % 31) - 15) / 64.0f));
        shared[i] = ((int)(i % 127) - 63) / 97.0f;
    }
    const int32_t selected[] = {53, 7, 100, 1};
    const float weights[] = {0.2f, 0.3f, 0.7f, 0.8f};
    uint32_t matrices[3 * EXPERTS];
    for (unsigned i = 0; i < 3 * EXPERTS; i++) { matrices[i] = i < 2 * EXPERTS ? 2 : 3; }
    ds4_gpu_tensor *input = tensor(sizeof(x), x), *cur = tensor(sizeof(x), x), *out = tensor(sizeof(x), NULL);
    ds4_gpu_tensor *sh = tensor(sizeof(shared), shared), *ids = tensor(sizeof(selected), selected);
    ds4_gpu_tensor *w = tensor(sizeof(weights), weights), *norm = tensor(sizeof(x), NULL);
    CHECK(ds4_gpu_minimax_moe(moe, out, input, ids, w, sh, matrices, 1, 0, NULL));
    CHECK(ds4_gpu_minimax_residual(ctx, cur, out, norm, 0, 1, NULL, NULL, 0,
        DS4_GVQ_ROWS_F32, DS4_MINIMAX_NORM_F32));
    CHECK(ds4_gpu_tensor_read(cur, 0, expected, sizeof(x)));
    CHECK(ds4_gpu_tensor_read(norm, 0, expected_norm, sizeof(x)));
    for (unsigned i = 0; i < H; i++) { CHECK(isfinite(expected[i]) && isfinite(expected_norm[i])); }
    ds4_minimax_residual_out residual = {cur, norm, NULL, NULL, 0, DS4_GVQ_ROWS_F32, DS4_MINIMAX_NORM_F32};
    for (unsigned next = 0; next < 2; next++) {
        residual.next = next ? DS4_GVQ_NO_LINK : 0;
        CHECK(ds4_gpu_tensor_write(cur, 0, x, sizeof(x)));
        memset(got, 0xa5, sizeof(got));
        CHECK(ds4_gpu_tensor_write(out, 0, got, sizeof(got)) && ds4_gpu_tensor_write(norm, 0, got, sizeof(got)));
        CHECK(ds4_gpu_minimax_moe(moe, out, input, ids, w, sh, matrices, 1, 0, &residual));
        CHECK(ds4_gpu_tensor_read(cur, 0, got, sizeof(got)) && !memcmp(got, expected, sizeof(got)));
        CHECK(ds4_gpu_tensor_read(norm, 0, got, sizeof(got)));
        if (!next) { CHECK(!memcmp(got, expected_norm, sizeof(got))); }
        else { for (unsigned i = 0; i < sizeof(got); i++) { CHECK(((uint8_t *)got)[i] == 0xa5); } }
        CHECK(ds4_gpu_tensor_read(out, 0, got, sizeof(got)));
        for (unsigned i = 0; i < sizeof(got); i++) { CHECK(((uint8_t *)got)[i] == 0xa5); }
    }
    residual.next = 0; residual.norm_format = DS4_MINIMAX_NORM_Q8;
    CHECK(!ds4_gpu_minimax_moe(moe, out, input, ids, w, sh, matrices, 1, 0, &residual));
    residual.norm_format = DS4_MINIMAX_NORM_F32; residual.format = DS4_GVQ_ROWS_BF16;
    CHECK(!ds4_gpu_minimax_moe(moe, out, input, ids, w, sh, matrices, 1, 0, &residual));
    ds4_gpu_tensor_free(norm); ds4_gpu_tensor_free(w); ds4_gpu_tensor_free(ids); ds4_gpu_tensor_free(sh);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(cur); ds4_gpu_tensor_free(input);
    ds4_gpu_minimax_moe_free(moe); ds4_gpu_gvq_free(ctx); ds4_gpu_gvq_unbind(model);
    ds4_gpu_unregister_model_map(map); CHECK(!munmap(map, bytes));
    puts("MINIMAX_MOE_API_PASS: cur/norm exact, dead output untouched, unsupported formats rejected");
}

static double score_fp8(unsigned code) {
    return code < 8 ? code / 512. : ldexp(1. + (code & 7) / 8., (int)(code >> 3) - 7);
}

static double attn_value(const uint8_t *pages, unsigned block, unsigned head, unsigned token, unsigned dim) {
    const double levels[] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    const uint8_t *v = pages + ((size_t)block * 4 + head) * DS4_MINIMAX_KV_PAGE + DS4_MINIMAX_KV_PAGE / 2;
    const unsigned code = (v[token * 64 + dim / 2] >> (4 * (dim % 2))) & 15;
    return levels[code & 7] * (code & 8 ? -1 : 1) * score_fp8(v[128 * 64 + token * 8 + dim / 16]);
}

static void run_attn(void) {
    CHECK(getenv("DS4_MINIMAX_ATTN_PACK") && !getenv("DS4_MINIMAX_ATTN_SIMT"));
    const unsigned cases[][2] = {{9, 125}, {31, 2047}, {33, 8191}, {2048, 63488}};
    for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        const unsigned rows = cases[c][0], pos = cases[c][1], nb = (pos + rows + 127) / 128;
        const size_t count = (size_t)rows * 64 * 128, bytes = count * sizeof(float);
        const size_t page_bytes = (size_t)nb * 4 * DS4_MINIMAX_KV_PAGE, bn = (size_t)rows * 4 * 16;
        float *q = malloc(bytes), *reference = malloc(bytes);
        uint16_t *q16 = malloc(count * sizeof(uint16_t));
        uint8_t *pages = malloc(page_bytes), *reverse = malloc(page_bytes), *got = malloc(bytes + 16);
        int32_t *blocks = malloc(bn * sizeof(int32_t)), *table = malloc(nb * sizeof(int32_t));
        CHECK(q && reference && q16 && pages && reverse && got && blocks && table);
        for (size_t i = 0; i < count; i++) {
            q16[i] = bf16(i < 64 * 128 ? 0.f : ((int)((i * 37) % 1021) - 510) / 1024.f);
            q[i] = widen(q16[i]);
        }
        unsigned seed = 47;
        for (unsigned b = 0; b < nb; b++) for (unsigned h = 0; h < 4; h++) for (unsigned side = 0; side < 2; side++) {
            uint8_t *p = pages + ((size_t)b * 4 + h) * DS4_MINIMAX_KV_PAGE + side * DS4_MINIMAX_KV_PAGE / 2;
            for (unsigned i = 0; i < 128 * 72; i++) {
                seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
                p[i] = i < 128 * 64 ? (uint8_t)seed : (uint8_t)(0x20 + seed % 25);
            }
        }
        for (unsigned r = 0; r < rows; r++) for (unsigned h = 0; h < 4; h++) {
            const unsigned last = (pos + r) / 128, n = last < 15 ? last + 1 : 16;
            for (unsigned j = 0; j < 16; j++) {
                blocks[((size_t)r * 4 + h) * 16 + j] = j >= n ? -1 : n == 1 ? 0 : (int32_t)(j * last / (n - 1));
            }
        }
        for (unsigned b = 0; b < nb; b++) {
            table[b] = (int32_t)(nb - b - 1);
            memcpy(reverse + (size_t)table[b] * 4 * DS4_MINIMAX_KV_PAGE,
                pages + (size_t)b * 4 * DS4_MINIMAX_KV_PAGE, 4 * DS4_MINIMAX_KV_PAGE);
        }
        ds4_gpu_tensor *qt = tensor(bytes, q), *out = tensor(bytes + 16, NULL), *pool = tensor(page_bytes, pages);
        ds4_gpu_tensor *bt = tensor(bn * sizeof(int32_t), blocks), *tt = tensor(nb * sizeof(int32_t), table);
        uint64_t hash = 1469598103934665603ull;
        for (unsigned indirect = 0; indirect < 2; indirect++) {
            CHECK(ds4_gpu_tensor_write(pool, 0, indirect ? reverse : pages, page_bytes));
            for (unsigned flags = 0; flags < 4; flags++) {
                const bool qb = (flags & DS4_MINIMAX_BF16_Q) != 0, hb = (flags & DS4_MINIMAX_BF16_HEADS) != 0;
                CHECK(ds4_gpu_tensor_write(qt, 0, qb ? (void *)q16 : (void *)q, count * (qb ? 2 : 4)));
                memset(got, 0xa5, bytes + 16); CHECK(ds4_gpu_tensor_write(out, 0, got, bytes + 16));
                CHECK(ds4_gpu_minimax_attn(out, qt, pool, indirect ? tt : NULL, bt, NULL, rows, pos, flags));
                CHECK(ds4_gpu_tensor_read(out, 0, got, bytes + 16));
                for (size_t i = count * (hb ? 2 : 4); i < bytes + 16; i++) { CHECK(got[i] == 0xa5); }
                for (size_t i = 0; i < count; i++) {
                    const float v = hb ? widen(((uint16_t *)got)[i]) : ((float *)got)[i];
                    CHECK(isfinite(v) && fabsf(v) <= 6.01f);
                    if (!indirect && !flags) {
                        reference[i] = v; uint32_t bits; memcpy(&bits, &v, sizeof(bits));
                        hash = (hash ^ bits) * 1099511628211ull;
                    } else { CHECK(v == reference[i]); }
                }
            }
        }
        // A zero query attends uniformly, including its causal partial block.
        for (unsigned h = 0; h < 4; h++) for (unsigned d = 0; d < 128; d++) {
            double sum = 0; unsigned n = 0;
            for (unsigned j = 0; j < 16 && blocks[h * 16 + j] >= 0; j++) {
                const unsigned b = (unsigned)blocks[h * 16 + j];
                for (unsigned k = 0; k < 128 && b * 128 + k <= pos; k++) { sum += attn_value(pages, b, h, k, d); n++; }
            }
            const double expected = sum / n;
            CHECK(fabs(reference[(size_t)h * 16 * 128 + d] - expected) < .00002 + .005 * fabs(expected));
        }
        printf("MINIMAX_ATTN_PASS rows=%u pos=%u hash=%016llx\n", rows, pos, (unsigned long long)hash);
        ds4_gpu_tensor_free(qt); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(pool);
        ds4_gpu_tensor_free(bt); ds4_gpu_tensor_free(tt);
        free(q); free(reference); free(q16); free(pages); free(reverse); free(got); free(blocks); free(table);
    }
}

static void score_quantize(float *q, size_t count) {
    const double levels[] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    for (size_t g = 0; g < count; g += 16) {
        float amax = 0;
        for (unsigned i = 0; i < 16; i++) {
            q[g + i] = widen(bf16(q[g + i]));
            amax = fmaxf(amax, fabsf(q[g + i]));
        }
        // Compare in input units so -ffast-math cannot move a nearest-even tie via reciprocal division.
        const double wanted = fmaxf(amax, 6.f / 512.f);
        unsigned sc = 0;
        for (unsigned i = 1; i < 127; i++) {
            const double d = fabs(6 * score_fp8(i) - wanted), old = fabs(6 * score_fp8(sc) - wanted);
            if (d < old || (d == old && !(i & 1))) { sc = i; }
        }
        const double scale = score_fp8(sc);
        for (unsigned i = 0; i < 16; i++) {
            const double v = q[g + i];
            unsigned code = 0;
            for (unsigned j = 1; j < 8; j++) {
                const double d = fabs(fabs(v) - levels[j] * scale), old = fabs(fabs(v) - levels[code] * scale);
                if (d < old || (d == old && !(j & 1))) { code = j; }
            }
            q[g + i] = (float)(levels[code] * scale * (v < 0 ? -1 : 1));
        }
    }
}

static void run_score(void) {
    CHECK(getenv("DS4_MINIMAX_SCORE_FP4"));
    const unsigned cases[][2] = {{9, 2047}, {31, 2039}, {33, 4081}, {128, 65408}, {2048, 63488}, {9, 1048567}};
    const double levels[] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        const unsigned rows = cases[c][0], pos = cases[c][1], total = pos + rows;
        const unsigned stride = (total + DS4_MINIMAX_BLOCK - 1) / DS4_MINIMAX_BLOCK;
        const size_t qn = (size_t)rows * DS4_MINIMAX_KV_HEADS * DS4_MINIMAX_HEAD;
        const size_t sn = (size_t)rows * DS4_MINIMAX_KV_HEADS * stride;
        const size_t bn = (size_t)rows * DS4_MINIMAX_KV_HEADS * DS4_MINIMAX_TOPK;
        float *q = malloc(qn * sizeof(float)), *scores = calloc(sn, sizeof(float));
        uint8_t *keys = malloc((size_t)total * DS4_MINIMAX_FP4_ROW);
        int32_t *blocks = malloc(bn * sizeof(int32_t));
        CHECK(q && scores && keys && blocks);
        unsigned seed = 42;
        for (size_t i = 0; i < qn; i++) { q[i] = ((int)(rand_r(&seed) % 4097) - 2048) / 1024.f; }
        for (unsigned i = 0; i < DS4_MINIMAX_HEAD; i++) {
            q[i] = 0;
            q[DS4_MINIMAX_HEAD + i] *= 1e-4f;  // exercise the encoder's minimum scale
        }
        for (unsigned t = 0; t < total; t++) {
            for (unsigned d = 0; d < DS4_MINIMAX_FP4_ROW; d++) {
                keys[(size_t)t * DS4_MINIMAX_FP4_ROW + d] = d < DS4_MINIMAX_FP4_DATA ?
                    (uint8_t)rand_r(&seed) : (uint8_t)(0x28 + rand_r(&seed) % 17);
            }
        }
        ds4_gpu_tensor *query = tensor(qn * sizeof(float), q), *pool = tensor((size_t)total * DS4_MINIMAX_FP4_ROW, keys);
        ds4_gpu_tensor *score = tensor(sn * sizeof(float), scores), *picked = tensor(bn * sizeof(int32_t), NULL);
        CHECK(ds4_gpu_minimax_select(picked, score, rows, query, pool, rows, pos));
        CHECK(ds4_gpu_tensor_read(score, 0, scores, sn * sizeof(float)));
        CHECK(ds4_gpu_tensor_read(picked, 0, blocks, bn * sizeof(int32_t)));
        for (size_t i = 0; i < qn; i++) { q[i] = widen(bf16(q[i])); }
        if (!getenv("DS4_MINIMAX_SCORE_FP4") || strcmp(getenv("DS4_MINIMAX_SCORE_FP4"), "0") != 0) {
            for (size_t i = 0; i < qn; i += 16) {
                float high[16], low[16];
                memcpy(high, q + i, sizeof(high));
                score_quantize(high, 16);
                for (unsigned j = 0; j < 16; j++) { low[j] = q[i + j] - high[j]; }
                score_quantize(low, 16);
                for (unsigned j = 0; j < 16; j++) { q[i + j] = high[j] + low[j]; }
            }
        }
        const unsigned probes[] = {0, rows / 2, rows - 1};
        for (unsigned i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) {
            const unsigned row = probes[i], own = (pos + row) / DS4_MINIMAX_BLOCK;
            for (unsigned h = 0; h < DS4_MINIMAX_KV_HEADS; h++) {
                const int32_t *list = blocks + ((size_t)row * DS4_MINIMAX_KV_HEADS + h) * DS4_MINIMAX_TOPK;
                CHECK(list[DS4_MINIMAX_TOPK - 1] == (int32_t)own);
                for (unsigned j = 1; j < DS4_MINIMAX_TOPK; j++) { CHECK(list[j - 1] < list[j]); }
                const unsigned bs[] = {0, own / 2, own};
                for (unsigned j = 0; j < sizeof(bs) / sizeof(bs[0]); j++) {
                    const unsigned b = bs[j];
                    double best = -INFINITY;
                    for (unsigned t = b * DS4_MINIMAX_BLOCK; t < (b + 1) * DS4_MINIMAX_BLOCK && t <= pos + row; t++) {
                        double sum = 0;
                        for (unsigned d = 0; d < DS4_MINIMAX_HEAD; d++) {
                            const uint8_t *key = keys + (size_t)t * DS4_MINIMAX_FP4_ROW;
                            const unsigned code = (key[d / 2] >> (4 * (d % 2))) & 15;
                            const unsigned sc = key[DS4_MINIMAX_FP4_DATA + d / 16];
                            const double scale = ldexp(1. + (sc & 7) / 8., (int)(sc >> 3) - 7);
                            sum += q[((size_t)row * DS4_MINIMAX_KV_HEADS + h) * DS4_MINIMAX_HEAD + d] *
                                scale * levels[code & 7] * (code & 8 ? -1 : 1);
                        }
                        if (sum > best) { best = sum; }
                    }
                    const double got = scores[((size_t)row * DS4_MINIMAX_KV_HEADS + h) * stride + b];
                    if (fabs(got - best) >= 1e-4) {
                        fprintf(stderr, "score row=%u head=%u block=%u got=%.12g expected=%.12g error=%.12g\n",
                            row, h, b, got, best, got - best);
                    }
                    CHECK(fabs(got - best) < 1e-4);
                }
            }
        }
        printf("MINIMAX_SCORE rows=%u pos=%u CPU_scores_and_causal_selection PASS\n", rows, pos);
        ds4_gpu_tensor_free(query); ds4_gpu_tensor_free(pool); ds4_gpu_tensor_free(score); ds4_gpu_tensor_free(picked);
        free(q); free(keys); free(scores); free(blocks);
    }
}

int main(int argc, char **argv) {
    CHECK(argc <= 2 && ds4_gpu_init());
    if (argc == 2 && !strcmp(argv[1], "attention")) { run_attn(); ds4_gpu_cleanup(); return 0; }
    if (argc == 2 && !strcmp(argv[1], "score")) { run_score(); ds4_gpu_cleanup(); return 0; }
    if (argc == 2 && !strcmp(argv[1], "moe")) { run_moe(); ds4_gpu_cleanup(); return 0; }
    if (argc == 2 && !strcmp(argv[1], "proj")) {
        run_proj(); ds4_gpu_cleanup(); puts("MINIMAX_PROJ_PASS"); return 0;
    }
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
