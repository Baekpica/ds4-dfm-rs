/* Production Q4 bind/embedding and small-to-wide dispatch. Construct the
 * affine weights independently; exact Q8 inputs isolate MMQ layout errors. */
#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_glmvq.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

enum { Q4_TYPE = 12, BLOCK = 256, PACKED = 144, ALIGNMENT = 128,
       OUTPUTS = 256, MAX_TOKENS = 65 };
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GVQ Q4 FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static void run(unsigned cols) {
    const size_t wb = (size_t)OUTPUTS * cols / BLOCK * PACKED;
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    const size_t bytes = (ALIGNMENT + wb + page - 1) / page * page;
    uint8_t *map = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    float *w = malloc((size_t)OUTPUTS * cols * sizeof(float));
    float *x = malloc((size_t)MAX_TOKENS * cols * sizeof(float));
    float *got = malloc((size_t)MAX_TOKENS * (cols > OUTPUTS ? cols : OUTPUTS) * sizeof(float));
    CHECK(map != MAP_FAILED && w && x && got);
    for (unsigned row = 0; row < OUTPUTS; row++) {
        for (unsigned b = 0; b < cols / BLOCK; b++) {
            uint8_t *p = map + ALIGNMENT + ((size_t)row * cols / BLOCK + b) * PACKED;
            const uint16_t half = 0x1c00u; /* 2^-8, exact in both half formats. */
            memcpy(p, &half, 2); memcpy(p + 2, &half, 2);
            for (unsigned group = 0; group < 8; group++) {
                const unsigned scale = group + 1, min = group + 9;
                if (group < 4) {
                    p[4 + group] |= scale;
                    p[8 + group] |= min;
                } else {
                    p[12 + group - 4] = (scale & 15u) | ((min & 15u) << 4);
                    p[4 + group - 4] |= (scale >> 4) << 6;
                    p[8 + group - 4] |= (min >> 4) << 6;
                }
                for (unsigned i = 0; i < 32; i++) {
                    const unsigned q = (row * 17 + b * 11 + group * 7 + i * 3) % 16;
                    p[16 + group / 2 * 32 + i] |= q << (4 * (group & 1));
                    w[(size_t)row * cols + b * BLOCK + group * 32 + i] =
                        ((float)(scale * q) - min) / 256.0f;
                }
            }
        }
    }
    for (unsigned t = 0; t < MAX_TOKENS; t++) {
        for (unsigned k = 0; k < cols; k++) {
            x[(size_t)t * cols + k] = k % 32 == 0 ? 127.0f : (float)((int)((k + 37 * t) % 63) - 31);
        }
    }
    CHECK(!mprotect(map, bytes, PROT_READ));
    CHECK(ds4_gpu_model_source_bind(map, bytes, DS4_MSRC_ROLE_PRIMARY, -1,
        DS4_RESIDENCY_EAGER_DEVICE, "base", "GVQ Q4 test") >= 0);
    CHECK(ds4_gpu_set_model_map_range(map, bytes, 0, bytes, wb));
    ds4_unit_tensor_in tensor = {ALIGNMENT, wb, DS4_TCAT_ROUTED_EXPERT, 1};
    ds4_unit_compile_params params = {0};
    params.max_unit_bytes = bytes; params.vmm_granularity = 256;
    params.device_promote = 1; params.promote_experts = 1;
    ds4_phys_unit unit;
    uint64_t populated = 0;
    CHECK(ds4_units_compile(&tensor, 1, &params, &unit) == 1);
    CHECK(!ds4_gpu_model_units_bind(map, &unit, 1));
    CHECK(ds4_gpu_materialize_model_plan(map, bytes, &populated));
    const ds4_gvq_record record = {ALIGNMENT, wb, Q4_TYPE, OUTPUTS, cols,
        DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK};
    ds4_gvq_model *model = ds4_gpu_gvq_bind(map, bytes, &record, 1);
    const size_t xb = (size_t)MAX_TOKENS * cols * sizeof(float);
    ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(xb), *out = ds4_gpu_tensor_alloc(xb);
    CHECK(model && input && out && ds4_gpu_tensor_write(input, 0, x, xb));
    const unsigned widths[] = {1, 8, 9, 32, MAX_TOKENS};
    for (unsigned mode = 0; mode < 2; mode++) {
        CHECK(!setenv("DS4_GLMVQ_Q4_DIRECT", mode ? "1" : "0", 1));
        const uint64_t scratch = ds4_gpu_gvq_linear_bytes(OUTPUTS, cols, MAX_TOKENS, Q4_TYPE, DS4_GVQ_BF16);
        ds4_gvq_ctx *ctx = ds4_gpu_gvq_create(model, scratch, MAX_TOKENS);
        CHECK(ctx);
        for (unsigned j = 0; j < sizeof(widths) / sizeof(widths[0]); j++) {
            const unsigned tokens = widths[j];
            CHECK(ds4_gpu_gvq_linear(ctx, out, 0, input, tokens, DS4_GVQ_BF16));
            CHECK(ds4_gpu_tensor_read(out, 0, got, (size_t)tokens * OUTPUTS * sizeof(float)));
            double max_ratio = 0;
            for (unsigned t = 0; t < tokens; t++) {
                for (unsigned row = 0; row < OUTPUTS; row++) {
                    double want = 0, absolute = 0;
                    for (unsigned k = 0; k < cols; k++) {
                        const double product = (double)w[(size_t)row * cols + k] * x[(size_t)t * cols + k];
                        want += product; absolute += fabs(product);
                    }
                    const double u = 0x1p-24, ops = cols + 32;
                    const double half_ulp = exp2(floor(log2(fmax(fabs(want), 0x1p-126))) - 8);
                    const double bound = ops * u / (1 - ops * u) * absolute + half_ulp + 0x1p-126;
                    const double error = fabs(got[(size_t)t * OUTPUTS + row] - want);
                    CHECK(isfinite(got[(size_t)t * OUTPUTS + row]) && error <= bound);
                    max_ratio = fmax(max_ratio, error / bound);
                }
            }
            printf("GVQ Q4 cols=%u tokens=%u direct=%u bound_ratio=%.6f\n", cols, tokens, mode, max_ratio);
        }
        ds4_gpu_gvq_free(ctx);
    }
    /* Table gather uses the generic embedding entry, followed by the GLM
     * BF16 boundary. Repeated/nonconsecutive rows catch stride mistakes. */
    int32_t ids[MAX_TOKENS];
    for (unsigned t = 0; t < MAX_TOKENS; t++) { ids[t] = (t * 37) % OUTPUTS; }
    ds4_gpu_tensor *tokens = ds4_gpu_tensor_alloc(sizeof(ids));
    CHECK(tokens && ds4_gpu_tensor_write(tokens, 0, ids, sizeof(ids)));
    CHECK(ds4_gpu_embed_tokens_quant_tensor(out, tokens, map, bytes, ALIGNMENT,
        Q4_TYPE, OUTPUTS, MAX_TOKENS, cols));
    CHECK(ds4_gpu_gvq_round(out, (size_t)MAX_TOKENS * cols));
    CHECK(ds4_gpu_tensor_read(out, 0, got, xb));
    for (unsigned t = 0; t < MAX_TOKENS; t++) {
        CHECK(!memcmp(got + (size_t)t * cols, w + (size_t)ids[t] * cols, cols * sizeof(float)));
    }
    ds4_gpu_tensor_free(tokens); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(input);
    ds4_gpu_gvq_unbind(model);
    ds4_gpu_unregister_model_map(map);
    munmap(map, bytes); free(got); free(x); free(w);
}

int main(void) {
    CHECK(ds4_gpu_init());
    run(512); run(4096);
    ds4_gpu_cleanup();
    puts("GLMVQ_Q4_ADAPTER_PASS");
    return 0;
}
