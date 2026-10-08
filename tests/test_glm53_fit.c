/* Run the actual bank admission/retry coordinator without model/GPU loads.
 * Allocation failures are injected after a successful memory estimate;
 * graph construction and GPU memory estimates remain fixture boundaries. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ROWS = 2048u, WINDOW = 4096u, CONTEXT = 8192u, MAX_BANKS = 2u,
       GLM53_PREFILL_DEFAULT = ROWS, GLM53_PREFILL_MAX = ROWS,
       GLM53_PREFILL_WINDOW = WINDOW, GLM53_DENSE_MIN_ROWS = 128u,
       DS4_N_EXPERT = 288u };
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GLM bank fit FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

typedef enum { GLM53_STRICT_BANKS, GLM53_FIT_BANKS } glm53_fit_mode;
typedef enum { GLM53_BANK_MTP_OFF, GLM53_BANK_MTP_ON } glm53_bank_mtp;
typedef struct { uint32_t count; } ds4_glm53_stream;
typedef struct { ds4_glm53_stream glm53_stream; bool vision_ready; } ds4_engine;
typedef struct { void *checkpoint_slab; uint32_t prefill_cap; } ds4_glm53_batch_runtime;
typedef struct {
    ds4_engine *e;
    uint32_t ctx_size, prefill_cap, raw_cap, seq_cap, max_seq;
    uint64_t serial_reserve;
    ds4_glm53_batch_runtime *glm53;
    int *bank_hist;
    uint32_t *bank_hist_len;
    uint8_t *bank_hist_valid;
    uint64_t *bank_gen, *bank_last_use;
    bool supports_partial_reuse, mtp;
} ds4_batch_ctx;
typedef struct { uint64_t banks_total; } metrics;
static metrics observed;
static uint32_t fit_banks, attempts[MAX_BANKS], calls;

static void *xcalloc(size_t count, size_t size) {
    void *ptr = calloc(count, size);
    CHECK(ptr);
    return ptr;
}

static void *xmalloc(size_t size) {
    void *ptr = malloc(size);
    CHECK(ptr);
    return ptr;
}

static bool ds4_engine_has_mtp(const ds4_engine *e) { return e && false; }
static uint64_t glm53_bank_bytes_for(uint32_t ctx, uint32_t slots,
        uint32_t banks, glm53_bank_mtp mtp) {
    (void)slots; (void)mtp;
    return ctx ? banks : 0u;
}
static int ds4_gpu_mem_info(uint64_t *free_bytes, uint64_t *total_bytes) {
    *free_bytes = *total_bytes = MAX_BANKS;
    return 0;
}
static uint64_t ds4_gpu_substrate_outstanding(void) { return 0u; }
static uint64_t ds4_batch_fit_headroom_bytes(int ctx) { (void)ctx; return 0u; }
static metrics *ds4_metrics_get(void) { return &observed; }
static void ds4_metric_set(uint64_t *metric, uint64_t value) { *metric = value; }
static void glm53_batch_free(ds4_glm53_batch_runtime *rt) { free(rt); }

static uint32_t glm53_graph_row_cap(uint32_t ctx, uint32_t slots);
static uint32_t glm53_prefill_cap(uint32_t ctx, uint32_t slots);
static ds4_glm53_batch_runtime *glm53_batch_create(ds4_engine *e,
        uint32_t ctx, uint32_t banks) {
    CHECK(calls < MAX_BANKS);
    attempts[calls++] = banks;
    if (banks > fit_banks) { return NULL; }
    ds4_glm53_batch_runtime *rt = xcalloc(1u, sizeof(*rt));
    rt->prefill_cap = banks == 1u ? glm53_prefill_cap(ctx, e->glm53_stream.count)
        : glm53_graph_row_cap(ctx, e->glm53_stream.count);
    return rt;
}

#include GLM53_FIT_FIXTURE

static void release(ds4_batch_ctx *ctx) {
    if (!ctx) { return; }
    glm53_batch_free(ctx->glm53);
    free(ctx->bank_hist); free(ctx->bank_hist_len); free(ctx->bank_hist_valid);
    free(ctx->bank_gen); free(ctx->bank_last_use); free(ctx);
}

int main(int argc, char **argv) {
    CHECK(argc == 2);
    const char *name = argv[1];
    CHECK(setenv("DS4_GLM53_PREFILL_ROWS", "2048", 1) == 0);
    if (strcmp(name, "policy-retry") != 0) {
        CHECK(setenv("DS4_GLM53_PREFILL_WINDOW", "4096", 1) == 0);
    }
    ds4_engine engine = {.glm53_stream = {.count = 2u * DS4_N_EXPERT}};
    glm53_fit_mode mode = GLM53_FIT_BANKS;
    uint32_t requested = MAX_BANKS;
    uint32_t expected_cap = WINDOW, expected_calls = MAX_BANKS;
    fit_banks = 1u;
    if (strcmp(name, "two") == 0) {
        fit_banks = MAX_BANKS; expected_cap = ROWS; expected_calls = 1u;
    } else if (strcmp(name, "one") == 0) {
        requested = 1u; expected_calls = 1u;
    } else if (strcmp(name, "disabled") == 0) {
        CHECK(setenv("DS4_GLM53_PREFILL_WINDOW", "0", 1) == 0);
        expected_cap = ROWS;
    } else if (strcmp(name, "small") == 0) {
        engine.glm53_stream.count = 8u; expected_cap = ROWS;
    } else if (strcmp(name, "strict") == 0) {
        mode = GLM53_STRICT_BANKS; expected_calls = 1u;
    } else if (strcmp(name, "fail") == 0) {
        fit_banks = 0u;
    } else {
        CHECK(strcmp(name, "retry") == 0 || strcmp(name, "policy-retry") == 0);
    }

    ds4_batch_ctx *ctx = NULL;
    char error[256] = {0};
    const int rc = glm53_batch_ctx_create(&engine, CONTEXT, (int)requested,
        mode, &ctx, error, sizeof(error));
    CHECK(calls == expected_calls);
    CHECK(attempts[0] == requested);
    if (mode == GLM53_STRICT_BANKS || !fit_banks) {
        CHECK(rc != 0 && !ctx && strstr(error, "allocation failed"));
    } else {
        CHECK(rc == 0 && ctx);
        CHECK(ctx->max_seq == fit_banks);
        CHECK(ctx->ctx_size == CONTEXT && ctx->seq_cap == CONTEXT);
        CHECK(ctx->prefill_cap == expected_cap);
        CHECK(ctx->prefill_cap == ctx->glm53->prefill_cap);
        CHECK(observed.banks_total == ctx->max_seq);
    }
    release(ctx);
    printf("GLM bank fit %s PASS\n", name);
    return 0;
}
