/* The Python driver extracts the production stage and run-images functions.
 * Host tensors prove transfer bytes and resource lifetime, not GPU arithmetic. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DS4_N_EMBD 8u
#define STAGE_BYTES (8ull << 20)
#define FIXTURE_LAYERS 2u
#define FIXTURE_IMAGES 2u

typedef struct { uint64_t bytes; const void *data; } ds4_tensor;
typedef struct { int unused; } ds4_model;
typedef struct { uint64_t bytes; uint8_t *data; } ds4_gpu_tensor;
typedef struct {
    const ds4_tensor *norm1, *qkv_w, *qkv_b, *o_w, *o_b;
    const ds4_tensor *norm2, *up_w, *down_w;
} ds4_v41_vision_block_weights;
typedef struct {
    const ds4_tensor *patch_w, *patch_b, *out_norm;
    const ds4_tensor *mm0_w, *mm0_b, *mm1_w, *mm1_b;
    ds4_v41_vision_block_weights blk[FIXTURE_LAYERS];
} ds4_v41_vision_weights;
typedef struct {
    struct { ds4_v41_vision_weights v41_vision; } weights;
    ds4_model model, vision_model;
} ds4_engine;
typedef struct {
    uint32_t vis_layers, vis_dim, vis_heads, vis_inter, vis_ds, vis_patch;
    float vis_theta;
} ds4_v41_cfg;
typedef struct {
    float *patches;
    uint32_t n_vit_h, n_vit_w, n_llm_h, n_llm_w;
} v41_image;
typedef struct {
    ds4_gpu_tensor *x, *al;
    uint32_t n, nh, nw, n_llm;
} v41_vact;
struct ds4_v41_images {
    uint32_t n;
    v41_image *img;
    uint32_t n_rows;
};

static ds4_v41_cfg g_ds4_v41 = {FIXTURE_LAYERS, 4, 1, 8, 1, 1, 10000.f};
static struct {
    uint32_t pin_attempts, pin_allocs, pin_frees, clock_calls, drop_calls;
    uint32_t writes, pinned_writes, direct_weights;
    uint64_t pin_bytes, max_write;
    bool pin_after_clock;
} stats;
static void *pin;
static const void *weight_source;
static uint32_t live_tensors, fail_write;
static bool fail_pin;

static void *ds4_gpu_host_alloc(uint64_t bytes) {
    stats.pin_attempts++;
    stats.pin_bytes = bytes;
    stats.pin_after_clock = stats.clock_calls > 0;
    if (pin) { abort(); }
    if (fail_pin) { return NULL; }
    pin = malloc((size_t)bytes);
    if (!pin) { abort(); }
    stats.pin_allocs++;
    return pin;
}

static void ds4_gpu_host_free(void *p) {
    if (!p || p != pin) { abort(); }
    free(pin);
    pin = NULL;
    stats.pin_frees++;
}

static ds4_gpu_tensor *v41_alloc(uint64_t bytes, bool *ok) {
    ds4_gpu_tensor *t = calloc(1, sizeof *t);
    if (!t) { abort(); }
    t->bytes = bytes;
    t->data = calloc((size_t)bytes, 1);
    if (!t->data) { abort(); }
    live_tensors++;
    (void)ok;
    return t;
}

static void ds4_gpu_tensor_free(ds4_gpu_tensor *t) {
    if (!t) { return; }
    free(t->data);
    free(t);
    live_tensors--;
}

static uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t->bytes; }

static int ds4_gpu_tensor_write(ds4_gpu_tensor *t, uint64_t offset,
                                const void *source, uint64_t bytes) {
    if (!t || offset > t->bytes || bytes > t->bytes - offset) { abort(); }
    stats.writes++;
    if (bytes > stats.max_write) { stats.max_write = bytes; }
    if (pin && (uintptr_t)source >= (uintptr_t)pin &&
        (uintptr_t)source - (uintptr_t)pin < stats.pin_bytes) {
        if (bytes > stats.pin_bytes - ((uintptr_t)source - (uintptr_t)pin)) { abort(); }
        stats.pinned_writes++;
    } else if (source == weight_source) {
        stats.direct_weights++;
    }
    if (fail_write && stats.writes == fail_write) { return 0; }
    memcpy(t->data + offset, source, (size_t)bytes);
    return 1;
}

static const void *tensor_data(const ds4_model *vm, const ds4_tensor *t) {
    (void)vm;
    return t->data;
}

static double now_sec(void) { return ++stats.clock_calls; }
static int ds4_gpu_synchronize(void) { return 1; }
static void v41_vis_drop_pages(const ds4_model *vm) {
    (void)vm;
    stats.drop_calls++;
}

/* Arithmetic is outside this host transfer/lifetime fixture. */
#define ds4_gpu_v41_matmul_bf16_dev_tensor(...) (1)
#define ds4_gpu_v41_vision_bias_add_tensor(...) (1)
#define ds4_gpu_v41_round_bf16_tensor(...) (1)
#define ds4_gpu_v41_add_tensor(...) (1)
#define ds4_gpu_v41_rms_norm_dev_tensor(...) (1)
#define ds4_gpu_v41_vision_rope_tensor(...) (1)
#define ds4_gpu_v41_dense_attn_tensor(...) (1)
#define ds4_gpu_v41_swiglu_pair_tensor(...) (1)
#define ds4_gpu_v41_vision_unfold_tensor(...) (1)
#define ds4_gpu_v41_gelu_tensor(...) (1)
#include DS41_VISION_STAGE_FIXTURE

static void reset_stats(void) {
    if (pin || live_tensors) { abort(); }
    memset(&stats, 0, sizeof stats);
    fail_write = 0;
    fail_pin = false;
    weight_source = NULL;
}

enum check_result { CHECK_FAIL, CHECK_PASS };

static int check(const char *name, enum check_result result) {
    printf("%s: %s\n", result == CHECK_PASS ? "PASS" : "FAIL", name);
    return result == CHECK_PASS ? 0 : 1;
}

static int test_copy(void) {
    reset_stats();
    const uint64_t bytes = 2u * STAGE_BYTES + 137u;
    uint8_t *source = malloc((size_t)bytes);
    if (!source) { abort(); }
    for (uint64_t i = 0; i < bytes; i++) {
        source[i] = (uint8_t)((i * 131u) ^ (i >> 17) ^ (i >> 23));
    }
    const ds4_tensor tensor = {bytes, source};
    ds4_gpu_tensor *slot = NULL;
    ds4_model model = {0};
    void *stage = ds4_gpu_host_alloc(STAGE_BYTES);
    bool ok = test_stage(&model, &tensor, &slot, stage);
    int failures = check("staged multi-chunk/tail bytes match source",
        ok && slot && memcmp(source, slot->data, (size_t)bytes) == 0);
    failures += check("staged copy uses only bounded pinned writes",
        stats.writes == 3 && stats.pinned_writes == 3 && stats.max_write <= STAGE_BYTES);
    ds4_gpu_tensor_free(slot);
    ds4_gpu_host_free(stage);

    reset_stats();
    slot = NULL;
    ok = test_stage(&model, &tensor, &slot, NULL);
    failures += check("NULL staging retains direct byte-exact transfer",
        ok && slot && stats.writes == 1 && stats.pin_attempts == 0 &&
        memcmp(source, slot->data, (size_t)bytes) == 0);
    ds4_gpu_tensor_free(slot);

    reset_stats();
    slot = NULL;
    stage = ds4_gpu_host_alloc(STAGE_BYTES);
    fail_write = 2;
    ok = test_stage(&model, &tensor, &slot, stage);
    failures += check("failed chunk aborts the transfer",
        !ok && stats.writes == 2);
    ds4_gpu_tensor_free(slot);
    ds4_gpu_host_free(stage);
    free(source);
    return failures;
}

static ds4_engine fixture(const ds4_tensor *weight) {
    ds4_engine e = {0};
    ds4_v41_vision_weights *v = &e.weights.v41_vision;
    v->patch_w = v->patch_b = v->out_norm = weight;
    v->mm0_w = v->mm0_b = v->mm1_w = v->mm1_b = weight;
    for (uint32_t i = 0; i < FIXTURE_LAYERS; i++) {
        ds4_v41_vision_block_weights *b = &v->blk[i];
        b->norm1 = b->qkv_w = b->qkv_b = b->o_w = b->o_b = weight;
        b->norm2 = b->up_w = b->down_w = weight;
    }
    weight_source = weight->data;
    return e;
}

enum run_result { RUN_OK, RUN_FAIL };

static bool run_images(ds4_engine *e, enum run_result expected) {
    float patches[2][12] = {{0}, {0}};
    v41_image images[FIXTURE_IMAGES] = {
        {patches[0], 2, 2, 2, 2}, {patches[1], 1, 2, 1, 2}};
    v41_vact acts[FIXTURE_IMAGES] = {{0}};
    const bool success = expected == RUN_OK;
    bool ok = v41_vis_run_images(e, images, FIXTURE_IMAGES, acts) == success;
    for (uint32_t i = 0; i < FIXTURE_IMAGES; i++) {
        ok = ok && !acts[i].x && (success ? acts[i].al != NULL : acts[i].al == NULL);
        ds4_gpu_tensor_free(acts[i].al);
    }
    return ok && live_tensors == 0;
}

static int test_requests(void) {
    const uint8_t data[64] = {1, 2, 3, 4};
    const ds4_tensor weight = {sizeof data, data};
    reset_stats();
    unsetenv("DS41_VISION_PINNED");
    ds4_engine e = fixture(&weight);
    int failures = check("two-image tower succeeds and frees temporary tensors",
        run_images(&e, RUN_OK));
    failures += check("one request owns one 8MiB pinned window",
        stats.pin_attempts == 1 && stats.pin_allocs == 1 &&
        stats.pin_bytes == STAGE_BYTES && stats.pin_frees == 1 && !pin);
    failures += check("pin allocation is included in tower timing", stats.pin_after_clock);
    failures += check("all tower weights use the request's pinned window",
        stats.pinned_writes > 0 && stats.direct_weights == 0);

    reset_stats();
    e = fixture(&weight);
    fail_pin = true;
    failures += check("pin allocation failure retains a working direct path",
        run_images(&e, RUN_OK) && stats.pin_attempts == 1 && !stats.pin_allocs &&
        !stats.pin_frees && !pin && stats.direct_weights > 0 && !stats.pinned_writes);

    reset_stats();
    e = fixture(&weight);
    fail_write = 2;
    failures += check("tower write failure frees outputs and pinned storage",
        run_images(&e, RUN_FAIL) && stats.pin_allocs == 1 && stats.pin_frees == 1 &&
        !pin && stats.drop_calls == 1);

    reset_stats();
    e = fixture(&weight);
    bool repeated = true;
    for (uint32_t i = 1; i <= 3; i++) {
        repeated = run_images(&e, RUN_OK) && repeated && !pin &&
            stats.pin_allocs == i && stats.pin_frees == i;
    }
    failures += check("repeated requests release each pinned window", repeated);

    reset_stats();
    e = fixture(&weight);
    setenv("DS41_VISION_PINNED", "0", 1);
    failures += check("disabled staging never allocates pinned memory",
        run_images(&e, RUN_OK) && !stats.pin_attempts && !stats.pin_frees &&
        !stats.pinned_writes && stats.direct_weights > 0);
    setenv("DS41_VISION_PINNED", "1", 1);
    failures += check("next request can re-enable bounded staging",
        run_images(&e, RUN_OK) && stats.pin_allocs == 1 && stats.pin_frees == 1 && !pin);
    unsetenv("DS41_VISION_PINNED");
    return failures;
}

static int test_admission(void) {
    reset_stats();
    enum { FIRST_ROWS = 8, SECOND_ROWS = 5 };
    v41_image images[FIXTURE_IMAGES] = {
        {NULL, 2, 2, 2, 2}, {NULL, 1, 2, 1, 2}};
    const struct ds4_v41_images one = {1, images, FIRST_ROWS};
    const struct ds4_v41_images two = {2, images, FIRST_ROWS + SECOND_ROWS};
    const uint64_t row_bytes = DS4_N_EMBD * sizeof(float);
    /* The 2x2 fixture has 1296 scratch, 192 residual/output and 512 weight
     * bytes. The smaller image adds 96 residual/output bytes and five rows. */
    const uint64_t one_bytes = (192ull << 20) + 1296u + 192u + 512u +
        FIRST_ROWS * row_bytes + STAGE_BYTES;
    const uint64_t two_bytes = one_bytes + 96u + SECOND_ROWS * row_bytes;
    int failures = check("admission charges one window per image request, none for NULL",
        ds4_v41_images_bytes(NULL) == 0 && ds4_v41_images_bytes(&one) == one_bytes &&
        ds4_v41_images_bytes(&two) == two_bytes);

    setenv("DS41_VISION_PINNED", "0", 1);
    const uint64_t disabled_bytes = ds4_v41_images_bytes(&two);
    setenv("DS41_VISION_PINNED", "1", 1);
    const uint8_t data[64] = {1, 2, 3, 4};
    const ds4_tensor weight = {sizeof data, data};
    ds4_engine e = fixture(&weight);
    fail_pin = true;
    failures += check("disabled and allocation-fallback budgets remain conservative",
        run_images(&e, RUN_OK) && stats.pin_attempts == 1 && !stats.pin_allocs &&
        disabled_bytes == two_bytes && ds4_v41_images_bytes(&two) == two_bytes);
    unsetenv("DS41_VISION_PINNED");
    return failures;
}

int main(void) {
    int failures = test_copy();
    failures += test_requests();
    failures += test_admission();
    printf("vision staging: %d failure(s), no GPU/model loaded\n", failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
