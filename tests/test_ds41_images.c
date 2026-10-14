/* Actual bridge and native image-set builder; inference is a bounded stub.
 * Invalid views must fail before native copies or tower reads can begin. */
#include <limits.h>
#include <sys/resource.h>
#include <sys/wait.h>

#include "../native/bridge/ds4_bridge.c"
#include "../ds4_gpu.h"

#define DS4_N_EMBD 8u

typedef struct {
    float *patches;
    uint32_t n_vit_h, n_vit_w, n_llm_h, n_llm_w, n_tok, row0, start;
    uint64_t sig;
} v41_image;

struct ds4_v41_images {
    uint32_t n;
    v41_image *img;
    uint32_t n_rows, np;
    int32_t *row_of;
    uint64_t *sig_of;
    uint32_t row_base;
    ds4_gpu_tensor *embd;
    int encoded;
};

typedef struct {
    uint32_t vis_layers, vis_patch, vis_ds, vis_max_tok;
} ds4_v41_cfg;

static ds4_v41_cfg g_ds4_v41 = {1u, 14u, 3u, 1024u};
static unsigned allocations, run_calls, tower_calls, tensor_frees;

static void *xmalloc(size_t bytes) {
    if (bytes > (1u << 20)) { abort(); }
    allocations++;
    void *p = malloc(bytes);
    if (!p) { abort(); }
    return p;
}

static void *xcalloc(size_t n, size_t bytes) {
    if (bytes && n > (1u << 20) / bytes) { abort(); }
    void *p = xmalloc(n * bytes);
    memset(p, 0, n * bytes);
    return p;
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *t) {
    if (t != (ds4_gpu_tensor *)1) { abort(); }
    tensor_frees++;
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *t, uint64_t offset, void *dst, uint64_t bytes) {
    if (t != (ds4_gpu_tensor *)1 || offset || bytes % sizeof(float)) { abort(); }
    for (uint64_t i = 0; i < bytes / sizeof(float); i++) { ((float *)dst)[i] = (float)i; }
    return 1;
}

typedef struct { uint32_t n_llm; ds4_gpu_tensor *al; } v41_vact;
static bool v41_vis_run_images(ds4_engine *e, const v41_image *im, uint32_t n, v41_vact *a) {
    (void)e;
    if (n != 1u) { abort(); }
    tower_calls++;
    a->n_llm = im->n_llm_h * im->n_llm_w;
    a->al = (ds4_gpu_tensor *)1;
    return true;
}

#include DS41_IMAGES_FIXTURE

int ds4_v41_engram_count(void *engine) { (void)engine; return 0; }
ds4_v41_feed *ds4_v41_feed_open(void *engine, int n_prompt,
    ds4_v41_rows_fn rows, void *rows_ud, ds4_v41_wait_fn wait, void *wait_ud,
    ds4_v41_next_fn next, void *next_ud) {
    (void)engine; (void)n_prompt; (void)rows; (void)rows_ud;
    (void)wait; (void)wait_ud; (void)next; (void)next_ud;
    abort();
}
const ds41_engram_feed *ds4_v41_feed_handle(const ds4_v41_feed *feed) {
    (void)feed;
    return NULL;
}
void ds4_v41_feed_close(ds4_v41_feed *feed) { (void)feed; }
void ds4_engine_v41_set_progress(int (*cb)(void *, const char *, int, int), void *ud) {
    (void)cb; (void)ud;
}
int ds4_v41_generate_argmax(void *engine, const int *prompt, int n_prompt,
    int n_predict, int no_engram, int verify_k, struct ds4_v41_images *imgs,
    const ds41_engram_feed *feed, int (*emit)(int, void *), void *emit_ud) {
    (void)engine; (void)prompt; (void)n_prompt; (void)n_predict;
    (void)no_engram; (void)verify_k; (void)imgs; (void)feed; (void)emit; (void)emit_ud;
    run_calls++;
    return 0;
}

enum image_case {
    VALID, VALID_ODD, VALID_MANY, PATCH_CONFIG, PATCH_SHORT, PATCH_LONG, DOWNSAMPLE, TOTAL_ROWS,
    ZERO_GRID, GRID_OVERFLOW, PATCH_OVERFLOW, NULL_PATCH, NULL_INPUT,
    NULL_ROWS, NULL_SIG, ROW_RANGE, ROW_ORDER, SPAN_SPLIT, IMAGE_SIG,
    TEXT_SIG, ZERO_SIG, NO_TOWER, NO_DOWNSAMPLE, TOO_MANY,
    BRIDGE_VALID, BRIDGE_TEXT, BRIDGE_LENGTH, BRIDGE_COUNT, BRIDGE_NULL, BRIDGE_ZERO,
    GOLDEN_VALID, GOLDEN_SHORT, GOLDEN_LONG, GOLDEN_CONFIG, GOLDEN_ZERO,
    GOLDEN_OVERFLOW, GOLDEN_NULL_OUTPUT,
};

static int run_case(enum image_case test) {
    float patches[3u * 3u * 3u * 14u * 14u] = {0};
    ds4_v41_image_src image = {
        .patches = patches, .patches_len = sizeof patches / sizeof patches[0],
        .n_vit_h = 3u, .n_vit_w = 3u, .n_llm_h = 1u, .n_llm_w = 1u,
    };
    ds4_v41_image_src many[5];
    const ds4_v41_image_src *input = &image;
    int32_t rows[22] = {-1, 0, 1, 2, 3, -1};
    uint64_t sig[22] = {0, 17, 17, 17, 17, 0};
    const int32_t *row_of = rows;
    const uint64_t *sig_of = sig;
    uint32_t n_rows = 4u, np = 6u;
    int n = 1;
    switch (test) {
        case VALID_ODD:
            image.n_vit_h = 2u; image.n_vit_w = 4u; image.n_llm_w = 2u;
            image.patches_len = 2u * 4u * 3u * 14u * 14u;
            rows[5] = 4; sig[5] = 17; n_rows = 5u;
            break;
        case VALID_MANY:
            for (unsigned i = 0; i < 5u; i++) { many[i] = image; }
            for (unsigned i = 0; i < 20u; i++) { rows[i + 1] = (int32_t)i; sig[i + 1] = 17u + i / 4u; }
            rows[21] = -1; input = many; n = 5; n_rows = 20u; np = 22u;
            break;
        case PATCH_CONFIG: image.patches_len = 3u * 3u * 3u; break;
        case PATCH_SHORT: image.patches_len--; break;
        case PATCH_LONG: image.patches_len++; break;
        case DOWNSAMPLE: image.n_llm_h = image.n_llm_w = 3u; break;
        case TOTAL_ROWS: n_rows++; break;
        case ZERO_GRID: image.n_vit_h = 0; break;
        case GRID_OVERFLOW: image.n_vit_h = image.n_vit_w = UINT32_MAX; break;
        case PATCH_OVERFLOW: g_ds4_v41.vis_patch = UINT32_MAX; break;
        case NULL_PATCH: image.patches = NULL; break;
        case NULL_INPUT: input = NULL; break;
        case NULL_ROWS: row_of = NULL; break;
        case NULL_SIG: sig_of = NULL; break;
        case ROW_RANGE: rows[4] = 4; break;
        case ROW_ORDER: rows[2] = 2; rows[3] = 1; break;
        case SPAN_SPLIT: rows[2] = -1; sig[2] = 0; rows[5] = 1; sig[5] = 17; break;
        case IMAGE_SIG: sig[3] = 18; break;
        case TEXT_SIG: sig[0] = 17; break;
        case ZERO_SIG: memset(sig, 0, sizeof sig); break;
        case NO_TOWER: g_ds4_v41.vis_layers = 0; break;
        case NO_DOWNSAMPLE: g_ds4_v41.vis_ds = 0; break;
        case TOO_MANY: n = INT_MAX; break;
        default: break;
    }
    if (test >= GOLDEN_VALID) {
        ds4_bridge_model model = {.engine = (ds4_engine *)1};
        uint64_t count = image.patches_len;
        uint32_t h = 3u, w = 3u, oh = 23u, ow = 29u;
        uint32_t *out_h = &oh;
        if (test == GOLDEN_SHORT) { count--; }
        if (test == GOLDEN_LONG) { count++; }
        if (test == GOLDEN_CONFIG) { count = 3u * 3u * 3u; }
        if (test == GOLDEN_ZERO) { h = 0; }
        if (test == GOLDEN_OVERFLOW) { h = w = UINT32_MAX; }
        if (test == GOLDEN_NULL_OUTPUT) { out_h = NULL; }
        float *out = ds4_bridge_v41_vision_run(&model, patches, count, h, w, out_h, &ow);
        int valid = out && oh == 1u && ow == 1u && tower_calls == 1u && tensor_frees == 1u;
        if (valid) {
            for (unsigned i = 0; i < DS4_N_EMBD; i++) { valid &= out[i] == (float)i; }
        }
        const int rejected = !out && !tower_calls && !allocations && oh == 23u && ow == 29u;
        ds4_bridge_v41_vision_rows_free(out);
        return test == GOLDEN_VALID ? valid : rejected;
    }
    if (test >= BRIDGE_VALID) {
        ds4_bridge_model model = {.engine = (ds4_engine *)1};
        const int32_t prompt[] = {1, 2, 2, 2, 2, 3};
        ds4_bridge_v41_image raw = {
            .patches = patches, .patches_len = image.patches_len,
            .n_vit_h = 3u, .n_vit_w = 3u, .n_llm_h = 1u, .n_llm_w = 1u,
        };
        ds4_bridge_v41_images images = {
            .img = &raw, .n = 1, .row_of = rows, .sig_of = sig, .np = 6u, .n_rows = 4u,
        };
        if (test == BRIDGE_LENGTH) { images.np = 5u; }
        if (test == BRIDGE_COUNT) { images.n = -1; }
        if (test == BRIDGE_NULL) { images.img = NULL; }
        if (test == BRIDGE_ZERO) { images.n = 0; }
        char err[128] = {0};
        const int rc = ds4_bridge_v41_generate(&model, prompt, 6, 1, 1, 0,
            test == BRIDGE_TEXT ? NULL : &images,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, err, sizeof err);
        if (test == BRIDGE_VALID || test == BRIDGE_TEXT) { return rc == 0 && run_calls == 1u; }
        return rc != 0 && run_calls == 0u && allocations == 0u;
    }
    struct ds4_v41_images *images = ds4_v41_images_build(input, n, row_of, np, n_rows, sig_of);
    const int valid = images != NULL;
    ds4_v41_images_free(images);
    return test <= VALID_MANY ? valid : !valid && allocations == 0u;
}

int main(void) {
    const struct rlimit no_core = {0, 0};
    (void)setrlimit(RLIMIT_CORE, &no_core);
    const char *names[] = {
        "valid loaded geometry", "ceiling downsample grid", "five images",
        "wrong patch config", "short patch extent", "long patch extent",
        "wrong downsample", "wrong total rows", "zero grid", "grid overflow", "patch overflow",
        "null patches", "null image source", "null rows", "null signatures", "row out of range",
        "row order", "split span", "changed span signature", "text signature", "zero image signature",
        "no tower", "zero downsample", "image count overflow", "valid bridge", "text bridge",
        "bridge prompt length", "negative bridge count", "null bridge images", "zero bridge count",
        "valid golden extent", "short golden extent", "long golden extent", "wrong golden patch config",
        "zero golden grid", "golden grid overflow", "null golden output",
    };
    unsigned failures = 0;
    for (unsigned i = 0; i < sizeof names / sizeof names[0]; i++) {
        fflush(NULL);
        const pid_t child = fork();
        if (child < 0) { return 1; }
        if (child == 0) { _exit(run_case((enum image_case)i) ? 0 : 1); }
        int status = 0;
        if (waitpid(child, &status, 0) != child) { return 1; }
        const int ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
        printf("%s: %s\n", names[i], ok ? "PASS" : "FAIL");
        failures += !ok;
    }
    printf("ds41 native image admission: %s (%u failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
