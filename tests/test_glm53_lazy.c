/* Actual session policy and row allocation; GPU tensors and persistent
 * state are mocked at toy dimensions. No model or CUDA process is started. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { GLM53_PREFILL_DEFAULT = 2048u, GLM53_PREFILL_MAX = 2048u,
    GLM53_PREFILL_WINDOW = 4096u, GLM53_DENSE_MIN_ROWS = 128u,
    DS4_GLM53_RESIDENT_CTX_MAX = 1048576u, GLM53_DIAG_CTX_MAX = 2048u,
    DS4_MODEL_FAMILY_GLM53 = 1, DS4_MODEL_FAMILY = DS4_MODEL_FAMILY_GLM53,
    DS4_BACKEND_CUDA = 1, DS4_DISTRIBUTED_NONE = 0,
    DS4_GOVC_SERIAL_SESSION = 0, DS4_MEMC_SESSION_TENSORS = 0,
    DS4_LOG_TIMING = 0, DS4_N_EXPERT = 288u, DS4_N_EXPERT_USED = 8u,
    DS4_N_LAYER = 2u, DS4_N_NEXTN_PREDICT = 0u,
    DS4_N_HC = 4u, DS4_N_EMBD = 4u, DS4_N_VOCAB = 8u,
    DS4_N_HEAD = 1u, DS4_N_KEY_MLA = 2u, DS4_N_KDA_HEAD_DIM = 2u,
    DS4_N_SSM_CONV = 2u, DS4_N_LORA_Q = 2u, DS4_N_KV_LORA = 2u,
    DS4_N_FF_DENSE = 4u, DS4_N_FF_EXP = 2u, DS4_N_INDEXER_HEAD = 1u,
    DS4_N_INDEXER_HEAD_DIM = 2u, DS4_GLM53_POOL_SIZE = 4u,
    DS4_GLM53_INDEX_TOPK = 4u, DS4_GLM53_MAX_SELECTED = 4u,
    GLM53_MTP_SAVES = 2u, CONTEXT = 8192u, NARROW_ROWS = 1024u };
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GLM session FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)
#define ds4_log(...) ((void)0)

typedef int ds4_backend;
typedef struct { uint32_t count; } ds4_glm53_stream;
typedef struct { int unused; } ds4_model;
typedef struct { int unused; } ds4_weights;
typedef struct { uint64_t bytes; } ds4_gpu_tensor;
#include GLM53_ROW_FIXTURE
typedef struct {
    bool ready, expanded_diag;
    uint32_t ctx_cap, row_cap, window_cap, pool_cap;
    ds4_glm53_stream *stream;
    uint64_t workspace_bytes, kv_bytes, state_bytes;
#define FIELD(name, cols, type) ds4_gpu_tensor *name;
    GLM53_ROW_BUFFERS(FIELD)
#undef FIELD
    ds4_gpu_tensor *logits, *kda_scratch;
} ds4_glm53_gpu_graph;
typedef struct {
    ds4_backend backend;
    bool metal_ready, dspark_ready, glm53_mtp;
    struct { int role; } distributed;
    ds4_glm53_stream glm53_stream;
    ds4_model model;
    ds4_weights weights;
} ds4_engine;
typedef struct {
    ds4_engine *engine;
    int ctx_size;
    uint64_t generation, graph_alloc_bytes;
    uint32_t prefill_cap, glm53_rows, glm53_window;
    bool graph_pending, glm53_graph_ready;
    float *logits;
    ds4_glm53_gpu_graph glm53_graph;
} ds4_session;
typedef struct {
    int fits, fail_open;
    uint64_t need_bytes, headroom_bytes, avail_bytes, deficit_bytes;
} ds4_session_graph_fit_quote;

static uint64_t free_bytes, live_bytes, planned;
static struct { uint64_t graph_fit_refusals; } metrics;
static void *xcalloc(size_t count, size_t size) {
    void *ptr = calloc(count, size); CHECK(ptr); return ptr;
}
static void *xmalloc(size_t size) {
    void *ptr = malloc(size); CHECK(ptr); return ptr;
}
static int ds4_gpu_mem_info(uint64_t *avail, uint64_t *total) {
    *avail = *total = free_bytes; return 0;
}
static uint64_t ds4_gpu_substrate_outstanding(void) { return 0u; }
static uint64_t ds4_session_graph_headroom_bytes(void) { return 0u; }
static void ds4_metric_add(uint64_t *value, uint64_t add) { *value += add; }
#define ds4_metrics_get() (&metrics)
static void ds4_gov_publish_use(int scope, uint64_t intent, uint64_t committed) {
    (void)scope;
    if (!committed) { planned = intent; }
}
static uint64_t session_tensors_census_live(void) { return live_bytes; }
static void ds4_gpu_mem_scope_begin(int scope) { (void)scope; }
static void ds4_gpu_mem_scope_end(void) {}
static ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    ds4_gpu_tensor *t = xmalloc(sizeof(*t));
    t->bytes = bytes; live_bytes += bytes; return t;
}
static void ds4_gpu_tensor_free(ds4_gpu_tensor *t) {
    if (!t) { return; }
    live_bytes -= t->bytes; free(t);
}
static void glm53_graph_free(ds4_glm53_gpu_graph *g) {
#define FREE(name, cols, type) ds4_gpu_tensor_free(g->name);
    GLM53_ROW_BUFFERS(FREE)
#undef FREE
    ds4_gpu_tensor_free(g->logits); ds4_gpu_tensor_free(g->kda_scratch);
    memset(g, 0, sizeof(*g));
}
static bool ds4_glm53_layer_is_kda(uint32_t il) { return il == 0u; }
static uint64_t glm53_dense_reserve(uint32_t rows, uint32_t ctx) {
    (void)rows; (void)ctx; return 0u;
}
static uint64_t glm53_mtp_bytes(uint32_t ctx) { return ctx * sizeof(float); }
static bool glm53_mtp_enable(ds4_glm53_gpu_graph *g) { (void)g; return true; }
static uint64_t ds4_gpu_solar_kda_prefill_scratch_bytes(uint32_t rows,
        uint32_t heads, uint32_t dim) { return (uint64_t)rows * heads * dim * sizeof(float); }
static bool glm53_graph_state(ds4_glm53_gpu_graph *g,
        const ds4_model *m, const ds4_weights *w) { (void)g; (void)m; (void)w; return true; }
static bool glm53_graph_reset(ds4_glm53_gpu_graph *g) { return g != NULL; }
static bool ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *t, float value, uint64_t count) {
    (void)value; return t && t->bytes == count * sizeof(float);
}
static bool ds4_session_lazy_graph_enabled(void) {
    const char *v = getenv("DS4_SESSION_LAZY_GRAPH");
    return !v || strcmp(v, "0") != 0;
}

#include GLM53_SESSION_FIXTURE

static void release(ds4_session *s) {
    glm53_graph_free(&s->glm53_graph); free(s->logits); free(s);
}

int main(int argc, char **argv) {
    CHECK(argc == 2);
    const char *name = argv[1];
    const bool eager = strcmp(name, "eager") == 0;
    const bool retry = strcmp(name, "retry") == 0;
    const bool window_on = strcmp(name, "window-on") == 0;
    const bool window_off = strcmp(name, "window-off") == 0;
    const uint32_t rows = window_on || window_off ? GLM53_PREFILL_MAX : NARROW_ROWS;
    const uint32_t window = window_on ? GLM53_PREFILL_WINDOW : 0u;
    ds4_engine engine = {.backend = DS4_BACKEND_CUDA, .metal_ready = true,
        .glm53_stream.count = window_on || window_off ? 2u * DS4_N_EXPERT : 0u};
    CHECK(unsetenv("DS4_SESSION_GRAPH_FIT") == 0);
    CHECK(unsetenv("DS4_GLM53_DSA_EXPANDED") == 0);
    CHECK(setenv("DS4_SESSION_LAZY_GRAPH", eager ? "0" : "1", 1) == 0);
    CHECK(setenv("DS4_GLM53_PREFILL_ROWS", rows == NARROW_ROWS ? "1024" : "2048", 1) == 0);
    CHECK(setenv("DS4_GLM53_PREFILL_WINDOW", window ? "4096" : "0", 1) == 0);
    CHECK(glm53_prefill_cap(CONTEXT, engine.glm53_stream.count) == (window ? window : rows));
    const uint64_t intent = glm53_graph_bytes_for(CONTEXT, engine.glm53_stream.count);
    free_bytes = intent;
    ds4_session *first = NULL, *second = NULL;
    CHECK(create(&first, &engine, CONTEXT) == 0);
    CHECK(first->graph_pending == !eager);
    CHECK(first->prefill_cap == (window ? window : rows));
    if (retry) {
        free_bytes = 0u;
        CHECK(ds4_session_alloc_graph(first) != 0);
        CHECK(!first->glm53_graph_ready && live_bytes == 0u);
    }
    CHECK(setenv("DS4_GLM53_PREFILL_ROWS", "2048", 1) == 0);
    CHECK(setenv("DS4_GLM53_PREFILL_WINDOW", window_off ? "4096" : "0", 1) == 0);
    free_bytes = UINT64_MAX / 2u;
    CHECK(create(&second, &engine, GLM53_PREFILL_WINDOW) == 0);
    free_bytes = intent;
    if (!eager) { CHECK(ds4_session_alloc_graph(first) == 0); }
    CHECK(first->glm53_graph_ready);
    CHECK(planned == intent || eager);
    CHECK(first->glm53_graph.row_cap == rows);
    CHECK(first->glm53_graph.window_cap == window);
    CHECK(first->prefill_cap == (window ? window : rows));
    free_bytes = UINT64_MAX / 2u;
    if (!eager) { CHECK(ds4_session_alloc_graph(second) == 0); }
    CHECK(second->glm53_graph.row_cap == GLM53_PREFILL_MAX);
    CHECK(second->glm53_graph.window_cap == (window_off ? GLM53_PREFILL_WINDOW : 0u));
    release(first); release(second);
    CHECK(live_bytes == 0u);
    printf("GLM session %s PASS\n", name);
    return 0;
}
