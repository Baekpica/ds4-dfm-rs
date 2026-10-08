/* The real Prefill coordinator with mocked arithmetic and GPU execution.
 * This checks absolute positions, chunk boundaries and transient-I/O unwind;
 * actual KDA/DSA values, KV bytes and bank admission have separate gates. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ROW_CAP = 2048u, WINDOW = 4096u, MAX_ROWS = WINDOW + 1u,
       CONTEXT = WINDOW * 4u, DS4_N_VOCAB = 1024u, DS4_N_EMBD = 4u, MAX_CALLS = 3u };
enum { GLM53_PREFILL_DEFAULT = ROW_CAP, GLM53_PREFILL_MAX = ROW_CAP,
       GLM53_PREFILL_WINDOW = WINDOW, GLM53_DENSE_MIN_ROWS = 128u, DS4_N_EXPERT = 288u };
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GLM prefill FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

typedef struct { uint32_t id; } ds4_gpu_tensor;
typedef struct { uint32_t unused; } ds4_model;
typedef struct { uint32_t unused; } ds4_weights;
typedef struct { bool transient; uint64_t hot; uint32_t finishes; } ds4_glm53_stream;
typedef struct {
    bool ready;
    uint32_t cache_len, ctx_cap, row_cap, window_cap, last_rows;
    ds4_glm53_stream *stream;
    ds4_gpu_tensor *hc_cur, *hc_next;
} ds4_glm53_gpu_graph;
typedef struct { uint32_t unused; } glm53_stream_stats;
enum call_kind { CALL_WINDOW, CALL_BATCH };
enum failure { FAIL_NONE, FAIL_WINDOW, FAIL_TAIL, FAIL_BIND };
typedef struct { enum call_kind kind; uint32_t pos, rows; } call;
static call calls[MAX_CALLS];
static uint32_t call_count, bind_count, start_pos, total_rows;
static enum failure failure;
static int tokens[MAX_ROWS];
static float embeddings[MAX_ROWS * DS4_N_EMBD], logits[DS4_N_VOCAB];

static glm53_stream_stats glm53_stream_snap(const ds4_glm53_stream *s) {
    CHECK(s);
    return (glm53_stream_stats){0};
}

static void glm53_stream_delta(const ds4_glm53_stream *s,
        glm53_stream_stats before, uint32_t pos, uint32_t rows) {
    (void)before;
    CHECK(s && pos == start_pos && rows == total_rows);
}

static void glm53_stream_finish(ds4_glm53_stream *s) {
    CHECK(s);
    s->transient = false;
    s->finishes++;
}

static void glm53_stream_hot_begin(ds4_glm53_stream *s, uint32_t rows) {
    CHECK(s && rows);
}

static void glm53_stream_hot_end(ds4_glm53_stream *s) {
    CHECK(s);
}

static void record(ds4_glm53_gpu_graph *g, enum call_kind kind,
        const int *input, uint32_t rows, uint32_t pos, const float *embeds, float *out) {
    CHECK(call_count < MAX_CALLS && pos >= start_pos && g->cache_len == pos);
    CHECK(input == tokens + pos - start_pos && rows && rows <= total_rows - (pos - start_pos));
    CHECK(embeds == embeddings + (uint64_t)(pos - start_pos) * DS4_N_EMBD);
    CHECK(out == (pos - start_pos + rows == total_rows ? logits : NULL));
    calls[call_count++] = (call){kind, pos, rows};
}

static bool glm53_graph_window(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_weights *w, const int *input, uint32_t rows, uint32_t pos,
        const float *embeds, float *out) {
    CHECK(m && w && rows > g->row_cap && rows <= WINDOW - pos % WINDOW);
    record(g, CALL_WINDOW, input, rows, pos, embeds, out);
    g->stream->transient = true;
    /* A failing helper may have changed only part of the layer state. Its
     * caller must invalidate the frontier without relying on helper cleanup. */
    if (failure == FAIL_WINDOW) { return false; }
    g->cache_len = pos + rows;
    g->last_rows = (rows - 1u) % g->row_cap + 1u;
    return true;
}

static bool glm53_graph_bind(ds4_glm53_gpu_graph *active,
        const ds4_glm53_gpu_graph *owner, uint32_t rows) {
    CHECK(rows && rows <= owner->row_cap);
    bind_count++;
    if (failure == FAIL_BIND) { return false; }
    *active = *owner;
    return true;
}

static void glm53_graph_unbind(ds4_glm53_gpu_graph *g) {
    memset(g, 0, sizeof(*g));
}

static bool glm53_graph_batch(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_weights *w, const int *input, uint32_t rows, uint32_t pos,
        const float *embeds, float *out) {
    CHECK(m && w && rows <= g->row_cap);
    record(g, CALL_BATCH, input, rows, pos, embeds, out);
    if (failure == FAIL_TAIL) { return false; }
    /* Model the odd-layer residual swap so the coordinator must publish it. */
    ds4_gpu_tensor *swap = g->hc_cur;
    g->hc_cur = g->hc_next;
    g->hc_next = swap;
    return true;
}

static void *ds4_gpu_tensor_ptr(const ds4_gpu_tensor *t) {
    return (void *)t;
}

#include GLM53_PREFILL_FIXTURE

static bool run_prefill(uint32_t pos, uint32_t rows, uint32_t window,
        enum failure fault, ds4_glm53_gpu_graph *g, ds4_glm53_stream *s) {
    static ds4_gpu_tensor cur = {1u}, next = {2u};
    static const uint64_t HOT = UINT64_C(0x1234);
    *s = (ds4_glm53_stream){.hot = HOT};
    *g = (ds4_glm53_gpu_graph){.ready = true, .cache_len = pos, .ctx_cap = CONTEXT,
        .row_cap = ROW_CAP, .window_cap = window, .stream = s, .hc_cur = &cur, .hc_next = &next};
    failure = fault;
    call_count = bind_count = 0u;
    start_pos = pos;
    total_rows = rows;
    for (uint32_t i = 0u; i < rows; i++) { tokens[i] = (int)(i % DS4_N_VOCAB); }
    const ds4_model model = {0};
    const ds4_weights weights = {0};
    const bool ok = glm53_graph_prefill(g, &model, &weights, tokens, rows, pos, embeddings, logits);
    CHECK(s->hot == HOT);
    return ok;
}

static void check_unwind(enum failure fault) {
    ds4_glm53_gpu_graph g;
    ds4_glm53_stream s;
    CHECK(!run_prefill(0u, MAX_ROWS, WINDOW, fault, &g, &s));
    CHECK(!s.transient && s.finishes == 1u);
    if (fault == FAIL_TAIL || fault == FAIL_WINDOW) { CHECK(g.cache_len == UINT32_MAX); }
    else { CHECK(g.cache_len != MAX_ROWS); }
    CHECK(calls[0].kind == CALL_WINDOW && calls[0].pos == 0u && calls[0].rows == WINDOW);
    if (fault == FAIL_TAIL) {
        CHECK(call_count == 2u && calls[1].kind == CALL_BATCH);
        CHECK(calls[1].pos == WINDOW && calls[1].rows == 1u);
    } else { CHECK(call_count == 1u); }
    printf("GLM prefill: failure=%u releases window staging without committing final frontier\n", fault);
}

typedef struct { uint32_t pos, rows, window, count; call expected[MAX_CALLS]; } split_case;
static void check_alignment(void) {
    const split_case cases[] = {
        {0u, ROW_CAP, WINDOW, 1u, {{CALL_BATCH, 0u, ROW_CAP}}},
        {0u, ROW_CAP + 1u, WINDOW, 1u, {{CALL_WINDOW, 0u, ROW_CAP + 1u}}},
        {0u, WINDOW - 1u, WINDOW, 1u, {{CALL_WINDOW, 0u, WINDOW - 1u}}},
        {0u, WINDOW, WINDOW, 1u, {{CALL_WINDOW, 0u, WINDOW}}},
        {0u, MAX_ROWS, WINDOW, 2u, {{CALL_WINDOW, 0u, WINDOW}, {CALL_BATCH, WINDOW, 1u}}},
        {WINDOW - 1u, MAX_ROWS, WINDOW, 2u,
            {{CALL_BATCH, WINDOW - 1u, 1u}, {CALL_WINDOW, WINDOW, WINDOW}}},
        {WINDOW, MAX_ROWS, WINDOW, 2u,
            {{CALL_WINDOW, WINDOW, WINDOW}, {CALL_BATCH, WINDOW * 2u, 1u}}},
        {WINDOW + 1u, MAX_ROWS, WINDOW, 2u,
            {{CALL_WINDOW, WINDOW + 1u, WINDOW - 1u}, {CALL_BATCH, WINDOW * 2u, 2u}}},
        /* Bank creation qualifies disabling the window separately. Here the
         * coordinator must honor an already-disabled effective window. */
        {0u, MAX_ROWS, 0u, 3u,
            {{CALL_BATCH, 0u, ROW_CAP}, {CALL_BATCH, ROW_CAP, ROW_CAP}, {CALL_BATCH, WINDOW, 1u}}},
    };
    for (uint32_t i = 0u; i < sizeof(cases) / sizeof(*cases); i++) {
        const split_case *c = &cases[i];
        ds4_glm53_gpu_graph g;
        ds4_glm53_stream s;
        CHECK(run_prefill(c->pos, c->rows, c->window, FAIL_NONE, &g, &s));
        CHECK(g.cache_len == c->pos + c->rows && g.last_rows && g.last_rows <= ROW_CAP);
        CHECK(call_count == c->count && !s.transient);
        uint32_t windows = 0u, batches = 0u;
        for (uint32_t n = 0u; n < c->count; n++) {
            CHECK(calls[n].kind == c->expected[n].kind && calls[n].pos == c->expected[n].pos);
            CHECK(calls[n].rows == c->expected[n].rows);
            windows += calls[n].kind == CALL_WINDOW;
            batches += calls[n].kind == CALL_BATCH;
        }
        CHECK(s.finishes == (windows ? 1u : 0u) && bind_count == batches);
        CHECK(g.hc_cur->id == (batches % 2u ? 2u : 1u));
    }
    puts("GLM prefill: 2048+1, 4095/4096/4097 alignment and disabled windows passed");
}

static void check_caps(void) {
    CHECK(unsetenv("DS4_GLM53_PREFILL_ROWS") == 0);
    CHECK(unsetenv("DS4_GLM53_PREFILL_WINDOW") == 0);
    const uint32_t slots = 2u * DS4_N_EXPERT;
    CHECK(glm53_window_cap(CONTEXT, slots) == WINDOW);
    CHECK(glm53_window_cap(WINDOW - 1u, slots) == 0u);
    CHECK(glm53_window_cap(CONTEXT, slots - 1u) == 0u);
    CHECK(setenv("DS4_GLM53_PREFILL_ROWS", "127", 1) == 0);
    CHECK(glm53_window_cap(CONTEXT, slots) == 0u);
    CHECK(setenv("DS4_GLM53_PREFILL_ROWS", "128", 1) == 0);
    CHECK(glm53_window_cap(CONTEXT, slots) == WINDOW);
    CHECK(setenv("DS4_GLM53_PREFILL_WINDOW", "0", 1) == 0);
    CHECK(glm53_window_cap(CONTEXT, slots) == 0u);
    puts("GLM prefill: default window preserves small-cache, context and row fallbacks");
}

int main(int argc, char **argv) {
    CHECK(argc == 2);
    if (!strcmp(argv[1], "tail")) { check_unwind(FAIL_TAIL); return 0; }
    if (!strcmp(argv[1], "window")) { check_unwind(FAIL_WINDOW); return 0; }
    if (!strcmp(argv[1], "bind")) { check_unwind(FAIL_BIND); return 0; }
    if (!strcmp(argv[1], "alignment")) { check_alignment(); return 0; }
    if (!strcmp(argv[1], "caps")) { check_caps(); return 0; }
    CHECK(false);
    return 1;
}
