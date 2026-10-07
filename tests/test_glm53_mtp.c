/* Model-free control gate. Tensor operations are bounded byte-array mocks;
 * predictor/target arithmetic is a deterministic stateful test oracle. */
#include "../ds4_gpu.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

enum {
    DS4_N_EMBD = 2, DS4_N_KV_LORA = 2, DS4_N_VOCAB = 8,
    GLM53_MTP_ROWS = 4, GLM53_MTP_SAVES = GLM53_MTP_ROWS + 1,
    FIXTURE_CTX = 32, FIXTURE_START = 5, STATE_FLOATS = 8
};
typedef struct { int unused; } ds4_model, ds4_weights;
struct ds4_gpu_tensor { unsigned char *data; uint64_t bytes; int owner; };
typedef struct {
    bool ready, mtp_ready;
    uint32_t ctx_cap, cache_len, last_rows;
    uint64_t state_bytes;
    ds4_gpu_tensor *state_pool, *last_hidden, *mtp_hidden, *logits;
    ds4_gpu_tensor *mtp_kv, *mtp_concat, *mtp_journal;
    float *mtp_logit_rows;
    struct glm53_mtp_epoch *mtp_epoch;
    uint64_t mtp_rewind_epoch;
    uint32_t mtp_len, mtp_min, mtp_trial_n, mtp_trial_start;
    uint32_t mtp_rewind_n;
    uint32_t mtp_saved_len[GLM53_MTP_SAVES], mtp_saved_min[GLM53_MTP_SAVES];
    int mtp_trial[GLM53_MTP_ROWS];
} ds4_glm53_gpu_graph;
static uint64_t allocated;
static unsigned copies, forwards, steps;
static int fail_copy;
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GLM MTP FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    ds4_gpu_tensor *t = calloc(1u, sizeof(*t));
    if (!t) { return NULL; }
    t->data = calloc(1u, bytes);
    if (!t->data) { free(t); return NULL; }
    t->bytes = bytes; t->owner = 1;
    allocated += bytes;
    return t;
}
ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *t, uint64_t off, uint64_t bytes) {
    if (!t || off > t->bytes || bytes > t->bytes - off) { return NULL; }
    ds4_gpu_tensor *v = calloc(1u, sizeof(*v));
    if (v) { v->data = t->data + off; v->bytes = bytes; }
    return v;
}
void ds4_gpu_tensor_free(ds4_gpu_tensor *t) {
    if (!t) { return; }
    if (t->owner) { allocated -= t->bytes; free(t->data); }
    free(t);
}
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t ? t->bytes : 0u; }
int ds4_gpu_tensor_write(ds4_gpu_tensor *t, uint64_t off, const void *p, uint64_t bytes) {
    if (!t || off > t->bytes || bytes > t->bytes - off) { return 0; }
    memcpy(t->data + off, p, bytes);
    return 1;
}
int ds4_gpu_tensor_read(const ds4_gpu_tensor *t, uint64_t off, void *p, uint64_t bytes) {
    if (!t || off > t->bytes || bytes > t->bytes - off) { return 0; }
    memcpy(p, t->data + off, bytes);
    return 1;
}
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t off,
        const ds4_gpu_tensor *src, uint64_t from, uint64_t bytes) {
    copies++;
    if (!dst || !src || off > dst->bytes || bytes > dst->bytes - off ||
        from > src->bytes || bytes > src->bytes - from ||
        (fail_copy && --fail_copy == 0)) { return 0; }
    memmove(dst->data + off, src->data + from, bytes);
    return 1;
}

#define GLM53_MTP_STATE
#include "../ds4_glm53_mtp.inc"
#undef GLM53_MTP_STATE

static bool fixture_step(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_weights *w, const ds4_gpu_tensor *hidden, int token,
        uint32_t pos, int *next) {
    (void)m; (void)w;
    if (pos >= g->ctx_cap || (g->mtp_len && pos != g->mtp_len)) { return false; }
    if (!g->mtp_len) { g->mtp_min = pos; }
    const float *h = (const float *)hidden->data;
    float *out = (float *)g->mtp_hidden->data;
    const uint16_t kv[] = {(uint16_t)(h[0] + token), (uint16_t)(h[1] + pos)};
    if (!ds4_gpu_tensor_write(g->mtp_kv, (uint64_t)pos * sizeof(kv), kv, sizeof(kv))) {
        return false;
    }
    out[0] = h[0] + (float)token + 0.5f;
    out[1] = h[1] + (float)pos + 0.25f;
    *next = (token + 1) % DS4_N_VOCAB;
    g->mtp_len = pos + 1u;
    steps++;
    return true;
}

static bool fixture_forward(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_weights *w, int token, uint32_t pos,
        const float *embedding, float *logits) {
    (void)embedding;
    if (pos != g->cache_len || pos >= g->ctx_cap) { return false; }
    if (g->mtp_len && g->mtp_len != pos - 1u) { g->mtp_len = g->mtp_min = 0u; }
    int unused;
    if (!fixture_step(g, m, w, g->last_hidden, token, pos - 1u, &unused)) { return false; }
    float *state = (float *)g->state_pool->data;
    state[0] += (float)(token + pos); /* recurrent state */
    state[1] = state[1] * 2.0f + (float)token; /* convolution carry */
    state[2] = (float)(pos % 4u); state[3] += (float)token; /* pool tails */
    float *hidden = (float *)g->last_hidden->data;
    hidden[0] = (float)(100u * pos + (unsigned)token);
    hidden[1] = hidden[0] + 0.5f;
    for (unsigned i = 0u; i < DS4_N_VOCAB; i++) { logits[i] = -(float)i; }
    logits[(token + 1) % DS4_N_VOCAB] = 10.0f;
    CHECK(ds4_gpu_tensor_write(g->logits, 0u, logits, DS4_N_VOCAB * sizeof(float)));
    g->cache_len = pos + 1u;
    forwards++;
    return true;
}

#define glm53_mtp_step fixture_step
#define glm53_graph_forward_token fixture_forward
#define GLM53_MTP_TRIAL
#include "../ds4_glm53_mtp.inc"
#undef GLM53_MTP_TRIAL
#undef glm53_graph_forward_token
#undef glm53_mtp_step

static void init(ds4_glm53_gpu_graph *g, float *logits) {
    memset(g, 0, sizeof(*g));
    g->ready = true; g->ctx_cap = FIXTURE_CTX; g->cache_len = FIXTURE_START;
    g->state_bytes = STATE_FLOATS * sizeof(float);
    g->state_pool = ds4_gpu_tensor_alloc(g->state_bytes);
    g->last_hidden = ds4_gpu_tensor_view(g->state_pool, 4u * sizeof(float), DS4_N_EMBD * sizeof(float));
    g->mtp_hidden = ds4_gpu_tensor_view(g->state_pool, 6u * sizeof(float), DS4_N_EMBD * sizeof(float));
    g->logits = ds4_gpu_tensor_alloc(DS4_N_VOCAB * sizeof(float));
    CHECK(g->state_pool && g->last_hidden && g->mtp_hidden && g->logits && glm53_mtp_enable(g));
    float state[STATE_FLOATS];
    for (unsigned i = 0u; i < STATE_FLOATS; i++) { state[i] = (float)i; }
    CHECK(ds4_gpu_tensor_write(g->state_pool, 0u, state, sizeof(state)));
    for (unsigned i = 0u; i < DS4_N_VOCAB; i++) { logits[i] = (float)i; }
    CHECK(ds4_gpu_tensor_write(g->logits, 0u, logits, DS4_N_VOCAB * sizeof(float)));
}

static void release(ds4_glm53_gpu_graph *g) {
    glm53_mtp_free(g);
    ds4_gpu_tensor_free(g->last_hidden); ds4_gpu_tensor_free(g->mtp_hidden);
    ds4_gpu_tensor_free(g->state_pool); ds4_gpu_tensor_free(g->logits);
}

static void check_rewind(void) {
    ds4_model m = {0}; ds4_weights w = {0};
    for (unsigned keep = 0u; keep < GLM53_MTP_ROWS; keep++) {
        ds4_glm53_gpu_graph g, base;
        float logits[DS4_N_VOCAB], expected[DS4_N_VOCAB];
        init(&g, logits); init(&base, expected);
        int tokens[GLM53_MTP_ROWS], target[GLM53_MTP_ROWS];
        CHECK(glm53_mtp_trial(&g, &m, &w, 4, GLM53_MTP_ROWS, logits, tokens, target));
        CHECK(glm53_mtp_accept(&g, GLM53_MTP_ROWS, logits));
        for (unsigned i = 0u; i < keep; i++) {
            CHECK(fixture_forward(&base, &m, &w, tokens[i], base.cache_len, NULL, expected));
        }
        const unsigned before = forwards;
        CHECK(glm53_mtp_rewind(&g, FIXTURE_START + keep, logits));
        CHECK(forwards == before && g.cache_len == base.cache_len);
        CHECK(memcmp(g.state_pool->data, base.state_pool->data, g.state_bytes) == 0);
        CHECK(memcmp(logits, expected, sizeof(logits)) == 0);
        CHECK(g.mtp_len == base.mtp_len && g.mtp_min == base.mtp_min);
        release(&g); release(&base);
        CHECK(!allocated);
    }
    puts("GLM MTP: retained-cycle rewind passed");
}

static void check_stale(void) {
    ds4_model m = {0}; ds4_weights w = {0};
    ds4_glm53_gpu_graph g, clone;
    float logits[DS4_N_VOCAB], other[DS4_N_VOCAB];
    init(&g, logits); init(&clone, other);
    glm53_mtp_free(&clone);
    CHECK(glm53_mtp_clone(&clone, &g));
    int tokens[GLM53_MTP_ROWS], target[GLM53_MTP_ROWS];
    CHECK(glm53_mtp_trial(&g, &m, &w, 4, GLM53_MTP_ROWS, logits, tokens, target));
    CHECK(glm53_mtp_accept(&g, GLM53_MTP_ROWS, logits));
    ((float *)clone.state_pool->data)[0] += 1000.0f;
    CHECK(glm53_mtp_trial(&clone, &m, &w, 1, 2u, other, tokens, target));
    CHECK(glm53_mtp_accept(&clone, 2u, other));
    unsigned char state[STATE_FLOATS * sizeof(float)];
    memcpy(state, g.state_pool->data, sizeof(state));
    CHECK(!glm53_mtp_rewind(&g, FIXTURE_START + 1u, logits));
    CHECK(g.cache_len == FIXTURE_START + GLM53_MTP_ROWS);
    CHECK(memcmp(state, g.state_pool->data, sizeof(state)) == 0);
    release(&clone); release(&g);
    CHECK(!allocated);
    puts("GLM MTP: shared-bank stale rewind rejected");
}

static void check_pending(void) {
    ds4_model m = {0}; ds4_weights w = {0};
    ds4_glm53_gpu_graph g, clone;
    float logits[DS4_N_VOCAB], other[DS4_N_VOCAB];
    init(&g, logits); init(&clone, other);
    glm53_mtp_free(&clone);
    CHECK(glm53_mtp_clone(&clone, &g));
    int tokens[GLM53_MTP_ROWS], target[GLM53_MTP_ROWS];
    CHECK(glm53_mtp_trial(&g, &m, &w, 4, GLM53_MTP_ROWS, logits, tokens, target));
    ((float *)clone.state_pool->data)[0] += 1000.0f;
    CHECK(glm53_mtp_trial(&clone, &m, &w, 1, 2u, other, tokens, target));
    unsigned char state[STATE_FLOATS * sizeof(float)];
    memcpy(state, g.state_pool->data, sizeof(state));
    CHECK(!glm53_mtp_accept(&g, 1u, logits));
    CHECK(g.mtp_trial_n == GLM53_MTP_ROWS);
    CHECK(memcmp(state, g.state_pool->data, sizeof(state)) == 0);
    release(&clone); release(&g);
    CHECK(!allocated);
    puts("GLM MTP: shared-bank stale pending commit rejected");
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "rewind") == 0) { check_rewind(); return 0; }
    if (argc == 2 && strcmp(argv[1], "stale") == 0) { check_stale(); return 0; }
    if (argc == 2 && strcmp(argv[1], "pending") == 0) { check_pending(); return 0; }
    ds4_model m = {0}; ds4_weights w = {0};
    for (unsigned n = 1u; n <= GLM53_MTP_ROWS; n++) {
        for (unsigned keep = 0u; keep <= n; keep++) {
            ds4_glm53_gpu_graph g, base;
            float logits[DS4_N_VOCAB], expected[DS4_N_VOCAB];
            init(&g, logits); init(&base, expected);
            int tokens[GLM53_MTP_ROWS], target[GLM53_MTP_ROWS];
            CHECK(glm53_mtp_trial(&g, &m, &w, 4, n, logits, tokens, target));
            CHECK(g.cache_len == FIXTURE_START + n && g.mtp_trial_n == n);
            CHECK(!glm53_mtp_trial(&g, &m, &w, 4, n, logits, tokens, target));
            CHECK(!glm53_mtp_accept(&g, n + 1u, logits) && g.mtp_trial_n == n);
            for (unsigned i = 0u; i < n; i++) {
                CHECK(tokens[i] == (int)((4u + i) % DS4_N_VOCAB));
                CHECK(target[i] == (tokens[i] + 1) % DS4_N_VOCAB);
                if (i < keep) {
                    CHECK(fixture_forward(&base, &m, &w, tokens[i], base.cache_len, NULL, expected));
                }
            }
            const unsigned before = forwards;
            CHECK(glm53_mtp_accept(&g, keep, logits));
            CHECK(forwards == before); /* Commit never re-executes a target row. */
            CHECK(g.cache_len == base.cache_len && !g.mtp_trial_n);
            CHECK(g.mtp_len == base.mtp_len && g.mtp_min == base.mtp_min);
            CHECK(memcmp(g.state_pool->data, base.state_pool->data, g.state_bytes) == 0);
            CHECK(memcmp(logits, expected, sizeof(logits)) == 0);
            if (g.mtp_len > g.mtp_min) {
                const uint64_t off = (uint64_t)g.mtp_min * DS4_N_KV_LORA * sizeof(uint16_t);
                const uint64_t bytes = (uint64_t)(g.mtp_len - g.mtp_min) * DS4_N_KV_LORA * sizeof(uint16_t);
                CHECK(memcmp(g.mtp_kv->data + off, base.mtp_kv->data + off, bytes) == 0);
            }
            ds4_gpu_tensor *cursor = ds4_gpu_tensor_alloc(glm53_mtp_ckpt_bytes(&g));
            CHECK(cursor && glm53_mtp_ckpt(&g, cursor, 0u, GLM53_MTP_SAVE));
            g.mtp_len = g.mtp_min = 0u;
            CHECK(glm53_mtp_ckpt(&g, cursor, 0u, GLM53_MTP_RESTORE));
            CHECK(g.mtp_len == base.mtp_len && g.mtp_min == base.mtp_min);
            ds4_gpu_tensor_free(cursor);
            release(&g); release(&base);
            CHECK(!allocated);
        }
    }
    CHECK(copies && steps && forwards);
    puts("GLM MTP: recursive drafts, every accepted prefix, abort, state and cursor restore passed");
    return 0;
}
