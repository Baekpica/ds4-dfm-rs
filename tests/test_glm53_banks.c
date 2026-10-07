/* Model-free lifecycle test: real bank walkers, byte-backed device tensors.
 * This checks frontier ownership, not GLM arithmetic or CUDA execution. */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { DS4_MULTISEQ_MAX_SEQ = 8, DS4_MAX_LAYER = 4, DS4_N_LAYER = 4,
    DS4_N_NEXTN_PREDICT = 1, DS4_N_VOCAB = 7, DS4_N_EMBD = 3,
    DS4_N_HC = 2, DS4_N_HEAD = 2, DS4_N_KEY_MLA = 4, DS4_N_KV_LORA = 4,
    DS4_N_KDA_HEAD_DIM = 2, DS4_N_SSM_CONV = 2,
    DS4_GLM53_POOL_SIZE = 4, DS4_N_INDEXER_HEAD_DIM = 2,
    DS4_PARTIAL_PERIODIC_TARGET = 24, GLM53_MTP_SAVES = 5 };
typedef struct { uint8_t *data; uint64_t bytes; unsigned owned; } ds4_gpu_tensor;
typedef struct {
    uint64_t bank_refs[2], last_use;
    uint32_t pos;
    uint8_t logits_valid;
} ds4_partial_checkpoint;
typedef struct { uint32_t count; } ds4_glm53_stream;
typedef struct { int unused; } ds4_model;
typedef struct { int unused; } ds4_weights;
typedef struct {
    bool ready, expanded_diag, mtp_ready;
    uint32_t ctx_cap, cache_len, row_cap, pool_cap, last_rows, mtp_len, mtp_min;
    uint64_t state_bytes, control_bytes;
    ds4_gpu_tensor *state_pool, *control_pool, *logits, *kda_scratch;
    ds4_gpu_tensor *last_hidden, *mtp_hidden, *mtp_kv, *mtp_concat, *mtp_journal;
    float *mtp_logit_rows;
    void *mtp_epoch;
    ds4_gpu_tensor *recurrent[DS4_MAX_LAYER], *q_conv_state[DS4_MAX_LAYER];
    ds4_gpu_tensor *k_conv_state[DS4_MAX_LAYER], *v_conv_state[DS4_MAX_LAYER];
    ds4_gpu_tensor *q_conv_weight[DS4_MAX_LAYER], *k_conv_weight[DS4_MAX_LAYER];
    ds4_gpu_tensor *v_conv_weight[DS4_MAX_LAYER], *decay_scale[DS4_MAX_LAYER];
    ds4_gpu_tensor *dt_bias[DS4_MAX_LAYER], *o_norm[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_tail_k[DS4_MAX_LAYER], *layer_tail_gate[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_kv[DS4_MAX_LAYER], *layer_pool[DS4_MAX_LAYER];
} ds4_glm53_gpu_graph;
typedef struct { ds4_model model; ds4_weights weights; ds4_glm53_stream glm53_stream; } ds4_engine;
typedef enum { GLM53_MTP_SAVE, GLM53_MTP_RESTORE } glm53_mtp_move;
static unsigned prefill_calls, teacher_calls;
static uint64_t span_need, usable_mem = UINT64_MAX, observed_reserve;

static bool ds4_glm53_layer_is_kda(uint32_t il) { return il != 1u; }
static bool partial_checkpoint_ref(const ds4_partial_checkpoint *cp, uint32_t b) {
    return (cp->bank_refs[0] & (1ull << b)) != 0u;
}
static void partial_checkpoint_set_ref(ds4_partial_checkpoint *cp, uint32_t b) {
    cp->bank_refs[0] |= 1ull << b;
}
static void partial_checkpoint_clear_ref(ds4_partial_checkpoint *cp, uint32_t b) {
    cp->bank_refs[0] &= ~(1ull << b);
    if (!cp->bank_refs[0] && !cp->bank_refs[1]) { cp->pos = 0u; cp->logits_valid = 0u; }
}
static ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    ds4_gpu_tensor *t = calloc(1u, sizeof(*t)); assert(t);
    t->data = calloc(1u, bytes); assert(t->data);
    t->bytes = bytes; t->owned = 1u; return t;
}
static void ds4_gpu_tensor_free(ds4_gpu_tensor *t) {
    if (!t) { return; }
    if (t->owned) { free(t->data); }
    free(t);
}
static uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t ? t->bytes : 0u; }
static void *ds4_gpu_tensor_ptr(ds4_gpu_tensor *t) { return t ? t->data : NULL; }
static ds4_gpu_tensor *ds4_gpu_tensor_view(ds4_gpu_tensor *t, uint64_t off, uint64_t n) {
    if (!t || off > t->bytes || n > t->bytes - off) { return NULL; }
    ds4_gpu_tensor *v = calloc(1u, sizeof(*v)); assert(v);
    v->data = t->data + off; v->bytes = n; return v;
}
static int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t d,
        const ds4_gpu_tensor *src, uint64_t s, uint64_t n) {
    if (!dst || !src || d > dst->bytes || s > src->bytes ||
        n > dst->bytes - d || n > src->bytes - s) { return 0; }
    memmove(dst->data + d, src->data + s, n); return 1;
}
static int ds4_gpu_synchronize(void) { return 1; }
static uint64_t batch_span_need(const ds4_gpu_tensor *t, uint64_t o, uint64_t n) {
    (void)t; (void)o; (void)n; return span_need;
}
static uint64_t ds4_mem_usable_beyond(uint64_t reserve) {
    observed_reserve = reserve; return usable_mem;
}
static int ds4_gpu_tensor_ensure(ds4_gpu_tensor *t, uint64_t o, uint64_t n) {
    return t && o <= t->bytes && n <= t->bytes - o;
}
static uint64_t ds4_gpu_tensor_trim(ds4_gpu_tensor *t, uint64_t o, uint64_t n) {
    (void)t; (void)o; return n;
}
static ds4_gpu_tensor *ds4_gpu_tensor_reserve(uint64_t n) { return ds4_gpu_tensor_alloc(n); }
static uint64_t ds4_gpu_vmm_demand_page(void) { return 0u; }
static void *xcalloc(size_t n, size_t s) { void *p = calloc(n, s); assert(p); return p; }
static uint64_t glm53_graph_bytes_for(uint32_t c, uint32_t s) { (void)c; (void)s; return 1u; }
static bool glm53_graph_diag(void) { return false; }
static void glm53_graph_layers(uint32_t *k, uint32_t *d) { *k = 2u; *d = 1u; }
static bool glm53_graph_bind(ds4_glm53_gpu_graph *g, ds4_glm53_gpu_graph *s, uint32_t n) {
    (void)n; *g = *s; return true;
}
static bool glm53_graph_alloc(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_weights *w, uint32_t c, ds4_glm53_stream *s) {
    (void)g; (void)m; (void)w; (void)c; (void)s; abort();
}
static void glm53_graph_free(ds4_glm53_gpu_graph *g) { (void)g; abort(); }
static bool glm53_graph_reset(ds4_glm53_gpu_graph *g) {
    memset(g->state_pool->data, 0, g->state_bytes); g->cache_len = 0u; return true;
}
static bool glm53_graph_prefill(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_weights *w, const int *t, uint32_t r, uint32_t p, const float *e, float *l) {
    (void)m; (void)w; (void)t; (void)e;
    prefill_calls++; g->cache_len = p + r;
    if (l) { memset(l, 0, DS4_N_VOCAB * sizeof(float)); }
    return true;
}
static bool glm53_graph_forward_token(ds4_glm53_gpu_graph *g, const ds4_model *m,
        const ds4_weights *w, int t, uint32_t p, const float *e, float *l) {
    teacher_calls++;
    return glm53_graph_prefill(g, m, w, &t, 1u, p, e, l);
}
static bool ds4_engine_has_mtp(ds4_engine *e) { (void)e; return false; }
static bool glm53_mtp_enable(ds4_glm53_gpu_graph *g) { (void)g; abort(); }
static bool glm53_mtp_clone(ds4_glm53_gpu_graph *g, ds4_glm53_gpu_graph *s) {
    (void)g; (void)s; return true;
}
static bool glm53_mtp_copy(ds4_glm53_gpu_graph *d, ds4_glm53_gpu_graph *s, uint32_t n) {
    if (!s->mtp_ready || d == s) { return true; }
    const uint32_t limit = n ? n - 1u : 0u;
    const uint32_t len = s->mtp_len < limit ? s->mtp_len : limit;
    const uint32_t min = s->mtp_min < len ? s->mtp_min : len;
    const uint64_t row = DS4_N_KV_LORA * sizeof(uint16_t);
    if (len > min && !ds4_gpu_tensor_copy(d->mtp_kv, min * row, s->mtp_kv,
                                          min * row, (len - min) * row)) { return false; }
    d->mtp_len = len; d->mtp_min = min; return true;
}
static uint64_t glm53_mtp_ckpt_bytes(const ds4_glm53_gpu_graph *g) { (void)g; return 8u; }
static bool glm53_mtp_ckpt(ds4_glm53_gpu_graph *g, ds4_gpu_tensor *t,
        uint64_t base, glm53_mtp_move mode) {
    if (mode == GLM53_MTP_SAVE) {
        memcpy(t->data + base, &g->mtp_len, 4u);
        memcpy(t->data + base + 4u, &g->mtp_min, 4u);
    } else {
        memcpy(&g->mtp_len, t->data + base, 4u);
        memcpy(&g->mtp_min, t->data + base + 4u, 4u);
    }
    return true;
}

#include "../ds4_glm53_batch.inc"

typedef struct {
    ds4_glm53_batch_runtime *glm53;
    uint32_t max_seq, seq_cap;
    uint64_t serial_reserve, bank_gen[DS4_MULTISEQ_MAX_SEQ];
    uint8_t bank_hist_valid[DS4_MULTISEQ_MAX_SEQ];
    uint32_t bank_hist_len[DS4_MULTISEQ_MAX_SEQ];
    int *bank_hist;
} ds4_batch_ctx;
enum { DS4_SESSION_PAYLOAD_U32_FIELDS = 13 };
static void payload_set_err(char *err, size_t len, const char *msg) {
    if (len) { snprintf(err, len, "%s", msg); }
}
static int payload_read_u32(FILE *fp, uint32_t *out, uint64_t *left,
        char *err, size_t len) {
    (void)err; (void)len;
    if (*left < sizeof(*out) || fread(out, sizeof(*out), 1u, fp) != 1u) { return 1; }
    *left -= sizeof(*out); return 0;
}
/* The payload codec has separate byte-exact tests. Supply a finite restored
 * frontier here to exercise the real bank loader and checkpoint lifecycle. */
static uint64_t glm53_payload_bytes(const ds4_glm53_gpu_graph *g, uint32_t n) {
    (void)g; (void)n; abort();
}
static int glm53_payload_save(ds4_glm53_gpu_graph *g, const int *tokens,
        uint32_t n, const float *logits, FILE *fp, char *err, size_t len) {
    (void)g; (void)tokens; (void)n; (void)logits; (void)fp; (void)err; (void)len; abort();
}
static int glm53_payload_restore(ds4_glm53_gpu_graph *g, FILE *fp, uint64_t *left,
        const uint32_t *h, int **tokens, float *logits, char *err, size_t len) {
    (void)fp; (void)err; (void)len;
    *tokens = xcalloc(h[7], sizeof(**tokens));
    for (uint32_t i = 0; i < h[7]; i++) { (*tokens)[i] = (int)(i % DS4_N_VOCAB); }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) { logits[i] = (float)i + 0.25f; }
    memset(g->state_pool->data, 37, g->state_bytes);
    g->cache_len = h[7]; g->mtp_len = h[7] - 1u; g->mtp_min = 1u;
    *left = 0u; return 0;
}
#define GLM53_BATCH_API
#include "../ds4_glm53_batch.inc"
#undef GLM53_BATCH_API

static void setup(ds4_glm53_gpu_graph *g) {
    memset(g, 0, sizeof(*g)); g->ready = true; g->ctx_cap = 16u; g->state_bytes = 128u;
    g->state_pool = ds4_gpu_tensor_alloc(g->state_bytes);
    g->layer_kv[1] = ds4_gpu_tensor_alloc(16u * DS4_N_KV_LORA * sizeof(uint16_t));
    g->layer_pool[1] = ds4_gpu_tensor_alloc(4u * DS4_N_INDEXER_HEAD_DIM * sizeof(uint16_t));
}
static void teardown(ds4_glm53_gpu_graph *g) {
    ds4_gpu_tensor_free(g->state_pool); ds4_gpu_tensor_free(g->layer_kv[1]);
    ds4_gpu_tensor_free(g->layer_pool[1]);
}
static void clone_test(void) {
    ds4_glm53_gpu_graph source, bank; setup(&source);
    source.control_bytes = 16u;
    source.control_pool = ds4_gpu_tensor_alloc(source.control_bytes);
    source.last_hidden = ds4_gpu_tensor_view(source.state_pool, 96u, 12u);
    source.mtp_hidden = ds4_gpu_tensor_view(source.state_pool, 108u, 12u);
    source.recurrent[0] = ds4_gpu_tensor_view(source.state_pool, 0u, 64u);
    source.layer_tail_k[1] = ds4_gpu_tensor_view(source.state_pool, 64u, 16u);
    source.layer_tail_gate[1] = ds4_gpu_tensor_view(source.state_pool, 80u, 16u);
    source.q_conv_weight[0] = ds4_gpu_tensor_view(source.control_pool, 0u, 8u);
    memset(source.state_pool->data, 37, source.state_bytes);
    assert(glm53_bank_clone(&bank, &source));
    assert(bank.state_pool->data != source.state_pool->data);
    assert(bank.layer_kv[1]->data != source.layer_kv[1]->data);
    assert(bank.layer_pool[1]->data != source.layer_pool[1]->data);
    assert(bank.control_pool->data == source.control_pool->data);
    assert(bank.q_conv_weight[0]->data == source.q_conv_weight[0]->data);
    assert(bank.last_hidden->data == bank.state_pool->data + 96u);
    assert(bank.layer_tail_gate[1]->data == bank.state_pool->data + 80u);
    memset(bank.state_pool->data, 91, bank.state_bytes);
    assert(source.state_pool->data[0] == 37u);
    ds4_gpu_tensor_free(bank.last_hidden); ds4_gpu_tensor_free(bank.mtp_hidden);
    ds4_gpu_tensor_free(bank.recurrent[0]); ds4_gpu_tensor_free(bank.layer_tail_k[1]);
    ds4_gpu_tensor_free(bank.layer_tail_gate[1]); ds4_gpu_tensor_free(bank.q_conv_weight[0]);
    ds4_gpu_tensor_free(bank.control_pool); ds4_gpu_tensor_free(bank.logits);
    ds4_gpu_tensor_free(source.last_hidden); ds4_gpu_tensor_free(source.mtp_hidden);
    ds4_gpu_tensor_free(source.recurrent[0]); ds4_gpu_tensor_free(source.layer_tail_k[1]);
    ds4_gpu_tensor_free(source.layer_tail_gate[1]); ds4_gpu_tensor_free(source.q_conv_weight[0]);
    ds4_gpu_tensor_free(source.control_pool); teardown(&bank); teardown(&source);
}
static void prefill_test(void) {
    ds4_glm53_gpu_graph graph; setup(&graph); graph.mtp_ready = true;
    float logits[DS4_N_VOCAB]; uint8_t valid = 0u, failed = 0u;
    ds4_glm53_batch_runtime rt = {.graph=&graph, .max_seq=1u,
        .bank_logits=logits, .bank_logits_valid=&valid, .failed=&failed};
    ds4_engine e = {0}; const int token = 1;
    prefill_calls = teacher_calls = 0u;
    assert(glm53_bank_prefill(&rt, &e, 0u, &token, 1u, 0u, GLM53_FINAL_ROWS));
    assert(prefill_calls == 1u && teacher_calls == 0u);
    const uint32_t bank = 0u, pos = 1u;
    assert(glm53_bank_decode(&rt, &e, &bank, &token, &pos, 1u));
    assert(prefill_calls == 2u && teacher_calls == 1u);
    teardown(&graph);
}
static void old_mtp_window_test(void) {
    ds4_glm53_gpu_graph graph[2]; setup(&graph[0]); setup(&graph[1]);
    for (unsigned b = 0; b < 2; b++) {
        graph[b].mtp_ready = true;
        graph[b].mtp_kv = ds4_gpu_tensor_alloc(16u * DS4_N_KV_LORA * sizeof(uint16_t));
    }
    memset(graph[0].mtp_kv->data, 37, graph[0].mtp_kv->bytes);
    memset(graph[1].mtp_kv->data, 239, graph[1].mtp_kv->bytes);
    graph[0].cache_len = 5u; graph[0].mtp_min = 1u; graph[0].mtp_len = 4u;
    float logits[2u * DS4_N_VOCAB] = {0}, cp_logits[GLM53_BANK_CHECKPOINTS * DS4_N_VOCAB];
    uint8_t valid[2] = {1u, 0u}, failed[2] = {0u, 0u};
    ds4_glm53_batch_runtime rt = {.graph=graph, .max_seq=2u, .bank_logits=logits,
        .bank_logits_valid=valid, .failed=failed, .checkpoint_logits=cp_logits,
        .checkpoint_slot_bytes=136u};
    rt.checkpoint_slab = ds4_gpu_tensor_alloc(136u * GLM53_BANK_CHECKPOINTS);
    assert(glm53_ckpt_capture(&rt, 0u, 5u, GLM53_HAS_LOGITS, 0u));
    graph[0].cache_len = 10u; graph[0].mtp_min = 8u; graph[0].mtp_len = 9u;
    uint32_t pos = 0u;
    const int cp = glm53_ckpt_find(&rt, 0u, 5u, 6u); assert(cp >= 0);
    assert(glm53_ckpt_restore(&rt, 0u, 1u, (uint32_t)cp, 5u, &pos));
    assert(pos == 5u && graph[1].mtp_min == 1u && graph[1].mtp_len == 4u);
    const uint64_t row = DS4_N_KV_LORA * sizeof(uint16_t);
    assert(!memcmp(graph[1].mtp_kv->data + row, graph[0].mtp_kv->data + row, 3u * row));
    assert(graph[1].mtp_kv->data[0] == 239u && graph[1].mtp_kv->data[4u * row] == 239u);
    assert(graph[0].cache_len == 10u && graph[0].mtp_min == 8u && graph[0].mtp_len == 9u);
    ds4_gpu_tensor_free(rt.checkpoint_slab);
    for (unsigned b = 0; b < 2; b++) { ds4_gpu_tensor_free(graph[b].mtp_kv); teardown(&graph[b]); }
}
static void frontier_test(void) {
    ds4_glm53_gpu_graph graph[2]; setup(&graph[0]); setup(&graph[1]);
    float logits[2u * DS4_N_VOCAB], cp_logits[GLM53_BANK_CHECKPOINTS * DS4_N_VOCAB];
    uint8_t valid[2] = {1u, 0u}, failed[2] = {0u, 0u};
    ds4_glm53_batch_runtime rt = {.graph=graph, .max_seq=2u, .bank_logits=logits,
        .bank_logits_valid=valid, .failed=failed, .checkpoint_logits=cp_logits,
        .checkpoint_slot_bytes=136u};
    rt.checkpoint_slab = ds4_gpu_tensor_alloc(rt.checkpoint_slot_bytes * GLM53_BANK_CHECKPOINTS);
    for (uint32_t i = 0; i < graph[0].state_bytes; i++) { graph[0].state_pool->data[i] = (uint8_t)(i + 7u); }
    for (uint32_t i = 0; i < graph[0].layer_kv[1]->bytes; i++) { graph[0].layer_kv[1]->data[i] = (uint8_t)(i + 3u); }
    for (uint32_t i = 0; i < graph[0].layer_pool[1]->bytes; i++) { graph[0].layer_pool[1]->data[i] = (uint8_t)(i + 11u); }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) { logits[i] = (float)i + 0.25f; }
    graph[0].cache_len = 9u;
    assert(glm53_ckpt_capture(&rt, 0u, 9u, GLM53_HAS_LOGITS, 0u));
    uint8_t saved_state[128]; memcpy(saved_state, graph[0].state_pool->data, sizeof(saved_state));
    graph[0].cache_len = 13u; memset(graph[0].state_pool->data, 91, graph[0].state_bytes);
    memset(graph[1].layer_kv[1]->data, 237, graph[1].layer_kv[1]->bytes);
    memset(graph[1].layer_pool[1]->data, 239, graph[1].layer_pool[1]->bytes);
    assert(!glm53_bank_copy(&rt, 0u, 1u, 9u)); /* KDA cannot truncate by count. */
    assert(glm53_bank_copy(&rt, 0u, 1u, 13u));
    assert(graph[1].cache_len == 13u && graph[0].cache_len == 13u);
    assert(!memcmp(graph[0].state_pool->data, graph[1].state_pool->data, 128u));
    assert(!memcmp(graph[0].layer_kv[1]->data, graph[1].layer_kv[1]->data, 13u * 8u));
    assert(graph[1].layer_kv[1]->data[13u * 8u] == 237u);
    assert(!memcmp(graph[0].layer_pool[1]->data, graph[1].layer_pool[1]->data, 3u * 4u));
    assert(graph[1].layer_pool[1]->data[3u * 4u] == 239u);
    assert(valid[1] && !memcmp(logits, logits + DS4_N_VOCAB, DS4_N_VOCAB * sizeof(float)));
    int cp = glm53_ckpt_find(&rt, 0u, 12u, 14u); assert(cp >= 0);
    uint32_t pos = 0u;
    assert(glm53_ckpt_restore(&rt, 0u, 1u, (uint32_t)cp, 12u, &pos));
    assert(pos == 9u && graph[1].cache_len == 9u && graph[0].cache_len == 13u);
    assert(!memcmp(saved_state, graph[1].state_pool->data, sizeof(saved_state)));
    assert(graph[0].state_pool->data[0] == 91u);
    rt.checkpoint[cp].logits_valid = 0u;
    assert(glm53_ckpt_find(&rt, 1u, 9u, 9u) == -1);
    assert(glm53_ckpt_find(&rt, 1u, 9u, 10u) == cp);
    assert(glm53_ckpt_restore(&rt, 0u, 0u, (uint32_t)cp, 12u, &pos));
    assert(graph[0].cache_len == 9u && !memcmp(saved_state, graph[0].state_pool->data, sizeof(saved_state)));
    glm53_bank_reset(&rt, 1u); assert(!valid[1] && graph[1].cache_len == 0u);
    assert(!partial_checkpoint_ref(&rt.checkpoint[cp], 1u));
    ds4_gpu_tensor_free(rt.checkpoint_slab); teardown(&graph[0]); teardown(&graph[1]);
}
static void loaded_frontier_test(void) {
    ds4_glm53_gpu_graph graph; setup(&graph); graph.mtp_ready = true;
    float logits[DS4_N_VOCAB], cp_logits[GLM53_BANK_CHECKPOINTS * DS4_N_VOCAB];
    uint8_t valid = 0u, failed = 0u;
    ds4_glm53_batch_runtime rt = {.graph=&graph, .max_seq=1u, .bank_logits=logits,
        .bank_logits_valid=&valid, .failed=&failed, .checkpoint_logits=cp_logits,
        .checkpoint_slot_bytes=136u};
    rt.checkpoint_slab = ds4_gpu_tensor_alloc(rt.checkpoint_slot_bytes * GLM53_BANK_CHECKPOINTS);
    int history[16];
    ds4_batch_ctx ctx = {.glm53=&rt, .max_seq=1u, .seq_cap=16u,
        .serial_reserve=64u, .bank_hist=history};
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {0}; h[7] = 5u;
    FILE *fp = tmpfile(); assert(fp);
    assert(fwrite(h, sizeof(h), 1u, fp) == 1u); rewind(fp);
    char err[128] = {0};
    (void)glm53_bank_save_payload;
    assert(!glm53_bank_load_payload(&ctx, 0u, fp, sizeof(h), err, sizeof(err)));
    assert(ctx.bank_hist_valid[0] && ctx.bank_hist_len[0] == 5u && ctx.bank_gen[0] == 1u);
    assert(history[4] == 4 && valid && !failed);

    /* An appended answer changes recurrent state. An edited question must
     * restore the loaded prefix instead of restarting from token zero. */
    graph.cache_len = 10u; graph.mtp_min = 8u; graph.mtp_len = 9u;
    memset(graph.state_pool->data, 91, graph.state_bytes);
    memset(logits, 0, sizeof(logits));
    assert(glm53_ckpt_capture(&rt, 0u, 10u, GLM53_HAS_LOGITS, ctx.serial_reserve));
    const int cp = glm53_ckpt_find(&rt, 0u, 5u, 6u); assert(cp >= 0);
    uint32_t pos = 0u;
    assert(glm53_ckpt_restore(&rt, 0u, 0u, (uint32_t)cp, 5u, &pos));
    assert(pos == 5u && graph.cache_len == 5u && graph.mtp_min == 1u && graph.mtp_len == 4u);
    for (uint32_t i = 0; i < graph.state_bytes; i++) { assert(graph.state_pool->data[i] == 37u); }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) { assert(logits[i] == (float)i + 0.25f); }

    /* Lack of checkpoint funding must not invalidate a successful load. */
    rewind(fp); span_need = 1u; usable_mem = 0u; observed_reserve = 0u;
    assert(!glm53_bank_load_payload(&ctx, 0u, fp, sizeof(h), err, sizeof(err)));
    assert(observed_reserve == ctx.serial_reserve && valid && !failed && graph.cache_len == 5u);
    assert(glm53_ckpt_find(&rt, 0u, 5u, 6u) == -1);
    span_need = 0u; usable_mem = UINT64_MAX;
    fclose(fp); ds4_gpu_tensor_free(rt.checkpoint_slab); teardown(&graph);
}
int main(void) {
    (void)glm53_batch_create; (void)glm53_ckpt_trim; (void)glm53_bank_bytes_for;
    old_mtp_window_test(); prefill_test(); clone_test(); frontier_test(); loaded_frontier_test();
    puts("GLM compact bank frontier: PASS"); return 0;
}
