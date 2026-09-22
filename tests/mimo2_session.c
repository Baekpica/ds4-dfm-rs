/* Exercise production session control against a deterministic forward stub.
 * These tests cover state transitions, not model arithmetic or GPU ownership. */
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

enum { DS4_N_VOCAB = 152576 };
typedef struct { int *v; int len, cap; } ds4_tokens;
typedef struct { unsigned cap, context, position; bool failed; void *logits; } ds4_mimo2_graph;
typedef struct { int model, weights; } ds4_engine;
typedef struct {
    ds4_engine *engine;
    ds4_mimo2_graph mimo2_graph;
    bool mimo2_graph_ready, checkpoint_valid, mtp_draft_valid;
    ds4_tokens checkpoint;
    unsigned generation;
    float *logits;
    void (*progress)(void *, const char *, int, int);
    void *progress_ud;
} ds4_session;
static unsigned forwards;
static bool fail_forward, bad_logits, fail_sync;
static int ds4_gpu_synchronize(void) { return !fail_sync; }
static void payload_set_err(char *s, size_t n, const char *msg) { if (n) { snprintf(s, n, "%s", msg); } }
static int ds4_gpu_tensor_read(void *tensor, unsigned off, float *out, size_t bytes) {
    (void)tensor; (void)off;
    for (size_t i = 0; i < bytes / sizeof(float); i++) { out[i] = 0.0f; }
    if (bad_logits) { out[0] = NAN; }
    return 1;
}
static int ds4_session_ensure_graph(ds4_session *s, char *err, size_t n) {
    (void)err; (void)n; s->mimo2_graph_ready = true; return 0;
}
static bool ds4_tokens_starts_with(const ds4_tokens *a, const ds4_tokens *b) {
    return a->len >= b->len && !memcmp(a->v, b->v, b->len * sizeof(int));
}
static void ds4_tokens_copy(ds4_tokens *a, const ds4_tokens *b) {
    assert(b->len <= a->cap); memcpy(a->v, b->v, b->len * sizeof(int)); a->len = b->len;
}
static void token_vec_push(ds4_tokens *v, int token) { assert(v->len < v->cap); v->v[v->len++] = token; }
static bool mimo2_forward(ds4_mimo2_graph *g, const void *m, const void *w,
                           const int *tokens, unsigned n, unsigned pos) {
    (void)m; (void)w; (void)tokens;
    assert(pos == g->position && n <= g->cap && pos + n <= g->context);
    forwards++;
    if (fail_forward) { g->failed = true; return false; }
    g->position += n;
    return true;
}
static void ds4_session_invalidate(ds4_session *s);
#include "../ds4_mimo2_session.inc"
static void ds4_session_invalidate(ds4_session *s) {
    s->generation++; s->checkpoint_valid = false; s->checkpoint.len = 0;
    s->mtp_draft_valid = false;
    if (s->mimo2_graph_ready) { (void)mimo2_reset(&s->mimo2_graph); }
}
int main(void) {
    ds4_engine engine = {0};
    int checkpoint[32], input[] = {1, 2, 3, 4, 5, 6, 7};
    ds4_session s = {.engine = &engine, .mimo2_graph = {.cap = 3, .context = 32},
        .checkpoint = {.v = checkpoint, .cap = 32}, .logits = calloc(DS4_N_VOCAB, sizeof(float))};
    assert(s.logits);
    ds4_tokens prompt = {.v = input, .len = 5, .cap = 7};
    char err[128];
    assert(!mimo2_session_sync(&s, &prompt, err, sizeof(err)));
    assert(forwards == 2 && s.checkpoint_valid && s.checkpoint.len == 5 && s.mimo2_graph.position == 5);
    assert(!mimo2_session_sync(&s, &prompt, err, sizeof(err)) && forwards == 2);
    prompt.len = 7;
    assert(!mimo2_session_sync(&s, &prompt, err, sizeof(err)) && forwards == 3);
    assert(!mimo2_session_eval(&s, 8, err, sizeof(err)));
    assert(s.checkpoint.len == 8 && s.mimo2_graph.position == 8);
    input[0] = 9; prompt.len = 3;
    assert(!mimo2_session_sync(&s, &prompt, err, sizeof(err)));
    assert(s.checkpoint.len == 3 && s.mimo2_graph.position == 3);
    for (unsigned i = 0; i < 3; i++) {
        const int pads[] = {151655, 151656, 151669};
        input[0] = pads[i];
        assert(mimo2_session_sync(&s, &prompt, err, sizeof(err)));
        assert(s.checkpoint_valid && s.checkpoint.len == 3);
    }
    input[0] = 9; prompt.len = 5; fail_forward = true;
    assert(mimo2_session_sync(&s, &prompt, err, sizeof(err)));
    assert(!s.checkpoint_valid && s.checkpoint.len == 0 && s.mimo2_graph.failed);
    fail_forward = false; fail_sync = true;
    assert(mimo2_session_sync(&s, &prompt, err, sizeof(err)) && s.mimo2_graph.failed);
    fail_sync = false;
    assert(!mimo2_session_sync(&s, &prompt, err, sizeof(err)));
    bad_logits = true;
    assert(mimo2_session_eval(&s, 8, err, sizeof(err)));
    assert(!s.checkpoint_valid && s.mimo2_graph.failed);
    bad_logits = false;
    assert(!mimo2_session_sync(&s, &prompt, err, sizeof(err)));
    unsigned before = forwards;
    assert(mimo2_session_eval(&s, -1, err, sizeof(err)) && forwards == before);
    free(s.logits);
    puts("session control: prefix reuse, branch replay, media rejection, forward/sync/NaN failures and recovery passed");
    return 0;
}
