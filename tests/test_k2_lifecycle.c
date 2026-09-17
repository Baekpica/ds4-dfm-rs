/* K2-only serial/disk/bank lifecycle. HTTP restart is a separate gate.
 * Usage: test_k2_lifecycle first-shard.gguf [context=1024]
 * The two-bank fork uses this small context, not the qualified 32K/one-bank claim. */
#include "../ds4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char error[256];
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "K2 lifecycle FAIL %d: %s: %s\n", __LINE__, #x, error); exit(1); \
} } while (0)
enum { K2_MODEL_ID = 8, DECODE = 16 };

static void greedy_trace(ds4_session *session, int out[DECODE],
                         float *rows, int vocab) {
    for (int i = 0; i < DECODE; i++) {
        if (rows) {
            CHECK(ds4_session_copy_logits(session, rows + (size_t)i * vocab, vocab) == vocab);
        }
        out[i] = ds4_session_argmax(session);
        CHECK(out[i] >= 0);
        if (i + 1 < DECODE) {
            CHECK(ds4_session_eval(session, out[i], error, sizeof(error)) == 0);
        }
    }
}

static void greedy(ds4_session *session, int out[DECODE]) {
    greedy_trace(session, out, NULL, 0);
}

static void sync_trace(ds4_session *session, const ds4_tokens *prompt,
                       const char *label) {
    const int live = ds4_session_pos(session);
    const int common = ds4_session_common_prefix(session, prompt);
    const int span = ds4_session_exaone_rewind_span(session);
    CHECK(span == 0); /* K2 keeps exact extensions, never an edited LCP. */
    int start = common;
    if (start > 0 && start == prompt->len) { start--; }
    if (live - start > span) { start = 0; }
    const uint64_t before = ds4_session_generation(session);
    CHECK(ds4_session_sync(session, prompt, error, sizeof(error)) == 0);
    /* C has no host last_plan accessor. This is the native EXAONE rule
     * derived from public state, not a measured count of GPU rows. */
    fprintf(stderr, "K2 sync %s live=%d prompt=%d lcp=%d span=%d derived_start=%d cap=%d generation=%llu/%llu\n",
            label, live, prompt->len, common, span, start,
            ds4_session_prefill_cap(session), (unsigned long long)before,
            (unsigned long long)ds4_session_generation(session));
}

static void logits_report(const char *label, const float *a, const float *b,
                          int vocab, int actual, int expected) {
    double sq = 0, scale = 0, max_abs = 0;
    float next_a = -INFINITY, next_b = -INFINITY;
    for (int i = 0; i < vocab; i++) {
        CHECK(isfinite(a[i]) && isfinite(b[i]));
        const double d = (double)a[i] - b[i];
        sq += d * d;
        scale += (double)b[i] * b[i];
        if (fabs(d) > max_abs) { max_abs = fabs(d); }
        if (i != actual && a[i] > next_a) { next_a = a[i]; }
        if (i != expected && b[i] > next_b) { next_b = b[i]; }
    }
    fprintf(stderr, "K2 logits %s max_abs=%.9g rel_rms=%.9g top=%d/%d margin=%.9g/%.9g warm_pair=%.9g/%.9g cold_pair=%.9g/%.9g\n",
            label, max_abs, sqrt(sq / (scale + 1e-30)), actual, expected,
            a[actual] - next_a, b[expected] - next_b,
            a[actual], a[expected], b[actual], b[expected]);
}

static void cold_compare(ds4_engine *engine, int context,
                         ds4_session *warm, const ds4_tokens *prompt,
                         const char *label) {
    const int vocab = ds4_engine_vocab_size(engine);
    float *a = malloc((size_t)DECODE * vocab * sizeof(*a));
    float *b = malloc((size_t)DECODE * vocab * sizeof(*b));
    CHECK(a && b);
    int actual[DECODE], expected[DECODE];
    greedy_trace(warm, actual, a, vocab);
    /* Reuse one allocation; reset forces a complete cold prefill. */
    ds4_session_invalidate(warm);
    sync_trace(warm, prompt, "cold-control");
    greedy_trace(warm, expected, b, vocab);
    int first = -1;
    for (int i = 0; i < DECODE; i++) {
        if (actual[i] != expected[i]) { first = i; break; }
    }
    fprintf(stderr, "K2 suffix %s ctx=%d prompt=%d greedy=%d first_diff=%d\n",
            label, context, prompt->len, DECODE, first);
    logits_report("prefill", a, b, vocab, actual[0], expected[0]);
    if (first >= 0) {
        /* Up to this first flip both decode histories are identical. Later
         * rows are conditional on different tokens and cannot diagnose it. */
        logits_report("first-diff", a + (size_t)first * vocab,
                      b + (size_t)first * vocab, vocab, actual[first], expected[first]);
        for (int i = 0; i <= first; i++) {
            fprintf(stderr, "K2 greedy step=%d warm=%d cold=%d\n", i, actual[i], expected[i]);
        }
        ds4_session_invalidate(warm);
        sync_trace(warm, prompt, "cold-repeat");
        CHECK(ds4_session_copy_logits(warm, a, vocab) == vocab);
        int repeated[DECODE];
        greedy(warm, repeated);
        const int same_logits = memcmp(a, b, (size_t)vocab * sizeof(*a)) == 0;
        const int same_tokens = memcmp(repeated, expected, sizeof(repeated)) == 0;
        logits_report("cold-repeat", a, b, vocab, repeated[0], expected[0]);
        fprintf(stderr, "K2 cold-repeat exact_logits=%d greedy_equal=%d\n", same_logits, same_tokens);
        CHECK(same_logits && same_tokens);
    }
    CHECK(memcmp(actual, expected, sizeof(actual)) == 0);
    /* Cross-prefill-width arithmetic is reported for review. Exact payload
     * restoration has its own bit-identical full-logit assertion. */
    free(a);
    free(b);
}

typedef struct {
    ds4_tokens *prompt;
    int cached, fork, place, sent;
    int got_cached, got_computed, got_bank, n;
    int tokens[DECODE];
} request;

static int admitted(void *ud, void *user, int cached, int computed, int bank) {
    (void)user;
    request *r = ud;
    r->got_cached = cached;
    r->got_computed = computed;
    r->got_bank = bank;
    return 1;
}

static int admit(void *ud, ds4_cont_request *out) {
    request *r = ud;
    if (r->sent++) { return 0; }
    *out = (ds4_cont_request){.tokens = r->prompt->v, .n = r->prompt->len,
        .max_new = DECODE, .eos = -1, .n_cached = r->cached,
        .fork_bank = r->fork, .place_bank = r->place, .on_admitted = admitted};
    return 1;
}

static void done(void *ud, void *user, const int *tokens, int n, int finish) {
    (void)user;
    (void)finish;
    request *r = ud;
    CHECK(tokens && n == DECODE);
    memcpy(r->tokens, tokens, sizeof(r->tokens));
    r->n = n;
}

static void bank_run(ds4_batch_ctx *batch, request *r, const int expected[DECODE]) {
    CHECK(ds4_engine_continuous_generate(batch, admit, NULL, done, r,
                                         error, sizeof(error)) == 0);
    CHECK(r->n == DECODE && memcmp(r->tokens, expected, sizeof(r->tokens)) == 0);
    CHECK(r->got_cached == r->cached);
    CHECK(r->got_computed == r->prompt->len - r->cached);
    CHECK(r->got_bank == r->place - 1);
}

int main(int argc, char **argv) {
    CHECK(argc == 2 || argc == 3);
    const int context = argc == 3 ? atoi(argv[2]) : 1024;
    CHECK(context >= 256);
    CHECK(setenv("DS4_SESSION_LAZY_GRAPH", "1", 1) == 0);
    ds4_engine_options options = {.model_path = argv[1], .backend = DS4_BACKEND_CUDA,
                                  .n_threads = 8, .defer_boot_prewarm = true};
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    CHECK(ds4_engine_open(&engine, &options) == 0);
    CHECK(ds4_engine_model_id(engine) == K2_MODEL_ID);
    ds4_tokens base = {0}, append = {0}, edited = {0};
    ds4_tokenize_text(engine, "A traveler visits Paris in France. The capital of France is", &base);
    ds4_tokenize_text(engine, "A traveler visits Rome in Italy. The capital of Italy is", &edited);
    CHECK(base.len > 3 && edited.len > 3);
    CHECK(ds4_session_create(&session, engine, context) == 0);
    CHECK(ds4_session_sync(session, &base, error, sizeof(error)) == 0);
    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(*logits));
    float *restored = malloc((size_t)vocab * sizeof(*restored));
    CHECK(logits && restored);
    CHECK(ds4_session_copy_logits(session, logits, vocab) == vocab);
    ds4_session_snapshot saved = {0}, bad = {0};
    ds4_session_payload_file disk = {0};
    CHECK(ds4_session_save_snapshot(session, &saved, error, sizeof(error)) == 0);
    CHECK(saved.len == ds4_session_payload_bytes(session) && saved.len > 0);
    CHECK(ds4_session_stage_payload(session, &disk, error, sizeof(error)) == 0);
    CHECK(disk.bytes == saved.len);
    for (int i = 0; i < base.len; i++) { ds4_tokens_push(&append, base.v[i]); }
    ds4_tokens_push(&append, ds4_session_argmax(session));
    ds4_session_free(session);
    session = NULL;
    CHECK(ds4_session_create(&session, engine, context) == 0);
    FILE *fp = fopen(disk.path, "rb");
    CHECK(fp && ds4_session_load_payload(session, fp, disk.bytes, error, sizeof(error)) == 0);
    CHECK(fclose(fp) == 0);
    CHECK(ds4_session_pos(session) == base.len);
    CHECK(ds4_session_copy_logits(session, restored, vocab) == vocab);
    CHECK(memcmp(logits, restored, (size_t)vocab * sizeof(*logits)) == 0);
    sync_trace(session, &append, "append");
    cold_compare(engine, context, session, &append, "append");
    sync_trace(session, &edited, "edited");
    cold_compare(engine, context, session, &edited, "edited");
    bad.ptr = malloc((size_t)saved.len);
    CHECK(bad.ptr);
    memcpy(bad.ptr, saved.ptr, (size_t)saved.len);
    bad.len = bad.cap = saved.len;
    bad.ptr[0] ^= 1; /* Invalid wire magic invalidates the current state. */
    CHECK(ds4_session_load_snapshot(session, &bad, error, sizeof(error)) != 0);
    CHECK(ds4_session_pos(session) == 0);
    CHECK(ds4_session_load_snapshot(session, &saved, error, sizeof(error)) == 0);
    CHECK(ds4_session_sync(session, &append, error, sizeof(error)) == 0);
    int oracle[DECODE], edit_oracle[DECODE];
    greedy(session, oracle);
    ds4_session_invalidate(session);
    CHECK(ds4_session_sync(session, &edited, error, sizeof(error)) == 0);
    greedy(session, edit_oracle);
    ds4_session_free(session);

    ds4_batch_ctx *batch = NULL;
    CHECK(ds4_batch_ctx_create_fit(engine, context, 2, 16, &batch, error, sizeof(error)) == 0);
    CHECK(!ds4_batch_ctx_supports_partial_reuse(batch));
    fp = fopen(disk.path, "rb");
    CHECK(fp && ds4_cont_bank_restore_payload(batch, 0, fp, disk.bytes, error, sizeof(error)) == 0);
    CHECK(fclose(fp) == 0);
    request fork = {.prompt = &append, .cached = base.len, .fork = 1, .place = 2};
    bank_run(batch, &fork, oracle);
    const int *committed = NULL;
    CHECK(ds4_batch_ctx_bank_committed(batch, 0, &committed) == base.len);
    CHECK(memcmp(committed, base.v, (size_t)base.len * sizeof(int)) == 0);
    ds4_session_payload_file bank_disk = {0};
    CHECK(ds4_cont_bank_stage_payload(batch, 0, &bank_disk, error, sizeof(error)) == 0);
    ds4_batch_ctx_destroy(batch);
    batch = NULL;
    CHECK(ds4_batch_ctx_create_fit(engine, context, 1, 16, &batch, error, sizeof(error)) == 0);
    fp = fopen(bank_disk.path, "rb");
    CHECK(fp && ds4_cont_bank_restore_payload(batch, 0, fp, bank_disk.bytes, error, sizeof(error)) == 0);
    CHECK(fclose(fp) == 0);
    request after_disk = {.prompt = &append, .cached = base.len, .place = 1};
    bank_run(batch, &after_disk, oracle);
    request edit = {.prompt = &edited, .cached = 0, .place = 1};
    bank_run(batch, &edit, edit_oracle);
    ds4_batch_ctx_destroy(batch);
    ds4_session_payload_file_free(&bank_disk);
    ds4_session_payload_file_free(&disk);
    ds4_session_snapshot_free(&bad);
    ds4_session_snapshot_free(&saved);
    ds4_tokens_free(&base);
    ds4_tokens_free(&append);
    ds4_tokens_free(&edited);
    free(logits);
    free(restored);
    ds4_engine_close(engine);
    puts("K2 lifecycle passed: exact logits, append/edit, corrupt reset, disk, exact bank fork; no MTP");
    return 0;
}
