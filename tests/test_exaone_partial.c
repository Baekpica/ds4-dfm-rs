/* Full-model, width-matched EXAONE checkpoint proof. The source advances far
 * enough to overwrite every local window before fork and in-place restore. */
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "EXAONE partial FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)

enum { CONTEXT = 1024, CHUNK = 32, PREFIX = 200, ADVANCED = 480, SUFFIX = 32, DECODE = 16 };

static void prefill(ds4_batch_ctx *ctx, unsigned bank, const int *tokens,
                    unsigned start, unsigned end) {
    for (unsigned pos = start; pos < end;) {
        unsigned n = end - pos;
        if (n > CHUNK) { n = CHUNK; }
        CHECK(family_banked_prefill(ctx, bank, tokens + pos, n, pos, pos + n == end, NULL, 0));
        pos += n;
    }
}

/* Hash every physical row, including inactive slack: restore/fork may not
 * write a single source byte. Full-vocabulary comparisons below gate output. */
static uint64_t bank_hash(ds4_exaone_gpu_graph *g) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned il = 0; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
        const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_kv[il]);
        uint8_t *data = xmalloc(bytes);
        CHECK(ds4_gpu_tensor_read(g->layer_kv[il], 0, data, bytes));
        for (uint64_t i = 0; i < bytes; i++) { hash = (hash ^ data[i]) * UINT64_C(1099511628211); }
        free(data);
    }
    return hash;
}

static void check_logits(const float *got, const float *want) {
    double max_abs = 0;
    for (unsigned i = 0; i < DS4_N_VOCAB; i++) {
        CHECK(isfinite(got[i]) && isfinite(want[i]));
        const double delta = fabs((double)got[i] - want[i]);
        if (delta > max_abs) { max_abs = delta; }
    }
    fprintf(stderr, "EXAONE partial full-vocab max_abs=%.9g\n", max_abs);
    CHECK(max_abs == 0);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <first-K-EXAONE-model-shard.gguf>\n", argv[0]);
        return 2;
    }
    setenv("DS4_SESSION_LAZY_GRAPH", "1", 1);
    setenv("DS4_NO_BOOT_PREWARM", "1", 1);
    setenv("DS4_SERVER_FORK_PARTIAL", "1", 1);
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_CUDA,
        .n_threads = 8, .defer_boot_prewarm = true};
    ds4_engine *engine = NULL;
    CHECK(ds4_engine_open(&engine, &opt) == 0);
    CHECK(DS4_MODEL_VARIANT == DS4_VARIANT_KEXAONE_236B);
    char err[256] = {0};
    ds4_batch_ctx *ctx = NULL;
    CHECK(ds4_batch_ctx_create_fit(engine, CONTEXT, 2, CHUNK, &ctx, err, sizeof(err)) == 0);
    CHECK(ctx && ctx->max_seq == 2 && ds4_batch_ctx_supports_partial_reuse(ctx));
    ds4_exaone_batch_runtime *rt = ctx->exaone;
    ds4_tokens text = {0};
    ds4_tokenize_text(engine, "The checkpoint preserves local attention and the complete global prefix. Count upward: 1, 2, 3, 4, 5, 6, 7, 8.\n", &text);
    CHECK(text.len > 0);
    int trunk[ADVANCED], branch[PREFIX + SUFFIX];
    for (unsigned i = 0; i < ADVANCED; i++) { trunk[i] = text.v[i % (unsigned)text.len]; }
    memcpy(branch, trunk, sizeof(branch));
    for (unsigned i = PREFIX; i < PREFIX + SUFFIX; i++) { branch[i] = text.v[(i + 7) % (unsigned)text.len]; }
    const size_t logits_bytes = DS4_N_VOCAB * sizeof(float);
    float *expected = xmalloc((size_t)DECODE * logits_bytes);
    int greedy[DECODE];

    /* Cold reference uses the same prefill boundaries as replay. This isolates
     * checkpoint plumbing from legitimate cross-width floating-point changes. */
    prefill(ctx, 1, branch, 0, PREFIX);
    prefill(ctx, 1, branch, PREFIX, PREFIX + SUFFIX);
    for (unsigned n = 0; n < DECODE; n++) {
        float *row = rt->bank_logits + DS4_N_VOCAB;
        memcpy(expected + (size_t)n * DS4_N_VOCAB, row, logits_bytes);
        greedy[n] = argmax_f32(row, DS4_N_VOCAB);
        if (n + 1 < DECODE) {
            const uint32_t bank = 1, pos = PREFIX + SUFFIX + n;
            CHECK(family_banked_decode(ctx, &bank, &greedy[n], &pos, 1));
        }
    }
    prefill(ctx, 0, trunk, 0, PREFIX);
    CHECK(exaone_ckpt_capture(rt, 0, PREFIX, true, 0));
    const int slot = exaone_ckpt_find(rt, 0, PREFIX, PREFIX);
    CHECK(slot >= 0);
    prefill(ctx, 0, trunk, PREFIX, ADVANCED);
    const uint64_t source_hash = bank_hash(&rt->graph[0]);

    for (unsigned pass = 0; pass < 2; pass++) {
        const unsigned dst = pass == 0 ? 1 : 0;
        unsigned restored = 0;
        CHECK(exaone_ckpt_restore(rt, 0, dst, (unsigned)slot, PREFIX, &restored));
        CHECK(restored == PREFIX && rt->cache_len[dst] == PREFIX);
        prefill(ctx, dst, branch, PREFIX, PREFIX + SUFFIX);
        for (unsigned n = 0; n < DECODE; n++) {
            const float *row = rt->bank_logits + (size_t)dst * DS4_N_VOCAB;
            check_logits(row, expected + (size_t)n * DS4_N_VOCAB);
            CHECK(argmax_f32(row, DS4_N_VOCAB) == greedy[n]);
            if (n + 1 < DECODE) {
                const uint32_t pos = PREFIX + SUFFIX + n;
                CHECK(family_banked_decode(ctx, &dst, &greedy[n], &pos, 1));
            }
        }
        if (pass == 0) { CHECK(bank_hash(&rt->graph[0]) == source_hash); }
    }
    free(expected);
    ds4_tokens_free(&text);
    ds4_batch_ctx_destroy(ctx);
    ds4_engine_close(engine);
    puts("EXAONE partial: wrapped LLLG fork/truncate, full-vocabulary logits and 16 greedy tokens PASS");
    return 0;
}
