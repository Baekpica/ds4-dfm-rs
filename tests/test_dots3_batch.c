/* Width-matched dots3 text-bank proof across both the local ring and DSA
 * top-2048 boundary. No new arithmetic or MTP execution is used. */
#include "../ds4.c"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "dots3 bank FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)
enum { CONTEXT = 4096, CHUNK = 128, PREFIX = 2304, ADVANCED = 3328, SUFFIX = 128, DECODE = 16 };

static void prefill(ds4_batch_ctx *ctx, unsigned bank, const int *tokens,
                    unsigned start, unsigned end) {
    if (!start) { bank_hist_reset(ctx, bank); family_banked_reset(ctx, bank); }
    for (unsigned pos = start; pos < end;) {
        unsigned n = end - pos;
        if (n > CHUNK) { n = CHUNK; }
        CHECK(family_banked_prefill(ctx, bank, tokens + pos, n, pos, pos + n == end, NULL, 0));
        bank_hist_append_n(ctx, bank, tokens + pos, n);
        pos += n;
    }
}

/* Check all physical rows, including slack; a fork may not alter its source. */
static uint64_t bank_hash(ds4_dots3_gpu_graph *g) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned il = 0; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
        ds4_gpu_tensor *parts[] = {g->layer_kv_latent[il], g->layer_k_pe[il], g->layer_idx_k[il]};
        for (unsigned part = 0; part < 3; part++) {
            if (!parts[part]) { continue; }
            const uint64_t bytes = ds4_gpu_tensor_bytes(parts[part]);
            uint8_t *data = xmalloc(bytes);
            CHECK(ds4_gpu_tensor_read(parts[part], 0, data, bytes));
            for (uint64_t i = 0; i < bytes; i++) { hash = (hash ^ data[i]) * UINT64_C(1099511628211); }
            free(data);
        }
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
    fprintf(stderr, "dots3 bank full-vocab max_abs=%.9g\n", max_abs);
    CHECK(max_abs == 0);
}

typedef struct {
    ds4_batch_ctx *ctx;
    const int *prompt;
    int length, cached, source, target, budget, admitted, tokens, done;
    const float *expected;
    const int *greedy;
    bool cancel;
} bank_case;

static int admitted(void *ud, void *user, int cached, int computed, int bank) {
    (void)user;
    bank_case *c = ud;
    CHECK(cached == c->cached && computed == c->length - c->cached && bank == c->target);
    c->admitted++;
    return 1;
}

static int alive(void *ud, void *user) {
    (void)user;
    return !((bank_case *)ud)->cancel;
}

static int admit(void *ud, ds4_cont_request *req) {
    bank_case *c = ud;
    if (c->admitted) { return 0; }
    *req = (ds4_cont_request){.tokens = c->prompt, .n = c->length,
        .max_new = c->budget, .eos = -1, .n_cached = c->cached,
        .fork_bank = c->source + 1, .place_bank = c->target + 1,
        .on_admitted = admitted, .alive = alive, .user = c};
    return 1;
}

static int token(void *ud, void *user, int value) {
    (void)user;
    bank_case *c = ud;
    CHECK(c->tokens < c->budget && value == c->greedy[c->tokens]);
    check_logits(family_banked_logits(c->ctx, c->target),
                 c->expected + (size_t)c->tokens * DS4_N_VOCAB);
    c->tokens++;
    return 1;
}

static void done(void *ud, void *user, const int *tokens, int n, int finish) {
    (void)user; (void)finish;
    bank_case *c = ud;
    CHECK(n == (c->cancel ? 0 : c->budget));
    if (n) { CHECK(memcmp(tokens, c->greedy, (size_t)n * sizeof(int)) == 0); }
    c->done++;
}

static void run_case(bank_case *c) {
    char err[256] = {0};
    CHECK(ds4_engine_continuous_generate(c->ctx, admit, token, done, c, err, sizeof(err)) == 0);
    CHECK(c->admitted == 1 && c->done == 1 && c->tokens == (c->cancel ? 0 : c->budget));
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <first-dots3-MQ87-model-shard.gguf>\n", argv[0]); return 2; }
    setenv("DS4_SESSION_LAZY_GRAPH", "1", 1);
    setenv("DS4_NO_BOOT_PREWARM", "1", 1);
    setenv("DS4_DOTS3_BATCH", "1", 1);
    setenv("DS4_DOTS3_PREFILL_CHUNK", "128", 1);
    setenv("DS4_CONT_PREFILL_CHUNK", "128", 1);
    setenv("DS4_SERVER_FORK_PARTIAL", "1", 1);
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_CUDA,
        .n_threads = 8, .defer_boot_prewarm = true};
    ds4_engine *engine = NULL;
    CHECK(ds4_engine_open(&engine, &opt) == 0);
    CHECK(DS4_MODEL_VARIANT == DS4_VARIANT_DOTS3_NOTE_PREV);
    CHECK(ds4_engine_supports_batching(engine));
    char err[256] = {0};
    ds4_batch_ctx *ctx = NULL;
    CHECK(ds4_batch_ctx_create_fit(engine, CONTEXT, 2, CHUNK, &ctx, err, sizeof(err)) == 0);
    CHECK(ctx && ctx->max_seq == 2 && ds4_batch_ctx_supports_partial_reuse(ctx));
    ds4_dots3_batch_runtime *rt = ctx->dots3;
    ds4_tokens text = {0};
    ds4_tokenize_text(engine, "The checkpoint preserves local latent attention and the complete DSA prefix. Count upward: 1, 2, 3, 4, 5, 6, 7, 8.\n", &text);
    CHECK(text.len > 0);
    int trunk[ADVANCED], branch[PREFIX + SUFFIX];
    for (unsigned i = 0; i < ADVANCED; i++) { trunk[i] = text.v[i % (unsigned)text.len]; }
    memcpy(branch, trunk, sizeof(branch));
    for (unsigned i = PREFIX; i < PREFIX + SUFFIX; i++) { branch[i] = text.v[(i + 7) % (unsigned)text.len]; }
    const size_t logits_bytes = DS4_N_VOCAB * sizeof(float);
    float *expected = xmalloc((size_t)DECODE * logits_bytes);
    int greedy[DECODE];

    /* Width-matched cold reference separates state plumbing from legitimate
     * arithmetic changes caused by choosing a different prefill width. */
    prefill(ctx, 1, branch, 0, PREFIX);
    prefill(ctx, 1, branch, PREFIX, PREFIX + SUFFIX);
    for (unsigned n = 0; n < DECODE; n++) {
        float *row = rt->bank_logits + DS4_N_VOCAB;
        memcpy(expected + (size_t)n * DS4_N_VOCAB, row, logits_bytes);
        greedy[n] = argmax_f32(row, DS4_N_VOCAB);
        if (n + 1 < DECODE) {
            const uint32_t bank = 1, pos = PREFIX + SUFFIX + n;
            CHECK(family_banked_decode(ctx, &bank, &greedy[n], &pos, 1));
            bank_hist_append(ctx, bank, greedy[n]);
        }
    }
    prefill(ctx, 0, trunk, 0, PREFIX);
    CHECK(dots3_ckpt_capture(rt, 0, PREFIX, true, 0));
    prefill(ctx, 0, trunk, PREFIX, ADVANCED);
    const uint64_t source_hash = bank_hash(&rt->graph[0]);
    for (unsigned pass = 0; pass < 2; pass++) {
        bank_case c = {.ctx = ctx, .prompt = branch, .length = PREFIX + SUFFIX,
            .cached = PREFIX, .source = 0, .target = pass == 0 ? 1 : 0,
            .budget = DECODE, .expected = expected, .greedy = greedy};
        run_case(&c);
        if (pass == 0) {
            CHECK(bank_hash(&rt->graph[0]) == source_hash);
            CHECK(ctx->bank_hist_len[0] == ADVANCED);
        }
    }

    /* Exact fork, persisted payload recovery, and malformed-load invalidation
     * use the public lifecycle rather than assigning graph pointers. */
    const int frontier = PREFIX + SUFFIX + DECODE - 1;
    const float *last = expected + (size_t)(DECODE - 1) * DS4_N_VOCAB;
    bank_case exact = {.ctx = ctx, .prompt = ctx->bank_hist, .length = frontier,
        .cached = frontier, .source = 0, .target = 1, .budget = 1,
        .expected = last, .greedy = &greedy[DECODE - 1]};
    run_case(&exact);
    ds4_session_payload_file payload = {0};
    CHECK(ds4_cont_bank_stage_payload(ctx, 1, &payload, err, sizeof(err)) == 0);
    CHECK(payload.bytes == ds4_cont_bank_payload_bytes(ctx, 1));
    FILE *fp = fopen(payload.path, "rb");
    CHECK(fp && ds4_cont_bank_restore_payload(ctx, 0, fp, payload.bytes, err, sizeof(err)) == 0);
    CHECK(fclose(fp) == 0);
    const int *restored = NULL;
    CHECK(ds4_batch_ctx_bank_committed(ctx, 0, &restored) == frontier);
    CHECK(memcmp(restored, ctx->bank_hist + ctx->seq_cap, (size_t)frontier * sizeof(int)) == 0);
    check_logits(rt->bank_logits, last);
    const uint64_t generation = ds4_batch_ctx_bank_generation(ctx, 1);
    fp = fopen(payload.path, "rb");
    CHECK(fp && ds4_cont_bank_restore_payload(ctx, 1, fp, payload.bytes - 7, err, sizeof(err)) != 0);
    CHECK(fclose(fp) == 0);
    CHECK(!ctx->bank_hist_valid[1] && rt->graph[1].cache_len == 0);
    CHECK(ds4_batch_ctx_bank_generation(ctx, 1) != generation);
    fp = fopen(payload.path, "rb");
    CHECK(fp && ds4_cont_bank_restore_payload(ctx, 1, fp, payload.bytes, err, sizeof(err)) == 0);
    CHECK(fclose(fp) == 0);
    const uint32_t banks[] = {0, 1}, positions[] = {(uint32_t)frontier, (uint32_t)frontier};
    const int next[] = {greedy[DECODE - 1], greedy[DECODE - 1]};
    CHECK(family_banked_decode(ctx, banks, next, positions, 2));
    check_logits(rt->bank_logits, rt->bank_logits + DS4_N_VOCAB);
    bank_hist_append(ctx, 0, next[0]); bank_hist_append(ctx, 1, next[1]);
    bank_case cancel = {.ctx = ctx, .prompt = branch, .length = PREFIX + SUFFIX,
        .source = -1, .target = 1, .budget = 1, .cancel = true};
    run_case(&cancel);
    CHECK(rt->graph[1].cache_len == 0 && ctx->bank_hist_len[1] == 0);
    prefill(ctx, 1, branch, 0, 8);
    CHECK(rt->graph[1].cache_len == 8 && ctx->bank_hist_valid[1]);

    ds4_session_payload_file_free(&payload);
    free(expected);
    ds4_tokens_free(&text);
    const int short_token = argmax_f32(rt->bank_logits + DS4_N_VOCAB, DS4_N_VOCAB);
    ds4_batch_ctx_destroy(ctx);
    ds4_tokens prompts[2] = {{.v = branch, .len = 8, .cap = 8}, {.v = branch, .len = 8, .cap = 8}};
    const int budgets[] = {1, 1}, eos[] = {-1, -1};
    ds4_batch_gen_result got[2] = {0};
    CHECK(ds4_engine_batched_generate_ex(engine, prompts, 2, 64, budgets, eos,
                                         got, err, sizeof(err)) == 0);
    for (unsigned i = 0; i < 2; i++) {
        CHECK(got[i].n_tokens == 1 && got[i].tokens[0] == short_token);
        free(got[i].tokens);
    }
    ds4_engine_close(engine);
    puts("dots3 banks: DSA/local wrap, full-vocabulary +16 greedy parity, partial/exact fork, truncate, payload, cancel/reuse PASS");
    return 0;
}
