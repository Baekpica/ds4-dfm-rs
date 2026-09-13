/* Step continuous lane: two banks must decode byte-identically to two
 * independent serial sessions, fork must clone a committed prefix, and a bank
 * disk-KV checkpoint must reload and continue. Optional owner-imported MTP
 * also checks predictors, partial forks and speculative emission boundaries. */
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "Step cont FAIL line %d: %s (%s)\n", __LINE__, #x, err); exit(1); \
} } while (0)
static char err[256];
static bool speculate;

/* C oracle for the production Rust accepted_prefix policy. */
static int accept_prefix(const int *tokens, const int *target, int n, int eos) {
    int keep = 1;
    while (keep < n && tokens[keep - 1] != eos && tokens[keep] == target[keep - 1]) { keep++; }
    return keep;
}

enum { GEN = 16, MAX_REQ = 4 };

typedef struct {
    const int *tokens;
    int n;
    int fork_bank; /* +1, or 0 */
    int n_cached;
    int out[GEN];
    int out_n;
    int placed_bank;
    int stop_after, seen;
    int force_after, force_token;
    int eos;
    bool stop_eos;
} cont_req;

typedef struct {
    cont_req *reqs;
    int count, next;
} driver;

static int override_cb(void *ud, void *user) {
    (void)ud;
    cont_req *r = user;
    return r->force_after && r->seen == r->force_after
        ? DS4_SAMPLE_OVERRIDE_TOKEN(r->force_token) : DS4_SAMPLE_OVERRIDE_NONE;
}

static int admit_cb(void *ud, ds4_cont_request *req) {
    driver *d = ud;
    if (d->next >= d->count) { return 0; }
    cont_req *r = &d->reqs[d->next++];
    memset(req, 0, sizeof(*req));
    req->tokens = r->tokens;
    req->n = r->n;
    req->max_new = GEN;
    req->eos = r->stop_eos ? r->eos : -1;
    req->temperature = 0.0f;    /* greedy argmax */
    req->sample_override = override_cb;
    req->step_accept = speculate ? accept_prefix : NULL;
    req->user = r;
    req->bank_used = &r->placed_bank;
    req->fork_bank = r->fork_bank;
    req->n_cached = r->n_cached;
    return 1;
}

static int token_cb(void *ud, void *user, int token) {
    (void)ud; (void)token;
    cont_req *r = user;
    r->seen++;
    return !r->stop_after || r->seen < r->stop_after;
}

static void done_cb(void *ud, void *user, const int *tokens, int n, int finish) {
    (void)ud; (void)finish;
    cont_req *r = user;
    r->out_n = n < GEN ? n : GEN;
    for (int i = 0; i < r->out_n; i++) { r->out[i] = tokens[i]; }
}

/* Independent serial reference: prefill then greedy-decode GEN tokens. */
static void serial_reference(ds4_engine *e, int ctx, const int *prompt, int n,
                             int *out, int *out_n) {
    ds4_session *s = NULL;
    CHECK(!ds4_session_create(&s, e, ctx) && s);
    ds4_tokens toks = {0};
    for (int i = 0; i < n; i++) { ds4_tokens_push(&toks, prompt[i]); }
    CHECK(!ds4_session_sync(s, &toks, err, sizeof(err)));
    int count = 0;
    for (int i = 0; i < GEN; i++) {
        const int tok = ds4_session_argmax(s);
        CHECK(tok >= 0 && !ds4_session_eval(s, tok, err, sizeof(err)));
        out[count++] = tok;
    }
    *out_n = count;
    ds4_session_free(s);
    ds4_tokens_free(&toks);
}

static void serial_spec_generate(ds4_session *s, int *out) {
    int count = 1;
    out[0] = ds4_session_argmax(s);
    while (count < GEN) {
        int tokens[S37_VERIFY], target[S37_VERIFY];
        int trial = ds4_session_step37_trial(s, out[count - 1], GEN - count + 1,
                                            tokens, target, S37_VERIFY, err, sizeof(err));
        CHECK(trial > 0);
        const int keep = accept_prefix(tokens, target, trial, -1);
        CHECK(!ds4_session_step37_commit(s, keep, err, sizeof(err)));
        for (int i = 1; i < keep && count < GEN; i++) { out[count++] = tokens[i]; }
        if (count < GEN) { out[count++] = ds4_session_argmax(s); }
    }
}

static ds4_session *serial_spec_reference(ds4_engine *e, int ctx, const int *prompt,
                                          int n, int *out) {
    ds4_session *s = NULL;
    CHECK(!ds4_session_create(&s, e, ctx));
    ds4_tokens input = {.v = (int *)prompt, .len = n, .cap = n};
    CHECK(!ds4_session_sync(s, &input, err, sizeof(err)));
    serial_spec_generate(s, out);
    return s;
}

static void same_rows(ds4_step37_graph *a, ds4_step37_graph *b,
                       unsigned layer, unsigned pos) {
    const unsigned cap = a->kv_cap[layer];
    CHECK(cap == b->kv_cap[layer]);
    const size_t row = 2u * S37_KV * sizeof(uint16_t);
    const size_t bytes = cap * row;
    unsigned char *x = xmalloc(bytes), *y = xmalloc(bytes);
    CHECK(ds4_gpu_tensor_read(a->kv[layer], 0, x, bytes));
    CHECK(ds4_gpu_tensor_read(b->kv[layer], 0, y, bytes));
    unsigned start = step37_sliding(layer) && pos > S37_WINDOW ? pos - S37_WINDOW : 0;
    for (unsigned p = start; p < pos; p++) { CHECK(!memcmp(x + (p % cap) * row, y + (p % cap) * row, row)); }
    free(x); free(y);
}

static void same_bank(ds4_batch_ctx *ctx, unsigned bank, ds4_session *s) {
    ds4_step37_batch_runtime *rt = ctx->step37;
    CHECK(ctx->bank_hist_len[bank] == (unsigned)s->checkpoint.len);
    CHECK(!memcmp(rt->bank_logits + (size_t)bank * DS4_N_VOCAB, s->logits, DS4_N_VOCAB * sizeof(float)));
    for (unsigned il = 0; il < STEP37_LAYERS; il++) {
        same_rows(&rt->graph[bank], &s->step37_graph, il, rt->graph[bank].position);
    }
    ds4_step37_spec *a = &rt->spec[bank], *b = &s->step37_spec;
    CHECK(step37_spec_valid(a) && a->position == b->position && a->tail_rows == b->tail_rows);
    for (unsigned d = 0; d < STEP37_DRAFT_LAYERS; d++) {
        CHECK(a->draft.position[d] == b->draft.position[d]);
        same_rows(&a->draft.graph, &b->draft.graph, STEP37_LAYERS + d, a->draft.position[d]);
    }
    float x[STEP37_DRAFT_LAYERS * S37_HIDDEN], y[STEP37_DRAFT_LAYERS * S37_HIDDEN];
    const size_t bytes = a->tail_rows * S37_HIDDEN * sizeof(float);
    CHECK(ds4_gpu_tensor_read(a->tail, 0, x, bytes) && ds4_gpu_tensor_read(b->tail, 0, y, bytes));
    CHECK(!memcmp(x, y, bytes));
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 5) { return 2; }
    const ds4_host_shape host = {.variant = DS4_VARIANT_STEP37_FLASH};
    ds4_host_shape_install(&host);
    model_apply_host_shape();
    ds4_host_shape_clear();
    ds4_engine e = {.backend = DS4_BACKEND_CUDA, .metal_ready = true};
    model_open(&e.model, argv[1], false, false);
    weights_bind(&e.weights, &e.model, false, 0, UINT32_MAX, true, false);
    e.vocab.n_vocab = DS4_N_VOCAB;
    e.vocab.eos_id = -1;
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(e.model.map, e.model.size));
    if (argc == 5) {
        model_open(&e.mtp_model, argv[3], false, false);
        step37_bind_draft(&e.step37_mtp, &e.mtp_model);
        CHECK(ds4_gpu_import_model_ipc_manifest(e.mtp_model.map, e.mtp_model.size, argv[4], "mtp"));
        model_release_mapping_cache(&e.mtp_model);
        e.mtp_ready = true;
        e.mtp_draft_tokens = STEP37_DRAFT_LAYERS;
    }

    ds4_tokens prompt = {0};
    FILE *fp = fopen(argv[2], "r");
    CHECK(fp != NULL);
    int token;
    while (fscanf(fp, "%d", &token) == 1) { ds4_tokens_push(&prompt, token); }
    fclose(fp);
    CHECK(prompt.len >= 300);

    /* Without the env the family stays on the serial lane. */
    CHECK(!ds4_engine_supports_batching(&e));
    setenv("DS4_STEP37_BATCH", "1", 1);
    CHECK(ds4_engine_supports_batching(&e));

    const int ctx = prompt.len + GEN + 8;

    /* Serial references for two distinct prompts. */
    int ref_a[GEN], ref_b[GEN], na = 0, nb = 0;
    const int len_a = 160, len_b = prompt.len; /* B keeps the whole prompt */
    serial_reference(&e, ctx, prompt.v, len_a, ref_a, &na);
    serial_reference(&e, ctx, prompt.v, len_b, ref_b, &nb);
    CHECK(na == GEN && nb == GEN);
    CHECK(!session_tensors_census_live());

    ds4_batch_ctx *ctx_b = NULL;
    CHECK(!ds4_batch_ctx_create_fit(&e, ctx, 2, ctx * 2, &ctx_b, err, sizeof(err)) && ctx_b);
    CHECK(ds4_batch_ctx_max_seq(ctx_b) == 2);
    CHECK((ctx_b->step37->spec != NULL) == e.mtp_ready);

    cont_req reqs[2] = {
        {.tokens = prompt.v, .n = len_a},
        {.tokens = prompt.v, .n = len_b},
    };
    driver d = {.reqs = reqs, .count = 2};
    CHECK(!ds4_engine_continuous_generate(ctx_b, admit_cb, NULL, done_cb, &d, err, sizeof(err)));
    CHECK(reqs[0].out_n == GEN && reqs[1].out_n == GEN);
    if (e.mtp_ready) {
        for (unsigned b = 0; b < 2; b++) {
            CHECK(step37_spec_valid(&ctx_b->step37->spec[b]));
            CHECK(ctx_b->step37->spec[b].position == ctx_b->bank_hist_len[b]);
        }
    }
    CHECK(!memcmp(reqs[0].out, ref_a, GEN * sizeof(int)));
    CHECK(!memcmp(reqs[1].out, ref_b, GEN * sizeof(int)));
    printf("Step cont: two banks match two serial sessions byte-for-byte (%d tokens each)\n", GEN);

    /* Full-frontier fork: bank A is idle with a committed frontier F (the lane
     * commits F = len_a + GEN - 1 rows; the last generated token stays pending
     * as the next input).  Read F exactly, clone the whole frontier into a
     * fresh bank without re-prefill, and continue greedily.  A serial session
     * prefilling those F committed tokens must produce the same stream. */
    const int src_bank = reqs[0].placed_bank;
    const int *committed = NULL;
    const int fcount = ds4_batch_ctx_bank_committed(ctx_b, src_bank, &committed);
    CHECK(fcount > len_a && fcount <= len_a + GEN && committed);
    static int frontier[1024];
    CHECK(fcount <= (int)(sizeof(frontier) / sizeof(frontier[0])));
    for (int i = 0; i < fcount; i++) { frontier[i] = committed[i]; }

    int ref_fork[GEN], nf = 0;
    serial_reference(&e, ctx, frontier, fcount, ref_fork, &nf);
    CHECK(nf == GEN && !session_tensors_census_live());

    cont_req fork_req = {
        .tokens = frontier, .n = fcount,
        .fork_bank = src_bank + 1, .n_cached = fcount,
    };
    driver fd = {.reqs = &fork_req, .count = 1};
    const uint64_t forks_before = ctx_b->fork_admits;
    CHECK(!ds4_engine_continuous_generate(ctx_b, admit_cb, NULL, done_cb, &fd, err, sizeof(err)));
    CHECK(ctx_b->fork_admits == forks_before + 1u);
    CHECK(fork_req.out_n == GEN && !memcmp(fork_req.out, ref_fork, GEN * sizeof(int)));
    printf("Step cont: full-frontier fork clones %d committed rows and matches serial\n", fcount);

    /* Bank disk KV: snapshot the forked bank, restore in place, verify the
     * committed frontier reloads byte-exactly. */
    const int bank = fork_req.placed_bank;
    const uint64_t bytes = ds4_cont_bank_payload_bytes(ctx_b, (uint32_t)bank);
    CHECK(bytes != 0u);
    FILE *snap = tmpfile();
    CHECK(snap && !ds4_cont_bank_save_payload(ctx_b, (uint32_t)bank, snap, err, sizeof(err)));
    CHECK((uint64_t)ftell(snap) == bytes);
    for (unsigned il = 0; il < STEP37_LAYERS; il++) {
        ds4_step37_graph *g = &ctx_b->step37->graph[bank];
        CHECK(ds4_gpu_tensor_fill_f32(g->kv[il], 0, (uint64_t)g->kv_cap[il] * S37_KV));
    }
    if (e.mtp_ready) {
        ds4_step37_spec *s = &ctx_b->step37->spec[bank];
        for (unsigned il = STEP37_LAYERS; il < STEP37_LAYERS + STEP37_DRAFT_LAYERS; il++) {
            CHECK(ds4_gpu_tensor_fill_f32(s->draft.graph.kv[il], 0, (uint64_t)s->draft.graph.kv_cap[il] * S37_KV));
        }
        CHECK(ds4_gpu_tensor_fill_f32(s->tail, 0, STEP37_DRAFT_LAYERS * S37_HIDDEN));
    }
    CHECK(step37_batch_runtime_reset_bank(ctx_b->step37, (uint32_t)bank));
    rewind(snap);
    CHECK(!ds4_cont_bank_restore_payload(ctx_b, (uint32_t)bank, snap, bytes, err, sizeof(err)));
    FILE *roundtrip = tmpfile();
    CHECK(roundtrip && !ds4_cont_bank_save_payload(ctx_b, (uint32_t)bank, roundtrip, err, sizeof(err)));
    CHECK((uint64_t)ftell(roundtrip) == bytes);
    rewind(snap); rewind(roundtrip);
    unsigned char saved[4096], loaded[4096];
    size_t read;
    while ((read = fread(saved, 1, sizeof(saved), snap))) {
        CHECK(fread(loaded, 1, read, roundtrip) == read && !memcmp(saved, loaded, read));
    }
    CHECK(!ferror(snap));
    fclose(roundtrip);
    fclose(snap);
    const int *rt = NULL;
    const int rn = ds4_batch_ctx_bank_committed(ctx_b, bank, &rt);
    CHECK(rn == fcount + GEN - 1 && rt);
    for (int i = 0; i < fcount; i++) { CHECK(rt[i] == frontier[i]); }
    for (int i = 0; i < GEN - 1; i++) { CHECK(rt[fcount + i] == ref_fork[i]); }
    /* The restore must carry live frontier logits (not an all-zero
     * distribution), so the bank's argmax is the real pending token -- the
     * last token the fork generated but had not yet committed. */
    CHECK(ctx_b->step37->bank_logits_valid[bank]);
    const int restored_next =
        sample_argmax(ctx_b->step37->bank_logits + (size_t)bank * DS4_N_VOCAB, DS4_N_VOCAB);
    CHECK(restored_next == ref_fork[GEN - 1]);
    printf("Step cont: bank disk KV round-trips %d committed tokens; frontier logits live\n", rn);

    /* Disk restore changes lineage, so old prefix checkpoints cannot survive
     * it. Rebuild a source, then restore its exact prompt checkpoint after
     * decode has advanced beyond that frontier. */
    CHECK(step37_ckpt_find(ctx_b->step37, (uint32_t)bank, (uint32_t)len_a, (uint32_t)len_a) < 0);
    cont_req trunk = {.tokens = prompt.v, .n = len_a};
    driver td = {.reqs = &trunk, .count = 1};
    CHECK(!ds4_engine_continuous_generate(ctx_b, admit_cb, NULL, done_cb, &td, err, sizeof(err)));
    CHECK(ds4_batch_ctx_supports_partial_reuse(ctx_b));
    cont_req cut = {.tokens = prompt.v, .n = len_a,
                    .fork_bank = trunk.placed_bank + 1, .n_cached = len_a};
    driver pd = {.reqs = &cut, .count = 1};
    const uint64_t partial_before = ctx_b->fork_partial;
    CHECK(!ds4_engine_continuous_generate(ctx_b, admit_cb, NULL, done_cb, &pd, err, sizeof(err)));
    CHECK(ctx_b->fork_partial == partial_before + 1);
    CHECK(cut.out_n == GEN && !memcmp(cut.out, ref_a, GEN * sizeof(int)));
    if (e.mtp_ready) {
        CHECK(step37_spec_valid(&ctx_b->step37->spec[cut.placed_bank]));
        CHECK(ctx_b->step37->spec[cut.placed_bank].position == ctx_b->bank_hist_len[cut.placed_bank]);
    }
    printf("Step cont: partial fork restores %d-row prompt checkpoint, %d tokens exact\n", len_a, GEN);

    if (e.mtp_ready) {
        int ref0[GEN], ref1[GEN];
        ds4_session *s0 = serial_spec_reference(&e, ctx, prompt.v, len_a, ref0);
        ds4_session *s1 = serial_spec_reference(&e, ctx, prompt.v, len_b, ref1);
        cont_req pair[2] = {{.tokens = prompt.v, .n = len_a}, {.tokens = prompt.v, .n = len_b}};
        driver sd = {.reqs = pair, .count = 2};
        speculate = true;
        const uint64_t drafts = ds4_metric_read(&ds4_metrics_get()->spec_drafts);
        CHECK(!ds4_engine_continuous_generate(ctx_b, admit_cb, token_cb, done_cb, &sd, err, sizeof(err)));
        CHECK(ds4_metric_read(&ds4_metrics_get()->spec_drafts) > drafts);
        CHECK(pair[0].out_n == GEN && !memcmp(pair[0].out, ref0, sizeof(ref0)));
        CHECK(pair[1].out_n == GEN && !memcmp(pair[1].out, ref1, sizeof(ref1)));
        same_bank(ctx_b, (unsigned)pair[0].placed_bank, s0);
        same_bank(ctx_b, (unsigned)pair[1].placed_bank, s1);
        puts("Step cont: two-bank MTP matches serial tokens, full logits, target and predictor KV");

        const int *source = NULL;
        const int source_n = ds4_batch_ctx_bank_committed(ctx_b, pair[0].placed_bank, &source);
        ds4_tokens source_copy = {0};
        for (int i = 0; i < source_n; i++) { ds4_tokens_push(&source_copy, source[i]); }
        int continued[GEN];
        serial_spec_generate(s0, continued);
        cont_req sf = {.tokens = source_copy.v, .n = source_n,
                       .fork_bank = pair[0].placed_bank + 1, .n_cached = source_n};
        driver sfd = {.reqs = &sf, .count = 1};
        CHECK(!ds4_engine_continuous_generate(ctx_b, admit_cb, token_cb, done_cb, &sfd, err, sizeof(err)));
        CHECK(sf.out_n == GEN && !memcmp(sf.out, continued, sizeof(continued)));
        same_bank(ctx_b, (unsigned)sf.placed_bank, s0);
        ds4_tokens_free(&source_copy);
        ds4_session_free(s0); ds4_session_free(s1);

        s0 = serial_spec_reference(&e, ctx, prompt.v, len_a, ref0);
        cont_req sp = {.tokens = prompt.v, .n = len_a,
                       .fork_bank = pair[0].placed_bank + 1, .n_cached = len_a};
        driver spd = {.reqs = &sp, .count = 1};
        const uint64_t before = ctx_b->fork_partial;
        CHECK(!ds4_engine_continuous_generate(ctx_b, admit_cb, token_cb, done_cb, &spd, err, sizeof(err)));
        CHECK(ctx_b->fork_partial == before + 1 && sp.out_n == GEN && !memcmp(sp.out, ref0, sizeof(ref0)));
        same_bank(ctx_b, (unsigned)sp.placed_bank, s0);
        ds4_session_free(s0);
        puts("Step cont: MTP full and partial fork preserve target/predictor state exactly");

        cont_req cancel = {.tokens = prompt.v, .n = len_a, .stop_after = 2};
        driver cd = {.reqs = &cancel, .count = 1};
        CHECK(!ds4_engine_continuous_generate(ctx_b, admit_cb, token_cb, done_cb, &cd, err, sizeof(err)));
        CHECK(cancel.out_n == 2 && !memcmp(cancel.out, ref0, 2 * sizeof(int)));
        const unsigned b = (unsigned)cancel.placed_bank;
        CHECK(ctx_b->bank_hist_len[b] > (unsigned)len_a && ctx_b->bank_hist_len[b] <= (unsigned)len_a + 2);
        CHECK(step37_spec_valid(&ctx_b->step37->spec[b]));
        CHECK(ctx_b->step37->spec[b].position == ctx_b->bank_hist_len[b]);
        puts("Step cont: cancellation commits no unreported speculative tokens");

        cont_req forced = {.tokens = prompt.v, .n = len_a, .stop_after = 3,
                           .force_after = 2, .force_token = 42};
        driver forced_d = {.reqs = &forced, .count = 1};
        CHECK(!ds4_engine_continuous_generate(ctx_b, admit_cb, token_cb, done_cb, &forced_d, err, sizeof(err)));
        CHECK(forced.out_n == 3 && !memcmp(forced.out, ref0, 2 * sizeof(int)) && forced.out[2] == 42);
        CHECK(ctx_b->bank_hist_len[forced.placed_bank] == (unsigned)len_a + 2);

        cont_req stopped = {.tokens = prompt.v, .n = len_a, .stop_eos = true, .eos = ref0[1]};
        driver stopped_d = {.reqs = &stopped, .count = 1};
        CHECK(!ds4_engine_continuous_generate(ctx_b, admit_cb, token_cb, done_cb, &stopped_d, err, sizeof(err)));
        const int stop_n = ref0[0] == ref0[1] ? 1 : 2;
        CHECK(stopped.out_n == stop_n && !memcmp(stopped.out, ref0, stop_n * sizeof(int)));
        CHECK(ctx_b->bank_hist_len[stopped.placed_bank] <= (unsigned)len_a + stop_n);
        puts("Step cont: forced protocol token and EOS stop speculative emission");
    }

    ds4_batch_ctx_destroy(ctx_b);
    CHECK(!session_tensors_census_live());
    ds4_gpu_cleanup();
    model_close(&e.model);
    if (e.mtp_ready) { model_close(&e.mtp_model); }
    ds4_tokens_free(&prompt);
    puts("Step cont: two-bank parity, fork, and bank disk KV PASS");
    return 0;
}
