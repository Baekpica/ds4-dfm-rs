/* Step continuous lane: two banks must decode byte-identically to two
 * independent serial sessions, fork must clone a committed prefix, and a bank
 * disk-KV checkpoint must reload and continue.  One MQ83 mapping, base only
 * (the banked lane runs ordinary decode; MTP stays on the serial session). */
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "Step cont FAIL line %d: %s (%s)\n", __LINE__, #x, err); exit(1); \
} } while (0)
static char err[256];

enum { GEN = 16, MAX_REQ = 4 };

typedef struct {
    const int *tokens;
    int n;
    int fork_bank; /* +1, or 0 */
    int n_cached;
    int out[GEN];
    int out_n;
    int placed_bank;
} cont_req;

typedef struct {
    cont_req *reqs;
    int count, next;
} driver;

static int admit_cb(void *ud, ds4_cont_request *req) {
    driver *d = ud;
    if (d->next >= d->count) { return 0; }
    cont_req *r = &d->reqs[d->next++];
    memset(req, 0, sizeof(*req));
    req->tokens = r->tokens;
    req->n = r->n;
    req->max_new = GEN;
    req->eos = -1;              /* run the full GEN budget, ignore EOS */
    req->temperature = 0.0f;    /* greedy argmax */
    req->sample_override = NULL;
    req->user = r;
    req->bank_used = &r->placed_bank;
    req->fork_bank = r->fork_bank;
    req->n_cached = r->n_cached;
    return 1;
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

int main(int argc, char **argv) {
    if (argc != 3) { return 2; }
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

    cont_req reqs[2] = {
        {.tokens = prompt.v, .n = len_a},
        {.tokens = prompt.v, .n = len_b},
    };
    driver d = {.reqs = reqs, .count = 2};
    CHECK(!ds4_engine_continuous_generate(ctx_b, admit_cb, NULL, done_cb, &d, err, sizeof(err)));
    CHECK(reqs[0].out_n == GEN && reqs[1].out_n == GEN);
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
    rewind(snap);
    CHECK(!ds4_cont_bank_restore_payload(ctx_b, (uint32_t)bank, snap, bytes, err, sizeof(err)));
    fclose(snap);
    const int *rt = NULL;
    const int rn = ds4_batch_ctx_bank_committed(ctx_b, bank, &rt);
    CHECK(rn == fcount + GEN - 1 && rt);
    for (int i = 0; i < fcount; i++) { CHECK(rt[i] == frontier[i]); }
    for (int i = 0; i < GEN - 1; i++) { CHECK(rt[fcount + i] == ref_fork[i]); }
    printf("Step cont: bank disk KV round-trips %d committed tokens\n", rn);

    ds4_batch_ctx_destroy(ctx_b);
    CHECK(!session_tensors_census_live());
    ds4_gpu_cleanup();
    model_close(&e.model);
    ds4_tokens_free(&prompt);
    puts("Step cont: two-bank parity, fork, and bank disk KV PASS");
    return 0;
}
