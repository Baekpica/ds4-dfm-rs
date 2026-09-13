/* One main mapping plus owner-imported MTP. Compare committed transitions
 * with an independent, width-matched target graph and a width-one control. */
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "Step speculation FAIL line %d: %s (%s)\n", __LINE__, #x, err); exit(1); \
} } while (0)
static char err[256];

static void same_kv(ds4_step37_graph *g, ds4_step37_graph *r) {
    CHECK(g->position == r->position);
    for (unsigned il = 0; il < STEP37_LAYERS; il++) {
        const unsigned cap = g->kv_cap[il];
        CHECK(cap == r->kv_cap[il]);
        const size_t row_bytes = 2 * S37_KV * sizeof(uint16_t), bytes = cap * row_bytes;
        unsigned char *a = xmalloc(bytes), *b = xmalloc(bytes);
        CHECK(ds4_gpu_tensor_read(g->kv[il], 0, a, bytes) && ds4_gpu_tensor_read(r->kv[il], 0, b, bytes));
        unsigned start = step37_sliding(il) && g->position > S37_WINDOW ? g->position - S37_WINDOW : 0;
        for (unsigned pos = start; pos < g->position; pos++) {
            CHECK(!memcmp(a + (pos % cap) * row_bytes, b + (pos % cap) * row_bytes, row_bytes));
        }
        free(a); free(b);
    }
}

int main(int argc, char **argv) {
    CHECK(argc == 5);
    const ds4_host_shape host = {.variant = DS4_VARIANT_STEP37_FLASH};
    ds4_host_shape_install(&host); model_apply_host_shape(); ds4_host_shape_clear();
    ds4_engine e = {.backend = DS4_BACKEND_CUDA, .metal_ready = true,
                    .mtp_ready = true, .mtp_draft_tokens = STEP37_DRAFT_LAYERS};
    model_open(&e.model, argv[1], false, false);
    weights_bind(&e.weights, &e.model, false, 0, UINT32_MAX, true, false);
    model_open(&e.mtp_model, argv[2], false, false);
    step37_bind_draft(&e.step37_mtp, &e.mtp_model);
    e.vocab.n_vocab = DS4_N_VOCAB;
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(e.model.map, e.model.size));
    CHECK(ds4_gpu_import_model_ipc_manifest(e.mtp_model.map, e.mtp_model.size, argv[3], "mtp"));
    model_release_mapping_cache(&e.mtp_model);
    ds4_tokens prompt = {0};
    FILE *fp = fopen(argv[4], "r"); CHECK(fp);
    int token;
    while (fscanf(fp, "%d", &token) == 1) { ds4_tokens_push(&prompt, token); }
    CHECK(fclose(fp) == 0 && prompt.len > 0);
    enum { GENERATED = 32, CHUNK = 64 };
    const unsigned ctx = (unsigned)prompt.len + GENERATED + S37_VERIFY;
    setenv("DS4_STEP37_PREFILL_CHUNK", "64", 1);
    setenv("DS4_SESSION_LAZY_GRAPH", "1", 1);
    ds4_session *s = NULL;
    CHECK(!ds4_session_create(&s, &e, (int)ctx) && ds4_session_graph_pending(s));
    const uint64_t estimate = ds4_engine_session_graph_bytes_estimate(&e, (int)ctx);
    CHECK(!ds4_session_sync(s, &prompt, err, sizeof(err)));
    CHECK(step37_spec_valid(&s->step37_spec) && s->step37_spec.position == (unsigned)prompt.len);
    CHECK(ds4_session_graph_bytes_committed(s) == estimate);
    const uint64_t generation = s->generation;
    CHECK(!ds4_session_sync(s, &prompt, err, sizeof(err)) && s->generation == generation);
    ds4_step37_graph reference, serial;
    CHECK(step37_graph_alloc(&reference, &e.model, &e.weights, ctx, ctx < CHUNK ? ctx : CHUNK) &&
          step37_graph_alloc(&serial, &e.model, &e.weights, ctx, ctx < CHUNK ? ctx : CHUNK));
    for (unsigned pos = 0; pos < (unsigned)prompt.len;) {
        unsigned n = (unsigned)prompt.len - pos;
        if (n > CHUNK) { n = CHUNK; }
        CHECK(step37_forward(&reference, &e.model, &e.weights, prompt.v + pos, n, pos));
        CHECK(step37_forward(&serial, &e.model, &e.weights, prompt.v + pos, n, pos));
        pos += n;
    }
    same_kv(&s->step37_graph, &reference);
    const size_t logbytes = DS4_N_VOCAB * sizeof(float);
    float *logits = xmalloc(logbytes), *control = xmalloc(logbytes);
    unsigned generated = 0, cycles = 0, proposed = 0, accepted = 0;
    unsigned kept_mask = 0;
    const bool truncate = getenv("STEP37_TEST_TRUNCATE") != NULL;
    while (generated < GENERATED) {
        int tokens[S37_VERIFY], target[S37_VERIFY];
        const int first = ds4_session_argmax(s), before = ds4_session_pos(s);
        const int n = ds4_session_step37_trial(s, first, GENERATED - (int)generated,
                                               tokens, target, S37_VERIFY, err, sizeof(err));
        CHECK(n > 0 && n <= S37_VERIFY && tokens[0] == first && ds4_session_pos(s) == before);
        CHECK(ds4_session_argmax(s) == -1 && ds4_session_eval(s, first, err, sizeof(err)) &&
              ds4_session_sync(s, &prompt, err, sizeof(err)) &&
              ds4_session_step37_trial(s, first, 1, tokens, target, S37_VERIFY, err, sizeof(err)) < 0);
        CHECK(ds4_session_step37_commit(s, 0, err, sizeof(err)) &&
              ds4_session_step37_commit(s, n + 1, err, sizeof(err)) && ds4_session_pos(s) == before);
        CHECK(step37_forward(&reference, &e.model, &e.weights, tokens, (unsigned)n, reference.position));
        for (int row = 0; row < n; row++) {
            CHECK(step37_head(&reference, &e.model, &e.weights, (unsigned)row) &&
                  ds4_gpu_tensor_read(reference.logits, 0, logits, logbytes));
            CHECK(target[row] == sample_argmax(logits, DS4_N_VOCAB));
            CHECK(step37_head(&s->step37_graph, &e.model, &e.weights, (unsigned)row) &&
                  ds4_gpu_tensor_read(s->step37_graph.logits, 0, control, logbytes));
            CHECK(!memcmp(logits, control, logbytes));
        }
        int keep = 1;
        while (keep < n && tokens[keep] == target[keep - 1]) { keep++; }
        /* A caller may stop before the longest accepted prefix. Exercise
         * every commit length while preserving greedy token verification. */
        if (truncate && cycles < S37_VERIFY && keep > (int)cycles + 1) { keep = (int)cycles + 1; }
        CHECK(!ds4_session_step37_commit(s, keep, err, sizeof(err)));
        CHECK(ds4_session_step37_commit(s, keep, err, sizeof(err)));
        CHECK(step37_rewind(&reference, (unsigned)before + (unsigned)keep) &&
              step37_head(&reference, &e.model, &e.weights, (unsigned)keep - 1) &&
              ds4_gpu_tensor_read(reference.logits, 0, logits, logbytes));
        CHECK(!memcmp(logits, s->logits, logbytes));
        same_kv(&s->step37_graph, &reference);
        for (int i = 0; i < keep; i++) {
            CHECK(ds4_gpu_tensor_read(serial.logits, 0, control, logbytes));
            CHECK(tokens[i] == sample_argmax(control, DS4_N_VOCAB));
            CHECK(step37_forward(&serial, &e.model, &e.weights, tokens + i, 1, serial.position));
            ds4_tokens_push(&prompt, tokens[i]);
        }
        CHECK(step37_spec_valid(&s->step37_spec) && s->step37_spec.position == reference.position);
        printf("cycle %u: draft=%d keep=%d; complete logits/live KV exact, mode-0 tokens match\n", cycles, n - 1, keep);
        fflush(stdout);
        proposed += (unsigned)n - 1; accepted += (unsigned)keep - 1;
        kept_mask |= 1u << ((unsigned)keep - 1);
        generated += (unsigned)keep; cycles++;
    }
    CHECK(!truncate || kept_mask == (1u << S37_VERIFY) - 1);
    ds4_session_rewind(s, s->checkpoint.len - 1);
    CHECK(!s->checkpoint_valid && step37_spec_valid(&s->step37_spec) && !s->step37_spec.position);
    CHECK(!ds4_session_sync(s, &prompt, err, sizeof(err)));
    ds4_session_invalidate(s);
    CHECK(!s->checkpoint_valid && !s->step37_spec.position && !s->step37_trial_n);
    printf("Step speculation PASS: %u tokens, %u cycles, %u/%u draft acceptance; memory=%" PRIu64 "\n",
           generated, cycles, accepted, proposed, estimate);
    step37_graph_free(&reference); step37_graph_free(&serial);
    ds4_session_free(s); ds4_gpu_cleanup(); model_close(&e.mtp_model); model_close(&e.model);
    ds4_tokens_free(&prompt); free(logits); free(control);
    return 0;
}
