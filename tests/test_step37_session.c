/* One MQ83 mapping: public lifecycle versus the already gated eager graph. */
#include "../ds4.c"

static void check(bool ok, const char *message) {
    if (!ok) { ds4_die(message); }
}

static void emitted(void *out, int token) {
    *(int *)out = token;
}

static void policy(ds4_engine *e) {
    ds4_step37_graph g = {.cap = 64, .context = 1024, .position = 832};
    check(step37_rewind(&g, 768) && !step37_rewind(&g, 767),
          "Step repeated rewind exceeded retained KV");
    check(!ds4_engine_supports_batching(e), "Step entered DeepSeek batching");
    ds4_session_graph_fit_quote q;
    setenv("DS4_SESSION_GRAPH_FIT", "0", 1);
    const int contexts[] = {1, 64, 512, 1024, 262144};
    for (unsigned i = 0; i < sizeof(contexts) / sizeof(contexts[0]); i++) {
        int ctx = contexts[i];
        ds4_context_memory m = ds4_context_memory_estimate(e->backend, ctx);
        check(m.total_bytes && m.total_bytes == ds4_engine_session_graph_bytes_estimate(e, ctx) &&
              ds4_engine_session_graph_fit_quote(e, ctx, &q) && q.fail_open &&
              q.need_bytes == m.total_bytes, "Step memory quotes disagree");
    }
    check(!ds4_engine_session_graph_fit_quote(e, 262145, &q) &&
          !ds4_engine_session_graph_bytes_estimate(e, 262145), "Step oversized context admitted");
    e->backend = DS4_BACKEND_CPU;
    check(!ds4_engine_session_graph_fit_quote(e, 32, &q), "Step CPU context admitted");
    e->backend = DS4_BACKEND_CUDA;
    unsetenv("DS4_SESSION_GRAPH_FIT");
    puts("Step session policy: context, memory and repeated rewind PASS");
}

static void same(ds4_session *s, ds4_step37_graph *r) {
    const unsigned end = (unsigned)ds4_session_pos(s);
    check(r->position == end, "Step reference frontier differs");
    float *logits = xmalloc(DS4_N_VOCAB * sizeof(float));
    check(ds4_gpu_tensor_read(r->logits, 0, logits, DS4_N_VOCAB * sizeof(float)) &&
          !memcmp(logits, s->logits, DS4_N_VOCAB * sizeof(float)), "Step session logits differ");
    free(logits);
    for (unsigned il = 0; il < STEP37_LAYERS; il++) {
        unsigned rows = end < r->kv_cap[il] ? end : r->kv_cap[il];
        const size_t bytes = (size_t)rows * 2 * S37_KV * sizeof(uint16_t);
        void *a = xmalloc(bytes), *b = xmalloc(bytes);
        check(ds4_gpu_tensor_read(s->step37_graph.kv[il], 0, a, bytes) &&
              ds4_gpu_tensor_read(r->kv[il], 0, b, bytes) && !memcmp(a, b, bytes),
              "Step session KV differs");
        free(a); free(b);
    }
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) { return 2; }
    const ds4_host_shape host = {.variant = DS4_VARIANT_STEP37_FLASH};
    ds4_host_shape_install(&host);
    model_apply_host_shape();
    ds4_host_shape_clear();
    ds4_engine e = {.backend = DS4_BACKEND_CUDA, .metal_ready = true};
    policy(&e);
    if (argc == 2 && !strcmp(argv[1], "--policy")) { return 0; }
    if (argc != 3) { return 2; }
    model_open(&e.model, argv[1], false, false);
    weights_bind(&e.weights, &e.model, false, 0, UINT32_MAX, true, false);
    e.vocab.n_vocab = DS4_N_VOCAB;
    check(ds4_gpu_init() && ds4_gpu_set_model_map(e.model.map, e.model.size), "Step GPU init");
    ds4_tokens prompt = {0};
    FILE *input = fopen(argv[2], "r");
    check(input != NULL, "Step token fixture missing");
    int token;
    while (fscanf(input, "%d", &token) == 1) { ds4_tokens_push(&prompt, token); }
    fclose(input);
    check(prompt.len > 128 && prompt.len % 64 == 0, "Step fixture needs multiple 64-row chunks");
    setenv("DS4_STEP37_PREFILL_CHUNK", "64", 1);
    setenv("DS4_SESSION_LAZY_GRAPH", "1", 1);
    ds4_session *s = NULL;
    const int ctx = prompt.len + 16;
    check(!ds4_session_create(&s, &e, ctx) && s && ds4_session_graph_pending(s) &&
          !ds4_session_graph_bytes_committed(s), "Step lazy session create");
    const uint64_t estimate = ds4_engine_session_graph_bytes_estimate(&e, ctx);
    char err[256] = {0};
    prompt.len -= 64;
    check(!ds4_session_sync(s, &prompt, err, sizeof(err)), err);
    uint64_t extended_generation = ds4_session_generation(s);
    prompt.len += 64;
    check(!ds4_session_sync(s, &prompt, err, sizeof(err)) &&
          ds4_session_generation(s) == extended_generation, "Step prefix extension rebuilt state");
    const uint64_t measured = ds4_session_graph_bytes_committed(s);
    check(measured >= estimate && measured - estimate < 1048576, "Step memory estimate drift");
    const uint64_t generation = ds4_session_generation(s);
    check(!ds4_session_sync(s, &prompt, err, sizeof(err)) &&
          generation == ds4_session_generation(s), "Step no-op sync invalidated state");
    const int saved = prompt.v[prompt.len - 1];
    prompt.v[prompt.len - 1] = -1;
    check(ds4_session_sync(s, &prompt, err, sizeof(err)) &&
          ds4_session_eval(s, DS4_N_VOCAB, err, sizeof(err)) &&
          ds4_session_pos(s) == prompt.len && generation == ds4_session_generation(s),
          "Step invalid input changed checkpoint");
    prompt.v[prompt.len - 1] = saved;
    ds4_step37_graph reference;
    check(step37_graph_alloc(&reference, &e.model, &e.weights, ctx, 64), "Step reference alloc");
    for (unsigned pos = 0; pos < (unsigned)prompt.len; pos += 64) {
        check(step37_forward(&reference, &e.model, &e.weights, prompt.v + pos, 64, pos),
              "Step reference prefill");
    }
    same(s, &reference);
    token = ds4_session_argmax(s);
    check(!ds4_session_eval(s, token, err, sizeof(err)) &&
          step37_forward(&reference, &e.model, &e.weights, &token, 1, prompt.len), err);
    same(s, &reference);
    ds4_tokens_push(&prompt, token);
    const int end = prompt.len;
    ds4_session_rewind(s, end);
    check(!ds4_session_sync(s, &prompt, err, sizeof(err)), "Step no-op rewind");
    ds4_session_rewind(s, end - 1);
    check(ds4_session_eval(s, token, err, sizeof(err)), "Step rewind exposed stale logits");
    check(!ds4_session_sync(s, &prompt, err, sizeof(err)), err);
    /* Public rewind can rebuild. Compare with the same complete chunk plan. */
    check(step37_rewind(&reference, 0), "Step reference reset");
    for (unsigned pos = 0; pos < (unsigned)prompt.len;) {
        unsigned n = (unsigned)prompt.len - pos;
        if (n > 64) { n = 64; }
        check(step37_forward(&reference, &e.model, &e.weights, prompt.v + pos, n, pos), "Step reference replay");
        pos += n;
    }
    same(s, &reference);
    ds4_session_invalidate(s);
    check(ds4_session_eval(s, token, err, sizeof(err)), "Step invalidated decode accepted");
    check(!ds4_session_sync(s, &prompt, err, sizeof(err)), err);
    same(s, &reference);
    FILE *fp = tmpfile();
    check(fp && !ds4_session_payload_bytes(s) && ds4_session_save_payload(s, fp, err, sizeof(err)) &&
          ds4_session_load_payload(s, fp, 0, err, sizeof(err)), "Step entered DeepSeek disk payload");
    fclose(fp);
    const int expected = ds4_session_argmax(s);
    step37_graph_free(&reference);
    ds4_session_free(s);
    check(!session_tensors_census_live(), "Step leaked session tensors");
    int generated = -1;
    check(!ds4_engine_generate_argmax(&e, &prompt, 1, ctx, emitted, NULL, &generated, NULL, NULL) &&
          generated == expected && !session_tensors_census_live(), "Step public generation dispatch");
    ds4_gpu_cleanup();
    model_close(&e.model);
    ds4_tokens_free(&prompt);
    printf("Step session: lazy/no-op/invalid/decode/rewind/reset logits and 45-layer KV PASS; "
           "estimate=%llu measured=%llu\n", (unsigned long long)estimate, (unsigned long long)measured);
    return 0;
}
