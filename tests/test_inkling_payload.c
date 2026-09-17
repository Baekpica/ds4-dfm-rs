/* Payload lifecycle without weights, including physical local-ring wrap. */
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static void fill_tensor(ds4_gpu_tensor *tensor, size_t bytes, unsigned seed) {
    unsigned char *data = xmalloc(bytes);
    for (size_t i = 0; i < bytes; i++) { data[i] = (unsigned char)(i * 13 + seed); }
    CHECK(ds4_gpu_tensor_write(tensor, 0, data, bytes));
    free(data);
}

static void check_tensor(ds4_gpu_tensor *tensor, size_t bytes, unsigned seed) {
    unsigned char *data = xmalloc(bytes);
    CHECK(ds4_gpu_tensor_read(tensor, 0, data, bytes));
    for (size_t i = 0; i < bytes; i++) {
        const unsigned char want = (unsigned char)(i * 13 + seed);
        if (data[i] != want) {
            fprintf(stderr, "state byte %zu/%zu seed=%u got=%u expected=%u\n",
                    i, bytes, seed, data[i], want);
            CHECK(data[i] == want);
        }
    }
    free(data);
}

static void check_graph(ds4_inkling_graph *g, unsigned n) {
    for (unsigned i = 0; i < g->n_layers; i++) {
        inkling_layer_state *s = &g->layer[i];
        const unsigned rows = n < s->capacity ? n : s->capacity;
        if (rows) { check_tensor(s->kv, (size_t)rows * 2 * IK_KV * sizeof(uint16_t), i); }
        for (unsigned j = 0; j < IK_CONV_STREAMS; j++) {
            check_tensor(s->conv[j], (size_t)IK_HISTORY * (j < 2 ? IK_KV : IK_HIDDEN) * sizeof(float),
                         i + j + 1);
        }
    }
}

static void fake_graph(ds4_inkling_graph *g, unsigned layers, unsigned n, unsigned ctx) {
    g->context = ctx;
    g->cap = 1;
    g->position = n;
    g->n_layers = layers;
    for (unsigned i = 0; i < layers; i++) {
        inkling_layer_state *s = &g->layer[i];
        const bool global = layers == INKLING_LAYERS ? i % 6 == 5 : (i == 1 || i == 3);
        s->capacity = global ? ctx : IK_LOCAL;
        size_t bytes = (size_t)s->capacity * 2 * IK_KV * sizeof(uint16_t);
        s->kv = ds4_gpu_tensor_alloc(bytes);
        CHECK(s->kv);
        fill_tensor(s->kv, bytes, i);
        for (unsigned j = 0; j < IK_CONV_STREAMS; j++) {
            bytes = (size_t)IK_HISTORY * (j < 2 ? IK_KV : IK_HIDDEN) * sizeof(float);
            s->conv[j] = ds4_gpu_tensor_alloc(bytes);
            CHECK(s->conv[j]);
            fill_tensor(s->conv[j], bytes, i + j + 1);
        }
    }
}

static void roundtrip(unsigned n, unsigned mtp) {
    ds4_engine engine = {.backend = DS4_BACKEND_CUDA, .metal_ready = true, .mtp_ready = mtp != 0};
    ds4_session s = {.engine = &engine, .ctx_size = IK_LOCAL + 64,
                     .checkpoint_valid = true, .inkling_graph_ready = true};
    fake_graph(&s.inkling_graph, INKLING_LAYERS, n, s.ctx_size);
    if (mtp) {
        ds4_inkling_spec *spec = &s.inkling_spec;
        spec->position = n;
        spec->tail_rows = n < INKLING_DRAFT_LAYERS ? n : INKLING_DRAFT_LAYERS;
        const unsigned stable = n - spec->tail_rows;
        fake_graph(&spec->draft.graph, INKLING_DRAFT_LAYERS, stable, s.ctx_size);
        for (unsigned i = 0; i < INKLING_DRAFT_LAYERS; i++) { spec->draft.positions[i] = stable; }
        const size_t bytes = (size_t)INKLING_DRAFT_LAYERS * IK_HIDDEN * sizeof(float);
        spec->tail = ds4_gpu_tensor_alloc(bytes);
        CHECK(spec->tail);
        fill_tensor(spec->tail, bytes, 127);
    }
    s.logits = xcalloc(DS4_N_VOCAB, sizeof(float));
    for (unsigned i = 0; i < n; i++) { ds4_tokens_push(&s.checkpoint, (int)i + 1); }
    for (unsigned i = 0; i < DS4_N_VOCAB; i++) { s.logits[i] = (float)i * 0.125f; }
    ds4_session_snapshot saved = {0}, again = {0};
    char err[256] = {0};
    CHECK(ds4_session_payload_bytes(&s) > 0);
    CHECK(ds4_session_save_snapshot(&s, &saved, err, sizeof(err)) == 0);
    CHECK(saved.len == ds4_session_payload_bytes(&s));
    CHECK(ds4_session_load_snapshot(&s, &saved, err, sizeof(err)) == 0);
    CHECK(ds4_session_pos(&s) == (int)n);
    check_graph(&s.inkling_graph, n);
    if (mtp) {
        check_graph(&s.inkling_spec.draft.graph, n - s.inkling_spec.tail_rows);
        check_tensor(s.inkling_spec.tail,
                     (size_t)s.inkling_spec.tail_rows * IK_HIDDEN * sizeof(float), 127);
    }
    CHECK(ds4_session_save_snapshot(&s, &again, err, sizeof(err)) == 0);
    CHECK(saved.len == again.len && !memcmp(saved.ptr, again.ptr, saved.len));

    /* Reject incomplete, wrong-family, wrong-token and uncommitted state. */
    saved.len--;
    CHECK(ds4_session_load_snapshot(&s, &saved, err, sizeof(err)) != 0);
    CHECK(!s.checkpoint_valid && !ds4_session_payload_bytes(&s));
    saved.len++;
    CHECK(ds4_session_load_snapshot(&s, &saved, err, sizeof(err)) == 0);
    unsigned char *raw = saved.ptr;
    raw[5 * sizeof(uint32_t)] ^= 1;
    CHECK(ds4_session_load_snapshot(&s, &saved, err, sizeof(err)) != 0);
    raw[5 * sizeof(uint32_t)] ^= 1;
    CHECK(ds4_session_load_snapshot(&s, &saved, err, sizeof(err)) == 0);
    raw[8 * sizeof(uint32_t)] ^= INKLING_PAYLOAD_MTP;
    CHECK(ds4_session_load_snapshot(&s, &saved, err, sizeof(err)) != 0);
    raw[8 * sizeof(uint32_t)] ^= INKLING_PAYLOAD_MTP;
    const unsigned invalid[] = {INKLING_VALID_VOCAB, IK_IMAGE_TOKEN, IK_AUDIO_TOKEN};
    for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        payload_put_u32(raw + DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t), invalid[i]);
        CHECK(ds4_session_load_snapshot(&s, &saved, err, sizeof(err)) != 0);
        CHECK(!s.checkpoint_valid && !ds4_session_payload_bytes(&s));
    }
    payload_put_u32(raw + DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t), 1);
    CHECK(ds4_session_load_snapshot(&s, &saved, err, sizeof(err)) == 0);
    s.inkling_trial_n = 1;
    CHECK(!ds4_session_payload_bytes(&s));
    s.inkling_trial_n = 0;
    s.checkpoint.v[0] = IK_IMAGE_TOKEN;
    CHECK(!ds4_session_payload_bytes(&s));
    s.checkpoint.v[0] = IK_AUDIO_TOKEN;
    CHECK(!ds4_session_payload_bytes(&s));
    s.checkpoint.v[0] = 1;
    if (mtp) {
        s.inkling_spec.draft.positions[0]++;
        CHECK(!ds4_session_payload_bytes(&s));
        s.inkling_spec.draft.positions[0]--;
    }
    printf("Inkling payload: n=%u mtp=%u bytes=%llu exact roundtrip and rejection passed\n",
           n, mtp, (unsigned long long)saved.len);
    ds4_session_snapshot_free(&saved);
    ds4_session_snapshot_free(&again);
    token_vec_free(&s.checkpoint);
    free(s.logits);
    inkling_graph_free(&s.inkling_graph);
    if (mtp) { inkling_spec_free(&s.inkling_spec); }
}

int main(void) {
    const ds4_host_shape host = {.variant = DS4_VARIANT_INKLING_SMALL};
    ds4_host_shape_install(&host);
    model_apply_host_shape();
    ds4_host_shape_clear();
    CHECK(ds4_gpu_init());
    const unsigned rows[] = {3, IK_LOCAL - 1, IK_LOCAL + 17};
    for (unsigned mtp = 0; mtp < 2; mtp++) {
        for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); i++) { roundtrip(rows[i], mtp); }
    }
    ds4_gpu_cleanup();
    return 0;
}
