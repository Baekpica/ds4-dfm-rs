/* Public native session lifecycle over one real MQ85GB model mapping. */
#include "../ds4.c"

static void check(int ok, const char *message) {
    if (!ok) {
        ds4_die(message);
    }
}

static void check_image_sync(ds4_session *s) {
    float pixels[2 * 40 * 40 * 3];
    for (unsigned i = 0; i < sizeof(pixels) / sizeof(pixels[0]); i++) {
        pixels[i] = ((int)(i * 13 % 257) - 128) / 63.0f;
    }
    int ids[] = {200000, 200005, 200054, 200010, 200001};
    ds4_tokens prompt = {.v = ids, .len = 5, .cap = 5};
    ds4_inkling_pixels image = {.pixels = pixels, .pixel_count = 2 * 40 * 40 * 3,
                                .token_offset = 2, .token_count = 1};
    char err[256] = {0};
    size_t bytes = INKLING_VALID_VOCAB * sizeof(float);
    float *first = xmalloc(bytes), *second = xmalloc(bytes), *got = xmalloc(bytes);
    check(ds4_session_sync_inkling(s, &prompt, &image, 1, NULL, 0, err, sizeof(err)) == 0, err);
    memcpy(first, s->logits, bytes);
    uint64_t generation = ds4_session_generation(s);
    check(ds4_session_sync_inkling(s, &prompt, &image, 1, NULL, 0, err, sizeof(err)) == 0 &&
          ds4_session_generation(s) > generation && memcmp(first, s->logits, bytes) == 0,
          "repeated Inkling image changed logits or reused token-only identity");
    for (unsigned i = 0; i < sizeof(pixels) / sizeof(pixels[0]); i++) {
        pixels[i] = -pixels[i];
    }
    check(ds4_session_sync_inkling(s, &prompt, &image, 1, NULL, 0, err, sizeof(err)) == 0, err);
    memcpy(second, s->logits, bytes);
    check(memcmp(first, second, bytes) != 0, "changed Inkling image reused old KV");
    generation = ds4_session_generation(s);
    check(ds4_session_sync(s, &prompt, err, sizeof(err)) != 0,
          "Inkling image placeholders accepted without pixels");
    image.pixel_count--;
    check(ds4_session_sync_inkling(s, &prompt, &image, 1, NULL, 0, err, sizeof(err)) != 0,
          "Inkling image pixel length mismatch accepted");
    image.pixel_count++;
    image.token_offset = 1;
    check(ds4_session_sync_inkling(s, &prompt, &image, 1, NULL, 0, err, sizeof(err)) != 0,
          "Inkling image features accepted on text token");
    image.token_offset = 2;
    pixels[0] = NAN;
    check(ds4_session_sync_inkling(s, &prompt, &image, 1, NULL, 0, err, sizeof(err)) != 0 &&
          ds4_session_generation(s) == generation && ds4_session_pos(s) == prompt.len,
          "invalid Inkling pixels changed checkpoint frontier");
    check(ds4_session_copy_logits(s, got, INKLING_VALID_VOCAB) == INKLING_VALID_VOCAB &&
          memcmp(second, got, bytes) == 0, "invalid Inkling image changed logits");
    free(first); free(second); free(got);
    puts("Inkling image session: repeat/changed-image identity and invalid-input state passed");
}

static void check_audio_sync(ds4_session *s) {
    int32_t codes[2 * IK_AUDIO_BINS];
    float pixels[IK_IMAGE_PIXELS] = {0};
    for (unsigned i = 0; i < 2 * IK_AUDIO_BINS; i++) {
        codes[i] = (i * 7 + 3) % IK_AUDIO_LEVELS;
    }
    int ids[] = {200000, 200005, 200054, 200020, 200053, 200053, 200043, 200010, 200001};
    ds4_tokens prompt = {.v = ids, .len = 9, .cap = 9};
    ds4_inkling_pixels image = {.pixels = pixels, .pixel_count = IK_IMAGE_PIXELS,
                                .token_offset = 2, .token_count = 1};
    ds4_inkling_audio audio = {.codes = codes, .code_count = 2 * IK_AUDIO_BINS,
                               .token_offset = 4, .token_count = 2};
    char err[256] = {0};
    const size_t bytes = INKLING_VALID_VOCAB * sizeof(float);
    float *first = xmalloc(bytes), *second = xmalloc(bytes);
    check(ds4_session_sync_inkling(s, &prompt, &image, 1, &audio, 1, err, sizeof(err)) == 0, err);
    memcpy(first, s->logits, bytes);
    uint64_t generation = ds4_session_generation(s);
    check(ds4_session_sync_inkling(s, &prompt, &image, 1, &audio, 1, err, sizeof(err)) == 0 &&
          ds4_session_generation(s) > generation && memcmp(first, s->logits, bytes) == 0,
          "repeated Inkling audio changed logits or reused token identity");
    for (unsigned i = 0; i < 2 * IK_AUDIO_BINS; i++) {
        codes[i] = 15 - codes[i];
    }
    check(ds4_session_sync_inkling(s, &prompt, &image, 1, &audio, 1, err, sizeof(err)) == 0 &&
          memcmp(first, s->logits, bytes) != 0, "changed Inkling audio reused old KV");
    memcpy(second, s->logits, bytes);
    generation = ds4_session_generation(s);
    check(ds4_session_sync_inkling(s, &prompt, &image, 1, NULL, 0, err, sizeof(err)) != 0,
          "uncovered Inkling audio accepted");
    audio.code_count--;
    check(ds4_session_sync_inkling(s, &prompt, &image, 1, &audio, 1, err, sizeof(err)) != 0,
          "wrong Inkling code length accepted");
    audio.code_count++;
    audio.token_offset = 2;
    check(ds4_session_sync_inkling(s, &prompt, &image, 1, &audio, 1, err, sizeof(err)) != 0,
          "overlapping Inkling audio/image accepted");
    audio.token_offset = 4;
    codes[159] = 16;
    check(ds4_session_sync_inkling(s, &prompt, &image, 1, &audio, 1, err, sizeof(err)) != 0 &&
          ds4_session_generation(s) == generation && ds4_session_pos(s) == prompt.len &&
          memcmp(second, s->logits, bytes) == 0, "invalid Inkling audio changed checkpoint");
    codes[159] = 4;
    prompt.v = ids + 3; prompt.len = 6;
    audio.token_offset = 1;
    check(ds4_session_sync_inkling(s, &prompt, NULL, 0, &audio, 1, err, sizeof(err)) == 0,
          "audio-only Inkling sync failed");
    free(first); free(second);
    puts("Inkling audio session: mixed/audio-only, repeat/change and invalid state passed");
}

static void check_targets(const ds4_session *s, const ds4_session *reference) {
    const ds4_inkling_graph *g = &s->inkling_graph, *r = &reference->inkling_graph;
    check(g->position == r->position && g->position == (unsigned)s->checkpoint.len,
          "Inkling MTP target frontier differs");
    check(memcmp(s->logits, reference->logits, INKLING_VALID_VOCAB * sizeof(float)) == 0,
          "Inkling MTP target logits differ");
    const size_t capacity = (size_t)(g->context + IK_LOCAL) * 2 * IK_KV * sizeof(uint16_t);
    unsigned char *got = xmalloc(capacity), *want = xmalloc(capacity);
    for (unsigned i = 0; i < INKLING_LAYERS; i++) {
        const unsigned valid = g->position < g->layer[i].capacity ? g->position : g->layer[i].capacity;
        size_t bytes = (size_t)valid * 2 * IK_KV * sizeof(uint16_t);
        check(ds4_gpu_tensor_read(g->layer[i].kv, 0, got, bytes) &&
              ds4_gpu_tensor_read(r->layer[i].kv, 0, want, bytes) && memcmp(got, want, bytes) == 0,
              "Inkling MTP committed target KV differs");
        for (unsigned j = 0; j < IK_CONV_STREAMS; j++) {
            bytes = IK_HISTORY * (j < 2 ? IK_KV : IK_HIDDEN) * sizeof(float);
            check(ds4_gpu_tensor_read(g->layer[i].conv[j], 0, got, bytes) &&
                  ds4_gpu_tensor_read(r->layer[i].conv[j], 0, want, bytes) && memcmp(got, want, bytes) == 0,
                  "Inkling MTP committed target convolution differs");
        }
    }
    free(got); free(want);
}

static void check_mtp(ds4_session *s, const ds4_tokens *prompt) {
    if (!s->engine->mtp_ready) {
        return;
    }
    /* A second session shares the same target mapping; no second weight copy. */
    ds4_engine base_engine = *s->engine;
    base_engine.mtp_ready = false;
    ds4_session *base = NULL;
    char err[256] = {0};
    check(ds4_session_create(&base, &base_engine, s->ctx_size) == 0, "Inkling MTP reference session failed");
    check(ds4_session_sync(s, prompt, err, sizeof(err)) == 0 &&
          ds4_session_sync(base, prompt, err, sizeof(err)) == 0, err);
    ds4_tokens transcript = {0};
    ds4_tokens_copy(&transcript, prompt);
    unsigned cycles = 0, generated = 0;
    while (generated < 18) {
        int tokens[IK_VERIFY_ROWS], target[IK_VERIFY_ROWS];
        const int first = ds4_session_argmax(s), before = ds4_session_pos(s);
        check(first == ds4_session_argmax(base), "Inkling MTP first token differs");
        const int n = ds4_session_inkling_trial(s, first, 18 - (int)generated,
                                               tokens, target, IK_VERIFY_ROWS, err, sizeof(err));
        check(n > 0 && n <= IK_VERIFY_ROWS && n <= 18 - (int)generated, err);
        check(ds4_session_pos(s) == before && tokens[0] == first, "Inkling trial committed tokens early");
        check(ds4_session_eval(s, first, err, sizeof(err)) != 0 &&
              ds4_session_sync(s, &transcript, err, sizeof(err)) != 0 &&
              ds4_session_inkling_trial(s, first, 1, tokens, target, IK_VERIFY_ROWS, err, sizeof(err)) < 0,
              "Inkling allowed another operation during pending verification");
        check(ds4_session_inkling_commit(s, 0, err, sizeof(err)) != 0 &&
              ds4_session_inkling_commit(s, n + 1, err, sizeof(err)) != 0 &&
              ds4_session_pos(s) == before, "Inkling accepted invalid commit bounds");
        int keep = 1;
        while (keep < n && tokens[keep - 1] != 200006 && tokens[keep] == target[keep - 1]) {
            keep++;
        }
        check(ds4_session_inkling_commit(s, keep, err, sizeof(err)) == 0, err);
        check(ds4_session_inkling_commit(s, keep, err, sizeof(err)) != 0,
              "Inkling committed one trial twice");
        for (int i = 0; i < keep; i++) {
            check(tokens[i] == ds4_session_argmax(base), "Inkling MTP accepted a non-greedy token");
            check(ds4_session_eval(base, tokens[i], err, sizeof(err)) == 0, err);
            ds4_tokens_push(&transcript, tokens[i]);
        }
        check_targets(s, base);
        check(s->inkling_spec.position == (unsigned)transcript.len && inkling_spec_valid(&s->inkling_spec),
              "Inkling draft boundary did not follow committed target");
        generated += (unsigned)keep;
        cycles++;
    }
    ds4_session_invalidate(s);
    check(ds4_session_sync(s, &transcript, err, sizeof(err)) == 0, err);
    check_targets(s, base);
    /* A failed restore must poison state; the host owns the one generation
     * change when it invalidates both native and Rust timelines. */
    int tokens[IK_VERIFY_ROWS], target[IK_VERIFY_ROWS];
    const int first = ds4_session_argmax(s);
    check(ds4_session_inkling_trial(s, first, 2, tokens, target,
                                    IK_VERIFY_ROWS, err, sizeof(err)) == 2, err);
    const uint64_t generation = ds4_session_generation(s);
    s->inkling_graph.undo[INKLING_LAYERS - 1].streams = 0;
    check(ds4_session_inkling_commit(s, 1, err, sizeof(err)) != 0,
          "Inkling committed incomplete target restore");
    check(ds4_session_generation(s) == generation,
          "Inkling failed commit changed host-owned generation");
    check(!s->checkpoint_valid && s->inkling_graph.failed &&
          s->inkling_spec.draft.graph.failed && !s->inkling_trial_n,
          "Inkling failed commit left usable state");
    ds4_session_invalidate(s);
    check(ds4_session_generation(s) == generation + 1,
          "Inkling recovery changed generation more than once");
    check(ds4_session_sync(s, &transcript, err, sizeof(err)) == 0, err);
    check_targets(s, base);
    printf("Inkling native MTP: %u cycles, %u greedy tokens, all target logits/KV/convolution exact\n", cycles, generated);
    ds4_session_free(base);
    ds4_tokens_free(&transcript);
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 4) {
        fprintf(stderr, "usage: %s <MQ85GB-first.gguf> [<MTP.gguf> <manifest>]\n", argv[0]);
        return 2;
    }
    const ds4_host_shape host = {.variant = DS4_VARIANT_INKLING_SMALL};
    ds4_host_shape_install(&host);
    model_apply_host_shape();
    ds4_host_shape_clear();
    ds4_engine e = {.backend = DS4_BACKEND_CUDA, .metal_ready = true};
    check(!ds4_engine_supports_batching(&e), "Inkling entered DeepSeek batching");
    check(ds4_engine_hidden_f32_values(&e) == IK_HIDDEN &&
          ds4_engine_n_hc(&e) == 1, "Inkling hidden stream contract");
    ds4_session_graph_fit_quote quote;
    setenv("DS4_SESSION_GRAPH_FIT", "0", 1);
    check(ds4_engine_session_graph_fit_quote(&e, 32, &quote) &&
          quote.fail_open && quote.need_bytes, "Inkling fit override ignored");
    check(!ds4_engine_session_graph_fit_quote(&e,
              (int)DS4_SHAPE_INKLING_SMALL.rope_orig_ctx + 1, &quote),
          "Inkling fit override bypassed context limit");
    unsetenv("DS4_SESSION_GRAPH_FIT");

    model_open(&e.model, argv[1], false, false);
    weights_bind(&e.weights, &e.model, false, 0, UINT32_MAX, true, false);
    e.vocab.n_vocab = INKLING_VALID_VOCAB;
    check(ds4_gpu_init() && ds4_gpu_set_model_map(e.model.map, e.model.size),
          "Inkling GPU/map initialization failed");
    if (argc == 4) {
        model_open(&e.mtp_model, argv[2], false, false);
        inkling_bind_draft(&e.inkling_mtp, &e.mtp_model);
        check(ds4_gpu_import_model_ipc_manifest(e.mtp_model.map, e.mtp_model.size, argv[3], "mtp"),
              "Inkling MTP owner import failed");
        e.mtp_ready = true;
        e.mtp_draft_tokens = INKLING_DRAFT_LAYERS;
    }
    setenv("DS4_SESSION_LAZY_GRAPH", "1", 1);
    ds4_session *s = NULL;
    const int context = 32;
    check(ds4_session_create(&s, &e, context) == 0 && s &&
          ds4_session_graph_pending(s), "Inkling lazy session create failed");
    check(ds4_session_graph_bytes_committed(s) == 0, "pending graph owns memory");
    const uint64_t estimate = ds4_engine_session_graph_bytes_estimate(&e, context);
    check(estimate && ds4_engine_session_graph_fit_quote(&e, context, &quote) &&
          quote.need_bytes == estimate, "Inkling graph quote mismatch");

    const int fixture[] = {976, 9029, 328, 10128, 382};
    ds4_tokens prompt = {0};
    for (unsigned i = 0; i < sizeof(fixture) / sizeof(fixture[0]); i++) {
        ds4_tokens_push(&prompt, fixture[i]);
    }
    char err[256] = {0};
    const size_t bytes = INKLING_VALID_VOCAB * sizeof(float);
    float *base = xmalloc(bytes), *got = xmalloc(bytes), *extended = xmalloc(bytes);
    check(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0, err);
    check(!ds4_session_graph_pending(s) && ds4_session_pos(s) == prompt.len,
          "Inkling sync frontier mismatch");
    const uint64_t measured = ds4_session_graph_bytes_committed(s);
    check(measured >= estimate && measured - estimate < 1024 * 1024,
          "Inkling memory estimate misses actual allocation");
    check(ds4_session_copy_logits(s, base, INKLING_VALID_VOCAB) == INKLING_VALID_VOCAB,
          "Inkling valid vocabulary readback failed");
    check(ds4_session_set_logits(s, base, INKLING_VALID_VOCAB) == 0 &&
          ds4_session_copy_logits(s, got, INKLING_VALID_VOCAB - 1) == 0,
          "Inkling logits size contract failed");
    for (unsigned i = INKLING_VALID_VOCAB; i < DS4_N_VOCAB; i++) {
        check(s->logits[i] == -INFINITY, "Inkling restored logits include padding");
    }
    const uint64_t generation = ds4_session_generation(s);
    check(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0 &&
          ds4_session_generation(s) == generation, "no-op sync changed generation");
    check(ds4_session_eval(s, INKLING_VALID_VOCAB, err, sizeof(err)) != 0 &&
          ds4_session_pos(s) == prompt.len, "invalid token changed frontier");
    const int next = ds4_session_argmax(s);
    check(next >= 0 && next < INKLING_VALID_VOCAB &&
          ds4_session_eval(s, next, err, sizeof(err)) == 0, err);
    check(ds4_session_copy_logits(s, extended, INKLING_VALID_VOCAB) == INKLING_VALID_VOCAB,
          "Inkling decode readback failed");
    ds4_tokens_push(&prompt, next);
    ds4_session_invalidate(s);
    check(ds4_session_eval(s, next, err, sizeof(err)) != 0,
          "invalidated session allowed decode");
    check(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0 &&
          ds4_session_copy_logits(s, got, INKLING_VALID_VOCAB) == INKLING_VALID_VOCAB &&
          memcmp(got, extended, bytes) == 0, "cold sync differs from decode extension");

    ds4_session_rewind(s, 3);
    check(ds4_session_pos(s) == 3 &&
          ds4_session_eval(s, next, err, sizeof(err)) != 0,
          "rewound convolution state allowed decode");
    prompt.len = 5;
    check(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0 &&
          ds4_session_copy_logits(s, got, INKLING_VALID_VOCAB) == INKLING_VALID_VOCAB &&
          memcmp(got, base, bytes) == 0, "rewind replay differs from cold sync");
    ds4_session_invalidate(s);
    prompt.len = 3;
    check(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0, err);
    prompt.len = 5;
    check(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0 &&
          ds4_session_copy_logits(s, got, INKLING_VALID_VOCAB) == INKLING_VALID_VOCAB &&
          memcmp(got, base, bytes) == 0, "prefix extension differs from cold sync");

    FILE *fp = tmpfile();
    check(fp && !ds4_session_payload_bytes(s) &&
          ds4_session_save_payload(s, fp, err, sizeof(err)) != 0,
          "Inkling entered DeepSeek payload format");
    fclose(fp);
    printf("Inkling session: lazy alloc, no-op/extend/decode/reset/rewind parity; "
           "estimate=%llu measured=%llu\n", (unsigned long long)estimate,
           (unsigned long long)measured);
    check_mtp(s, &prompt);
    check_image_sync(s);
    check_audio_sync(s);
    ds4_session_free(s);
    check(session_tensors_census_live() == 0, "Inkling leaked session tensors");
    ds4_gpu_cleanup();
    model_close(&e.model);
    if (e.mtp_ready) {
        model_close(&e.mtp_model);
    }
    ds4_tokens_free(&prompt);
    free(base);
    free(got);
    free(extended);
    return 0;
}
