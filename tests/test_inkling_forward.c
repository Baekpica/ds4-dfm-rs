/* Real MQ85GB graph integration: full prefill versus incremental decode. */
#include "../ds4.c"

typedef struct { unsigned char *data; size_t bytes; } inkling_snapshot;

static void check_seed(const ds4_inkling_graph *g, const ds4_model *m,
                        const ds4_weights *w, unsigned rows) {
    const size_t count = (size_t)rows * IK_HIDDEN, bytes = count * sizeof(float);
    float *raw = xmalloc(bytes), *seed = xmalloc(bytes);
    const uint16_t *weights = tensor_data(m, w->output_norm);
    if (!ds4_gpu_tensor_read(g->buf[IK_X], 0, raw, bytes) ||
        !ds4_gpu_tensor_read(g->buf[IK_FINAL], 0, seed, bytes)) {
        ds4_die("Inkling normalized hidden read failed");
    }
    for (unsigned row = 0; row < rows; row++) {
        double sum = 0;
        for (unsigned c = 0; c < IK_HIDDEN; c++) {
            const double v = raw[row * IK_HIDDEN + c]; sum += v * v;
        }
        const double scale = 1.0 / sqrt(sum / IK_HIDDEN + 1e-6);
        for (unsigned c = 0; c < IK_HIDDEN; c++) {
            uint32_t bits = (uint32_t)weights[c] << 16; float weight;
            memcpy(&weight, &bits, sizeof(weight));
            const double want = raw[row * IK_HIDDEN + c] * scale * weight;
            const float got = seed[row * IK_HIDDEN + c];
            memcpy(&bits, &got, sizeof(bits));
            /* Half a BF16 relative ulp plus FP32 norm reduction rounding. */
            if (!isfinite(got) || (bits & 0xffffu) || fabs(got - want) > 0.004 * fabs(want) + 1e-7) {
                ds4_die("Inkling draft seed must retain final norm before muP division");
            }
        }
    }
    free(raw); free(seed);
}

static inkling_snapshot read_state(const ds4_inkling_graph *g) {
    size_t bytes = 0;
    for (unsigned i = 0; i < INKLING_LAYERS; i++) {
        const unsigned valid = g->position < g->layer[i].capacity
                                 ? g->position : g->layer[i].capacity;
        bytes += (size_t)valid * 2 * IK_KV * sizeof(uint16_t);
        bytes += IK_HISTORY * (2 * IK_KV + 2 * IK_HIDDEN) * sizeof(float);
    }
    inkling_snapshot s = {xmalloc(bytes), bytes};
    size_t offset = 0;
    for (unsigned i = 0; i < INKLING_LAYERS; i++) {
        const unsigned valid = g->position < g->layer[i].capacity
                                 ? g->position : g->layer[i].capacity;
        size_t count = (size_t)valid * 2 * IK_KV * sizeof(uint16_t);
        if (!ds4_gpu_tensor_read(g->layer[i].kv, 0, s.data + offset, count)) {
            ds4_die("Inkling KV state read failed");
        }
        offset += count;
        for (unsigned j = 0; j < 4; j++) {
            count = IK_HISTORY * (j < 2 ? IK_KV : IK_HIDDEN) * sizeof(float);
            if (!ds4_gpu_tensor_read(g->layer[i].conv[j], 0, s.data + offset, count)) {
                ds4_die("Inkling convolution state read failed");
            }
            offset += count;
        }
    }
    return s;
}

static void write_state(ds4_inkling_graph *g, const inkling_snapshot *s, unsigned pos) {
    size_t offset = 0;
    for (unsigned i = 0; i < INKLING_LAYERS; i++) {
        const unsigned valid = pos < g->layer[i].capacity ? pos : g->layer[i].capacity;
        size_t bytes = (size_t)valid * 2 * IK_KV * sizeof(uint16_t);
        if (offset + bytes > s->bytes ||
            !ds4_gpu_tensor_write(g->layer[i].kv, 0, s->data + offset, bytes)) {
            ds4_die("Inkling fixture KV restore failed");
        }
        offset += bytes;
        for (unsigned j = 0; j < IK_CONV_STREAMS; j++) {
            bytes = IK_HISTORY * (j < 2 ? IK_KV : IK_HIDDEN) * sizeof(float);
            if (offset + bytes > s->bytes ||
                !ds4_gpu_tensor_write(g->layer[i].conv[j], 0, s->data + offset, bytes)) {
                ds4_die("Inkling fixture convolution restore failed");
            }
            offset += bytes;
        }
    }
    g->position = pos;
    inkling_snapshot check = read_state(g);
    if (offset != s->bytes || check.bytes != s->bytes || memcmp(check.data, s->data, s->bytes)) {
        ds4_die("Inkling fixture state restore is not exact");
    }
    free(check.data);
}

static void check_verify(ds4_inkling_graph *g, const ds4_model *m,
                          const ds4_weights *w, const int *tokens, unsigned n) {
    if (n < IK_VERIFY_ROWS) {
        return;
    }
    const unsigned count = IK_VERIFY_ROWS;
    const size_t logits_bytes = INKLING_VALID_VOCAB * sizeof(float);
    const size_t seed_bytes = IK_HIDDEN * sizeof(float);
    inkling_snapshot states[IK_VERIFY_ROWS];
    float *logits = xmalloc(count * logits_bytes), *got = xmalloc(logits_bytes);
    float *seeds = xmalloc(count * seed_bytes), *seed = xmalloc(seed_bytes);
    int tops[IK_VERIFY_ROWS], trial[IK_VERIFY_ROWS], verified[IK_VERIFY_ROWS];
    if (!inkling_graph_reset(g) || !inkling_graph_forward(g, m, w, tokens, n)) {
        ds4_die("Inkling verification prefix failed");
    }
    inkling_snapshot prefix = read_state(g);
    verified[0] = -1;
    if (inkling_target_verify(g, m, w, tokens, 1, verified) || verified[0] != -1 ||
        g->position != n || g->failed) {
        ds4_die("Inkling target verification requires a journal");
    }
    for (unsigned i = 0; i < count; i++) {
        float *row = logits + (size_t)i * INKLING_VALID_VOCAB;
        if (!inkling_graph_forward(g, m, w, tokens + i, 1) ||
            !ds4_gpu_tensor_read(g->logits, 0, row, logits_bytes) ||
            !ds4_gpu_tensor_read(g->buf[IK_FINAL], 0, seeds + (size_t)i * IK_HIDDEN, seed_bytes)) {
            ds4_die("Inkling accepted-prefix baseline failed");
        }
        tops[i] = 0;
        for (unsigned j = 0; j < INKLING_VALID_VOCAB; j++) {
            if (!isfinite(row[j])) {
                ds4_die("Inkling verification baseline has nonfinite logits");
            }
            if (row[j] > row[tops[i]]) {
                tops[i] = (int)j;
            }
        }
        states[i] = read_state(g);
    }
    if (!inkling_graph_track(g, count)) {
        ds4_die("Inkling target journal allocation failed");
    }
    write_state(g, &prefix, n);
    memcpy(trial, tokens, count * sizeof(*trial));
    trial[count - 1] = INKLING_VALID_VOCAB;
    for (unsigned i = 0; i < count; i++) {
        verified[i] = -1;
    }
    if (inkling_target_verify(g, m, w, trial, count, verified) ||
        inkling_target_verify(g, m, w, tokens, 0, verified) ||
        inkling_target_verify(g, m, w, tokens, count + 1, verified) ||
        inkling_target_verify(g, m, w, tokens, 1, NULL) || g->position != n || g->failed) {
        ds4_die("Inkling target verification accepted invalid input");
    }
    g->position = g->context;
    if (inkling_target_verify(g, m, w, tokens, 1, verified) || g->failed) {
        ds4_die("Inkling target verification exceeded context");
    }
    g->position = n;
    inkling_snapshot unchanged_prefix = read_state(g);
    if (unchanged_prefix.bytes != prefix.bytes ||
        memcmp(unchanged_prefix.data, prefix.data, prefix.bytes)) {
        ds4_die("Inkling invalid verification changed state");
    }
    free(unchanged_prefix.data);
    for (unsigned i = 0; i < count; i++) {
        if (verified[i] != -1) {
            ds4_die("Inkling invalid verification published argmax output");
        }
    }
    for (unsigned keep = 1; keep <= count; keep++) {
        write_state(g, &prefix, n);
        for (unsigned i = 0; i < count; i++) {
            /* Change only rejected rows; accepted logits/state must stay exact. */
            trial[i] = i < keep ? tokens[i] : (tokens[i] + 1009) % INKLING_VALID_VOCAB;
            verified[i] = -1;
        }
        if (!inkling_target_verify(g, m, w, trial, count, verified) ||
            memcmp(verified, tops, keep * sizeof(*tops))) {
            ds4_die("Inkling target verification disagrees with scalar argmax");
        }
        inkling_snapshot pending = read_state(g);
        g->undo[INKLING_LAYERS - 1].streams = 0;
        if (inkling_target_keep(g, m, w, keep) || g->failed) {
            ds4_die("Inkling target committed an incomplete final-layer journal");
        }
        g->undo[INKLING_LAYERS - 1].streams = IK_CONV_MASK;
        if (inkling_target_keep(g, m, w, 0) || inkling_target_keep(g, m, w, count + 1) ||
            g->failed || g->position != n + count) {
            ds4_die("Inkling target accepted an invalid commit length");
        }
        inkling_snapshot unchanged = read_state(g);
        if (pending.bytes != unchanged.bytes || memcmp(pending.data, unchanged.data, pending.bytes)) {
            ds4_die("Inkling invalid commit changed pending state");
        }
        free(pending.data); free(unchanged.data);
        if (!inkling_target_keep(g, m, w, keep) || g->position != n + keep ||
            !ds4_gpu_tensor_read(g->logits, 0, got, logits_bytes) ||
            !ds4_gpu_tensor_read(g->buf[IK_FINAL], (keep - 1) * seed_bytes, seed, seed_bytes)) {
            ds4_die("Inkling target accepted-prefix commit failed");
        }
        inkling_snapshot committed = read_state(g);
        const inkling_snapshot *want = &states[keep - 1];
        if (committed.bytes != want->bytes || memcmp(committed.data, want->data, want->bytes) ||
            memcmp(got, logits + (size_t)(keep - 1) * INKLING_VALID_VOCAB, logits_bytes) ||
            memcmp(seed, seeds + (size_t)(keep - 1) * IK_HIDDEN, seed_bytes) ||
            inkling_target_keep(g, m, w, keep) || g->failed) {
            ds4_die("Inkling committed target state, logits or hidden differ");
        }
        free(committed.data);
        printf("Inkling target keep %u/%u: argmax/logits/seed/state exact\n", keep, count);
    }
    write_state(g, &prefix, n);
    if (!inkling_target_verify(g, m, w, tokens, 1, verified) || verified[0] != tops[0] ||
        !inkling_target_keep(g, m, w, 1) || g->position != n + 1) {
        ds4_die("Inkling one-row verification/commit failed");
    }
    inkling_snapshot single = read_state(g);
    if (single.bytes != states[0].bytes || memcmp(single.data, states[0].data, single.bytes) ||
        !ds4_gpu_tensor_read(g->logits, 0, got, logits_bytes) || memcmp(got, logits, logits_bytes)) {
        ds4_die("Inkling one-row committed state/logits differ");
    }
    free(single.data);
    puts("Inkling one-row target verification and input/commit rejection passed");
    for (unsigned i = 0; i < count; i++) {
        free(states[i].data);
    }
    free(prefix.data); free(logits); free(got); free(seeds); free(seed);
    inkling_track_free(g);
}

static int check_padding(const ds4_inkling_graph *g) {
    const unsigned count = DS4_N_VOCAB - INKLING_VALID_VOCAB;
    float *padding = xmalloc(count * sizeof(float));
    if (!ds4_gpu_tensor_read(g->logits, INKLING_VALID_VOCAB * sizeof(float),
                             padding, count * sizeof(float))) {
        ds4_die("Inkling padded logits read failed");
    }
    int failed = 0;
    for (unsigned i = 0; i < count; i++) {
        if (padding[i] != -INFINITY) {
            fprintf(stderr, "Inkling padded logit %u is not masked: %g\n",
                    i + INKLING_VALID_VOCAB, padding[i]);
            failed = 1;
            break;
        }
    }
    free(padding);
    return failed;
}

static float **load_features(const int *tokens, unsigned rows) {
    const char *dir = getenv("INKLING_TEST_MEDIA");
    if (!dir) {
        return NULL;
    }
    FILE *files[2];
    const char *names[] = {"audio-output.f32", "image-stage-3.f32"};
    for (unsigned i = 0; i < 2; i++) {
        char path[1024];
        int n = snprintf(path, sizeof(path), "%s/%s", dir, names[i]);
        if (n < 0 || n >= (int)sizeof(path) || !(files[i] = fopen(path, "rb"))) {
            ds4_die("cannot read Inkling media reference");
        }
    }
    float **features = xcalloc(rows, sizeof(*features));
    for (unsigned i = 0; i < rows; i++) {
        if (tokens[i] != 200053 && tokens[i] != 200054) {
            continue;
        }
        features[i] = xmalloc(IK_HIDDEN * sizeof(float));
        if (fread(features[i], sizeof(float), IK_HIDDEN, files[tokens[i] - 200053]) != IK_HIDDEN) {
            ds4_die("too few Inkling media reference rows");
        }
    }
    for (unsigned i = 0; i < 2; i++) {
        if (fgetc(files[i]) != EOF || fclose(files[i]) != 0) {
            ds4_die("Inkling fixture rows must match all placeholder tokens");
        }
    }
    return features;
}

static bool test_forward(ds4_inkling_graph *g, const ds4_model *m,
                          const ds4_weights *w, const int *tokens, unsigned n,
                          float **features) {
    return features ? inkling_graph_media(g, m, w, tokens, n, (const float *const *)features)
                    : inkling_graph_forward(g, m, w, tokens, n);
}

static void check_features(ds4_inkling_graph *g, const ds4_model *m,
                            const ds4_weights *w, const int *tokens,
                            unsigned n, float **features) {
    if (!features) {
        return;
    }
    const size_t bytes = (size_t)n * IK_HIDDEN * sizeof(float);
    float *text = xmalloc(bytes), *media = xmalloc(bytes);
    if (!inkling_embed_rows(g, m, w, tokens, n, NULL) ||
        !ds4_gpu_tensor_read(g->buf[IK_X], 0, text, bytes) ||
        !inkling_embed_rows(g, m, w, tokens, n, (const float *const *)features) ||
        !ds4_gpu_tensor_read(g->buf[IK_X], 0, media, bytes)) {
        ds4_die("Inkling input features read failed");
    }
    for (unsigned i = 0; i < n; i++) {
        const float *want = features[i] ? features[i] : text + i * IK_HIDDEN;
        if (memcmp(media + i * IK_HIDDEN, want, IK_HIDDEN * sizeof(float)) != 0) {
            ds4_die("Inkling media was reordered or text-normalized twice");
        }
    }
    free(text);
    free(media);
    puts("Inkling media replaces normalized embeddings exactly; text rows unchanged");
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <MQ85GB-first.gguf> <token> <token> [...]\n", argv[0]);
        return 2;
    }
    const unsigned rows = (unsigned)argc - 2;
    int *tokens = xcalloc(rows, sizeof(*tokens));
    for (unsigned i = 0; i < rows; i++) {
        tokens[i] = atoi(argv[i + 2]);
    }
    float **features = load_features(tokens, rows);
    const ds4_host_shape host = {.variant = DS4_VARIANT_INKLING_SMALL};
    ds4_host_shape_install(&host);
    model_apply_host_shape();
    ds4_host_shape_clear();
    ds4_model model;
    model_open(&model, argv[1], false, false);
    ds4_weights weights;
    weights_bind(&weights, &model, false, 0, UINT32_MAX, true, false);
    if (!ds4_gpu_init()) {
        ds4_die("Inkling GPU initialization failed");
    }
    /* Match production startup: aligned Q8 changes the reduction order.
     * The existing CUDA opt-outs also exercise the original raw tier. */
    ds4_gpu_tensor_record *records = xcalloc(model.n_tensors, sizeof(*records));
    for (uint64_t i = 0; i < model.n_tensors; i++) {
        const ds4_tensor *t = &model.tensors[i];
        records[i].name = t->name.ptr;
        records[i].name_len = (uint32_t)t->name.len;
        records[i].type = t->type;
        records[i].ndim = t->ndim;
        memcpy(records[i].dims, t->dim, sizeof(t->dim));
        records[i].offset = t->abs_offset;
        records[i].bytes = t->bytes;
    }
    int built = ds4_gpu_build_derived_artifacts_from_records(
        model.map, model.size, records, (uint32_t)model.n_tensors);
    free(records);
    if (built > 0) {
        model_release_mapping_cache(&model);
    }
    if (!ds4_gpu_set_model_map(model.map, model.size)) {
        ds4_die("Inkling GPU/map initialization failed");
    }
    ds4_inkling_graph g;
    if (!inkling_graph_alloc(&g, &model, &weights, rows + 16, rows)) {
        ds4_die("Inkling graph allocation failed");
    }
    check_features(&g, &model, &weights, tokens, rows, features);
    float *prefill = xcalloc(INKLING_VALID_VOCAB, sizeof(float));
    float *decode = xcalloc(INKLING_VALID_VOCAB, sizeof(float));
    const size_t bytes = INKLING_VALID_VOCAB * sizeof(float);
    const size_t seed_bytes = (size_t)rows * IK_HIDDEN * sizeof(float);
    float *full_seed = xmalloc(seed_bytes), *decode_seed = xmalloc(seed_bytes);
    const char *trace = getenv("INKLING_TEST_TRACE");
    char trace_prefix[1024];
    if (trace) {
        snprintf(trace_prefix, sizeof(trace_prefix), "%s/prefill", trace);
        setenv("DS4_METAL_GRAPH_DUMP_PREFIX", trace_prefix, 1);
    }
    fprintf(stderr, "Inkling: prefill %u tokens\n", rows);
    if (!test_forward(&g, &model, &weights, tokens, rows, features) ||
        !ds4_gpu_tensor_read(g.logits, 0, prefill, bytes)) {
        ds4_die("Inkling prefill failed");
    }
    check_seed(&g, &model, &weights, rows);
    if (!ds4_gpu_tensor_read(g.buf[IK_FINAL], 0, full_seed, seed_bytes)) {
        ds4_die("Inkling full hidden read failed");
    }
    const char *dump = getenv("INKLING_TEST_LOGITS");
    if (dump) {
        FILE *fp = fopen(dump, "wb");
        if (!fp || fwrite(prefill, 1, bytes, fp) != bytes || fclose(fp) != 0) {
            ds4_die("Inkling reference logits write failed");
        }
    }
    inkling_snapshot full_state = read_state(&g);
    if (!inkling_graph_reset(&g)) {
        ds4_die("Inkling reset failed");
    }
    if (trace) {
        snprintf(trace_prefix, sizeof(trace_prefix), "%s/decode", trace);
        setenv("DS4_METAL_GRAPH_DUMP_PREFIX", trace_prefix, 1);
    }
    for (unsigned i = 0; i < rows; i++) {
        fprintf(stderr, "Inkling: decode position %u\n", i);
        if (!test_forward(&g, &model, &weights, &tokens[i], 1, features ? features + i : NULL)) {
            ds4_die("Inkling decode failed");
        }
        if (!ds4_gpu_tensor_read(g.buf[IK_FINAL], 0, decode_seed + (size_t)i * IK_HIDDEN,
                                  IK_HIDDEN * sizeof(float))) {
            ds4_die("Inkling decode hidden read failed");
        }
    }
    if (!ds4_gpu_tensor_read(g.logits, 0, decode, bytes)) {
        ds4_die("Inkling logits read failed");
    }
    inkling_snapshot decode_state = read_state(&g);
    const int state_diff = full_state.bytes != decode_state.bytes ||
                          memcmp(full_state.data, decode_state.data, full_state.bytes) != 0;
    double err2 = 0, ref2 = 0, max_abs = 0;
    unsigned ptop = 0, dtop = 0;
    for (unsigned i = 0; i < INKLING_VALID_VOCAB; i++) {
        if (!isfinite(prefill[i]) || !isfinite(decode[i])) {
            ds4_die("nonfinite Inkling logits");
        }
        double err = (double)prefill[i] - decode[i];
        err2 += err * err;
        ref2 += (double)decode[i] * decode[i];
        max_abs = fmax(max_abs, fabs(err));
        if (prefill[i] > prefill[ptop]) {
            ptop = i;
        }
        if (decode[i] > decode[dtop]) {
            dtop = i;
        }
    }
    const double rel = sqrt(err2 / fmax(ref2, 1e-30));
    printf("Inkling %u-token full/decode: rel_rms=%g max_abs=%g top=%u/%u\n",
           rows, rel, max_abs, ptop, dtop);
    printf("Inkling committed KV/convolution: %zu bytes %s\n", full_state.bytes,
           state_diff ? "DIFFER" : "EXACT");
    int failed = memcmp(prefill, decode, bytes) != 0 || ptop != dtop ||
                 g.position != rows || ref2 == 0 || state_diff ||
                 memcmp(full_seed, decode_seed, seed_bytes) != 0;
    const unsigned chunks[] = {2, 3, 7};
    for (unsigned j = 0; j < sizeof(chunks) / sizeof(chunks[0]); j++) {
        const unsigned chunk = chunks[j];
        if (chunk >= rows) {
            continue;
        }
        if (!inkling_graph_reset(&g)) {
            ds4_die("Inkling chunk reset failed");
        }
        if (trace) {
            snprintf(trace_prefix, sizeof(trace_prefix), "%s/chunk-%u", trace, chunk);
            setenv("DS4_METAL_GRAPH_DUMP_PREFIX", trace_prefix, 1);
        }
        for (unsigned pos = 0; pos < rows; pos += chunk) {
            unsigned count = rows - pos < chunk ? rows - pos : chunk;
            if (!test_forward(&g, &model, &weights, tokens + pos, count, features ? features + pos : NULL)) {
                ds4_die("Inkling chunk forward failed");
            }
            if (!ds4_gpu_tensor_read(g.buf[IK_FINAL], 0, decode_seed + (size_t)pos * IK_HIDDEN,
                                      (size_t)count * IK_HIDDEN * sizeof(float))) {
                ds4_die("Inkling chunk hidden read failed");
            }
        }
        if (!ds4_gpu_tensor_read(g.logits, 0, decode, bytes)) {
            ds4_die("Inkling chunk logits read failed");
        }
        inkling_snapshot chunk_state = read_state(&g);
        const int mismatch = memcmp(prefill, decode, bytes) != 0 ||
            memcmp(full_seed, decode_seed, seed_bytes) != 0 ||
            full_state.bytes != chunk_state.bytes ||
            memcmp(full_state.data, chunk_state.data, full_state.bytes) != 0;
        printf("Inkling chunk %u logits/state: %s\n", chunk, mismatch ? "DIFFER" : "EXACT");
        failed |= mismatch;
        free(chunk_state.data);
    }
    failed |= check_padding(&g);
    const int invalid[] = {-1, INKLING_VALID_VOCAB};
    for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        if (inkling_graph_forward(&g, &model, &weights, &invalid[i], 1) ||
            g.position != rows) {
            failed = 1;
        }
    }
    if (inkling_graph_forward(&g, &model, &weights, tokens, 0) ||
        inkling_graph_forward(&g, &model, &weights, tokens, g.cap + 1) ||
        g.position != rows) {
        failed = 1;
    }
    if (features) {
        float value[IK_HIDDEN] = {0};
        const float *bad[] = {value};
        const int text_token = 1, image_token = 200054;
        if (inkling_graph_media(&g, &model, &weights, &text_token, 1, bad) ||
            g.position != rows || g.failed) {
            ds4_die("Inkling accepted media on an ordinary text token");
        }
        value[IK_HIDDEN - 1] = NAN;
        if (inkling_graph_media(&g, &model, &weights, &image_token, 1, bad) ||
            g.position != rows || g.failed) {
            ds4_die("Inkling accepted nonfinite media or mutated the frontier");
        }
        inkling_snapshot state = read_state(&g);
        if (state.bytes != full_state.bytes || memcmp(state.data, full_state.data, state.bytes) != 0) {
            ds4_die("invalid Inkling features mutated committed state");
        }
        free(state.data);
        for (unsigned i = 0; i < rows; i++) {
            free(features[i]);
        }
        free(features);
    }
    if (!features) {
        check_verify(&g, &model, &weights, tokens, rows);
    }
    inkling_graph_free(&g);
    ds4_gpu_cleanup();
    model_close(&model);
    free(prefill);
    free(decode);
    free(full_state.data);
    free(decode_state.data);
    free(tokens);
    free(full_seed); free(decode_seed);
    return failed;
}
