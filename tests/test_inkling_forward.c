/* Real MQ85GB graph integration: full prefill versus incremental decode. */
#include "../ds4.c"

typedef struct { unsigned char *data; size_t bytes; } inkling_snapshot;

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
    float *prefill = xcalloc(INKLING_VALID_VOCAB, sizeof(float));
    float *decode = xcalloc(INKLING_VALID_VOCAB, sizeof(float));
    const size_t bytes = INKLING_VALID_VOCAB * sizeof(float);
    const char *trace = getenv("INKLING_TEST_TRACE");
    char trace_prefix[1024];
    if (trace) {
        snprintf(trace_prefix, sizeof(trace_prefix), "%s/prefill", trace);
        setenv("DS4_METAL_GRAPH_DUMP_PREFIX", trace_prefix, 1);
    }
    fprintf(stderr, "Inkling: prefill %u tokens\n", rows);
    if (!inkling_graph_forward(&g, &model, &weights, tokens, rows) ||
        !ds4_gpu_tensor_read(g.logits, 0, prefill, bytes)) {
        ds4_die("Inkling prefill failed");
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
        if (!inkling_graph_forward(&g, &model, &weights, &tokens[i], 1)) {
            ds4_die("Inkling decode failed");
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
                 g.position != rows || ref2 == 0 || state_diff;
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
            if (!inkling_graph_forward(&g, &model, &weights, tokens + pos, count)) {
                ds4_die("Inkling chunk forward failed");
            }
        }
        if (!ds4_gpu_tensor_read(g.logits, 0, decode, bytes)) {
            ds4_die("Inkling chunk logits read failed");
        }
        inkling_snapshot chunk_state = read_state(&g);
        const int mismatch = memcmp(prefill, decode, bytes) != 0 ||
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
    inkling_graph_free(&g);
    ds4_gpu_cleanup();
    model_close(&model);
    free(prefill);
    free(decode);
    free(full_state.data);
    free(decode_state.data);
    free(tokens);
    return failed;
}
