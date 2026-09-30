/* Save logical BF16 K/V plus exact E4M3 codes/F32 scales. SWA rows are
 * packed by absolute position, so restore may use another prefill width. */
enum { N05_PAYLOAD_MAGIC = 0x35304e4e, N05_KV_WIDTH = N05_KEY + N05_VALUE };
typedef enum { N05_SAVE, N05_LOAD } naive_payload_dir;

static unsigned naive_saved_rows(unsigned il, unsigned n) {
    return naive_is_dsa(il) || n < N05_WINDOW ? n : N05_WINDOW;
}

static uint64_t naive_payload_body(unsigned n) {
    uint64_t bytes = (uint64_t)n * sizeof(uint32_t) + (uint64_t)N05_VOCAB * sizeof(float);
    for (unsigned il = 0; il < N05_LAYERS; il++) {
        bytes += (uint64_t)naive_saved_rows(il, n) * naive_kv_heads(il) * N05_KV_WIDTH * sizeof(uint16_t);
        if (naive_is_dsa(il)) { bytes += (uint64_t)n * (N05_INDEX_DIM + sizeof(float)); }
    }
    return bytes;
}

static bool naive_payload_layout(const ds4_naive_graph *g, unsigned n) {
    if (!g || !g->cap || g->cap > N05_PREFILL_MAX || g->cap > g->context ||
        !n || n > g->context || g->context > N05_CONTEXT || !g->logits) { return false; }
    for (unsigned il = 0; il < N05_LAYERS; il++) {
        if (!g->kv[il] || g->kv_cap[il] != naive_kv_capacity(il, g->context, g->cap) ||
            (naive_is_dsa(il) && (!g->codes[il] || !g->scales[il]))) { return false; }
    }
    return true;
}

static uint64_t naive_payload_bytes(const ds4_naive_graph *g, unsigned n) {
    if (!naive_payload_layout(g, n) || g->failed || g->position != n) { return 0; }
    return DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) + naive_payload_body(n);
}

static int naive_payload_rows(naive_payload_dir dir, FILE *fp, ds4_naive_graph *g,
        unsigned il, unsigned n, uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    unsigned rows = naive_saved_rows(il, n), first = n - rows;
    const uint64_t row = naive_kv_heads(il) * N05_KV_WIDTH * sizeof(uint16_t);
    while (rows) {
        const unsigned slot = first % g->kv_cap[il];
        const unsigned tail = g->kv_cap[il] - slot, span = rows < tail ? rows : tail;
        const int rc = dir == N05_LOAD
            ? payload_read_tensor_span(fp, g->kv[il], slot * row, span * row,
                buf, DS4_SESSION_IO_CHUNK, left, err, errlen)
            : payload_write_tensor_span(fp, g->kv[il], slot * row, span * row,
                buf, DS4_SESSION_IO_CHUNK, err, errlen);
        if (rc) { return rc; }
        first += span; rows -= span;
    }
    if (!naive_is_dsa(il)) { return 0; }
    ds4_gpu_tensor *history[] = {g->codes[il], g->scales[il]};
    const uint64_t bytes[] = {(uint64_t)n * N05_INDEX_DIM, (uint64_t)n * sizeof(float)};
    for (unsigned i = 0; i < 2; i++) {
        const int rc = dir == N05_LOAD
            ? payload_read_tensor_span(fp, history[i], 0, bytes[i], buf, DS4_SESSION_IO_CHUNK, left, err, errlen)
            : payload_write_tensor_span(fp, history[i], 0, bytes[i], buf, DS4_SESSION_IO_CHUNK, err, errlen);
        if (rc) { return rc; }
    }
    return 0;
}

static int naive_payload_save(ds4_naive_graph *g, const int *tokens, unsigned n,
        const float *logits, FILE *fp, char *err, size_t errlen) {
    if (!fp || !tokens || !logits || !naive_payload_bytes(g, n)) {
        payload_set_err(err, errlen, "invalid Naive snapshot state"); return 1;
    }
    for (unsigned i = 0; i < n; i++) {
        if (tokens[i] < 0 || (unsigned)tokens[i] >= N05_VOCAB) { payload_set_err(err, errlen, "invalid Naive snapshot token"); return 1; }
    }
    for (unsigned i = 0; i < N05_VOCAB; i++) {
        if (!isfinite(logits[i])) { payload_set_err(err, errlen, "invalid Naive snapshot logits"); return 1; }
    }
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!buf || !ds4_gpu_synchronize()) { free(buf); payload_set_err(err, errlen, "Naive snapshot synchronization failed"); return 1; }
    const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION, g->context, g->cap,
        N05_LAYERS, N05_PAYLOAD_MAGIC, N05_KV_WIDTH, n, N05_INDEX_DIM,
        sizeof(uint16_t), N05_WINDOW, N05_VOCAB, n,
    };
    int rc = 0;
    for (unsigned i = 0; !rc && i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) { rc = payload_write_u32(fp, h[i], err, errlen); }
    for (unsigned i = 0; !rc && i < n; i++) { rc = payload_write_u32(fp, (uint32_t)tokens[i], err, errlen); }
    if (!rc) { rc = payload_write_logits(fp, logits, buf, err, errlen); }
    for (unsigned il = 0; !rc && il < N05_LAYERS; il++) { rc = naive_payload_rows(N05_SAVE, fp, g, il, n, buf, NULL, err, errlen); }
    free(buf);
    return rc;
}

static int naive_payload_restore(ds4_naive_graph *g, FILE *fp, uint64_t *left,
        const uint32_t *h, int **tokens_out, float *logits, char *err, size_t errlen) {
    if (!g || !fp || !left || !h || !tokens_out || !logits) { return 1; }
    g->failed = true; g->position = 0; *tokens_out = NULL;
    const unsigned n = h[7];
    if (!naive_payload_layout(g, n) || h[0] != DS4_SESSION_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_PAYLOAD_VERSION || h[2] < n || h[2] > N05_CONTEXT ||
        !h[3] || h[3] > N05_PREFILL_MAX || h[3] > h[2] || h[4] != N05_LAYERS ||
        h[5] != N05_PAYLOAD_MAGIC || h[6] != N05_KV_WIDTH || h[8] != N05_INDEX_DIM ||
        h[9] != sizeof(uint16_t) || h[10] != N05_WINDOW || h[11] != N05_VOCAB ||
        h[12] != n || *left != naive_payload_body(n)) {
        payload_set_err(err, errlen, "incompatible Naive snapshot header or length"); return 1;
    }
    int *tokens = malloc((size_t)n * sizeof(*tokens));
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!tokens || !buf || !ds4_gpu_synchronize()) {
        free(tokens); free(buf); payload_set_err(err, errlen, "Naive restore synchronization failed"); return 1;
    }
    int rc = 0;
    for (unsigned i = 0; !rc && i < n; i++) {
        uint32_t token = 0;
        rc = payload_read_u32(fp, &token, left, err, errlen);
        if (!rc && token >= N05_VOCAB) { payload_set_err(err, errlen, "invalid Naive snapshot token"); rc = 1; }
        tokens[i] = (int)token;
    }
    if (!rc) { rc = payload_read_bytes(fp, logits, (uint64_t)N05_VOCAB * sizeof(float), left, err, errlen); }
    for (unsigned i = 0; !rc && i < N05_VOCAB; i++) {
        if (!isfinite(logits[i])) { payload_set_err(err, errlen, "invalid Naive snapshot logits"); rc = 1; }
    }
    for (unsigned il = 0; !rc && il < N05_LAYERS; il++) { rc = naive_payload_rows(N05_LOAD, fp, g, il, n, buf, left, err, errlen); }
    if (!rc && (*left || !ds4_gpu_tensor_write(g->logits, 0, logits, (uint64_t)N05_VOCAB * sizeof(float)) ||
        !ds4_gpu_synchronize())) { payload_set_err(err, errlen, "Naive restore finalization failed"); rc = 1; }
    free(buf);
    if (rc) { free(tokens); return 1; }
    g->position = n; g->failed = false; *tokens_out = tokens;
    return 0;
}
