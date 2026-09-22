/* Persist logical KV rows, not ring capacity. A different prefill width may
 * restore the same prefix into different physical slots. MTP is not encoded. */
enum { M2_PAYLOAD_MAGIC = 0x324f4d49, M2_KV_WIDTH = 192 + 128 };
typedef enum { M2_PAYLOAD_SAVE, M2_PAYLOAD_LOAD } mimo2_payload_dir;

static unsigned mimo2_saved_rows(unsigned il, unsigned n) {
    return mimo2_is_full(il) || n < M2_WINDOW ? n : M2_WINDOW;
}

static uint64_t mimo2_payload_body(unsigned n) {
    uint64_t bytes = (uint64_t)n * sizeof(uint32_t) + (uint64_t)DS4_N_VOCAB * sizeof(float);
    for (unsigned il = 0; il < MIMO2_LAYERS; il++) {
        bytes += (uint64_t)mimo2_saved_rows(il, n) *
            (mimo2_is_full(il) ? 4 : 8) * M2_KV_WIDTH * sizeof(uint16_t);
    }
    return bytes;
}

static bool mimo2_payload_layout(const ds4_mimo2_graph *g, unsigned n) {
    if (!g || !g->cap || g->cap > M2_MAX_BATCH || g->cap > g->context || !n || n > g->context || g->context > M2_CONTEXT) { return false; }
    for (unsigned il = 0; il < MIMO2_LAYERS; il++) {
        if (!g->kv[il] || g->kv_cap[il] != mimo2_kv_capacity(il, g->context, g->cap)) { return false; }
    }
    return g->logits != NULL;
}

static uint64_t mimo2_payload_bytes(const ds4_mimo2_graph *g, unsigned n) {
    if (!mimo2_payload_layout(g, n) || g->failed || g->position != n) { return 0; }
    return DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) + mimo2_payload_body(n);
}

static int mimo2_payload_ring(mimo2_payload_dir dir, FILE *fp, ds4_mimo2_graph *g,
                              unsigned il, unsigned n, uint8_t *buf,
                              uint64_t *left, char *err, size_t errlen) {
    unsigned rows = mimo2_saved_rows(il, n), first = n - rows;
    const uint64_t row = (mimo2_is_full(il) ? 4 : 8) * M2_KV_WIDTH * sizeof(uint16_t);
    while (rows) {
        const unsigned slot = first % g->kv_cap[il];
        unsigned span = g->kv_cap[il] - slot;
        if (span > rows) { span = rows; }
        const int rc = dir == M2_PAYLOAD_LOAD
            ? payload_read_tensor_span(fp, g->kv[il], slot * row, span * row,
                                        buf, DS4_SESSION_IO_CHUNK, left, err, errlen)
            : payload_write_tensor_span(fp, g->kv[il], slot * row, span * row,
                                         buf, DS4_SESSION_IO_CHUNK, err, errlen);
        if (rc) { return rc; }
        first += span;
        rows -= span;
    }
    return 0;
}

static int mimo2_payload_save(ds4_mimo2_graph *g, const int *tokens, unsigned n,
                              const float *logits, FILE *fp, char *err, size_t errlen) {
    if (!fp || !tokens || !logits || !mimo2_payload_bytes(g, n)) {
        payload_set_err(err, errlen, "invalid MiMo snapshot state"); return 1;
    }
    for (unsigned i = 0; i < n; i++) {
        if (tokens[i] < 0 || (unsigned)tokens[i] >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "invalid MiMo snapshot token"); return 1;
        }
    }
    for (unsigned i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) { payload_set_err(err, errlen, "invalid MiMo snapshot logits"); return 1; }
    }
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!buf || !ds4_gpu_synchronize()) {
        free(buf); payload_set_err(err, errlen, "MiMo snapshot allocation or synchronization failed"); return 1;
    }
    const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION, g->context, g->cap,
        MIMO2_LAYERS, M2_PAYLOAD_MAGIC, M2_KV_WIDTH, n, 0, sizeof(uint16_t),
        M2_WINDOW, DS4_N_VOCAB, n,
    };
    int rc = 0;
    for (unsigned i = 0; !rc && i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) { rc = payload_write_u32(fp, h[i], err, errlen); }
    for (unsigned i = 0; !rc && i < n; i++) { rc = payload_write_u32(fp, (uint32_t)tokens[i], err, errlen); }
    if (!rc) { rc = payload_write_logits(fp, logits, buf, err, errlen); }
    for (unsigned il = 0; !rc && il < MIMO2_LAYERS; il++) {
        rc = mimo2_payload_ring(M2_PAYLOAD_SAVE, fp, g, il, n, buf, NULL, err, errlen);
    }
    free(buf);
    return rc;
}

static int mimo2_payload_restore(ds4_mimo2_graph *g, FILE *fp, uint64_t *left,
                                 const uint32_t *h, int **tokens_out, float *logits,
                                 char *err, size_t errlen) {
    if (!g || !fp || !left || !h || !tokens_out || !logits) { return 1; }
    /* A failed/partial restore can never be reused as a committed prefix. */
    g->failed = true;
    g->position = 0;
    *tokens_out = NULL;
    const unsigned n = h[7];
    if (!mimo2_payload_layout(g, n) || h[0] != DS4_SESSION_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_PAYLOAD_VERSION || h[2] < n || h[2] > M2_CONTEXT ||
        !h[3] || h[3] > M2_MAX_BATCH || h[3] > h[2] || h[4] != MIMO2_LAYERS || h[5] != M2_PAYLOAD_MAGIC ||
        h[6] != M2_KV_WIDTH || h[8] || h[9] != sizeof(uint16_t) ||
        h[10] != M2_WINDOW || h[11] != DS4_N_VOCAB || h[12] != n ||
        *left != mimo2_payload_body(n)) {
        payload_set_err(err, errlen, "incompatible MiMo snapshot header or length"); return 1;
    }
    int *tokens = malloc((size_t)n * sizeof(*tokens));
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!tokens || !buf || !ds4_gpu_synchronize()) {
        free(tokens); free(buf); payload_set_err(err, errlen, "MiMo restore allocation or synchronization failed"); return 1;
    }
    int rc = 0;
    for (unsigned i = 0; !rc && i < n; i++) {
        uint32_t token = 0;
        rc = payload_read_u32(fp, &token, left, err, errlen);
        if (!rc && token >= DS4_N_VOCAB) { payload_set_err(err, errlen, "invalid MiMo snapshot token"); rc = 1; }
        tokens[i] = (int)token;
    }
    if (!rc) { rc = payload_read_bytes(fp, logits, (uint64_t)DS4_N_VOCAB * sizeof(float), left, err, errlen); }
    for (unsigned i = 0; !rc && i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) { payload_set_err(err, errlen, "invalid MiMo snapshot logits"); rc = 1; }
    }
    for (unsigned il = 0; !rc && il < MIMO2_LAYERS; il++) {
        rc = mimo2_payload_ring(M2_PAYLOAD_LOAD, fp, g, il, n, buf, left, err, errlen);
    }
    if (!rc && (*left || !ds4_gpu_tensor_write(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) ||
                !ds4_gpu_synchronize())) { payload_set_err(err, errlen, "MiMo restore finalization failed"); rc = 1; }
    free(buf);
    if (rc) { free(tokens); return 1; }
    g->position = n;
    g->failed = false;
    *tokens_out = tokens;
    return 0;
}
