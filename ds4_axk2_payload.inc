/* Snapshot ("AXK1"): tokens, logits, then every layer's latent rows [0, n),
 * their exponents, rope keys and indexer keys (with exponents) in position
 * order. Rows are position-indexed, so a payload loads into a graph of any
 * chunk size; the key format is part of the header. */
enum { AXK2_PAYLOAD_MAGIC = 0x314b5841 }; /* "AXK1" */
typedef enum { AXK2_PAYLOAD_SAVE, AXK2_PAYLOAD_LOAD } axk2_payload_dir;

// Bytes of one token's rows across a layer's caches.
static uint32_t axk2_payload_row(void) {
    return AXK2_KV_LORA + 1u + AXK2_ROPE * 2u +
        (axk2_key_format() == DS4_AXK2_KEYS_BF16 ? AXK2_IDX_DIM * 2u : AXK2_IDX_DIM + 1u);
}

static uint64_t axk2_payload_body(uint32_t n) {
    return (uint64_t)n * sizeof(uint32_t) + (uint64_t)DS4_N_VOCAB * sizeof(float) +
        (uint64_t)AXK2_LAYERS * n * axk2_payload_row();
}

static uint64_t axk2_payload_bytes(const ds4_axk2_graph *g, uint32_t n) {
    if (!g->ready || g->failed || !n || n > g->ctx_cap || g->position != n) { return 0; }
    return DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) + axk2_payload_body(n);
}

static int axk2_payload_span(axk2_payload_dir dir, FILE *fp, ds4_gpu_tensor *t, uint64_t offset, uint64_t bytes,
        uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    if (!bytes) { return 0; }
    return dir == AXK2_PAYLOAD_LOAD
        ? payload_read_tensor_span(fp, t, offset, bytes, buf, DS4_SESSION_IO_CHUNK, left, err, errlen)
        : payload_write_tensor_span(fp, t, offset, bytes, buf, DS4_SESSION_IO_CHUNK, err, errlen);
}

static int axk2_payload_state(axk2_payload_dir dir, FILE *fp, ds4_axk2_graph *g, uint32_t n,
        uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    const bool fp8 = axk2_key_format() == DS4_AXK2_KEYS_FP8;
    const uint64_t key = fp8 ? AXK2_IDX_DIM : (uint64_t)AXK2_IDX_DIM * 2u;
    int rc = 0;
    for (uint32_t il = 0; !rc && il < AXK2_LAYERS; il++) {
        rc = axk2_payload_span(dir, fp, g->latent[il], 0, (uint64_t)n * AXK2_KV_LORA, buf, left, err, errlen) ||
            axk2_payload_span(dir, fp, g->latent[il], (uint64_t)g->ctx_cap * AXK2_KV_LORA, n, buf, left, err, errlen) ||
            axk2_payload_span(dir, fp, g->k_pe[il], 0, (uint64_t)n * AXK2_ROPE * 2u, buf, left, err, errlen) ||
            axk2_payload_span(dir, fp, g->keys[il], 0, (uint64_t)n * key, buf, left, err, errlen) ||
            (fp8 && axk2_payload_span(dir, fp, g->keys[il], (uint64_t)g->ctx_cap * AXK2_IDX_DIM, n, buf, left, err,
                errlen));
    }
    return rc;
}

static int axk2_payload_save(ds4_axk2_graph *g, const int *tokens, uint32_t n,
        const float *logits, FILE *fp, char *err, size_t errlen) {
    if (!fp || !tokens || !logits || !axk2_payload_bytes(g, n)) {
        payload_set_err(err, errlen, "invalid A.X-K2 snapshot state");
        return 1;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "invalid A.X-K2 snapshot token");
            return 1;
        }
    }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) { payload_set_err(err, errlen, "invalid A.X-K2 snapshot logits"); return 1; }
    }
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!buf || !ds4_gpu_synchronize()) {
        free(buf);
        payload_set_err(err, errlen, "A.X-K2 snapshot allocation or synchronization failed");
        return 1;
    }
    const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION, g->ctx_cap, g->row_cap,
        AXK2_LAYERS, AXK2_PAYLOAD_MAGIC, (uint32_t)axk2_key_format(), n, 0u, axk2_payload_row(),
        AXK2_LAYERS, DS4_N_VOCAB, n,
    };
    int rc = 0;
    for (uint32_t i = 0; !rc && i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) { rc = payload_write_u32(fp, h[i], err, errlen); }
    for (uint32_t i = 0; !rc && i < n; i++) { rc = payload_write_u32(fp, (uint32_t)tokens[i], err, errlen); }
    if (!rc) { rc = payload_write_logits(fp, logits, buf, err, errlen); }
    if (!rc) { rc = axk2_payload_state(AXK2_PAYLOAD_SAVE, fp, g, n, buf, NULL, err, errlen); }
    free(buf);
    return rc;
}

static int axk2_payload_restore(ds4_axk2_graph *g, FILE *fp, uint64_t *left,
        const uint32_t *h, int **tokens_out, float *logits, char *err, size_t errlen) {
    if (!g || !fp || !left || !h || !tokens_out || !logits) { return 1; }
    /* A failed or partial restore can never be reused as a committed prefix. */
    g->failed = true;
    g->position = 0;
    *tokens_out = NULL;
    const uint32_t n = h[7];
    if (!g->ready || !n || n > g->ctx_cap || h[0] != DS4_SESSION_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_PAYLOAD_VERSION || h[2] < n || h[2] > AXK2_CONTEXT || !h[3] || h[3] > h[2] ||
        h[4] != AXK2_LAYERS || h[5] != AXK2_PAYLOAD_MAGIC || h[6] != (uint32_t)axk2_key_format() || h[8] ||
        h[9] != axk2_payload_row() || h[10] != AXK2_LAYERS || h[11] != DS4_N_VOCAB || h[12] != n ||
        *left != axk2_payload_body(n)) {
        payload_set_err(err, errlen, "incompatible A.X-K2 snapshot header or length");
        return 1;
    }
    int *tokens = malloc((size_t)n * sizeof(*tokens));
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!tokens || !buf || !ds4_gpu_synchronize()) {
        free(tokens);
        free(buf);
        payload_set_err(err, errlen, "A.X-K2 restore allocation or synchronization failed");
        return 1;
    }
    int rc = 0;
    for (uint32_t i = 0; !rc && i < n; i++) {
        uint32_t token = 0;
        rc = payload_read_u32(fp, &token, left, err, errlen);
        if (!rc && token >= DS4_N_VOCAB) { payload_set_err(err, errlen, "invalid A.X-K2 snapshot token"); rc = 1; }
        tokens[i] = (int)token;
    }
    if (!rc) { rc = payload_read_bytes(fp, logits, (uint64_t)DS4_N_VOCAB * sizeof(float), left, err, errlen); }
    for (uint32_t i = 0; !rc && i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) { payload_set_err(err, errlen, "invalid A.X-K2 snapshot logits"); rc = 1; }
    }
    if (!rc) { rc = axk2_payload_state(AXK2_PAYLOAD_LOAD, fp, g, n, buf, left, err, errlen); }
    if (!rc && (*left || !ds4_gpu_tensor_write(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) ||
                !ds4_gpu_synchronize())) {
        payload_set_err(err, errlen, "A.X-K2 restore finalization failed");
        rc = 1;
    }
    free(buf);
    if (rc) { free(tokens); return 1; }
    g->position = n;
    g->failed = false;
    *tokens_out = tokens;
    return 0;
}
