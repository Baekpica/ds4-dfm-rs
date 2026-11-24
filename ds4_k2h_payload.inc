/* Snapshot ("K2H1"): tokens, logits, then every layer's NVFP4 K/V blocks
 * holding positions [0, n) (whole blocks; rows past n are rewritten before
 * any read). Pages are position-indexed, so a payload loads into a graph of
 * any chunk size or context of at least n. */
enum { K2H_PAYLOAD_MAGIC = 0x3148324b }; /* "K2H1" */
typedef enum { K2H_PAYLOAD_SAVE, K2H_PAYLOAD_LOAD } k2h_payload_dir;

static uint64_t k2h_payload_blocks(uint32_t n) { return (n + DS4_K2H_KV_BLOCK - 1u) / DS4_K2H_KV_BLOCK; }

static uint64_t k2h_payload_body(uint32_t n) {
    return (uint64_t)n * sizeof(uint32_t) + (uint64_t)DS4_N_VOCAB * sizeof(float) +
        (uint64_t)K2H_LAYERS * k2h_payload_blocks(n) * DS4_K2H_BLOCK_BYTES;
}

static uint64_t k2h_payload_bytes(const ds4_k2h_graph *g, uint32_t n) {
    if (!g->ready || g->failed || !n || n > g->ctx_cap || g->position != n) { return 0; }
    return DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) + k2h_payload_body(n);
}

static int k2h_payload_state(k2h_payload_dir dir, FILE *fp, ds4_k2h_graph *g, uint32_t n,
        uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    const uint64_t bytes = k2h_payload_blocks(n) * DS4_K2H_BLOCK_BYTES;
    int rc = 0;
    for (uint32_t il = 0; !rc && il < K2H_LAYERS; il++) {
        rc = dir == K2H_PAYLOAD_LOAD
            ? payload_read_tensor_span(fp, g->pages[il], 0, bytes, buf, DS4_SESSION_IO_CHUNK, left, err, errlen)
            : payload_write_tensor_span(fp, g->pages[il], 0, bytes, buf, DS4_SESSION_IO_CHUNK, err, errlen);
    }
    return rc;
}

static int k2h_payload_save(ds4_k2h_graph *g, const int *tokens, uint32_t n,
        const float *logits, FILE *fp, char *err, size_t errlen) {
    if (!fp || !tokens || !logits || !k2h_payload_bytes(g, n)) {
        payload_set_err(err, errlen, "invalid K2-Horizon snapshot state");
        return 1;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "invalid K2-Horizon snapshot token");
            return 1;
        }
    }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) { payload_set_err(err, errlen, "invalid K2-Horizon snapshot logits"); return 1; }
    }
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!buf || !ds4_gpu_synchronize()) {
        free(buf);
        payload_set_err(err, errlen, "K2-Horizon snapshot allocation or synchronization failed");
        return 1;
    }
    const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION, g->ctx_cap, g->row_cap,
        K2H_LAYERS, K2H_PAYLOAD_MAGIC, 0u, n, 0u, DS4_K2H_BLOCK_BYTES,
        K2H_LAYERS, DS4_N_VOCAB, n,
    };
    int rc = 0;
    for (uint32_t i = 0; !rc && i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) { rc = payload_write_u32(fp, h[i], err, errlen); }
    for (uint32_t i = 0; !rc && i < n; i++) { rc = payload_write_u32(fp, (uint32_t)tokens[i], err, errlen); }
    if (!rc) { rc = payload_write_logits(fp, logits, buf, err, errlen); }
    if (!rc) { rc = k2h_payload_state(K2H_PAYLOAD_SAVE, fp, g, n, buf, NULL, err, errlen); }
    free(buf);
    return rc;
}

static int k2h_payload_restore(ds4_k2h_graph *g, FILE *fp, uint64_t *left,
        const uint32_t *h, int **tokens_out, float *logits, char *err, size_t errlen) {
    if (!g || !fp || !left || !h || !tokens_out || !logits) { return 1; }
    /* A failed or partial restore can never be reused as a committed prefix. */
    g->failed = true;
    g->position = 0;
    *tokens_out = NULL;
    const uint32_t n = h[7];
    if (!g->ready || !n || n > g->ctx_cap || h[0] != DS4_SESSION_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_PAYLOAD_VERSION || h[2] < n || h[2] > K2H_CONTEXT || !h[3] || h[3] > h[2] ||
        h[4] != K2H_LAYERS || h[5] != K2H_PAYLOAD_MAGIC || h[6] || h[8] || h[9] != DS4_K2H_BLOCK_BYTES ||
        h[10] != K2H_LAYERS || h[11] != DS4_N_VOCAB || h[12] != n || *left != k2h_payload_body(n)) {
        payload_set_err(err, errlen, "incompatible K2-Horizon snapshot header or length");
        return 1;
    }
    int *tokens = malloc((size_t)n * sizeof(*tokens));
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!tokens || !buf || !ds4_gpu_synchronize()) {
        free(tokens);
        free(buf);
        payload_set_err(err, errlen, "K2-Horizon restore allocation or synchronization failed");
        return 1;
    }
    int rc = 0;
    for (uint32_t i = 0; !rc && i < n; i++) {
        uint32_t token = 0;
        rc = payload_read_u32(fp, &token, left, err, errlen);
        if (!rc && token >= DS4_N_VOCAB) { payload_set_err(err, errlen, "invalid K2-Horizon snapshot token"); rc = 1; }
        tokens[i] = (int)token;
    }
    if (!rc) { rc = payload_read_bytes(fp, logits, (uint64_t)DS4_N_VOCAB * sizeof(float), left, err, errlen); }
    for (uint32_t i = 0; !rc && i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) { payload_set_err(err, errlen, "invalid K2-Horizon snapshot logits"); rc = 1; }
    }
    if (!rc) { rc = k2h_payload_state(K2H_PAYLOAD_LOAD, fp, g, n, buf, left, err, errlen); }
    if (!rc && (*left || !ds4_gpu_tensor_write(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) ||
                !ds4_gpu_synchronize())) {
        payload_set_err(err, errlen, "K2-Horizon restore finalization failed");
        rc = 1;
    }
    free(buf);
    if (rc) { free(tokens); return 1; }
    g->position = n;
    g->failed = false;
    *tokens_out = tokens;
    return 0;
}
