/* Snapshot: tokens, logits, the state pool (every KDA tile and convolution
 * tail plus the carried hidden row), then each MLA block's latent and rope
 * key rows [0, n) and, with MTP, the draft layer's rows [0, n-1). Rejected
 * speculative rows never enter it. */
enum { L31_PAYLOAD_MAGIC = 0x3133474c }; /* "LG31" */
typedef enum { L31_PAYLOAD_SAVE, L31_PAYLOAD_LOAD } ling31_payload_dir;

static uint64_t ling31_payload_body(const ds4_ling31_graph *g, uint32_t n) {
    const uint64_t draft = g->mtp_drafts ? (uint64_t)(n - 1u) * L31_KV_ROW * sizeof(uint16_t) : 0u;
    return (uint64_t)n * sizeof(uint32_t) + (uint64_t)DS4_N_VOCAB * sizeof(float) + g->state_bytes +
        (uint64_t)LING31_MLA_LAYERS * n * L31_KV_ROW * sizeof(uint16_t) + draft;
}

static uint64_t ling31_payload_bytes(const ds4_ling31_graph *g, uint32_t n) {
    if (!g->ready || g->failed || g->mtp_n || !n || n > g->ctx_cap || g->position != n ||
        (g->mtp_drafts && g->mtp_len != n - 1u)) { return 0; }
    return DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) + ling31_payload_body(g, n);
}

static int ling31_payload_state(ling31_payload_dir dir, FILE *fp, ds4_ling31_graph *g, uint32_t n,
        uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    ds4_gpu_tensor *spans[1u + 2u * (LING31_MLA_LAYERS + 1u)];
    uint64_t bytes[1u + 2u * (LING31_MLA_LAYERS + 1u)];
    uint32_t count = 0;
    spans[count] = g->state_pool;
    bytes[count++] = g->state_bytes;
    const uint32_t layers = g->mtp_drafts ? LING31_LAYERS + 1u : LING31_LAYERS;
    for (uint32_t il = 0; il < layers; il++) {
        if (!ds4_ling31_layer_is_mla(il)) { continue; }
        const uint32_t rows = il == LING31_MTP ? n - 1u : n;
        spans[count] = g->latent[il];
        bytes[count++] = (uint64_t)rows * L31_KV_LORA * sizeof(uint16_t);
        spans[count] = g->k_pe[il];
        bytes[count++] = (uint64_t)rows * L31_ROPE * sizeof(uint16_t);
    }
    int rc = 0;
    for (uint32_t i = 0; !rc && i < count; i++) {
        if (!bytes[i]) { continue; }
        rc = dir == L31_PAYLOAD_LOAD
            ? payload_read_tensor_span(fp, spans[i], 0, bytes[i], buf, DS4_SESSION_IO_CHUNK, left, err, errlen)
            : payload_write_tensor_span(fp, spans[i], 0, bytes[i], buf, DS4_SESSION_IO_CHUNK, err, errlen);
    }
    return rc;
}

static int ling31_payload_save(ds4_ling31_graph *g, const int *tokens, uint32_t n,
        const float *logits, FILE *fp, char *err, size_t errlen) {
    if (!fp || !tokens || !logits || !ling31_payload_bytes(g, n)) {
        payload_set_err(err, errlen, "invalid Ling-3.1 snapshot state");
        return 1;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "invalid Ling-3.1 snapshot token");
            return 1;
        }
    }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) { payload_set_err(err, errlen, "invalid Ling-3.1 snapshot logits"); return 1; }
    }
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!buf || !ds4_gpu_synchronize()) {
        free(buf);
        payload_set_err(err, errlen, "Ling-3.1 snapshot allocation or synchronization failed");
        return 1;
    }
    const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION, g->ctx_cap, g->row_cap,
        LING31_LAYERS, L31_PAYLOAD_MAGIC, (uint32_t)g->state_bytes, n, g->mtp_drafts ? 1u : 0u, L31_KV_ROW,
        LING31_MLA_LAYERS, DS4_N_VOCAB, n,
    };
    int rc = 0;
    for (uint32_t i = 0; !rc && i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) { rc = payload_write_u32(fp, h[i], err, errlen); }
    for (uint32_t i = 0; !rc && i < n; i++) { rc = payload_write_u32(fp, (uint32_t)tokens[i], err, errlen); }
    if (!rc) { rc = payload_write_logits(fp, logits, buf, err, errlen); }
    if (!rc) { rc = ling31_payload_state(L31_PAYLOAD_SAVE, fp, g, n, buf, NULL, err, errlen); }
    free(buf);
    return rc;
}

static int ling31_payload_restore(ds4_ling31_graph *g, FILE *fp, uint64_t *left,
        const uint32_t *h, int **tokens_out, float *logits, char *err, size_t errlen) {
    if (!g || !fp || !left || !h || !tokens_out || !logits) { return 1; }
    /* A failed or partial restore can never be reused as a committed prefix. */
    g->failed = true;
    g->position = 0;
    *tokens_out = NULL;
    const uint32_t n = h[7];
    if (!g->ready || !n || n > g->ctx_cap || h[0] != DS4_SESSION_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_PAYLOAD_VERSION || h[2] < n || h[2] > L31_CONTEXT || !h[3] || h[3] > h[2] ||
        h[4] != LING31_LAYERS || h[5] != L31_PAYLOAD_MAGIC || h[6] != (uint32_t)g->state_bytes ||
        h[8] != (g->mtp_drafts ? 1u : 0u) || h[9] != L31_KV_ROW || h[10] != LING31_MLA_LAYERS || h[11] != DS4_N_VOCAB ||
        h[12] != n || *left != ling31_payload_body(g, n)) {
        payload_set_err(err, errlen, "incompatible Ling-3.1 snapshot header or length");
        return 1;
    }
    int *tokens = malloc((size_t)n * sizeof(*tokens));
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!tokens || !buf || !ds4_gpu_synchronize()) {
        free(tokens);
        free(buf);
        payload_set_err(err, errlen, "Ling-3.1 restore allocation or synchronization failed");
        return 1;
    }
    int rc = 0;
    for (uint32_t i = 0; !rc && i < n; i++) {
        uint32_t token = 0;
        rc = payload_read_u32(fp, &token, left, err, errlen);
        if (!rc && token >= DS4_N_VOCAB) { payload_set_err(err, errlen, "invalid Ling-3.1 snapshot token"); rc = 1; }
        tokens[i] = (int)token;
    }
    if (!rc) { rc = payload_read_bytes(fp, logits, (uint64_t)DS4_N_VOCAB * sizeof(float), left, err, errlen); }
    for (uint32_t i = 0; !rc && i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) { payload_set_err(err, errlen, "invalid Ling-3.1 snapshot logits"); rc = 1; }
    }
    if (!rc) { rc = ling31_payload_state(L31_PAYLOAD_LOAD, fp, g, n, buf, left, err, errlen); }
    if (!rc && (*left || !ds4_gpu_tensor_write(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) ||
                !ds4_gpu_synchronize())) {
        payload_set_err(err, errlen, "Ling-3.1 restore finalization failed");
        rc = 1;
    }
    free(buf);
    if (rc) { free(tokens); return 1; }
    g->position = n;
    g->mtp_len = g->mtp_drafts ? n - 1u : 0u;
    g->failed = false;
    *tokens_out = tokens;
    return 0;
}
