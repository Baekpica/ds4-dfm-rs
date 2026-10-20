/* Snapshot ("BEM1"): tokens, logits, every full layer's FP8 rows [0, n) and
 * every sliding layer's last min(n, 512) ring rows in position order. The
 * rings are re-placed by position on restore, so a payload loads into a
 * graph of any chunk size. */
enum { BEAM_PAYLOAD_MAGIC = 0x314d4542 }; /* "BEM1" */
typedef enum { BEAM_PAYLOAD_SAVE, BEAM_PAYLOAD_LOAD } beam_payload_dir;

static uint32_t beam_payload_window(uint32_t n) { return n < BEAM_WINDOW ? n : BEAM_WINDOW; }

static uint64_t beam_payload_body(uint32_t n) {
    const uint32_t swa = BEAM_LAYERS - BEAM_FULL_LAYERS;
    return (uint64_t)n * sizeof(uint32_t) + (uint64_t)DS4_N_VOCAB * sizeof(float) +
        (uint64_t)BEAM_FULL_LAYERS * n * beam_full_row() + (uint64_t)swa * beam_payload_window(n) * BEAM_SWA_ROW;
}

static uint64_t beam_payload_bytes(const ds4_beam_graph *g, uint32_t n) {
    if (!g->ready || g->failed || !n || n > g->ctx_cap || g->position != n) { return 0; }
    return DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) + beam_payload_body(n);
}

static int beam_payload_span(beam_payload_dir dir, FILE *fp, ds4_gpu_tensor *t, uint64_t offset, uint64_t bytes,
        uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    if (!bytes) { return 0; }
    return dir == BEAM_PAYLOAD_LOAD
        ? payload_read_tensor_span(fp, t, offset, bytes, buf, DS4_SESSION_IO_CHUNK, left, err, errlen)
        : payload_write_tensor_span(fp, t, offset, bytes, buf, DS4_SESSION_IO_CHUNK, err, errlen);
}

static int beam_payload_state(beam_payload_dir dir, FILE *fp, ds4_beam_graph *g, uint32_t n,
        uint8_t *buf, uint64_t *left, char *err, size_t errlen) {
    const uint32_t keep = beam_payload_window(n), first = n - keep;
    int rc = 0;
    for (uint32_t il = 0; !rc && il < BEAM_LAYERS; il++) {
        if (ds4_beam_layer_is_full(il)) {
            rc = beam_payload_span(dir, fp, g->kv[il], 0, (uint64_t)n * beam_full_row(), buf, left, err, errlen);
            continue;
        }
        // Positions [first, n) occupy slots p % swa_cap: at most two spans.
        const uint32_t slot = first % g->swa_cap;
        const uint32_t head = keep < g->swa_cap - slot ? keep : g->swa_cap - slot;
        rc = beam_payload_span(dir, fp, g->kv[il], (uint64_t)slot * BEAM_SWA_ROW, (uint64_t)head * BEAM_SWA_ROW,
                buf, left, err, errlen) ||
            beam_payload_span(dir, fp, g->kv[il], 0, (uint64_t)(keep - head) * BEAM_SWA_ROW, buf, left, err, errlen);
    }
    return rc;
}

static int beam_payload_save(ds4_beam_graph *g, const int *tokens, uint32_t n,
        const float *logits, FILE *fp, char *err, size_t errlen) {
    if (!fp || !tokens || !logits || !beam_payload_bytes(g, n)) {
        payload_set_err(err, errlen, "invalid Beam snapshot state");
        return 1;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "invalid Beam snapshot token");
            return 1;
        }
    }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) { payload_set_err(err, errlen, "invalid Beam snapshot logits"); return 1; }
    }
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!buf || !ds4_gpu_synchronize()) {
        free(buf);
        payload_set_err(err, errlen, "Beam snapshot allocation or synchronization failed");
        return 1;
    }
    const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION, g->ctx_cap, g->row_cap,
        BEAM_LAYERS, BEAM_PAYLOAD_MAGIC, beam_payload_window(n), n, 0u, beam_full_row(),
        BEAM_FULL_LAYERS, DS4_N_VOCAB, n,
    };
    int rc = 0;
    for (uint32_t i = 0; !rc && i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) { rc = payload_write_u32(fp, h[i], err, errlen); }
    for (uint32_t i = 0; !rc && i < n; i++) { rc = payload_write_u32(fp, (uint32_t)tokens[i], err, errlen); }
    if (!rc) { rc = payload_write_logits(fp, logits, buf, err, errlen); }
    if (!rc) { rc = beam_payload_state(BEAM_PAYLOAD_SAVE, fp, g, n, buf, NULL, err, errlen); }
    free(buf);
    return rc;
}

static int beam_payload_restore(ds4_beam_graph *g, FILE *fp, uint64_t *left,
        const uint32_t *h, int **tokens_out, float *logits, char *err, size_t errlen) {
    if (!g || !fp || !left || !h || !tokens_out || !logits) { return 1; }
    /* A failed or partial restore can never be reused as a committed prefix. */
    g->failed = true;
    g->position = 0;
    *tokens_out = NULL;
    const uint32_t n = h[7];
    if (!g->ready || !n || n > g->ctx_cap || h[0] != DS4_SESSION_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_PAYLOAD_VERSION || h[2] < n || h[2] > BEAM_CONTEXT || !h[3] || h[3] > h[2] ||
        h[4] != BEAM_LAYERS || h[5] != BEAM_PAYLOAD_MAGIC || h[6] != beam_payload_window(n) || h[8] ||
        h[9] != beam_full_row() || h[10] != BEAM_FULL_LAYERS || h[11] != DS4_N_VOCAB || h[12] != n ||
        *left != beam_payload_body(n)) {
        payload_set_err(err, errlen, "incompatible Beam snapshot header or length");
        return 1;
    }
    int *tokens = malloc((size_t)n * sizeof(*tokens));
    uint8_t *buf = malloc(DS4_SESSION_IO_CHUNK);
    if (!tokens || !buf || !ds4_gpu_synchronize()) {
        free(tokens);
        free(buf);
        payload_set_err(err, errlen, "Beam restore allocation or synchronization failed");
        return 1;
    }
    int rc = 0;
    for (uint32_t i = 0; !rc && i < n; i++) {
        uint32_t token = 0;
        rc = payload_read_u32(fp, &token, left, err, errlen);
        if (!rc && token >= DS4_N_VOCAB) { payload_set_err(err, errlen, "invalid Beam snapshot token"); rc = 1; }
        tokens[i] = (int)token;
    }
    if (!rc) { rc = payload_read_bytes(fp, logits, (uint64_t)DS4_N_VOCAB * sizeof(float), left, err, errlen); }
    for (uint32_t i = 0; !rc && i < DS4_N_VOCAB; i++) {
        if (!isfinite(logits[i])) { payload_set_err(err, errlen, "invalid Beam snapshot logits"); rc = 1; }
    }
    if (!rc) { rc = beam_payload_state(BEAM_PAYLOAD_LOAD, fp, g, n, buf, left, err, errlen); }
    if (!rc && (*left || !ds4_gpu_tensor_write(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) ||
                !ds4_gpu_synchronize())) {
        payload_set_err(err, errlen, "Beam restore finalization failed");
        rc = 1;
    }
    free(buf);
    if (rc) { free(tokens); return 1; }
    g->position = n;
    g->failed = false;
    *tokens_out = tokens;
    return 0;
}
